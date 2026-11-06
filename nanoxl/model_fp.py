"""FP32 reference GPT-2 (nanoGPT architecture) and the Shakespeare-char dataset.

Weight-compatible with nanoGPT's `GPT.state_dict()` (original nanoGPT key names:
`transformer.{wte,wpe,h.N.{ln_1,attn.c_attn,attn.c_proj,ln_2,mlp.c_fc,mlp.c_proj},ln_f}`,
`lm_head.weight` tied to `wte`). nanoGPT-extended (commit 24e8407) names the MLP `ff`
instead of `mlp`; `load_nanogpt_state_dict` accepts both.

Architecture choices (frozen in ADR-002):
- LayerNorm with bias, eps = 1e-5 (torch default).
- GELU: exact erf form (`F.gelu` default, what nanoGPT trains with). The integer
  golden model uses a 256-entry LUT, which can encode either form at no cost.
- Learned absolute position embedding, multi-head causal attention, tied LM head.
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


CONFIGS = {
    "tiny": GPTConfig(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0),
    "shakespeare": GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
}


# --------------------------------------------------------------------------- data
def load_shakespeare(root: Path | None = None):
    """Return (train uint16, val uint16, chars). 90/10 split like nanoGPT prepare.py."""
    root = Path(root or REPO / "data" / "raw")
    path = root / "input.txt"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], chars


# --------------------------------------------------------------------------- model
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # tied
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for blk in self.transformer.h:
            x = blk(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_nanogpt_state_dict(model: GPT, sd: dict):
    """Load a nanoGPT / nanoGPT-extended state_dict (handles `_orig_mod.` and `.ff.`)."""
    fixed = {}
    for k, v in sd.items():
        k = k.removeprefix("_orig_mod.")
        k = k.replace(".ff.", ".mlp.")
        if k.endswith(".attn.bias"):  # causal-mask buffer of old nanoGPT
            continue
        fixed[k] = v
    fixed.setdefault("lm_head.weight", fixed["transformer.wte.weight"])  # tied
    model.load_state_dict(fixed, strict=True)
    return model


def load_checkpoint(path) -> GPT:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    cfg.dropout = 0.0
    m = GPT(cfg)
    load_nanogpt_state_dict(m, ck["model"])
    m.eval()
    return m


def save_checkpoint(path, model: GPT, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    torch.save({"model": sd, "config": asdict(model.config), **(extra or {})}, path)


def numpy_weights(model: GPT) -> dict:
    """state_dict as float32 numpy, Linear weights kept in torch [out, in] layout."""
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}


@torch.no_grad()
def eval_loss(model: GPT, data: np.ndarray, block: int | None = None, max_tokens: int | None = None,
              batch: int = 16) -> float:
    """Mean CE (nats) over non-overlapping windows of length `block` covering `data`."""
    model.eval()
    block = block or model.config.block_size
    d = data if max_tokens is None else data[: max_tokens + 1]
    n = (len(d) - 1) // block
    x = torch.from_numpy(d[: n * block].astype(np.int64)).view(n, block)
    y = torch.from_numpy(d[1: n * block + 1].astype(np.int64)).view(n, block)
    tot = 0.0
    for i in range(0, n, batch):
        logits, _ = model(x[i:i + batch])
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y[i:i + batch].reshape(-1),
                               reduction="sum").item()
    return tot / (n * block)
