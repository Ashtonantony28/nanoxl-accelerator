"""Float64 numpy forward of the FP32 GPT that exposes every quantization point.

Used for calibration (activation ranges) and as the float side of per-op error tables.
Matches nanoxl.model_fp.GPT (eval mode) to ~1e-5.
"""
from __future__ import annotations

import math

import numpy as np

from nanoxl.quant.ops import gelu_f


def layernorm_f(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax_f(x, axis=-1):
    m = x.max(axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis, keepdims=True)


def forward_f(W: dict, cfg, idx: np.ndarray, taps: dict | None = None):
    """idx int [B,T] -> logits [B,T,V] (float64). W: numpy state_dict (torch layout).
    taps (optional) receives every intermediate: resid.{l}.{in,mid}, ln_1.{l}, q/k/v.{l},
    scores.{l} (pre-softmax, scaled by 1/sqrt(hs)), att.{l}, y.{l}, fc.{l}, gelu.{l}, ln_f, logits."""
    f = lambda k: W[k].astype(np.float64)  # noqa: E731
    B, T = idx.shape
    d, H = cfg.n_embd, cfg.n_head
    hs = d // H
    x = f("transformer.wte.weight")[idx] + f("transformer.wpe.weight")[:T][None]
    tap = taps if taps is not None else {}
    causal = np.tril(np.ones((T, T), bool))
    for l in range(cfg.n_layer):
        p = f"transformer.h.{l}."
        tap[f"resid.{l}.in"] = x
        h = layernorm_f(x, f(p + "ln_1.weight"), f(p + "ln_1.bias"))
        tap[f"ln_1.{l}"] = h
        qkv = h @ f(p + "attn.c_attn.weight").T + f(p + "attn.c_attn.bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        tap[f"q.{l}"], tap[f"k.{l}"], tap[f"v.{l}"] = q, k, v
        q = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hs)
        tap[f"scores.{l}"] = s
        s = np.where(causal, s, -np.inf)
        a = softmax_f(s)
        tap[f"att.{l}"] = a
        y = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        tap[f"y.{l}"] = y
        x = x + y @ f(p + "attn.c_proj.weight").T + f(p + "attn.c_proj.bias")
        tap[f"resid.{l}.mid"] = x
        h = layernorm_f(x, f(p + "ln_2.weight"), f(p + "ln_2.bias"))
        tap[f"ln_2.{l}"] = h
        u = h @ f(p + "mlp.c_fc.weight").T + f(p + "mlp.c_fc.bias")
        tap[f"fc.{l}"] = u
        g = gelu_f(u)
        tap[f"gelu.{l}"] = g
        x = x + g @ f(p + "mlp.c_proj.weight").T + f(p + "mlp.c_proj.bias")
    tap["resid.out"] = x
    h = layernorm_f(x, f("transformer.ln_f.weight"), f("transformer.ln_f.bias"))
    tap["ln_f"] = h
    logits = h @ f("transformer.wte.weight").T
    tap["logits"] = logits
    return logits
