"""Integer golden model of the whole nanoGPT (contract C5, ADR-002). This file is the spec.

Pure numpy integer inference from a quantized bundle (nanoxl.quant.quantize / nanoxl.export):
embeddings -> N x [LN1 -> QKV -> scores -> softmax -> P@V -> proj -> residual add,
LN2 -> FC -> GELU -> proj -> residual add] -> LN_f -> LM head (int16 logits) -> argmax.
Every matmul is nanoxl.numerics.gemm_i8 followed by nanoxl.numerics.requant (C1/C2).

Shapes: one sequence at a time; `forward(tokens, pos0, cache)` processes a chunk (prefill
or a single decode step) and appends K/V to the cache, so decode-with-cache and full
recompute are the same computation row by row (bit-exact by construction; tested).
"""
from __future__ import annotations

import numpy as np

from nanoxl import numerics as nx
from nanoxl.quant import ops

I64 = np.int64
_GEMM = nx.gemm_i8


def gemm_exact_f64(a, w):
    """Fast exact equivalent of gemm_i8 via float64 BLAS (|sum| <= 2^26 << 2^53). Test-checked."""
    assert a.dtype == np.int8 and w.dtype == np.int8 and a.shape[1] == w.shape[0] <= 4096
    return (a.astype(np.float64) @ w.astype(np.float64)).astype(np.int32)


def set_gemm(fn):
    """Swap the GEMM implementation (nx.gemm_i8 by default; gemm_exact_f64 for long evals)."""
    global _GEMM
    _GEMM = fn


class Golden:
    def __init__(self, arrays: dict, meta: dict):
        self.A, self.meta = arrays, meta
        self.L, self.H, self.d = meta["n_layer"], meta["n_head"], meta["n_embd"]
        self.V, self.T = meta["vocab_size"], meta["block_size"]
        self.hs = self.d // self.H
        q = meta["quant"]
        self.rb = q["res_bits"]
        self.gelu_mode, self.exp_mode = q["gelu"], q["exp"]
        self.group = q["group"]
        self.sat_count: dict[str, int] = {}
        self._w8 = {}
        for k in list(arrays):
            if k.endswith(".w4"):  # unpack INT4 once (C1b)
                self._w8[k[:-3]] = nx.unpack_i4(arrays[k])
        if self.exp_mode == "poly":
            coef = tuple(meta["exp_poly"])
            self._exp_fn = lambda d: ops.exp2_neg_poly_i(d, coef)
        else:
            self._exp_fn = None

    # ------------------------------------------------------------ primitives
    def linear(self, name, x, out_bits):
        """x int8 [M,K] -> requant(acc + bias) int{out_bits} [M,N].
        INT8: acc = gemm_i8(x, W).  INT4 (W4A8): acc = sum_g gm[g,n] * gemm_i8(x[:,g], W4[g,:])."""
        A = self.A
        if name + ".w" in A:
            acc = _GEMM(x, A[name + ".w"]).astype(I64)
        else:
            w = self._w8[name]
            gm = A[name + ".gm"].astype(I64)
            G = self.group
            acc = np.zeros((x.shape[0], w.shape[1]), I64)
            for g in range(w.shape[0] // G):
                acc += gm[g] * _GEMM(np.ascontiguousarray(x[:, g * G:(g + 1) * G]), w[g * G:(g + 1) * G])
            assert np.all(np.abs(acc) < (1 << 31))
        return self._requant(acc, A[name + ".mult"], A[name + ".shift"], A[name + ".bias"],
                             out_bits, name)

    def _requant(self, acc, mult, shift, bias, out_bits, name):
        x = np.asarray(acc, I64) + np.asarray(bias, I64)
        x = ((x + (1 << 31)) % (1 << 32)) - (1 << 31)
        y = nx.rdivpot(nx.srdhm(x, mult), shift).astype(I64)
        ops.sat(y, out_bits, counter=self.sat_count, name=name)
        return nx.requant(acc, mult, shift, bias, out_bits)

    def layernorm(self, name, x):
        c = self.A[name + ".c"]
        cd = dict(mu_mult=int(c[0]), mu_shift=int(c[1]), sqrt_d=int(c[2]), eps_q=int(c[3]))
        z = ops.ln_norm_i(x, cd, self.A["lut.rsqrt"], counter=self.sat_count)
        return ops.ln_affine_i(z, self.A[name + ".g"], self.A[name + ".b"], int(c[4]),
                               counter=self.sat_count, name=name)

    def gelu(self, l, u):
        if self.gelu_mode == "lut":
            return ops.gelu_lut_i(u, self.A[f"h{l}.gelu.lut"])
        sc = self.meta["scales"]  # study-only float-modelled I-BERT polynomial
        g = ops.gelu_ibert_f(u.astype(np.float64) * sc[f"h{l}.fc"]) / sc[f"h{l}.gelu"]
        return np.clip(np.sign(g) * np.floor(np.abs(g) + 0.5), -128, 127).astype(np.int8)

    def res_add(self, x, r):
        return ops.sat(np.asarray(x, I64) + r, self.rb, counter=self.sat_count, name="resid").astype(
            np.int16 if self.rb == 16 else np.int8)

    # ------------------------------------------------------------ model
    def new_cache(self):
        return dict(k=np.zeros((self.L, self.T, self.d), np.int8),
                    v=np.zeros((self.L, self.T, self.d), np.int8), n=0)

    def forward(self, tokens, cache=None, trace: dict | None = None):
        """tokens int [t] appended at positions cache['n'] .. cache['n']+t-1.
        Returns int16 logits [t, V] (scale meta.scales.logits)."""
        A, d, H, hs = self.A, self.d, self.H, self.hs
        cache = cache if cache is not None else self.new_cache()
        tokens = np.asarray(tokens, I64)
        t, p0 = len(tokens), cache["n"]
        assert p0 + t <= self.T, "context overflow (v1 policy: stop at block_size)"
        tr = trace if trace is not None else {}
        x = self.res_add(A["emb.tok"][tokens], A["emb.pos"][p0:p0 + t].astype(I64))
        pos_q = np.arange(p0, p0 + t)[:, None]
        pos_k = np.arange(p0 + t)[None, :]
        mask = pos_k <= pos_q                                   # causal, [t, p0+t]
        for l in range(self.L):
            hp = f"h{l}."
            tr[f"resid.{l}.in"] = x
            h = self.layernorm(hp + "ln_1", x)
            tr[f"ln_1.{l}"] = h
            qkv = self.linear(hp + "c_attn", h, 8)
            q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
            cache["k"][l, p0:p0 + t] = k
            cache["v"][l, p0:p0 + t] = v
            K, Vv = cache["k"][l, :p0 + t], cache["v"][l, :p0 + t]
            tr[f"q.{l}"], tr[f"k.{l}"], tr[f"v.{l}"] = q, k, v
            y = np.empty((t, d), np.int8)
            ls = int(A[hp + "score.ls"][0])
            sc_all, p_all = [], []
            for hh in range(H):
                sl = slice(hh * hs, (hh + 1) * hs)
                acc = _GEMM(np.ascontiguousarray(q[:, sl]), np.ascontiguousarray(K[:, sl].T)).astype(I64)
                s = self._requant(acc << ls, A[hp + "score.mult"], A[hp + "score.shift"], 0, 16,
                                  hp + "score")
                pr = ops.softmax_i(s, mask, A["lut.exp2"], A["lut.recip"], self._exp_fn)
                pv = _GEMM(np.where(mask, pr, 0).astype(np.int8), np.ascontiguousarray(Vv[:, sl]))
                y[:, sl] = self._requant(pv, A[hp + "pv.mult"], A[hp + "pv.shift"], 0, 8, hp + "pv")
                sc_all.append(s)
                p_all.append(pr)
            tr[f"scores.{l}"], tr[f"att.{l}"] = np.stack(sc_all), np.stack(p_all)
            tr[f"y.{l}"] = y
            x = self.res_add(x, self.linear(hp + "attn_proj", y, self.rb))
            tr[f"resid.{l}.mid"] = x
            h = self.layernorm(hp + "ln_2", x)
            tr[f"ln_2.{l}"] = h
            u = self.linear(hp + "c_fc", h, 8 if self.gelu_mode == "lut" else 16)
            tr[f"fc.{l}"] = u
            g = self.gelu(l, u)
            tr[f"gelu.{l}"] = g
            x = self.res_add(x, self.linear(hp + "mlp_proj", g, self.rb))
        cache["n"] = p0 + t
        tr["resid.out"] = x
        h = self.layernorm("ln_f", x)
        tr["ln_f"] = h
        logits = self.linear("lm_head", h, 16)
        tr["logits"] = logits
        return logits

    def generate(self, prompt, n_new, use_cache=True):
        """Greedy decode. argmax over int16 logits, ties -> lowest index. Stops at block_size."""
        prompt = [int(t) for t in prompt]
        out = list(prompt)
        if use_cache:
            cache = self.new_cache()
            logits = self.forward(prompt, cache)
            for _ in range(n_new):
                nxt = int(np.argmax(logits[-1]))
                out.append(nxt)
                if len(out) >= self.T or len(out) - len(prompt) >= n_new:
                    break
                logits = self.forward([nxt], cache)
        else:
            for _ in range(n_new):
                logits = self.forward(out[-self.T:])
                out.append(int(np.argmax(logits[-1])))
                if len(out) >= self.T:
                    break
        return out[len(prompt):]


def load(path) -> Golden:
    from nanoxl.export import load_bundle
    return Golden(*load_bundle(path))
