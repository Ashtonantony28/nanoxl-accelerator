"""Host driver (design-plan §6.5): bundle -> DDR image -> per-step programs on the command ring ->
logits -> host-side sampling -> ``generate(prompt, n)``.

The driver does exactly what nanoxl.funcsim.HostRunner models (docs/isa.md §6): host embedding
lookup ``x0 = sat16(emb.tok[t] + emb.pos[p])``, one compiled program per step written to the ring at
the tail, DOORBELL, wait for END, read int16 logits, pick the next token. Everything device-facing
goes through a ``Backend`` (CSRs + one shared DDR buffer), so the same code drives the board
(``PynqBackend``) and the emulator (``SimBackend``).
"""
from __future__ import annotations

import time

import numpy as np

from .. import isa
from ..compiler import Compiler, relocate
from ..graph import from_bundle
from ..hwconfig import KV260, HwConfig
from ..quant import ops as qops
from . import csr as R

# tiny-shakespeare character set (nanoGPT shakespeare_char, sorted; 65 symbols)
SHAKESPEARE_CHARS = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class CharCodec:
    def __init__(self, chars: str = SHAKESPEARE_CHARS):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        bad = sorted({c for c in s if c not in self.stoi})
        if bad:
            raise ValueError(f"characters not in the vocabulary: {bad!r}")
        return [self.stoi[c] for c in s]

    def decode(self, toks) -> str:
        return "".join(self.chars[int(t)] for t in toks)


class DriverError(RuntimeError):
    def __init__(self, msg, code=0, index=0, words=None):
        super().__init__(msg)
        self.code, self.index, self.words = code, index, words


class DriverTimeout(DriverError):
    pass


def sample(logits_row, top_k=None, temperature=1.0, rng=None, scale=1.0) -> int:
    """Next token from one int16 logits row. Greedy (argmax, ties -> lowest index, as the golden model)
    when ``top_k`` is None or ``temperature`` <= 0; otherwise top-k sampling of softmax(scale * l / T)."""
    l = np.asarray(logits_row, np.int64)
    if top_k is None or temperature <= 0:
        return int(np.argmax(l))
    rng = rng if rng is not None else np.random.default_rng()
    k = max(1, min(int(top_k), len(l)))
    idx = np.argsort(-l, kind="stable")[:k]
    z = l[idx].astype(np.float64) * scale / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(idx[rng.choice(k, p=p)])


class NanoXL:
    """One accelerator instance running one bundle. ``n_slots`` KV slots (batch)."""

    def __init__(self, backend, arrays, meta, hw: HwConfig = KV260, name: str = "bundle", n_slots: int = 1,
                 timeout: float = 10.0, use_irq: bool = False, **compiler_kw):
        self.be = backend
        self.A, self.meta = arrays, meta
        g, params = from_bundle(arrays, meta, name)
        self.cfg = g.cfg
        self.comp = Compiler(g, params, hw, n_slots=n_slots, **compiler_kw)
        self.timeout, self.use_irq = timeout, use_irq
        self.logit_scale = float(meta.get("scales", {}).get("logits", 1.0))
        self.host_sat = {}
        self.pos = [0] * n_slots
        self.programs = 0
        self.last_perf = {}
        if self.be.read32(R.ID) != R.ID_VALUE:
            raise DriverError(f"no NanoXL at the CSR base (ID 0x{self.be.read32(R.ID):08x})")
        self._load_image()
        self.reset()

    # ---------------------------------------------------------------- setup
    def _load_image(self):
        img = self.comp.img
        self.mem, self.phys = self.be.alloc(img.size)
        if self.phys % 4096:
            raise DriverError("DDR buffer must be 4 KiB aligned")
        self.delta = self.phys - img.base
        self.mem[:] = 0
        for off, data in img.chunks:
            self.mem[off:off + len(data)] = np.frombuffer(data, np.uint8)
        ring_addr, ring_bytes = img.regions["cmd_ring"]
        self.ring_off = ring_addr - img.base
        self.ring_words = ring_bytes // isa.WORD_BYTES
        if self.ring_words >= 1 << 24:
            raise DriverError("ring larger than RING_SIZE (24 bits)")
        self.be.flush()

    def reset(self):
        """Soft reset, ring configuration, IRQ enables; KV positions restart at 0."""
        be = self.be
        be.write32(R.CTRL, R.CTRL_SOFT_RESET)
        t0 = time.monotonic()
        while be.read32(R.CTRL) & R.CTRL_SOFT_RESET:
            if time.monotonic() - t0 > self.timeout:
                raise DriverTimeout("soft reset did not complete")
        be.write32(R.RING_BASE, self.phys + self.ring_off)
        be.write32(R.RING_SIZE, self.ring_words)
        be.write32(R.IRQ_EN, R.IRQ_END | R.IRQ_ERR)
        be.write32(R.IRQ_STATUS, 0x7)
        self.tail = be.read32(R.RING_HEAD)
        self.ends = be.read32(R.END_COUNT)
        self.pos = [0] * len(self.pos)

    # ---------------------------------------------------------------- ring
    def _free_words(self):
        head = self.be.read32(R.RING_HEAD)
        return (head - self.tail - 1) % self.ring_words

    def submit(self, cmds) -> int:
        """Relocate, encode and write a program at the ring tail, ring the doorbell. Returns the
        END_COUNT value that marks its completion (the program must end with END)."""
        if not cmds or cmds[-1].op != "END":
            raise DriverError("a program must end with END")
        words = isa.encode_stream(relocate(cmds, self.delta), aligned=True)
        n = len(words) // isa.WORD_BYTES
        if n >= self.ring_words:
            raise DriverError(f"program of {n} words does not fit the {self.ring_words}-word ring")
        t0 = time.monotonic()
        while self._free_words() < n:
            self._check_error()
            if time.monotonic() - t0 > self.timeout:
                raise DriverTimeout("ring full")
        self._write_ring(words)
        self.tail = (self.tail + n) % self.ring_words
        self.be.write32(R.DOORBELL, self.tail)
        self.ends = (self.ends + 1) & 0xFFFFFFFF
        return self.ends

    def submit_raw(self, words: bytes) -> None:
        """Write already-encoded words at the tail and ring (tests / bring-up; no END bookkeeping)."""
        self._write_ring(words)
        self.tail = (self.tail + len(words) // isa.WORD_BYTES) % self.ring_words
        self.be.write32(R.DOORBELL, self.tail)

    def _write_ring(self, words: bytes):
        a = np.frombuffer(words, np.uint8).reshape(-1, isa.WORD_BYTES)
        n = len(a)
        first = min(n, self.ring_words - self.tail)
        o = self.ring_off + self.tail * isa.WORD_BYTES
        self.mem[o:o + first * isa.WORD_BYTES] = a[:first].reshape(-1)
        if n > first:  # wrap
            self.mem[self.ring_off:self.ring_off + (n - first) * isa.WORD_BYTES] = a[first:].reshape(-1)
        self.be.flush()

    def _check_error(self):
        code = self.be.read32(R.ERR_CODE)
        if code:
            idx = self.be.read32(R.ERR_INDEX)
            w = [self.be.read32(R.ERR_W0 + 4 * i) for i in range(4)]
            raise DriverError(f"accelerator halted: ERR {code} ({R.ERR_NAMES.get(code, '?')}) at ring index {idx}",
                              code, idx, w)

    def wait(self, end_count: int):
        """Poll (or IRQ-wait) until END_COUNT reaches ``end_count``; W1C the END bit. Raises on error."""
        be = self.be
        t0 = time.monotonic()
        while True:
            self._check_error()
            if ((be.read32(R.END_COUNT) - end_count) & 0xFFFFFFFF) < 1 << 31:
                be.write32(R.IRQ_STATUS, R.IRQ_END)
                return
            left = self.timeout - (time.monotonic() - t0)
            if left <= 0:
                st = be.read32(R.STATUS)
                raise DriverTimeout(f"timeout waiting for END #{end_count} (STATUS 0x{st:04x}, "
                                    f"head {be.read32(R.RING_HEAD)}, tail {self.tail})")
            if self.use_irq:
                be.wait_irq(left)
            elif be.poll_interval:
                time.sleep(be.poll_interval)

    # ---------------------------------------------------------------- model I/O
    def _io(self, prog, name):
        addr, rows, width, bits = prog.io[name]
        return addr - self.comp.img.base, rows, width, {8: np.int8, 16: np.int16, 32: np.int32}[bits]

    def embed(self, tokens, p0):
        t = np.asarray(tokens, np.int64)
        x = self.A["emb.tok"][t].astype(np.int64) + self.A["emb.pos"][p0:p0 + len(t)].astype(np.int64)
        return qops.sat(x, 16, counter=self.host_sat, name="resid").astype(np.int16)

    def step(self, tokens_per_slot):
        """Append tokens_per_slot[b] to slot b for every b; returns int16 logits per slot."""
        seqs = [(self.pos[b], len(t)) for b, t in enumerate(tokens_per_slot)]
        prog = self.comp.compile_step(seqs)
        x0 = np.concatenate([self.embed(t, self.pos[b]) for b, t in enumerate(tokens_per_slot)])
        off, rows, width, dt = self._io(prog, "x")
        assert x0.shape == (rows, width)
        self.mem[off:off + x0.nbytes] = x0.astype(dt).view(np.uint8).reshape(-1)
        self.be.flush()
        p0 = self.perf()
        self.wait(self.submit(prog.cmds))
        p1 = self.perf()
        self.last_perf = {k: (p1[k] - p0[k]) % (1 << (64 if k == "bytes" else 32)) for k in p1}
        self.programs += 1
        self.be.invalidate()
        off, rows, width, dt = self._io(prog, "logits")
        logits = self.mem[off:off + rows * width * np.dtype(dt).itemsize].view(dt).reshape(rows, width).copy()
        out, r = [], 0
        for b, (_, n) in enumerate(seqs):
            out.append(logits[r:r + n])
            r += n
            self.pos[b] += n
        return out

    def prefill(self, prompt, chunk=None):
        chunk = chunk or len(prompt)
        return np.concatenate([self.step([list(prompt[i:i + chunk])])[0] for i in range(0, len(prompt), chunk)])

    def generate(self, prompt, n, top_k=None, temperature=1.0, seed=None, chunk=None, trace=None):
        """Prefill ``prompt`` (token ids) in slot 0 and produce up to ``n`` new tokens (stops at ctx).
        Greedy by default (bit-identical to nanoxl.golden.Golden.generate); ``top_k`` samples."""
        assert len(self.pos) == 1, "generate() drives slot 0 of a single-slot driver"
        if self.pos[0]:
            self.reset()
        rng = np.random.default_rng(seed)
        logits = self.prefill(list(prompt), chunk)
        if trace is not None:
            trace.append(logits)
        out = []
        while len(out) < n:
            out.append(sample(logits[-1], top_k, temperature, rng, self.logit_scale))
            if len(prompt) + len(out) >= self.cfg.ctx or len(out) >= n:
                break
            logits = self.step([[out[-1]]])[0]
            if trace is not None:
                trace.append(logits)
        return out

    # ---------------------------------------------------------------- counters
    def perf(self) -> dict:
        r = self.be.read32
        return {"cycles": r(R.PERF_CYCLES), "busy_dma": r(R.PERF_BUSY_DMA), "busy_arr": r(R.PERF_BUSY_ARR),
                "busy_vec": r(R.PERF_BUSY_VEC), "retired": r(R.PERF_RETIRED),
                "bytes": r(R.PERF_BYTES_LO) | (r(R.PERF_BYTES_HI) << 32)}

    def saturation(self) -> dict:
        return {n: self.be.read32(R.SAT_BASE + 4 * i) for i, n in enumerate(R.SAT_NAMES)}

    def clear_counters(self):
        self.be.write32(R.PERF_CTRL, 3)
