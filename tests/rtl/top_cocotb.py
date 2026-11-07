"""nanoxl_top end to end (SG7c, docs/top.md §5): the real host driver (nanoxl.driver.NanoXL) runs in a
cocotb bridge thread and talks to the RTL through an AXI4-Lite master (CSRs) and a cocotbext-axi
AxiRam that holds the DDR image (fetch master and DMA master share one memory). Every step's int16
logits, the final KV cache and the per-op saturation counters are compared with nanoxl.golden; the
greedy tokens with artifacts/models/tiny/golden_ref.json.

Environment (set by tests/rtl/test_top.py):
  TOP_BUNDLE w8a8 | w4a8      TOP_KV onchip | ddr       TOP_LAYERS 0 = all, else truncate the model
  TOP_PROMPT tokens of the reference prompt to prefill   TOP_DECODE greedy decode steps after it
  TOP_N, TOP_L array size / lanes (must equal the RTL parameters)   TOP_P_AXI random AXI stall prob.
  TOP_STRESS 1 = lowered shifts (every clamp fires)      TOP_PBUF_KB compiler's PBUF ring size
  TOP_PERF_JSON optional path: measured vs modelled cycles are written there
"""
from __future__ import annotations

import json
import logging
import mmap
import os
import random
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.task import bridge, resume
from cocotb.triggers import ClockCycles, First, ReadOnly, RisingEdge
from cocotbext.axi import AxiBus, AxiLiteBus, AxiLiteMaster, AxiRam, AxiReadBus
from cocotbext.axi.axi_ram import AxiRamRead

from nanoxl import isa
from nanoxl.cyclemodel import simulate
from nanoxl.driver import NanoXL
from nanoxl.driver import csr as R
from nanoxl.driver.backends import Backend
from nanoxl.export import load_bundle
from nanoxl.golden import Golden
from nanoxl.hwconfig import KV260, KiB

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "artifacts" / "models" / "tiny"
REF = json.loads((TINY / "golden_ref.json").read_text())
PHYS = 0x5000_0000             # driver buffer (image built at 0x4000_0000: relocation exercised)


def env(name, default):
    return type(default)(os.environ.get(name, default))


def top_hw(n, l, kv_onchip, pbuf_kb=32):
    """HwConfig the RTL test instance implements (nanoxl_top defaults: buffer capacities). A smaller
    `pbuf_kb` makes the compiler's PBUF ring wrap inside a program, so different parameter blocks land
    on the same PBUF lines (exercises the DMA -> VEC residency snoop)."""
    return KV260.with_(array_n=n, lanes=l, kv_onchip=kv_onchip, abuf=64 * KiB, wbuf=32 * KiB,
                       acc=64 * KiB, pbuf=pbuf_kb * KiB, kvbuf=128 * KiB)


def prepare(arrays, meta, layers, stress):
    """Optionally cut the model to its first `layers` blocks, and (stress) lower requant / LN shifts so
    that every saturating clamp of the hardware fires (golden and device run the same bundle)."""
    m = dict(meta)
    if layers and layers < meta["n_layer"]:
        m["n_layer"] = layers
    if stress:
        arrays = dict(arrays)
        for l in range(m["n_layer"]):
            for name, d in (("c_attn", 3), ("attn_proj", 3), ("c_fc", 3), ("score", 3), ("pv", 3)):
                k = f"h{l}.{name}.shift"
                arrays[k] = np.maximum(arrays[k] - d, 0).astype(arrays[k].dtype)
            k = f"h{l}.ln_1.c"
            arrays[k] = arrays[k].copy()
            arrays[k][4] -= 4
        arrays["lm_head.shift"] = np.maximum(arrays["lm_head.shift"] - 1, 0).astype(arrays["lm_head.shift"].dtype)
    return arrays, m


class RtlBackend(Backend):
    """Driver backend over the RTL: CSR reads/writes are AXI4-Lite transactions on the simulator
    (resume-converted coroutines), DDR is the AxiRam's memory (shared numpy view)."""

    poll_interval = 0

    def __init__(self, dut, axil, mem, phys):
        self.dut, self.axil, self.mem, self.phys = dut, axil, mem, phys
        self.n_csr = 0

        @resume
        async def rd(off):
            return int(await self.axil.read_dword(off))

        @resume
        async def wr(off, val):
            await self.axil.write_dword(off, int(val) & 0xFFFFFFFF)

        @resume
        async def wirq(cycles):
            if int(self.dut.irq.value):
                return True
            await First(RisingEdge(self.dut.irq), ClockCycles(self.dut.clk, cycles))
            return bool(int(self.dut.irq.value))

        self._rd, self._wr, self._wirq = rd, wr, wirq

    def read32(self, off):
        self.n_csr += 1
        return self._rd(off)

    def write32(self, off, val):
        self.n_csr += 1
        self._wr(off, val)

    def alloc(self, nbytes):
        self.nbytes = nbytes
        return np.frombuffer(self.mem, np.uint8, count=nbytes, offset=self.phys), self.phys

    def wait_irq(self, timeout):
        return self._wirq(200_000)

    def view(self, off, n):
        return np.frombuffer(self.mem, np.uint8, count=n, offset=self.phys + off)


class VecMonitor:
    """Snapshot of the VEC engine's five saturation counters at every `done` (per-op counts)."""

    def __init__(self, dut):
        self.dut = dut
        self.snaps = []
        cocotb.start_soon(self._run())

    def _sat(self):
        v = int(self.dut.u_vec.sat_cnt.value)
        return [(v >> (32 * i)) & 0xFFFFFFFF for i in range(5)]   # rq, alu, ln_out, ln_z, score

    async def _run(self):
        done = self.dut.u_vec.done
        while True:
            await RisingEdge(done)
            await ReadOnly()
            self.snaps.append(self._sat())
            while True:
                await RisingEdge(self.dut.clk)
                await ReadOnly()
                if int(done.value):
                    self.snaps.append(self._sat())
                else:
                    break


def per_op_sat(vec_cmds, snaps):
    """Per golden counter name, from the counter deltas at each VEC-queue done (in queue order)."""
    assert len(snaps) == len(vec_cmds), (len(snaps), len(vec_cmds))
    out, prev = {}, [0] * 5
    unit = {"REQUANT": 0, "VEC": 1, "LN": 2, "SOFTMAX_ONLINE": 4}
    for c, s in zip(vec_cmds, snaps):
        d = [a - b for a, b in zip(s, prev)]
        prev = s
        u = unit.get(c.op)
        for i, x in enumerate(d):
            if not x:
                continue
            if c.op == "LN" and i == 3:
                out["ln_z"] = out.get("ln_z", 0) + x
            elif i == u:
                out[c.note] = out.get(c.note, 0) + x
            else:
                raise AssertionError(f"{c.op} ({c.note}) moved saturation counter {i} by {x}")
    return out


@cocotb.test()
async def top_tiny(dut):
    bundle = env("TOP_BUNDLE", "w8a8")
    kv_mode = env("TOP_KV", "onchip")
    layers = env("TOP_LAYERS", 0)
    stress = env("TOP_STRESS", 0)
    n_prompt = env("TOP_PROMPT", 16)
    n_dec = env("TOP_DECODE", 8)
    N, L = env("TOP_N", 8), env("TOP_L", 4)
    p_axi = env("TOP_P_AXI", 0.0)
    perf_json = os.environ.get("TOP_PERF_JSON", "")
    log = dut._log
    for n in ("md_axi", "mf_axi", "s_axil"):
        logging.getLogger(f"cocotb.{dut._name}.{n}").setLevel(logging.WARNING)

    cocotb.start_soon(Clock(dut.clk, 4, unit="ns").start())
    mem = mmap.mmap(-1, 1 << 32)
    ram = AxiRam(AxiBus.from_prefix(dut, "md_axi"), dut.clk, dut.rst_n, reset_active_level=False,
                 size=1 << 32, mem=mem)
    fram = AxiRamRead(AxiReadBus.from_prefix(dut, "mf_axi"), dut.clk, dut.rst_n,
                      reset_active_level=False, size=1 << 32, mem=mem)
    axil = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axil"), dut.clk, dut.rst_n,
                         reset_active_level=False)
    for lg in (ram.write_if.log, ram.read_if.log, fram.log, axil.write_if.log, axil.read_if.log):
        lg.setLevel(logging.WARNING)
    if p_axi:
        rng = random.Random(1)

        def pause(p):
            r = random.Random(rng.random())
            while True:
                yield r.random() < p
        for ch in (ram.read_if.ar_channel, ram.read_if.r_channel, ram.write_if.aw_channel,
                   ram.write_if.w_channel, ram.write_if.b_channel, fram.ar_channel, fram.r_channel):
            ch.set_pause_generator(pause(p_axi))
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 8)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 4)

    A, meta = load_bundle(TINY / f"{bundle}.npz")
    A, meta = prepare(A, meta, layers, stress)
    hw = top_hw(N, L, kv_mode == "onchip", env("TOP_PBUF_KB", 32))
    be = RtlBackend(dut, axil, mem, PHYS)
    mon = VecMonitor(dut)
    prompt = REF["prompt"][:n_prompt]
    full = (not layers or layers >= 4) and not stress
    ref_tokens = REF["bundles"][bundle]["tokens"] if full and n_prompt == len(REF["prompt"]) else None

    def host():
        """Blocking host program: exactly what nanoxl.driver does for generate(), step by step,
        plus golden comparisons."""
        dev = NanoXL(be, A, meta, hw, name=bundle, use_irq=True, timeout=1e9)
        g = Golden(A, meta)
        cache = g.new_cache()
        dev.clear_counters()
        res = {"steps": [], "vec_cmds": [], "tokens": []}
        toks = list(prompt)
        for s in range(n_dec + 1):
            feed = toks if s == 0 else [res["tokens"][-1]]
            seqs = [(dev.pos[0], len(feed))]
            prog = dev.comp.compile_step(seqs)
            res["vec_cmds"] += [c for c in prog.cmds if c.queue == isa.Q_VEC and c.op not in ("WAIT", "SIGNAL")]
            logits = dev.step([feed])[0]
            ref = g.forward(feed, cache)
            if not np.array_equal(logits, ref):
                bad = np.argwhere(logits != ref)
                raise AssertionError(f"step {s}: logits differ at {len(bad)} entries, first {bad[0].tolist()} "
                                     f"rtl {logits[tuple(bad[0])]} golden {ref[tuple(bad[0])]}")
            t = simulate(prog.cmds, hw)
            res["steps"].append({"rows": len(feed), "pos": seqs[0][0], "cycles": dev.last_perf["cycles"],
                                 "model": t.cycles, "busy_dma": dev.last_perf["busy_dma"],
                                 "busy_arr": dev.last_perf["busy_arr"], "busy_vec": dev.last_perf["busy_vec"],
                                 "model_busy": dict(t.busy), "cmds": len(prog.cmds)})
            res["tokens"].append(int(np.argmax(logits[-1])))
        # KV cache: golden cache[k|v][layer, pos, d] vs the device's [layer][K|V][head][pos][dh]
        cfg = dev.cfg
        H, dh, ctx = cfg.n_head, cfg.d_head, cfg.ctx
        nkv = dev.comp.kv_slot
        if kv_mode == "onchip":
            img = dev.comp.img
            dump = img.size                    # scratch DDR past the image
            cmds = []
            for off in range(0, nkv, 65536):
                rb = min(65536, nkv - off)
                cmds.append(isa.C("DMA_STORE", chan=2, sp_buf=isa.KVBUF, sp_addr=off,
                                  ddr_addr=img.base + dump + off, rows=1, row_bytes=rb, ddr_stride=rb))
            cmds.append(isa.C("END"))
            dev.wait(dev.submit(cmds))
            kv = be.view(dump, nkv).copy()
        else:
            kv = be.view(dev.comp.kv_ddr - dev.comp.img.base, nkv).copy()
        kv = kv.view(np.int8).reshape(cfg.n_layer, 2, H, ctx, dh)
        for kind, name in ((0, "k"), (1, "v")):
            gk = cache[name].reshape(cfg.n_layer, ctx, H, dh).transpose(0, 2, 1, 3)
            if not np.array_equal(kv[:, kind], gk):
                bad = np.argwhere(kv[:, kind] != gk)[0].tolist()
                raise AssertionError(f"KV {name} differs first at [layer, head, pos, i] = {bad}: "
                                     f"rtl {kv[:, kind][tuple(bad)]} golden {gk[tuple(bad)]}")
        res["golden_sat"] = dict(g.sat_count)
        res["host_sat"] = dict(dev.host_sat)
        res["csr_sat"] = dev.saturation()
        res["err"] = be.read32(R.ERR_CODE)
        res["status"] = be.read32(R.STATUS)
        res["n_csr"] = be.n_csr
        return res

    res = await bridge(host)()
    await ClockCycles(dut.clk, 4)
    assert res["err"] == 0, f"ERR_CODE {res['err']}"
    assert res["status"] & R.ST_QUEUES_IDLE

    # per-op saturation counters: RTL VEC-engine deltas per command (+ host embedding) == golden
    sat = per_op_sat(res["vec_cmds"], mon.snaps)
    for k, v in res["host_sat"].items():
        sat[k] = sat.get(k, 0) + v
    gs = {k: v for k, v in res["golden_sat"].items() if v}
    sat = {k: v for k, v in sat.items() if v}
    log.info("saturation per op: rtl %s golden %s", sat, gs)
    assert sat == gs, (sat, gs)
    if stress:
        assert all(sat.get(k) for k in ("h0.c_attn", "h0.attn_proj", "h0.score", "h0.pv", "resid", "h0.ln_1")), sat
    cs = res["csr_sat"]
    last = mon.snaps[-1] if mon.snaps else [0] * 5
    tot = dict(zip(R.SAT_NAMES, last))       # SAT_i == vec_seq.sat_cnt[i] (A1-02 canonical map)
    assert cs == tot, (cs, tot)

    if ref_tokens is not None:
        assert res["tokens"] == ref_tokens[:len(res["tokens"])], (res["tokens"], ref_tokens[:len(res["tokens"])])
        log.info("tokens match golden_ref.json: %s", res["tokens"])
    for s in res["steps"]:
        log.info("PERF %s N=%d L=%d kv=%s step pos=%d rows=%d: measured %d cycles, cyclemodel %d (%.2fx); "
                 "busy dma/arr/vec %d/%d/%d model %s", bundle, N, L, kv_mode, s["pos"], s["rows"], s["cycles"],
                 s["model"], s["cycles"] / max(1, s["model"]), s["busy_dma"], s["busy_arr"], s["busy_vec"],
                 s["model_busy"])
    if perf_json:
        Path(perf_json).write_text(json.dumps({"bundle": bundle, "kv": kv_mode, "N": N, "L": L,
                                               "layers": layers, "steps": res["steps"],
                                               "tokens": res["tokens"], "sat": sat}, indent=1))
