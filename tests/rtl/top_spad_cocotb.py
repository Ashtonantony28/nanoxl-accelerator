"""buf_spad (rtl/buf_spad.sv): banked physical scratchpad with rotating-priority port arbitration.

Random traffic on every read and write port (multi-line beats at any start line, wrapping at the buffer
end, byte strobes), valid held until ready as every requester does. Reference: a line-array model in
which a write granted at edge t is visible to reads granted at edges > t (a read granted at the same
edge returns the old line, also with the bank output register, SP_OREG = 1). Checks every response
(latency 1 + OREG, data), that two grants of one cycle never share a bank, and the starvation bound (a
request waits < number of ports cycles).

Env: SP_LINES, SP_NB, SP_RPL, SP_WPL (comma lists, lines per beat per port), SP_CYCLES, SP_OREG.
"""
from __future__ import annotations

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly


XZ0 = str.maketrans("xzXZuUwW-", "000000000")


def cfg():
    lines = int(os.environ.get("SP_LINES", "64"))
    nb = int(os.environ.get("SP_NB", "4"))
    rpl = [int(x) for x in os.environ.get("SP_RPL", "1,2,4,1").split(",")]
    wpl = [int(x) for x in os.environ.get("SP_WPL", "1,4,2").split(",")]
    return lines, nb, rpl, wpl, max(rpl + wpl), int(os.environ.get("SP_CYCLES", "4000")), \
        int(os.environ.get("SP_OREG", "0"))


@cocotb.test()
async def spad_random(dut):
    LINES, NB, RPL, WPL, PMAX, CYC, OREG = cfg()
    NR, NW = len(RPL), len(WPL)
    rng = random.Random(7)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.r_valid.value = 0
    dut.w_valid.value = 0
    dut.r_addr.value = 0
    dut.w_addr.value = 0
    dut.w_data.value = 0
    dut.w_strb.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    mem = [0] * LINES                       # model, 128-bit ints (bank_spad zero-initialises)
    rq = [None] * NR                        # pending read: (addr, age)
    wq = [None] * NW                        # pending write: (addr, data lines, strb lines, age)
    pipe = [{} for _ in range(1 + OREG)]    # per grant edge in flight: port -> expected lines
    max_wait_r = max_wait_w = 0
    n_rd = n_wr = 0
    # hot region so that ports collide often; sometimes anywhere (wrap at the end)
    def addr():
        return rng.randrange(LINES) if rng.random() < 0.3 else (LINES - 3 + rng.randrange(8)) % LINES \
            if rng.random() < 0.2 else rng.randrange(8)
    p_req = 0.6
    for cyc in range(CYC):
        await FallingEdge(dut.clk)
        if cyc % 500 == 0:
            p_req = rng.choice([0.2, 0.6, 1.0])
        rv = ra = wv = wa = wd = ws = 0
        for p in range(NR):
            if rq[p] is None and rng.random() < p_req:
                rq[p] = [addr(), 0]
            if rq[p] is not None:
                rv |= 1 << p
                ra |= rq[p][0] << (17 * p)
        for p in range(NW):
            if wq[p] is None and rng.random() < p_req:
                n = WPL[p]
                wq[p] = [addr(), [rng.getrandbits(128) for _ in range(n)],
                         [rng.getrandbits(16) if rng.random() < 0.5 else 0xFFFF for _ in range(n)], 0]
            if wq[p] is not None:
                wv |= 1 << p
                wa |= wq[p][0] << (17 * p)
                for j in range(WPL[p]):
                    wd |= wq[p][1][j] << (128 * (p * PMAX + j))
                    ws |= wq[p][2][j] << (16 * (p * PMAX + j))
        dut.r_valid.value = rv
        dut.r_addr.value = ra
        dut.w_valid.value = wv
        dut.w_addr.value = wa
        dut.w_data.value = wd
        dut.w_strb.value = ws
        await ReadOnly()
        # responses for the grants of the edge 1 + OREG cycles ago
        expect = pipe.pop(0)
        dv = int(dut.d_valid.value)
        dd = int(str(dut.d_data.value).translate(XZ0), 2) if dv else 0   # idle bank outputs may be X
        for p in range(NR):
            got = (dv >> p) & 1
            assert got == (p in expect), f"cycle {cyc}: port {p} d_valid {got}, expected {p in expect}"
            if got:
                for j, e in enumerate(expect[p]):
                    v = (dd >> (128 * (p * PMAX + j))) & ((1 << 128) - 1)
                    assert v == e, f"cycle {cyc}: read port {p} line {j}: {v:032x} != {e:032x}"
        expect = {}
        pipe.append(expect)
        rr = int(dut.r_ready.value)
        wr = int(dut.w_ready.value)
        banks = set()
        for p in range(NR):
            if rq[p] is None:
                assert not (rr >> p) & 1
                continue
            if (rr >> p) & 1:
                lines = [(rq[p][0] + j) % LINES for j in range(RPL[p])]
                bk = {l % NB for l in lines}
                assert not (bk & banks), f"cycle {cyc}: read bank conflict"
                banks |= bk
                expect[p] = [mem[l] for l in lines]          # read-first: before this edge's writes
                max_wait_r = max(max_wait_r, rq[p][1])
                rq[p] = None
                n_rd += 1
            else:
                rq[p][1] += 1
                assert rq[p][1] < NR, f"read port {p} starved {rq[p][1]} cycles"
        banks = set()
        for p in range(NW):
            if wq[p] is None:
                assert not (wr >> p) & 1
                continue
            if (wr >> p) & 1:
                for j in range(WPL[p]):
                    l = (wq[p][0] + j) % LINES
                    assert l % NB not in banks, f"cycle {cyc}: write bank conflict"
                    banks.add(l % NB)
                    old, new, st = mem[l], wq[p][1][j], wq[p][2][j]
                    m = sum(0xFF << (8 * b) for b in range(16) if (st >> b) & 1)
                    mem[l] = (old & ~m) | (new & m)
                max_wait_w = max(max_wait_w, wq[p][3])
                wq[p] = None
                n_wr += 1
            else:
                wq[p][3] += 1
                assert wq[p][3] < NW, f"write port {p} starved {wq[p][3]} cycles"
    dut._log.info("buf_spad: %d reads, %d writes granted; max wait %d (read) %d (write) cycles",
                  n_rd, n_wr, max_wait_r, max_wait_w)
    assert n_rd > CYC // 2 and n_wr > CYC // 3
