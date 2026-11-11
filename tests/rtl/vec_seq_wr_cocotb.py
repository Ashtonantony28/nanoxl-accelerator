"""vec_seq_wr wide mode and its simulation-only walker check (AUDIT-2 A2-05; test_vec_seq_wr.py).

One row of up to 2 * L int32 lanes (two vec_unit beats, each wider than a 16P-byte port beat) is written from
line 0. Each case sets the keep bits of the two beats; the bench collects the port writes and compares
them with the lines the row should produce (bytes of kept lanes, strobes = keep). Env: L, P, CASE:
  legal     both beats fully kept (what every producer emits)
  interior  beat 0 keeps lanes 0..3 and the last four (interior hole, last chunk still kept): allowed
  ragged    the row is L + L/2 + 1 elements, so beat 1 keeps only its first L/2 + 1 lanes: allowed
  violate   beat 0 (not final) keeps lanes 0..L/4-1 only, so its trailing chunks are empty: the writer
            emits fewer chunks than the walker expects and chk_err must rise (chk_nofatal is set so
            the bench observes it instead of the $fatal that stops any other run).
"""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge

L = int(os.environ.get("L", "16"))
P = int(os.environ.get("P", "1"))
CASE = os.environ.get("CASE", "legal")


def keeps():
    full = (1 << L) - 1
    if CASE == "interior":
        return [0xF | (0xF << (L - 4)), full]
    if CASE == "ragged":
        return [full, (1 << (L // 2 + 1)) - 1]
    if CASE == "violate":
        return [(1 << (L // 4)) - 1, full]
    return [full, full]


@cocotb.test()
async def vec_seq_wr_wide(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for s in (dut.start, dut.clr, dut.i_valid, dut.i_last, dut.w_ready):
        s.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    if CASE == "violate":
        dut.chk_nofatal.value = 1
    dut.w_ready.value = 1
    n = L + L // 2 + 1 if CASE == "ragged" else 2 * L   # row length in int32 elements
    dut.d_buf.value, dut.d_base.value, dut.d_so.value, dut.d_no.value = 1, 0, 0, 1
    dut.d_si.value, dut.d_ni.value, dut.d_len.value, dut.d_len_last.value = 0, 1, n, n
    dut.d_ebl.value = 2
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    writes = []

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.w_valid.value) and int(dut.w_ready.value):
                writes.append((int(dut.w_addr.value), int(dut.w_data.value), int(dut.w_strb.value)))
    cocotb.start_soon(collect())

    ks = keeps()
    vals = [[((b * L + j + 1) * 0x01010101) & 0xFFFFFFFF for j in range(L)] for b in range(2)]
    for b in range(2):
        dut.i_data.value = sum(v << (32 * j) for j, v in enumerate(vals[b]))
        dut.i_keep.value = ks[b]
        dut.i_last.value = int(b == 1)
        dut.i_valid.value = 1
        while True:
            await RisingEdge(dut.clk)
            if int(dut.i_ready.value):
                break
    dut.i_valid.value = 0
    for _ in range(200):
        await RisingEdge(dut.clk)
        if not int(dut.busy.value):
            break
    await ClockCycles(dut.clk, 4)

    err = int(dut.chk_err.value)
    if CASE == "violate":
        assert err == 1, "A2-05 check did not fire on a non-final beat with empty trailing chunks"
        return
    assert err == 0, "A2-05 check fired on a legal stream"
    img = {}                                      # line -> (data, strobe) the row should produce
    for b in range(2):
        for j in range(L):
            if (ks[b] >> j) & 1:
                line, off = divmod(4 * (b * L + j), 16)
                d, s = img.get(line, (0, 0))
                img[line] = (d | (vals[b][j] << (8 * off)), s | (0xF << off))
    got = {}
    for addr, data, strb in writes:
        for q in range(P):
            s = (strb >> (16 * q)) & 0xFFFF
            if s:
                d = (data >> (128 * q)) & ((1 << 128) - 1)
                mask = sum(0xFF << (8 * i) for i in range(16) if (s >> i) & 1)
                assert addr + q not in got, f"line {addr + q} written twice"
                got[addr + q] = (d & mask, s)
    assert got == img, (sorted(got), sorted(img))
