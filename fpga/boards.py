"""As-built FPGA configurations (SG9a): the nanoxl_top parameters each bitstream is built with, and the
`HwConfig` the host compiler/driver must use with that bitstream.

Single source of truth for:
  * fpga/<board>/build.tcl  -- the `CONFIG.<param>` values of the nanoxl_0 cell (tests/fpga checks they match)
  * fpga/yosys/estimate.py  -- resource estimates
  * fpga/run_board.py       -- the driver CLI wrapper used on the board

Why these differ from nanoxl/hwconfig.py (ADR-003): the RTL needs power-of-two buffer capacities (line
addresses wrap at the capacity) and the scratchpads must fit the device's BRAM/URAM granularity
(docs/fpga.md §2). The compiler's capacity for every buffer must be <= the RTL's, which `hwconfig()`
guarantees by construction.
"""
from __future__ import annotations

from nanoxl.hwconfig import KV260, PYNQ_Z2, KiB, HwConfig

LINE = 16  # bytes per scratchpad line

# nanoxl_top parameters (names exactly as in rtl/nanoxl_top.sv / rtl/nanoxl_axi_wrap.v)
RTL_PARAMS = {
    "kv260": dict(
        N=32, L=64, M_DEPTH=64,
        ABUF_LINES=32768,      # 512 KiB  -> URAM (8 banks x 4096 lines = 2 URAM288 each)
        WBUF_LINES=8192,       # 128 KiB  -> BRAM (8 banks x 1024 lines = 4 RAMB36 each)
        ACC_LINES=16384,       # 256 KiB  -> URAM (8 banks x 2048 lines)
        PBUF_LINES=4096,       # 64 KiB   -> BRAM (8 banks x 512 lines)
        KV_LINES=65536,        # 1 MiB    -> URAM (8 banks x 8192 lines = 4 URAM288 each)
        NB=8,                  # = widest port (ACC, N/4 lines); auto (16) would waste half of every URAM
        VP=4,                  # 64-byte VEC engine ports (SG10b)
        CTX_MAX=256, D_MAX=384, NG=6, NSLOT=16, NCTX=8, MAXB=256, LDF=512,
    ),
    "pynq-z2": dict(
        N=16, L=16, M_DEPTH=64,
        ABUF_LINES=8192,       # 128 KiB  (8 banks x 1024 lines = 4 RAMB36 each)
        WBUF_LINES=4096,       # 64 KiB
        ACC_LINES=4096,        # 64 KiB
        PBUF_LINES=2048,       # 32 KiB   (256-line banks: 2 RAMB36 each, half used)
        KV_LINES=8192,         # 128 KiB  (tiny fits on chip; shakespeare uses the DDR-KV mode)
        NB=0,                  # auto = 8 (2 x widest port: N/4 = VP = 4 lines)
        VP=4,
        CTX_MAX=256, D_MAX=384, NG=6, NSLOT=16, NCTX=8, MAXB=256, LDF=512,
    ),
}

CLOCK_MHZ = {"kv260": 250.0, "pynq-z2": 150.0}   # pl_clk0 / FCLK_CLK0 targets (ADR-003)
PART = {"kv260": "xck26-sfvc784-2LV-c", "pynq-z2": "xc7z020clg400-1"}
CSR_BASE = {"kv260": 0xA000_0000, "pynq-z2": 0x4000_0000}   # address of nanoxl_0/s_axil (4 KiB)

# Device capacities used for the budget check (DS987 K26 SOM, DS190 Zynq-7020).
DEVICE = {
    "kv260": dict(name="K26 (XCK26)", LUT=117_120, FF=234_240, DSP=1_248, BRAM36=144, URAM=64),
    "pynq-z2": dict(name="XC7Z020", LUT=53_200, FF=106_400, DSP=220, BRAM36=140, URAM=0),
}


def capacities(board: str) -> dict:
    """Buffer capacities in bytes of the as-built RTL."""
    p = RTL_PARAMS[board]
    return {k: p[f"{k.upper()}_LINES"] * LINE for k in ("abuf", "wbuf", "acc", "pbuf")} | \
        {"kvbuf": p["KV_LINES"] * LINE}


def hwconfig(board: str, kv_onchip: bool | None = None) -> HwConfig:
    """HwConfig for the compiler / driver when running on this bitstream. Timing knobs (DDR bandwidth,
    clock) stay the ADR-003 board assumptions; structure (N, lanes, capacities, 1-row weight port)
    is the as-built RTL."""
    base = {"kv260": KV260, "pynq-z2": PYNQ_Z2}[board]
    p = RTL_PARAMS[board]
    cap = capacities(board)
    kw = dict(name=f"{board}-fpga", array_n=p["N"], lanes=p["L"], clock_mhz=CLOCK_MHZ[board],
              wload_rows_per_cycle=1, **cap)
    kw["panel_bytes"] = min(base.panel_bytes, cap["wbuf"] // 4)
    if kv_onchip is not None:
        kw["kv_onchip"] = kv_onchip
    return base.with_(**kw)


__all__ = ["RTL_PARAMS", "CLOCK_MHZ", "PART", "CSR_BASE", "DEVICE", "capacities", "hwconfig", "KiB"]
