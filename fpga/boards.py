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
    # First-bitstream configuration, ADR-006: the largest array / vector width whose Yosys LUT count
    # (logic + LUTRAM) stays <= 70 % of the XCK26 (62 %); N = 16 is the step-up once Vivado has counted.
    "kv260": dict(
        N=8, L=4, M_DEPTH=64,
        ABUF_LINES=32768,      # 512 KiB  -> URAM (2 banks x 16384 lines = 8 URAM288 each)
        WBUF_LINES=8192,       # 128 KiB  -> URAM (2 banks x 4096 lines = 2 URAM288 each)
        ACC_LINES=16384,       # 256 KiB  -> URAM (2 banks x 8192 lines = 4 URAM288 each)
        PBUF_LINES=4096,       # 64 KiB   -> URAM (2 banks x 2048 lines = 2 URAM288 each)
        KV_LINES=32768,        # 512 KiB  -> URAM (2 banks x 16384 lines): tiny on chip; shakespeare
                               #            (1.18 MB of KV) runs the DDR-KV mode at any power-of-two size
        NB=2,                  # = widest port (array ACC, N/4 = 2 lines); halves the 5 routers vs NB = 4
        RAM_STYLE=2,           # banks of >= 2048 lines in URAM288 (all five here), the rest RAMB36 (SG11b)
        VP=1,                  # 16-byte VEC ports: at L = 4 one line already carries a full int32 beat
        CTX_MAX=256, D_MAX=384, NG=6, NSLOT=16, NCTX=8, MAXB=256, LDF=512,
    ),
    # Reference only: the smallest useful PYNQ-Z2 build does NOT fit the XC7Z020 (ADR-006: Yosys 123 %
    # of its LUTs). Kept so the flow, the estimate and the emulator tests stay exercised.
    "pynq-z2": dict(
        N=8, L=4, M_DEPTH=64,
        ABUF_LINES=8192,       # 128 KiB  (2 banks x 4096 lines = 16 RAMB36 each)
        WBUF_LINES=4096,       # 64 KiB
        ACC_LINES=4096,        # 64 KiB
        PBUF_LINES=2048,       # 32 KiB
        KV_LINES=8192,         # 128 KiB  (tiny fits on chip; shakespeare uses the DDR-KV mode)
        NB=2,                  # = widest port (N/4 = 2 lines)
        RAM_STYLE=1,           # every bank in RAMB36
        VP=1,
        CTX_MAX=256, D_MAX=384, NG=6, NSLOT=16, NCTX=8, MAXB=256, LDF=512,
    ),
}

CLOCK_MHZ = {"kv260": 200.0, "pynq-z2": 150.0}   # pl_clk0 / FCLK_CLK0 targets (ADR-006: KV260 200 until timed)
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
