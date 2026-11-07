"""Accelerator configuration shared by the compiler (capacities, tiling) and the cycle model (timing)."""
from __future__ import annotations

from dataclasses import dataclass, replace

KiB = 1024
MiB = 1024 * KiB


@dataclass(frozen=True)
class HwConfig:
    name: str = "kv260"
    array_n: int = 32            # N x N weight-stationary array
    clock_mhz: float = 250.0
    lanes: int = 64              # vector-unit lanes (elements per cycle)
    ddr_gbps: float = 8.0        # achievable PL<->DDR bandwidth, GB/s (1e9 bytes)
    ddr_latency_ns: float = 300.0
    burst_bytes: int = 256       # AXI burst size the DMA issues (one burst per cycle max)
    # scratchpad capacities (bytes)
    abuf: int = 512 * KiB        # activations (int8 / int16)
    wbuf: int = 128 * KiB        # weight ring (>= 2 panels => double buffering)
    acc: int = 256 * KiB         # int32 accumulators
    pbuf: int = 64 * KiB         # requant / LN / softmax params, GELU LUTs (ring)
    kvbuf: int = 1536 * KiB      # on-chip KV tier (URAM), or a staging ring in DDR-KV mode
    kv_onchip: bool = True
    panel_bytes: int = 32 * KiB  # target weight-panel size; the WBUF ring holds wbuf/panel panels in flight
    # micro-architecture timing knobs
    wload_rows_per_cycle: int = 2   # weight-buffer rows (N bytes each) moved into the array per cycle (ADR-003)
    fill_drain: int = 0             # 0 => 2*N (skewed); GEMV mode uses N
    issue_cycles: int = 2           # command processor: cycles per 16-byte command
    fifo_depth: int = 32            # per-queue command FIFO depth
    vec_latency: int = 8            # pipeline latency of vector ops
    row_latency_ln: int = 24        # per-row reduction + rsqrt latency
    row_latency_softmax: int = 24   # per-row max/sum reduction + reciprocal latency
    sfu_row_bubble: int = 4         # per-row stall between LN / softmax passes not hidden by row interleaving (estimate)

    def with_(self, **kw) -> "HwConfig":
        return replace(self, **kw)

    @property
    def clock_hz(self) -> float:
        return self.clock_mhz * 1e6

    @property
    def ddr_bytes_per_cycle(self) -> float:
        return self.ddr_gbps * 1e9 / self.clock_hz

    @property
    def peak_gops(self) -> float:
        return 2 * self.array_n ** 2 * self.clock_hz / 1e9

    @property
    def onchip_bytes(self) -> int:
        return self.abuf + self.wbuf + self.acc + self.pbuf + self.kvbuf


KV260 = HwConfig()
# PYNQ-Z2: 630 KB of BRAM in total, no URAM: KV cache lives in DDR with a small staging ring.
PYNQ_Z2 = HwConfig(name="pynq-z2", array_n=16, clock_mhz=150.0, lanes=16, ddr_gbps=1.5,
                   ddr_latency_ns=400.0, burst_bytes=128,
                   abuf=192 * KiB, wbuf=64 * KiB, acc=96 * KiB, pbuf=32 * KiB, kvbuf=96 * KiB, panel_bytes=16 * KiB, wload_rows_per_cycle=1,
                   kv_onchip=False)
BOARDS = {"kv260": KV260, "pynq-z2": PYNQ_Z2}
