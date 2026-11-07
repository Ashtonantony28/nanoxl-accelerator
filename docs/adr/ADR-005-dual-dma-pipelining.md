# ADR-005: Second DMA master (ISA `chan`), DMA command pipelining, SIGNAL carried on done

Status: **accepted and implemented** (SG10a / M8, 2026-11-07). Extends contract C3 (control plane,
`docs/control.md`) and the board integration (`docs/top.md`). The ISA (C4) is unchanged: DMA
commands already had a `chan` field (word bit 6), which was ignored until now.
Evidence: `tests/rtl/test_ctrl.py::ctrl_dual_dma` and the randomized ctrl streams (two DMA engines,
scoreboard on every SIGNAL/WAIT edge), `tests/rtl/test_dma.py::dma_pipelined_loads`, formal job
`queue` (`formal/run_formal.py`), `tests/py/test_dma_channels.py`, `tests/rtl/test_top.py`
(bit-exact tiny end to end, all four bundles/KV modes), `tests/py/test_cyclemodel.py`.

## Context

ADR-003 and AUDIT-1 ranked the DMA as the KV260 decode bottleneck. There was one 128-bit master
(16 B/cycle = 4 GB/s at 250 MHz, half the board's DDR), and it ran one command at a time: each
command paid the full DDR latency, and a SIGNAL in the DMA queue blocked the queue until every
earlier DMA command was done. The modelled shakespeare W8A8 decode was 332 tok/s, against a 752 tok/s
DDR weight bound.

## Decision

### 1. One DMA queue, two engines (not two queues)
`cmd_proc` keeps **one** DMA command queue. It dispatches to two `dma_engine`s by `chan[0]`: ch0 is
on `md_axi` and scratchpad port 0, ch1 is on `md1_axi` and port 7/3. Both engines share the queue's
SIGNAL/WAIT state.

* Same-channel commands issue and complete in program order. Different channels run concurrently
  and may complete out of order.
* `DMA_STORE` (0x09) is a **fence** across both channels. It issues only when nothing is outstanding
  on either channel, and nothing issues while it runs. This keeps a store's read-after-write on
  scratchpad data that either channel loaded, without per-buffer tracking.
* A DMA **SIGNAL** covers both channels: it fires once every earlier DMA command on *both*
  channels is done. A DMA **WAIT** holds both channels.

Rejected: two queues (one per channel), each with its own SIGNAL/WAIT. That would double the
semaphore state and make the compiler pair SIGNALs per channel. It would also change the semantics of
existing programs, because a SIGNAL would cover only half of the earlier DMAs. With one queue,
every existing program means the same thing: all of them use `chan = 0`, so they are cycle
compatible apart from pipelining.

### 2. SIGNAL carried on done
A SIGNAL at the queue head no longer blocks. It is moved into a pending buffer (PSIG = 4 entries).
Each entry records how many commands are still outstanding on each channel, and each channel's
`done` decrements its count. Pending SIGNALs fire in order once their counts reach zero. When nothing
is pending or outstanding, a SIGNAL fires directly (bypass). The same queue logic serves ARRAY and VEC
(NCH = 1), so their SIGNALs are carried too. A full pending buffer stalls the head, which costs
throughput but not correctness.

### 3. Pipelined DMA engine
`dma_engine` accepts up to NCMD = 4 commands. An address stage issues the AR bursts of the next
command while the data stage still receives the previous one, through descriptor FIFOs `aq` and `rq`.
LOADs therefore overlap DDR latency. A STORE runs alone: it is accepted only into an empty engine,
and nothing is accepted after it until it is done.

The realigner bubble (one cycle per row with an unaligned DDR offset) was **not** removed. The
compiler coalesces contiguous rows into commands of up to 64 KiB, so one bubble per command is
below 0.1 % of decode time.

### 4. Board mapping
| board | `mf_axi` (fetch) | `md_axi` (DMA ch0) | `md1_axi` (DMA ch1) |
|---|---|---|---|
| KV260 (ZU5EV) | S_AXI_HPC0_FPD | S_AXI_HP0_FPD | S_AXI_HP1_FPD |
| PYNQ-Z2 (Z7020) | shares S_AXI_HP0 (2:1 SmartConnect, low traffic) | S_AXI_HP0 | S_AXI_HP1 |

The RTL masters are 128-bit. The KV260 HP ports are 128-bit, so each master moves 16 B/cycle, and the
two together reach the board's 8 GB/s DDR. On PYNQ-Z2 the HP ports are 64-bit (`axi_port_bytes = 8` in
the cycle model, through a width converter), and the 1.5 GB/s DDR is the limit. In both cases the
cycle model shares the DDR bandwidth between the channels when both stream.

### 5. Cross-channel hazards: rejected, not ordered
Only these order ch0 against ch1:
* same-channel program order
* a STORE fence
* a DMA SIGNAL / WAIT
* a barrier

Two LOADs on different channels that write overlapping scratchpad bytes, with none of these between
them, are a write-after-write **race**. `funcsim.check_hazards` rejects such a program. The hardware
does not order them.

The compiler's `SyncBuilder` avoids races in two ways: it moves the second LOAD to the first LOAD's
channel, or orders the two with a DMA→DMA semaphore (sem 6). In the compiled tiny and shakespeare
programs no LOAD needs to move.

### 6. Compiler channel policy
Each weight-panel command (`WBUF`, W8 and dense W4; `CH_ALT`) goes to the channel that has moved
fewer bytes so far, so each channel carries 40–60 % of the weight bytes. KV loads use `CH_KV | kind`
(K on ch0, V on ch1). Stores use ch0; they fence both channels anyway. Programs with `chan = 0` everywhere stay valid.

## Consequences

* The CSR map is unchanged. PERF_RETIRED and PERF_BYTES count both channels, an AXI error on
  either engine raises the DMA error code, and the processor is idle only when both channels are.
* `cmd_queue` now has its own FIFO (the same behaviour as `fifo_sync`) so the formal proof can
  see its entries. The proof (`queue`: QDEPTH 2, MAXOUT 2, NCH 2, PSIG 2) shows by k-induction
  that a carried SIGNAL fires only after every earlier engine command on each channel is done.
* Measured, the tiny `test_top` decode is 8.8 % faster (W8A8) at 75-cycle DDR latency (§Measurements).
  Tiny is array/VEC bound at N = 16, so the large gain is the modelled one at shakespeare scale.

## Measurements

See `docs/top.md` §6 for the full tables. Modelled shakespeare (`estimate(SHAKESPEARE, board)`,
`PRE_SG10A` vs built):

| board, W | decode tok/s before → after | DDR weight bound | prefill ms before → after |
|---|---|---|---|
| KV260, W8 | 332 → 619 | 752 | 14.73 → 14.01 |
| KV260, W4 | 594 → 651 | 1500 | 14.75 → 14.31 |
| PYNQ-Z2, W8 | 99 → 128 | 141 | 88.1 → 84.9 |
| PYNQ-Z2, W4 | 175 → 186 | 281 | 85.5 → 84.0 |

On KV260, W4 decode is now array bound (one weight row per cycle). The next fix in
`bench/results/fixes.md` is the 2-row array weight port.
