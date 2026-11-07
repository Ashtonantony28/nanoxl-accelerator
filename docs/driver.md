# Host driver (`nanoxl/driver/`, SG4c)

The PYNQ-side Python API from design-plan §6.5. It loads a bundle and builds the DDR image with the
compiler. For each step it writes the compiled program to the command ring, rings the doorbell,
waits for END, reads the int16 logits and samples the next token on the host. The same code drives
the board and an emulator, because everything the device sees goes through a `Backend`: 32-bit CSR
reads and writes, plus one physically contiguous DDR buffer.

| file | what |
|---|---|
| `driver/csr.py` | CSR offsets, bit fields and error codes (docs/control.md §5) |
| `driver/backends.py` | `Backend` interface, `SimBackend` (funcsim plus CSR emulation), `PynqBackend` (board, thin, imported lazily) |
| `driver/host.py` | `NanoXL` (image load, ring, submit / wait, step, prefill, `generate`), `sample`, `CharCodec` |
| `driver/__main__.py` | CLI |

Tests: `tests/py/test_driver.py` (`python3 -m pytest -q tests/py -k driver`, about 20 s).

## 1. API

```python
from nanoxl.driver import open_bundle, CharCodec, SimBackend, PynqBackend
dev = open_bundle("artifacts/models/tiny/w8a8.npz")             # SimBackend on the KV260 config
# dev = open_bundle(path, PynqBackend("nanoxl.bit"), hw=KV260)   # on the board
codec = CharCodec()
toks = dev.generate(codec.encode("ROMEO:"), 50)                  # greedy, bit-exact vs nanoxl.golden
toks = dev.generate(codec.encode("ROMEO:"), 50, top_k=10, temperature=0.8, seed=1)
print(codec.decode(toks), dev.perf(), dev.saturation())
```

* `NanoXL(backend, arrays, meta, hw, n_slots=1, timeout=10, use_irq=False, **compiler_kw)`. It checks
  the ID register, allocates the buffer (`backend.alloc(image.size)`) and copies the image chunks in.
  It then soft-resets the device and programs RING_BASE / RING_SIZE (the image's `cmd_ring` region, 4 MiB
  = 262,144 words by default) and IRQ_EN (END | error).
* **Relocation.** The compiler builds the image at `0x4000_0000`. The driver shifts every DDR address of
  every program by `phys - base` (`compiler.relocate`), so any 4 KiB-aligned CMA buffer works. Host I/O
  (the `x` input rows and the `logits` output) is addressed by buffer offset.
* `step(tokens_per_slot)`: compiles one program for this step, writes `x0 = sat16(emb.tok[t] + emb.pos[p])`
  (embedding done on the host, counted as `resid`), then calls `submit` and `wait` and reads the logits.
  `prefill(prompt, chunk)` and `generate(prompt, n, top_k, temperature, seed, chunk)` build on it.
  `generate` resets the KV positions when it is called again.
* `submit(cmds)` relocates and encodes the program (aligned) and writes it at the ring tail. If the
  program crosses the ring end, it is written in two pieces. `submit` keeps one slot free and waits for
  room by reading RING_HEAD. It writes DOORBELL = the new tail and returns the END_COUNT value that marks
  completion. `wait(n)` polls END_COUNT (or, with `use_irq`, waits on the irq line), clears the END bit
  (W1C) and raises `DriverError(code, index, words)` when ERR_CODE is set, or `DriverTimeout` when the
  timeout expires. Recovery is `reset()`, which does a soft reset and reprograms the ring.
* `sample(logits, top_k, temperature, rng, scale)`. Greedy is argmax with ties going to the lowest
  index, which is the golden model's rule. Top-k computes softmax(`scale · logit / T`) over the k
  largest logits, with `scale` = the bundle's `scales.logits`.
* `perf()` reads PERF_CYCLES / BUSY_* / RETIRED / BYTES. `saturation()` reads SAT_0..3
  (requant, ALU, LN z, LN out). `clear_counters()` writes PERF_CTRL = 3. `last_perf` holds the
  counter deltas of the last step.

CLI:

```
python3 -m nanoxl.driver --bundle artifacts/models/tiny/w8a8.npz --prompt "ROMEO:" --n 50
python3 -m nanoxl.driver --bundle ... --tokens 12,0,0,19 --n 50 --json        # token ids in, JSON out
python3 -m nanoxl.driver --bundle ... --backend pynq --bitstream nanoxl.bit --prompt "ROMEO:"
```

On the sim backend the stderr line also reports the modelled device time. That value is the sum of
PERF_CYCLES, which the SimBackend fills in from the cycle model.

## 2. SimBackend: what it emulates

The SimBackend runs the ring contents with `nanoxl.funcsim` on the driver's own DDR buffer (shared,
in place). The scratchpads and the on-chip KV tier persist between programs, as on the hardware. The
CSR semantics follow docs/control.md §1 and §5:

* **Ring.** Words in `[head, tail)` are fetched from `RING_BASE + 16·i` and wrap at RING_SIZE. Writing
  DOORBELL with RING_SIZE = 0 or tail ≥ RING_SIZE raises error 6 (RING).
* **Queues.** Three in-order queues of depth 8. Dispatch moves one word per step in stream order. A
  full queue blocks the dispatcher. After every dispatch, each queue head advances as far as it can:
  an engine command executes in funcsim (done = results visible); `WAIT(s, k)` needs at least k, then
  subtracts k; `SIGNAL` stalls while the counter is at 255. So a WAIT whose SIGNAL never comes, or more
  than 255 + 8 outstanding SIGNALs, leaves the machine stuck the way the RTL is stuck: STATUS busy,
  no error. The driver then times out.
* **END / IRQ** are barriers that execute only when all queues are empty. END sets IRQ_STATUS bit 0 and
  increments END_COUNT. IRQ(code) sets bit 1 and IRQ_CODE. Neither stops dispatch: if tail has already
  moved past END, the next program runs.
* **Errors** halt dispatch and latch the first error. A word that `isa.decode` rejects gives error 1
  (OPCODE), 2 (RESV) or 3 (FIELD), with ERR_INDEX and ERR_W0..3. A buffer-range fault in funcsim gives
  error 7 (ENGINE). A failed ring fetch gives error 4. Every halt also sets IRQ_STATUS bit 2. Commands
  dispatched before the bad word complete; none after it run.
* **IRQ_STATUS** is W1C, and the irq line is `|(IRQ_STATUS & IRQ_EN)`.
* **Soft reset** (CTRL bit 1) completes at once and reads back 0. It clears head = tail = 0, ERR, the
  queues, the semaphores and the SFU tables (`eng_clr`). It keeps RING_BASE / SIZE, IRQ_EN, IRQ_STATUS,
  the perf counters and the scratchpad contents.
* **Single step** (CTRL bit 0): each DBG_STEP write dispatches one word or executes one barrier.
  DBG_LAST holds the ring index of the last dispatched word. SEM_SEL / SEM_VAL read one semaphore.
* **Counters.** PERF_RETIRED counts every word. PERF_BYTES counts DMA bytes. PERF_CYCLES and
  PERF_BUSY_DMA/ARR/VEC come from `cyclemodel.simulate` over the words dispatched since the queues were
  last idle, so they are modelled, not measured. SAT_0..3 are the funcsim saturation events per unit,
  32-bit saturating. PERF_CTRL bit 0 clears the perf counters, and bit 1 clears the SAT counters.

Not emulated: AXI timing and fetch errors from RRESP, queue overflow (STATUS bit 15), and the soft-reset
busy window. The PYNQ backend has no cache-coherence modelling either, because the CMA buffer is
allocated uncached.

## 3. Board bring-up (KV260): UNTESTED until hardware exists

Everything in this section is a plan written from the PYNQ / Kria documentation. None of it has been
run. SG9 owns the overlay; the driver side is `PynqBackend`.

1. **OS image.** Flash the Ubuntu 22.04 image for Kria (Certified Ubuntu for AMD Kria, KV260) to a
   microSD card of at least 32 GB. Boot it, then update the boot firmware if the Kria docs require it
   for that image (`sudo xmutil bootfw_update` per the Kria wiki).
2. **PYNQ for Kria.** `git clone --recursive https://github.com/Xilinx/Kria-PYNQ && cd Kria-PYNQ &&
   sudo bash install.sh -b KV260`. This installs PYNQ into a venv (`source /etc/profile.d/pynq_venv.sh`)
   and a Jupyter server. Pin the PYNQ version in an ADR (design-plan §11).
3. **Overlay files.** Copy `nanoxl.bit` and `nanoxl.hwh`, which share a base name, from the Vivado
   project (SG9, `fpga/kv260/`) to the board. The block design must expose:
   * the ctrl_top AXI4-Lite slave on HPM0_FPD. Its name in `Overlay.ip_dict` is the `--ip` argument
     (default `nanoxl_0`);
   * the fetch master and the DMA master into S_AXI_HP0/HP1_FPD at 128 bits. Use one SmartConnect or two
     ports. Two DMA masters on two ports are what the 8 GB/s assumption in ADR-003 needs;
   * `irq` to pl_ps_irq0, if `use_irq` is wanted.
   Run `xmutil unloadapp` first, so that the default k26 app does not hold the PL.
4. **CMA buffer.** The DDR image is `image.size` bytes (tiny about 5 MiB with the 4 MiB command ring;
   shakespeare about 15 MiB). `pynq.allocate(shape=(n,), dtype=uint8, cacheable=False)` returns a
   physically contiguous, 4 KiB-aligned buffer, and the driver relocates every program to its
   `physical_address`. If the allocation fails, raise the CMA pool with the `cma=` boot argument
   (for example `cma=512M`) or the device tree.
5. **First contact.** In this order:
   1. read ID (0x4E584C43) and VERSION;
   2. write and read back RING_BASE / RING_SIZE;
   3. one `END` program: END_COUNT goes to 1 and IRQ_STATUS bit 0 is set;
   4. a DMA bandwidth program (large DMA_LOAD / DMA_STORE), to measure the PL DDR bandwidth before
      anything else (design-plan §11);
   5. the tiny model:
      `python3 -m nanoxl.driver --backend pynq --bitstream nanoxl.bit --bundle w8a8.npz --tokens 12,0,0,19,30,17,25,21,27,10,0,19,53,53,42,1 --n 100 --json`.
      The tokens must equal `artifacts/models/tiny/golden_ref.json`, because the device is bit-exact
      with the golden model.
6. **Debug.** If a run halts, the ERR_* registers give the code and the ring index of the bad word.
   Single step (CTRL bit 0 plus DBG_STEP) and SEM_VAL localize a hang. Compare PERF_* with
   `cyclemodel` (M6 target: within 10 %) and SAT_* with the golden saturation counts. Reproduce
   anything suspicious in the SimBackend, then in Verilator, before debugging on the board.

PYNQ-Z2 (Zynq-7000) works the same way with the PYNQ v3 SD image. Its HP ports are 64 bits wide (see
ADR-003), and the CMA pool is smaller. Also untested.
