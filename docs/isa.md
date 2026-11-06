# NanoXL command ISA and DDR image format (contract C4)

Status: M1 draft, owned by SG4. Executable form: `nanoxl/isa.py` (encoder/decoder, validation),
`nanoxl/funcsim.py` (semantics), `nanoxl/cyclemodel.py` (timing). Tests: `tests/py/test_isa.py`.
The field tables at the end of this file are generated from `nanoxl.isa.LAYOUTS` by
`isa.markdown_tables()`; a test fails if they drift.

## 1. Command word

* Every command is one **128-bit word, little-endian** (byte 0 holds bits [7:0]). A command ring in
  DDR is a plain array of words, so a 128-bit AXI read returns exactly one command.
* Bits [5:0] are the **opcode**. Opcode 0x00 is illegal, so zeroed memory never executes.
* The remaining fields follow LSB-first in the order of the table for that opcode. Bits above the
  last field are reserved and **must be zero**; the decoder rejects a word with non-zero reserved bits.
* Field kinds: `u` unsigned; `m1` a count stored minus one (12 bits encode 1..4096, so a zero count
  cannot be expressed); `s` two's complement.

## 2. Execution model

* The command processor fetches words in order and dispatches each to one of three **in-order
  queues**: DMA, ARRAY (systolic array) and VEC (vector unit + SFUs). CTRL commands (`IRQ`, `END`)
  are barriers: they wait until all three queues have drained.
* Within a queue, commands behave as if executed one after another (the hardware interlocks
  same-queue hazards, e.g. consecutive accumulating GEMMs into the same accumulator rows).
* Across queues there is **no implicit ordering**. `SIGNAL(queue=q, sem=s)` is executed by queue q
  once all earlier commands of q have completed, and increments counting semaphore s.
  `WAIT(queue=q, sem=s, count=k)` blocks queue q until semaphore s is at least k, then subtracts k.
  32 semaphores exist; counters are at least 8 bits wide (the cycle model reports the largest
  number of outstanding signals; it is 1-2 for compiler output).
* **Compiler invariants** (checked by `funcsim.check_hazards` and the cycle model):
  1. every semaphore has a single producer queue and a single consumer queue;
  2. the SIGNALs a WAIT consumes appear *earlier in the stream* than the WAIT. With in-order
     dispatch this makes the stream deadlock-free for any queue FIFO depth >= 1;
  3. every pair of conflicting accesses (RAW, WAR, WAW on any 16-byte block of any scratchpad or of
     DDR) from different queues is ordered by a SIGNAL -> WAIT chain.
  The semaphore numbering used by the compiler is `sem = index of (producer, consumer)` in
  DMA->ARRAY 0, DMA->VEC 1, ARRAY->DMA 2, ARRAY->VEC 3, VEC->DMA 4, VEC->ARRAY 5.
* `END` raises the done interrupt and stops fetching; `IRQ(code)` raises an interrupt with `code`
  in a CSR and continues.

## 3. Scratchpads and addressing

| id | name | contents | element types | address bits |
|---|---|---|---|---|
| 0 | ABUF | activations, attention probabilities | int8, int16 | 20 (1 MiB) |
| 1 | WBUF | weight ring (double-buffered panels) | int8, packed int4 (C1b) | 21 (2 MiB) |
| 2 | ACC | int32 accumulator tiles, attention scores | int32 | 20 |
| 3 | PBUF | requant records, LN blocks, softmax ctx, GELU LUTs | mixed | 20 |
| 4 | KVBUF | on-chip KV tier, or KV staging ring in DDR-KV mode | int8 | 21 |

* Addresses are **byte addresses** within the named buffer. A command that can only touch one
  buffer for an operand (e.g. GEMM's A operand is always ABUF) carries just the address.
* All scratchpad **row strides are in units of 16-byte lines** (`isa.LINE`); a stride of 0 in
  `DMA_*.sp_stride` means "rows packed back to back" (stride = row_bytes). The compiler pads tensor
  row strides to 16 bytes. Base addresses may be byte-granular (column offsets of tiles); wave-2
  RTL may require 16-byte alignment of bases, which the compiler already gives for every tensor
  base and every tile column offset that is a multiple of 16 bytes.
* DDR addresses are 32-bit physical (PL view). `compiler.relocate()` rebases a stream when the
  driver's CMA buffer lands somewhere else.

### Tensor layouts
* Activations `[M, width]` row-major with a padded row stride. Weights in WBUF: a panel
  `[K, N]` row-major and dense (`N` bytes per row, or `N/2` bytes packed int4 with C1b nibble
  order). `GEMM.w_trans=1` reads the operand as `[N, K]` (used for `K^T` straight from the KV
  cache). Accumulator tiles `[M, N]` int32 with `c_stride`.
* KV cache: `[slot][layer][K|V][head][pos][d_head]` int8, both on chip and in DDR, so one head's K
  for all positions is one contiguous burst.
* PBUF records: requant record = 16 bytes `{int32 bias, int32 mult, int32 shift, pad}` per output
  channel (`REQUANT.per_tensor=1` reads only record 0). LN block = `float32 {in_scale, out_scale,
  eps, 0}` then `gamma[cols]`, `beta[cols]` (float32 **placeholder**, replaced when ADR-002 freezes
  C5). Softmax ctx = 16 bytes (`float32 scale`, placeholder) followed by 16 bytes of running
  state per row for multi-tile online softmax (SG5/SG8). GELU LUT = 256 int8 indexed by the
  input byte `uint8(x)`.

## 4. Command semantics

| op | queue | semantics |
|---|---|---|
| DMA_LOAD / DMA_STORE | DMA | Move `rows x row_bytes` between DDR (`ddr_stride` bytes per row; 0 = dense) and scratchpad `sp_buf` (`sp_stride` lines; 0 = dense). `chan` is an AXI-port hint: 0 weights/params, 1 activations, 2 KV. A load after a store to overlapping DDR in the DMA queue observes the store. |
| GEMM | ARRAY | `C[M,N] (+)= A[M,K] @ W[K,N]` exact int32 (C1/C1b). A from ABUF (`a_stride` lines); W from WBUF or KVBUF (`w_buf`), dense; C in ACC (`c_stride` lines). `accumulate=0` overwrites, `1` adds (int32 wraparound; cannot overflow for total K <= 4096). The array walks N/n x K/n tiles internally. `gemv=1` selects the low-latency path that skips skew/deskew (a timing hint; results are identical). `int4=1`: W is C1b-packed, N even. `w_trans=1`: W stored `[N,K]`. |
| REQUANT | VEC | `dst[M,N] = requant(src[M,N], records)` per C2, int8 or int16 (`out16`). Per-channel records start at `param_addr`; tile column offset n0 is `param_addr + 16 n0`. |
| VEC | VEC | Elementwise over `len` contiguous elements: COPY, ADD, SUB, MUL, MAX, MIN, ADDI (src0+imm), SHRR (C2 round-half-away right shift by imm), SHL. Computed exactly, saturated to `dt_out`. |
| LN | VEC | Row-wise LayerNorm of `[rows, cols]` int8/int16 (`in16`) to int8. **Float placeholder** until C5. |
| SOFTMAX_ONLINE | VEC | Row-wise softmax of int32 scores to int8 probabilities. `causal`: row i may see column j iff `j <= i + diag`. `first/last` delimit a multi-tile online pass (running max/sum in ctx). **Float placeholder**, single tile only, until C5/SG5. |
| GELU | VEC | `dst[i] = lut[uint8(src[i])]` over `len` bytes. Exact. |
| KV_APPEND | VEC | For `t < rows`, `h < heads`: copy `src[t, h*dh : (h+1)*dh]` to `dst + (h*ctx + pos + t)*dh`. |
| ATTN_TILE | ARRAY | Reserved for SG8: fused flash-decoding tile (QK^T, online softmax, PV) over `len` keys. Encoded and costed, not yet executed by funcsim. |
| WAIT / SIGNAL | named | Semaphores, see section 2. |
| IRQ / END | CTRL | Barrier + interrupt / end of program. |

## 5. Field widths and why they suffice

Sized for `shakespeare` (d=384, ctx=256, 6 layers, d_ff=1536) and for the top of the v1 envelope,
`m25` (d=512, ctx=1024, 8 layers, d_ff=2048, vocab 4096, about 25M parameters).
`test_isa_fields_cover_v1_models` and the compiler's m25 test check the worst cases.

| field | width | range | worst case in v1 |
|---|---|---|---|
| M, N, K, rows, cols (m1) | 12 | 1..4096 | K <= 4096 is contract C1; prefill M <= ctx = 1024 |
| DMA row_bytes (m1) | 16 | 1..65536 | m25 one head of K for all positions: 1024 x 64 = 64 KiB |
| DMA ddr_stride | 24 | 16 MiB | KV head stride ctx x dh = 64 KiB; logits row 16 KiB |
| DMA sp_stride / REQUANT / LN / KV strides | 12 lines | 64 KiB | qkv row 3d = 1.5 KiB |
| GEMM a_stride, c_stride; SOFTMAX strides | 10 lines | 16 KiB | A row d_ff = 2 KiB; scores row 4 x ctx = 4 KiB |
| ABUF / ACC / PBUF address | 20 | 1 MiB | KV260 ABUF 512 KiB |
| WBUF / KVBUF address | 21 | 2 MiB | KV260 URAM (2.25 MiB total) KV tier 1.5 MiB |
| DDR address | 32 | 4 GiB | KV260 has 4 GB; relocatable |
| VEC len (m1) | 20 | 1M elements | residual add of a 1024 x 512 int16 tile = 512K |
| KV pos / ctx | 12 | 4096 | ctx 1024 |
| KV heads (m1) / dh (m1) | 6 / 8 | 64 / 256 | 8 heads, dh 64 |
| SOFTMAX diag (s) | 13 | +-4096 | query offset up to ctx |
| semaphores | 5 | 32 | 6 used |

Growing past these (e.g. GPT-2 124M from DDR, d=768, d_ff=3072, vocab 50257) needs the compiler
to tile the LM head along N (already supported: panels are <= 4092 columns) and nothing else.

## 6. DDR image

Built once per (model, hardware config) by `compiler.Compiler`; regions are 4 KiB aligned, in order:

| region | contents |
|---|---|
| `cmd_ring` | command words (default 4 MiB; the host writes each step's program here) |
| `params` | requant records, LN blocks, softmax ctx + PV record per attention layer, GELU LUTs |
| `weights` | GEMM weights **pre-tiled into panels** `[kc, nt]` (dense, int8 or packed int4) in the order the compiler streams them, so each panel is one contiguous burst |
| `kv` | DDR-KV mode only: `[slot][layer][K/V][head][ctx][dh]` |
| `io_<tensor>` | inputs (host-side embedding rows, int16 residual) and outputs (int32 logits) |

The host does the embedding lookup (`x0 = wte[tok] + wpe[pos]`, a few hundred bytes per token) and
the sampling/argmax in v1.

## 7. Field tables (generated)

#### END (opcode 0x01, queue CTRL)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |

#### IRQ (opcode 0x02, queue CTRL)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| code | [21:6] | u |

#### WAIT (opcode 0x03, queue explicit)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| queue | [7:6] | u |
| sem | [12:8] | u |
| count | [20:13] | u |

#### SIGNAL (opcode 0x04, queue explicit)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| queue | [7:6] | u |
| sem | [12:8] | u |

#### DMA_LOAD (opcode 0x08, queue DMA)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| chan | [7:6] | u |
| sp_buf | [10:8] | u |
| sp_addr | [31:11] | u |
| ddr_addr | [63:32] | u |
| rows | [75:64] | m1 |
| row_bytes | [91:76] | m1 |
| ddr_stride | [115:92] | u |
| sp_stride | [127:116] | u |

#### DMA_STORE (opcode 0x09, queue DMA)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| chan | [7:6] | u |
| sp_buf | [10:8] | u |
| sp_addr | [31:11] | u |
| ddr_addr | [63:32] | u |
| rows | [75:64] | m1 |
| row_bytes | [91:76] | m1 |
| ddr_stride | [115:92] | u |
| sp_stride | [127:116] | u |

#### GEMM (opcode 0x10, queue ARRAY)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| int4 | [6:6] | u |
| gemv | [7:7] | u |
| w_trans | [8:8] | u |
| accumulate | [9:9] | u |
| a_addr | [29:10] | u |
| w_buf | [30:30] | u |
| w_addr | [51:31] | u |
| c_addr | [71:52] | u |
| M | [83:72] | m1 |
| N | [95:84] | m1 |
| K | [107:96] | m1 |
| a_stride | [117:108] | u |
| c_stride | [127:118] | u |

#### ATTN_TILE (opcode 0x11, queue ARRAY)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| first | [6:6] | u |
| last | [7:7] | u |
| q_addr | [27:8] | u |
| k_addr | [48:28] | u |
| v_addr | [69:49] | u |
| o_addr | [89:70] | u |
| ctx_addr | [109:90] | u |
| rows | [115:110] | m1 |
| len | [127:116] | m1 |

#### VEC (opcode 0x18, queue VEC)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| vop | [10:6] | u |
| dt_in | [12:11] | u |
| dt_out | [14:13] | u |
| src0_buf | [17:15] | u |
| src0_addr | [38:18] | u |
| src1_buf | [41:39] | u |
| src1_addr | [62:42] | u |
| dst_buf | [65:63] | u |
| dst_addr | [86:66] | u |
| len | [106:87] | m1 |
| imm | [122:107] | s |

#### REQUANT (opcode 0x19, queue VEC)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| out16 | [6:6] | u |
| per_tensor | [7:7] | u |
| src_addr | [27:8] | u |
| dst_addr | [47:28] | u |
| param_addr | [67:48] | u |
| M | [79:68] | m1 |
| N | [91:80] | m1 |
| src_stride | [103:92] | u |
| dst_stride | [115:104] | u |

#### LN (opcode 0x1a, queue VEC)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| in16 | [6:6] | u |
| src_addr | [26:7] | u |
| dst_addr | [46:27] | u |
| param_addr | [66:47] | u |
| rows | [78:67] | m1 |
| cols | [90:79] | m1 |
| src_stride | [102:91] | u |
| dst_stride | [114:103] | u |

#### SOFTMAX_ONLINE (opcode 0x1b, queue VEC)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| first | [6:6] | u |
| last | [7:7] | u |
| causal | [8:8] | u |
| src_addr | [28:9] | u |
| dst_addr | [48:29] | u |
| ctx_addr | [68:49] | u |
| rows | [80:69] | m1 |
| cols | [92:81] | m1 |
| diag | [105:93] | s |
| src_stride | [115:106] | u |
| dst_stride | [125:116] | u |

#### GELU (opcode 0x1c, queue VEC)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| src_addr | [25:6] | u |
| dst_addr | [45:26] | u |
| lut_addr | [65:46] | u |
| len | [85:66] | m1 |

#### KV_APPEND (opcode 0x1d, queue VEC)

| field | bits | kind |
|---|---|---|
| opcode | [5:0] | u |
| src_addr | [25:6] | u |
| dst_addr | [46:26] | u |
| pos | [58:47] | u |
| heads | [64:59] | m1 |
| dh | [72:65] | m1 |
| ctx | [84:73] | m1 |
| rows | [96:85] | m1 |
| src_stride | [108:97] | u |

## 8. Golden vectors

Bytes in memory order (byte 0 first). The RTL command decoder must decode each to the fields shown.

| command | bytes |
|---|---|
| `END()` | `01000000000000000000000000000000` |
| `IRQ(code=48879)` | `c2bb2f00000000000000000000000000` |
| `WAIT(queue=1, sem=0, count=1)` | `43200000000000000000000000000000` |
| `SIGNAL(queue=0, sem=31)` | `041f0000000000000000000000000000` |
| `DMA_LOAD(chan=0, sp_buf=1, sp_addr=65536, ddr_addr=1074934784, rows=384, row_bytes=160, ddr_stride=160, sp_stride=0)` | `08010008003412407ff109000a000000` |
| `DMA_STORE(chan=2, sp_buf=0, sp_addr=64, ddr_addr=4294967280, rows=4096, row_bytes=65536, ddr_stride=16777215, sp_stride=4095)` | `89000200f0ffffffffffffffffffffff` |
| `GEMM(int4=0, gemv=1, w_trans=0, accumulate=1, a_addr=384, w_buf=0, w_addr=32768, c_addr=4096, M=1, N=160, K=384, a_stride=24, c_stride=40)` | `90020600004000000100f0097f81010a` |
| `GEMM(int4=1, gemv=0, w_trans=0, accumulate=0, a_addr=1048575, w_buf=1, w_addr=2097151, c_addr=1048575, M=4096, N=4096, K=4096, a_stride=1023, c_stride=1023)` | `50fcffffffffffffffffffffffffffff` |
| `VEC(vop=1, dt_in=1, dt_out=1, src0_buf=0, src0_addr=256, src1_buf=0, src1_addr=1024, dst_buf=0, dst_addr=256, len=384, imm=-1)` | `5828000400001000000480bf00f8ff07` |
| `REQUANT(out16=1, per_tensor=0, src_addr=8192, dst_addr=768, param_addr=64, M=128, N=96, src_stride=24, dst_stride=48)` | `5900200030004000f0075f8001300000` |
| `LN(in16=1, src_addr=0, dst_addr=196608, param_addr=2048, rows=1, cols=384, src_stride=48, dst_stride=24)` | `5a000000001800040080bf80010c0000` |
| `SOFTMAX_ONLINE(first=1, last=1, causal=1, src_addr=256, dst_addr=512, ctx_addr=16, rows=64, cols=200, diag=-5, src_stride=50, dst_stride=13)` | `db01020040002000e0078e61ffcbd000` |
| `GELU(src_addr=4096, dst_addr=4096, lut_addr=32512, len=1536)` | `1c0004004000c01ffc17000000000000` |
| `KV_APPEND(src_addr=16384, dst_addr=73728, pos=255, heads=6, dh=64, ctx=256, rows=1, src_stride=72)` | `1d00100080847f287efe010090000000` |
| `ATTN_TILE(first=1, last=0, q_addr=1, k_addr=2, v_addr=3, o_addr=4, ctx_addr=5, rows=64, len=4096)` | `51010020000006000001001400c0ffff` |
