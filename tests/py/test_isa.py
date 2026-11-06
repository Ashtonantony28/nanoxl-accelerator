"""ISA (contract C4): encoder/decoder round trip, field validation, golden byte vectors."""
from pathlib import Path

import numpy as np
import pytest

from nanoxl import isa
from nanoxl.isa import C, IsaError

ROOT = Path(__file__).resolve().parents[2]

# Golden vectors: the RTL command decoder (wave 2) must decode these bytes to these fields.
# Bytes are the 16-byte command word, little-endian, written as hex in memory order.
GOLDEN = [
    (C("END"), "01000000000000000000000000000000"),
    (C("IRQ", code=0xBEEF), "c2bb2f00000000000000000000000000"),
    (C("WAIT", queue=1, sem=0, count=1), "43200000000000000000000000000000"),
    (C("SIGNAL", queue=0, sem=31), "041f0000000000000000000000000000"),
    (C("DMA_LOAD", chan=0, sp_buf=1, sp_addr=0x10000, ddr_addr=0x40123400, rows=384, row_bytes=160,
       ddr_stride=160, sp_stride=0), "08010008003412407ff109000a000000"),
    (C("DMA_STORE", chan=2, sp_buf=0, sp_addr=64, ddr_addr=0xFFFFFFF0, rows=4096, row_bytes=65536,
       ddr_stride=(1 << 24) - 1, sp_stride=4095), "89000200f0ffffffffffffffffffffff"),
    (C("GEMM", int4=0, gemv=1, w_trans=0, accumulate=1, a_addr=384, w_buf=0, w_addr=32768, c_addr=4096,
       M=1, N=160, K=384, a_stride=24, c_stride=40), "90020600004000000100f0097f81010a"),
    (C("GEMM", int4=1, gemv=0, w_trans=0, accumulate=0, a_addr=(1 << 20) - 1, w_buf=1, w_addr=(1 << 21) - 1,
       c_addr=(1 << 20) - 1, M=4096, N=4096, K=4096, a_stride=1023, c_stride=1023),
     "50fcffffffffffffffffffffffffffff"),
    (C("VEC", vop=1, dt_in=1, dt_out=1, src0_buf=0, src0_addr=256, src1_buf=0, src1_addr=1024, dst_buf=0,
       dst_addr=256, len=384, imm=-1), "5828000400001000000480bf00f8ff07"),
    (C("REQUANT", out16=1, per_tensor=0, src_addr=8192, dst_addr=768, param_addr=64, M=128, N=96,
       src_stride=24, dst_stride=48), "5900200030004000f0075f8001300000"),
    (C("LN", in16=1, src_addr=0, dst_addr=196608, param_addr=2048, rows=1, cols=384, src_stride=48,
       dst_stride=24), "5a000000001800040080bf80010c0000"),
    (C("SOFTMAX_ONLINE", first=1, last=1, causal=1, src_addr=256, dst_addr=512, ctx_addr=16, rows=64, cols=200,
       diag=-5, src_stride=50, dst_stride=13), "db01020040002000e0078e61ffcbd000"),
    (C("GELU", src_addr=4096, dst_addr=4096, lut_addr=32512, len=1536), "1c0004004000c01ffc17000000000000"),
    (C("KV_APPEND", src_addr=16384, dst_addr=73728, pos=255, heads=6, dh=64, ctx=256, rows=1, src_stride=72),
     "1d00100080847f287efe010090000000"),
    (C("ATTN_TILE", first=1, last=0, q_addr=1, k_addr=2, v_addr=3, o_addr=4, ctx_addr=5, rows=64, len=4096),
     "51010020000006000001001400c0ffff"),
]


@pytest.mark.parametrize("cmd,hexs", GOLDEN, ids=[f"{c.op}{i}" for i, (c, _) in enumerate(GOLDEN)])
def test_isa_golden_vectors(cmd, hexs):
    assert isa.encode(cmd).hex() == hexs
    assert isa.decode(bytes.fromhex(hexs)) == cmd


def test_isa_golden_hand_checked():
    # WAIT(queue=ARRAY, sem=0, count=1): opcode 0x03 | queue 1 << 6 = 0x43; count field starts at bit 13
    w = int.from_bytes(bytes.fromhex(GOLDEN[2][1]), "little")
    assert w & 0x3F == 0x03 and (w >> 6) & 3 == 1 and (w >> 8) & 31 == 0 and (w >> 13) & 0xFF == 1
    # DMA_LOAD: ddr_addr sits at bits [63:32] when chan/sp_buf/sp_addr precede it
    w = int.from_bytes(bytes.fromhex(GOLDEN[4][1]), "little")
    assert (w >> 32) & 0xFFFFFFFF == 0x40123400 and (w >> 64) & 0xFFF == 383  # rows stored minus one


def _random_cmd(rng, op):
    f = {}
    for n, w, k in isa.LAYOUTS[op]:
        lo, hi = isa.field_range(w, k)
        f[n] = int(rng.integers(lo, hi + 1))
    if op in ("WAIT", "SIGNAL"):
        f["queue"] = int(rng.integers(0, 3))
    if op == "WAIT":
        f["count"] = max(1, f["count"])
    for b in ("sp_buf", "src0_buf", "src1_buf", "dst_buf"):
        if b in f:
            f[b] = int(rng.integers(0, 5))
    if op == "VEC":
        f["vop"] = int(rng.integers(0, len(isa.VOPS)))
        f["dt_in"], f["dt_out"] = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    if op == "GEMM" and f["int4"]:
        f["w_trans"] = 0
        f["N"] -= f["N"] % 2
    return C(op, **f)


def test_isa_roundtrip_random():
    rng = np.random.default_rng(1)
    for op in isa.OPCODES:
        for _ in range(200):
            c = _random_cmd(rng, op)
            b = isa.encode(c)
            assert len(b) == 16
            d = isa.decode(b)
            assert d == c, (c, d)
            assert isa.encode(d) == b


def test_isa_roundtrip_extremes():
    for op in isa.OPCODES:
        for pick in (0, 1):
            f = {}
            for n, w, k in isa.LAYOUTS[op]:
                f[n] = isa.field_range(w, k)[pick]
            if op in ("WAIT", "SIGNAL"):
                f["queue"] = 2
            if op == "WAIT":
                f["count"] = max(1, f["count"])
            for b in ("sp_buf", "src0_buf", "src1_buf", "dst_buf"):
                if b in f:
                    f[b] = 4 * pick
            if op == "VEC":
                f["vop"], f["dt_in"], f["dt_out"] = 8 * pick, 2 * pick, 2 * pick
            if op == "GEMM" and pick:
                f["w_trans"] = 0
            c = C(op, **f)
            assert isa.decode(isa.encode(c)) == c


def test_isa_stream_roundtrip():
    cmds = [c for c, _ in GOLDEN]
    assert isa.decode_stream(isa.encode_stream(cmds)) == cmds


@pytest.mark.parametrize("op", list(isa.OPCODES))
def test_isa_field_range_validation(op):
    for n, w, k in isa.LAYOUTS[op]:
        lo, hi = isa.field_range(w, k)
        for bad in (lo - 1, hi + 1):
            c = C(op)
            if op in ("WAIT",):
                c.f["count"] = 1
            c.f[n] = bad
            with pytest.raises(IsaError):
                isa.encode(c)


def test_isa_semantic_validation():
    with pytest.raises(IsaError):
        isa.encode(C("GEMM", int4=1, w_trans=1, N=2))
    with pytest.raises(IsaError):
        isa.encode(C("GEMM", int4=1, N=3))
    with pytest.raises(IsaError):
        isa.encode(C("WAIT", queue=0, sem=1, count=0))
    with pytest.raises(IsaError):
        isa.encode(C("SIGNAL", queue=isa.Q_CTRL))
    with pytest.raises(IsaError):
        isa.encode(C("DMA_LOAD", sp_buf=7))
    with pytest.raises(IsaError):
        isa.encode(C("VEC", vop=31))
    with pytest.raises(IsaError):
        isa.encode(C("VEC", dt_in=3))
    with pytest.raises(IsaError):
        C("GEMM", bogus=1)
    with pytest.raises(IsaError):
        isa.encode(C("GEMM", M=True))  # bools are not field values


def test_isa_decoder_rejects_garbage():
    with pytest.raises(IsaError):
        isa.decode(bytes(16))  # opcode 0 is illegal (zeroed memory must not execute)
    with pytest.raises(IsaError):
        isa.decode(bytes.fromhex("3f" + "00" * 15))
    with pytest.raises(IsaError):
        isa.decode(bytes(15))
    end_with_junk = bytearray(16)
    end_with_junk[0] = isa.OPCODES["END"]
    end_with_junk[15] = 0x80
    with pytest.raises(IsaError):
        isa.decode(bytes(end_with_junk))
    with pytest.raises(IsaError):
        isa.decode_stream(bytes(17))


def test_isa_layouts_fit_and_are_unique():
    assert len(set(isa.OPCODES.values())) == len(isa.OPCODES)
    assert all(v < 1 << isa.OPCODE_BITS for v in isa.OPCODES.values())
    for op in isa.OPCODES:
        bm = isa.bit_map(op)
        assert bm[-1][2] < 128
        for (_, l0, h0, _), (_, l1, _, _) in zip(bm, bm[1:]):
            assert l1 == h0 + 1


def test_isa_docs_tables_in_sync():
    doc = (ROOT / "docs" / "isa.md").read_text()
    assert isa.markdown_tables() in doc, "regenerate the field tables in docs/isa.md from isa.markdown_tables()"
    for c, h in GOLDEN:
        assert h in doc


def test_isa_fields_cover_v1_models():
    """Worst-case field values for the largest v1 model (m25: d=512, ctx=1024, d_ff=2048) and the
    biggest scratchpads fit. The compiler test compiles m25 end to end as well."""
    from nanoxl.graph import M25
    d, ctx, dff = M25.d_model, M25.ctx, M25.d_ff
    isa.encode(C("GEMM", a_stride=dff // 16, c_stride=4 * ctx // 16, M=ctx, N=ctx, K=dff))
    isa.encode(C("KV_APPEND", pos=ctx - 1, heads=M25.n_head, dh=M25.d_head, ctx=ctx, rows=ctx,
                 src_stride=3 * d // 16, dst_addr=(1 << 21) - 1))
    isa.encode(C("SOFTMAX_ONLINE", rows=ctx, cols=ctx, diag=ctx - 1, src_stride=4 * ctx // 16, dst_stride=ctx // 16))
    isa.encode(C("DMA_LOAD", rows=ctx, row_bytes=ctx * M25.d_head, ddr_stride=ctx * M25.d_head))
    isa.encode(C("VEC", len=ctx * d))
