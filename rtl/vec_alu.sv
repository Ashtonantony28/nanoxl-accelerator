// vec_alu.sv -- L-lane elementwise integer ALU. Semantics: docs/vector-unit.md,
// executable reference: nanoxl/vec_ref.py::alu. Bit-exact.
//
// Lanes are int32 (narrower operands arrive sign-extended). Every op computes an exact
// 64-bit signed intermediate w, then saturates w to the per-beat output format
// (fmt 0 = int8, 1 = int16, 2/3 = int32); the result is sign-extended back to 32 bits.
//
//   op  name   w                                  op  name   w
//   0   PASS   a                                  6   SRA    a >>> shamt     (floor)
//   1   ADD    a + b                              7   SRAR   RDivPOT(a, shamt)
//   2   SUB    a - b                              8   SHL    a * 2^shamt
//   3   MUL    RDivPOT(a * b, shamt)              9   CLAMP  min(max(a, lo), hi)
//   4   MAX    max(a, b)                          10-15      reserved = PASS
//   5   MIN    min(a, b)
// RDivPOT = RoundingDivideByPOT (C2b, round half away from zero) on 64-bit values.
//
// All controls (op, fmt, shamt, lo, hi) are per beat, so back-to-back beats of
// different commands may share the pipeline.
//
// Implementation (SG11c, docs/fpga.md C4). The output saturates to at most int32, so the
// datapath carries w clamped to W = 35 bits: clamp35(w) saturates to every fmt exactly like w,
// with the same flag. Every non-MUL op is exact within 35 bits except SHL, which is computed
// as (a << shamt) mod 2^35 plus an overflow test on a's top bits.
// Pipeline depth 3: S1 add / subtract / compare | S2 32-bit shift, rounding, clamp |
//                   S3 output saturation + sat flag. 1 beat per cycle.
// Handshake: whole-pipeline stall, in_ready = !out_valid || out_ready, except while a MUL beat
// is in the multiplier (in_ready = 0).
//
// MUL: the compiler emits no VEC MUL (and vec_seq forwards MUL only with shamt 0), so the
// 32x32 product is built over several cycles from four 17x17 signed partial products (one DSP
// per lane, operands and product registered) instead of a combinational 64x64 multiply:
//   a = aH*2^16 + aL, b = bH*2^16 + bL  (aH, bH signed 16 bits; aL, bL unsigned 16 bits)
//   a*b = aL*bL + (aL*bH + aH*bL)*2^16 + aH*bH*2^32
// then shifted right one bit per cycle (shamt cycles) keeping the round and sticky bits,
// clamped to 34 bits, rounded, and entered into S1 as an already-computed value:
//   edge k: accept (the pipeline gets a bubble)    edges k+1..k+4: partial products (ms 1-4)
//   edges k+2..k+5: accumulate (ms 2-5)            edges k+6..k+5+shamt: shift (ms 6)
//   edge k+6+shamt: result -> S1 (when the pipeline advances); next beat accepted at k+7+shamt.
// A MUL beat therefore costs MUL_CYC = 7 + shamt cycles (7 for shamt 0) with latency
// 9 + shamt; all other ops keep 1 beat per cycle and depth 3. Results are bit-identical to the
// former combinational 64x64 form.
module vec_alu #(
    parameter int L = 4
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [L*32-1:0] in_a,
    input  logic [L*32-1:0] in_b,
    input  logic [3:0]      in_op,
    input  logic [1:0]      in_fmt,
    input  logic [4:0]      in_shamt,
    input  logic [31:0]     in_lo,
    input  logic [31:0]     in_hi,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*32-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic [L-1:0]    out_sat,
    output logic            out_last,
    input  logic            sat_clr,
    output logic [31:0]     sat_count,
    output logic            sat_sticky
);
    localparam logic [3:0] OP_ADD = 4'd1, OP_SUB = 4'd2, OP_MUL = 4'd3,
                           OP_MAX = 4'd4, OP_MIN = 4'd5, OP_SRA = 4'd6, OP_SRAR = 4'd7,
                           OP_SHL = 4'd8, OP_CLAMP = 4'd9;
    localparam int D = 3;
    localparam int W = 35;                 // datapath width: clamp35 of the exact w

    logic [D-1:0] v;
    logic         adv;
    assign adv       = !v[D-1] || out_ready;
    assign out_valid = v[D-1];

    // ---- MUL sequencer (shared by all lanes) ----
    logic [2:0] ms;                    // 0 idle, 1..5 multiply / accumulate, 6 shift + push
    logic [4:0] mcnt;                  // remaining right shifts
    logic       acc_in, mul_in, push;  // accept a non-MUL beat / a MUL beat; MUL result -> S1
    logic [1:0] mfmt;
    logic [4:0] msh;
    logic [L-1:0] mkeep;
    logic       mlast;
    assign in_ready = adv && (ms == 3'd0);
    assign acc_in   = in_valid && in_ready && (in_op != OP_MUL);
    assign mul_in   = in_valid && in_ready && (in_op == OP_MUL);
    assign push     = (ms == 3'd6) && (mcnt == 5'd0) && adv;

    always_ff @(posedge clk) begin
        if (!rst_n) ms <= 3'd0;
        else if (mul_in) ms <= 3'd1;
        else if (ms == 3'd6) begin
            if (push) ms <= 3'd0;
        end else if (ms != 3'd0) ms <= ms + 3'd1;
        if (mul_in) begin
            mfmt  <= in_fmt;
            msh   <= in_shamt;
            mkeep <= in_keep;
            mlast <= in_last;
        end
        if (ms == 3'd5) mcnt <= msh;
        else if (ms == 3'd6 && mcnt != 5'd0) mcnt <= mcnt - 5'd1;
    end

    // Sideband: stage 1 and 2 controls, keep/last for all stages.
    logic [3:0]  op1;
    logic [1:0]  fmt1, fmt2;
    logic [4:0]  sh1;
    logic signed [31:0] lo1, hi1;
    logic [L-1:0] keep_p [D];
    logic [D-1:0] last_p;

    always_ff @(posedge clk) begin
        if (!rst_n) v <= '0;
        else if (adv) v <= {v[D-2:0], acc_in || push};
    end
    always_ff @(posedge clk) begin
        if (adv) begin
            op1  <= push ? OP_MUL : in_op;      // S2 passes a MUL result through
            fmt1 <= push ? mfmt : in_fmt;
            sh1  <= in_shamt;
            lo1  <= in_lo;
            hi1  <= in_hi;
            fmt2 <= fmt1;
            keep_p[0] <= push ? mkeep : in_keep;
            for (int k = 1; k < D; k++) keep_p[k] <= keep_p[k-1];
            last_p <= {last_p[D-2:0], push ? mlast : in_last};
        end
    end

    genvar i;
    generate
        for (i = 0; i < L; i++) begin : g_lane
            logic signed [31:0]  a32, b32;
            logic signed [W-1:0] w0, w1, w2c, w2, mres;
            logic signed [31:0]  w1l, cl1, cl2;
            logic signed [31:0]  o3;
            logic                s3;

            // ---- multiplier: registered operands, one 17x17 product per cycle ----
            logic        [31:0] ma, mb;
            logic signed [16:0] mx, my;
            logic signed [33:0] mp;         // product register (DSP MREG / PREG)
            logic signed [35:0] macc, msum;
            logic        [15:0] mlo0, mlo1;
            logic signed [63:0] mP;         // product, then shifted right one bit per cycle
            logic               mrb, mst;   // last bit shifted out / OR of the earlier ones
            // ms = 1: aL*bL, 2: aL*bH, 3: aH*bL, 4: aH*bH
            assign mx = (ms == 3'd1 || ms == 3'd2) ? $signed({1'b0, ma[15:0]}) : $signed({ma[31], ma[31:16]});
            assign my = (ms == 3'd1 || ms == 3'd3) ? $signed({1'b0, mb[15:0]}) : $signed({mb[31], mb[31:16]});
            assign msum = macc + {{2{mp[33]}}, mp};
            always_ff @(posedge clk) begin
                if (mul_in) begin
                    ma <= in_a[i*32 +: 32];
                    mb <= in_b[i*32 +: 32];
                end
                if (ms >= 3'd1 && ms <= 3'd4) mp <= mx * my;
                case (ms)
                    3'd2: begin                          // mp = aL*bL (>= 0)
                        mlo0 <= mp[15:0];
                        macc <= {{18{1'b0}}, mp[33:16]};
                    end
                    3'd3: macc <= msum;                  // + aL*bH
                    3'd4: begin                          // + aH*bL, then shift out 16 bits
                        mlo1 <= msum[15:0];
                        macc <= {{16{msum[35]}}, msum[35:16]};
                    end
                    3'd5: begin                          // + aH*bH: the 64-bit product
                        mP  <= {msum[31:0], mlo1, mlo0};
                        mrb <= 1'b0;
                        mst <= 1'b0;
                    end
                    3'd6: if (mcnt != 5'd0) begin       // RDivPOT, one bit per cycle
                        mP  <= mP >>> 1;
                        mrb <= mP[0];
                        mst <= mst | mrb;
                    end
                    default: ;
                endcase
            end
            // result: clamp to 34 bits, then + round (half away from zero: the bit below the
            // cut is set and, for a negative product, some lower bit too); fits 35 bits
            logic signed [33:0] mcl;
            logic               mrnd;
            assign mcl  = (mP[63:33] == {31{mP[63]}}) ? mP[33:0]
                                                     : (mP[63] ? {1'b1, 33'd0} : {1'b0, {33{1'b1}}});
            assign mrnd = mrb && (!mP[63] || mst);
            assign mres = {mcl[33], mcl} + {{(W-1){1'b0}}, mrnd};

            // S1 (combinational from the inputs, registered into w1): exact in 33 bits
            assign a32 = in_a[i*32 +: 32];
            assign b32 = in_b[i*32 +: 32];
            // one 33-bit adder: a + b for ADD, a - b for everything else; MAX / MIN select on
            // the sign of a - b (exact in 33 bits)
            logic signed [32:0] as33, bx33;
            logic               lt;
            assign bx33 = (in_op == OP_ADD) ? {b32[31], b32} : ~{b32[31], b32};
            assign as33 = {a32[31], a32} + bx33 + {32'd0, in_op != OP_ADD};
            assign lt   = as33[32];                         // a < b (when not ADD)
            assign w0 = (in_op == OP_ADD || in_op == OP_SUB) ? {{(W-33){as33[32]}}, as33} :
                        (in_op == OP_MAX) ? {{(W-32){(lt ? b32[31] : a32[31])}}, (lt ? b32 : a32)} :
                        (in_op == OP_MIN) ? {{(W-32){(lt ? a32[31] : b32[31])}}, (lt ? a32 : b32)} :
                                            {{(W-32){a32[31]}}, a32};

            // S2 (w1 holds a sign-extended int32 for SRA, SRAR, SHL and CLAMP)
            assign w1l = w1[31:0];
            assign cl1 = (w1l < lo1) ? lo1 : w1l;
            assign cl2 = (cl1 > hi1) ? hi1 : cl1;
            // right: 32-bit arithmetic shift + RDivPOT rounding increment (SRAR)
            logic signed [31:0]  sra;
            logic signed [W-1:0] sr, sl, slc;
            logic        [30:0]  rmask, rhalf;
            logic        [31:0]  lmask;
            logic                rnd;
            assign sra   = w1l >>> sh1;
            assign rmask = (31'd1 << sh1) - 31'd1;          // shamt low bits
            assign rhalf = rmask ^ (rmask >> 1);            // bit shamt-1 (0 for shamt 0)
            assign rnd   = (op1 == OP_SRAR) && (|(w1l[30:0] & rhalf)) &&
                           (!w1l[31] || (|(w1l[30:0] & (rmask >> 1))));
            assign sr    = {{(W-32){sra[31]}}, sra} + {{(W-1){1'b0}}, rnd};
            // left: a * 2^shamt fits 35 bits iff a's bits at positions >= 34 - shamt all equal
            // its sign bit; otherwise clamp to the 35-bit extreme of its sign
            assign sl    = {{(W-32){w1l[31]}}, w1l} <<< sh1;
            assign lmask = 32'hFFFF_FFFF << (6'd34 - {1'b0, sh1});
            assign slc   = (((w1l ^ {32{w1l[31]}}) & lmask) == 32'd0) ? sl
                         : (w1l[31] ? {1'b1, {(W-1){1'b0}}} : {1'b0, {(W-1){1'b1}}});
            assign w2c = (op1 == OP_SRA || op1 == OP_SRAR) ? sr :
                         (op1 == OP_SHL)   ? slc :
                         (op1 == OP_CLAMP) ? {{(W-32){cl2[31]}}, cl2} : w1;

            // S3: saturation to fmt. w2 fits the format iff its bits above the format's
            // sign bit all equal the sign bit; otherwise it saturates toward its sign.
            logic fits;
            assign fits = (fmt2 == 2'd0) ? (w2[W-1:7]  == {(W-7){w2[W-1]}}) :
                          (fmt2 == 2'd1) ? (w2[W-1:15] == {(W-15){w2[W-1]}}) :
                                           (w2[W-1:31] == {(W-31){w2[W-1]}});

            always_ff @(posedge clk) begin
                if (adv) begin
                    w1 <= push ? mres : w0;
                    w2 <= w2c;
                    if (fits) begin
                        o3 <= w2[31:0];
                        s3 <= 1'b0;
                    end else begin
                        s3 <= 1'b1;
                        case (fmt2)
                            2'd0:    o3 <= w2[W-1] ? -32'sd128      : 32'sd127;
                            2'd1:    o3 <= w2[W-1] ? -32'sd32768    : 32'sd32767;
                            default: o3 <= w2[W-1] ? 32'sh8000_0000 : 32'sh7FFF_FFFF;
                        endcase
                    end
                end
            end
            assign out_data[i*32 +: 32] = o3;
            assign out_sat[i] = s3 & keep_p[D-1][i];
        end
    endgenerate

    assign out_keep = keep_p[D-1];
    assign out_last = last_p[D-1];

    sat_counter #(.N(L), .W(32)) u_sat (
        .clk(clk), .rst_n(rst_n), .clr(sat_clr), .en(out_valid && out_ready),
        .inc(out_sat), .count(sat_count), .sticky(sat_sticky)
    );
endmodule
