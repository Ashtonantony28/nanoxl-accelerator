// srdhm.sv -- gemmlowp SaturatingRoundingDoublingHighMul, contract C2a, pipelined.
//
//   y = (a == INT32_MIN && b == INT32_MIN) ? INT32_MAX
//                                          : trunc_div(a*b + nudge, 2^31)
//   nudge = 2^30 if a*b >= 0 else 1 - 2^30;  trunc_div = C-style (toward zero)
//
// Implementation identity (SG11c): for p = a*b < 0, p + nudge = p + 1 - 2^30 is negative, and
// C truncation of a negative x by 2^31 is floor((x + 2^31 - 1) / 2^31) = floor((p + 2^30) / 2^31);
// for p >= 0 it is floor((p + 2^30) / 2^31) directly. So in both cases
//   trunc_div(p + nudge, 2^31) = (p + 2^30) >>> 31 = ((p >>> 30) + 1) >>> 1,
// one 34-bit increment instead of two 64-bit adds and a sign-dependent bias.
//
// Latency: 3 enabled clock edges. Operands present on a/b at enabled edge k
// appear on y after enabled edge k+2 (i.e. y is the third register).
//   stage 1: partial products (DSP multiply, M register)
//   stage 2: product register (DSP P register / partial-product sum)
//   stage 3: +2^30, >>> 31, overflow select
// No reset on the datapath; the caller tracks validity.
//
// NDSP = 4 (default): one 32x32 product, which synthesis splits into four 27x18 / 25x18 DSP tiles.
// NDSP = 3 (C6 option): a = aH*2^26 + aL (aL 26 bits unsigned, aH 6 bits signed), b = bH*2^17 + bL
//   (bL 17 bits unsigned, bH 15 bits signed): aL*bL, aL*bH and aH*bL each fit one DSP48E2 /
//   DSP48E1 (27x18 / 25x18 signed); the 6x15 corner aH*bH is built from shifted partial products
//   in fabric (~50 LUTs). One DSP less per instance for ~100 LUTs (the sum of the partial
//   products also moves to fabric): worth it only when DSPs, not LUTs, are the limit.
module srdhm #(
    parameter int NDSP = 4
) (
    input  logic               clk,
    input  logic               en,
    input  logic signed [31:0] a,
    input  logic signed [31:0] b,
    output logic signed [31:0] y
);
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [63:0] p2;            // bits [29:0] do not reach y
    logic        [33:0] q;             // (p >>> 30) + 1 (mod 2^34); y = q[32:1]
    /* verilator lint_on UNUSEDSIGNAL */
    logic               ovf1, ovf2;

    // stage 3: ((p >>> 30) + 1) >>> 1, low 32 bits
    assign q = p2[63:30] + 34'd1;

    generate
        if (NDSP == 3) begin : g_3dsp
            logic        [25:0] aL;
            logic signed [5:0]  aH;
            logic        [16:0] bL;
            logic signed [14:0] bH;
            logic signed [43:0] t_ll;      // aL*bL  (43 bits unsigned)
            logic signed [41:0] t_lh;      // aL*bH
            logic signed [23:0] t_hl;      // aH*bL
            logic signed [20:0] t_hh;      // aH*bH (fabric)
            (* use_dsp = "no" *) logic signed [20:0] hh;
            assign aL = a[25:0];
            assign aH = a[31:26];
            assign bL = b[16:0];
            assign bH = b[31:17];
            always_comb begin
                // aH*bH = sum_k aH[k] * bH * 2^k, with weight -2^5 for the sign bit aH[5]
                hh = 21'sd0;
                for (int k = 0; k < 5; k++)
                    if (aH[k]) hh = hh + (21'(bH) <<< k);
                if (aH[5]) hh = hh - (21'(bH) <<< 5);
            end
            always_ff @(posedge clk) begin
                if (en) begin
                    t_ll <= $signed({1'b0, aL}) * $signed({1'b0, bL});
                    t_lh <= $signed({1'b0, aL}) * bH;
                    t_hl <= aH * $signed({1'b0, bL});
                    t_hh <= hh;
                    p2   <= 64'(t_ll) + (64'(t_lh) <<< 17) + (64'(t_hl) <<< 26) + (64'(t_hh) <<< 43);
                end
            end
        end else begin : g_4dsp
            logic signed [63:0] p1;
            always_ff @(posedge clk) begin
                if (en) begin
                    p1 <= 64'(a) * 64'(b);
                    p2 <= p1;
                end
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (en) begin
            ovf1 <= (a == 32'sh8000_0000) && (b == 32'sh8000_0000);
            ovf2 <= ovf1;
            y    <= ovf2 ? 32'sh7FFF_FFFF : $signed(q[32:1]);
        end
    end
endmodule
