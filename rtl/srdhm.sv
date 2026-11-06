// srdhm.sv -- gemmlowp SaturatingRoundingDoublingHighMul, contract C2a, pipelined.
//
//   y = (a == INT32_MIN && b == INT32_MIN) ? INT32_MAX
//                                          : trunc_div(a*b + nudge, 2^31)
//   nudge = 2^30 if a*b >= 0 else 1 - 2^30;  trunc_div = C-style (toward zero)
//
// Latency: 3 enabled clock edges. Operands present on a/b at enabled edge k
// appear on y after enabled edge k+2 (i.e. y is the third register).
//   stage 1: 64-bit signed product (DSP multiply)
//   stage 2: product pipeline register (DSP P register; lets the 32x32 multiply
//            cascade across DSP48E2 slices at 250 MHz)
//   stage 3: nudge add, truncating /2^31, overflow select
// No reset on the datapath; the caller tracks validity.
module srdhm (
    input  logic               clk,
    input  logic               en,
    input  logic signed [31:0] a,
    input  logic signed [31:0] b,
    output logic signed [31:0] y
);
    logic signed [63:0] p1, p2;
    logic               ovf1, ovf2;
    logic signed [63:0] a64, b64;

    // Combinational stage-3 logic.
    logic signed [63:0] s, sadj, q;
    assign a64 = {{32{a[31]}}, a};
    assign b64 = {{32{b[31]}}, b};
    always_comb begin
        s    = p2 + ((p2 >= 64'sd0) ? 64'sh0000_0000_4000_0000 : -64'sh0000_0000_3FFF_FFFF);
        // C truncating division by 2^31: bias negative values by (2^31 - 1) before >>>.
        sadj = s + ((s < 64'sd0) ? 64'sh0000_0000_7FFF_FFFF : 64'sd0);
        q    = sadj >>> 31;
    end

    always_ff @(posedge clk) begin
        if (en) begin
            p1   <= a64 * b64;
            ovf1 <= (a == 32'sh8000_0000) && (b == 32'sh8000_0000);
            p2   <= p1;
            ovf2 <= ovf1;
            y    <= ovf2 ? 32'sh7FFF_FFFF : q[31:0];
        end
    end
endmodule
