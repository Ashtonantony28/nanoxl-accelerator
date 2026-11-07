// sfu_recip.sv -- C5 RECIP primitive, one lane: 1/S ~= y * 2^-(15+b).
//
//   b  = bitlen(S)  (>= 16 for S >= 2^15)
//   m  = S >> (b - 16)                        in [2^15, 2^16)
//   y0 = lut.recip[(m >> 7) - 256]            (= m[14:7])
//   y  = (y0 * (2^32 - m*y0)) >> 31           (m*y0 < 2^32, so 2^32 - m*y0 is in (0, 2^32])
//
// Contract domain: S in [2^15, 2^24] (softmax row sums). The unit is bit-exact against
// nanoxl.quant.ops.recip_i for every S in [2^15, 2^SW) and for any 16-bit table contents.
// S < 2^15 is outside the contract (normalised by a left shift instead; result unspecified).
// Pipeline: 4 enabled edges: E1 bitlen + normalise | E2 LUT read | E3 2^32 - m*y0 |
// E4 y0 * (...) product register; y and b are wires off the E4 registers.
module sfu_recip #(
    parameter int SW = 32
) (
    input  logic          clk,
    input  logic          en,
    input  logic [SW-1:0] s,
    output logic [16:0]   y,
    output logic [5:0]    b,
    input  logic          lut_we,
    input  logic [7:0]    lut_waddr,
    input  logic [15:0]   lut_wdata
);
    function automatic logic [5:0] bitlen(input logic [SW-1:0] x);
        bitlen = '0;
        for (int i = 0; i < SW; i++) if (x[i]) bitlen = 6'(i + 1);
    endfunction

    logic [5:0]    bl;
    logic [SW-1:0] sn;
    always_comb begin
        bl = bitlen(s);
        sn = s << (6'(SW) - bl);          // msb of s moved to bit SW-1 (truncating)
    end

    logic [15:0] m1, m2;
    logic [5:0]  b1, b2, b3, b4;
    logic [15:0] y0, y03;
    logic [32:0] t3;
    logic [48:0] p4;

    lut_rom #(.DW(16), .AW(8)) u_lut (
        .clk(clk), .we(lut_we), .waddr(lut_waddr), .wdata(lut_wdata),
        .en(en), .raddr(m1[14:7]), .rdata(y0)
    );

    always_ff @(posedge clk) begin
        if (en) begin
            m1  <= sn[SW-1 -: 16];
            b1  <= bl;
            m2  <= m1;
            b2  <= b1;
            t3  <= 33'h1_0000_0000 - 33'({16'd0, m2} * {16'd0, y0});
            y03 <= y0;
            b3  <= b2;
            p4  <= 49'(y03) * 49'(t3);
            b4  <= b3;
        end
    end
    assign y = p4[47:31];
    assign b = b4;
endmodule
