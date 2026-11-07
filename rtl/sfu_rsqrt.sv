// sfu_rsqrt.sv -- C5 RSQRT primitive, one lane: 1/sqrt(Q) ~= y * 2^-(22+k).
//
//   e  = bitlen(Q) - 1;  k = (e - 14) >> 1     (floor; k in [-7, (QW-15)/2])
//   m  = k >= 0 ? Q >> 2k : Q << -2k            in [2^14, 2^16)
//   y0 = lut.rsqrt[m >> 8]
//   t  = (y0*y0) >> 14;  u = (m*t) >> 16;  y = (y0 * (3*2^14 - u)) >> 15   (floor)
//
// Contract domain: Q in [1, 2^46] (LayerNorm variance sum, QW = 48). Bit-exact against
// nanoxl.quant.ops.rsqrt_i for every Q in [1, 2^QW) and for any 16-bit table contents
// (y is signed: with arbitrary tables 3*2^14 - u can go negative). Q = 0 is outside the
// contract; the unit treats it as Q = 1.
// Pipeline: 5 enabled edges: E1 bitlen + normalise | E2 LUT read | E3 t | E4 u, 3*2^14-u |
// E5 product register; y and k are wires off the E5 registers.
module sfu_rsqrt #(
    parameter int QW = 48
) (
    input  logic                clk,
    input  logic                en,
    input  logic [QW-1:0]       q,
    output logic signed [21:0]  y,
    output logic signed [7:0]   k,
    input  logic                lut_we,
    input  logic [7:0]          lut_waddr,
    input  logic [15:0]         lut_wdata
);
    function automatic logic [6:0] bitlen(input logic [QW-1:0] x);
        bitlen = '0;
        for (int i = 0; i < QW; i++) if (x[i]) bitlen = 7'(i + 1);
    endfunction

    logic [QW-1:0]     qq;
    logic signed [8:0] e, sh2;
    logic [QW+15:0]    wide;
    always_comb begin
        qq   = (q == '0) ? QW'(1) : q;
        e    = $signed({2'b00, bitlen(qq)}) - 9'sd1;
        sh2  = (e - 9'sd14) & ~9'sd1;                  // 2k = floor((e-14)/2)*2
        if (sh2 >= 0) wide = {16'd0, qq} >> sh2;
        else          wide = {16'd0, qq} << (-sh2);
    end

    logic [15:0]        m1, m2, m3;
    logic signed [7:0]  k1, k2, k3, k4, k5;
    logic [15:0]        y0, y03, y04;
    logic [17:0]        t3;
    logic signed [19:0] v4;
    logic signed [36:0] p5;
    logic [33:0]        mt;
    logic [31:0]        yy;

    lut_rom #(.DW(16), .AW(8)) u_lut (
        .clk(clk), .we(lut_we), .waddr(lut_waddr), .wdata(lut_wdata),
        .en(en), .raddr(m1[15:8]), .rdata(y0)
    );

    always_comb begin
        mt = {18'd0, m3} * {16'd0, t3};
        yy = {16'd0, y0} * {16'd0, y0};
    end

    always_ff @(posedge clk) begin
        if (en) begin
            m1  <= wide[15:0];
            k1  <= 8'(sh2 >>> 1);
            m2  <= m1;   k2 <= k1;
            t3  <= yy[31:14];
            y03 <= y0;   m3 <= m2;   k3 <= k2;
            v4  <= 20'sd49152 - $signed({2'b00, mt[33:16]});
            y04 <= y03;  k4 <= k3;
            p5  <= $signed({21'd0, y04}) * 37'(v4);
            k5  <= k4;
        end
    end
    assign y = 22'(p5 >>> 15);
    assign k = k5;
endmodule
