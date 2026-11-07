// sfu_exp2.sv -- C5 EXP2 primitive, one lane: e = 2^(-d/256) in Q.15.
//
//   q = d >> 8;  r = d & 255;  e = (q >= 16) ? 0 : lut.exp2[r] >> q
//
// d is uint16 (row max - score in softmax), e <= lut.exp2[0] (32768 for the bundle table).
// The table is loaded through the write port (lut_we/lut_waddr/lut_wdata) from the bundle's
// `lut.exp2`; any 16-bit contents are handled exactly.
// Pipeline: 2 enabled edges (E1 LUT read + q register, E2 shift; e is the E2 register).
// No reset on the datapath; the caller tracks validity and stalls the pipeline with en.
module sfu_exp2 (
    input  logic        clk,
    input  logic        en,
    input  logic [15:0] d,
    output logic [15:0] e,
    input  logic        lut_we,
    input  logic [7:0]  lut_waddr,
    input  logic [15:0] lut_wdata
);
    logic [15:0] lv;
    logic [7:0]  q1;
    lut_rom #(.DW(16), .AW(8)) u_lut (
        .clk(clk), .we(lut_we), .waddr(lut_waddr), .wdata(lut_wdata),
        .en(en), .raddr(d[7:0]), .rdata(lv)
    );
    always_ff @(posedge clk) begin
        if (en) begin
            q1 <= d[15:8];
            e  <= (q1 >= 8'd16) ? 16'd0 : (lv >> q1[3:0]);
        end
    end
endmodule
