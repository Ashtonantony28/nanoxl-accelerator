// sfu_elem.sv -- L-lane elementwise special-function pipeline (contract C5 primitives).
//
//   op  name   lane input (int32 lane)        lane output (int32 lane)
//   0   EXP2   d = lane[15:0] (uint16)        e (Q.15, zero-extended)
//   1   RECIP  S = lane (uint32, >= 2^15)     {2'b0, b[5:0], 7'b0, y[16:0]}   1/S ~= y*2^-(15+b)
//   2   RSQRT  Q = lane (uint32, >= 1)        {k[7:0], y[23:0]} (both signed) 1/sqrt(Q) ~= y*2^-(22+k)
//   3   GELU   x = lane[7:0]                  sign-extended gelu.lut[slot][x & 0xFF]
//   4-7        -                              lane passed through unchanged
// Lanes with keep = 0 return 0. keep/last travel with the beat. `slot` (GELU table, one
// per layer) is a per-beat input like op, so beats of different rows can share the pipe.
//
// Tables are loaded at run time (lut_we one-hot: [0] exp2, [1] recip, [2] rsqrt, [3] GELU;
// lut_waddr = entry for exp2/recip/rsqrt, {slot, entry} for GELU). Write them while idle.
//
// Pipeline: 6 stages, 1 beat per cycle, whole-pipeline stall (in_ready = !out_valid ||
// out_ready). Each primitive is fed from the tap of the input delay line that makes all
// of them finish at stage 5 (rsqrt 5 edges, recip 4, exp2 2, GELU 1); stage 6 selects and
// packs. A beat accepted at edge k is on out_* after edge k+5.
module sfu_elem #(
    parameter int L  = 4,
    parameter int NG = 4          // GELU table slots
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [2:0]      in_op,
    input  logic [7:0]      in_slot,
    input  logic [L*32-1:0] in_data,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*32-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic            out_last,
    input  logic [3:0]      lut_we,
    input  logic [15:0]     lut_waddr,
    input  logic [15:0]     lut_wdata
);
    localparam int D   = 6;
    localparam int GSW = (NG > 1) ? $clog2(NG) : 1;
    localparam int GAW = GSW + 8;

    logic [D-1:0] v;
    logic         adv;
    assign adv       = !v[D-1] || out_ready;
    assign in_ready  = adv;
    assign out_valid = v[D-1];

    // sideband delay line (index i = after i edges)
    logic [2:0]   op_p   [D];
    logic [GSW-1:0] slot_p [D];
    logic [L-1:0] keep_p [D];
    logic [D-1:0] last_p;
    always_ff @(posedge clk) begin
        if (!rst_n) v <= '0;
        else if (adv) v <= {v[D-2:0], in_valid};
    end
    always_ff @(posedge clk) begin
        if (adv) begin
            op_p[1]   <= in_op;
            slot_p[1] <= in_slot[GSW-1:0];
            keep_p[1] <= in_keep;
            last_p[1] <= in_last;
            for (int i = 2; i < D; i++) begin
                op_p[i]   <= op_p[i-1];
                slot_p[i] <= slot_p[i-1];
                keep_p[i] <= keep_p[i-1];
                last_p[i] <= last_p[i-1];
            end
        end
    end

    // output-stage results of every primitive, per lane
    logic [15:0]        e_l  [L];
    logic [16:0]        ry_l [L];
    logic [5:0]         rb_l [L];
    logic signed [21:0] qy_l [L];
    logic signed [7:0]  qk_l [L];
    logic [7:0]         g_l  [L];
    logic [31:0]        x5_l [L];

    genvar j;
    generate
        for (j = 0; j < L; j++) begin : g_lane
            logic [31:0] x0, x1, x2, x3, x4;
            assign x0 = in_data[j*32 +: 32];
            always_ff @(posedge clk) begin
                if (adv) begin
                    x1 <= x0; x2 <= x1; x3 <= x2; x4 <= x3; x5_l[j] <= x4;
                end
            end
            sfu_rsqrt #(.QW(32)) u_rsqrt (
                .clk(clk), .en(adv), .q(x0), .y(qy_l[j]), .k(qk_l[j]),
                .lut_we(lut_we[2]), .lut_waddr(lut_waddr[7:0]), .lut_wdata(lut_wdata)
            );
            sfu_recip #(.SW(32)) u_recip (
                .clk(clk), .en(adv), .s(x1), .y(ry_l[j]), .b(rb_l[j]),
                .lut_we(lut_we[1]), .lut_waddr(lut_waddr[7:0]), .lut_wdata(lut_wdata)
            );
            sfu_exp2 u_exp2 (
                .clk(clk), .en(adv), .d(x3[15:0]), .e(e_l[j]),
                .lut_we(lut_we[0]), .lut_waddr(lut_waddr[7:0]), .lut_wdata(lut_wdata)
            );
            lut_rom #(.DW(8), .AW(GAW)) u_gelu (
                .clk(clk), .we(lut_we[3]), .waddr(lut_waddr[GAW-1:0]), .wdata(lut_wdata[7:0]),
                .en(adv), .raddr({slot_p[4], x4[7:0]}), .rdata(g_l[j])
            );
        end
    endgenerate

    // stage 6: select and pack
    logic [31:0] r_l [L];
    generate
        for (j = 0; j < L; j++) begin : g_pack
            wire [2:0] op5 = op_p[D-1];
            wire       k5  = keep_p[D-1][j];
            assign r_l[j] = !k5           ? 32'd0 :
                            (op5 == 3'd0) ? {16'd0, e_l[j]} :
                            (op5 == 3'd1) ? {2'b00, rb_l[j], 7'd0, ry_l[j]} :
                            (op5 == 3'd2) ? {qk_l[j], {2{qy_l[j][21]}}, qy_l[j]} :
                            (op5 == 3'd3) ? {{24{g_l[j][7]}}, g_l[j]} : x5_l[j];
        end
    endgenerate
    always_ff @(posedge clk) begin
        if (adv) begin
            out_keep <= keep_p[D-1];
            out_last <= last_p[D-1];
            for (int i = 0; i < L; i++) out_data[i*32 +: 32] <= r_l[i];
        end
    end
endmodule
