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
// Tables are loaded at run time, one 16-byte PBUF line per write (docs/isa.md §3): lut_we
// one-hot [0] exp2, [1] recip, [2] rsqrt (line 0..31 = 8 uint16 entries, little-endian), [3]
// GELU (lut_waddr = {slot, line 0..15}, 16 int8 entries). Write them while idle.
//
// Lane sharing (docs/fpga.md §2.3 C3): EXP2 / RECIP / RSQRT have EL primitive lanes (EL
// divides L). A beat of one of those ops is split into R = L/EL sub-beats of EL lanes that
// enter the pipeline on consecutive cycles (sub-beat 0 straight from in_data, the others
// from a shift register), and the output register assembles them by shifting EL lanes in
// from the top; out_valid rises with the last sub-beat. Those ops therefore take R cycles
// per beat (in_ready is low for the R-1 cycles after the accept) and R-1 cycles of extra
// latency. GELU and the pass-through ops keep L lanes and one beat per cycle. EL = L is the
// unshared pipeline. The compiler emits no elementwise EXP2 / RECIP / RSQRT (softmax and LN
// have their own primitives), so only those stand-alone sub-ops are slower.
//
// Pipeline: 6 stages, whole-pipeline stall (in_ready = (!out_valid || out_ready) and no
// sub-beat pending). Each primitive is fed from the tap of the input delay line that makes
// all of them finish at stage 5 (rsqrt 5 edges, recip 4, exp2 2, GELU 1); stage 6 selects
// and packs. A GELU / pass-through beat accepted at edge k is on out_* after edge k+5, an
// EXP2 / RECIP / RSQRT beat after edge k+5+R-1.
module sfu_elem #(
    parameter int L  = 4,
    parameter int NG = 4,         // GELU table slots
    parameter int EL = L          // EXP2 / RECIP / RSQRT primitive lanes (divides L)
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
    input  logic [11:0]     lut_waddr,
    input  logic [127:0]    lut_wdata
);
    localparam int D   = 6;
    localparam int GSW = (NG > 1) ? $clog2(NG) : 1;
    localparam int GAW = GSW + 8;
    localparam int R   = L / EL;                 // sub-beats per shared-op beat
    localparam int RW  = (R > 1) ? $clog2(R) : 1;
    localparam int HL  = (R > 1) ? L - EL : EL;  // lanes held for sub-beats 1..R-1

    // ---------------- sub-beat issue ----------------
    logic [D-1:0]  v;               // token valid (index i = after i+1 edges)
    logic [D-1:0]  f;               // token is the last (or only) sub-beat of its beat
    logic          adv, in_fire, slow_in;
    logic [RW-1:0] pend;            // sub-beats still to issue from the hold register
    assign out_valid = v[D-1] && f[D-1];
    assign adv       = !out_valid || out_ready;
    assign in_ready  = adv && (pend == '0);
    assign in_fire   = in_valid && in_ready;
    assign slow_in   = (R > 1) && (in_op <= 3'd2);

    logic [HL*32-1:0] h_data;
    logic [HL-1:0]    h_ksub;
    logic [2:0]       h_op;
    logic [GSW-1:0]   h_slot;
    logic [L-1:0]     h_keep;
    logic             h_last;

    // stage-0 token
    logic           t_valid, t_fin, t_last;
    logic [2:0]     t_op;
    logic [GSW-1:0] t_slot;
    logic [L-1:0]   t_keep;
    logic [EL-1:0]  t_ksub;
    logic [31:0]    t_x [EL];       // primitive-lane inputs
    always_comb begin
        if (pend != '0) begin
            t_valid = 1'b1;
            t_fin   = (pend == RW'(1));
            t_op    = h_op;
            t_slot  = h_slot;
            t_keep  = h_keep;
            t_last  = h_last;
            t_ksub  = h_ksub[EL-1:0];
            for (int i = 0; i < EL; i++) t_x[i] = h_data[i*32 +: 32];
        end else begin
            t_valid = in_valid;
            t_fin   = !slow_in;
            t_op    = in_op;
            t_slot  = in_slot[GSW-1:0];
            t_keep  = in_keep;
            t_last  = in_last;
            t_ksub  = in_keep[EL-1:0];
            for (int i = 0; i < EL; i++) t_x[i] = in_data[i*32 +: 32];
        end
    end
    generate
        if (R > 1) begin : g_hold
            always_ff @(posedge clk) begin
                if (!rst_n) pend <= '0;
                else if (adv) begin
                    if (pend != '0)              pend <= pend - 1'b1;
                    else if (in_fire && slow_in) pend <= RW'(R - 1);
                end
            end
            always_ff @(posedge clk) begin
                if (adv) begin
                    if (pend != '0) begin
                        h_data <= h_data >> (EL*32);
                        h_ksub <= h_ksub >> EL;
                    end else begin
                        h_data <= in_data[L*32-1:EL*32];
                        h_ksub <= in_keep[L-1:EL];
                        h_op   <= in_op;
                        h_slot <= in_slot[GSW-1:0];
                        h_keep <= in_keep;
                        h_last <= in_last;
                    end
                end
            end
        end else begin : g_nohold
            assign pend   = '0;
            assign h_data = '0;
            assign h_ksub = '0;
            assign h_op   = '0;
            assign h_slot = '0;
            assign h_keep = '0;
            assign h_last = 1'b0;
        end
    endgenerate

    // sideband delay line (index i = after i edges)
    logic [2:0]     op_p   [D];
    logic [GSW-1:0] slot_p [D];
    logic [L-1:0]   keep_p [D];
    logic [EL-1:0]  ksub_p [D];
    logic [D-1:0]   last_p;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v <= '0;
            f <= '0;
        end else if (adv) begin
            v <= {v[D-2:0], t_valid};
            f <= {f[D-2:0], t_fin};
        end
    end
    always_ff @(posedge clk) begin
        if (adv) begin
            op_p[1]   <= t_op;
            slot_p[1] <= t_slot;
            keep_p[1] <= t_keep;
            ksub_p[1] <= t_ksub;
            last_p[1] <= t_last;
            for (int i = 2; i < D; i++) begin
                op_p[i]   <= op_p[i-1];
                slot_p[i] <= slot_p[i-1];
                keep_p[i] <= keep_p[i-1];
                ksub_p[i] <= ksub_p[i-1];
                last_p[i] <= last_p[i-1];
            end
        end
    end

    // ---------------- shared primitives (EL lanes) ----------------
    logic [15:0]        e_l  [EL];
    logic [16:0]        ry_l [EL];
    logic [5:0]         rb_l [EL];
    logic signed [21:0] qy_l [EL];
    logic signed [7:0]  qk_l [EL];
    logic [31:0]        rs_l [EL];      // packed stage-5 result, masked

    genvar j;
    generate
        for (j = 0; j < EL; j++) begin : g_prim
            logic [31:0] x1, x2, x3;
            always_ff @(posedge clk) begin
                if (adv) begin
                    x1 <= t_x[j]; x2 <= x1; x3 <= x2;
                end
            end
            sfu_rsqrt #(.QW(32)) u_rsqrt (
                .clk(clk), .en(adv), .q(t_x[j]), .y(qy_l[j]), .k(qk_l[j]),
                .lut_we(lut_we[2]), .lut_waddr(lut_waddr[4:0]), .lut_wdata(lut_wdata)
            );
            sfu_recip #(.SW(32)) u_recip (
                .clk(clk), .en(adv), .s(x1), .y(ry_l[j]), .b(rb_l[j]),
                .lut_we(lut_we[1]), .lut_waddr(lut_waddr[4:0]), .lut_wdata(lut_wdata)
            );
            sfu_exp2 u_exp2 (
                .clk(clk), .en(adv), .d(x3[15:0]), .e(e_l[j]),
                .lut_we(lut_we[0]), .lut_waddr(lut_waddr[4:0]), .lut_wdata(lut_wdata)
            );
            wire [2:0] op5 = op_p[D-1];
            assign rs_l[j] = !ksub_p[D-1][j] ? 32'd0 :
                             (op5 == 3'd0) ? {16'd0, e_l[j]} :
                             (op5 == 3'd1) ? {2'b00, rb_l[j], 7'd0, ry_l[j]} :
                                             {qk_l[j], {2{qy_l[j][21]}}, qy_l[j]};
        end
    endgenerate

    // ---------------- full-width lanes: GELU and pass-through ----------------
    logic [7:0]  g_l  [L];
    logic [31:0] x5_l [L];
    logic [31:0] r_l  [L];
    generate
        for (j = 0; j < L; j++) begin : g_lane
            logic [31:0] x1, x2, x3, x4;
            always_ff @(posedge clk) begin
                if (adv) begin
                    x1 <= in_data[j*32 +: 32]; x2 <= x1; x3 <= x2; x4 <= x3; x5_l[j] <= x4;
                end
            end
            lut_rom #(.DW(8), .AW(GAW), .WN(16)) u_gelu (
                .clk(clk), .we(lut_we[3]), .waddr(lut_waddr[GAW-5:0]), .wdata(lut_wdata),
                .en(adv), .raddr({slot_p[4], x4[7:0]}), .rdata(g_l[j])
            );
            wire [2:0] op5 = op_p[D-1];
            assign r_l[j] = !keep_p[D-1][j] ? 32'd0 :
                            (op5 == 3'd3)   ? {{24{g_l[j][7]}}, g_l[j]} : x5_l[j];
        end
    endgenerate

    // ---------------- stage 6: select and pack ----------------
    // A shared-op sub-beat shifts its EL lanes in from the top, so after R sub-beats
    // sub-beat 0 sits in lanes 0..EL-1.
    logic [EL*32-1:0] rs_flat;
    logic [L*32-1:0]  rs_next;
    always_comb begin
        for (int i = 0; i < EL; i++) rs_flat[i*32 +: 32] = rs_l[i];
    end
    generate
        if (R > 1) begin : g_shift
            assign rs_next = {rs_flat, out_data[L*32-1:EL*32]};
        end else begin : g_noshift
            assign rs_next = rs_flat;
        end
    endgenerate
    always_ff @(posedge clk) begin
        if (adv) begin
            out_keep <= keep_p[D-1];
            out_last <= last_p[D-1];
            if (op_p[D-1] <= 3'd2) out_data <= rs_next;
            else for (int i = 0; i < L; i++) out_data[i*32 +: 32] <= r_l[i];
        end
    end
endmodule
