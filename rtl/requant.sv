// requant.sv -- L-lane pipelined requantizer, contract C2 (docs/contracts.md), bit-exact
// with nanoxl.numerics.requant.
//
//   x   = wrap32(acc + bias)
//   y   = RoundingDivideByPOT(SRDHM(x, mult), shift)
//   out = clamp(y, int8 or int16)          (in_out16 selects int16, per beat)
//
// Per-lane (per output channel) acc/bias/mult/shift; one result per lane per cycle.
// Pipeline depth D = 6 (a beat accepted at clock edge k is on out_* after edge k+5,
// i.e. 6 register stages):
//   S1 bias add (wrap32) | S2-S4 srdhm (product, product reg, nudge+trunc) |
//   S5 RoundingDivideByPOT | S6 clamp + saturation flag
// Handshake: whole-pipeline stall. in_ready = !out_valid || out_ready (combinational
// from out_ready). Full throughput while out_ready is high.
// out_data lane i = int16 result (int8 results sign-extended to 16 bits).
// out_sat[i] = lane i saturated AND out_keep[i]. sat_count counts saturating kept lanes
// on output transfers (sticky, saturating counter; sat_clr clears).
module requant #(
    parameter int L    = 4,
    parameter int NDSP = 4     // srdhm DSP tiles per lane: 4, or 3 (fabric corner, rtl/srdhm.sv)
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [L*32-1:0] in_acc,
    input  logic [L*32-1:0] in_bias,
    input  logic [L*32-1:0] in_mult,
    input  logic [L*5-1:0]  in_shift,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    input  logic            in_out16,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*16-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic [L-1:0]    out_sat,
    output logic            out_last,
    input  logic            sat_clr,
    output logic [31:0]     sat_count,
    output logic            sat_sticky
);
    localparam int D = 6;

    logic [D-1:0] v;                 // v[k]: stage k+1 holds a valid beat
    logic         adv;
    assign adv       = !v[D-1] || out_ready;
    assign in_ready  = adv;
    assign out_valid = v[D-1];

    // Sideband pipeline (keep, last, out16), one entry per stage.
    logic [L-1:0] keep_p [D];
    logic [D-1:0] last_p;
    logic [D-1:0] o16_p;

    always_ff @(posedge clk) begin
        if (!rst_n) v <= '0;
        else if (adv) v <= {v[D-2:0], in_valid};
    end
    always_ff @(posedge clk) begin
        if (adv) begin
            keep_p[0] <= in_keep;
            for (int k = 1; k < D; k++) keep_p[k] <= keep_p[k-1];
            last_p <= {last_p[D-2:0], in_last};
            o16_p  <= {o16_p[D-2:0], in_out16};
        end
    end

    // C2b RoundingDivideByPOT on int32, exponent 0..31 (round half away from zero).
    function automatic logic signed [31:0] rdivpot32(input logic signed [31:0] x,
                                                     input logic [4:0] e);
        logic [31:0] mask, rem, thr;
        mask = (32'd1 << e) - 32'd1;
        rem  = x & mask;
        thr  = (mask >> 1) + {31'd0, x[31]};
        rdivpot32 = (x >>> e) + ((rem > thr) ? 32'sd1 : 32'sd0);
    endfunction

    genvar i;
    generate
        for (i = 0; i < L; i++) begin : g_lane
            logic signed [31:0] x1, m1, q4, y5;
            logic        [4:0]  sh1, sh2, sh3, sh4;
            logic signed [15:0] o6;
            logic               s6;

            always_ff @(posedge clk) begin
                if (adv) begin
                    // S1: wrapping 32-bit add (the carry out is dropped).
                    x1  <= in_acc[i*32 +: 32] + in_bias[i*32 +: 32];
                    m1  <= in_mult[i*32 +: 32];
                    sh1 <= in_shift[i*5 +: 5];
                    sh2 <= sh1;
                    sh3 <= sh2;
                    sh4 <= sh3;
                    // S5
                    y5  <= rdivpot32(q4, sh4);
                end
            end

            // S2..S4
            srdhm #(.NDSP(NDSP)) u_srdhm (.clk(clk), .en(adv), .a(x1), .b(m1), .y(q4));

            // S6: clamp to int8 / int16 (mode bit of the beat now in S5). y5 fits iff its bits
            // above the format's sign bit equal the sign bit; else it saturates toward its sign.
            logic fits;
            assign fits = o16_p[D-2] ? (y5[31:15] == {17{y5[31]}}) : (y5[31:7] == {25{y5[31]}});
            always_ff @(posedge clk) begin
                if (adv) begin
                    if (fits) begin
                        o6 <= y5[15:0];
                        s6 <= 1'b0;
                    end else begin
                        o6 <= o16_p[D-2] ? (y5[31] ? 16'sh8000 : 16'sh7FFF)
                                         : (y5[31] ? -16'sd128 : 16'sd127);
                        s6 <= 1'b1;
                    end
                end
            end
            assign out_data[i*16 +: 16] = o6;
            assign out_sat[i] = s6 & keep_p[D-1][i];
        end
    endgenerate

    assign out_keep = keep_p[D-1];
    assign out_last = last_p[D-1];

    sat_counter #(.N(L), .W(32)) u_sat (
        .clk(clk), .rst_n(rst_n), .clr(sat_clr), .en(out_valid && out_ready),
        .inc(out_sat), .count(sat_count), .sticky(sat_sticky)
    );
endmodule
