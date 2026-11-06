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
// Pipeline depth 3: S1 wide op (adder / 32x32 multiplier / compare) |
//                   S2 shift, rounding, clamp | S3 output saturation + sat flag.
// Handshake: whole-pipeline stall, in_ready = !out_valid || out_ready.
// out_sat[i] = lane i saturated AND out_keep[i]; sat_count counts them on transfers.
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
    localparam logic [3:0] OP_PASS = 4'd0, OP_ADD = 4'd1, OP_SUB = 4'd2, OP_MUL = 4'd3,
                           OP_MAX = 4'd4, OP_MIN = 4'd5, OP_SRA = 4'd6, OP_SRAR = 4'd7,
                           OP_SHL = 4'd8, OP_CLAMP = 4'd9;
    localparam int D = 3;

    logic [D-1:0] v;
    logic         adv;
    assign adv       = !v[D-1] || out_ready;
    assign in_ready  = adv;
    assign out_valid = v[D-1];

    // Sideband: stage 1 and 2 controls, keep/last for all stages.
    logic [3:0]  op1;
    logic [1:0]  fmt1, fmt2;
    logic [4:0]  sh1;
    logic signed [31:0] lo1, hi1;
    logic [L-1:0] keep_p [D];
    logic [D-1:0] last_p;

    always_ff @(posedge clk) begin
        if (!rst_n) v <= '0;
        else if (adv) v <= {v[D-2:0], in_valid};
    end
    always_ff @(posedge clk) begin
        if (adv) begin
            op1  <= in_op;
            fmt1 <= in_fmt;
            sh1  <= in_shamt;
            lo1  <= in_lo;
            hi1  <= in_hi;
            fmt2 <= fmt1;
            keep_p[0] <= in_keep;
            for (int k = 1; k < D; k++) keep_p[k] <= keep_p[k-1];
            last_p <= {last_p[D-2:0], in_last};
        end
    end

    function automatic logic signed [63:0] rdivpot64(input logic signed [63:0] x,
                                                     input logic [4:0] e);
        logic [63:0] mask, rem, thr;
        mask = (64'd1 << e) - 64'd1;
        rem  = x & mask;
        thr  = (mask >> 1) + ((x < 64'sd0) ? 64'd1 : 64'd0);
        rdivpot64 = (x >>> e) + ((rem > thr) ? 64'sd1 : 64'sd0);
    endfunction

    genvar i;
    generate
        for (i = 0; i < L; i++) begin : g_lane
            logic signed [63:0] a64, b64, w0, w1, w2c, w2;
            logic signed [63:0] lo64, hi64, smin, smax;
            logic signed [31:0] o3;
            logic               s3;

            // S1 (combinational from the inputs, registered into w1)
            assign a64 = {{32{in_a[i*32+31]}}, in_a[i*32 +: 32]};
            assign b64 = {{32{in_b[i*32+31]}}, in_b[i*32 +: 32]};
            always_comb begin
                case (in_op)
                    OP_ADD:  w0 = a64 + b64;
                    OP_SUB:  w0 = a64 - b64;
                    OP_MUL:  w0 = a64 * b64;
                    OP_MAX:  w0 = (a64 > b64) ? a64 : b64;
                    OP_MIN:  w0 = (a64 < b64) ? a64 : b64;
                    default: w0 = a64;
                endcase
            end

            // S2
            assign lo64 = {{32{lo1[31]}}, lo1};
            assign hi64 = {{32{hi1[31]}}, hi1};
            always_comb begin
                case (op1)
                    OP_MUL, OP_SRAR: w2c = rdivpot64(w1, sh1);
                    OP_SRA:          w2c = w1 >>> sh1;
                    OP_SHL:          w2c = w1 <<< sh1;
                    OP_CLAMP: begin
                        w2c = (w1 < lo64) ? lo64 : w1;
                        w2c = (w2c > hi64) ? hi64 : w2c;
                    end
                    default:         w2c = w1;
                endcase
            end

            // S3 saturation bounds
            always_comb begin
                case (fmt2)
                    2'd0:    begin smin = -64'sd128;        smax = 64'sd127;        end
                    2'd1:    begin smin = -64'sd32768;      smax = 64'sd32767;      end
                    default: begin smin = -64'sd2147483648; smax = 64'sd2147483647; end
                endcase
            end

            always_ff @(posedge clk) begin
                if (adv) begin
                    w1 <= w0;
                    w2 <= w2c;
                    if (w2 > smax)      begin o3 <= smax[31:0]; s3 <= 1'b1; end
                    else if (w2 < smin) begin o3 <= smin[31:0]; s3 <= 1'b1; end
                    else                begin o3 <= w2[31:0];   s3 <= 1'b0; end
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
