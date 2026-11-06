// vec_unit.sv -- vector unit sequencer: runs one command over a row of `len` elements,
// streamed as ceil(len/L) beats of L int32 lanes, from source streams to a destination
// stream. Semantics and opcode map: docs/vector-unit.md; reference nanoxl/vec_ref.py.
//
// Opcode space (cmd_op, 5 bits):
//   0x00-0x0F  vec_alu op (low 4 bits; 0x0A-0x0F reserved = PASS)
//   0x10       REQUANT   a = acc, b = bias, p = per-lane mult/shift (contract C2)
//   0x11       REDUCE    a only; result on the r_* port (sum, sumsq, max, min)
//   0x12-0x17  reserved: executed as ALU PASS, raises sticky illegal_op
//   0x18-0x1F  SFU hook: forwarded on the sfu_in_* port with sfu_in_op = cmd_op[2:0]
//              (when HAS_SFU = 0 they are executed as ALU PASS and raise illegal_op)
//
// Streams (all valid/ready, a transfer happens when both are high on a rising edge):
//   a  : operand A beats (always consumed)
//   b  : operand B beats, consumed only by binary ALU ops (ADD SUB MUL MAX MIN) and
//        REQUANT, and only when cmd_b_imm = 0; otherwise B = cmd_imm_b broadcast
//   p  : REQUANT per-lane mult (32b) / shift (5b), consumed only when cmd_p_imm = 0;
//        otherwise mult = cmd_mult, shift = cmd_shamt for every lane
//   d  : destination beats for all ops except REDUCE; d_keep marks lanes < len,
//        d_last marks the final beat of the row; lanes with d_keep = 0 read as 0
//   r  : one REDUCE result per REDUCE command
// The lanes of the final beat at positions >= len are zeroed before they reach any unit,
// and saturation flags of those lanes are never counted.
// Commands are accepted while the input side is idle. Up to RECQ commands may be in
// flight in the output side, so back-to-back rows pipeline without draining.
// len = 0 is a no-op (accepted, produces nothing, including no REDUCE result).
module vec_unit #(
    parameter int L       = 4,
    parameter int RW      = 16,   // row-length field width
    parameter int ACC_W   = 48,
    parameter int HAS_SFU = 0
) (
    input  logic             clk,
    input  logic             rst_n,
    // ---- command ----
    input  logic             cmd_valid,
    output logic             cmd_ready,
    input  logic [4:0]       cmd_op,
    input  logic [RW-1:0]    cmd_len,
    input  logic [1:0]       cmd_fmt,      // 0 int8, 1 int16, 2/3 int32 (REQUANT: 0 int8, else int16)
    input  logic [4:0]       cmd_shamt,
    input  logic [31:0]      cmd_lo,
    input  logic [31:0]      cmd_hi,
    input  logic             cmd_b_imm,
    input  logic [31:0]      cmd_imm_b,
    input  logic             cmd_p_imm,
    input  logic [31:0]      cmd_mult,
    // ---- source streams ----
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [L*32-1:0]  a_data,
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [L*32-1:0]  b_data,
    input  logic             p_valid,
    output logic             p_ready,
    input  logic [L*32-1:0]  p_mult,
    input  logic [L*5-1:0]   p_shift,
    // ---- destination stream ----
    output logic             d_valid,
    input  logic             d_ready,
    output logic [L*32-1:0]  d_data,
    output logic [L-1:0]     d_keep,
    output logic             d_last,
    // ---- reduce result ----
    output logic             r_valid,
    input  logic             r_ready,
    output logic [ACC_W-1:0] r_sum,
    output logic [ACC_W-1:0] r_sumsq,
    output logic [31:0]      r_max,
    output logic [31:0]      r_min,
    output logic             r_sq_err,
    // ---- SFU hook (SG5) ----
    output logic             sfu_in_valid,
    input  logic             sfu_in_ready,
    output logic [2:0]       sfu_in_op,
    output logic [1:0]       sfu_in_fmt,
    output logic [4:0]       sfu_in_shamt,
    output logic [31:0]      sfu_in_imm,
    output logic [L*32-1:0]  sfu_in_data,
    output logic [L-1:0]     sfu_in_keep,
    output logic             sfu_in_last,
    input  logic             sfu_out_valid,
    output logic             sfu_out_ready,
    input  logic [L*32-1:0]  sfu_out_data,
    input  logic [L-1:0]     sfu_out_keep,
    input  logic             sfu_out_last,
    // ---- status ----
    input  logic             stat_clr,
    output logic [31:0]      rq_sat_count,
    output logic [31:0]      alu_sat_count,
    output logic             sat_sticky,
    output logic             illegal_op,
    output logic             busy
);
    localparam logic [1:0] U_ALU = 2'd0, U_RQ = 2'd1, U_RED = 2'd2, U_SFU = 2'd3;
    localparam int RECQ = 4;

    function automatic logic [1:0] unit_of(input logic [4:0] op);
        if (op[4] == 1'b0)                   unit_of = U_ALU;
        else if (op == 5'h10)                unit_of = U_RQ;
        else if (op == 5'h11)                unit_of = U_RED;
        else if (op[4:3] == 2'b11 && HAS_SFU != 0) unit_of = U_SFU;
        else                                 unit_of = U_ALU;
    endfunction
    function automatic logic is_illegal(input logic [4:0] op);
        is_illegal = op[4] && !(op == 5'h10 || op == 5'h11 ||
                                (op[4:3] == 2'b11 && HAS_SFU != 0));
    endfunction

    // ---------------- command latch / input side ----------------
    logic          busy_in;
    logic [1:0]    c_unit;
    logic [3:0]    c_aop;
    logic          c_need_b, c_need_p;
    logic [4:0]    c_op;
    logic [1:0]    c_fmt;
    logic [4:0]    c_shamt;
    logic [31:0]   c_lo, c_hi, c_imm_b, c_mult;
    logic          c_b_imm, c_p_imm;
    logic [RW:0]   rem;                 // elements not yet issued
    logic          srcs_ok, u_ready, in_fire;
    logic          alu_in_ready, rq_in_ready, red_in_ready;

    // output-side record queue (unit of each in-flight elementwise command)
    logic [1:0]    rq_unit [RECQ];
    logic [$clog2(RECQ):0] rq_cnt;
    logic [$clog2(RECQ)-1:0] rq_wp, rq_rp;
    logic          rq_full, rq_empty, rq_push, rq_pop;
    assign rq_full  = (rq_cnt == ($clog2(RECQ)+1)'(RECQ));
    assign rq_empty = (rq_cnt == 0);

    logic [1:0] n_unit;
    assign n_unit   = unit_of(cmd_op);
    assign cmd_ready = !busy_in && !rq_full;
    logic cmd_fire;
    assign cmd_fire = cmd_valid && cmd_ready;
    assign rq_push  = cmd_fire && (cmd_len != '0) && (n_unit != U_RED);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy_in <= 1'b0;
        end else if (cmd_fire) begin
            busy_in  <= (cmd_len != '0);
            rem      <= {1'b0, cmd_len};
            c_unit   <= n_unit;
            c_aop    <= (cmd_op[4] == 1'b0) ? cmd_op[3:0] : 4'd0;
            c_need_b <= !cmd_b_imm && ((n_unit == U_RQ) ||
                        (n_unit == U_ALU && cmd_op[4] == 1'b0 &&
                         (cmd_op[3:0] >= 4'd1 && cmd_op[3:0] <= 4'd5)));
            c_need_p <= !cmd_p_imm && (n_unit == U_RQ);
            c_op     <= cmd_op;
            c_fmt    <= cmd_fmt;
            c_shamt  <= cmd_shamt;
            c_lo     <= cmd_lo;
            c_hi     <= cmd_hi;
            c_b_imm  <= cmd_b_imm;
            c_imm_b  <= cmd_imm_b;
            c_p_imm  <= cmd_p_imm;
            c_mult   <= cmd_mult;
        end else if (in_fire) begin
            if (rem <= (RW+1)'(L)) busy_in <= 1'b0;
            rem <= rem - (RW+1)'(L);
        end
    end

    // keep mask / last for the beat being issued
    logic [L-1:0] keep;
    logic         last;
    always_comb begin
        for (int j = 0; j < L; j++) keep[j] = ((RW+1)'(j) < rem);
        last = (rem <= (RW+1)'(L));
    end

    // operand selection with masking of lanes >= len
    logic [L*32-1:0] am, bm, pm;
    logic [L*5-1:0]  ps;
    always_comb begin
        for (int j = 0; j < L; j++) begin
            am[j*32 +: 32] = keep[j] ? a_data[j*32 +: 32] : 32'd0;
            bm[j*32 +: 32] = keep[j] ? (c_b_imm ? c_imm_b : b_data[j*32 +: 32]) : 32'd0;
            pm[j*32 +: 32] = c_p_imm ? c_mult  : p_mult[j*32 +: 32];
            ps[j*5 +: 5]   = c_p_imm ? c_shamt : p_shift[j*5 +: 5];
        end
    end

    always_comb begin
        srcs_ok = busy_in && a_valid && (!c_need_b || b_valid) && (!c_need_p || p_valid);
        case (c_unit)
            U_ALU:   u_ready = alu_in_ready;
            U_RQ:    u_ready = rq_in_ready;
            U_RED:   u_ready = red_in_ready;
            default: u_ready = sfu_in_ready;
        endcase
    end
    assign in_fire = srcs_ok && u_ready;
    assign a_ready = busy_in && (!c_need_b || b_valid) && (!c_need_p || p_valid) && u_ready;
    assign b_ready = busy_in && c_need_b && a_valid && (!c_need_p || p_valid) && u_ready;
    assign p_ready = busy_in && c_need_p && a_valid && (!c_need_b || b_valid) && u_ready;

    // ---------------- units ----------------
    logic alu_out_valid, alu_out_ready, alu_out_last;
    logic [L*32-1:0] alu_out_data;
    logic [L-1:0] alu_out_keep, alu_out_sat;
    logic alu_sticky, rq_sticky;

    vec_alu #(.L(L)) u_alu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(srcs_ok && c_unit == U_ALU), .in_ready(alu_in_ready),
        .in_a(am), .in_b(bm), .in_op(c_aop), .in_fmt(c_fmt), .in_shamt(c_shamt),
        .in_lo(c_lo), .in_hi(c_hi), .in_keep(keep), .in_last(last),
        .out_valid(alu_out_valid), .out_ready(alu_out_ready), .out_data(alu_out_data),
        .out_keep(alu_out_keep), .out_sat(alu_out_sat), .out_last(alu_out_last),
        .sat_clr(stat_clr), .sat_count(alu_sat_count), .sat_sticky(alu_sticky)
    );

    logic rq_out_valid, rq_out_ready, rq_out_last;
    logic [L*16-1:0] rq_out_data;
    logic [L-1:0] rq_out_keep, rq_out_sat;

    requant #(.L(L)) u_rq (
        .clk(clk), .rst_n(rst_n),
        .in_valid(srcs_ok && c_unit == U_RQ), .in_ready(rq_in_ready),
        .in_acc(am), .in_bias(bm), .in_mult(pm), .in_shift(ps),
        .in_keep(keep), .in_last(last), .in_out16(c_fmt != 2'd0),
        .out_valid(rq_out_valid), .out_ready(rq_out_ready), .out_data(rq_out_data),
        .out_keep(rq_out_keep), .out_sat(rq_out_sat), .out_last(rq_out_last),
        .sat_clr(stat_clr), .sat_count(rq_sat_count), .sat_sticky(rq_sticky)
    );

    vec_reduce #(.L(L), .ACC_W(ACC_W)) u_red (
        .clk(clk), .rst_n(rst_n),
        .in_valid(srcs_ok && c_unit == U_RED), .in_ready(red_in_ready),
        .in_data(am), .in_keep(keep), .in_last(last),
        .out_valid(r_valid), .out_ready(r_ready), .out_sum(r_sum), .out_sumsq(r_sumsq),
        .out_max(r_max), .out_min(r_min), .out_sq_err(r_sq_err)
    );

    // SFU hook
    assign sfu_in_valid = srcs_ok && c_unit == U_SFU;
    assign sfu_in_op    = c_op[2:0];
    assign sfu_in_fmt   = c_fmt;
    assign sfu_in_shamt = c_shamt;
    assign sfu_in_imm   = c_imm_b;
    assign sfu_in_data  = am;
    assign sfu_in_keep  = keep;
    assign sfu_in_last  = last;

    // ---------------- output side ----------------
    logic [1:0] h;
    assign h = rq_unit[rq_rp];
    logic            o_valid, o_last;
    logic [L*32-1:0] o_data;
    logic [L-1:0]    o_keep;
    always_comb begin
        o_valid = 1'b0; o_last = 1'b0; o_data = '0; o_keep = '0;
        if (!rq_empty) begin
            case (h)
                U_RQ: begin
                    o_valid = rq_out_valid; o_last = rq_out_last; o_keep = rq_out_keep;
                    for (int j = 0; j < L; j++)
                        o_data[j*32 +: 32] = {{16{rq_out_data[j*16+15]}}, rq_out_data[j*16 +: 16]};
                end
                U_SFU: begin
                    o_valid = sfu_out_valid; o_last = sfu_out_last; o_keep = sfu_out_keep;
                    o_data = sfu_out_data;
                end
                default: begin
                    o_valid = alu_out_valid; o_last = alu_out_last; o_keep = alu_out_keep;
                    o_data = alu_out_data;
                end
            endcase
        end
        for (int j = 0; j < L; j++) if (!o_keep[j]) o_data[j*32 +: 32] = 32'd0;
    end
    assign d_valid = o_valid;
    assign d_data  = o_data;
    assign d_keep  = o_keep;
    assign d_last  = o_last;
    assign alu_out_ready = !rq_empty && h == U_ALU && d_ready;
    assign rq_out_ready  = !rq_empty && h == U_RQ  && d_ready;
    assign sfu_out_ready = !rq_empty && h == U_SFU && d_ready;
    assign rq_pop = d_valid && d_ready && d_last;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rq_cnt <= '0; rq_wp <= '0; rq_rp <= '0;
        end else begin
            if (rq_push) begin rq_unit[rq_wp] <= n_unit; rq_wp <= rq_wp + 1'b1; end
            if (rq_pop) rq_rp <= rq_rp + 1'b1;
            rq_cnt <= rq_cnt + {{$clog2(RECQ){1'b0}}, rq_push} - {{$clog2(RECQ){1'b0}}, rq_pop};
        end
    end

    // ---------------- status ----------------
    always_ff @(posedge clk) begin
        if (!rst_n) illegal_op <= 1'b0;
        else illegal_op <= (stat_clr ? 1'b0 : illegal_op) | (cmd_fire && is_illegal(cmd_op));
    end
    assign sat_sticky = alu_sticky | rq_sticky;
    // busy: a command is issuing, an elementwise row is still draining, or a reduce
    // result is pending (in the pipeline or on r_*).
    logic [2:0] red_inflight;           // <= 4 rows can be in the 3-stage pipe + result reg
    logic       red_inc, red_dec;
    assign red_inc = in_fire && c_unit == U_RED && last;
    assign red_dec = r_valid && r_ready;
    always_ff @(posedge clk) begin
        if (!rst_n) red_inflight <= '0;
        else red_inflight <= red_inflight + {2'b0, red_inc} - {2'b0, red_dec};
    end
    assign busy = busy_in || !rq_empty || (red_inflight != '0);
endmodule
