// sfu_vec.sv -- vector unit with the C5 SFU block attached to its hook: vec_unit
// (HAS_SFU = 1) and sfu wired port group to port group, as docs/vector-unit.md §5.6
// prescribes. vec_unit itself is unchanged. Opcodes 0x18-0x1F reach the SFU with sub-op
// = cmd_op[2:0] and imm = cmd_imm_b (table / LN / score-record slot, softmax causal
// limit); see docs/sfu.md.
module sfu_vec #(
    parameter int L       = 4,
    parameter int RW      = 16,
    parameter int ACC_W   = 48,
    parameter int CTX_MAX = 256,
    parameter int D_MAX   = 384,
    parameter int NG      = 4,
    parameter int NSLOT   = 16,
    parameter int NCTX    = 8
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             cmd_valid,
    output logic             cmd_ready,
    input  logic [4:0]       cmd_op,
    input  logic [RW-1:0]    cmd_len,
    input  logic [1:0]       cmd_fmt,
    input  logic [4:0]       cmd_shamt,
    input  logic [31:0]      cmd_lo,
    input  logic [31:0]      cmd_hi,
    input  logic             cmd_b_imm,
    input  logic [31:0]      cmd_imm_b,
    input  logic             cmd_p_imm,
    input  logic [31:0]      cmd_mult,
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
    output logic             d_valid,
    input  logic             d_ready,
    output logic [L*32-1:0]  d_data,
    output logic [L-1:0]     d_keep,
    output logic             d_last,
    output logic             r_valid,
    input  logic             r_ready,
    output logic [ACC_W-1:0] r_sum,
    output logic [ACC_W-1:0] r_sumsq,
    output logic [31:0]      r_max,
    output logic [31:0]      r_min,
    output logic             r_sq_err,
    input  logic             cfg_we,
    input  logic [3:0]       cfg_region,
    input  logic [7:0]       cfg_slot,
    input  logic [11:0]      cfg_line,
    input  logic [12:0]      cfg_cols,
    input  logic [127:0]     cfg_data,
    input  logic             stat_clr,
    output logic [31:0]      rq_sat_count,
    output logic [31:0]      alu_sat_count,
    output logic [31:0]      sm_sat_count,
    output logic [31:0]      ln_z_sat_count,
    output logic [31:0]      ln_out_sat_count,
    output logic             sat_sticky,
    output logic             illegal_op,
    output logic             sfu_err_len,
    output logic             busy
);
    logic            si_valid, si_ready, si_last, so_valid, so_ready, so_last;
    logic [2:0]      si_op;
    logic [1:0]      si_fmt;
    logic [4:0]      si_shamt;
    logic [31:0]     si_imm;
    logic [L*32-1:0] si_data, so_data;
    logic [L-1:0]    si_keep, so_keep;
    logic            v_sticky, v_illegal, v_busy, s_sticky, s_illegal, s_busy;

    vec_unit #(.L(L), .RW(RW), .ACC_W(ACC_W), .HAS_SFU(1)) u_vec (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op), .cmd_len(cmd_len),
        .cmd_fmt(cmd_fmt), .cmd_shamt(cmd_shamt), .cmd_lo(cmd_lo), .cmd_hi(cmd_hi),
        .cmd_b_imm(cmd_b_imm), .cmd_imm_b(cmd_imm_b), .cmd_p_imm(cmd_p_imm), .cmd_mult(cmd_mult),
        .a_valid(a_valid), .a_ready(a_ready), .a_data(a_data),
        .b_valid(b_valid), .b_ready(b_ready), .b_data(b_data),
        .p_valid(p_valid), .p_ready(p_ready), .p_mult(p_mult), .p_shift(p_shift),
        .d_valid(d_valid), .d_ready(d_ready), .d_data(d_data), .d_keep(d_keep), .d_last(d_last),
        .r_valid(r_valid), .r_ready(r_ready), .r_sum(r_sum), .r_sumsq(r_sumsq), .r_max(r_max),
        .r_min(r_min), .r_sq_err(r_sq_err),
        .sfu_in_valid(si_valid), .sfu_in_ready(si_ready), .sfu_in_op(si_op),
        .sfu_in_fmt(si_fmt), .sfu_in_shamt(si_shamt), .sfu_in_imm(si_imm),
        .sfu_in_data(si_data), .sfu_in_keep(si_keep), .sfu_in_last(si_last),
        .sfu_out_valid(so_valid), .sfu_out_ready(so_ready), .sfu_out_data(so_data),
        .sfu_out_keep(so_keep), .sfu_out_last(so_last),
        .stat_clr(stat_clr), .rq_sat_count(rq_sat_count), .alu_sat_count(alu_sat_count),
        .sat_sticky(v_sticky), .illegal_op(v_illegal), .busy(v_busy)
    );

    sfu #(.L(L), .CTX_MAX(CTX_MAX), .D_MAX(D_MAX), .NG(NG), .NSLOT(NSLOT), .NCTX(NCTX)) u_sfu (
        .clk(clk), .rst_n(rst_n),
        .in_valid(si_valid), .in_ready(si_ready), .in_op(si_op), .in_fmt(si_fmt),
        .in_shamt(si_shamt), .in_imm(si_imm), .in_data(si_data), .in_keep(si_keep),
        .in_last(si_last),
        .out_valid(so_valid), .out_ready(so_ready), .out_data(so_data), .out_keep(so_keep),
        .out_last(so_last),
        .cfg_we(cfg_we), .cfg_region(cfg_region), .cfg_slot(cfg_slot), .cfg_line(cfg_line),
        .cfg_cols(cfg_cols), .cfg_data(cfg_data),
        .stat_clr(stat_clr), .sm_sat_count(sm_sat_count), .ln_z_sat_count(ln_z_sat_count),
        .ln_out_sat_count(ln_out_sat_count), .sat_sticky(s_sticky), .illegal_op(s_illegal),
        .err_len(sfu_err_len), .busy(s_busy)
    );

    assign sat_sticky = v_sticky | s_sticky;
    assign illegal_op = v_illegal | s_illegal;
    assign busy       = v_busy | s_busy;
endmodule
