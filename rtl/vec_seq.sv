// vec_seq.sv -- VEC engine of the command processor (SG7b): executes the VEC-queue ISA
// commands (docs/isa.md §4) on vec_unit + SFU (rtl/sfu_vec.sv).
//
//   VEC             rows of <= CH elements, vec_alu op (COPY=PASS, ADD, SUB, MUL, MAX, MIN,
//                   ADDI = ADD with B = imm, SHRR = SRAR, SHL), any buffers, int8/16/32
//   REQUANT         per row of the ACC tile: vec_unit REQUANT; per-channel records from a
//                   record buffer (columns in chunks of RQ_MAX), or per-tensor immediates
//   LN              per row: SFU LN (sub-op 5) with the block's LN slot
//   SOFTMAX_ONLINE  per row: SFU SOFTMAX_RQ (sub-op 6), ctx-record slot, causal limit
//   GELU            rows of <= CH bytes: SFU GELU (sub-op 3) with the table's slot
//   KV_APPEND       per (t, head): vec_unit PASS int8 of dh bytes, ABUF -> KVBUF
//   LUT_LOAD        32 PBUF lines -> SFU table (pbuf_loader)
// Parameter blocks are loaded by pbuf_loader, which keeps them resident per slot.
//
// Engine port (docs/control.md §2): cmd_valid/cmd_ready/cmd_word; one `done` pulse per
// accepted command, in order, at least one cycle after acceptance, once the command's last
// scratchpad write has been accepted. Commands execute one at a time (a 2-entry queue
// accepts the next ones), so every command sees the results of the previous one.
// `err` (sticky; cleared by reset or clr) reports an illegal command (skipped, still done)
// or a unit error; err_code holds the first cause.
//
// Scratchpad ports: three read ports (A: src0 / scores / acc / activations, B: src1,
// P: PBUF parameters) and one write port (D), each moving PL 16-byte lines per beat from any
// start line (PL = 1: the DMA's sp_* line port; line j of a beat in bits [128j +: 128], its
// strobes in [16j +: 16]). PBUF writes by other units must be reported on pb_snoop_*.
// Throughput per port: PL lines per cycle (docs/vector-unit.md §9.3).
module vec_seq #(
    parameter int L       = 4,
    parameter int AW      = 17,
    parameter int CTX_MAX = 256,
    parameter int D_MAX   = 384,
    parameter int NG      = 4,
    parameter int NSLOT   = 16,
    parameter int NCTX    = 8,
    parameter int RQ_MAX  = 256,
    parameter int CH      = 4096,
    parameter int FD      = 16,
    parameter int PL      = 1           // lines per beat on the A, B, P and D ports (16PL bytes)
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             clr,
    // engine command port
    input  logic             cmd_valid,
    output logic             cmd_ready,
    input  logic [127:0]     cmd_word,
    output logic             done,
    output logic             err,
    output logic [3:0]       err_code,
    output logic             busy,
    // read port A
    output logic             ra_arvalid,
    input  logic             ra_arready,
    output logic [2:0]       ra_arbuf,
    output logic [AW-1:0]    ra_araddr,
    input  logic             ra_rvalid,
    input  logic [PL*128-1:0] ra_rdata,
    // read port B
    output logic             rb_arvalid,
    input  logic             rb_arready,
    output logic [2:0]       rb_arbuf,
    output logic [AW-1:0]    rb_araddr,
    input  logic             rb_rvalid,
    input  logic [PL*128-1:0] rb_rdata,
    // read port P (PBUF)
    output logic             rp_arvalid,
    input  logic             rp_arready,
    output logic [AW-1:0]    rp_araddr,
    input  logic             rp_rvalid,
    input  logic [PL*128-1:0] rp_rdata,
    // write port D
    output logic             wd_valid,
    input  logic             wd_ready,
    output logic [2:0]       wd_buf,
    output logic [AW-1:0]    wd_addr,
    output logic [PL*128-1:0] wd_data,
    output logic [PL*16-1:0]  wd_strb,
    // PBUF writes of other units (DMA)
    input  logic             pb_snoop_valid,
    input  logic [AW-1:0]    pb_snoop_line,
    // saturation counters (CSR SAT_i): 0 requant, 1 ALU, 2 LN out, 3 LN z, 4 softmax score
    input  logic             sat_clr,
    output logic [5*32-1:0]  sat_cnt,
    // parameter-load statistics
    output logic [31:0]      st_hits,
    output logic [31:0]      st_misses,
    output logic [31:0]      st_lines
);
    localparam int LL  = $clog2(L);
    localparam int LCH = $clog2(CH);
    localparam logic [2:0] B_ABUF = 3'd0, B_ACC = 3'd2, B_PBUF = 3'd3, B_KV = 3'd4;
    localparam logic [2:0] K_LUT = 3'd0, K_GELU = 3'd1, K_LN = 3'd2, K_CTX = 3'd3, K_REC = 3'd4;
    localparam logic [3:0] E_OP = 4'd1, E_FIELD = 4'd2, E_IMM = 4'd3, E_SM = 4'd4, E_SIZE = 4'd5,
                           E_LUT = 4'd6, E_KV = 4'd7, E_LS = 4'd8, E_UNIT = 4'd9;
    // command kinds
    localparam logic [2:0] C_VEC = 3'd0, C_RQ = 3'd1, C_LN = 3'd2, C_SM = 3'd3, C_GELU = 3'd4,
                           C_KV = 3'd5, C_LUT = 3'd6;

    logic rst_i;
    assign rst_i = rst_n && !clr;

    // ---------------- command queue + field decode ----------------
    logic         cq_valid, cq_pop;
    logic [127:0] cq_word, cw;
    fifo_sync #(.W(128), .DEPTH(2)) u_cq (
        .clk(clk), .rst_n(rst_i), .in_valid(cmd_valid), .in_ready(cmd_ready), .in_data(cmd_word),
        .out_valid(cq_valid), .out_ready(cq_pop), .out_data(cq_word), .count()
    );

    logic [5:0]  f_opcode;
    logic        f_known, f_resv_ok;
    logic [1:0]  f_q;
    logic        is_vec, is_rq, is_ln, is_sm, is_gelu, is_kv, is_lut;
    logic [4:0]  vec_vop;
    logic [1:0]  vec_dt_in, vec_dt_out, lut_table;
    logic [2:0]  vec_src0_buf, vec_src1_buf, vec_dst_buf;
    logic [20:0] vec_src0_addr, vec_src1_addr, vec_dst_addr, vec_len, gelu_len, kv_dst_addr;
    logic [15:0] vec_imm;
    logic        rq_out16, rq_pt, ln_in16, sm_first, sm_last, sm_causal;
    logic [19:0] rq_src, rq_dst, rq_par, ln_src, ln_dst, ln_par, sm_src, sm_dst, sm_ctx;
    logic [19:0] gelu_src, gelu_dst, gelu_lut, kv_src, lut_src;
    logic [12:0] rq_m, rq_n, ln_rows, ln_cols, sm_rows, sm_cols, sm_diag, kv_ctx, kv_rows;
    logic [11:0] rq_ss, rq_ds, ln_ss, ln_ds, kv_pos, kv_ss;
    logic [9:0]  sm_ss, sm_ds;
    logic [6:0]  kv_heads;
    logic [8:0]  kv_dh;

    /* verilator lint_off PINCONNECTEMPTY */
    cmd_fields u_f (
        .w(cw), .opcode(f_opcode), .known(f_known), .resv_ok(f_resv_ok), .fixed_q(f_q),
        .is_end(), .is_irq(), .is_wait(), .is_signal(), .is_dma_load(), .is_dma_store(),
        .is_gemm(), .is_attn_tile(), .is_vec(is_vec), .is_requant(is_rq), .is_ln(is_ln),
        .is_softmax_online(is_sm), .is_gelu(is_gelu), .is_kv_append(is_kv),
        .is_gemm_gs(), .is_lut_load(is_lut),
        .irq_code(), .wait_queue(), .wait_sem(), .wait_count(), .signal_queue(), .signal_sem(),
        .dma_load_chan(), .dma_load_sp_buf(), .dma_load_sp_addr(), .dma_load_ddr_addr(),
        .dma_load_rows(), .dma_load_row_bytes(), .dma_load_ddr_stride(), .dma_load_sp_stride(),
        .dma_store_chan(), .dma_store_sp_buf(), .dma_store_sp_addr(), .dma_store_ddr_addr(),
        .dma_store_rows(), .dma_store_row_bytes(), .dma_store_ddr_stride(), .dma_store_sp_stride(),
        .gemm_int4(), .gemm_gemv(), .gemm_w_trans(), .gemm_accumulate(), .gemm_a_addr(),
        .gemm_w_buf(), .gemm_w_addr(), .gemm_c_addr(), .gemm_m(), .gemm_n(), .gemm_k(),
        .gemm_a_stride(), .gemm_c_stride(),
        .attn_tile_first(), .attn_tile_last(), .attn_tile_q_addr(), .attn_tile_k_addr(),
        .attn_tile_v_addr(), .attn_tile_o_addr(), .attn_tile_ctx_addr(), .attn_tile_rows(),
        .attn_tile_len(),
        .vec_vop(vec_vop), .vec_dt_in(vec_dt_in), .vec_dt_out(vec_dt_out),
        .vec_src0_buf(vec_src0_buf), .vec_src0_addr(vec_src0_addr), .vec_src1_buf(vec_src1_buf),
        .vec_src1_addr(vec_src1_addr), .vec_dst_buf(vec_dst_buf), .vec_dst_addr(vec_dst_addr),
        .vec_len(vec_len), .vec_imm(vec_imm),
        .requant_out16(rq_out16), .requant_per_tensor(rq_pt), .requant_src_addr(rq_src),
        .requant_dst_addr(rq_dst), .requant_param_addr(rq_par), .requant_m(rq_m),
        .requant_n(rq_n), .requant_src_stride(rq_ss), .requant_dst_stride(rq_ds),
        .ln_in16(ln_in16), .ln_src_addr(ln_src), .ln_dst_addr(ln_dst), .ln_param_addr(ln_par),
        .ln_rows(ln_rows), .ln_cols(ln_cols), .ln_src_stride(ln_ss), .ln_dst_stride(ln_ds),
        .softmax_online_first(sm_first), .softmax_online_last(sm_last),
        .softmax_online_causal(sm_causal), .softmax_online_src_addr(sm_src),
        .softmax_online_dst_addr(sm_dst), .softmax_online_ctx_addr(sm_ctx),
        .softmax_online_rows(sm_rows), .softmax_online_cols(sm_cols),
        .softmax_online_diag(sm_diag), .softmax_online_src_stride(sm_ss),
        .softmax_online_dst_stride(sm_ds),
        .gelu_src_addr(gelu_src), .gelu_dst_addr(gelu_dst), .gelu_lut_addr(gelu_lut),
        .gelu_len(gelu_len),
        .kv_append_src_addr(kv_src), .kv_append_dst_addr(kv_dst_addr), .kv_append_pos(kv_pos),
        .kv_append_heads(kv_heads), .kv_append_dh(kv_dh), .kv_append_ctx(kv_ctx),
        .kv_append_rows(kv_rows), .kv_append_src_stride(kv_ss),
        .gemm_gs_gemv(), .gemm_gs_accumulate(), .gemm_gs_a_addr(), .gemm_gs_w_addr(),
        .gemm_gs_c_addr(), .gemm_gs_m(), .gemm_gs_n(), .gemm_gs_k(), .gemm_gs_a_stride(),
        .gemm_gs_c_stride(), .gemm_gs_glog2(),
        .lut_load_table(lut_table), .lut_load_src_addr(lut_src)
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // ---------------- per-command state ----------------
    typedef enum logic [3:0] {S_IDLE, S_DEC, S_RQC, S_LREQ, S_LOAD, S_SET, S_RUN0, S_RUN, S_FIN}
        state_t;
    state_t st;
    logic [2:0]   kind;
    logic [1:0]   settle;
    // load request
    logic [2:0]   l_kind;
    logic [AW-1:0] l_line;
    logic [12:0]  l_cols, l_n;
    logic [1:0]   l_table;
    logic [7:0]   slot;
    // stream descriptors (A, B, D)
    logic [2:0]   a_buf, b_buf, d_buf;
    logic [AW-1:0] a_base, a_so, a_si, b_base, b_so, d_base, d_so, d_si;
    logic [12:0]  s_no;
    logic [6:0]   s_ni;
    logic [18:0]  s_nrows;
    logic [15:0]  s_len, s_len_last;
    logic [1:0]   a_ebl, d_ebl;
    logic         use_b;
    // vec_unit row template
    logic [4:0]   t_op;
    logic [1:0]   t_fmt;
    logic [4:0]   t_shamt;
    logic         t_b_imm, t_p_imm;
    logic [31:0]  t_imm_b;
    // softmax per-row limit
    logic         t_causal;
    logic [12:0]  t_cols;
    logic signed [13:0] t_diag1;            // diag + 1
    // requant
    logic         rq_chan;                  // per-channel records on B/P
    logic [12:0]  c0, nc_w;
    logic [AW-1:0] rq_src_l, rq_dst_l, rq_par_l;
    logic [12:0]  rq_mq, rq_nq;
    logic [11:0]  rq_ssq, rq_dsq;
    logic         rq_o16, rq_ptq;
    // row issue
    logic [18:0]  ri;
    logic         lut_ok [3];

    // ---------------- units ----------------
    logic            ld_idle, ld_done, ld_hit, ld_req;
    logic [7:0]      ld_slot;
    logic            cfg_we;
    logic [3:0]      cfg_region;
    logic [7:0]      cfg_slot;
    logic [11:0]     cfg_line;
    logic [12:0]     cfg_cols;
    logic [127:0]    cfg_data;
    logic [PL-1:0]   rec_we;
    logic [12:0]     rec_idx;
    logic [PL*128-1:0] rec_data;
    logic [PL-1:0]   own_snoop;             // lines of an accepted PBUF write that carry a strobe
    for (genvar j = 0; j < PL; j++) begin : g_snoop
        assign own_snoop[j] = wd_valid && wd_ready && wd_buf == B_PBUF && (|wd_strb[j*16 +: 16]);
    end

    pbuf_loader #(.AW(AW), .NSLOT(NSLOT), .NG(NG), .NCTX(NCTX), .FD(FD), .P(PL)) u_ld (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .req_valid(ld_req), .idle(ld_idle), .req_kind(l_kind), .req_line(l_line),
        .req_cols(l_cols), .req_table(l_table), .req_n(l_n),
        .done(ld_done), .slot(ld_slot), .hit(ld_hit),
        .snoop0_valid(pb_snoop_valid), .snoop0_line(pb_snoop_line),
        .snoop1_valid(own_snoop), .snoop1_line(wd_addr),
        .ar_valid(rp_arvalid), .ar_ready(rp_arready), .ar_addr(rp_araddr),
        .r_valid(rp_rvalid), .r_data(rp_rdata),
        .cfg_we(cfg_we), .cfg_region(cfg_region), .cfg_slot(cfg_slot), .cfg_line(cfg_line),
        .cfg_cols(cfg_cols), .cfg_data(cfg_data),
        .rec_we(rec_we), .rec_idx(rec_idx), .rec_data(rec_data),
        .st_hits(st_hits), .st_misses(st_misses), .st_lines(st_lines)
    );

    // record buffer (REQUANT): bias, mult, shift per output channel of the current chunk;
    // up to PL consecutive records written per cycle (record rec_idx + j from line j).
    // The PL lines are rotated by rec_idx mod PL into slots (slot q holds the record whose index
    // is q mod PL), so every entry has one fixed write source (docs/fpga.md C9) instead of a
    // PL-way mux.
    localparam int PLB  = (PL > 1) ? $clog2(PL) : 1;
    logic [31:0] rec_bias  [RQ_MAX];
    logic [31:0] rec_mult  [RQ_MAX];
    logic [4:0]  rec_shift [RQ_MAX];
    logic        ls_bad;
    logic [PL*128-1:0] rec_rot;
    logic [PL-1:0]     rec_we_rot;
    logic [13:0]       rec_idx_q [PL];
    logic [PLB-1:0]    rec_r;
    assign rec_r = (PL > 1) ? PLB'(rec_idx) : '0;
    always_comb begin
        rec_rot    = rec_data;
        rec_we_rot = rec_we;
        for (int k = 0; k < PLB; k++) begin
            if (PL > 1 && rec_r[k]) begin
                rec_rot    = (PL*128)'({rec_rot, rec_rot} >> ((PL - (1 << k)) * 128));
                rec_we_rot = PL'({rec_we_rot, rec_we_rot} >> (PL - (1 << k)));
            end
        end
        // slot q holds line (q - rec_idx) mod PL, i.e. record rec_idx + ((q - rec_idx) mod PL)
        for (int q = 0; q < PL; q++)
            rec_idx_q[q] = 14'(rec_idx) + 14'((PLB'(q) - rec_r) & PLB'(PL - 1));
    end
    // one always_ff per entry (generate loop): portable to Verilator < 5.036, which rejects
    // delayed array assignments inside procedural for loops (BLKLOOPINIT)
    for (genvar ge = 0; ge < RQ_MAX; ge++) begin : g_rec
        always_ff @(posedge clk) begin
            if (rec_we != '0 && rec_we_rot[ge % PL] && rec_idx_q[ge % PL] == 14'(ge)) begin
                rec_bias[ge]  <= rec_rot[(ge % PL)*128 +: 32];
                rec_mult[ge]  <= rec_rot[(ge % PL)*128 + 32 +: 32];
                rec_shift[ge] <= rec_rot[(ge % PL)*128 + 64 +: 5];
            end
        end
    end
    logic [PL-1:0] ls_bad_j;
    for (genvar j = 0; j < PL; j++) begin : g_lsbad
        assign ls_bad_j[j] = rec_we[j] && (rec_data[j*128 + 96 +: 32] != '0);
    end
    assign ls_bad = |ls_bad_j;

    // A / B readers, D writer
    logic            rd_start, ra_busy, rb_busy, wr_busy;
    logic            a_v, a_r, a_l, b_v, b_r, b_l;
    logic [L*32-1:0] a_d, b_d;
    vec_seq_rd #(.L(L), .AW(AW), .FD(FD), .P(PL)) u_ra (
        .clk(clk), .rst_n(rst_n), .clr(clr), .start(rd_start), .d_buf(a_buf), .d_base(a_base),
        .d_so(a_so), .d_no(s_no), .d_si(a_si), .d_ni(s_ni), .d_nrows(s_nrows), .d_len(s_len),
        .d_len_last(s_len_last), .d_ebl(a_ebl), .busy(ra_busy),
        .ar_valid(ra_arvalid), .ar_ready(ra_arready), .ar_buf(ra_arbuf), .ar_addr(ra_araddr),
        .r_valid(ra_rvalid), .r_data(ra_rdata),
        .o_valid(a_v), .o_ready(a_r), .o_data(a_d), .o_last(a_l)
    );
    vec_seq_rd #(.L(L), .AW(AW), .FD(FD), .P(PL)) u_rb (
        .clk(clk), .rst_n(rst_n), .clr(clr), .start(rd_start && use_b), .d_buf(b_buf),
        .d_base(b_base), .d_so(b_so), .d_no(s_no), .d_si('0), .d_ni(7'd1), .d_nrows(s_nrows),
        .d_len(s_len), .d_len_last(s_len_last), .d_ebl(a_ebl), .busy(rb_busy),
        .ar_valid(rb_arvalid), .ar_ready(rb_arready), .ar_buf(rb_arbuf), .ar_addr(rb_araddr),
        .r_valid(rb_rvalid), .r_data(rb_rdata),
        .o_valid(b_v), .o_ready(b_r), .o_data(b_d), .o_last(b_l)
    );

    logic            vd_valid, vd_ready, vd_last;
    logic [L*32-1:0] vd_data;
    logic [L-1:0]    vd_keep;
    vec_seq_wr #(.L(L), .AW(AW), .P(PL)) u_wd (
        .clk(clk), .rst_n(rst_n), .clr(clr), .start(rd_start), .d_buf(d_buf), .d_base(d_base),
        .d_so(d_so), .d_no(s_no), .d_si(d_si), .d_ni(s_ni), .d_len(s_len),
        .d_len_last(s_len_last), .d_ebl(d_ebl), .busy(wr_busy),
        .i_valid(vd_valid), .i_ready(vd_ready), .i_data(vd_data), .i_keep(vd_keep),
        .i_last(vd_last),
        .w_valid(wd_valid), .w_ready(wd_ready), .w_buf(wd_buf), .w_addr(wd_addr),
        .w_data(wd_data), .w_strb(wd_strb)
    );

    // vec_unit + SFU
    logic            vu_cmd_valid, vu_cmd_ready;
    logic [15:0]     vu_len;
    logic [31:0]     vu_imm_b, vu_mult;
    logic [4:0]      vu_shamt;
    logic            vu_b_valid, vu_b_ready, vu_p_valid, vu_p_ready;
    logic [L*32-1:0] vu_b_data, vu_p_mult;
    logic [L*5-1:0]  vu_p_shift;
    logic [31:0]     rq_cnt, alu_cnt, sm_cnt, lnz_cnt, lno_cnt;
    logic            vu_sticky, vu_illegal, vu_err_len, vu_busy;

    sfu_vec #(.L(L), .RW(16), .CTX_MAX(CTX_MAX), .D_MAX(D_MAX), .NG(NG), .NSLOT(NSLOT),
              .NCTX(NCTX)) u_vu (
        .clk(clk), .rst_n(rst_i),
        .cmd_valid(vu_cmd_valid), .cmd_ready(vu_cmd_ready), .cmd_op(t_op), .cmd_len(vu_len),
        .cmd_fmt(t_fmt), .cmd_shamt(vu_shamt), .cmd_lo('0), .cmd_hi('0), .cmd_b_imm(t_b_imm),
        .cmd_imm_b(vu_imm_b), .cmd_p_imm(t_p_imm), .cmd_mult(vu_mult),
        .a_valid(a_v), .a_ready(a_r), .a_data(a_d),
        .b_valid(vu_b_valid), .b_ready(vu_b_ready), .b_data(vu_b_data),
        .p_valid(vu_p_valid), .p_ready(vu_p_ready), .p_mult(vu_p_mult), .p_shift(vu_p_shift),
        .d_valid(vd_valid), .d_ready(vd_ready), .d_data(vd_data), .d_keep(vd_keep),
        .d_last(vd_last),
        .r_valid(), .r_ready(1'b1), .r_sum(), .r_sumsq(), .r_max(), .r_min(), .r_sq_err(),
        .cfg_we(cfg_we), .cfg_region(cfg_region), .cfg_slot(cfg_slot), .cfg_line(cfg_line),
        .cfg_cols(cfg_cols), .cfg_data(cfg_data),
        .stat_clr(sat_clr), .rq_sat_count(rq_cnt), .alu_sat_count(alu_cnt),
        .sm_sat_count(sm_cnt), .ln_z_sat_count(lnz_cnt), .ln_out_sat_count(lno_cnt),
        .sat_sticky(vu_sticky), .illegal_op(vu_illegal), .sfu_err_len(vu_err_len),
        .busy(vu_busy)
    );
    assign sat_cnt = {sm_cnt, lnz_cnt, lno_cnt, alu_cnt, rq_cnt};

    // B / P stream sources: B reader (VEC binary ops) or the record buffer (REQUANT)
    logic [15:0] rb_beat, rb_nb;
    always_comb begin
        for (int j = 0; j < L; j++) begin
            int e;
            e = (int'(rb_beat) << LL) + j;
            vu_b_data[j*32 +: 32] = rq_chan ? rec_bias[e % RQ_MAX] : b_d[j*32 +: 32];
            vu_p_mult[j*32 +: 32] = rec_mult[e % RQ_MAX];
            vu_p_shift[j*5 +: 5]  = rec_shift[e % RQ_MAX];
        end
    end
    assign vu_b_valid = rq_chan ? (st == S_RUN) : b_v;
    assign vu_p_valid = rq_chan && st == S_RUN;
    assign b_r        = !rq_chan && vu_b_ready;
    always_ff @(posedge clk) begin
        if (st == S_RUN0) rb_beat <= '0;
        else if (rq_chan && vu_b_valid && vu_b_ready)
            rb_beat <= (rb_beat == rb_nb - 16'd1) ? '0 : rb_beat + 16'd1;
    end

    // row command
    logic        row_last;
    logic signed [14:0] vis;
    logic [12:0] nvis;
    assign row_last = (ri == s_nrows - 19'd1);
    always_comb begin
        vis  = 15'(signed'({2'b0, ri[12:0]})) + 15'(t_diag1);
        nvis = !t_causal ? 13'd0 : (vis >= 15'(signed'({2'b0, t_cols}))) ? t_cols : 13'(vis);
        vu_len   = row_last ? s_len_last : s_len;
        vu_imm_b = (kind == C_SM) ? {3'b0, nvis, 8'd0, slot} :
                   (kind == C_LN || kind == C_GELU) ? {24'd0, slot} :
                   (kind == C_RQ) ? rec_bias[0] : t_imm_b;
        vu_mult  = rec_mult[0];
        vu_shamt = (kind == C_RQ) ? rec_shift[0] : t_shamt;
    end
    assign vu_cmd_valid = (st == S_RUN) && (ri != s_nrows);

    // ---------------- decode helpers ----------------
    function automatic logic [AW-1:0] ln(input logic [20:0] a);
        ln = AW'(a >> 4);
    endfunction

    logic [12:0] vec_rows, gelu_rows;
    assign vec_rows  = 13'((22'(vec_len) + 22'(CH - 1)) >> LCH);
    assign gelu_rows = 13'((22'(gelu_len) + 22'(CH - 1)) >> LCH);

    logic [3:0] dec_err;
    always_comb begin
        dec_err = '0;
        if (!f_known || !f_resv_ok || f_q != 2'd2) dec_err = E_OP;
        else if (is_vec) begin
            if (vec_vop > 5'd8 || vec_dt_in == 2'd3 || vec_dt_out == 2'd3 ||
                vec_src0_buf > 3'd4 || vec_src1_buf > 3'd4 || vec_dst_buf > 3'd4) dec_err = E_FIELD;
            else if ((vec_vop == 5'd7 || vec_vop == 5'd8) && (vec_imm[15] || vec_imm[14:5] != '0))
                dec_err = E_IMM;
        end else if (is_ln) begin
            if (32'(ln_cols) > D_MAX) dec_err = E_SIZE;
            else if (!lut_ok[2]) dec_err = E_LUT;
        end else if (is_sm) begin
            if (!sm_first || !sm_last || (sm_causal && sm_diag[12])) dec_err = E_SM;
            else if (32'(sm_cols) > CTX_MAX) dec_err = E_SIZE;
            else if (!lut_ok[0] || !lut_ok[1]) dec_err = E_LUT;
        end else if (is_kv) begin
            if (kv_dh[3:0] != '0 || 14'(kv_pos) + 14'(kv_rows) > 14'(kv_ctx)) dec_err = E_KV;
        end else if (is_lut) begin
            if (lut_table == 2'd3) dec_err = E_FIELD;
        end
    end

    // ---------------- main sequencer ----------------
    logic all_issued;
    assign all_issued = (ri == s_nrows);
    assign nc_w       = (rq_nq - c0 > 13'(RQ_MAX)) ? 13'(RQ_MAX) : rq_nq - c0;
    assign cq_pop     = (st == S_IDLE) && cq_valid;
    assign ld_req     = (st == S_LREQ) && ld_idle && !vu_busy;
    assign rd_start   = (st == S_RUN0);

    always_ff @(posedge clk) begin
        if (!rst_i) begin
            st       <= S_IDLE;
            done     <= 1'b0;
            err      <= 1'b0;
            err_code <= '0;
            for (int t = 0; t < 3; t++) lut_ok[t] <= 1'b0;
        end else begin
            done <= 1'b0;
            if (ls_bad && !err) begin err <= 1'b1; err_code <= E_LS; end
            if ((vu_err_len || vu_illegal) && !err) begin err <= 1'b1; err_code <= E_UNIT; end
            case (st)
                S_IDLE: if (cq_valid) begin
                    cw <= cq_word;
                    st <= S_DEC;
                end
                S_DEC: begin
                    // defaults
                    use_b    <= 1'b0;
                    rq_chan  <= 1'b0;
                    s_ni     <= 7'd1;
                    a_si     <= '0;
                    d_si     <= '0;
                    t_b_imm  <= 1'b0;
                    t_p_imm  <= 1'b0;
                    t_shamt  <= '0;
                    t_imm_b  <= '0;
                    t_fmt    <= 2'd0;
                    t_causal <= 1'b0;
                    t_cols   <= '0;
                    t_diag1  <= '0;
                    slot     <= '0;
                    if (dec_err != '0) begin
                        if (!err) begin err <= 1'b1; err_code <= dec_err; end
                        st <= S_FIN;
                    end else if (is_vec) begin
                        kind       <= C_VEC;
                        a_buf      <= vec_src0_buf;
                        a_base     <= ln(vec_src0_addr);
                        a_so       <= AW'((CH / 16) << vec_dt_in);
                        b_buf      <= vec_src1_buf;
                        b_base     <= ln(vec_src1_addr);
                        b_so       <= AW'((CH / 16) << vec_dt_in);
                        d_buf      <= vec_dst_buf;
                        d_base     <= ln(vec_dst_addr);
                        d_so       <= AW'((CH / 16) << vec_dt_out);
                        s_no       <= vec_rows;
                        s_nrows    <= 19'(vec_rows);
                        s_len      <= 16'(CH);
                        s_len_last <= 16'(((vec_len - 21'd1) & 21'(CH - 1)) + 21'd1);
                        a_ebl      <= vec_dt_in;
                        d_ebl      <= vec_dt_out;
                        t_fmt      <= vec_dt_out;
                        use_b      <= (vec_vop >= 5'd1 && vec_vop <= 5'd5);
                        case (vec_vop)
                            5'd6:    begin t_op <= 5'h01; t_b_imm <= 1'b1;
                                           t_imm_b <= 32'(signed'(vec_imm)); end
                            default: t_op <= vec_vop;    // COPY=PASS 0, ADD..MIN 1..5, SHRR=SRAR 7, SHL 8
                        endcase
                        t_shamt    <= (vec_vop == 5'd7 || vec_vop == 5'd8) ? vec_imm[4:0] : 5'd0;
                        st         <= S_RUN0;
                    end else if (is_rq) begin
                        kind     <= C_RQ;
                        rq_src_l <= ln(21'(rq_src));
                        rq_dst_l <= ln(21'(rq_dst));
                        rq_par_l <= ln(21'(rq_par));
                        rq_mq    <= rq_m;
                        rq_nq    <= rq_n;
                        rq_ssq   <= rq_ss;
                        rq_dsq   <= rq_ds;
                        rq_o16   <= rq_out16;
                        rq_ptq   <= rq_pt;
                        rq_chan  <= !rq_pt;
                        t_op     <= 5'h10;
                        t_fmt    <= {1'b0, rq_out16};
                        t_b_imm  <= rq_pt;
                        t_p_imm  <= rq_pt;
                        c0       <= '0;
                        st       <= S_RQC;
                    end else if (is_ln) begin
                        kind       <= C_LN;
                        a_buf      <= B_ABUF;
                        a_base     <= ln(21'(ln_src));
                        a_so       <= AW'(ln_ss);
                        d_buf      <= B_ABUF;
                        d_base     <= ln(21'(ln_dst));
                        d_so       <= AW'(ln_ds);
                        s_no       <= ln_rows;
                        s_nrows    <= 19'(ln_rows);
                        s_len      <= 16'(ln_cols);
                        s_len_last <= 16'(ln_cols);
                        a_ebl      <= {1'b0, ln_in16};
                        d_ebl      <= 2'd0;
                        t_op       <= 5'h1D;
                        l_kind     <= K_LN;
                        l_line     <= ln(21'(ln_par));
                        l_cols     <= ln_cols;
                        st         <= S_LREQ;
                    end else if (is_sm) begin
                        kind       <= C_SM;
                        a_buf      <= B_ACC;
                        a_base     <= ln(21'(sm_src));
                        a_so       <= AW'(sm_ss);
                        d_buf      <= B_ABUF;
                        d_base     <= ln(21'(sm_dst));
                        d_so       <= AW'(sm_ds);
                        s_no       <= sm_rows;
                        s_nrows    <= 19'(sm_rows);
                        s_len      <= 16'(sm_cols);
                        s_len_last <= 16'(sm_cols);
                        a_ebl      <= 2'd2;
                        d_ebl      <= 2'd0;
                        t_op       <= 5'h1E;
                        t_causal   <= sm_causal;
                        t_cols     <= sm_cols;
                        t_diag1    <= 14'(signed'(sm_diag)) + 14'sd1;
                        l_kind     <= K_CTX;
                        l_line     <= ln(21'(sm_ctx));
                        st         <= S_LREQ;
                    end else if (is_gelu) begin
                        kind       <= C_GELU;
                        a_buf      <= B_ABUF;
                        a_base     <= ln(21'(gelu_src));
                        a_so       <= AW'(CH / 16);
                        d_buf      <= B_ABUF;
                        d_base     <= ln(21'(gelu_dst));
                        d_so       <= AW'(CH / 16);
                        s_no       <= gelu_rows;
                        s_nrows    <= 19'(gelu_rows);
                        s_len      <= 16'(CH);
                        s_len_last <= 16'(((gelu_len - 21'd1) & 21'(CH - 1)) + 21'd1);
                        a_ebl      <= 2'd0;
                        d_ebl      <= 2'd0;
                        t_op       <= 5'h1B;
                        l_kind     <= K_GELU;
                        l_line     <= ln(21'(gelu_lut));
                        st         <= S_LREQ;
                    end else if (is_kv) begin
                        kind       <= C_KV;
                        a_buf      <= B_ABUF;
                        a_base     <= ln(21'(kv_src));
                        a_so       <= AW'(kv_ss);
                        a_si       <= AW'(kv_dh >> 4);
                        d_buf      <= B_KV;
                        d_base     <= ln(kv_dst_addr) + AW'((21'(kv_pos) * 21'(kv_dh)) >> 4);
                        d_so       <= AW'(kv_dh >> 4);
                        d_si       <= AW'((22'(kv_ctx) * 22'(kv_dh)) >> 4);
                        s_no       <= kv_rows;
                        s_ni       <= kv_heads;
                        s_nrows    <= 19'(kv_rows) * 19'(kv_heads);
                        s_len      <= 16'(kv_dh);
                        s_len_last <= 16'(kv_dh);
                        a_ebl      <= 2'd0;
                        d_ebl      <= 2'd0;
                        t_op       <= 5'h00;
                        st         <= S_RUN0;
                    end else begin           // LUT_LOAD
                        kind    <= C_LUT;
                        l_kind  <= K_LUT;
                        l_line  <= ln(21'(lut_src));
                        l_table <= lut_table;
                        st      <= S_LREQ;
                    end
                end
                S_RQC: begin                 // next REQUANT column chunk
                    a_buf      <= B_ACC;
                    a_base     <= rq_src_l + AW'(c0 >> 2);
                    a_so       <= AW'(rq_ssq);
                    d_buf      <= B_ABUF;
                    d_base     <= rq_dst_l + AW'((14'(c0) << rq_o16) >> 4);
                    d_so       <= AW'(rq_dsq);
                    s_no       <= rq_mq;
                    s_nrows    <= 19'(rq_mq);
                    s_len      <= 16'(nc_w);
                    s_len_last <= 16'(nc_w);
                    rb_nb      <= 16'((14'(nc_w) + 14'(L - 1)) >> LL);
                    a_ebl      <= 2'd2;
                    d_ebl      <= {1'b0, rq_o16};
                    l_kind     <= K_REC;
                    l_line     <= rq_par_l + AW'(c0);
                    l_n        <= rq_ptq ? 13'd1 : nc_w;
                    st         <= (rq_ptq && c0 != '0) ? S_RUN0 : S_LREQ;
                end
                S_LREQ: if (ld_req) st <= S_LOAD;
                S_LOAD: if (ld_done) begin
                    slot   <= ld_slot;
                    settle <= 2'd2;
                    if (kind == C_LUT) begin
                        lut_ok[l_table] <= 1'b1;
                        st <= S_FIN;
                    end else begin
                        st <= S_SET;
                    end
                end
                S_SET: begin
                    settle <= settle - 2'd1;
                    if (settle == 2'd0) st <= S_RUN0;
                end
                S_RUN0: begin
                    ri <= '0;
                    st <= S_RUN;
                end
                S_RUN: begin
                    if (vu_cmd_valid && vu_cmd_ready) ri <= ri + 19'd1;
                    if (all_issued && !ra_busy && !rb_busy && !wr_busy && !vd_valid) begin
                        if (kind == C_RQ && 14'(c0) + 14'(RQ_MAX) < 14'(rq_nq)) begin
                            c0 <= c0 + 13'(RQ_MAX);
                            st <= S_RQC;
                        end else begin
                            st <= S_FIN;
                        end
                    end
                end
                S_FIN: begin
                    done <= 1'b1;
                    st   <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end

    assign busy = (st != S_IDLE) || cq_valid;

    logic unused;
    assign unused = ^{f_opcode, a_l, b_l, vu_sticky, vu_p_ready, ld_hit, kv_ctx};
endmodule
