// array_seq.sv -- ARRAY engine: ISA GEMM / GEMM_GS words -> array_ctrl tile commands
// (docs/array.md §7, docs/control.md §2, docs/isa.md §3-4, ADR-004).
//
// Engine port (control.md §2): cmd_valid/cmd_ready + the 128-bit ISA word; `done` pulses
// once per accepted command, in order, >= 1 cycle after acceptance, when the command's
// last ACC write has been accepted by the scratchpad (= visible). At most OUTS commands
// are accepted and not yet done. Opcodes other than GEMM / GEMM_GS (ATTN_TILE is
// reserved) set the sticky `err` and still produce their done pulse.
//
// Execution of one command C[M,N] (+)= A[M,K] @ W[K,N]:
//   for m0 in 0..M step M_DEPTH:            (row chunk = array_ctrl accumulator rows)
//     for n0 in 0..N step NA:               (output column tile, NA = array size)
//       for k0 in 0..K step KT:             (KT = NA; GEMM_GS: min(NA, G) so a tile
//                                            never straddles a scale group)
//         one array_ctrl tile: m = mc, k = kt, n = nt, abase 0, acc = (k0 > 0),
//         emit = last k-tile, int4, gs
//   and each emitted chain (mc rows x nt columns) is written to ACC at
//   c_addr + m*c_stride*16 + 4*n0, adding the old ACC row when `accumulate` = 1.
// Operand layouts are the compiler's (docs/isa.md §3): A rows at a_addr + m*a_stride*16;
// W dense [K, N] int8 (row = N bytes), packed int4 [K, N/2] (C1b), GEMM_GS block = packed
// int4 then gm uint8 [ceil(K/G), N] at w_addr + rup(K*N/2, 16), or w_trans [N, K] int8.
// A row segment, a weight row segment and the scale row are read through byte-granular
// segment readers (array_seq_rd) from line ports; w_trans tiles go through a
// double-buffered NA x NA byte transposer. Throughput = array_ctrl's max(m, k) per tile
// (k + 1 for GEMM_GS tiles, the scale beat) while every segment is one port beat.
//
// ACC ordering: the old rows of an accumulating command are read only after every
// earlier command is done (all its writes accepted), so a GEMM that accumulates into
// the tile written by the previous GEMM (K-split panels) sees its result. Within one
// command each ACC row/column tile is read once before it is written.
module array_seq #(
    parameter int N        = 8,                 // array size NA (power of two, >= 4)
    parameter int M_DEPTH  = 64,                // array_ctrl accumulator rows = row chunk
    parameter int OUTS     = 4,                 // commands accepted and not yet done
    parameter int RCRED    = 8,                 // beats in flight per read port (power of 2)
    parameter int APL      = (N + 15) / 16,     // lines per A-port beat
    parameter int WPL      = (N + 15) / 16      // lines per W-port beat
) (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 clr,           // engine soft reset (ctrl_top eng_clr)
    // engine command port
    input  logic                 cmd_valid,
    output logic                 cmd_ready,
    input  logic [127:0]         cmd_word,
    output logic                 done,
    output logic                 err,
    output logic                 idle,
    // A read port (ABUF)
    output logic                 a_arvalid,
    input  logic                 a_arready,
    output logic [2:0]           a_arbuf,
    output logic [16:0]          a_araddr,
    input  logic                 a_rvalid,
    input  logic [APL*128-1:0]   a_rdata,
    // W read port (WBUF / KVBUF)
    output logic                 w_arvalid,
    input  logic                 w_arready,
    output logic [2:0]           w_arbuf,
    output logic [16:0]          w_araddr,
    input  logic                 w_rvalid,
    input  logic [WPL*128-1:0]   w_rdata,
    // ACC read port
    output logic                 cr_arvalid,
    input  logic                 cr_arready,
    output logic [2:0]           cr_arbuf,
    output logic [16:0]          cr_araddr,
    input  logic                 cr_rvalid,
    input  logic [N*32-1:0]      cr_rdata,     // ACC port beat = one INT32 row = N/4 lines
    // ACC write port
    output logic                 cw_valid,
    input  logic                 cw_ready,
    output logic [2:0]           cw_buf,
    output logic [16:0]          cw_addr,
    output logic [N*32-1:0]      cw_data,
    output logic [N*4-1:0]       cw_strb,
    // simulation checkers of array_ctrl
    output logic                 hazard_seen,
    output logic                 overflow_seen
);
    localparam int RB   = $clog2(N);
    localparam int KW   = $clog2(N + 1);
    localparam int MW   = $clog2(M_DEPTH + 1);
    localparam int ML2  = $clog2(M_DEPTH);
    localparam int CSW  = $clog2(OUTS) + 1;
    localparam int CPL  = N / 4;               // lines per ACC-port beat
    localparam int CLW  = $clog2(4 * N + 1);
    localparam logic [2:0] B_ABUF = 3'd0, B_WBUF = 3'd1, B_ACC = 3'd2, B_KVBUF = 3'd4;

    logic lrst_n;
    assign lrst_n = rst_n && !clr;

    // =================================================================== decode
    /* verilator lint_off PINMISSING */
    logic        is_gemm, is_gemm_gs;
    logic        g_int4, g_wtrans, g_acc, g_wbuf, s_acc;
    logic [19:0] g_a, g_c, s_a, s_c;
    logic [20:0] g_w, s_w;
    logic [12:0] g_m, g_n, g_k, s_m, s_n, s_k;
    logic [9:0]  g_as, g_cs, s_as, s_cs;
    logic [2:0]  s_g2;
    cmd_fields u_dec (
        .w(cmd_word), .is_gemm(is_gemm), .is_gemm_gs(is_gemm_gs),
        .gemm_int4(g_int4), .gemm_w_trans(g_wtrans), .gemm_accumulate(g_acc),
        .gemm_a_addr(g_a), .gemm_w_buf(g_wbuf), .gemm_w_addr(g_w), .gemm_c_addr(g_c),
        .gemm_m(g_m), .gemm_n(g_n), .gemm_k(g_k), .gemm_a_stride(g_as), .gemm_c_stride(g_cs),
        .gemm_gs_accumulate(s_acc), .gemm_gs_a_addr(s_a), .gemm_gs_w_addr(s_w),
        .gemm_gs_c_addr(s_c), .gemm_gs_m(s_m), .gemm_gs_n(s_n), .gemm_gs_k(s_k),
        .gemm_gs_a_stride(s_as), .gemm_gs_c_stride(s_cs), .gemm_gs_glog2(s_g2));
    /* verilator lint_on PINMISSING */

    // =================================================================== command / done
    logic [CSW-1:0] acc_cnt, done_cnt, outs;
    logic           cmd_fire, done_now;
    localparam logic [1:0] TG_IDLE = 2'd0, TG_SETUP = 2'd1, TG_RUN = 2'd2, TG_NOP = 2'd3;
    logic [1:0]     tg_st;

    assign outs      = acc_cnt - done_cnt;
    assign cmd_ready = (tg_st == TG_IDLE) && (outs < CSW'(OUTS));
    assign cmd_fire  = cmd_valid && cmd_ready;
    assign idle      = (tg_st == TG_IDLE) && (outs == '0);

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            acc_cnt  <= '0;
            done_cnt <= '0;
            done     <= 1'b0;
            err      <= 1'b0;
        end else begin
            acc_cnt  <= acc_cnt + CSW'(cmd_fire);
            done_cnt <= done_cnt + CSW'(done_now);
            done     <= done_now;
            if (cmd_fire && !(is_gemm || is_gemm_gs)) err <= 1'b1;
        end
    end

    // =================================================================== tile generator
    // command registers
    logic           r_int4, r_gs, r_trans, r_kv, r_acc;
    logic [20:0]    r_a, r_w, r_c, r_soff;
    logic [12:0]    r_m, r_n, r_k;
    logic [9:0]     r_as, r_cs;
    logic [2:0]     r_g2;
    logic [CSW-1:0] r_cseq;
    // loop state
    logic [12:0]    m0, n0, k0;
    logic [20:0]    a_m, c_m, w_k, s_kp, wt_n;

    // per-command constants
    logic [3:0]     ktl2;                   // log2 KT
    logic [12:0]    kt_full, gmask, rowb;
    logic [20:0]    wstep;
    logic [12:0]    n_half, n0_half;
    assign n_half  = {1'b0, r_n[12:1]};
    assign n0_half = {1'b0, n0[12:1]};
    always_comb begin
        ktl2 = 4'(RB);
        if (r_gs && (32'(r_g2) + 3 < RB)) ktl2 = 4'(r_g2) + 4'd3;
        kt_full = 13'd1 << ktl2;
        gmask   = (13'd8 << r_g2) - 13'd1;
        rowb    = (r_int4 || r_gs) ? n_half : r_n;
        wstep   = 21'(rowb) << ktl2;
    end

    // current tile
    logic [12:0]    k_rem, n_rem, m_rem;
    logic           lastk, lastn, lastm;
    logic [KW-1:0]  t_kt, t_nt;
    logic [MW-1:0]  t_mc;
    assign k_rem = r_k - k0;
    assign n_rem = r_n - n0;
    assign m_rem = r_m - m0;
    assign lastk = (k_rem <= kt_full);
    assign lastn = (n_rem <= 13'(N));
    assign lastm = (m_rem <= 13'(M_DEPTH));
    assign t_kt  = lastk ? KW'(k_rem) : KW'(kt_full);
    assign t_nt  = lastn ? KW'(n_rem) : KW'(N);
    assign t_mc  = lastm ? MW'(m_rem) : MW'(M_DEPTH);

    // tile FIFOs
    localparam int ACW = MW + 2 * KW + 4;                    // array_ctrl command
    localparam int WTW = 1 + 1 + 1 + 21 + 13 + KW + KW + 21 + KW + KW; // W tile
    localparam int ATW = 21 + 14 + MW + KW;                  // A tile
    localparam int JBW = 17 + 10 + MW + KW + 1 + 1 + CSW + 1; // write-back job

    logic           ac_in_ready, wt_in_ready, at_in_ready, wb_in_ready, pf_in_ready;
    logic           tile_fire, nop_push;
    logic [ACW-1:0] ac_in;
    logic [WTW-1:0] wt_in;
    logic [ATW-1:0] at_in;
    logic [JBW-1:0] jb_in;

    assign tile_fire = (tg_st == TG_RUN) && ac_in_ready && wt_in_ready && at_in_ready &&
                       (!lastk || (wb_in_ready && (!r_acc || pf_in_ready)));
    assign nop_push  = (tg_st == TG_NOP) && wb_in_ready;

    // W tile fields
    logic [20:0]   wt_base, wt_sc;
    logic [12:0]   wt_stride;
    logic [KW-1:0] wt_cnt, wt_len;
    always_comb begin
        if (r_trans) begin
            wt_base   = wt_n + 21'(k0);
            wt_stride = r_k;
            wt_cnt    = t_nt;
            wt_len    = t_kt;
        end else begin
            wt_base   = w_k + ((r_int4 || r_gs) ? 21'(n0_half) : 21'(n0));
            wt_stride = rowb;
            wt_cnt    = t_kt;
            wt_len    = (r_int4 || r_gs) ? KW'((t_nt + 1'b1) >> 1) : t_nt;
        end
        wt_sc = s_kp + 21'(n0);
    end
    /* verilator lint_off UNUSEDSIGNAL */
    logic [20:0] c_byte;                    // line aligned (alignment rule, N >= 4)
    logic [41:0] soff_prod;
    /* verilator lint_on UNUSEDSIGNAL */
    assign c_byte = c_m + 21'({n0, 2'b00});

    assign ac_in = {t_mc, t_kt, t_nt, (k0 != '0), lastk, (r_int4 || r_gs), r_gs};
    assign wt_in = {r_kv, r_trans, r_gs, wt_base, wt_stride, wt_cnt, wt_len, wt_sc, t_nt, t_kt};
    assign at_in = {a_m + 21'(k0), {r_as, 4'b0000}, t_mc, t_kt};
    assign jb_in = nop_push ? {17'd0, 10'd0, MW'(0), KW'(0), 1'b0, 1'b1, r_cseq, 1'b1}
                            : {c_byte[20:4], r_cs, t_mc, t_nt, r_acc, (lastn && lastm), r_cseq, 1'b0};

    assign soff_prod = 42'(r_k) * 42'(n_half);

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            tg_st <= TG_IDLE;
            r_int4 <= 1'b0; r_gs <= 1'b0; r_trans <= 1'b0; r_kv <= 1'b0; r_acc <= 1'b0;
            r_a <= '0; r_w <= '0; r_c <= '0; r_soff <= '0; r_m <= '0; r_n <= '0; r_k <= '0;
            r_as <= '0; r_cs <= '0; r_g2 <= '0; r_cseq <= '0;
            m0 <= '0; n0 <= '0; k0 <= '0; a_m <= '0; c_m <= '0; w_k <= '0; s_kp <= '0; wt_n <= '0;
        end else begin
            case (tg_st)
                TG_IDLE: if (cmd_fire) begin
                    r_cseq <= acc_cnt;
                    if (is_gemm) begin
                        r_int4 <= g_int4; r_gs <= 1'b0; r_trans <= g_wtrans; r_kv <= g_wbuf;
                        r_acc <= g_acc; r_a <= {1'b0, g_a}; r_w <= g_w; r_c <= {1'b0, g_c};
                        r_m <= g_m; r_n <= g_n; r_k <= g_k; r_as <= g_as; r_cs <= g_cs; r_g2 <= '0;
                        tg_st <= TG_SETUP;
                    end else if (is_gemm_gs) begin
                        r_int4 <= 1'b1; r_gs <= 1'b1; r_trans <= 1'b0; r_kv <= 1'b0;
                        r_acc <= s_acc; r_a <= {1'b0, s_a}; r_w <= s_w; r_c <= {1'b0, s_c};
                        r_m <= s_m; r_n <= s_n; r_k <= s_k; r_as <= s_as; r_cs <= s_cs; r_g2 <= s_g2;
                        tg_st <= TG_SETUP;
                    end else begin
                        tg_st <= TG_NOP;
                    end
                end
                TG_SETUP: begin
                    // GEMM_GS scale block offset rup(K * N/2, 16) (isa.gs_layout)
                    r_soff <= (21'(soff_prod) + 21'd15) & ~21'd15;
                    s_kp   <= r_w + ((21'(soff_prod) + 21'd15) & ~21'd15);
                    m0 <= '0; n0 <= '0; k0 <= '0;
                    a_m <= r_a; c_m <= r_c; w_k <= r_w; wt_n <= r_w;
                    tg_st <= TG_RUN;
                end
                TG_RUN: if (tile_fire) begin
                    if (!lastk) begin
                        k0  <= k0 + kt_full;
                        w_k <= w_k + wstep;
                        if (((k0 + kt_full) & gmask) == '0) s_kp <= s_kp + 21'(r_n);
                    end else begin
                        k0   <= '0;
                        w_k  <= r_w;
                        s_kp <= r_w + r_soff;
                        if (!lastn) begin
                            n0   <= n0 + 13'(N);
                            wt_n <= wt_n + (21'(r_k) << RB);
                        end else begin
                            n0   <= '0;
                            wt_n <= r_w;
                            if (!lastm) begin
                                m0  <= m0 + 13'(M_DEPTH);
                                a_m <= a_m + (21'(r_as) << (4 + ML2));
                                c_m <= c_m + (21'(r_cs) << (4 + ML2));
                            end else begin
                                tg_st <= TG_IDLE;
                            end
                        end
                    end
                end
                TG_NOP: if (nop_push) tg_st <= TG_IDLE;
                default: tg_st <= TG_IDLE;
            endcase
        end
    end

    // FIFOs between the tile generator and the stream generators
    logic           ac_v, wt_v, at_v, wb_v, pf_v;
    logic           ac_pop, wt_pop, at_pop, wb_pop, pf_pop;
    logic [ACW-1:0] ac_q;
    logic [WTW-1:0] wt_q;
    logic [ATW-1:0] at_q;
    logic [JBW-1:0] wb_q, pf_q;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [2:0]     c_ac, c_wt, c_at, c_wb, c_pf;
    /* verilator lint_on UNUSEDSIGNAL */

    fifo_sync #(.W(ACW), .DEPTH(4)) u_fac (.clk(clk), .rst_n(lrst_n),
        .in_valid(tile_fire), .in_ready(ac_in_ready), .in_data(ac_in),
        .out_valid(ac_v), .out_ready(ac_pop), .out_data(ac_q), .count(c_ac));
    fifo_sync #(.W(WTW), .DEPTH(4)) u_fwt (.clk(clk), .rst_n(lrst_n),
        .in_valid(tile_fire), .in_ready(wt_in_ready), .in_data(wt_in),
        .out_valid(wt_v), .out_ready(wt_pop), .out_data(wt_q), .count(c_wt));
    fifo_sync #(.W(ATW), .DEPTH(4)) u_fat (.clk(clk), .rst_n(lrst_n),
        .in_valid(tile_fire), .in_ready(at_in_ready), .in_data(at_in),
        .out_valid(at_v), .out_ready(at_pop), .out_data(at_q), .count(c_at));
    fifo_sync #(.W(JBW), .DEPTH(4)) u_fwb (.clk(clk), .rst_n(lrst_n),
        .in_valid((tile_fire && lastk) || nop_push), .in_ready(wb_in_ready), .in_data(jb_in),
        .out_valid(wb_v), .out_ready(wb_pop), .out_data(wb_q), .count(c_wb));
    fifo_sync #(.W(JBW), .DEPTH(4)) u_fpf (.clk(clk), .rst_n(lrst_n),
        .in_valid(tile_fire && lastk && r_acc), .in_ready(pf_in_ready), .in_data(jb_in),
        .out_valid(pf_v), .out_ready(pf_pop), .out_data(pf_q), .count(c_pf));

    // =================================================================== array_ctrl
    logic           ac_cmd_ready, ac_w_valid, ac_w_ready, ac_a_valid, ac_a_ready;
    logic           ac_out_valid, ac_out_ready;
    logic [N*8-1:0] ac_w_data, ac_a_data;
    logic [N*32-1:0] ac_out_data;
    /* verilator lint_off UNUSEDSIGNAL */
    logic           ac_out_last, ac_idle;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [MW-1:0]  q_m;
    logic [KW-1:0]  q_k, q_n;
    logic           q_acc, q_emit, q_int4, q_gs;
    assign {q_m, q_k, q_n, q_acc, q_emit, q_int4, q_gs} = ac_q;
    assign ac_pop = ac_v && ac_cmd_ready;

    array_ctrl #(.N(N), .M_DEPTH(M_DEPTH)) u_ctrl (
        .clk(clk), .rst_n(lrst_n),
        .cmd_valid(ac_v), .cmd_ready(ac_cmd_ready), .cmd_m(q_m), .cmd_k(q_k), .cmd_n(q_n),
        .cmd_abase('0), .cmd_acc(q_acc), .cmd_emit(q_emit), .cmd_int4(q_int4), .cmd_gs(q_gs),
        .w_valid(ac_w_valid), .w_ready(ac_w_ready), .w_data(ac_w_data),
        .a_valid(ac_a_valid), .a_ready(ac_a_ready), .a_data(ac_a_data),
        .out_valid(ac_out_valid), .out_ready(ac_out_ready), .out_data(ac_out_data),
        .out_last(ac_out_last),
        .idle(ac_idle), .hazard_seen(hazard_seen), .overflow_seen(overflow_seen));

    // =================================================================== A stream
    logic           as_v, as_last, as_fire, as_load, as_ready;
    logic [20:0]    as_addr;
    logic [13:0]    as_stride;
    logic [MW-1:0]  as_left;
    logic [KW-1:0]  as_len;
    logic [20:0]    at_base;
    logic [13:0]    at_stride;
    logic [MW-1:0]  at_cnt;
    logic [KW-1:0]  at_len;
    assign {at_base, at_stride, at_cnt, at_len} = at_q;
    assign as_fire = as_v && as_ready;
    assign as_last = as_fire && (as_left == MW'(1));
    assign as_load = (!as_v || as_last) && at_v;
    assign at_pop  = as_load;

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            as_v <= 1'b0; as_addr <= '0; as_stride <= '0; as_left <= '0; as_len <= '0;
        end else if (as_load) begin
            as_v <= 1'b1; as_addr <= at_base; as_stride <= at_stride; as_left <= at_cnt;
            as_len <= at_len;
        end else if (as_last) begin
            as_v <= 1'b0;
        end else if (as_fire) begin
            as_addr <= as_addr + 21'(as_stride);
            as_left <= as_left - 1'b1;
        end
    end

    /* verilator lint_off UNUSEDSIGNAL */
    logic a_tag, a_busy, w_busy, c_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    array_seq_rd #(.PL(APL), .SEGB(N), .TAGW(1), .RCRED(RCRED)) u_ard (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .seg_valid(as_v), .seg_ready(as_ready), .seg_buf(B_ABUF), .seg_addr(as_addr),
        .seg_len(as_len), .seg_tag(1'b0),
        .ar_valid(a_arvalid), .ar_ready(a_arready), .ar_buf(a_arbuf), .ar_addr(a_araddr),
        .r_valid(a_rvalid), .r_data(a_rdata),
        .out_valid(ac_a_valid), .out_ready(ac_a_ready), .out_data(ac_a_data), .out_tag(a_tag),
        .busy(a_busy));

    // =================================================================== W stream
    // tile: {kv, trans, gs, base, stride, cnt, len, sc_addr, nt, kt}
    logic           ws_v, ws_sc, ws_fire, ws_last, ws_load, ws_ready, ws_kv, ws_trans;
    logic [20:0]    ws_addr, ws_scaddr;
    logic [12:0]    ws_stride;
    logic [KW-1:0]  ws_left, ws_len, ws_nt, ws_kt;
    logic           wq_kv, wq_trans, wq_gs;
    logic [20:0]    wq_base, wq_sc;
    logic [12:0]    wq_stride;
    logic [KW-1:0]  wq_cnt, wq_len, wq_nt, wq_kt;
    assign {wq_kv, wq_trans, wq_gs, wq_base, wq_stride, wq_cnt, wq_len, wq_sc, wq_nt, wq_kt} = wt_q;
    assign ws_fire = ws_v && ws_ready;
    assign ws_last = ws_fire && !ws_sc && (ws_left == KW'(1));
    assign ws_load = (!ws_v || ws_last) && wt_v;
    assign wt_pop  = ws_load;

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            ws_v <= 1'b0; ws_sc <= 1'b0; ws_kv <= 1'b0; ws_trans <= 1'b0; ws_addr <= '0;
            ws_scaddr <= '0; ws_stride <= '0; ws_left <= '0; ws_len <= '0; ws_nt <= '0; ws_kt <= '0;
        end else if (ws_load) begin
            ws_v <= 1'b1; ws_sc <= wq_gs; ws_kv <= wq_kv; ws_trans <= wq_trans;
            ws_addr <= wq_base; ws_scaddr <= wq_sc; ws_stride <= wq_stride; ws_left <= wq_cnt;
            ws_len <= wq_len; ws_nt <= wq_nt; ws_kt <= wq_kt;
        end else if (ws_last) begin
            ws_v <= 1'b0;
        end else if (ws_fire) begin
            if (ws_sc) begin
                ws_sc <= 1'b0;
            end else begin
                ws_addr <= ws_addr + 21'(ws_stride);
                ws_left <= ws_left - 1'b1;
            end
        end
    end

    // segment tag: {trans, last column of a trans tile, kt}
    localparam int WTG = 2 + KW;
    logic [WTG-1:0] wr_tag;
    logic           wr_v, wr_ready, wr_trans, wr_lastc;
    logic [N*8-1:0] wr_data;
    logic [KW-1:0]  wr_kt;
    assign {wr_trans, wr_lastc, wr_kt} = wr_tag;

    array_seq_rd #(.PL(WPL), .SEGB(N), .TAGW(WTG), .RCRED(RCRED)) u_wrd (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .seg_valid(ws_v), .seg_ready(ws_ready), .seg_buf(ws_kv ? B_KVBUF : B_WBUF),
        .seg_addr(ws_sc ? ws_scaddr : ws_addr), .seg_len(ws_sc ? ws_nt : ws_len),
        .seg_tag({ws_trans, ws_trans && ws_left == KW'(1), ws_kt}),
        .ar_valid(w_arvalid), .ar_ready(w_arready), .ar_buf(w_arbuf), .ar_addr(w_araddr),
        .r_valid(w_rvalid), .r_data(w_rdata),
        .out_valid(wr_v), .out_ready(wr_ready), .out_data(wr_data), .out_tag(wr_tag),
        .busy(w_busy));

    // w_trans transposer: segments of a trans tile are W columns (stored [N, K] rows);
    // collect nt of them into a bank, then emit kt K-rows. Two banks, so collection of
    // the next tile overlaps emission. Non-trans segments bypass, but only while the
    // transposer is empty, which keeps the weight stream in order.
    logic [N*8-1:0] trc [0:2*N-1];          // {bank, column} -> column bytes (K order)
    logic           tw_bank, tr_bank;
    logic [RB-1:0]  tw_col, tr_row;
    logic [1:0]     tfull;
    logic [KW-1:0]  tkt [0:1];
    logic           tr_empty, tr_in, tr_out, byp;
    logic [N*8-1:0] trow;

    assign tr_empty = (tfull == 2'b00) && (tw_col == '0);
    assign tr_in    = wr_v && wr_trans && !tfull[tw_bank];
    assign byp      = wr_v && !wr_trans && tr_empty;
    assign ac_w_valid = tfull[tr_bank] || byp;
    assign ac_w_data  = tfull[tr_bank] ? trow : wr_data;
    assign tr_out   = tfull[tr_bank] && ac_w_ready;
    assign wr_ready = tr_in || (byp && ac_w_ready);

    always_comb begin
        for (int c = 0; c < N; c++)
            trow[c*8 +: 8] = trc[{tr_bank, RB'(c)}][{tr_row, 3'b000} +: 8];
    end

    always_ff @(posedge clk) begin
        if (tr_in) trc[{tw_bank, tw_col}] <= wr_data;
    end
    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            tw_bank <= 1'b0; tr_bank <= 1'b0; tw_col <= '0; tr_row <= '0; tfull <= '0;
            tkt[0] <= '0; tkt[1] <= '0;
        end else begin
            if (tr_in) begin
                if (wr_lastc) begin
                    tfull[tw_bank] <= 1'b1;
                    tkt[tw_bank]   <= wr_kt;
                    tw_bank        <= ~tw_bank;
                    tw_col         <= '0;
                end else begin
                    tw_col <= tw_col + 1'b1;
                end
            end
            if (tr_out) begin
                if ({1'b0, tr_row} == KW'(tkt[tr_bank] - 1'b1)) begin
                    tfull[tr_bank] <= 1'b0;
                    tr_bank        <= ~tr_bank;
                    tr_row         <= '0;
                end else begin
                    tr_row <= tr_row + 1'b1;
                end
            end
        end
    end

    // =================================================================== ACC prefetch
    // jobs: {cline, cs, mc, nt, acc, lastcmd, cseq, nop}
    logic [16:0]    pq_line;
    logic [9:0]     pq_cs;
    logic [MW-1:0]  pq_mc;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [KW-1:0]  pq_nt;
    logic           pq_acc, pq_lastcmd, pq_nop;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [CSW-1:0] pq_cseq;
    assign {pq_line, pq_cs, pq_mc, pq_nt, pq_acc, pq_lastcmd, pq_cseq, pq_nop} = pf_q;

    logic           ps_v, ps_ready, ps_fire, ps_last, ps_load;
    logic [16:0]    ps_line;
    logic [9:0]     ps_cs;
    logic [MW-1:0]  ps_left;
    assign ps_fire = ps_v && ps_ready;
    assign ps_last = ps_fire && (ps_left == MW'(1));
    // gate: every earlier command is done (its ACC writes accepted)
    assign ps_load = (!ps_v || ps_last) && pf_v && (pq_cseq == done_cnt);
    assign pf_pop  = ps_load;

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            ps_v <= 1'b0; ps_line <= '0; ps_cs <= '0; ps_left <= '0;
        end else if (ps_load) begin
            ps_v <= 1'b1; ps_line <= pq_line; ps_cs <= pq_cs; ps_left <= pq_mc;
        end else if (ps_last) begin
            ps_v <= 1'b0;
        end else if (ps_fire) begin
            ps_line <= ps_line + 17'(ps_cs);
            ps_left <= ps_left - 1'b1;
        end
    end

    logic             old_v, old_ready;
    logic [N*32-1:0]  old_data;
    /* verilator lint_off UNUSEDSIGNAL */
    logic             old_tag;
    /* verilator lint_on UNUSEDSIGNAL */
    array_seq_rd #(.PL(CPL), .SEGB(4 * N), .TAGW(1), .RCRED(RCRED)) u_crd (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .seg_valid(ps_v), .seg_ready(ps_ready), .seg_buf(B_ACC), .seg_addr({ps_line, 4'b0000}),
        .seg_len(CLW'(4 * N)), .seg_tag(1'b0),
        .ar_valid(cr_arvalid), .ar_ready(cr_arready), .ar_buf(cr_arbuf), .ar_addr(cr_araddr),
        .r_valid(cr_rvalid), .r_data(cr_rdata),
        .out_valid(old_v), .out_ready(old_ready), .out_data(old_data), .out_tag(old_tag),
        .busy(c_busy));

    // =================================================================== write-back
    logic [16:0]    jq_line;
    logic [9:0]     jq_cs;
    logic [MW-1:0]  jq_mc;
    logic [KW-1:0]  jq_nt;
    logic           jq_acc, jq_lastcmd, jq_nop;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [CSW-1:0] jq_cseq;
    /* verilator lint_on UNUSEDSIGNAL */
    assign {jq_line, jq_cs, jq_mc, jq_nt, jq_acc, jq_lastcmd, jq_cseq, jq_nop} = wb_q;

    logic           can_load, row_fire, row_last, nop_fire, w_done_q;
    logic [MW-1:0]  wb_row;
    logic [16:0]    wb_line, row_line;
    logic [N*32-1:0] sum;
    logic [N*4-1:0] strb;

    assign can_load = !cw_valid || cw_ready;
    assign row_fire = wb_v && !jq_nop && ac_out_valid && (!jq_acc || old_v) && can_load;
    assign row_last = row_fire && (wb_row == jq_mc - 1'b1);
    assign nop_fire = wb_v && jq_nop && !cw_valid;
    assign wb_pop   = row_last || nop_fire;
    assign ac_out_ready = row_fire;
    assign old_ready    = row_fire && jq_acc;
    assign row_line = (wb_row == '0) ? jq_line : wb_line;
    assign done_now = (cw_valid && cw_ready && w_done_q) || nop_fire;

    for (genvar c = 0; c < N; c++) begin : g_wb
        assign sum[c*32 +: 32] = ac_out_data[c*32 +: 32] + (jq_acc ? old_data[c*32 +: 32] : 32'd0);
        assign strb[c*4 +: 4]  = (c < 32'(jq_nt)) ? 4'hF : 4'h0;
    end

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            cw_valid <= 1'b0; cw_addr <= '0; cw_data <= '0; cw_strb <= '0; w_done_q <= 1'b0;
            wb_row <= '0; wb_line <= '0;
        end else begin
            if (row_fire) begin
                cw_valid <= 1'b1;
                cw_addr  <= row_line;
                cw_data  <= sum;
                cw_strb  <= strb;
                w_done_q <= row_last && jq_lastcmd;
                wb_line  <= row_line + 17'(jq_cs);
                wb_row   <= row_last ? '0 : wb_row + 1'b1;
            end else if (cw_ready) begin
                cw_valid <= 1'b0;
            end
        end
    end
    assign cw_buf = B_ACC;

`ifndef SYNTHESIS
    // a write-back row never arrives without its job (sim check)
    always @(posedge clk) begin
        if (lrst_n && ac_out_valid && !wb_v)
            $error("array_seq: array output without a write-back job");
    end
`endif
endmodule
