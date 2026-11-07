// cmd_proc.sv -- command processor: decode, dispatch into the DMA / ARRAY / VEC queues,
// semaphores, END / IRQ barriers, error halt and single-step (docs/control.md).
//
// One word per cycle is taken from the fetcher. Engine commands, WAITs and SIGNALs go to their
// queue in stream order (in-order dispatch: a full queue blocks the dispatcher). IRQ and END are
// barriers: they execute once all three queues are idle (nothing queued, nothing outstanding).
//   IRQ(code) -> irq_pulse with irq_code, then continue.
//   END       -> end_pulse (done interrupt), then continue with the next pending word if
//                software has already advanced tail past END; otherwise fetching simply
//                stops because the ring is empty.
// Errors halt dispatch and queue issue (outstanding engine commands finish): a decode error, an
// RRESP error on a fetched word, a DMA AXI error (dma_err) or an engine error (eng_err). The
// first error is captured with the ring index and the word. Only a (soft) reset clears it.
// Single-step: with step_mode = 1, each step_pulse lets exactly one word be dispatched
// (or one barrier execute).
module cmd_proc #(
    parameter int PW     = 24,
    parameter int QDEPTH = 8,
    parameter int MAXOUT = 4
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            clr,
    // word stream from the fetcher
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [127:0]    in_word,
    input  logic            in_rerr,
    input  logic [PW-1:0]   in_index,      // ring index of in_word (= head)
    // control
    input  logic            step_mode,
    input  logic            step_pulse,
    input  logic            cfg_err,       // ring configuration error (from the fetcher)
    input  logic            dma_err,       // DMA engine error (AXI response or bad command)
    input  logic [1:0]      eng_err,       // ARRAY / VEC engine error inputs
    // engines: index 0 DMA, 1 ARRAY, 2 VEC
    output logic [2:0]      eng_valid,
    input  logic [2:0]      eng_ready,
    output logic [3*128-1:0] eng_word,
    input  logic [2:0]      eng_done,
    // status
    output logic            halted,
    output logic [3:0]      err_code,
    output logic [PW-1:0]   err_index,
    output logic [127:0]    err_word,
    output logic            irq_pulse,
    output logic [15:0]     irq_code,
    output logic            end_pulse,
    output logic            all_idle,       // queues idle and no word waiting at a barrier
    output logic [2:0]      q_idle,
    output logic [2:0]      eng_busy,
    output logic [2:0]      retire_inc,     // commands completed this cycle (0..5)
    output logic [PW-1:0]   last_index,     // ring index of the last dispatched word
    output logic            q_overflow,
    input  logic [4:0]      sem_dbg_sel,
    output logic [7:0]      sem_dbg_val
);
    localparam logic [3:0] ERR_FETCH = 4'd4, ERR_DMA = 4'd5, ERR_RING = 4'd6, ERR_ENGINE = 4'd7;
    localparam logic [2:0] K_ENGINE = 3'd0, K_WAIT = 3'd1, K_SIGNAL = 3'd2, K_IRQ = 3'd3, K_END = 3'd4;

    // ---------------- decode ----------------
    logic [3:0]  d_err;
    logic [1:0]  d_route;
    logic [2:0]  d_kind;
    logic [4:0]  d_sem;
    logic [7:0]  d_wcnt;
    logic [15:0] d_irq;
    cmd_decode u_dec (.w(in_word), .err(d_err), .route(d_route), .kind(d_kind), .sem(d_sem),
                      .wait_count(d_wcnt), .irq_code(d_irq));

    // ---------------- queues + semaphores ----------------
    logic [2:0]  q_in_valid, q_in_ready, q_retire, q_ovf;
    logic [1:0]  q_kind;
    logic [127:0] q_word;
    logic [2:0]  sig_req, sig_gnt, wait_req, wait_gnt;
    logic [14:0] sig_sem, wait_sem;
    logic [23:0] wait_cnt;
    logic        hold;

    assign q_kind = (d_kind == K_WAIT) ? 2'd1 : (d_kind == K_SIGNAL) ? 2'd2 : 2'd0;
    assign q_word = (d_kind == K_ENGINE) ? in_word : {115'b0, d_wcnt, d_sem};

    genvar gq;
    for (gq = 0; gq < 3; gq++) begin : g_q
        cmd_queue #(.QDEPTH(QDEPTH), .MAXOUT(MAXOUT)) u_q (
            .clk(clk), .rst_n(rst_n), .clr(clr), .hold(hold),
            .in_valid(q_in_valid[gq]), .in_ready(q_in_ready[gq]), .in_kind(q_kind), .in_word(q_word),
            .eng_valid(eng_valid[gq]), .eng_ready(eng_ready[gq]), .eng_word(eng_word[gq*128 +: 128]),
            .eng_done(eng_done[gq]),
            .sig_req(sig_req[gq]), .sig_sem(sig_sem[gq*5 +: 5]), .sig_gnt(sig_gnt[gq]),
            .wait_req(wait_req[gq]), .wait_sem(wait_sem[gq*5 +: 5]), .wait_cnt(wait_cnt[gq*8 +: 8]),
            .wait_gnt(wait_gnt[gq]),
            .idle(q_idle[gq]), .eng_busy(eng_busy[gq]), .retire(q_retire[gq]), .overflow(q_ovf[gq]));
    end

    sem_unit #(.NQ(3), .NSEM(32), .CW(8)) u_sem (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .sig_req(sig_req), .sig_sem(sig_sem), .sig_gnt(sig_gnt),
        .wait_req(wait_req), .wait_sem(wait_sem), .wait_cnt(wait_cnt), .wait_gnt(wait_gnt),
        .dbg_sel(sem_dbg_sel), .dbg_val(sem_dbg_val));

    // ---------------- dispatch ----------------
    logic step_credit, step_ok, barrier_ok, is_bar, d_bad, take;
    assign step_ok    = !step_mode || step_credit;
    assign is_bar     = (d_kind == K_IRQ) || (d_kind == K_END);
    assign barrier_ok = &q_idle;
    assign d_bad      = in_rerr || (d_err != 4'd0);
    assign hold       = halted;

    always_comb begin
        q_in_valid = '0;
        take       = 1'b0;
        if (in_valid && !halted && step_ok && !d_bad) begin
            if (is_bar) begin
                take = barrier_ok;
            end else begin
                q_in_valid[d_route] = 1'b1;
                take = q_in_ready[d_route];
            end
        end
    end
    assign in_ready  = take;
    assign irq_pulse = take && (d_kind == K_IRQ);
    assign end_pulse = take && (d_kind == K_END);
    assign all_idle  = barrier_ok;

    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            halted      <= 1'b0;
            err_code    <= 4'd0;
            err_index   <= '0;
            err_word    <= '0;
            irq_code    <= '0;
            step_credit <= 1'b0;
            last_index  <= '0;
        end else begin
            if (step_pulse) step_credit <= 1'b1;
            else if (take)  step_credit <= 1'b0;
            if (take) last_index <= in_index;
            if (irq_pulse) irq_code <= d_irq;
            if (!halted) begin
                if (in_valid && step_ok && d_bad) begin
                    halted    <= 1'b1;
                    err_code  <= in_rerr ? ERR_FETCH : d_err;
                    err_index <= in_index;
                    err_word  <= in_word;
                end else if (dma_err) begin
                    halted   <= 1'b1;
                    err_code <= ERR_DMA;
                end else if (|eng_err) begin
                    halted   <= 1'b1;
                    err_code <= ERR_ENGINE;
                end else if (cfg_err) begin
                    halted   <= 1'b1;
                    err_code <= ERR_RING;
                end
            end
        end
    end

    // retired-command count: queue completions plus executed barriers
    assign retire_inc = 3'(q_retire[0]) + 3'(q_retire[1]) + 3'(q_retire[2]) + 3'(irq_pulse || end_pulse);
    assign q_overflow = |q_ovf;
endmodule
