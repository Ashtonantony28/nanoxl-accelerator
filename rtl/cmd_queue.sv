// cmd_queue.sv -- one in-order command queue (DMA, ARRAY or VEC), docs/isa.md §2, docs/control.md §1
// and ADR-005.
//
// Entries are {fence, ch, kind, word}: kind 0 = engine command (the ISA word), 1 = WAIT, 2 = SIGNAL
// (word = {count[7:0], sem[4:0]} in the low bits, filled in by the dispatcher). `ch` picks one of
// NCH engine channels (the DMA queue has two DMA engines, ARRAY / VEC one); `fence` marks a command
// that must run alone across all channels (DMA_STORE). The head entry:
//   engine  issued on channel ch (eng_valid[ch]/eng_ready[ch], the 128-bit word on eng_word): up to
//           MAXOUT commands may be outstanding per channel; each channel completes its commands in
//           order with one eng_done[ch] pulse each. A fence command issues only when nothing is
//           outstanding on any channel, and nothing issues while a fence command is outstanding.
//   WAIT    requests the semaphore unit and blocks the queue until granted.
//   SIGNAL  fires (requests the semaphore unit) once every earlier engine command of this queue,
//           on every channel, has completed. It does not block the queue ("carried on done"):
//           it leaves the queue into a pending-signal buffer of PSIG entries that records, per
//           channel, how many earlier commands are still outstanding; the counts go down with
//           eng_done and the oldest pending SIGNAL fires when all its counts are 0. Pending
//           SIGNALs fire in stream order. When the buffer is empty and nothing is outstanding, the
//           head SIGNAL requests the semaphore directly (same timing as a blocking SIGNAL). A full
//           buffer, or a full semaphore counter, stalls the SIGNAL (never lost, never wrapped).
// `hold` (halt after an error / soft-reset drain) stops issue and semaphore requests; outstanding
// commands finish.
module cmd_queue #(
    parameter int QDEPTH = 8,
    parameter int MAXOUT = 4,     // engine commands accepted but not yet done, per channel (>= 1)
    parameter int NCH    = 1,     // engine channels (1 or 2)
    parameter int PSIG   = 4,     // pending (carried) SIGNALs, power of two >= 2
    parameter int WW     = 128    // word width (128; the formal model uses 13: WAIT/SIGNAL fields only)
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            clr,
    input  logic            hold,
    // from the dispatcher
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [1:0]      in_kind,
    input  logic            in_ch,
    input  logic            in_fence,
    input  logic [WW-1:0]   in_word,
    // engine ports
    output logic [NCH-1:0]  eng_valid,
    input  logic [NCH-1:0]  eng_ready,
    output logic [WW-1:0]   eng_word,
    input  logic [NCH-1:0]  eng_done,
    // semaphore unit
    output logic            sig_req,
    output logic [4:0]      sig_sem,
    input  logic            sig_gnt,
    output logic            wait_req,
    output logic [4:0]      wait_sem,
    output logic [7:0]      wait_cnt,
    input  logic            wait_gnt,
    // status
    output logic            idle,       // nothing queued, nothing outstanding, no pending SIGNAL
    output logic            eng_busy,   // at least one command outstanding at an engine channel
    output logic [2:0]      retire,     // commands of this queue completed this cycle (0..NCH+2)
    output logic            overflow    // sticky: eng_done without an outstanding command
);
    localparam int OB = $clog2(MAXOUT + 1);
    localparam int PB = $clog2(PSIG);
    localparam int PC = $clog2(PSIG + 1);

    logic          h_valid, pop, h_fence, h_ch;
    logic [1:0]    h_kind;
    logic [WW-1:0] h_word;
    logic [OB-1:0] outst [NCH];
    logic [NCH-1:0] dn;                // eng_done of an outstanding command
    // command FIFO (first-word-fall-through, QDEPTH a power of two; same behaviour as fifo_sync, kept
    // inline so the formal proof of this module can see the entries)
    localparam int QB = (QDEPTH > 1) ? $clog2(QDEPTH) : 1;
    localparam int QC = $clog2(QDEPTH + 1);
    logic [WW+3:0] q_mem [QDEPTH];
    logic [QB:0]   q_wr, q_rd;
    logic [QC-1:0] qcount;
    assign qcount   = QC'(q_wr - q_rd);
    assign in_ready = qcount != QC'(QDEPTH);
    assign h_valid  = qcount != '0;
    assign {h_fence, h_ch, h_kind, h_word} = q_mem[q_rd[QB-1:0]];
    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            q_wr <= '0;
            q_rd <= '0;
        end else begin
            if (in_valid && in_ready) begin
                q_mem[q_wr[QB-1:0]] <= {in_fence, in_ch, in_kind, in_word};
                q_wr <= q_wr + 1'b1;
            end
            if (pop) q_rd <= q_rd + 1'b1;
        end
    end

    // ---------------- engine issue ----------------
    logic all_zero, fence_out;
    logic [NCH-1:0] room;
    always_comb begin
        all_zero = 1'b1;
        for (int c = 0; c < NCH; c++) begin
            dn[c]    = eng_done[c] && outst[c] != '0;
            room[c]  = 32'(outst[c]) < MAXOUT;
            all_zero = all_zero && outst[c] == '0;
        end
    end

    logic h_c;                             // the head's channel index
    assign h_c = (NCH > 1) ? h_ch : 1'b0;
    logic can_issue;
    assign can_issue = h_valid && !hold && (h_kind == 2'd0) && room[h_c] &&
                       (h_fence ? all_zero : !fence_out);
    always_comb begin
        eng_valid = '0;
        eng_valid[h_c] = can_issue;
    end
    assign eng_word = h_word;
    logic issued;
    assign issued = can_issue && eng_ready[h_c];

    // ---------------- carried SIGNALs ----------------
    logic [4:0]    p_sem [PSIG];
    logic [OB-1:0] p_left [PSIG*NCH];    // entry e, channel c at [e*NCH + c]
    logic [PB-1:0] p_rd, p_wr;
    logic [PC-1:0] p_cnt;
    logic          p_head_ok, h_sig, byp, p_push, p_fire;
    always_comb begin
        p_head_ok = 1'b1;
        for (int c = 0; c < NCH; c++) p_head_ok = p_head_ok && p_left[32'(p_rd) * NCH + c] == '0;
    end
    assign h_sig   = h_valid && !hold && (h_kind == 2'd2);
    // direct request: nothing pending and nothing outstanding
    assign byp     = h_sig && p_cnt == '0 && all_zero;
    assign sig_req = (p_cnt != '0) ? (!hold && p_head_ok) : byp;
    assign sig_sem = (p_cnt != '0) ? p_sem[p_rd] : h_word[4:0];
    assign p_fire  = (p_cnt != '0) && sig_req && sig_gnt;
    // the head SIGNAL moves into the buffer unless it fired directly this cycle
    assign p_push  = h_sig && !(byp && sig_gnt) && 32'(p_cnt) < PSIG;

    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            p_rd  <= '0;
            p_wr  <= '0;
            p_cnt <= '0;
        end else begin
            for (int e = 0; e < PSIG; e++)
                for (int c = 0; c < NCH; c++)
                    if (dn[c] && p_left[e * NCH + c] != '0) p_left[e * NCH + c] <= p_left[e * NCH + c] - 1'b1;
            if (p_push) begin
                p_sem[p_wr] <= h_word[4:0];
                for (int c = 0; c < NCH; c++) p_left[32'(p_wr) * NCH + c] <= outst[c] - OB'(dn[c]);
                p_wr <= p_wr + 1'b1;
            end
            if (p_fire) p_rd <= p_rd + 1'b1;
            p_cnt <= p_cnt + PC'(p_push) - PC'(p_fire);
        end
    end

    // ---------------- WAIT, pop ----------------
    assign wait_req  = h_valid && !hold && (h_kind == 2'd1);
    // WAIT/SIGNAL entries use an internal format set by the dispatcher: [4:0] sem, [12:5] count
    assign wait_sem  = h_word[4:0];
    assign wait_cnt  = h_word[12:5];
    assign pop       = issued || p_push || (byp && sig_gnt) || (wait_req && wait_gnt);

    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            for (int c = 0; c < NCH; c++) outst[c] <= '0;
            fence_out <= 1'b0;
            overflow  <= 1'b0;
        end else begin
            for (int c = 0; c < NCH; c++) begin
                outst[c] <= outst[c] + OB'(issued && h_c == 1'(c)) - OB'(dn[c]);
                if (eng_done[c] && outst[c] == '0) overflow <= 1'b1;
            end
            // a fence command runs alone: it is the only command outstanding until its done
            if (issued && h_fence) fence_out <= 1'b1;
            else if (|dn)          fence_out <= 1'b0;
        end
    end

    always_comb begin
        eng_busy = !all_zero;
        retire   = 3'(sig_req && sig_gnt) + 3'(wait_req && wait_gnt);
        for (int c = 0; c < NCH; c++) retire = retire + 3'(dn[c]);
    end
    assign idle = !h_valid && all_zero && p_cnt == '0;

`ifndef SYNTHESIS
`ifndef FORMAL
    // the pending counts never exceed what is outstanding (they track a suffix of it)
    always_ff @(posedge clk) begin
        if (rst_n && !clr && p_cnt != '0)
            for (int c = 0; c < NCH; c++)
                if (p_left[32'(p_rd) * NCH + c] > outst[c]) $error("cmd_queue: pending SIGNAL count above outstanding");
    end
`endif
`endif
`ifdef FORMAL
    // ---- properties (formal/queue.ys, NCH = 2): carried SIGNALs and the DMA_STORE fence ----
    logic f_init = 1'b1;
    always_ff @(posedge clk) f_init <= 1'b0;
    always_comb if (f_init) assume (!rst_n);
    // environment: the dispatcher and the engines keep their protocols
    always_comb begin
        assume (in_kind != 2'd3);
        if (in_kind != 2'd0) assume (!in_fence && !in_ch);
        for (int c = 0; c < NCH; c++) if (eng_done[c]) assume (outst[c] != '0);
        if (sig_gnt)  assume (sig_req);
        if (wait_gnt) assume (wait_req);
    end

    logic f_push, f_fire;
    assign f_push = in_valid && in_ready;
    assign f_fire = sig_req && sig_gnt;

    // FIFO contents: entry k (0 = head) and counts of engine commands per channel / SIGNALs, in all
    // entries (q_*) and in the first g_pos entries (a_*, those ahead of the tagged SIGNAL)
    localparam int GB = QC;
    logic [1:0]    e_kind [QDEPTH];
    logic          e_ch   [QDEPTH];
    logic [GB-1:0] q_eng [NCH];
    logic [GB-1:0] a_eng [NCH];
    logic [GB-1:0] q_sig, a_sig;
    logic [GB-1:0] g_pos;
    always_comb begin
        q_sig = '0;
        a_sig = '0;
        for (int c = 0; c < NCH; c++) begin q_eng[c] = '0; a_eng[c] = '0; end
        for (int k = 0; k < QDEPTH; k++) begin
            logic [WW+3:0] e;
            e = q_mem[QB'(32'(q_rd) + k)];
            e_kind[k] = e[WW+1:WW];
            e_ch[k]   = e[WW+2];
            if (k < 32'(qcount)) begin
                if (e_kind[k] == 2'd2) q_sig = q_sig + 1'b1;
                if (e_kind[k] == 2'd2 && k < 32'(g_pos)) a_sig = a_sig + 1'b1;
                for (int c = 0; c < NCH; c++)
                    if (e_kind[k] == 2'd0 && 32'(e_ch[k] && NCH > 1) == c) begin
                        q_eng[c] = q_eng[c] + 1'b1;
                        if (k < 32'(g_pos)) a_eng[c] = a_eng[c] + 1'b1;
                    end
            end
        end
    end

    // one tagged SIGNAL ((* anyseq *) picks which push): at its push, every engine command ahead of
    // it (queued or outstanding) on each channel is recorded in g_r; each done of that channel retires
    // one (per-channel in-order completion). g_ahead = unfired SIGNALs ahead of it; g_inq = it is
    // still in the FIFO at position g_pos.
    (* anyseq *) logic f_pick;
    logic          g_tag, g_inq;
    logic [OB+GB:0] g_r [NCH];
    logic [PC+GB:0] g_ahead;
    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            g_tag <= 1'b0;
            g_inq <= 1'b0;
        end else if (!g_tag) begin
            if (f_push && in_kind == 2'd2 && f_pick) begin
                g_tag   <= 1'b1;
                g_inq   <= 1'b1;
                g_pos   <= GB'(qcount) - GB'(pop);
                for (int c = 0; c < NCH; c++)
                    g_r[c] <= (OB+GB+1)'(q_eng[c]) + (OB+GB+1)'(outst[c]) - (OB+GB+1)'(dn[c]);
                g_ahead <= (PC+GB+1)'(q_sig) + (PC+GB+1)'(p_cnt) - (PC+GB+1)'(f_fire);
            end
        end else begin
            for (int c = 0; c < NCH; c++) if (dn[c] && g_r[c] != '0) g_r[c] <= g_r[c] - 1'b1;
            if (g_inq && pop) begin
                if (g_pos == '0) g_inq <= 1'b0;
                else             g_pos <= g_pos - 1'b1;
            end
            if (f_fire) begin
                if (g_ahead == '0) g_tag   <= 1'b0;
                else               g_ahead <= g_ahead - 1'b1;
            end
        end
    end

    // fence: a DMA_STORE runs alone
    logic g_fo, g_fc;
    always_ff @(posedge clk) begin
        if (!rst_n || clr)                    g_fo <= 1'b0;
        else if (issued && h_fence) begin     g_fo <= 1'b1; g_fc <= h_c; end
        else if (g_fo && dn[g_fc])            g_fo <= 1'b0;
    end

    always_comb begin
        if (!f_init && rst_n && !clr) begin
            // SIGNAL ordering: the tagged SIGNAL fires only after every earlier command of every
            // channel completed
            if (g_tag && f_fire && g_ahead == '0)
                for (int c = 0; c < NCH; c++) assert (g_r[c] == '0);
            // fence: issued only with nothing outstanding; nothing issues while it is outstanding
            if (issued && h_fence) assert (all_zero);
            assert (g_fo == fence_out);
            if (g_fo) begin
                assert (!issued);
                for (int c = 0; c < NCH; c++) assert (32'(outst[c]) == (32'(c) == 32'(g_fc) ? 1 : 0));
            end
            // bookkeeping invariants
            assert (32'(qcount) <= QDEPTH);
            for (int c = 0; c < NCH; c++) assert (32'(outst[c]) <= MAXOUT);
            assert (32'(p_cnt) <= PSIG);
            assert (PB'(p_wr - p_rd) == PB'(p_cnt));
            for (int k = 0; k < QDEPTH; k++)
                if (k < 32'(qcount)) begin
                    assert (e_kind[k] != 2'd3);
                    if (e_kind[k] != 2'd0) assert (!e_ch[k]);
                end
            for (int k = 0; k < PSIG; k++)
                if (k < 32'(p_cnt))
                    for (int c = 0; c < NCH; c++) begin
                        assert (p_left[32'(PB'(32'(p_rd) + k)) * NCH + c] <= outst[c]);
                        if (k > 0)
                            assert (p_left[32'(PB'(32'(p_rd) + k - 1)) * NCH + c] <= p_left[32'(PB'(32'(p_rd) + k)) * NCH + c]);
                    end
            if (g_tag && g_inq) begin           // tagged SIGNAL still queued at position g_pos
                assert (32'(g_pos) < 32'(qcount));
                assert (e_kind[g_pos] == 2'd2);
                assert (32'(g_ahead) == 32'(p_cnt) + 32'(a_sig));
                for (int c = 0; c < NCH; c++) assert (32'(g_r[c]) == 32'(outst[c]) + 32'(a_eng[c]));
            end
            if (g_tag && !g_inq) begin          // ... or pending at index g_ahead
                assert (32'(g_ahead) < 32'(p_cnt));
                for (int c = 0; c < NCH; c++)
                    assert (32'(g_r[c]) == 32'(p_left[32'(PB'(32'(p_rd) + 32'(g_ahead))) * NCH + c]));
            end
        end
    end
`endif
endmodule
