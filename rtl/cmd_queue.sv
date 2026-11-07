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
    parameter int PSIG   = 4      // pending (carried) SIGNALs, power of two >= 2
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
    input  logic [127:0]    in_word,
    // engine ports
    output logic [NCH-1:0]  eng_valid,
    input  logic [NCH-1:0]  eng_ready,
    output logic [127:0]    eng_word,
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
    logic [127:0]  h_word;
    logic [OB-1:0] outst [NCH];
    logic [NCH-1:0] dn;                // eng_done of an outstanding command
    /* verilator lint_off UNUSEDSIGNAL */
    logic [$clog2(QDEPTH+1)-1:0] qcount;
    /* verilator lint_on UNUSEDSIGNAL */

    fifo_sync #(.W(132), .DEPTH(QDEPTH)) u_q (
        .clk(clk), .rst_n(rst_n && !clr),
        .in_valid(in_valid), .in_ready(in_ready), .in_data({in_fence, in_ch, in_kind, in_word}),
        .out_valid(h_valid), .out_ready(pop), .out_data({h_fence, h_ch, h_kind, h_word}), .count(qcount));

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
    logic [OB-1:0] p_left [PSIG][NCH];
    logic [PB-1:0] p_rd, p_wr;
    logic [PC-1:0] p_cnt;
    logic          p_head_ok, h_sig, byp, p_push, p_fire;
    always_comb begin
        p_head_ok = 1'b1;
        for (int c = 0; c < NCH; c++) p_head_ok = p_head_ok && p_left[p_rd][c] == '0;
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
                    if (dn[c] && p_left[e][c] != '0) p_left[e][c] <= p_left[e][c] - 1'b1;
            if (p_push) begin
                p_sem[p_wr] <= h_word[4:0];
                for (int c = 0; c < NCH; c++) p_left[p_wr][c] <= outst[c] - OB'(dn[c]);
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
    // the pending counts never exceed what is outstanding (they track a suffix of it)
    always_ff @(posedge clk) begin
        if (rst_n && !clr && p_cnt != '0)
            for (int c = 0; c < NCH; c++)
                if (p_left[p_rd][c] > outst[c]) $error("cmd_queue: pending SIGNAL count above outstanding");
    end
`endif
endmodule
