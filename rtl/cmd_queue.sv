// cmd_queue.sv -- one in-order command queue (DMA, ARRAY or VEC), docs/isa.md §2 and docs/control.md.
//
// Entries are {kind, word}: kind 0 = engine command (the ISA word), 1 = WAIT, 2 = SIGNAL (word =
// {count[7:0], sem[4:0]} in the low bits, filled in by the dispatcher). The head entry:
//   engine  issued on the engine port (eng_valid/eng_ready, the 128-bit command word); up to
//           MAXOUT commands may be outstanding at the engine, which completes them in order
//           with one eng_done pulse each.
//   WAIT    requests the semaphore unit and blocks the queue until granted.
//   SIGNAL  waits until every earlier command of this queue has completed (outstanding = 0),
//           then requests the semaphore unit (granted unless the counter is full).
// `hold` (halt after an error) stops issue and semaphore requests; outstanding commands finish.
module cmd_queue #(
    parameter int QDEPTH = 8,
    parameter int MAXOUT = 4      // engine commands accepted but not yet done (>= 1)
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          clr,
    input  logic          hold,
    // from the dispatcher
    input  logic          in_valid,
    output logic          in_ready,
    input  logic [1:0]    in_kind,
    input  logic [127:0]  in_word,
    // engine port
    output logic          eng_valid,
    input  logic          eng_ready,
    output logic [127:0]  eng_word,
    input  logic          eng_done,
    // semaphore unit
    output logic          sig_req,
    output logic [4:0]    sig_sem,
    input  logic          sig_gnt,
    output logic          wait_req,
    output logic [4:0]    wait_sem,
    output logic [7:0]    wait_cnt,
    input  logic          wait_gnt,
    // status
    output logic          idle,       // nothing queued and nothing outstanding
    output logic          eng_busy,   // at least one command outstanding at the engine
    output logic          retire,     // a command of this queue completed this cycle
    output logic          overflow    // sticky: eng_done without an outstanding command
);
    localparam int OB = $clog2(MAXOUT + 1);

    logic          h_valid, pop;
    logic [1:0]    h_kind;
    logic [127:0]  h_word;
    logic [OB-1:0] outst;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [$clog2(QDEPTH+1)-1:0] qcount;
    /* verilator lint_on UNUSEDSIGNAL */

    fifo_sync #(.W(130), .DEPTH(QDEPTH)) u_q (
        .clk(clk), .rst_n(rst_n && !clr),
        .in_valid(in_valid), .in_ready(in_ready), .in_data({in_kind, in_word}),
        .out_valid(h_valid), .out_ready(pop), .out_data({h_kind, h_word}), .count(qcount));

    logic can_issue;
    assign can_issue = h_valid && !hold && (32'(outst) < MAXOUT);
    assign eng_valid = can_issue && (h_kind == 2'd0);
    assign eng_word  = h_word;
    assign sig_req   = h_valid && !hold && (h_kind == 2'd2) && (outst == '0);
    // WAIT/SIGNAL entries use an internal format set by the dispatcher: [4:0] sem, [12:5] count
    assign sig_sem   = h_word[4:0];
    assign wait_req  = h_valid && !hold && (h_kind == 2'd1);
    assign wait_sem  = h_word[4:0];
    assign wait_cnt  = h_word[12:5];
    assign pop       = (eng_valid && eng_ready) || (sig_req && sig_gnt) || (wait_req && wait_gnt);

    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            outst    <= '0;
            overflow <= 1'b0;
        end else begin
            outst <= outst + OB'(eng_valid && eng_ready) - OB'(eng_done && outst != '0);
            if (eng_done && outst == '0) overflow <= 1'b1;
        end
    end

    assign idle     = !h_valid && (outst == '0);
    assign eng_busy = (outst != '0);
    assign retire   = (eng_done && outst != '0) || (sig_req && sig_gnt) || (wait_req && wait_gnt);
endmodule
