// array_seq_rd.sv -- segment reader over a scratchpad line port (docs/array.md §7).
//
// Takes byte-granular segments {buf, addr, len <= SEGB} and returns each as SEGB bytes,
// byte 0 = the byte at addr (bytes >= len are don't-care). The line port is the one the
// DMA uses (docs/control.md §3): a read request names a buffer and a line address
// (byte address >> 4) and returns PL consecutive lines (PL*16 bytes, line i in bits
// [128i +: 128]); responses come back in request order, with any latency, without
// backpressure. A segment that crosses a PL-line window boundary costs two beats.
//
// Flow control: at most RCRED beats are requested and not yet consumed, and the
// response FIFO holds RCRED beats, so a response always has room. Throughput: one
// one-beat segment per cycle while the memory latency is below RCRED.
//
// clr (engine soft reset) empties the reader; responses still in flight for requests
// issued before clr are dropped as they arrive (counted in `drop`).
module array_seq_rd #(
    parameter int PL    = 1,
    parameter int SEGB  = 8,
    parameter int TAGW  = 1,
    parameter int RCRED = 8,                 // power of two
    parameter int LENW  = $clog2(SEGB + 1)
) (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 clr,
    // segment requests
    input  logic                 seg_valid,
    output logic                 seg_ready,
    input  logic [2:0]           seg_buf,
    input  logic [20:0]          seg_addr,
    input  logic [LENW-1:0]      seg_len,
    input  logic [TAGW-1:0]      seg_tag,
    // scratchpad line read port
    output logic                 ar_valid,
    input  logic                 ar_ready,
    output logic [2:0]           ar_buf,
    output logic [16:0]          ar_addr,
    input  logic                 r_valid,
    input  logic [PL*128-1:0]    r_data,
    // segments out
    output logic                 out_valid,
    input  logic                 out_ready,
    output logic [SEGB*8-1:0]    out_data,
    output logic [TAGW-1:0]      out_tag,
    output logic                 busy
);
    localparam int CW = $clog2(RCRED + 1);
    localparam int IW = 4 + 1 + TAGW;        // {off, two beats, tag}

    logic lrst_n;
    assign lrst_n = rst_n && !clr;

    // ------------------------------------------------------------- request side
    logic          iss2;                     // first beat of a two-beat segment issued
    logic          nb2, ar_fire, info_ready, cred_ok;
    logic [CW-1:0] outst;                    // beats requested and not yet consumed
    logic [CW-1:0] pend;                     // beats requested and not yet returned
    logic [CW-1:0] drop;                     // responses to discard (requests before clr)
    logic          r_keep;

    /* verilator lint_off WIDTHEXPAND */
    assign nb2      = ({1'b0, seg_addr[3:0]} + seg_len) > (PL * 16);
    /* verilator lint_on WIDTHEXPAND */
    assign cred_ok  = (outst + drop) < CW'(RCRED);
    assign ar_valid = seg_valid && cred_ok && (iss2 || info_ready) && !clr;
    assign ar_buf   = seg_buf;
    assign ar_addr  = seg_addr[20:4] + (iss2 ? 17'(PL) : 17'd0);
    assign ar_fire  = ar_valid && ar_ready;
    assign seg_ready = ar_fire && (iss2 || !nb2);
    assign r_keep   = r_valid && (drop == '0);

    // ------------------------------------------------------------- FIFOs
    logic            info_valid, info_pop, info_nb2;
    logic [3:0]      info_off;
    logic [TAGW-1:0] info_tag;
    logic            resp_valid, resp_pop;
    logic [PL*128-1:0] resp_data;
    /* verilator lint_off UNUSEDSIGNAL */
    logic            resp_in_ready;
    logic [$clog2(RCRED+1)-1:0] info_cnt, resp_cnt;
    /* verilator lint_on UNUSEDSIGNAL */

    fifo_sync #(.W(IW), .DEPTH(RCRED)) u_info (
        .clk(clk), .rst_n(lrst_n),
        .in_valid(ar_fire && !iss2), .in_ready(info_ready), .in_data({seg_addr[3:0], nb2, seg_tag}),
        .out_valid(info_valid), .out_ready(info_pop), .out_data({info_off, info_nb2, info_tag}),
        .count(info_cnt));

    fifo_sync #(.W(PL * 128), .DEPTH(RCRED)) u_resp (
        .clk(clk), .rst_n(lrst_n),
        .in_valid(r_keep), .in_ready(resp_in_ready), .in_data(r_data),
        .out_valid(resp_valid), .out_ready(resp_pop), .out_data(resp_data),
        .count(resp_cnt));

    // ------------------------------------------------------------- assembly
    logic              held;
    logic [PL*128-1:0] hold;
    logic [2*PL*128-1:0] wide;
    logic              take_first;

    assign take_first = info_valid && info_nb2 && !held && resp_valid;
    assign out_valid  = info_valid && resp_valid && (!info_nb2 || held);
    assign info_pop   = out_valid && out_ready;
    assign resp_pop   = info_pop || take_first;
    assign wide       = {resp_data, held ? hold : resp_data};
    /* verilator lint_off UNUSEDSIGNAL */
    logic [2*PL*128-1:0] shifted;
    /* verilator lint_on UNUSEDSIGNAL */
    assign shifted    = wide >> {info_off, 3'b000};
    assign out_data   = shifted[SEGB*8-1:0];
    assign out_tag    = info_tag;
    assign busy       = info_valid || resp_valid || (outst != '0) || iss2;

    always_ff @(posedge clk) begin
        if (!lrst_n) begin
            iss2  <= 1'b0;
            outst <= '0;
            held  <= 1'b0;
        end else begin
            if (ar_fire) iss2 <= !iss2 && nb2;
            outst <= outst + CW'(ar_fire) - CW'(resp_pop);
            if (take_first) begin
                held <= 1'b1;
                hold <= resp_data;
            end else if (info_pop) begin
                held <= 1'b0;
            end
        end
    end

    // in-flight accounting survives clr so that stale responses can be dropped
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pend <= '0;
            drop <= '0;
        end else begin
            pend <= pend + CW'(ar_fire) - CW'(r_valid);
            if (clr)          drop <= pend - CW'(r_valid);
            else if (r_valid && drop != '0) drop <= drop - 1'b1;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (lrst_n && r_keep && !resp_in_ready)
            $error("array_seq_rd: response FIFO overflow");
    end
`endif
endmodule
