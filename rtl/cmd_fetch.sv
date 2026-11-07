// cmd_fetch.sv -- command fetcher: AXI4 read master (read channels only) over the DDR ring.
//
// Fetches words in [fptr, tail) in INCR bursts of 16-byte beats (ARSIZE = 4), one burst in
// flight at a time, into a FDEPTH-word prefetch FIFO (space is reserved before the burst is
// issued, so RREADY is always 1). Each FIFO entry carries the RRESP error bit of its beat.
// `en` = 0 stops new bursts (halt, soft-reset drain); a burst already issued always completes.
// `out_*` is the word stream to the dispatcher; popping it advances head.
module cmd_fetch #(
    parameter int AW     = 32,
    parameter int IDW    = 4,
    parameter int PW     = 24,
    parameter int FDEPTH = 16,
    parameter int FBURST = 8
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            clr,          // soft reset (only while axi_idle)
    input  logic            en,
    input  logic [AW-1:0]   ring_base,
    input  logic [PW-1:0]   ring_size,
    input  logic [PW-1:0]   ring_tail,
    output logic [PW-1:0]   ring_head,
    output logic            cfg_ok,
    output logic            axi_idle,
    // AXI4 read address / data
    output logic [IDW-1:0]  m_axi_arid,
    output logic [AW-1:0]   m_axi_araddr,
    output logic [7:0]      m_axi_arlen,
    output logic [2:0]      m_axi_arsize,
    output logic [1:0]      m_axi_arburst,
    output logic            m_axi_arlock,
    output logic [3:0]      m_axi_arcache,
    output logic [2:0]      m_axi_arprot,
    output logic            m_axi_arvalid,
    input  logic            m_axi_arready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [IDW-1:0]  m_axi_rid,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [127:0]    m_axi_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]      m_axi_rresp,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic            m_axi_rlast,
    input  logic            m_axi_rvalid,
    output logic            m_axi_rready,
    // word stream to the dispatcher
    output logic            out_valid,
    input  logic            out_ready,
    output logic [127:0]    out_word,
    output logic            out_rerr
);
    logic [8:0]    burst_len;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [AW-1:0] burst_addr;
    logic [PW-1:0] fptr;
    /* verilator lint_on UNUSEDSIGNAL */
    logic          issue, consume;
    logic          busy;           // a burst is issued or being issued
    /* verilator lint_off UNUSEDSIGNAL */
    logic [$clog2(FDEPTH+1)-1:0] occ, fcount;
    logic          f_in_ready;
    /* verilator lint_on UNUSEDSIGNAL */

    cmd_ring_ptr #(.PW(PW), .AW(AW), .FDEPTH(FDEPTH), .FBURST(FBURST)) u_ring (
        .clk(clk), .rst_n(rst_n), .clr(clr), .base(ring_base), .size(ring_size), .tail(ring_tail),
        .issue(issue), .consume(consume), .cfg_ok(cfg_ok), .head(ring_head), .fptr(fptr),
        .burst_len(burst_len), .burst_addr(burst_addr), .occ(occ));

    // AR: the address phase registers the burst; `issue` books it in the ring pointers
    assign issue = !busy && en && (burst_len != 0);
    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            busy          <= 1'b0;
            m_axi_arvalid <= 1'b0;
        end else begin
            if (issue) begin
                busy          <= 1'b1;
                m_axi_arvalid <= 1'b1;
                m_axi_araddr  <= {burst_addr[AW-1:4], 4'b0};
                m_axi_arlen   <= 8'(burst_len - 9'd1);
            end
            if (m_axi_arvalid && m_axi_arready) m_axi_arvalid <= 1'b0;
            if (m_axi_rvalid && m_axi_rready && m_axi_rlast) busy <= 1'b0;
        end
    end
    assign m_axi_arid    = '0;
    assign m_axi_arsize  = 3'd4;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_rready  = 1'b1;    // credit reserved at issue
    assign axi_idle      = !busy;

    fifo_sync #(.W(129), .DEPTH(FDEPTH)) u_pf (
        .clk(clk), .rst_n(rst_n && !clr),
        .in_valid(m_axi_rvalid), .in_ready(f_in_ready), .in_data({m_axi_rresp[1], m_axi_rdata}),
        .out_valid(out_valid), .out_ready(out_ready), .out_data({out_rerr, out_word}),
        .count(fcount));
    assign consume = out_valid && out_ready;

`ifndef SYNTHESIS
    // the credit scheme guarantees the FIFO never refuses a beat
    always_ff @(posedge clk) begin
        if (rst_n && m_axi_rvalid && !f_in_ready)
            $error("cmd_fetch: prefetch FIFO overflow");
    end
`endif
endmodule
