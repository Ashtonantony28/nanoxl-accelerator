// scratchpad.sv -- single-clock 1R1W SRAM (BRAM/URAM-inferable) and a FIFO built on it.
//
// scratchpad: fixed-latency read port. rd_en in cycle t -> rd_data/rd_valid in cycle
// t + RD_LAT (RD_LAT = 1: registered read; 2: plus output register, maps to BRAM DOB_REG /
// URAM OREG for timing). Write is synchronous; a read of the address being written in the
// same cycle returns unspecified data (old or new) -- callers must bypass. No reset on the
// array; rd_valid is reset.
module scratchpad #(
    parameter int WIDTH  = 32,
    parameter int DEPTH  = 64,
    parameter int RD_LAT = 1
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     wr_en,
    input  logic [$clog2(DEPTH)-1:0] wr_addr,
    input  logic [WIDTH-1:0]         wr_data,
    input  logic                     rd_en,
    input  logic [$clog2(DEPTH)-1:0] rd_addr,
    output logic [WIDTH-1:0]         rd_data,
    output logic                     rd_valid
);
    logic [WIDTH-1:0] mem [0:DEPTH-1];
    logic [WIDTH-1:0] q1;
    logic             v1;

    always_ff @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end
    always_ff @(posedge clk) begin
        if (rd_en) q1 <= mem[rd_addr];
    end
    always_ff @(posedge clk) begin
        if (!rst_n) v1 <= 1'b0;
        else        v1 <= rd_en;
    end

    if (RD_LAT == 1) begin : g_lat1
        assign rd_data  = q1;
        assign rd_valid = v1;
    end else begin : g_lat2
        logic [WIDTH-1:0] q2;
        logic             v2;
        always_ff @(posedge clk) begin
            if (v1) q2 <= q1;
        end
        always_ff @(posedge clk) begin
            if (!rst_n) v2 <= 1'b0;
            else        v2 <= v1;
        end
        assign rd_data  = q2;
        assign rd_valid = v2;
    end
endmodule

// spad_fifo: FIFO on a scratchpad (RD_LAT=1) with a 2-entry first-word-fall-through output
// queue. Full throughput (one push and one pop per cycle). A push is legal while
// full == 0 (count < DEPTH); pushing into a full SRAM sets the sticky sim flag `overflow`.
// push -> out_valid latency is 3 cycles. Callers that push without looking at `full`
// (array_ctrl) must hold at most DEPTH entries outstanding (credits).
module spad_fifo #(
    parameter int W     = 32,
    parameter int DEPTH = 16
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       push,
    input  logic [W-1:0]               din,
    output logic                       full,
    output logic                       out_valid,
    input  logic                       out_ready,
    output logic [W-1:0]               out_data,
    output logic [$clog2(DEPTH+3)-1:0] count,
    output logic                       overflow
);
    localparam int AB = $clog2(DEPTH);
    localparam int CB = $clog2(DEPTH + 3);

    logic [AB-1:0]  wr_ptr, rd_ptr;
    logic [CB-1:0]  mem_cnt;
    logic [1:0]     oq_cnt;
    logic           rd_inflight;
    logic [W-1:0]   q0, q1, rdata;
    logic           pop, rd_issue;
    /* verilator lint_off UNUSEDSIGNAL */
    logic           rvalid;  // == rd_inflight
    /* verilator lint_on UNUSEDSIGNAL */
    logic [1:0]     cnt_after_pop;

    assign pop           = out_valid && out_ready;
    assign out_valid     = (oq_cnt != 0);
    assign out_data      = q0;
    assign cnt_after_pop = oq_cnt - {1'b0, pop};
    // a read may be issued if the queue will have room for it when it lands
    assign rd_issue      = (mem_cnt != 0) && ({1'b0, cnt_after_pop} + {2'b0, rd_inflight} < 3'd2);
    assign count         = mem_cnt + CB'(oq_cnt) + CB'(rd_inflight);
    assign full          = (count >= CB'(DEPTH));

    scratchpad #(.WIDTH(W), .DEPTH(DEPTH), .RD_LAT(1)) u_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en(push), .wr_addr(wr_ptr), .wr_data(din),
        .rd_en(rd_issue), .rd_addr(rd_ptr), .rd_data(rdata), .rd_valid(rvalid));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr      <= '0;
            rd_ptr      <= '0;
            mem_cnt     <= '0;
            oq_cnt      <= '0;
            rd_inflight <= 1'b0;
        end else begin
            if (push) wr_ptr <= (wr_ptr == AB'(DEPTH - 1)) ? '0 : wr_ptr + 1'b1;
            if (rd_issue) rd_ptr <= (rd_ptr == AB'(DEPTH - 1)) ? '0 : rd_ptr + 1'b1;
            mem_cnt     <= mem_cnt + CB'(push) - CB'(rd_issue);
            rd_inflight <= rd_issue;
            oq_cnt      <= cnt_after_pop + {1'b0, rd_inflight};
            // output queue: pop shifts q1 -> q0, then the landing read appends
            if (pop) q0 <= q1;
            if (rd_inflight) begin
                if (cnt_after_pop == 2'd0) q0 <= rdata;
                else                       q1 <= rdata;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (!rst_n) overflow <= 1'b0;
        else if (push && mem_cnt >= CB'(DEPTH)) overflow <= 1'b1;
    end
`else
    assign overflow = 1'b0;
`endif
endmodule
