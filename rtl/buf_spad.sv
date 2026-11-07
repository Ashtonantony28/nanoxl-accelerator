// buf_spad.sv -- one physical scratchpad buffer (ABUF / WBUF / ACC / PBUF / KVBUF) built from
// NB line-interleaved banks (spad_bank), with NR read ports and NW write ports
// (docs/top.md §3).
//
// Line l of the buffer lives in bank l % NB, row (l mod LINES) / NB. A request names a start
// line; port p moves PL[p] consecutive lines per beat (1 for the DMA and the VEC engine,
// ceil(N/16) for the array's A/W ports, N/4 for its ACC ports), so a beat touches PL[p]
// distinct banks whenever NB >= PL[p], whatever its start line. Line addresses wrap at LINES.
//
// Arbitration, every cycle, separately for reads and writes (each bank has one read and one
// write port): ports are visited in a rotating order (the start rotates by one every cycle),
// and a request is granted when none of its banks was taken by an earlier port in that
// order; ready = grant, so a port never waits more than NR (NW) cycles behind conflicts.
// ready depends combinationally on valid (never the reverse in any requester).
//
// Reads: fixed latency 1, d_valid[p] one cycle after the grant, lines of the beat in
// d_data[p] in address order (line j in bits [128j +: 128]); no backpressure, so responses
// come in request order per port. Writes: byte strobes per line; visible to every read
// granted at a later edge (a read granted at the same edge as a write of that line returns
// the old data).
module buf_spad #(
    parameter int LINES = 1024,              // power of two
    parameter int NB    = 4,                 // banks, power of two, >= max PL
    parameter int NR    = 7,
    parameter int NW    = 3,
    parameter int PMAX  = 4,                 // widest port in lines
    parameter logic [NR*8-1:0] RPL = '0,     // lines per beat of read port p in [8p +: 8]
    parameter logic [NW*8-1:0] WPL = '0      // lines per beat of write port p
) (
    input  logic                    clk,
    input  logic                    rst_n,
    // read ports
    input  logic [NR-1:0]           r_valid,
    output logic [NR-1:0]           r_ready,
    input  logic [NR*17-1:0]        r_addr,
    output logic [NR-1:0]           d_valid,
    output logic [NR*PMAX*128-1:0]  d_data,
    // write ports
    input  logic [NW-1:0]           w_valid,
    output logic [NW-1:0]           w_ready,
    input  logic [NW*17-1:0]        w_addr,
    input  logic [NW*PMAX*128-1:0]  w_data,
    input  logic [NW*PMAX*16-1:0]   w_strb
);
    localparam int LB    = $clog2(NB);
    localparam int DEPTH = LINES / NB;
    localparam int DW    = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam int LW    = $clog2(LINES);
    localparam int RW    = (NR > 1) ? $clog2(NR) : 1;
    localparam int WW    = (NW > 1) ? $clog2(NW) : 1;

    // ---------------------------------------------------------------- rotating start
    logic [RW-1:0] rr;
    logic [WW-1:0] wr;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rr <= '0;
            wr <= '0;
        end else begin
            rr <= (32'(rr) == NR - 1) ? '0 : rr + 1'b1;
            wr <= (32'(wr) == NW - 1) ? '0 : wr + 1'b1;
        end
    end

    // ---------------------------------------------------------------- bank signals
    logic [NB-1:0]   b_re, b_we;
    logic [DW-1:0]   b_raddr [NB];
    logic [DW-1:0]   b_waddr [NB];
    logic [127:0]    b_wdata [NB];
    logic [15:0]     b_wstrb [NB];
    logic [127:0]    b_rdata [NB];

    for (genvar b = 0; b < NB; b++) begin : g_bank
        spad_bank #(.DEPTH(DEPTH)) u_bank (
            .clk(clk), .we(b_we[b]), .waddr(b_waddr[b]), .wdata(b_wdata[b]), .wstrb(b_wstrb[b]),
            .re(b_re[b]), .raddr(b_raddr[b]), .rdata(b_rdata[b]));
    end

    // line -> (bank, row)
    /* verilator lint_off UNUSEDSIGNAL */
    function automatic logic [LB-1:0] bank_of(input logic [16:0] l);
        bank_of = LB'(l);
    endfunction
    function automatic logic [DW-1:0] row_of(input logic [16:0] l);
        logic [LW-1:0] lm;
        lm = LW'(l);
        row_of = DW'(lm >> LB);
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    // ---------------------------------------------------------------- read arbitration
    always_comb begin
        logic [NB-1:0] busy;
        busy    = '0;
        r_ready = '0;
        b_re    = '0;
        for (int b = 0; b < NB; b++) b_raddr[b] = '0;
        for (int k = 0; k < NR; k++) begin
            int p;
            logic [NB-1:0] m;
            logic [16:0]   a;
            p = (int'(rr) + k) % NR;
            a = r_addr[p*17 +: 17];
            m = '0;
            for (int j = 0; j < PMAX; j++)
                if (j < int'(RPL[p*8 +: 8])) m[bank_of(a + 17'(j))] = 1'b1;
            if (r_valid[p] && ((m & busy) == '0)) begin
                r_ready[p] = 1'b1;
                busy = busy | m;
                for (int j = 0; j < PMAX; j++)
                    if (j < int'(RPL[p*8 +: 8])) begin
                        b_re[bank_of(a + 17'(j))]    = 1'b1;
                        b_raddr[bank_of(a + 17'(j))] = row_of(a + 17'(j));
                    end
            end
        end
    end

    // read response: registered grant + start bank, lines rotated back into address order
    logic [NR-1:0] dv;
    logic [LB-1:0] ds [NR];
    always_ff @(posedge clk) begin
        if (!rst_n) dv <= '0;
        else        dv <= r_valid & r_ready;
        for (int p = 0; p < NR; p++)
            if (r_valid[p] && r_ready[p]) ds[p] <= bank_of(r_addr[p*17 +: 17]);
    end
    assign d_valid = dv;
    always_comb begin
        d_data = '0;
        for (int p = 0; p < NR; p++)
            for (int j = 0; j < PMAX; j++)
                if (j < int'(RPL[p*8 +: 8]))
                    d_data[(p*PMAX + j)*128 +: 128] = b_rdata[LB'(ds[p] + LB'(j))];
    end

    // ---------------------------------------------------------------- write arbitration
    always_comb begin
        logic [NB-1:0] busy;
        busy    = '0;
        w_ready = '0;
        b_we    = '0;
        for (int b = 0; b < NB; b++) begin
            b_waddr[b] = '0;
            b_wdata[b] = '0;
            b_wstrb[b] = '0;
        end
        for (int k = 0; k < NW; k++) begin
            int p;
            logic [NB-1:0] m;
            logic [16:0]   a;
            p = (int'(wr) + k) % NW;
            a = w_addr[p*17 +: 17];
            m = '0;
            for (int j = 0; j < PMAX; j++)
                if (j < int'(WPL[p*8 +: 8])) m[bank_of(a + 17'(j))] = 1'b1;
            if (w_valid[p] && ((m & busy) == '0)) begin
                w_ready[p] = 1'b1;
                busy = busy | m;
                for (int j = 0; j < PMAX; j++)
                    if (j < int'(WPL[p*8 +: 8])) begin
                        b_we[bank_of(a + 17'(j))]    = 1'b1;
                        b_waddr[bank_of(a + 17'(j))] = row_of(a + 17'(j));
                        b_wdata[bank_of(a + 17'(j))] = w_data[(p*PMAX + j)*128 +: 128];
                        b_wstrb[bank_of(a + 17'(j))] = w_strb[(p*PMAX + j)*16 +: 16];
                    end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if ((1 << LB) != NB || (1 << LW) != LINES || LINES < NB)
            $fatal(1, "buf_spad: LINES and NB must be powers of two, LINES >= NB");
        for (int p = 0; p < NR; p++)
            if (int'(RPL[p*8 +: 8]) > NB || int'(RPL[p*8 +: 8]) > PMAX)
                $fatal(1, "buf_spad: read port %0d wider than NB/PMAX", p);
        for (int p = 0; p < NW; p++)
            if (int'(WPL[p*8 +: 8]) > NB || int'(WPL[p*8 +: 8]) > PMAX)
                $fatal(1, "buf_spad: write port %0d wider than NB/PMAX", p);
    end
`endif
endmodule
