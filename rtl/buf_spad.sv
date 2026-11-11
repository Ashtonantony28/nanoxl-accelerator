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
// Reads: fixed latency RD_LAT = 1 + OREG, d_valid[p] RD_LAT cycles after the grant, lines of
// the beat in d_data[p] in address order (line j in bits [128j +: 128]); no backpressure, so
// responses come in request order per port. Writes: byte strobes per line; visible to every
// read granted at a later edge (a read granted at the same edge as a write of that line
// returns the old data). OREG = 1 adds the bank output register (URAM OREG / BRAM DOB_REG)
// and does not change the visibility rule: the array is read at the grant edge either way.
//
// Implementation (docs/fpga.md C2): everything indexed by a run-time value is a fixed mux.
//   * bank mask of a port: the constant mask of its PL lines, rotated by its start bank;
//   * pairwise conflicts c[p][q] = |(mask_p & mask_q), computed once;
//   * the grant vector for each of the NR (NW) rotations is a static priority chain over
//     c (port order fixed per rotation), and the rotation counter selects one of them;
//   * per bank, the owning port is one-hot (sel = grant & mask), so bank address, data and
//     strobes are AND-OR muxes; a write port's lines are barrel-rotated by its start bank
//     onto the banks, a read port's bank outputs barrel-rotated back into address order.
module buf_spad #(
    parameter int LINES = 1024,              // power of two
    parameter int NB    = 4,                 // banks, power of two, >= max PL
    parameter int NR    = 7,
    parameter int NW    = 3,
    parameter int PMAX  = 4,                 // widest port in lines
    parameter logic [NR*8-1:0] RPL = '0,     // lines per beat of read port p in [8p +: 8]
    parameter logic [NW*8-1:0] WPL = '0,     // lines per beat of write port p
    parameter int OREG  = 0,                 // 1: bank output register, read latency 2
    parameter int RAM_STYLE = 0              // spad_bank RAM_STYLE (0 auto, 1 block, 2 ultra, 3 distributed)
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
    localparam int LBW   = (LB > 0) ? LB : 1;
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

    // ---------------------------------------------------------------- banks
    logic [NB-1:0]   b_re, b_we;
    logic [DW-1:0]   b_raddr [NB];
    logic [DW-1:0]   b_waddr [NB];
    logic [127:0]    b_wdata [NB];
    logic [15:0]     b_wstrb [NB];
    logic [127:0]    b_rdata [NB];

    for (genvar b = 0; b < NB; b++) begin : g_bank
        spad_bank #(.DEPTH(DEPTH), .OREG(OREG), .RAM_STYLE(RAM_STYLE)) u_bank (
            .clk(clk), .we(b_we[b]), .waddr(b_waddr[b]), .wdata(b_wdata[b]), .wstrb(b_wstrb[b]),
            .re(b_re[b]), .raddr(b_raddr[b]), .rdata(b_rdata[b]));
    end

    // ---------------------------------------------------------------- per-port geometry
    // start bank s, row of the start line r0; bank b of the beat holds row r0 when b >= s, r0 + 1
    // when b < s (the beat wrapped round the banks); rows wrap at DEPTH.
    logic [LBW-1:0] rs  [NR];                 // read start bank
    logic [DW-1:0]  rr0 [NR], rr1 [NR];       // read rows
    logic [NB-1:0]  rm  [NR];                 // read bank masks
    logic [LBW-1:0] ws  [NW];
    logic [DW-1:0]  wr0 [NW], wr1 [NW];
    logic [NB-1:0]  wm  [NW];

    // m rotated left by s (bit i -> bit (i + s) mod NB), log2(NB) fixed stages
    function automatic logic [NB-1:0] rotl(input logic [NB-1:0] m, input logic [LBW-1:0] s);
        logic [NB-1:0] x;
        x = m;
        for (int k = 0; k < LB; k++)
            if (s[k]) x = NB'({x, x} >> (NB - (1 << k)));
        rotl = x;
    endfunction

    /* verilator lint_off UNUSEDSIGNAL */
    function automatic logic [DW-1:0] row_of(input logic [16:0] l);
        logic [LW-1:0] lm;
        lm = LW'(l);
        row_of = DW'(lm >> LB);
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    for (genvar p = 0; p < NR; p++) begin : g_rgeo
        localparam int PL = int'(RPL[p*8 +: 8]);
        localparam logic [NB-1:0] M0 = NB'((64'd1 << PL) - 64'd1);
        logic [16:0] a;
        assign a      = r_addr[p*17 +: 17];
        assign rs[p]  = (LB > 0) ? LBW'(a) : '0;
        assign rr0[p] = row_of(a);
        assign rr1[p] = rr0[p] + DW'(1);
        assign rm[p]  = rotl(M0, rs[p]);
    end
    for (genvar p = 0; p < NW; p++) begin : g_wgeo
        localparam int PL = int'(WPL[p*8 +: 8]);
        localparam logic [NB-1:0] M0 = NB'((64'd1 << PL) - 64'd1);
        logic [16:0] a;
        assign a      = w_addr[p*17 +: 17];
        assign ws[p]  = (LB > 0) ? LBW'(a) : '0;
        assign wr0[p] = row_of(a);
        assign wr1[p] = wr0[p] + DW'(1);
        assign wm[p]  = rotl(M0, ws[p]);
    end

    // ---------------------------------------------------------------- read arbitration
    // rc[p][q]: ports p and q want a common bank
    logic [NR-1:0] rc [NR];
    always_comb begin
        for (int p = 0; p < NR; p++)
            for (int q = 0; q < NR; q++)
                rc[p][q] = (p != q) && ((rm[p] & rm[q]) != '0);
    end
    // grant vector for rotation r: ports in the order r, r+1, ..., r+NR-1 (mod NR), static per r
    logic [NR-1:0] rg [NR];
    always_comb begin
        for (int r = 0; r < NR; r++) begin
            logic [NR-1:0] g;
            g = '0;
            for (int k = 0; k < NR; k++) begin
                // p = (r + k) mod NR: a constant after unrolling
                g[(r + k) % NR] = r_valid[(r + k) % NR] && ((rc[(r + k) % NR] & g) == '0);
            end
            rg[r] = g;
        end
    end
    always_comb begin
        r_ready = '0;
        for (int r = 0; r < NR; r++)
            r_ready = r_ready | (rg[r] & {NR{32'(rr) == r}});
    end
    // per bank: the (at most one) granted port that owns it
    always_comb begin
        for (int b = 0; b < NB; b++) begin
            logic [DW-1:0] ad;
            logic          re;
            ad = '0;
            re = 1'b0;
            for (int p = 0; p < NR; p++) begin
                logic s;
                s  = r_ready[p] && rm[p][b];
                re = re | s;
                ad = ad | ({DW{s}} & ((LB > 0 && 32'(b) < 32'(rs[p])) ? rr1[p] : rr0[p]));
            end
            b_re[b]    = re;
            b_raddr[b] = ad;
        end
    end

    // read response: grant + start bank delayed by the read latency, bank outputs rotated back
    // into address order (line j of port p = bank (s + j) mod NB)
    logic [NR-1:0]  dv0, dv1;
    logic [LBW-1:0] ds0 [NR];
    logic [LBW-1:0] ds1 [NR];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            dv0 <= '0;
            dv1 <= '0;
        end else begin
            dv0 <= r_valid & r_ready;
            dv1 <= dv0;
        end
        for (int p = 0; p < NR; p++) begin
            if (r_valid[p] && r_ready[p]) ds0[p] <= rs[p];
            ds1[p] <= ds0[p];
        end
    end
    assign d_valid = (OREG != 0) ? dv1 : dv0;

    // v rotated right by s lines of W bits (line i <- line (i + s) mod NB), log2(NB) fixed stages
    function automatic logic [NB*128-1:0] rotr_d(input logic [NB*128-1:0] v, input logic [LBW-1:0] s);
        logic [NB*128-1:0] x;
        x = v;
        for (int k = 0; k < LB; k++)
            if (s[k]) x = (NB*128)'({x, x} >> ((1 << k) * 128));
        rotr_d = x;
    endfunction
    // v rotated left by s lines of 144 bits (line i -> line (i + s) mod NB)
    function automatic logic [NB*144-1:0] rotl_w(input logic [NB*144-1:0] v, input logic [LBW-1:0] s);
        logic [NB*144-1:0] x;
        x = v;
        for (int k = 0; k < LB; k++)
            if (s[k]) x = (NB*144)'({x, x} >> ((NB - (1 << k)) * 144));
        rotl_w = x;
    endfunction

    logic [NB*128-1:0] b_rflat;
    always_comb begin
        for (int b = 0; b < NB; b++) b_rflat[b*128 +: 128] = b_rdata[b];
    end
    for (genvar p = 0; p < NR; p++) begin : g_rd
        localparam int PL = int'(RPL[p*8 +: 8]);
        logic [LBW-1:0]    sh;
        logic [NB*128-1:0] x;
        assign sh = (OREG != 0) ? ds1[p] : ds0[p];
        assign x  = rotr_d(b_rflat, sh);
        for (genvar j = 0; j < PMAX; j++) begin : g_l
            if (j < PL) begin : g_on
                assign d_data[(p*PMAX + j)*128 +: 128] = x[j*128 +: 128];
            end else begin : g_off
                assign d_data[(p*PMAX + j)*128 +: 128] = '0;
            end
        end
    end

    // ---------------------------------------------------------------- write arbitration
    logic [NW-1:0] wc [NW];
    always_comb begin
        for (int p = 0; p < NW; p++)
            for (int q = 0; q < NW; q++)
                wc[p][q] = (p != q) && ((wm[p] & wm[q]) != '0);
    end
    logic [NW-1:0] wg [NW];
    always_comb begin
        for (int r = 0; r < NW; r++) begin
            logic [NW-1:0] g;
            g = '0;
            for (int k = 0; k < NW; k++)
                g[(r + k) % NW] = w_valid[(r + k) % NW] && ((wc[(r + k) % NW] & g) == '0);
            wg[r] = g;
        end
    end
    always_comb begin
        w_ready = '0;
        for (int r = 0; r < NW; r++)
            w_ready = w_ready | (wg[r] & {NW{32'(wr) == r}});
    end

    // write lines of port p rotated onto the banks: bank (s + j) mod NB gets line j
    logic [NB*144-1:0] wl [NW];               // {strb, data} per bank
    for (genvar p = 0; p < NW; p++) begin : g_wrot
        localparam int PL = int'(WPL[p*8 +: 8]);
        logic [NB*144-1:0] x0;
        always_comb begin
            x0 = '0;
            for (int b = 0; b < NB; b++)
                if (b < PL) x0[b*144 +: 144] = {w_strb[(p*PMAX + b)*16 +: 16], w_data[(p*PMAX + b)*128 +: 128]};
        end
        assign wl[p] = rotl_w(x0, ws[p]);
    end
    always_comb begin
        for (int b = 0; b < NB; b++) begin
            logic [DW-1:0]  ad;
            logic [143:0]   d;
            logic           we;
            ad = '0;
            d  = '0;
            we = 1'b0;
            for (int p = 0; p < NW; p++) begin
                logic s;
                s  = w_ready[p] && wm[p][b];
                we = we | s;
                ad = ad | ({DW{s}} & ((LB > 0 && 32'(b) < 32'(ws[p])) ? wr1[p] : wr0[p]));
                d  = d | ({144{s}} & wl[p][b*144 +: 144]);
            end
            b_we[b]    = we;
            b_waddr[b] = ad;
            b_wdata[b] = d[127:0];
            b_wstrb[b] = d[143:128];
        end
    end

`ifndef SYNTHESIS
    initial begin
        if ((1 << LB) != NB || (1 << LW) != LINES || LINES < NB)
            $fatal(1, "buf_spad: LINES and NB must be powers of two, LINES >= NB");
        if (OREG != 0 && OREG != 1)
            $fatal(1, "buf_spad: OREG must be 0 or 1");
        for (int p = 0; p < NR; p++)
            if (int'(RPL[p*8 +: 8]) > NB || int'(RPL[p*8 +: 8]) > PMAX || int'(RPL[p*8 +: 8]) < 1)
                $fatal(1, "buf_spad: read port %0d wider than NB/PMAX (or zero)", p);
        for (int p = 0; p < NW; p++)
            if (int'(WPL[p*8 +: 8]) > NB || int'(WPL[p*8 +: 8]) > PMAX || int'(WPL[p*8 +: 8]) < 1)
                $fatal(1, "buf_spad: write port %0d wider than NB/PMAX (or zero)", p);
    end
`endif
endmodule
