// vec_seq_rd.sv -- VEC engine source stream: reads the lines of a row set (vec_seq_walk)
// over a P-line (16P-byte) scratchpad port and unpacks them into vec_unit beats of L int32
// lanes (int8 / int16 / int32 elements, sign-extended; 2^ebl bytes each).
//
// Port (the DMA's sp_* read shape, docs/control.md §3, widened to P lines per beat):
// requests ar_valid/ar_ready with ar_buf / ar_addr (first line of the beat, any line);
// responses r_valid / r_data (P lines, line j in bits [128j +: 128]) in request order, any
// latency, no backpressure. A credit counter keeps (beats in flight + beats buffered) <= FD,
// so a response always has room.
//
// Every row starts on a line boundary (docs/isa.md alignment rule), is read as
// ceil(lines/P) port beats from its first line (the last one ragged: lines past the row end
// are read but never used for row data) and sent as ceil(len/L) vec_unit beats, the last
// one with o_last. Lanes of a final beat past the row end carry zeros or neighbour bytes,
// which vec_unit masks (docs/vector-unit.md §5.3). P = 1 is the 16-byte line port.
//
// Engine clear (`clr`): all state is dropped at once except the count of lines in flight;
// their responses are discarded on arrival and no new request is issued until they are
// all back (`flush`), so a clear never pairs a stale response with a new request.
module vec_seq_rd #(
    parameter int L  = 4,
    parameter int AW = 17,
    parameter int FD = 16,
    parameter int P  = 1
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            clr,
    input  logic            start,
    input  logic [2:0]      d_buf,
    input  logic [AW-1:0]   d_base,
    input  logic [AW-1:0]   d_so,
    input  logic [12:0]     d_no,
    input  logic [AW-1:0]   d_si,
    input  logic [6:0]      d_ni,
    input  logic [18:0]     d_nrows,       // no * ni
    input  logic [15:0]     d_len,
    input  logic [15:0]     d_len_last,
    input  logic [1:0]      d_ebl,
    output logic            busy,
    output logic            ar_valid,
    input  logic            ar_ready,
    output logic [2:0]      ar_buf,
    output logic [AW-1:0]   ar_addr,
    input  logic            r_valid,
    input  logic [P*128-1:0] r_data,
    output logic            o_valid,
    input  logic            o_ready,
    output logic [L*32-1:0] o_data,
    output logic            o_last
);
    localparam int LL = $clog2(L);
    localparam int PL = $clog2(P);
    localparam int RW = 128 * P;                  // port beat ("chunk") width
    localparam int AL = (4 * L > 16 * P) ? (4 * L) / (16 * P) : 1;   // chunks in the assembly buffer
    localparam int AE = AL * 16 * P;              // assembly buffer bytes (>= 4L)
    localparam int CW = $clog2(FD + 1);

    logic rst_i;
    assign rst_i = rst_n && !clr;

    // ---------------- request side ----------------
    logic          w_act, w_step, w_rll;
    logic [AW-1:0] w_addr;
    logic [CW-1:0] inflight, fcount;
    logic          flush;
    logic [2:0]    buf_q;

    vec_seq_walk #(.AW(AW), .P(P)) u_walk (
        .clk(clk), .rst_n(rst_i), .start(start), .base(d_base), .so(d_so), .no(d_no),
        .si(d_si), .ni(d_ni), .len(d_len), .len_last(d_len_last), .ebl(d_ebl),
        .step(w_step), .act(w_act), .addr(w_addr), .row_last_line(w_rll)
    );

    assign ar_valid = w_act && !flush && (32'(inflight) + 32'(fcount) < FD);
    assign ar_buf   = buf_q;
    assign ar_addr  = w_addr;
    assign w_step   = ar_valid && ar_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            inflight <= '0;
            flush    <= 1'b0;
        end else begin
            inflight <= inflight + CW'(w_step) - CW'(r_valid);
            if (clr) flush <= (inflight + CW'(w_step) - CW'(r_valid)) != '0;
            else if (flush && inflight == CW'(r_valid)) flush <= 1'b0;
        end
        if (start && !w_act) buf_q <= d_buf;
    end

    // ---------------- response FIFO ----------------
    logic         f_valid, f_pop, f_in_ready;
    logic [RW-1:0] f_data;
    fifo_sync #(.W(RW), .DEPTH(FD)) u_fifo (
        .clk(clk), .rst_n(rst_i), .in_valid(r_valid && !flush), .in_ready(f_in_ready),
        .in_data(r_data), .out_valid(f_valid), .out_ready(f_pop), .out_data(f_data),
        .count(fcount)
    );

    // ---------------- unpacker ----------------
    logic [1:0]  ebl;
    logic [15:0] len_n, len_l;
    logic [18:0] rows_left;
    logic [13:0] lines_left;                  // port beats (chunks) of the row still to collect
    logic [15:0] beats_left;
    logic [3:0]  lpb_l2, bpl_l2;              // log2 chunks per beat / beats per chunk
    logic [7:0]  k;                           // chunks collected for the current beat group
    logic        full;
    logic [6:0]  sub;                         // beat index within a chunk (sub-chunk beats)
    logic [AL*RW-1:0] asm_b;
    logic        ov, ol;
    logic [L*32-1:0] od;
    logic        act;

    function automatic logic [13:0] nlines(input logic [15:0] n, input logic [1:0] e);
        logic [17:0] b;
        b = {2'b0, n} << e;
        nlines = 14'((b + 18'(16 * P - 1)) >> (4 + PL));
    endfunction
    function automatic logic [15:0] nbeats(input logic [15:0] n);
        nbeats = 16'((17'(n) + 17'(L - 1)) >> LL);
    endfunction

    assign act = (rows_left != '0);
    logic emit, last_slice, consume, can_fill;
    logic [7:0] k_eff;
    assign emit       = full && (!ov || o_ready);
    assign last_slice = (sub == 7'((1 << bpl_l2) - 1)) || (beats_left == 16'd1);
    assign consume    = emit && last_slice;
    assign can_fill   = act && (lines_left != '0) && (!full || consume);
    assign f_pop      = can_fill && f_valid;
    assign k_eff      = consume ? 8'd0 : k;

    // lanes of the slice `sub` of the assembly buffer, by element size (docs/fpga.md C9): slice
    // `sub` starts at byte sub * L * 2^ebl, i.e. at unit u = sub << ebl of L bytes, and only
    // narrow beats (L * 2^ebl < 16P bytes) have sub > 0, so u < NU = 16P / L. One barrel shift
    // of the buffer by u units (log2(NU) fixed stages), then lane j is byte / halfword / word j.
    localparam int NU = (16 * P > L) ? (16 * P) / L : 1;
    localparam int UW = (NU > 1) ? $clog2(NU) : 1;
    function automatic logic [AL*RW-1:0] shr_units(input logic [AL*RW-1:0] v, input logic [UW-1:0] u);
        logic [AL*RW-1:0] x;
        x = v;
        for (int i = 0; i < $clog2(NU); i++)
            if (u[i]) x = x >> ((1 << i) * L * 8);
        shr_units = x;
    endfunction
    logic [UW-1:0]    su;
    logic [AL*RW-1:0] asm_s;
    logic [L*32-1:0]  slice;
    /* verilator lint_off WIDTHTRUNC */
    assign su    = UW'({9'd0, sub} << ebl);
    /* verilator lint_on WIDTHTRUNC */
    assign asm_s = shr_units(asm_b, su);
    always_comb begin
        for (int j = 0; j < L; j++) begin
            case (ebl)
                2'd0:    slice[j*32 +: 32] = 32'($signed(asm_s[j*8 +: 8]));
                2'd1:    slice[j*32 +: 32] = 32'($signed(asm_s[j*16 +: 16]));
                default: slice[j*32 +: 32] = asm_s[j*32 +: 32];
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_i) begin
            rows_left <= '0;
            full      <= 1'b0;
            ov        <= 1'b0;
            k         <= '0;
            sub       <= '0;
            lines_left <= '0;
            beats_left <= '0;
        end else begin
            if (start && !act) begin
                ebl        <= d_ebl;
                len_n      <= d_len;
                len_l      <= d_len_last;
                rows_left  <= d_nrows;
                lines_left <= nlines(d_nrows == 19'd1 ? d_len_last : d_len, d_ebl);
                beats_left <= nbeats(d_nrows == 19'd1 ? d_len_last : d_len);
                lpb_l2     <= (32'(d_ebl) + LL > 4 + PL) ? 4'(32'(d_ebl) + LL - 4 - PL) : 4'd0;
                bpl_l2     <= (32'(d_ebl) + LL < 4 + PL) ? 4'(4 + PL - 32'(d_ebl) - LL) : 4'd0;
                k          <= '0;
                sub        <= '0;
                full       <= 1'b0;
            end
            if (o_ready) ov <= 1'b0;
            if (emit) begin
                ov <= 1'b1;
                od <= slice;
                ol <= (beats_left == 16'd1);
                if (consume) begin
                    full <= 1'b0;
                    k    <= '0;
                    sub  <= '0;
                end else begin
                    sub <= sub + 7'd1;
                end
                if (beats_left == 16'd1) begin
                    rows_left <= rows_left - 19'd1;
                    lines_left <= nlines(rows_left == 19'd2 ? len_l : len_n, ebl);
                    beats_left <= nbeats(rows_left == 19'd2 ? len_l : len_n);
                end else begin
                    beats_left <= beats_left - 16'd1;
                end
            end
            if (f_pop) begin
                // chunk k_eff of the buffer; the first chunk of a group clears the others
                for (int c = 0; c < AL; c++) begin
                    if (k_eff == 8'(c))       asm_b[c*RW +: RW] <= f_data;
                    else if (k_eff == 8'd0)   asm_b[c*RW +: RW] <= '0;
                end
                k          <= k_eff + 8'd1;
                full       <= (k_eff + 8'd1 == 8'(1 << lpb_l2)) || (lines_left == 14'd1);
                lines_left <= lines_left - 14'd1;
            end
        end
    end

    assign o_valid = ov;
    assign o_data  = od;
    assign o_last  = ol;
    assign busy    = w_act || act || ov || f_valid || (inflight != '0);

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_i && r_valid && !flush && !f_in_ready)
            $error("vec_seq_rd: response FIFO overflow (credit bug)");
    end
`endif
endmodule
