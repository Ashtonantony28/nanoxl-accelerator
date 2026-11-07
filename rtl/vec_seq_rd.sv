// vec_seq_rd.sv -- VEC engine source stream: reads the lines of a row set (vec_seq_walk)
// over a 16-byte scratchpad line port and unpacks them into vec_unit beats of L int32
// lanes (int8 / int16 / int32 elements, sign-extended; 2^ebl bytes each).
//
// Line port (same shape as the DMA's sp_* read side, docs/control.md §3): requests
// ar_valid/ar_ready with ar_buf / ar_addr (line address); responses r_valid / r_data in
// request order, any latency, no backpressure. A credit counter keeps
// (lines in flight + lines buffered) <= FD, so a response always has room.
//
// Every row starts on a line boundary (docs/isa.md alignment rule) and is sent as
// ceil(len/L) beats, the last one with o_last. Only the row's own lines are read: lanes of
// a final beat past the row end carry zeros or neighbour bytes of the last line, which
// vec_unit masks (docs/vector-unit.md §5.3).
//
// Engine clear (`clr`): all state is dropped at once except the count of lines in flight;
// their responses are discarded on arrival and no new request is issued until they are
// all back (`flush`), so a clear never pairs a stale response with a new request.
module vec_seq_rd #(
    parameter int L  = 4,
    parameter int AW = 17,
    parameter int FD = 16
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
    input  logic [127:0]    r_data,
    output logic            o_valid,
    input  logic            o_ready,
    output logic [L*32-1:0] o_data,
    output logic            o_last
);
    localparam int LL = $clog2(L);
    localparam int AL = (L > 4) ? L / 4 : 1;      // lines in the assembly buffer (4L bytes)
    localparam int CW = $clog2(FD + 1);

    logic rst_i;
    assign rst_i = rst_n && !clr;

    // ---------------- request side ----------------
    logic          w_act, w_step, w_rll;
    logic [AW-1:0] w_addr;
    logic [CW-1:0] inflight, fcount;
    logic          flush;
    logic [2:0]    buf_q;

    vec_seq_walk #(.AW(AW)) u_walk (
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
    logic [127:0] f_data;
    fifo_sync #(.W(128), .DEPTH(FD)) u_fifo (
        .clk(clk), .rst_n(rst_i), .in_valid(r_valid && !flush), .in_ready(f_in_ready),
        .in_data(r_data), .out_valid(f_valid), .out_ready(f_pop), .out_data(f_data),
        .count(fcount)
    );

    // ---------------- unpacker ----------------
    logic [1:0]  ebl;
    logic [15:0] len_n, len_l;
    logic [18:0] rows_left;
    logic [13:0] lines_left;
    logic [15:0] beats_left;
    logic [3:0]  lpb_l2, bpl_l2;              // log2 lines per beat / beats per line
    logic [7:0]  k;                           // lines collected for the current beat group
    logic        full;
    logic [3:0]  sub;                         // beat index within a line (sub-line beats)
    logic [AL*128-1:0] asm_b;
    logic        ov, ol;
    logic [L*32-1:0] od;
    logic        act;

    function automatic logic [13:0] nlines(input logic [15:0] n, input logic [1:0] e);
        logic [17:0] b;
        b = {2'b0, n} << e;
        nlines = 14'((b + 18'd15) >> 4);
    endfunction
    function automatic logic [15:0] nbeats(input logic [15:0] n);
        nbeats = 16'((17'(n) + 17'(L - 1)) >> LL);
    endfunction

    assign act = (rows_left != '0);
    logic emit, last_slice, consume, can_fill;
    logic [7:0] k_eff;
    assign emit       = full && (!ov || o_ready);
    assign last_slice = (sub == 4'((1 << bpl_l2) - 1)) || (beats_left == 16'd1);
    assign consume    = emit && last_slice;
    assign can_fill   = act && (lines_left != '0) && (!full || consume);
    assign f_pop      = can_fill && f_valid;
    assign k_eff      = consume ? 8'd0 : k;

    // lanes of the slice `sub` of the assembly buffer, by element size
    logic [L*32-1:0] slice;
    always_comb begin
        for (int j = 0; j < L; j++) begin
            int e;
            e = (int'(sub) << LL) + j;
            case (ebl)
                2'd0:    slice[j*32 +: 32] = 32'($signed(asm_b[(e % (AL*16))*8 +: 8]));
                2'd1:    slice[j*32 +: 32] = 32'($signed(asm_b[(e % (AL*8))*16 +: 16]));
                default: slice[j*32 +: 32] = asm_b[(e % (AL*4))*32 +: 32];
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
                lpb_l2     <= (32'(d_ebl) + LL > 4) ? 4'(32'(d_ebl) + LL - 4) : 4'd0;
                bpl_l2     <= (32'(d_ebl) + LL < 4) ? 4'(4 - 32'(d_ebl) - LL) : 4'd0;
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
                    sub <= sub + 4'd1;
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
                if (k_eff == 8'd0) begin
                    asm_b <= '0;
                    asm_b[127:0] <= f_data;
                end else begin
                    asm_b[int'(k_eff)*128 +: 128] <= f_data;
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
