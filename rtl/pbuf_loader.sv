// pbuf_loader.sv -- PBUF parameter loader of the VEC engine (docs/sfu.md §10).
//
// Copies parameter blocks from PBUF (one 16-byte line per cycle over a line read port) into
// the SFU configuration port (docs/sfu.md §2) or into the engine's requant record buffer,
// and keeps track of which blocks are already resident in an SFU slot.
//
//   kind        block (docs/isa.md §3)                lines                    target
//   K_LUT  0    LUT_LOAD table t (uint16[256])        32                       region t
//   K_GELU 1    GELU LUT (int8[256])                  16                       region 3, slot
//   K_LN   2    LN parameter block of `cols`          2+ceil(c/8)+ceil(c/4)    region 4, slot
//   K_CTX  3    softmax ctx = score record            1                        region 5, slot
//   K_REC  4    `n` requant records (16 B each)       n                        rec_* port
//
// Residency policy. Every SFU slot (3 LUT tables, NG GELU, NSLOT LN, NCTX ctx) has a tag:
// {valid, first PBUF line, end line, cols for LN}. A request whose block (same first line,
// same cols) is tagged valid is a hit and costs one cycle; otherwise the block is loaded
// into the first invalid slot of its kind, or the round-robin victim, and tagged. A tag is
// invalidated when any PBUF line inside [first, end) is written (snoop inputs: the DMA's
// PBUF writes and the engine's own), and all tags are dropped by reset / engine clear. So
// a hit always returns exactly the bytes PBUF holds now, for any program. K_REC is never
// cached (records go to a single buffer that each REQUANT chunk overwrites).
//
// Cost of a miss: n line reads, pipelined with up to FD in flight, so n + (read latency)
// + 2 cycles; a hit costs 1 cycle.
module pbuf_loader #(
    parameter int AW    = 17,
    parameter int NSLOT = 16,
    parameter int NG    = 4,
    parameter int NCTX  = 8,
    parameter int FD    = 16
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           clr,
    // request (one at a time; req_valid is a pulse while idle)
    input  logic           req_valid,
    output logic           idle,
    input  logic [2:0]     req_kind,
    input  logic [AW-1:0]  req_line,
    input  logic [12:0]    req_cols,     // K_LN
    input  logic [1:0]     req_table,    // K_LUT
    input  logic [12:0]    req_n,        // K_REC: number of records
    output logic           done,         // pulse: the block is in its slot / the records are written
    output logic [7:0]     slot,         // valid with done (K_GELU, K_LN, K_CTX)
    output logic           hit,          // valid with done
    // PBUF write snoop
    input  logic           snoop0_valid,
    input  logic [AW-1:0]  snoop0_line,
    input  logic           snoop1_valid,
    input  logic [AW-1:0]  snoop1_line,
    // PBUF line read port
    output logic           ar_valid,
    input  logic           ar_ready,
    output logic [AW-1:0]  ar_addr,
    input  logic           r_valid,
    input  logic [127:0]   r_data,
    // SFU configuration port
    output logic           cfg_we,
    output logic [3:0]     cfg_region,
    output logic [7:0]     cfg_slot,
    output logic [11:0]    cfg_line,
    output logic [12:0]    cfg_cols,
    output logic [127:0]   cfg_data,
    // requant records: index, {ls, shift, mult, bias} as the 4 int32 words of the record
    output logic           rec_we,
    output logic [12:0]    rec_idx,
    output logic [127:0]   rec_data,
    // statistics (cleared by reset only)
    output logic [31:0]    st_hits,
    output logic [31:0]    st_misses,
    output logic [31:0]    st_lines
);
    localparam logic [2:0] K_LUT = 3'd0, K_GELU = 3'd1, K_LN = 3'd2, K_CTX = 3'd3, K_REC = 3'd4;
    localparam int CW = $clog2(FD + 1);
    localparam int NT = 3 + NG + NSLOT + NCTX;     // tag entries: LUT 0..2, GELU, LN, CTX
    localparam int G0 = 3, N0 = 3 + NG, C0 = 3 + NG + NSLOT;

    logic rst_i;
    assign rst_i = rst_n && !clr;

    // ---------------- tags ----------------
    logic          t_v    [NT];
    logic [AW-1:0] t_line [NT];
    logic [AW:0]   t_end  [NT];
    logic [12:0]   t_cols [NT];
    logic [7:0]    rr_g, rr_n, rr_c;

    function automatic logic [13:0] ln_lines(input logic [12:0] c);
        ln_lines = 14'd2 + 14'((14'(c) + 14'd7) >> 3) + 14'((14'(c) + 14'd3) >> 2);
    endfunction

    // lookup (combinational on the request)
    logic [13:0] r_nl;
    int          lo, hi;
    logic        f_hit, f_free;
    int          f_hit_i, f_free_i;
    always_comb begin
        case (req_kind)
            K_LUT:   begin lo = int'(req_table); hi = int'(req_table) + 1; r_nl = 14'd32; end
            K_GELU:  begin lo = G0; hi = G0 + NG;    r_nl = 14'd16; end
            K_LN:    begin lo = N0; hi = N0 + NSLOT; r_nl = ln_lines(req_cols); end
            K_CTX:   begin lo = C0; hi = C0 + NCTX;  r_nl = 14'd1; end
            default: begin lo = 0;  hi = 0;          r_nl = 14'(req_n); end
        endcase
        f_hit = 1'b0; f_free = 1'b0; f_hit_i = 0; f_free_i = 0;
        for (int e = 0; e < NT; e++) begin
            if (e >= lo && e < hi) begin
                if (t_v[e] && t_line[e] == req_line && (req_kind != K_LN || t_cols[e] == req_cols)
                    && !f_hit) begin
                    f_hit = 1'b1; f_hit_i = e;
                end
                if (!t_v[e] && !f_free) begin
                    f_free = 1'b1; f_free_i = e;
                end
            end
        end
    end

    // ---------------- load engine ----------------
    logic          busy;
    logic [2:0]    k_kind;
    logic [AW-1:0] k_base;
    logic [13:0]   n_req, n_rsp, n_tot;
    logic [3:0]    k_region;
    logic [7:0]    k_slot;
    logic [12:0]   k_cols;
    logic [CW-1:0] inflight;
    logic          flush;
    logic          fin;
    int            victim;

    assign idle     = !busy && !fin;
    assign ar_valid = busy && !flush && (n_req != n_tot) && (32'(inflight) < FD);
    assign ar_addr  = k_base + AW'(n_req);

    always_comb begin
        case (req_kind)
            K_GELU:  victim = f_free ? f_free_i : G0 + int'(rr_g);
            K_LN:    victim = f_free ? f_free_i : N0 + int'(rr_n);
            K_CTX:   victim = f_free ? f_free_i : C0 + int'(rr_c);
            default: victim = lo;
        endcase
    end

    logic acc_rsp;
    assign acc_rsp = r_valid && !flush;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            inflight <= '0;
            flush    <= 1'b0;
        end else begin
            inflight <= inflight + CW'(ar_valid && ar_ready) - CW'(r_valid);
            if (clr) flush <= (inflight + CW'(ar_valid && ar_ready) - CW'(r_valid)) != '0;
            else if (flush && inflight == CW'(r_valid)) flush <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_i) begin
            busy <= 1'b0;
            fin  <= 1'b0;
            done <= 1'b0;
            rr_g <= '0; rr_n <= '0; rr_c <= '0;
            for (int e = 0; e < NT; e++) t_v[e] <= 1'b0;
        end else begin
            done <= 1'b0;
            fin  <= 1'b0;
            // snoop invalidation (before a new tag is set in the same cycle)
            for (int e = 0; e < NT; e++) begin
                if ((snoop0_valid && snoop0_line >= t_line[e] && {1'b0, snoop0_line} < t_end[e]) ||
                    (snoop1_valid && snoop1_line >= t_line[e] && {1'b0, snoop1_line} < t_end[e]))
                    t_v[e] <= 1'b0;
            end
            if (req_valid && !busy && !fin) begin
                if (req_kind != K_REC && f_hit) begin
                    fin  <= 1'b1;
                    slot <= 8'(f_hit_i - lo);
                    hit  <= 1'b1;
                end else begin
                    busy     <= 1'b1;
                    hit      <= 1'b0;
                    k_kind   <= req_kind;
                    k_base   <= req_line;
                    k_cols   <= req_cols;
                    n_req    <= '0;
                    n_rsp    <= '0;
                    n_tot    <= r_nl;
                    k_region <= (req_kind == K_LUT) ? {2'b0, req_table} :
                                (req_kind == K_GELU) ? 4'd3 : (req_kind == K_LN) ? 4'd4 : 4'd5;
                    if (req_kind != K_REC) begin
                        k_slot         <= 8'(victim - lo);
                        slot           <= 8'(victim - lo);
                        t_v[victim]    <= 1'b1;
                        t_line[victim] <= req_line;
                        t_end[victim]  <= {1'b0, req_line} + (AW+1)'(r_nl);
                        t_cols[victim] <= req_cols;
                        if (!f_free) begin
                            if (req_kind == K_GELU) rr_g <= (rr_g == 8'(NG - 1))    ? '0 : rr_g + 8'd1;
                            if (req_kind == K_LN)   rr_n <= (rr_n == 8'(NSLOT - 1)) ? '0 : rr_n + 8'd1;
                            if (req_kind == K_CTX)  rr_c <= (rr_c == 8'(NCTX - 1))  ? '0 : rr_c + 8'd1;
                        end
                    end
                end
            end
            if (fin) done <= 1'b1;
            if (busy) begin
                if (ar_valid && ar_ready) n_req <= n_req + 14'd1;
                if (acc_rsp) begin
                    n_rsp <= n_rsp + 14'd1;
                    if (n_rsp + 14'd1 == n_tot) begin
                        busy <= 1'b0;
                        fin  <= 1'b1;
                    end
                end
            end
        end
    end

    // responses go straight to the SFU cfg port / record port (neither can stall)
    always_ff @(posedge clk) begin
        if (!rst_i) begin
            cfg_we <= 1'b0;
            rec_we <= 1'b0;
        end else begin
            cfg_we     <= busy && acc_rsp && k_kind != K_REC;
            rec_we     <= busy && acc_rsp && k_kind == K_REC;
        end
        cfg_region <= k_region;
        cfg_slot   <= (k_kind == K_LUT) ? 8'd0 : k_slot;
        cfg_line   <= 12'(n_rsp);
        cfg_cols   <= k_cols;
        cfg_data   <= r_data;
        rec_idx    <= 13'(n_rsp);
        rec_data   <= r_data;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st_hits <= '0; st_misses <= '0; st_lines <= '0;
        end else begin
            if (req_valid && !busy && !fin && req_kind != K_REC) begin
                if (f_hit) st_hits <= st_hits + 32'd1;
                else       st_misses <= st_misses + 32'd1;
            end
            if (busy && acc_rsp) st_lines <= st_lines + 32'd1;
        end
    end
endmodule
