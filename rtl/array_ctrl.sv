// array_ctrl.sv -- tile controller around systolic_array (contract C3, docs/contracts.md).
//
// One command = one weight tile: out[m, 0..N-1] (+)= A[m, 0..k-1] @ W[0..k-1, 0..n-1]
// for m = 0..cmd_m-1, with 1 <= cmd_k, cmd_n <= N and 1 <= cmd_m <= M_DEPTH.
//   cmd_acc  = 0: overwrite accumulator rows abase+m;  1: add into them (K accumulation)
//   cmd_emit = 1: also stream the resulting INT32 rows on out_* (in m order, last flag)
//   cmd_int4 = 1: weight beats carry N packed INT4 weights (C1b) in their low N*4 bits
//   cmd_gs   = 1: W4A8 group scales (ADR-004, only with cmd_int4 = 1): the weight stream
//              carries k+1 beats, beat 0 = per-column uint8 multipliers s[c] (byte c), and
//              the accumulate step becomes ACC = (acc ? ACC : 0) + s[c] * psum[c]
// Streams (valid/ready, each beat accepted when valid && ready):
//   w_*  : cmd_k beats per command (K-row r = beat r), commands in order
//   a_*  : cmd_m beats per command (A row m = beat m), byte r = A[m, r]
//   out_*: cmd_m beats per emit command, out_data column c = INT32 result
// Bytes/nibbles at positions >= k (activations) or >= n (weights) are ignored (masked), so
// ragged tiles need no zero padding upstream. Output columns >= n carry 0 when every tile
// accumulated into those rows used the same n.
//
// Double buffering: the two PE weight banks alternate per command. The loader writes the
// next command's tile while the issuer streams the current one; a bank is rewritten right
// behind the last activation wavefront that reads it, so back-to-back tiles stream at
// max(M, K) cycles per tile with no bubble (docs/array.md).
// Output backpressure: the array never stalls. A row is issued only when a slot in the
// output FIFO is reserved for it (credits), so out_ready may drop at any time without loss.
module array_ctrl #(
    parameter int N        = 8,
    parameter int M_DEPTH  = 64,          // accumulator rows (power of 2)
    parameter int OF_DEPTH = 2 * N + 8,   // output FIFO depth; >= 2N+5 for full throughput
    parameter int GM_DEPTH = N + 4        // group-scale rows in flight (cmd_gs tiles); >= N+3 for full throughput
) (
    input  logic                         clk,
    input  logic                         rst_n,
    // command
    input  logic                         cmd_valid,
    output logic                         cmd_ready,
    input  logic [$clog2(M_DEPTH+1)-1:0] cmd_m,
    input  logic [$clog2(N+1)-1:0]       cmd_k,
    input  logic [$clog2(N+1)-1:0]       cmd_n,
    input  logic [$clog2(M_DEPTH)-1:0]   cmd_abase,
    input  logic                         cmd_acc,
    input  logic                         cmd_emit,
    input  logic                         cmd_int4,
    input  logic                         cmd_gs,
    // weight rows
    input  logic                         w_valid,
    output logic                         w_ready,
    input  logic [N*8-1:0]               w_data,
    // activation rows
    input  logic                         a_valid,
    output logic                         a_ready,
    input  logic [N*8-1:0]               a_data,
    // result rows
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [N*32-1:0]              out_data,
    output logic                         out_last,
    // status
    output logic                         idle,
    output logic                         hazard_seen,
    output logic                         overflow_seen
);
    localparam int MW    = $clog2(M_DEPTH + 1);
    localparam int AB    = $clog2(M_DEPTH);
    localparam int KW    = $clog2(N + 1);
    localparam int RB    = $clog2(N);
    localparam int CRW   = $clog2(OF_DEPTH + 1);
    localparam int TAG_W = AB + 4;          // {gs, last, emit, acc, addr}
    // group-scale product width: |psum| of an INT4 tile is <= N * 128 * 8 = N * 2^10,
    // so a signed (log2(N) + 12)-bit psum operand is exact (8 x 17 bits at N = 32, ADR-004)
    localparam int GSW   = $clog2(N) + 12;
    localparam int IFW   = $clog2(2 * N + 8);

    // ---------------------------------------------------------------- per-bank state
    // loaded[b]: bank b holds a complete tile that the issuer has not finished streaming.
    // d_*[b]: that tile's descriptor (written when its load completes).
    logic [1:0]     loaded;
    logic [MW-1:0]  d_m     [0:1];
    logic [KW-1:0]  d_k     [0:1];
    logic [AB-1:0]  d_abase [0:1];
    logic [1:0]     d_acc, d_emit, d_gen, d_gs;

    // ---------------------------------------------------------------- loader
    // Accepts the next command as soon as the previous load is done, but writes weight rows
    // into bank b only while loaded[b] == 0, i.e. after the issuer's last activation beat of
    // the previous tile in that bank. Because the array's write wavefront trails that
    // activation wavefront (systolic_array reuse rule W >= T_last + r), no drain wait is
    // needed: the first write lands >= 1 cycle after the last activation enters the array.
    logic [1:0]     ld_cnt;          // tile counter: [0] = bank, [1] = generation (checker)
    logic           ld_active, ld_bank, ld_gen, ld_int4, ld_acc, ld_emit, ld_gs, ld_sc;
    logic [RB-1:0]  ld_row;
    logic [KW-1:0]  ld_k, ld_n;
    logic [MW-1:0]  ld_m;
    logic [AB-1:0]  ld_abase;
    logic           ld_beat, ld_last, cmd_fire, sc_beat, gm_full;

    // ld_sc: the scale beat (beat 0 of a cmd_gs tile) is still to come. It goes to the
    // group-scale FIFO, not into a bank, so it may arrive while the bank is still busy.
    assign w_ready   = ld_active && (ld_sc ? !gm_full : !loaded[ld_bank]);
    assign sc_beat   = w_valid && w_ready && ld_sc;
    assign ld_beat   = w_valid && w_ready && !ld_sc;
    assign ld_last   = ld_beat && ({1'b0, ld_row} == KW'(ld_k - 1'b1));
    assign cmd_ready = !ld_active || ld_last;
    assign cmd_fire  = cmd_valid && cmd_ready;

    // column mask for ragged n (bytes in INT8 mode, nibbles in INT4 mode)
    logic [N*8-1:0] wmask;
    always_comb begin
        wmask = '0;
        for (int c = 0; c < N; c++) begin
            if (c < ld_n) begin
                if (ld_int4) wmask[c*4 +: 4] = 4'hF;
                else         wmask[c*8 +: 8] = 8'hFF;
            end
        end
    end

    // registered weight write port
    logic           wr_en_q, wr_bank_q, wr_gen_q, wr_int4_q;
    logic [RB-1:0]  wr_row_q;
    logic [N*8-1:0] wr_data_q;

    // ---------------------------------------------------------------- issuer
    logic           is_ptr;
    logic [MW-1:0]  is_row;
    logic [CRW-1:0] cred;
    logic           is_beat, is_last, cred_ok, out_pop;
    logic [N*8-1:0] amask;

    assign cred_ok = !d_emit[is_ptr] || (cred < CRW'(OF_DEPTH));
    assign a_ready = loaded[is_ptr] && cred_ok;
    assign is_beat = a_valid && a_ready;
    assign is_last = is_beat && (is_row == d_m[is_ptr] - 1'b1);
    assign out_pop = out_valid && out_ready;

    always_comb begin
        amask = '0;
        for (int r = 0; r < N; r++)
            if (r < d_k[is_ptr]) amask[r*8 +: 8] = 8'hFF;
    end

    logic             arr_in_valid, arr_in_sel, arr_in_gen;
    logic [N*8-1:0]   arr_in_a;
    logic [TAG_W-1:0] arr_in_tag;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ld_cnt <= '0; ld_active <= 1'b0; ld_bank <= 1'b0; ld_gen <= 1'b0; ld_int4 <= 1'b0;
            ld_acc <= 1'b0; ld_emit <= 1'b0; ld_gs <= 1'b0; ld_sc <= 1'b0;
            ld_row <= '0; ld_k <= '0; ld_n <= '0;
            ld_m <= '0; ld_abase <= '0;
            loaded <= '0; d_acc <= '0; d_emit <= '0; d_gen <= '0; d_gs <= '0;
            for (int b = 0; b < 2; b++) begin
                d_m[b] <= '0; d_k[b] <= '0; d_abase[b] <= '0;
            end
            wr_en_q <= 1'b0; wr_bank_q <= 1'b0; wr_gen_q <= 1'b0; wr_int4_q <= 1'b0;
            wr_row_q <= '0; wr_data_q <= '0;
            is_ptr <= 1'b0; is_row <= '0; cred <= '0;
            arr_in_valid <= 1'b0; arr_in_sel <= 1'b0; arr_in_gen <= 1'b0;
            arr_in_a <= '0; arr_in_tag <= '0;
        end else begin
            // loader
            if (cmd_fire) begin
                ld_active <= 1'b1;
                ld_cnt    <= ld_cnt + 1'b1;
                ld_bank   <= ld_cnt[0];
                ld_gen    <= ld_cnt[1];
                ld_row    <= '0;
                ld_k      <= cmd_k;
                ld_n      <= cmd_n;
                ld_m      <= cmd_m;
                ld_abase  <= cmd_abase;
                ld_acc    <= cmd_acc;
                ld_emit   <= cmd_emit;
                ld_int4   <= cmd_int4;
                ld_gs     <= cmd_gs;
                ld_sc     <= cmd_gs;
            end else if (ld_last) begin
                ld_active <= 1'b0;
            end else if (ld_beat) begin
                ld_row <= ld_row + 1'b1;
            end else if (sc_beat) begin
                ld_sc <= 1'b0;
            end
            if (ld_last) begin
                d_m[ld_bank]     <= ld_m;
                d_k[ld_bank]     <= ld_k;
                d_abase[ld_bank] <= ld_abase;
                d_acc[ld_bank]   <= ld_acc;
                d_emit[ld_bank]  <= ld_emit;
                d_gen[ld_bank]   <= ld_gen;
                d_gs[ld_bank]    <= ld_gs;
            end
            wr_en_q   <= ld_beat;
            wr_bank_q <= ld_bank;
            wr_gen_q  <= ld_gen;
            wr_int4_q <= ld_int4;
            wr_row_q  <= ld_row;
            wr_data_q <= w_data & wmask;

            // bank state (ld_last never targets a bank with loaded == 1, see w_ready)
            for (int b = 0; b < 2; b++) begin
                if (ld_last && ld_bank == b[0])        loaded[b] <= 1'b1;
                else if (is_last && is_ptr == b[0])    loaded[b] <= 1'b0;
            end

            // issuer
            if (is_beat) begin
                is_row <= is_last ? '0 : is_row + 1'b1;
                if (is_last) is_ptr <= ~is_ptr;
            end
            cred <= cred + CRW'(is_beat && d_emit[is_ptr]) - CRW'(out_pop);
            arr_in_valid <= is_beat;
            arr_in_sel   <= is_ptr;
            arr_in_gen   <= d_gen[is_ptr];
            arr_in_a     <= a_data & amask;
            arr_in_tag   <= {d_gs[is_ptr], is_last, d_emit[is_ptr], d_acc[is_ptr],
                             d_abase[is_ptr] + AB'(is_row)};
        end
    end

    // ---------------------------------------------------------------- array
    logic             arr_out_valid;
    logic [N*32-1:0]  arr_out_psum;
    logic [TAG_W-1:0] arr_out_tag;

    systolic_array #(.N(N), .TAG_W(TAG_W)) u_array (
        .clk(clk), .rst_n(rst_n),
        .in_valid(arr_in_valid), .in_a(arr_in_a), .in_sel(arr_in_sel), .in_gen(arr_in_gen),
        .in_tag(arr_in_tag),
        .w_wr_en(wr_en_q), .w_wr_bank(wr_bank_q), .w_wr_gen(wr_gen_q), .w_wr_row(wr_row_q),
        .w_wr_int4(wr_int4_q),
        .w_wr_data(wr_data_q),
        .out_valid(arr_out_valid), .out_psum(arr_out_psum), .out_tag(arr_out_tag),
        .hazard_seen(hazard_seen));

    // ---------------------------------------------------------------- accumulate (RMW)
    logic            s1_v, s1_fwd;
    logic [N*32-1:0] s1_psum, s1_fwd_data, acc_rdata, acc_old, sum;
    logic [TAG_W-1:0] s1_tag;
    logic            s1_gs, s1_last, s1_emit, s1_acc;
    logic [AB-1:0]   s1_addr, out_addr;
    /* verilator lint_off UNUSEDSIGNAL */
    logic            acc_rvalid, of_full, of_ovf;
    logic [$clog2(OF_DEPTH+3)-1:0] of_count;
    /* verilator lint_on UNUSEDSIGNAL */

    assign {s1_gs, s1_last, s1_emit, s1_acc, s1_addr} = s1_tag;
    assign out_addr = arr_out_tag[AB-1:0];

    scratchpad #(.WIDTH(N * 32), .DEPTH(M_DEPTH), .RD_LAT(1)) u_acc (
        .clk(clk), .rst_n(rst_n),
        .wr_en(s1_v), .wr_addr(s1_addr), .wr_data(sum),
        .rd_en(arr_out_valid && arr_out_tag[AB]), .rd_addr(out_addr),
        .rd_data(acc_rdata), .rd_valid(acc_rvalid));

    // group-scale FIFO: one N x uint8 row per cmd_gs tile, pushed by its scale beat (load
    // order = issue order = RMW order), popped when the tile's last row leaves the RMW stage.
    // The scale beat waits while the FIFO is full, so it can never overflow.
    logic [N*8-1:0] gm_row;
    /* verilator lint_off UNUSEDSIGNAL */
    logic           gm_valid, gm_ovf;
    logic [$clog2(GM_DEPTH+3)-1:0] gm_count;
    /* verilator lint_on UNUSEDSIGNAL */
    spad_fifo #(.W(N * 8), .DEPTH(GM_DEPTH)) u_gm (
        .clk(clk), .rst_n(rst_n),
        .push(sc_beat), .din(w_data), .full(gm_full),
        .out_valid(gm_valid), .out_ready(s1_v && s1_gs && s1_last), .out_data(gm_row),
        .count(gm_count), .overflow(gm_ovf));

    assign acc_old = s1_fwd ? s1_fwd_data : acc_rdata;
    for (genvar c = 0; c < N; c++) begin : g_sum
        logic signed [GSW-1:0]   ps_c;
        logic signed [GSW+8:0]   sc_c;
        logic        [31:0]      add_c;
        assign ps_c  = s1_psum[c*32 +: GSW];
        assign sc_c  = ps_c * $signed({1'b0, gm_row[c*8 +: 8]});
        assign add_c = s1_gs ? 32'(sc_c) : s1_psum[c*32 +: 32];
        assign sum[c*32 +: 32] = (s1_acc ? acc_old[c*32 +: 32] : 32'd0) + add_c;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_v <= 1'b0; s1_fwd <= 1'b0; s1_tag <= '0; s1_psum <= '0; s1_fwd_data <= '0;
        end else begin
            s1_v        <= arr_out_valid;
            s1_tag      <= arr_out_tag;
            s1_psum     <= arr_out_psum;
            // read-during-write bypass: the row being written now is the one read now
            s1_fwd      <= s1_v && (s1_addr == out_addr);
            s1_fwd_data <= sum;
        end
    end

    // ---------------------------------------------------------------- output FIFO
    spad_fifo #(.W(N * 32 + 1), .DEPTH(OF_DEPTH)) u_ofifo (
        .clk(clk), .rst_n(rst_n),
        .push(s1_v && s1_emit), .din({s1_last, sum}), .full(of_full),
        .out_valid(out_valid), .out_ready(out_ready), .out_data({out_last, out_data}),
        .count(of_count), .overflow(of_ovf));

    // ---------------------------------------------------------------- idle
    logic [IFW-1:0] inflight;
    always_ff @(posedge clk) begin
        if (!rst_n) inflight <= '0;
        else        inflight <= inflight + IFW'(is_beat) - IFW'(s1_v);
    end
    assign idle = !ld_active && (loaded == 2'b00) && (inflight == '0) && (cred == '0);

    // overflow_seen (sim only): output FIFO overflow, group-scale FIFO overflow, or a cmd_gs
    // row reaching the RMW stage without its scale row at the FIFO head
`ifndef SYNTHESIS
    logic gm_missing;
    always_ff @(posedge clk) begin
        if (!rst_n) gm_missing <= 1'b0;
        else if (s1_v && s1_gs && !gm_valid) gm_missing <= 1'b1;
    end
    assign overflow_seen = of_ovf | gm_ovf | gm_missing;
`else
    assign overflow_seen = 1'b0;
`endif
endmodule
