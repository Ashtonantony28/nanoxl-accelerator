// array_ctrl.sv -- tile controller around systolic_array (contract C3, docs/contracts.md).
//
// One command = one weight tile: out[m, 0..N-1] (+)= A[m, 0..k-1] @ W[0..k-1, 0..n-1]
// for m = 0..cmd_m-1, with 1 <= cmd_k, cmd_n <= N and 1 <= cmd_m <= M_DEPTH.
//   cmd_acc  = 0: overwrite accumulator rows abase+m;  1: add into them (K accumulation)
//   cmd_emit = 1: also stream the resulting INT32 rows on out_* (in m order, last flag)
//   cmd_int4 = 1: weight beats carry N packed INT4 weights (C1b) in their low N*4 bits
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
    parameter int OF_DEPTH = 2 * N + 8    // output FIFO depth; >= 2N+5 for full throughput
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
    localparam int TAG_W = AB + 3;          // {last, emit, acc, addr}
    localparam int IFW   = $clog2(2 * N + 8);

    // ---------------------------------------------------------------- per-bank state
    // loaded[b]: bank b holds a complete tile that the issuer has not finished streaming.
    // d_*[b]: that tile's descriptor (written when its load completes).
    logic [1:0]     loaded;
    logic [MW-1:0]  d_m     [0:1];
    logic [KW-1:0]  d_k     [0:1];
    logic [AB-1:0]  d_abase [0:1];
    logic [1:0]     d_acc, d_emit, d_gen;

    // ---------------------------------------------------------------- loader
    // Accepts the next command as soon as the previous load is done, but writes weight rows
    // into bank b only while loaded[b] == 0, i.e. after the issuer's last activation beat of
    // the previous tile in that bank. Because the array's write wavefront trails that
    // activation wavefront (systolic_array reuse rule W >= T_last + r), no drain wait is
    // needed: the first write lands >= 1 cycle after the last activation enters the array.
    logic [1:0]     ld_cnt;          // tile counter: [0] = bank, [1] = generation (checker)
    logic           ld_active, ld_bank, ld_gen, ld_int4, ld_acc, ld_emit;
    logic [RB-1:0]  ld_row;
    logic [KW-1:0]  ld_k, ld_n;
    logic [MW-1:0]  ld_m;
    logic [AB-1:0]  ld_abase;
    logic           ld_beat, ld_last, cmd_fire;

    assign w_ready   = ld_active && !loaded[ld_bank];
    assign ld_beat   = w_valid && w_ready;
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
            ld_acc <= 1'b0; ld_emit <= 1'b0; ld_row <= '0; ld_k <= '0; ld_n <= '0;
            ld_m <= '0; ld_abase <= '0;
            loaded <= '0; d_acc <= '0; d_emit <= '0; d_gen <= '0;
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
            end else if (ld_last) begin
                ld_active <= 1'b0;
            end else if (ld_beat) begin
                ld_row <= ld_row + 1'b1;
            end
            if (ld_last) begin
                d_m[ld_bank]     <= ld_m;
                d_k[ld_bank]     <= ld_k;
                d_abase[ld_bank] <= ld_abase;
                d_acc[ld_bank]   <= ld_acc;
                d_emit[ld_bank]  <= ld_emit;
                d_gen[ld_bank]   <= ld_gen;
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
            arr_in_tag   <= {is_last, d_emit[is_ptr], d_acc[is_ptr], d_abase[is_ptr] + AB'(is_row)};
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
    logic            s1_last, s1_emit, s1_acc;
    logic [AB-1:0]   s1_addr, out_addr;
    /* verilator lint_off UNUSEDSIGNAL */
    logic            acc_rvalid, of_full;
    logic [$clog2(OF_DEPTH+3)-1:0] of_count;
    /* verilator lint_on UNUSEDSIGNAL */

    assign {s1_last, s1_emit, s1_acc, s1_addr} = s1_tag;
    assign out_addr = arr_out_tag[AB-1:0];

    scratchpad #(.WIDTH(N * 32), .DEPTH(M_DEPTH), .RD_LAT(1)) u_acc (
        .clk(clk), .rst_n(rst_n),
        .wr_en(s1_v), .wr_addr(s1_addr), .wr_data(sum),
        .rd_en(arr_out_valid && arr_out_tag[AB]), .rd_addr(out_addr),
        .rd_data(acc_rdata), .rd_valid(acc_rvalid));

    assign acc_old = s1_fwd ? s1_fwd_data : acc_rdata;
    for (genvar c = 0; c < N; c++) begin : g_sum
        assign sum[c*32 +: 32] = (s1_acc ? acc_old[c*32 +: 32] : 32'd0) + s1_psum[c*32 +: 32];
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
        .count(of_count), .overflow(overflow_seen));

    // ---------------------------------------------------------------- idle
    logic [IFW-1:0] inflight;
    always_ff @(posedge clk) begin
        if (!rst_n) inflight <= '0;
        else        inflight <= inflight + IFW'(is_beat) - IFW'(s1_v);
    end
    assign idle = !ld_active && (loaded == 2'b00) && (inflight == '0) && (cred == '0);
endmodule
