// vec_reduce.sv -- horizontal reduction across lanes and across the beats of a row.
// Reference: nanoxl/vec_ref.py::reduce_row. Semantics: docs/vector-unit.md.
//
// One row = a run of beats ending with in_last. Lanes with in_keep[i]=0 are ignored
// (they contribute 0 to sums, nothing to max/min). For every row it produces, in one pass:
//   out_sum    = sum of kept int32 values                      (ACC_W-bit signed)
//   out_sumsq  = sum of squares of kept values taken as int16  (ACC_W-bit signed)
//   out_max / out_min over kept values (a row with no kept lane gives INT32_MIN / INT32_MAX)
//   out_sq_err = some kept value was outside the signed SQ_W-bit range (its square then
//                uses only the low SQ_W bits, so sumsq is not the true value)
// Exactness: with ACC_W = 48 the sum is exact for rows of up to 2^16 int32 values and the
// sum of squares for up to 2^17 int16 values (each square <= 2^30). A LayerNorm row of 384
// int16 values needs at most 39 bits. Beyond that the accumulators wrap modulo 2^ACC_W.
// Pipeline: S1 lane mask + square | S2 balanced adder / compare tree across lanes, NTS
//           register stages of TLV = 2 levels each (NTS = max(1, ceil(log2(L) / 2)): 1 for
//           L <= 4, 2 for L = 8 or 16, 3 for L = 32 or 64) | S3 row accumulator + result
//           register. Result valid 2 + NTS cycles after the last beat (L = 16: 4, L = 64: 5).
// Handshake: whole-pipeline stall while a result is waiting: in_ready = !out_valid || out_ready.
module vec_reduce #(
    parameter int L     = 4,
    parameter int ACC_W = 48,
    parameter int SQ_W  = 16
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             in_valid,
    output logic             in_ready,
    input  logic [L*32-1:0]  in_data,
    input  logic [L-1:0]     in_keep,
    input  logic             in_last,
    output logic             out_valid,
    input  logic             out_ready,
    output logic [ACC_W-1:0] out_sum,
    output logic [ACC_W-1:0] out_sumsq,
    output logic [31:0]      out_max,
    output logic [31:0]      out_min,
    output logic             out_sq_err
);
    localparam int P   = (L <= 1) ? 1 : (1 << $clog2(L));   // tree leaves (power of 2)
    localparam int LV  = $clog2(P);                          // tree levels
    localparam int TLV = 2;                                  // levels per tree stage
    localparam int NTS = (LV <= TLV) ? 1 : (LV + TLV - 1) / TLV;   // tree register stages

    logic adv;
    assign adv      = !out_valid || out_ready;
    assign in_ready = adv;

    // ---------------- S1: mask, square ----------------
    logic                    v1, last1, err1;
    logic signed [ACC_W-1:0] s1_sum [L];
    logic signed [ACC_W-1:0] s1_sq  [L];
    logic signed [31:0]      s1_max [L];
    logic signed [31:0]      s1_min [L];

    always_ff @(posedge clk) begin
        if (!rst_n) v1 <= 1'b0;
        else if (adv) v1 <= in_valid;
        if (adv) begin
            last1 <= in_last;
            err1  <= 1'b0;
            for (int j = 0; j < L; j++) begin
                logic signed [31:0]       d;
                logic signed [SQ_W-1:0]   dn;
                logic signed [2*SQ_W-1:0] sq;
                d  = in_data[j*32 +: 32];
                dn = d[SQ_W-1:0];
                sq = dn * dn;
                s1_sum[j] <= in_keep[j] ? {{(ACC_W-32){d[31]}}, d} : '0;
                s1_sq[j]  <= in_keep[j] ? {{(ACC_W-2*SQ_W){1'b0}}, sq} : '0;
                s1_max[j] <= in_keep[j] ? d : 32'sh8000_0000;
                s1_min[j] <= in_keep[j] ? d : 32'sh7FFF_FFFF;
                if (in_keep[j] && (d != {{(32-SQ_W){dn[SQ_W-1]}}, dn})) err1 <= 1'b1;
            end
        end
    end

    // ---------------- S2: balanced trees (heap layout: node 1 = root, leaves P..2P-1) -----
    // The tree is cut every TLV = 2 levels (C7, docs/fpga.md): the nodes at depth d with
    // 0 < d < LV and (LV - d) % 2 == 0 are registered (hr*), so the tree takes NTS register
    // stages; the last one holds the root (s2_*). L = 64: 3 stages, L = 16: 2, L <= 4: 1.
    logic signed [ACC_W-1:0] hs [2*P];
    logic signed [ACC_W-1:0] hq [2*P];
    logic signed [31:0]      hm [2*P];
    logic signed [31:0]      hn [2*P];
    /* verilator lint_off UNDRIVEN */
    logic signed [ACC_W-1:0] hrs [2*P];   // boundary-node registers (others unused)
    logic signed [ACC_W-1:0] hrq [2*P];
    logic signed [31:0]      hrm [2*P];
    logic signed [31:0]      hrn [2*P];
    /* verilator lint_on UNDRIVEN */
    logic signed [ACC_W-1:0] r_sum, r_sq;
    logic signed [31:0]      r_max, r_min;

    function automatic int depth_of(input int n);   // floor(log2(n)), n >= 1
        depth_of = 0;
        for (int k = 1; k < 31; k++) if (n >= (1 << k)) depth_of = k;
    endfunction
    function automatic logic is_cut(input int n);   // node n is a registered boundary
        int d;
        d = depth_of(n);
        is_cut = (d > 0) && (d < LV) && (((LV - d) % TLV) == 0);
    endfunction

    always_comb begin
        hs[0] = '0; hq[0] = '0; hm[0] = '0; hn[0] = '0;
        for (int n = 0; n < P; n++) begin
            if (n < L) begin
                hs[P+n] = s1_sum[n]; hq[P+n] = s1_sq[n]; hm[P+n] = s1_max[n]; hn[P+n] = s1_min[n];
            end else begin
                hs[P+n] = '0; hq[P+n] = '0; hm[P+n] = 32'sh8000_0000; hn[P+n] = 32'sh7FFF_FFFF;
            end
        end
        for (int n = P - 1; n >= 1; n--) begin
            logic signed [ACC_W-1:0] s0, s1, q0, q1;
            logic signed [31:0]      m0, m1, n0, n1;
            if (is_cut(2 * n)) begin
                s0 = hrs[2*n]; s1 = hrs[2*n+1]; q0 = hrq[2*n]; q1 = hrq[2*n+1];
                m0 = hrm[2*n]; m1 = hrm[2*n+1]; n0 = hrn[2*n]; n1 = hrn[2*n+1];
            end else begin
                s0 = hs[2*n];  s1 = hs[2*n+1];  q0 = hq[2*n];  q1 = hq[2*n+1];
                m0 = hm[2*n];  m1 = hm[2*n+1];  n0 = hn[2*n];  n1 = hn[2*n+1];
            end
            hs[n] = s0 + s1;
            hq[n] = q0 + q1;
            hm[n] = (m0 > m1) ? m0 : m1;
            hn[n] = (n0 < n1) ? n0 : n1;
        end
        r_sum = hs[1]; r_sq = hq[1]; r_max = hm[1]; r_min = hn[1];
    end

    for (genvar g = 2; g < P; g++) begin : g_cut
        localparam int GD = $clog2(g + 1) - 1;          // depth of node g
        if (GD > 0 && GD < LV && ((LV - GD) % TLV) == 0) begin : g_reg
            always_ff @(posedge clk) begin
                if (adv) begin
                    hrs[g] <= hs[g]; hrq[g] <= hq[g]; hrm[g] <= hm[g]; hrn[g] <= hn[g];
                end
            end
        end
    end

    // valid / last / err through the NTS - 1 inner tree stages
    // (shift registers of NTS bits; {x, in} drops x's top bit)
    logic [NTS-1:0] vt, lt, et;
    /* verilator lint_off WIDTHTRUNC */
    always_ff @(posedge clk) begin
        if (!rst_n) vt <= '0;
        else if (adv) vt <= {vt, v1};
        if (adv) begin
            lt <= {lt, last1};
            et <= {et, err1};
        end
    end
    /* verilator lint_on WIDTHTRUNC */

    logic                    v2, last2, err2;
    logic signed [ACC_W-1:0] s2_sum, s2_sq;
    logic signed [31:0]      s2_max, s2_min;
    assign v2    = vt[NTS-1];
    assign last2 = lt[NTS-1];
    assign err2  = et[NTS-1];
    always_ff @(posedge clk) begin
        if (adv) begin
            s2_sum <= r_sum;
            s2_sq  <= r_sq;
            s2_max <= r_max;
            s2_min <= r_min;
        end
    end

    // ---------------- S3: row accumulation ----------------
    logic                    first;            // next beat starts a new row
    logic signed [ACC_W-1:0] a_sum, a_sq, n_sum, n_sq;
    logic signed [31:0]      a_max, a_min, n_max, n_min;
    logic                    a_err, n_err;
    always_comb begin
        n_sum = first ? s2_sum : a_sum + s2_sum;
        n_sq  = first ? s2_sq  : a_sq + s2_sq;
        n_max = (first || s2_max > a_max) ? s2_max : a_max;
        n_min = (first || s2_min < a_min) ? s2_min : a_min;
        n_err = first ? err2 : (a_err | err2);
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            first     <= 1'b1;
            out_valid <= 1'b0;
        end else begin
            if (out_valid && out_ready) out_valid <= 1'b0;
            if (adv && v2) begin
                a_sum <= n_sum;
                a_sq  <= n_sq;
                a_max <= n_max;
                a_min <= n_min;
                a_err <= n_err;
                first <= last2;
                if (last2) begin
                    out_valid  <= 1'b1;
                    out_sum    <= n_sum;
                    out_sumsq  <= n_sq;
                    out_max    <= n_max;
                    out_min    <= n_min;
                    out_sq_err <= n_err;
                end
            end
        end
    end
endmodule
