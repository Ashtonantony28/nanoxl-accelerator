// pe.sv -- weight-stationary processing element (contract C1 / C3, docs/array.md).
//
// One INT8 x INT8 multiply-accumulate per cycle:
//   psum_out <= psum_in + a_in * w[a_sel_in]        (exact, INT32)
//   a_out    <= a_in   (activation passes right, one cycle per PE)
// Two weight registers (banks 0/1). The bank is chosen per activation by a_sel_in, which
// travels with the activation wavefront, so a tile switch needs no global swap pulse:
// the loader writes bank !b while activations tagged b are still computing.
// a_v_in/a_v_out carry the activation valid bit; it does not gate the datapath and is used
// only by the simulation hazard checker in systolic_array (synthesis removes it).
//
// The multiply is isolated in pe_mul below so that AMD WP486 2-MAC packing (two weights
// sharing one activation in one DSP48E2) is a local change: a pe2 cell replaces two
// horizontally adjacent PEs, see docs/array.md "DSP packing".
module pe #(
    parameter int AW   = 8,
    parameter int WW   = 8,
    parameter int ACCW = 32
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic signed [AW-1:0]   a_in,
    input  logic                   a_v_in,
    input  logic                   a_sel_in,
    output logic signed [AW-1:0]   a_out,
    output logic                   a_v_out,
    output logic                   a_sel_out,
    input  logic signed [ACCW-1:0] psum_in,
    output logic signed [ACCW-1:0] psum_out,
    input  logic                   w_wr_en,
    input  logic                   w_wr_bank,
    input  logic signed [WW-1:0]   w_wr_data
);
    logic signed [WW-1:0]    w0, w1, w_cur;
    logic signed [AW+WW-1:0] prod;

    assign w_cur = a_sel_in ? w1 : w0;

    // pe_mul: the only arithmetic that a DSP-packing change touches.
    always_comb prod = a_in * w_cur;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out     <= '0;
            a_v_out   <= 1'b0;
            a_sel_out <= 1'b0;
            psum_out  <= '0;
            w0        <= '0;
            w1        <= '0;
        end else begin
            a_out     <= a_in;
            a_v_out   <= a_v_in;
            a_sel_out <= a_sel_in;
            psum_out  <= psum_in + {{(ACCW-AW-WW){prod[AW+WW-1]}}, prod};
            if (w_wr_en) begin
                if (w_wr_bank) w1 <= w_wr_data;
                else           w0 <= w_wr_data;
            end
        end
    end
endmodule

// pe_pair -- two horizontally adjacent PEs (columns c, c+1 of one row) in one DSP: AMD WP486
// INT8 2-MAC packing, used by systolic_array when PACK = 1 (docs/array.md "DSP packing").
//
// Both weights multiply the same activation in the same cycle T (the cycle PE(r,c) would):
//   P  = (wA * 2^18 + wB) * a            one 27x18 multiply (pre-adder D + A, B = a)
//   lo = P[17:0] as signed 18 bits        = a * wB  (|a*wB| <= 2^14 < 2^17)
//   hi = P[35:18] + P[17]                 = a * wA  (correction: lo's sign borrows 1 from hi)
// Column c:   psum_out_a <= psum_in_a + hi       (end of cycle T, as an unpacked PE)
// Column c+1: lo is registered and added one cycle later, psum_out_b <= psum_in_b + lo_r
//             (end of cycle T+1, as an unpacked PE: its psum input arrives one cycle later).
// The activation leaves after 2 cycles (to the next pair at T+2). Both columns' weights are
// written in the same cycle (the column-c write wavefront: the array's write skew delays
// column c+1 by c instead of c+1), so the bank reuse / use rules of systolic_array are those of
// the unpacked array. Bit-identical to two pe cells; tests/rtl/test_pe.py checks the packing
// exhaustively over all a x wA x wB (Verilator; a strided 1/13 subset on Icarus).
module pe_pair #(
    parameter int ACCW = 32
) (
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic signed [7:0]      a_in,
    input  logic                   a_v_in,
    input  logic                   a_sel_in,
    output logic signed [7:0]      a_out,
    output logic                   a_v_out,
    output logic                   a_sel_out,
    input  logic signed [ACCW-1:0] psum_in_a,
    output logic signed [ACCW-1:0] psum_out_a,
    input  logic signed [ACCW-1:0] psum_in_b,
    output logic signed [ACCW-1:0] psum_out_b,
    input  logic                   w_wr_en,
    input  logic                   w_wr_bank,
    input  logic signed [7:0]      w_wr_a,
    input  logic signed [7:0]      w_wr_b
);
    logic signed [7:0]  wa0, wa1, wb0, wb1, wa, wb;
    logic signed [26:0] pre;
    logic signed [35:0] p;
    logic signed [17:0] lo, lo_r, hi;
    logic signed [7:0]  a_d;
    logic               v_d, s_d;

    assign wa  = a_sel_in ? wa1 : wa0;
    assign wb  = a_sel_in ? wb1 : wb0;
    assign pre = {wa[7], wa, 18'd0} + {{19{wb[7]}}, wb};   // wA * 2^18 + wB (27 bits)
    assign p   = pre * a_in;
    assign lo  = p[17:0];
    assign hi  = p[35:18] + {17'd0, p[17]};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_d <= '0; v_d <= 1'b0; s_d <= 1'b0;
            a_out <= '0; a_v_out <= 1'b0; a_sel_out <= 1'b0;
            psum_out_a <= '0; psum_out_b <= '0; lo_r <= '0;
            wa0 <= '0; wa1 <= '0; wb0 <= '0; wb1 <= '0;
        end else begin
            a_d <= a_in;  v_d <= a_v_in;  s_d <= a_sel_in;
            a_out <= a_d; a_v_out <= v_d; a_sel_out <= s_d;
            psum_out_a <= psum_in_a + {{(ACCW-18){hi[17]}}, hi};
            lo_r       <= lo;
            psum_out_b <= psum_in_b + {{(ACCW-18){lo_r[17]}}, lo_r};
            if (w_wr_en) begin
                if (w_wr_bank) begin wa1 <= w_wr_a; wb1 <= w_wr_b; end
                else           begin wa0 <= w_wr_a; wb0 <= w_wr_b; end
            end
        end
    end
endmodule
