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
