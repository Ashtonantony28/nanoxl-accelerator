// systolic_array.sv -- N x N weight-stationary INT8 systolic array (contract C3 core level).
//
// Orientation (C1): W [K,N] is stationary, PE(r,c) holds W[r,c] (r = K index, c = output
// column). An activation row A[m, 0..N-1] enters on in_a; element r is skewed by r cycles
// and enters PE row r from the left; partial sums flow down; column c is deskewed by
// N-1-c cycles so that the whole INT32 output row leaves on one cycle.
//
// Timing (cycle numbers relative to the cycle T in which in_valid is high):
//   PE(r,c) multiplies element r of that row during cycle T + r + c;
//   out_valid/out_psum/out_tag for that row are presented in cycle T + LAT, LAT = 2N-1.
// The pipeline never stalls; flow control is done by credits in array_ctrl.
//
// Weight load: one K-row per cycle. w_wr_en writes w_wr_data into row w_wr_row of bank
// w_wr_bank; the new value is used from the next cycle on. w_wr_int4=1 means w_wr_data
// holds N packed INT4 weights in its low N*4 bits (C1b nibble order: element c in bits
// [4c+3:4c]), each sign-extended to 8 bits here.
// Bank-reuse rule (checked in simulation by hazard_seen): a write to (row r, bank b) in
// cycle W is legal only if no valid activation tagged b sits at PE(r,c), c < N-1, in W;
// equivalently W >= T_last(b) + r + N - 1.
module systolic_array #(
    parameter int N     = 8,
    parameter int TAG_W = 1,
    parameter int AW    = 8,
    parameter int WW    = 8,
    parameter int ACCW  = 32
) (
    input  logic                  clk,
    input  logic                  rst_n,
    // activation row
    input  logic                  in_valid,
    input  logic [N*AW-1:0]       in_a,
    input  logic                  in_sel,
    input  logic [TAG_W-1:0]      in_tag,
    // weight row write
    input  logic                  w_wr_en,
    input  logic                  w_wr_bank,
    input  logic [$clog2(N)-1:0]  w_wr_row,
    input  logic                  w_wr_int4,
    input  logic [N*WW-1:0]       w_wr_data,
    // output row
    output logic                  out_valid,
    output logic [N*ACCW-1:0]     out_psum,
    output logic [TAG_W-1:0]      out_tag,
    // simulation-only sticky bank-reuse hazard flag (0 in synthesis)
    output logic                  hazard_seen
);
    localparam int LAT = 2 * N - 1;
    localparam int LW  = AW + 2;  // {v, sel, a} per skew lane

    // ---- input skew -------------------------------------------------------------------
    logic [N*LW-1:0] sk_in, sk_out;
    for (genvar r = 0; r < N; r++) begin : g_skin
        assign sk_in[r*LW +: LW] = {in_valid, in_sel, in_valid ? in_a[r*AW +: AW] : {AW{1'b0}}};
    end
    skew #(.LANES(N), .W(LW), .REVERSE(0)) u_skew (
        .clk(clk), .rst_n(rst_n), .din(sk_in), .dout(sk_out));

    // ---- weight unpack (INT4 sign extension) -------------------------------------------
    logic [N*WW-1:0] w_row;
    for (genvar c = 0; c < N; c++) begin : g_wunpack
        assign w_row[c*WW +: WW] = w_wr_int4 ? {{(WW-4){w_wr_data[c*4+3]}}, w_wr_data[c*4 +: 4]}
                                             : w_wr_data[c*WW +: WW];
    end

    // ---- PE grid -----------------------------------------------------------------------
    // a/v/sel buses: index (r*(N+1)+c) is the input of PE(r,c); c = N is the dangling output.
    // psum bus: index (r*N+c) is the psum input of PE(r,c); r = N is the column output.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [N*(N+1)*AW-1:0]   a_bus;  // column N outputs dangle
    /* verilator lint_on UNUSEDSIGNAL */
    wire [N*(N+1)-1:0]      v_bus, s_bus;
    wire [(N+1)*N*ACCW-1:0] p_bus;

    for (genvar c = 0; c < N; c++) begin : g_ptop
        assign p_bus[c*ACCW +: ACCW] = '0;
    end

    for (genvar r = 0; r < N; r++) begin : g_row
        assign a_bus[(r*(N+1))*AW +: AW] = sk_out[r*LW +: AW];
        assign s_bus[r*(N+1)]            = sk_out[r*LW + AW];
        assign v_bus[r*(N+1)]            = sk_out[r*LW + AW + 1];
        wire row_we = w_wr_en && (w_wr_row == ($clog2(N))'(r));
        for (genvar c = 0; c < N; c++) begin : g_col
            localparam int IH = r * (N + 1) + c;  // horizontal bus index of PE input
            localparam int IV = r * N + c;        // vertical bus index of PE input
            pe #(.AW(AW), .WW(WW), .ACCW(ACCW)) u_pe (
                .clk      (clk),
                .rst_n    (rst_n),
                .a_in     (a_bus[IH*AW +: AW]),
                .a_v_in   (v_bus[IH]),
                .a_sel_in (s_bus[IH]),
                .a_out    (a_bus[(IH+1)*AW +: AW]),
                .a_v_out  (v_bus[IH+1]),
                .a_sel_out(s_bus[IH+1]),
                .psum_in  (p_bus[IV*ACCW +: ACCW]),
                .psum_out (p_bus[(IV+N)*ACCW +: ACCW]),
                .w_wr_en  (row_we),
                .w_wr_bank(w_wr_bank),
                .w_wr_data(w_row[c*WW +: WW]));
        end
    end

    // ---- output deskew, valid/tag delay ---------------------------------------------------
    skew #(.LANES(N), .W(ACCW), .REVERSE(1)) u_deskew (
        .clk(clk), .rst_n(rst_n), .din(p_bus[N*N*ACCW +: N*ACCW]), .dout(out_psum));

    delay_pipe #(.W(TAG_W + 1), .D(LAT)) u_vtag (
        .clk(clk), .rst_n(rst_n), .din({in_valid, in_tag}), .dout({out_valid, out_tag}));

    // ---- simulation-only bank-reuse hazard checker ----------------------------------------
`ifndef SYNTHESIS
    logic hz_now;
    always_comb begin
        hz_now = 1'b0;
        for (int hr = 0; hr < N; hr++)
            for (int hc = 0; hc < N - 1; hc++)
                if (w_wr_en && 32'(w_wr_row) == hr && v_bus[hr*(N+1)+hc] && s_bus[hr*(N+1)+hc] == w_wr_bank)
                    hz_now = 1'b1;
    end
    always_ff @(posedge clk) begin
        if (!rst_n) hazard_seen <= 1'b0;
        else if (hz_now) hazard_seen <= 1'b1;
    end
`else
    assign hazard_seen = 1'b0;
`endif
endmodule
