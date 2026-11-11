// systolic_array.sv -- N x N weight-stationary INT8 systolic array (contract C3 core level).
//
// Orientation (C1): W [K,N] is stationary, PE(r,c) holds W[r,c] (r = K index, c = output
// column). An activation row A[m, 0..N-1] enters on in_a; element r is skewed by r cycles
// and enters PE row r from the left; partial sums flow down; column c is deskewed by
// N-1-c cycles so that the whole INT32 output row leaves on one cycle.
//
// Timing (cycle numbers relative to the cycle T in which in_valid is high):
//   PE(r,c) multiplies element r of that row during cycle T + r + c (PACK = 1 with an even N
//   and 8-bit operands: the pe_pair of columns 2p and 2p+1 multiplies both during T + r + 2p,
//   and column 2p+1's product joins its psum one cycle later, so the output timing is the same);
//   out_valid/out_psum/out_tag for that row are presented in cycle T + LAT, LAT = 2N-1.
// The pipeline never stalls; flow control is done by credits in array_ctrl.
//
// Weight load: one K-row per cycle. w_wr_en in cycle W writes w_wr_data into row w_wr_row
// of bank w_wr_bank. The write travels along the row with the activations: PE(r,c) is
// written at the end of cycle W + c (a skewed write wavefront); with packing (PACK = 1, the
// default) column 2p+1 is written with column 2p, at the end of cycle W + 2p, the cycle its
// pe_pair multiplies. The bank rules below hold unchanged in both cases, because the packed
// column's write and read move one cycle earlier together. w_wr_int4=1 means
// w_wr_data holds N packed INT4 weights in its low N*4 bits (C1b nibble order: element c
// in bits [4c+3:4c]), each sign-extended to 8 bits here.
//
// Bank rules, both relative to the activation wavefront of the same row:
//   reuse: a write of (row r, bank b) in cycle W must not overtake an older activation of
//          bank b in row r:  W >= T_last(b) + r  (T_last: last in_valid cycle using b);
//   use:   an activation of the new tile must not overtake the write:  T + r >= W_r + 1.
// Both are checked in simulation (hazard_seen) using a generation bit: in_gen/w_wr_gen
// identify the tile; any non-zero activation that multiplies with a weight of another
// generation (or never written since reset) sets hazard_seen. Zero activations (ragged-K
// padding) are exempt because their product does not depend on the weight.
module systolic_array #(
    parameter int N     = 8,
    parameter int TAG_W = 1,
    parameter int AW    = 8,
    parameter int WW    = 8,
    parameter int ACCW  = 32,
    parameter int PACK  = 1     // 1: WP486 INT8 2-MAC packing, one DSP per PE pair (pe_pair)
) (
    input  logic                  clk,
    input  logic                  rst_n,
    // activation row
    input  logic                  in_valid,
    input  logic [N*AW-1:0]       in_a,
    input  logic                  in_sel,
    input  logic                  in_gen,     // simulation checker only
    input  logic [TAG_W-1:0]      in_tag,
    // weight row write
    input  logic                  w_wr_en,
    input  logic                  w_wr_bank,
    input  logic                  w_wr_gen,   // simulation checker only
    input  logic [$clog2(N)-1:0]  w_wr_row,
    input  logic                  w_wr_int4,
    input  logic [N*WW-1:0]       w_wr_data,
    // output row
    output logic                  out_valid,
    output logic [N*ACCW-1:0]     out_psum,
    output logic [TAG_W-1:0]      out_tag,
    // simulation-only sticky hazard flag (0 in synthesis)
    output logic                  hazard_seen
);
    localparam int LAT = 2 * N - 1;
    localparam int RB  = $clog2(N);
    localparam int LW  = AW + 3;        // activation skew lane {gen, v, sel, a}
    localparam int WL  = WW + RB + 3;   // weight-write skew lane {gen, en, bank, row, w}
    // PACK needs 8-bit operands and an even N; otherwise the plain PE grid is built
    localparam int PK  = (PACK != 0 && N % 2 == 0 && AW == 8 && WW == 8) ? 1 : 0;

    // ---- input skew -------------------------------------------------------------------
    logic [N*LW-1:0] sk_in, sk_out;
    for (genvar r = 0; r < N; r++) begin : g_skin
        assign sk_in[r*LW +: LW] = {in_gen, in_valid, in_sel,
                                    in_valid ? in_a[r*AW +: AW] : {AW{1'b0}}};
    end
    skew #(.LANES(N), .W(LW), .REVERSE(0)) u_skew (
        .clk(clk), .rst_n(rst_n), .din(sk_in), .dout(sk_out));

    // ---- weight unpack (INT4 sign extension) and write skew ------------------------------
    logic [N*WL-1:0] wk_in, wk_out;
    for (genvar c = 0; c < N; c++) begin : g_wunpack
        logic [WW-1:0] wv;
        assign wv = w_wr_int4 ? {{(WW-4){w_wr_data[c*4+3]}}, w_wr_data[c*4 +: 4]}
                              : w_wr_data[c*WW +: WW];
        assign wk_in[c*WL +: WL] = {w_wr_gen, w_wr_en, w_wr_bank, w_wr_row, wv};
    end
    // Column c's write is delayed by c cycles; with PK, column 2p+1 is written together with
    // column 2p (delay 2p), the cycle its pe_pair multiplies both columns.
    if (PK == 0) begin : g_wskew
        skew #(.LANES(N), .W(WL), .REVERSE(0)) u_wskew (
            .clk(clk), .rst_n(rst_n), .din(wk_in), .dout(wk_out));
    end else begin : g_wskew_pk
        for (genvar c = 0; c < N; c++) begin : g_l
            localparam int WD = c - (c % 2);
            if (WD == 0) begin : g_wire
                assign wk_out[c*WL +: WL] = wk_in[c*WL +: WL];
            end else begin : g_dly
                delay_pipe #(.W(WL), .D(WD)) u_d (
                    .clk(clk), .rst_n(rst_n), .din(wk_in[c*WL +: WL]), .dout(wk_out[c*WL +: WL]));
            end
        end
    end

    // ---- PE grid -----------------------------------------------------------------------
    // a/v/sel buses: index (r*(N+1)+c) is the input of PE(r,c); c = N is the dangling output.
    // psum bus: index (r*N+c) is the psum input of PE(r,c); r = N is the column output.
    // (unpacked net arrays, one element per link: much faster in Icarus than one flat bus)
    /* verilator lint_off UNUSEDSIGNAL */
    wire [AW-1:0]   a_bus [0:N*(N+1)-1];  // column N outputs dangle
    wire            v_bus [0:N*(N+1)-1];
    wire            s_bus [0:N*(N+1)-1];
    /* verilator lint_on UNUSEDSIGNAL */
    wire [ACCW-1:0] p_bus [0:(N+1)*N-1];

    for (genvar c = 0; c < N; c++) begin : g_ptop
        assign p_bus[c] = '0;
    end

    for (genvar r = 0; r < N; r++) begin : g_row
        assign a_bus[r*(N+1)]            = sk_out[r*LW +: AW];
        assign s_bus[r*(N+1)]            = sk_out[r*LW + AW];
        assign v_bus[r*(N+1)]            = sk_out[r*LW + AW + 1];
        for (genvar c = 0; c < N; c++) begin : g_col
            localparam int IH = r * (N + 1) + c;  // horizontal bus index of PE input
            localparam int IV = r * N + c;        // vertical bus index of PE input
            // this column's (skewed) write: {gen, en, bank, row, w}
            wire          wc_en   = wk_out[c*WL + WW + RB + 1];
            wire          wc_bank = wk_out[c*WL + WW + RB];
            wire [RB-1:0] wc_row  = wk_out[c*WL + WW +: RB];
            wire          pe_we   = wc_en && (wc_row == RB'(r));
            if (PK == 0) begin : g_pe
                pe #(.AW(AW), .WW(WW), .ACCW(ACCW)) u_pe (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .a_in     (a_bus[IH]),
                    .a_v_in   (v_bus[IH]),
                    .a_sel_in (s_bus[IH]),
                    .a_out    (a_bus[IH+1]),
                    .a_v_out  (v_bus[IH+1]),
                    .a_sel_out(s_bus[IH+1]),
                    .psum_in  (p_bus[IV]),
                    .psum_out (p_bus[IV+N]),
                    .w_wr_en  (pe_we),
                    .w_wr_bank(wc_bank),
                    .w_wr_data(wk_out[c*WL +: WW]));
            end else if (c % 2 == 0) begin : g_pair
                // PE(r,c) and PE(r,c+1) in one pe_pair; both multiply the activation of
                // cycle T + r + c. The column c+1 bus entry mirrors the pair input (what that
                // PE multiplies, for the hazard checker).
                assign a_bus[IH+1] = a_bus[IH];
                assign v_bus[IH+1] = v_bus[IH];
                assign s_bus[IH+1] = s_bus[IH];
                pe_pair #(.ACCW(ACCW)) u_pp (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .a_in      (a_bus[IH]),
                    .a_v_in    (v_bus[IH]),
                    .a_sel_in  (s_bus[IH]),
                    .a_out     (a_bus[IH+2]),
                    .a_v_out   (v_bus[IH+2]),
                    .a_sel_out (s_bus[IH+2]),
                    .psum_in_a (p_bus[IV]),
                    .psum_out_a(p_bus[IV+N]),
                    .psum_in_b (p_bus[IV+1]),
                    .psum_out_b(p_bus[IV+1+N]),
                    .w_wr_en   (pe_we),
                    .w_wr_bank (wc_bank),
                    .w_wr_a    (wk_out[c*WL +: WW]),
                    .w_wr_b    (wk_out[(c+1)*WL +: WW]));
            end
        end
    end

    // ---- output deskew, valid/tag delay ---------------------------------------------------
    logic [N*ACCW-1:0] col_out;
    for (genvar c = 0; c < N; c++) begin : g_pbot
        assign col_out[c*ACCW +: ACCW] = p_bus[N*N+c];
    end
    skew #(.LANES(N), .W(ACCW), .REVERSE(1)) u_deskew (
        .clk(clk), .rst_n(rst_n), .din(col_out), .dout(out_psum));

    delay_pipe #(.W(TAG_W + 1), .D(LAT)) u_vtag (
        .clk(clk), .rst_n(rst_n), .din({in_valid, in_tag}), .dout({out_valid, out_tag}));

    // ---- simulation-only weight-generation checker ----------------------------------------
`ifndef SYNTHESIS
    logic [1:0]   wsh  [0:2*N*N-1];   // per PE per bank: {written, gen}
    logic [N-1:0] gsr  [0:N-1];       // per row: activation gen bit at columns 1..N-1
    function automatic logic act_gen(input int r, input int c);
        int cc;
        cc = (PK != 0) ? c - (c % 2) : c;      // packed: column 2p+1 multiplies with 2p
        act_gen = (cc == 0) ? sk_out[r*LW + AW + 2] : gsr[r][cc-1];
    endfunction

    // evaluated once per clock edge on the values the PEs use in the ending cycle
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            hazard_seen <= 1'b0;
            for (int i = 0; i < 2 * N * N; i++) wsh[i] <= 2'b00;
            for (int i = 0; i < N; i++) gsr[i] <= '0;
        end else begin
            for (int hr = 0; hr < N; hr++)
                for (int hc = 0; hc < N; hc++)
                    if (v_bus[hr*(N+1)+hc] && a_bus[hr*(N+1)+hc] != '0 &&
                        wsh[2*(hr*N+hc) + 32'(s_bus[hr*(N+1)+hc])] != {1'b1, act_gen(hr, hc)})
                        hazard_seen <= 1'b1;
            for (int hr = 0; hr < N; hr++)
                gsr[hr] <= {gsr[hr][N-2:0], sk_out[hr*LW + AW + 2]};
            for (int hc = 0; hc < N; hc++)
                if (wk_out[hc*WL + WW + RB + 1])
                    wsh[2*(32'(wk_out[hc*WL + WW +: RB])*N + hc) + 32'(wk_out[hc*WL + WW + RB])]
                        <= {1'b1, wk_out[hc*WL + WW + RB + 2]};
        end
    end
`else
    assign hazard_seen = 1'b0;
`endif
endmodule
