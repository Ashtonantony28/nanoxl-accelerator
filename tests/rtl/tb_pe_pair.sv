// tb_pe_pair.sv -- self-checking exhaustive test of rtl/pe.sv::pe_pair (WP486 2-MAC packing).
// One case per cycle, pipelined: case k = {wB, wA, a} (24 bits) for k = 0, STEP, 2*STEP, ...
// below 2^24 (STEP = 1: all int8 x int8 x int8 combinations).
//   cycle t:   write wA, wB of case k(t) into bank t&1
//   cycle t+1: activation a of k with sel = that bank, psum_in_a = fa(k)
//   cycle t+2: psum_in_b = fb(k); psum_out_a must be fa(k) + a*wA
//   cycle t+3: psum_out_b must be fb(k) + a*wB
// Both psums are full-range 32-bit values (wrapping adds, as in pe). The cocotb wrapper drives
// clk / rst_n, waits for done and reads errors / checked.
module tb_pe_pair #(
    parameter int STEP = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    output logic        done,
    output logic [31:0] errors,
    output logic [31:0] checked
);
    localparam int unsigned LAST = (1 << 24) - 1;

    function automatic logic [31:0] fa(input logic [23:0] k);
        fa = {8'd0, k} * 32'h9E37_79B1;
    endfunction
    function automatic logic [31:0] fb(input logic [23:0] k);
        fb = ({8'd0, k} * 32'h85EB_CA6B) ^ 32'h1234_5679;
    endfunction

    logic [24:0] cnt;                     // next case index (bit 24: past the end)
    logic [23:0] k1, k2, k3;
    logic        v1, v2, v3, par1, par0;

    logic signed [7:0]  a_in, w_a, w_b;
    logic signed [31:0] pia, pib, poa, pob;
    logic               a_v, sel;

    assign par0 = cnt[0] ^ cnt[1];         // any alternating-ish bank pattern works
    assign w_a  = cnt[15:8];
    assign w_b  = cnt[23:16];
    assign a_in = v1 ? k1[7:0] : 8'sd0;
    assign a_v  = v1;
    assign sel  = par1;
    assign pia  = fa(k1);
    assign pib  = fb(k2);

    pe_pair #(.ACCW(32)) dut (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in), .a_v_in(a_v), .a_sel_in(sel),
        .a_out(), .a_v_out(), .a_sel_out(),
        .psum_in_a(pia), .psum_out_a(poa), .psum_in_b(pib), .psum_out_b(pob),
        .w_wr_en(!cnt[24] && rst_n), .w_wr_bank(par0), .w_wr_a(w_a), .w_wr_b(w_b));

    logic signed [31:0] ea, eb, a2, wa2, a3, wb3;
    assign a2  = {{24{k2[7]}},  k2[7:0]};
    assign wa2 = {{24{k2[15]}}, k2[15:8]};
    assign a3  = {{24{k3[7]}},  k3[7:0]};
    assign wb3 = {{24{k3[23]}}, k3[23:16]};
    assign ea  = fa(k2) + a2 * wa2;
    assign eb  = fb(k3) + a3 * wb3;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cnt <= '0; v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
            errors <= '0; checked <= '0; done <= 1'b0;
        end else begin
            if (!cnt[24]) cnt <= (cnt + 25'(STEP) > 25'(LAST)) ? 25'h100_0000 : cnt + 25'(STEP);
            v1 <= !cnt[24]; k1 <= cnt[23:0]; par1 <= par0;
            v2 <= v1; k2 <= k1;
            v3 <= v2; k3 <= k2;
            // psum_out_a now holds case k2's column-c result, psum_out_b case k3's column c+1
            if (v2 && poa != ea) errors <= errors + 1;
            if (v3 && pob != eb) errors <= errors + 1;
            if (v3) checked <= checked + 1;
            if (cnt[24] && !v1 && !v2 && !v3) done <= 1'b1;
        end
    end
endmodule
