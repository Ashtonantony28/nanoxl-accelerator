// sat_counter.sv -- sticky saturation-event counter (contract C2: saturation must be countable).
//
// Each cycle with en=1 adds popcount(inc) to count. The counter saturates at 2^W-1
// (it never wraps). sticky goes high on the first event and stays high until clr.
// clr discards the old value but not the events of the same cycle:
// on a clr cycle count <= popcount(inc & {N{en}}).
module sat_counter #(
    parameter int N = 4,
    parameter int W = 32
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         clr,
    input  logic         en,
    input  logic [N-1:0] inc,
    output logic [W-1:0] count,
    output logic         sticky
);
    localparam int PW = $clog2(N + 1);
    logic [PW-1:0] pc;
    logic [W:0]    sum;
    always_comb begin
        pc = '0;
        for (int i = 0; i < N; i++) pc = pc + {{(PW-1){1'b0}}, inc[i] & en};
        sum = (clr ? {(W+1){1'b0}} : {1'b0, count}) + {{(W+1-PW){1'b0}}, pc};
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            count  <= '0;
            sticky <= 1'b0;
        end else begin
            count  <= sum[W] ? {W{1'b1}} : sum[W-1:0];
            sticky <= (clr ? 1'b0 : sticky) | (pc != '0);
        end
    end
endmodule
