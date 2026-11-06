// skew.sv -- triangular per-lane delay (input skew / output deskew) and a plain delay pipe.
//
// skew: lane i of din appears on dout after D(i) cycles,
//   REVERSE=0: D(i) = i              (activation skew into the array rows)
//   REVERSE=1: D(i) = LANES-1-i      (psum deskew out of the array columns)
// D=0 lanes are wires. All registers reset to 0 (synchronous, active-low).
module skew #(
    parameter int LANES   = 4,
    parameter int W       = 8,
    parameter int REVERSE = 0
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic [LANES*W-1:0] din,
    output logic [LANES*W-1:0] dout
);
    for (genvar i = 0; i < LANES; i++) begin : g_lane
        localparam int D = (REVERSE != 0) ? (LANES - 1 - i) : i;
        if (D == 0) begin : g_wire
            assign dout[i*W +: W] = din[i*W +: W];
        end else begin : g_dly
            delay_pipe #(.W(W), .D(D)) u_d (
                .clk(clk), .rst_n(rst_n), .din(din[i*W +: W]), .dout(dout[i*W +: W]));
        end
    end
endmodule

// delay_pipe: dout = din delayed by D >= 1 cycles, registers reset to 0.
module delay_pipe #(
    parameter int W = 8,
    parameter int D = 1
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic [W-1:0] din,
    output logic [W-1:0] dout
);
    logic [W-1:0] sr [0:D-1];
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int j = 0; j < D; j++) sr[j] <= '0;
        end else begin
            sr[0] <= din;
            for (int j = 1; j < D; j++) sr[j] <= sr[j-1];
        end
    end
    assign dout = sr[D-1];
endmodule
