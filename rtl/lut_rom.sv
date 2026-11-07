// lut_rom.sv -- runtime-loadable lookup table for the SFUs (contract C5: LUT contents are
// data from the model bundle; the RTL loads them, it never computes them).
//
// One write port (we/waddr/wdata, a write lands on the rising edge) and one synchronous
// read port with a clock enable: on an edge with en = 1, rdata <= mem[raddr].
// Instantiated once per consumer lane (the FPGA mapping replicates small tables into
// LUTRAM per read port anyway). Writes are meant to happen while the consumer is idle
// (docs/sfu.md §2); a read in the same cycle as a write to the same address returns the
// old entry. Contents are undefined until written (no reset on the storage).
module lut_rom #(
    parameter int DW = 16,
    parameter int AW = 8
) (
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] waddr,
    input  logic [DW-1:0] wdata,
    input  logic          en,
    input  logic [AW-1:0] raddr,
    output logic [DW-1:0] rdata
);
    logic [DW-1:0] mem [1 << AW];
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end
    always_ff @(posedge clk) begin
        if (en) rdata <= mem[raddr];
    end
endmodule
