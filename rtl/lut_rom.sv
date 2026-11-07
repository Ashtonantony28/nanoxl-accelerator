// lut_rom.sv -- runtime-loadable lookup table for the SFUs (contract C5: LUT contents are
// data from the model bundle; the RTL loads them, it never computes them).
//
// Write port: one write of WN consecutive entries per cycle (WN a power of two), so a
// 16-byte PBUF line (docs/isa.md §3) lands in one cycle: entry {waddr, k} <= wdata[k*DW +: DW],
// little-endian like the PBUF bytes. Read port: synchronous with clock enable, on an edge
// with en = 1, rdata <= mem[raddr].
// Instantiated once per consumer lane (the FPGA mapping replicates small tables into
// LUTRAM per read port anyway). Write while the consumer is idle (docs/sfu.md §2); a read
// in the same cycle as a write to the same entry returns the old entry. Contents are
// undefined until written (no reset on the storage).
module lut_rom #(
    parameter int DW = 16,
    parameter int AW = 8,
    parameter int WN = 1
) (
    input  logic                      clk,
    input  logic                      we,
    input  logic [AW-$clog2(WN)-1:0]  waddr,      // line address = entry address / WN
    input  logic [WN*DW-1:0]          wdata,
    input  logic                      en,
    input  logic [AW-1:0]             raddr,
    output logic [DW-1:0]             rdata
);
    logic [DW-1:0] mem [1 << AW];
    always_ff @(posedge clk) begin
        if (we)
            for (int k = 0; k < WN; k++)
                mem[AW'(waddr) * AW'(WN) + AW'(k)] <= wdata[k*DW +: DW];
    end
    always_ff @(posedge clk) begin
        if (en) rdata <= mem[raddr];
    end
endmodule
