// lut_rom.sv -- runtime-loadable lookup table for the SFUs (contract C5: LUT contents are
// data from the model bundle; the RTL loads them, it never computes them).
//
// Write port: one write of WN consecutive entries per cycle (WN a power of two), so a
// 16-byte PBUF line (docs/isa.md §3) lands in one cycle: entry {waddr, k} <= wdata[k*DW +: DW],
// little-endian like the PBUF bytes. Read port: synchronous with clock enable, on an edge
// with en = 1, rdata <= mem[raddr].
//
// Storage is line-organized (docs/fpga.md §2.3 C1): 2^AW / WN lines of WN*DW bits with one
// write port writing one whole line and one read port reading one whole line; the entry is
// selected from that line (a WN:1 mux) in front of the rdata register. A 1W/1R memory with
// an asynchronous read maps to LUTRAM (RAM32M16 / RAM64M8) instead of 2^AW*DW flip-flops
// plus a 2^AW:1 read mux; selecting before the register keeps the flip-flops at DW (a
// registered line read would cost WN*DW + log2(WN) and lets Yosys push the GELU tables into
// BRAM). The observable behaviour is unchanged.
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
    localparam int SW = $clog2(WN);              // entry-in-line index width
    localparam int LA = AW - SW;                 // line address width

    logic [WN*DW-1:0] mem [1 << LA];
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    logic [WN*DW-1:0] line_a;
    logic [DW-1:0]    ent [WN];
    genvar k;
    generate
        for (k = 0; k < WN; k++) begin : g_ent
            assign ent[k] = line_a[k*DW +: DW];
        end
        if (WN == 1) begin : g_w1
            assign line_a = mem[raddr];
            always_ff @(posedge clk) begin
                if (en) rdata <= ent[0];
            end
        end else begin : g_wn
            assign line_a = mem[raddr[AW-1:SW]];
            always_ff @(posedge clk) begin
                if (en) rdata <= ent[raddr[SW-1:0]];
            end
        end
    endgenerate
endmodule
