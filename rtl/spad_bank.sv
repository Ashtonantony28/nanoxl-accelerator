// spad_bank.sv -- one scratchpad bank: DEPTH lines of 16 bytes, simple dual port
// (one write with byte enables + one read per cycle), registered read (latency 1 + OREG).
// A read and a write of the same line at the same edge return the old line (read-first);
// the write is visible to every read issued at a later edge. No reset on the array.
// BRAM/URAM-inferable (SDP, byte-write enables, 128-bit data).
//
// OREG = 1: a second, unconditional output register (the URAM288 OREG / RAMB36 DOB_REG
// stage), read latency 2; the array is still read at the edge of the request, so the
// read-first / visibility rule above is unchanged.
// RAM_STYLE selects the primitive explicitly (Vivado and Yosys `ram_style` attribute):
// 0 = tool's choice, 1 = "block" (RAMB36/18), 2 = "ultra" (URAM288), 3 = "distributed".
module spad_bank #(
    parameter int DEPTH     = 256,
    parameter int OREG      = 0,
    parameter int RAM_STYLE = 0
) (
    input  logic                     clk,
    input  logic                     we,
    input  logic [$clog2(DEPTH)-1:0] waddr,
    input  logic [127:0]             wdata,
    input  logic [15:0]              wstrb,
    input  logic                     re,
    input  logic [$clog2(DEPTH)-1:0] raddr,
    output logic [127:0]             rdata
);
    logic [127:0] rd1;

    if (RAM_STYLE == 1) begin : g_block
        (* ram_style = "block" *) logic [127:0] mem [0:DEPTH-1];
        always_ff @(posedge clk) begin
            if (re) rd1 <= mem[raddr];
        end
        always_ff @(posedge clk) begin
            if (we) begin
                for (int b = 0; b < 16; b++)
                    if (wstrb[b]) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
            end
        end
`ifndef SYNTHESIS
        initial for (int i = 0; i < DEPTH; i++) mem[i] = '0;
`endif
    end else if (RAM_STYLE == 2) begin : g_ultra
        (* ram_style = "ultra" *) logic [127:0] mem [0:DEPTH-1];
        always_ff @(posedge clk) begin
            if (re) rd1 <= mem[raddr];
        end
        always_ff @(posedge clk) begin
            if (we) begin
                for (int b = 0; b < 16; b++)
                    if (wstrb[b]) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
            end
        end
`ifndef SYNTHESIS
        initial for (int i = 0; i < DEPTH; i++) mem[i] = '0;
`endif
    end else if (RAM_STYLE == 3) begin : g_dist
        (* ram_style = "distributed" *) logic [127:0] mem [0:DEPTH-1];
        always_ff @(posedge clk) begin
            if (re) rd1 <= mem[raddr];
        end
        always_ff @(posedge clk) begin
            if (we) begin
                for (int b = 0; b < 16; b++)
                    if (wstrb[b]) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
            end
        end
`ifndef SYNTHESIS
        initial for (int i = 0; i < DEPTH; i++) mem[i] = '0;
`endif
    end else begin : g_auto
        logic [127:0] mem [0:DEPTH-1];
        always_ff @(posedge clk) begin
            if (re) rd1 <= mem[raddr];
        end
        always_ff @(posedge clk) begin
            if (we) begin
                for (int b = 0; b < 16; b++)
                    if (wstrb[b]) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
            end
        end
`ifndef SYNTHESIS
        // zero-initialised so that a read of a never-written line is defined in every simulator
        initial for (int i = 0; i < DEPTH; i++) mem[i] = '0;
`endif
    end

    if (OREG != 0) begin : g_oreg
        logic [127:0] rd2;
        always_ff @(posedge clk) rd2 <= rd1;
        assign rdata = rd2;
    end else begin : g_noreg
        assign rdata = rd1;
    end
endmodule
