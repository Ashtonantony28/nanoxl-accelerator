// spad_bank.sv -- one scratchpad bank: DEPTH lines of 16 bytes, simple dual port
// (one write with byte enables + one read per cycle), registered read (latency 1).
// A read and a write of the same line at the same edge return the old line (read-first);
// the write is visible to every read issued at a later edge. No reset on the array.
// BRAM/URAM-inferable (SDP, byte-write enables, 128-bit data).
module spad_bank #(
    parameter int DEPTH = 256
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
    logic [127:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (re) rdata <= mem[raddr];
    end

    always_ff @(posedge clk) begin
        if (we) begin
            for (int b = 0; b < 16; b++)
                if (wstrb[b]) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
        end
    end

`ifndef SYNTHESIS
    // zero-initialised so that a read of a never-written line is defined in every simulator
    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end
`endif
endmodule
