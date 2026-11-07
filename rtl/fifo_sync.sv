// fifo_sync.sv -- generic single-clock first-word-fall-through FIFO (valid/ready).
//
// DEPTH must be a power of two (>= 2). Storage is a register array read combinationally at the
// read pointer (LUTRAM / FFs on an FPGA; for large BRAM-backed FIFOs see spad_fifo in
// scratchpad.sv). One push and one pop per cycle; a push into a full FIFO is impossible because
// in_ready = !full, and a pop of an empty FIFO is impossible because out_valid = !empty.
// in_ready does not depend on out_ready (no combinational path through the FIFO).
//
// Formal (yosys -formal, see formal/): count/pointer consistency, no overflow/underflow, and
// data order + integrity of one arbitrarily chosen ("tagged") element.
module fifo_sync #(
    parameter int W     = 8,
    parameter int DEPTH = 4
) (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [W-1:0]               in_data,
    output logic                       out_valid,
    input  logic                       out_ready,
    output logic [W-1:0]               out_data,
    output logic [$clog2(DEPTH+1)-1:0] count
);
    localparam int AB = $clog2(DEPTH);
    localparam int CB = $clog2(DEPTH + 1);

    logic [W-1:0]  mem [0:DEPTH-1];
    logic [AB:0]   wr_ptr, rd_ptr;   // one extra wrap bit
    logic          push, pop;

    assign count     = CB'(wr_ptr - rd_ptr);
    assign in_ready  = (count != CB'(DEPTH));
    assign out_valid = (count != '0);
    assign out_data  = mem[rd_ptr[AB-1:0]];
    assign push      = in_valid && in_ready;
    assign pop       = out_valid && out_ready;

    always_ff @(posedge clk) begin
        if (push) mem[wr_ptr[AB-1:0]] <= in_data;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
        end else begin
            if (push) wr_ptr <= wr_ptr + 1'b1;
            if (pop)  rd_ptr <= rd_ptr + 1'b1;
        end
    end

`ifdef FORMAL
    // ---- properties (checked by formal/fifo.ys + yosys-smtbmc, BMC and k-induction) ----
    logic f_init = 1'b1;
    always_ff @(posedge clk) f_init <= 1'b0;
    always_comb if (f_init) assume (!rst_n);

    // tagged element: (* anyseq *) chooses which push to follow
    (* anyseq *) logic f_pick;
    logic          f_tagged;
    logic [W-1:0]  f_tdata;
    logic [AB:0]   f_ahead;     // elements in front of the tagged one
    logic [AB:0]   f_tpos;
    assign f_tpos = rd_ptr + f_ahead;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            f_tagged <= 1'b0;
            f_ahead  <= '0;
        end else begin
            if (f_tagged) begin
                if (pop) begin
                    if (f_ahead == '0) f_tagged <= 1'b0;
                    else               f_ahead  <= f_ahead - 1'b1;
                end
            end else if (push && f_pick) begin
                f_tagged <= 1'b1;
                f_tdata  <= in_data;
                f_ahead  <= (AB+1)'(count) - (AB+1)'(pop);
            end
        end
    end

    always_comb begin
        if (!f_init && rst_n) begin
            assert (count <= CB'(DEPTH));                         // never overflows
            assert ((wr_ptr - rd_ptr) <= (AB+1)'(DEPTH));
            assert (out_valid == (count != 0));                    // never pops when empty
            assert (in_ready == (count < CB'(DEPTH)));             // never pushes when full
            if (f_tagged) begin
                assert ((AB+1)'(f_ahead) < (AB+1)'(count));         // still inside the FIFO
                assert (mem[f_tpos[AB-1:0]] == f_tdata);            // data unchanged
                if (f_ahead == '0) assert (out_data == f_tdata);  // leaves in order
            end
        end
    end
`endif
endmodule
