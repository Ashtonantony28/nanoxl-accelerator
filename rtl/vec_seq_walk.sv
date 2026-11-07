// vec_seq_walk.sv -- line-address walker of the VEC engine (docs/vector-unit.md §9).
//
// Walks the 16-byte lines of a two-level row set: row (o, i) for o < no, i < ni starts at
// line  base + o*so + i*si  and holds ceil(len * 2^ebl / 16) lines; the final row (o = no-1,
// i = ni-1) holds len_last elements instead of len. Rows are visited o-major, i-minor and
// the lines of a row in increasing order, P lines per beat: a beat starts at the row's first
// line (any line) or P lines after the previous beat of the row, and the row's final beat
// covers the remaining 1..P lines (lines past the row end belong to the beat but not to the
// row). `step` advances by one beat; `act` is high while a beat remains (`addr` is its first
// line). Addresses wrap at 2^AW lines. P = 1: one line per step.
module vec_seq_walk #(
    parameter int AW = 17,
    parameter int P  = 1
) (
    input  logic          clk,
    input  logic          rst_n,          // includes the engine clear
    input  logic          start,          // load the descriptor (ignored while act)
    input  logic [AW-1:0] base,
    input  logic [AW-1:0] so,
    input  logic [12:0]   no,
    input  logic [AW-1:0] si,
    input  logic [6:0]    ni,
    input  logic [15:0]   len,
    input  logic [15:0]   len_last,
    input  logic [1:0]    ebl,
    input  logic          step,
    output logic          act,
    output logic [AW-1:0] addr,
    output logic          row_last_line   // the beat at addr holds the final line of its row
);
    logic [AW-1:0] rowb, outb;
    logic [12:0]   o;
    logic [6:0]    i;
    logic [13:0]   k, nl;
    logic          last_row;

    function automatic logic [13:0] nlines(input logic [15:0] n, input logic [1:0] e);
        logic [17:0] b;
        b = {2'b0, n} << e;
        nlines = 14'((b + 18'd15) >> 4);
    endfunction

    assign last_row      = (o == no - 13'd1) && (i == ni - 7'd1);
    assign addr          = rowb + AW'(k);
    assign row_last_line = (15'(k) + 15'(P) >= 15'(nl));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            act <= 1'b0;
        end else if (!act) begin
            if (start) begin
                act  <= 1'b1;
                rowb <= base;
                outb <= base;
                o    <= '0;
                i    <= '0;
                k    <= '0;
                nl   <= (no == 13'd1 && ni == 7'd1) ? nlines(len_last, ebl) : nlines(len, ebl);
            end
        end else if (step) begin
            if (!row_last_line) begin
                k <= k + 14'(P);
            end else begin
                k <= '0;
                if (last_row) begin
                    act <= 1'b0;
                end else begin
                    if (i == ni - 7'd1) begin
                        i    <= '0;
                        o    <= o + 13'd1;
                        outb <= outb + so;
                        rowb <= outb + so;
                        nl   <= (o + 13'd1 == no - 13'd1 && ni == 7'd1) ? nlines(len_last, ebl)
                                                                         : nlines(len, ebl);
                    end else begin
                        i    <= i + 7'd1;
                        rowb <= rowb + si;
                        nl   <= (o == no - 13'd1 && i + 7'd1 == ni - 7'd1) ? nlines(len_last, ebl)
                                                                          : nlines(len, ebl);
                    end
                end
            end
        end
    end
endmodule
