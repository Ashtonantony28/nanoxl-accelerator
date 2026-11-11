// vec_seq_wr.sv -- VEC engine destination stream: packs vec_unit D beats (L int32 lanes,
// keep, last) into P-line (16P-byte) port beats of 2^ebl-byte elements (the low bytes of
// each lane) and writes them over a scratchpad write port, at the beat addresses of a row
// set (vec_seq_walk: from the row's first line, P lines apart). Byte strobes are the lanes'
// keep bits, so a row's final ragged beat writes only the row's own bytes (lines past the
// row end, and bytes past the row end in its last line, have strobe 0). Each row starts on
// a line boundary.
//
// Write port (the DMA's sp_* write shape, docs/control.md §3, P lines per beat):
//   w_valid/w_ready, w_buf, w_addr (first line), w_data[128P-1:0], w_strb[16P-1:0]
//   (line j in data bits [128j +: 128], strobes [16j +: 16])
// A write is complete (visible to later reads) when it is accepted.
// A 2-entry FIFO sits in front of the port, so i_ready never depends on w_ready
// combinationally.
module vec_seq_wr #(
    parameter int L  = 4,
    parameter int AW = 17,
    parameter int P  = 1
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            clr,
    input  logic            start,
    input  logic [2:0]      d_buf,
    input  logic [AW-1:0]   d_base,
    input  logic [AW-1:0]   d_so,
    input  logic [12:0]     d_no,
    input  logic [AW-1:0]   d_si,
    input  logic [6:0]      d_ni,
    input  logic [15:0]     d_len,
    input  logic [15:0]     d_len_last,
    input  logic [1:0]      d_ebl,
    output logic            busy,
    input  logic            i_valid,
    output logic            i_ready,
    input  logic [L*32-1:0] i_data,
    input  logic [L-1:0]    i_keep,
    input  logic            i_last,
    output logic            w_valid,
    input  logic            w_ready,
    output logic [2:0]      w_buf,
    output logic [AW-1:0]   w_addr,
    output logic [P*128-1:0] w_data,
    output logic [P*16-1:0]  w_strb
);
    localparam int LL = $clog2(L);
    localparam int PL = $clog2(P);
    localparam int CB = 16 * P;                   // bytes of a port beat ("chunk")
    localparam int BB = 4 * L;                    // bytes of a vec_unit beat at int32
    localparam int HB = (BB > CB) ? BB : CB;      // holding register bytes
    localparam int AL = (BB > CB) ? BB / CB : 1;  // chunks of a beat at int32
    localparam int FW = 3 + AW + CB * 8 + CB;

    logic rst_i;
    assign rst_i = rst_n && !clr;

    logic [1:0]  ebl;
    logic [2:0]  buf_q;
    logic [3:0]  lb_l2;                           // log2 bytes per beat
    always_ff @(posedge clk) begin
        if (start && !busy) begin
            ebl   <= d_ebl;
            buf_q <= d_buf;
            lb_l2 <= 4'(32'(d_ebl) + LL);
        end
    end

    // ---------------- walker (address of each pushed line) ----------------
    logic          w_act, w_step, w_rll;
    logic [AW-1:0] wk_addr;
    vec_seq_walk #(.AW(AW), .P(P)) u_walk (
        .clk(clk), .rst_n(rst_i), .start(start && !busy), .base(d_base), .so(d_so), .no(d_no),
        .si(d_si), .ni(d_ni), .len(d_len), .len_last(d_len_last), .ebl(d_ebl),
        .step(w_step), .act(w_act), .addr(wk_addr), .row_last_line(w_rll)
    );

    // ---------------- pack one beat into bytes + strobes ----------------
    logic [HB*8-1:0] pb;
    logic [HB-1:0]   ps;
    always_comb begin
        pb = '0;
        ps = '0;
        for (int j = 0; j < L; j++) begin
            case (ebl)
                2'd0: begin pb[j*8 +: 8]   = i_data[j*32 +: 8];  ps[j]       = i_keep[j];       end
                2'd1: begin pb[j*16 +: 16] = i_data[j*32 +: 16]; ps[j*2 +: 2] = {2{i_keep[j]}}; end
                default: begin pb[j*32 +: 32] = i_data[j*32 +: 32]; ps[j*4 +: 4] = {4{i_keep[j]}}; end
            endcase
        end
    end

    // ---------------- line FIFO to the port ----------------
    logic          f_in_valid, f_in_ready, f_out_valid;
    logic [CB*8-1:0] f_data_in;
    logic [CB-1:0]   f_strb_in;
    logic [FW-1:0] f_out;
    logic [1:0]    f_cnt;
    fifo_sync #(.W(FW), .DEPTH(2)) u_fifo (
        .clk(clk), .rst_n(rst_i), .in_valid(f_in_valid), .in_ready(f_in_ready),
        .in_data({buf_q, wk_addr, f_data_in, f_strb_in}), .out_valid(f_out_valid),
        .out_ready(w_ready), .out_data(f_out), .count(f_cnt)
    );
    assign w_valid = f_out_valid;
    assign {w_buf, w_addr, w_data, w_strb} = f_out;
    assign w_step  = f_in_valid && f_in_ready;

    // ---------------- wide beats (>= CB bytes): one beat -> several chunks ----------------
    logic            hv;
    logic [HB*8-1:0] hb;
    logic [HB-1:0]   hs;
    logic [7:0]      hm, hn;                      // current chunk, last chunk with a strobe
    // ---------------- narrow beats (< CB bytes): several beats -> one chunk ----------------
    logic [CB*8-1:0] cl;
    logic [CB-1:0]   cs;
    logic [6:0]      sub;
    logic            wide;
    assign wide = (lb_l2 >= 4'(4 + PL));

    logic [7:0] last_line;
    always_comb begin
        last_line = '0;
        for (int m = 0; m < AL; m++) if (|ps[m*CB +: CB]) last_line = 8'(m);
    end

    // narrow: merge the incoming beat into the chunk at slice `sub`
    logic [CB*8-1:0] cl_n;
    logic [CB-1:0]   cs_n;
    logic            nl_done;
    always_comb begin
        cl_n = cl;
        cs_n = cs;
        for (int p = 0; p < CB; p++) begin
            if ((p >> lb_l2) == int'(sub)) begin
                cl_n[p*8 +: 8] = pb[(p & ((1 << lb_l2) - 1))*8 +: 8] & {8{ps[p & ((1 << lb_l2) - 1)]}};
                cs_n[p]        = ps[p & ((1 << lb_l2) - 1)];
            end
        end
        nl_done = (sub == 7'((CB >> lb_l2) - 1)) || i_last;
    end

    always_comb begin
        if (wide) begin
            i_ready    = !hv || (f_in_ready && hm == hn);
            f_in_valid = hv;
            f_data_in  = hb[int'(hm)*CB*8 +: CB*8];
            f_strb_in  = hs[int'(hm)*CB +: CB];
        end else begin
            i_ready    = f_in_ready;
            f_in_valid = i_valid && nl_done;
            f_data_in  = cl_n;
            f_strb_in  = cs_n;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_i) begin
            hv  <= 1'b0;
            sub <= '0;
            cl  <= '0;
            cs  <= '0;
        end else if (wide) begin
            if (hv && f_in_ready) begin
                if (hm == hn) hv <= 1'b0;
                hm <= hm + 8'd1;
            end
            if (i_valid && i_ready) begin
                hv <= 1'b1;
                hb <= pb;
                hs <= ps;
                hm <= '0;
                hn <= last_line;
            end
        end else if (i_valid && i_ready) begin
            if (nl_done) begin
                sub <= '0;
                cl  <= '0;
                cs  <= '0;
            end else begin
                sub <= sub + 7'd1;
                cl  <= cl_n;
                cs  <= cs_n;
            end
        end
    end

    assign busy = w_act || hv || (cs != '0) || f_out_valid;

`ifndef SYNTHESIS
    // Simulation-only check (AUDIT-2 A2-05). A wide beat emits chunks 0..hn only (hn = the last
    // chunk with a kept lane), and the walker steps once per emitted chunk. The two stay in step
    // only if a chunk that ends a row is exactly the last emitted chunk of a beat with i_last, i.e.
    // every non-final beat of a row has a kept lane in its last chunk (true when keep is "the first
    // len lanes" and only the row's final beat is ragged, as every producer does today). A non-final
    // beat whose trailing chunk(s) are fully masked would make every later chunk of the row set
    // land one beat early. Interior masking that leaves the last chunk with a kept lane is fine
    // (the strobes carry it). The same check catches a final beat whose keep disagrees with the
    // row length. chk_err is sticky; the check is $fatal unless a bench sets
    // chk_nofatal (tests/rtl/vec_seq_wr_cocotb.py does, to observe chk_err).
    logic h_last;                                 // i_last of the beat in the holding register
    logic chk_err, chk_fail;
    logic chk_nofatal = 1'b0;
    assign chk_fail = rst_i && wide && w_step && (w_rll != (h_last && hm == hn));
    always_ff @(posedge clk) begin
        if (!rst_n) chk_err <= 1'b0;
        else if (chk_fail) chk_err <= 1'b1;
        if (wide && i_valid && i_ready) h_last <= i_last;
    end
    always @(posedge clk) begin
        if (rst_n && chk_fail) begin
            if (chk_nofatal)
                $display("vec_seq_wr: A2-05 check failed at %0t (chunk %0d of %0d, i_last %0d, row end %0d)",
                         $time, hm, hn, h_last, w_rll);
            else
                $fatal(1, "vec_seq_wr: A2-05 check failed: the emitted chunks of a wide beat disagree with the walker's row end (chunk %0d of %0d, i_last %0d, row end %0d)",
                       hm, hn, h_last, w_rll);
        end
    end
`endif
endmodule
