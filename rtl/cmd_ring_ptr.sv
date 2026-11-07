// cmd_ring_ptr.sv -- command-ring pointer logic of the fetcher.
//
// The ring is `size` 16-byte words at `base`. Software owns `tail` (the next word it will write;
// written through the DOORBELL CSR), hardware owns `head` (the next word to dispatch) and `fptr`
// (the next word to fetch). Words in [head, fptr) are fetched but not yet dispatched (in flight
// on AXI or sitting in the prefetch FIFO): `occ` of them, at most FDEPTH. The ring is empty when
// head == tail, so software keeps at least one slot free (at most size-1 words pending).
//
// Burst choice: burst_len = min(words up to tail, words up to the ring end, prefetch credit,
// words up to the next 4 KiB boundary, FBURST); 0 means nothing to fetch. `issue` takes a burst
// of exactly burst_len words. `consume` dispatches the word at head.
// cfg_ok = size != 0 && tail < size; with a bad configuration nothing is fetched.
//
// Formal (formal/ring.ys): head, fptr < size; occ == (fptr - head) mod size <= FDEPTH; the
// fetch pointer never passes tail; bursts never cross the ring end or a 4 KiB boundary.
module cmd_ring_ptr #(
    parameter int PW     = 24,   // ring size / pointer width in words
    parameter int AW     = 32,
    parameter int FDEPTH = 16,
    parameter int FBURST = 8     // max words per fetch burst (<= 256)
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          clr,
    input  logic [AW-1:0] base,
    input  logic [PW-1:0] size,
    input  logic [PW-1:0] tail,
    input  logic          issue,
    input  logic          consume,
    output logic          cfg_ok,
    output logic [PW-1:0] head,
    output logic [PW-1:0] fptr,
    output logic [8:0]    burst_len,
    output logic [AW-1:0] burst_addr,
    output logic [$clog2(FDEPTH+1)-1:0] occ
);
    localparam int OB = $clog2(FDEPTH + 1);

    logic [PW-1:0] avail, to_end;
    logic [8:0]    to_4k;
    logic [OB-1:0] credit;
    logic [PW-1:0] fptr_inc, head_inc;

    assign cfg_ok     = (size != '0) && (tail < size);
    assign avail      = (tail >= fptr) ? (tail - fptr) : (tail + size - fptr);
    assign to_end     = size - fptr;
    assign burst_addr = base + (AW'(fptr) << 4);
    assign to_4k      = 9'd256 - {1'b0, burst_addr[11:4]};
    assign credit     = OB'(FDEPTH) - occ;

    always_comb begin
        logic [PW-1:0] m;
        m = avail;
        if (to_end < m)              m = to_end;
        if (PW'(credit) < m)         m = PW'(credit);
        if (PW'(to_4k) < m)          m = PW'(to_4k);
        if (PW'(FBURST) < m)         m = PW'(FBURST);
        burst_len = cfg_ok ? 9'(m) : 9'd0;
    end

    assign fptr_inc = (fptr + PW'(burst_len) == size) ? '0 : fptr + PW'(burst_len);
    assign head_inc = (head + 1'b1 == size) ? '0 : head + 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n || clr) begin
            head <= '0;
            fptr <= '0;
            occ  <= '0;
        end else begin
            if (issue)   fptr <= fptr_inc;
            if (consume) head <= head_inc;
            occ <= occ + (issue ? OB'(burst_len) : '0) - OB'(consume);
        end
    end

`ifdef FORMAL
    logic f_init = 1'b1;
    always_ff @(posedge clk) f_init <= 1'b0;
    always_comb if (f_init) assume (!rst_n);
    always_comb assume (!clr);
    // environment: ring geometry fixed and legal; software only moves tail forward without
    // overtaking head; the fetcher only issues a non-empty burst and only consumes fetched words
    logic [PW-1:0] f_pend, f_pend_prev, f_tail_prev, f_size_prev;
    logic [AW-1:0] f_base_prev;
    assign f_pend = (tail >= head) ? (tail - head) : (tail + size - head);
    always_ff @(posedge clk) begin
        f_pend_prev <= f_pend;
        f_tail_prev <= tail;
        f_size_prev <= size;
        f_base_prev <= base;
    end
    always_comb begin
        assume (size != '0 && tail < size);
        assume (base[3:0] == 4'd0);
        if (!f_init) begin
            assume (size == f_size_prev);
            assume (base == f_base_prev);
        end
        if (issue)   assume (burst_len != 0);
        if (consume) assume (occ != 0);
    end
    // tail moves forward: pending (as seen from the old head) can only grow, unless head moved
    logic f_cons_prev;
    always_ff @(posedge clk) f_cons_prev <= consume;
    always_comb if (!f_init && rst_n && tail != f_tail_prev)
        assume (f_pend + PW'(f_cons_prev) >= f_pend_prev);

    logic [PW-1:0] f_occ_ring;
    assign f_occ_ring = (fptr >= head) ? (fptr - head) : (fptr + size - head);
    always_comb begin
        if (!f_init && rst_n) begin
            assert (head < size);
            assert (fptr < size);
            assert (32'(occ) <= FDEPTH);
            assert (PW'(occ) == f_occ_ring);              // fetched-not-dispatched words
            assert (f_occ_ring <= f_pend);                // fptr never passes tail
            if (burst_len != 0) begin
                assert (PW'(burst_len) <= avail);
                assert (32'(fptr) + 32'(burst_len) <= 32'(size));               // no ring wrap inside a burst
                assert (burst_addr[11:4] + 32'(burst_len) <= 256);                // no 4 KiB crossing
                assert (32'(occ) + 32'(burst_len) <= FDEPTH);
            end
        end
    end
`endif
endmodule
