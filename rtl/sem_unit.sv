// sem_unit.sv -- counting semaphores shared by the NQ command queues (WAIT / SIGNAL, docs/isa.md §2).
//
// Each queue q can present one SIGNAL request (sig_req[q], sig_sem) and one WAIT request
// (wait_req[q], wait_sem, wait_cnt) per cycle; a request is held until its grant.
//   * WAIT(s, k) is granted when the semaphore holds at least k after the WAITs of lower-numbered
//     queues on the same semaphore granted in this cycle; the grant subtracts k.
//   * SIGNAL(s) is granted unless the counter would exceed its maximum (2^CW - 1), counting the
//     SIGNALs of lower-numbered queues granted in this cycle; a full counter stalls the producer
//     rather than losing the signal. (Same-cycle WAIT grants are ignored for this check, which
//     only makes it conservative.)
// Counters therefore never go below zero and never wrap, and every granted SIGNAL is consumed
// by exactly one WAIT unit (no lost or duplicated signals). Formal: formal/sem.ys.
// `clr` (soft reset) zeroes all counters. `dbg_sel` / `dbg_val` read one counter for the CSRs.
module sem_unit #(
    parameter int NQ   = 3,
    parameter int NSEM = 32,
    parameter int CW   = 8      // counter width; WAIT.count is 8 bits in the ISA
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clr,
    input  logic [NQ-1:0]           sig_req,
    input  logic [NQ*5-1:0]         sig_sem,
    output logic [NQ-1:0]           sig_gnt,
    input  logic [NQ-1:0]           wait_req,
    input  logic [NQ*5-1:0]         wait_sem,
    input  logic [NQ*8-1:0]         wait_cnt,
    output logic [NQ-1:0]           wait_gnt,
    input  logic [4:0]              dbg_sel,
    output logic [CW-1:0]           dbg_val
);
    localparam int SB = (NSEM > 1) ? $clog2(NSEM) : 1;
    localparam int MAXC = (1 << CW) - 1;

    logic [CW-1:0] cnt [0:NSEM-1];

    logic [SB-1:0] ss [0:NQ-1];
    logic [SB-1:0] ws [0:NQ-1];
    logic [7:0]    wc [0:NQ-1];
    logic          s_ok [0:NQ-1];
    logic          w_ok [0:NQ-1];

    genvar gq;
    for (gq = 0; gq < NQ; gq++) begin : g_unpack
        assign ss[gq] = SB'(sig_sem[gq*5 +: 5]);
        assign ws[gq] = SB'(wait_sem[gq*5 +: 5]);
        assign wc[gq] = wait_cnt[gq*8 +: 8];
        // a semaphore id >= NSEM never exists (not grantable); NSEM = 32 in the product
        assign s_ok[gq] = (32'(sig_sem[gq*5 +: 5]) < NSEM);
        assign w_ok[gq] = (32'(wait_sem[gq*5 +: 5]) < NSEM);
    end

    // ---- grants: fixed priority by queue index among requests to the same semaphore ----
    always_comb begin
        wait_gnt = '0;
        sig_gnt  = '0;
        for (int q = 0; q < NQ; q++) begin
            int unsigned used_w;   // count already taken by earlier WAIT grants on ws[q]
            int unsigned used_s;   // SIGNALs already granted on ss[q]
            used_w = 0;
            used_s = 0;
            for (int p = 0; p < q; p++) begin
                if (wait_gnt[p] && ws[p] == ws[q]) used_w = used_w + 32'(wc[p]);
                if (sig_gnt[p]  && ss[p] == ss[q]) used_s = used_s + 1;
            end
            wait_gnt[q] = wait_req[q] && w_ok[q] && !clr &&
                          (32'(cnt[ws[q]]) >= used_w + 32'(wc[q]));
            sig_gnt[q]  = sig_req[q] && s_ok[q] && !clr &&
                          (32'(cnt[ss[q]]) + used_s + 1 <= MAXC);
        end
    end

    // ---- counter update ----
    always_ff @(posedge clk) begin
        for (int s = 0; s < NSEM; s++) begin
            if (!rst_n || clr) begin
                cnt[s] <= '0;
            end else begin
                int unsigned inc, dec;
                inc = 0;
                dec = 0;
                for (int q = 0; q < NQ; q++) begin
                    if (sig_gnt[q]  && 32'(ss[q]) == s) inc = inc + 1;
                    if (wait_gnt[q] && 32'(ws[q]) == s) dec = dec + 32'(wc[q]);
                end
                cnt[s] <= CW'(32'(cnt[s]) + inc - dec);
            end
        end
    end

    assign dbg_val = (32'(dbg_sel) < NSEM) ? cnt[SB'(dbg_sel)] : '0;

`ifdef FORMAL
    // ---- properties (formal/sem.ys): counts never negative, never wrap, no lost signals ----
    logic f_init = 1'b1;
    always_ff @(posedge clk) f_init <= 1'b0;
    always_comb if (f_init) assume (!rst_n);
    always_comb assume (!clr);

    // ghost totals per semaphore: granted signals and granted wait units (GW bits)
    localparam int GW = CW + 4;
    logic [GW-1:0] f_sig [0:NSEM-1];
    logic [GW-1:0] f_wt  [0:NSEM-1];
    // exact (wide) next value, to show the narrow counter never wraps either way
    logic signed [33:0] f_next [0:NSEM-1];
    always_comb begin
        for (int s = 0; s < NSEM; s++) begin
            f_next[s] = $signed({2'b0, 32'(cnt[s])});
            for (int q = 0; q < NQ; q++) begin
                if (sig_gnt[q]  && 32'(ss[q]) == s) f_next[s] = f_next[s] + 1;
                if (wait_gnt[q] && 32'(ws[q]) == s) f_next[s] = f_next[s] - $signed({26'b0, wc[q]});
            end
        end
    end
    always_ff @(posedge clk) begin
        for (int s = 0; s < NSEM; s++) begin
            if (!rst_n) begin
                f_sig[s] <= '0;
                f_wt[s]  <= '0;
            end else begin
                int unsigned inc, dec;
                inc = 0;
                dec = 0;
                for (int q = 0; q < NQ; q++) begin
                    if (sig_gnt[q]  && 32'(ss[q]) == s) inc = inc + 1;
                    if (wait_gnt[q] && 32'(ws[q]) == s) dec = dec + 32'(wc[q]);
                end
                f_sig[s] <= f_sig[s] + GW'(inc);
                f_wt[s]  <= f_wt[s]  + GW'(dec);
            end
        end
    end
    // bound the ghost counters so they cannot wrap within the proof
    always_comb for (int s = 0; s < NSEM; s++) assume (32'(f_sig[s]) < (1 << GW) - 8);

    always_comb begin
        if (!f_init && rst_n) begin
            for (int s = 0; s < NSEM; s++) begin
                assert (f_next[s] >= 0);                         // never negative
                assert (f_next[s] <= MAXC);                      // never wraps
                assert (f_wt[s] <= f_sig[s]);                    // never consumes more than signalled
                assert (32'(cnt[s]) == 32'(f_sig[s] - f_wt[s])); // no lost / duplicated signals
            end
            for (int q = 0; q < NQ; q++) begin
                // a grant only when the semaphore holds enough
                if (wait_gnt[q]) assert (32'(cnt[ws[q]]) >= 32'(wc[q]));
                // progress: a lone WAIT request that can be satisfied is granted
                if (wait_req[q] && w_ok[q] && 32'(cnt[ws[q]]) >= 32'(wc[q]) && (wait_req & ~(NQ'(1) << q)) == '0)
                    assert (wait_gnt[q]);
                // progress: a lone SIGNAL on a non-full counter is granted
                if (sig_req[q] && s_ok[q] && 32'(cnt[ss[q]]) < MAXC && (sig_req & ~(NQ'(1) << q)) == '0)
                    assert (sig_gnt[q]);
            end
        end
    end
`endif
endmodule
