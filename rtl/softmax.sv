// softmax.sv -- C5 integer softmax over one row, exact full-row 3-pass (ADR-002), with the
// optional score requantization of the ISA's SOFTMAX_ONLINE (docs/isa.md §3-4, ADR-004).
//
// Input: a row streamed as L-lane beats ending with in_last. Per row (in_* sideband constant
// over the row): in_rq selects the input format, in_slot the score record, in_nkeep the
// causal limit.
//   in_rq = 0: int16 logits s_j (Q.8 log2 units), low 16 bits of each int32 lane.
//   in_rq = 1: int32 scores; s_j = requant16(wrap32((acc_j << ls) + bias); mult, shift)
//              (C2, rtl/requant.sv) with the 16-byte score record {bias, mult, shift, ls}
//              of slot in_slot (loaded with ctx_we; = the PBUF "softmax ctx" record). Its
//              int16 clamps are counted on every lane with keep = 1 (sat_count): the golden
//              model counts h{l}.score over all keys of the step, masked or not.
//   Columns j >= in_nkeep are masked (causal; in_nkeep = 0 keeps every column), as are
//   lanes with keep = 0. Masked positions are excluded from max and sum and return p = 0.
//   pass 1 (while the row streams in): M = max_j s_j over kept j; s and masks are buffered
//   pass 2: d_j = M - s_j (uint16); e_j = EXP2(d_j); S = sum_j e_j
//           (y, b) = RECIP(S)
//   pass 3: p_j = min(127, rshr(e_j * (127*y), 15 + b))        (re-reads the buffer)
// Output: one beat per input beat, in order, keep/last unchanged (out_keep = in_keep), p
// in [0,127] (int8 scale 1/127) zero-extended in int32 lanes. Bit-exact against
// nanoxl.quant.ops.softmax_i (and funcsim SOFTMAX_ONLINE). A row with no kept position
// returns all zeros (outside the contract: the reference asserts).
//
// The row buffer holds CTX_MAX elements (ceil(CTX_MAX/L) beats). A longer row sets the
// sticky err_len flag; it still produces one output beat per input beat, but its data is
// unspecified. Tables (exp2, recip) are written one PBUF line (8 entries) per cycle.
//
// Score requant lanes (docs/fpga.md §2.3 C6): the requant front end has RQL lanes (RQL
// divides L). With RQL < L an int32 beat is requantized as R = L/RQL sub-beats on
// consecutive cycles (sub-beat 0 straight from in_data, the others from a shift register)
// and reassembled before the row buffer, so an in_rq row streams in at one beat per R
// cycles (in_ready low for the R-1 cycles after each accept). Results and clamp counts are
// unchanged (the clamps of a beat are counted over its sub-beats).
//
// Timing (docs/sfu.md §4), nb = ceil(len/L): nb cycles to stream in (R*nb when in_rq = 1,
// +6 for the requant pipeline), nb + 3 for pass 2, 5 for RECIP, nb + 5 for pass 3
// (out_ready = 1); the next row streams in after that: 3*nb + 13 cycles per row
// ((R+2)*nb + 19 with in_rq). Output backpressure stalls pass 3 only.
module softmax #(
    parameter int L       = 4,
    parameter int CTX_MAX = 256,
    parameter int NCTX    = 8,      // score-record slots (one per attention layer)
    parameter int RQL     = L       // score requant lanes (divides L)
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [L*32-1:0] in_data,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    input  logic            in_rq,
    input  logic [7:0]      in_slot,
    input  logic [15:0]     in_nkeep,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*32-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic            out_last,
    input  logic [1:0]      lut_we,      // [0] exp2, [1] recip
    input  logic [4:0]      lut_waddr,   // table line
    input  logic [127:0]    lut_wdata,
    input  logic            ctx_we,      // score record {int32 bias, mult, shift, ls} (LE)
    input  logic [7:0]      ctx_slot,
    input  logic [127:0]    ctx_data,
    input  logic            sat_clr,
    output logic [31:0]     sat_count,
    output logic            sat_sticky,
    output logic            busy,
    output logic            err_len
);
    localparam int NB  = (CTX_MAX + L - 1) / L;
    localparam int BAW = (NB > 1) ? $clog2(NB) : 1;
    localparam int SW  = 16 + $clog2(CTX_MAX + 1) + 1;     // sum of <= CTX_MAX 16-bit e's
    localparam int CW  = 16;                               // beat counter width
    localparam int CSW = (NCTX > 1) ? $clog2(NCTX) : 1;
    localparam int R   = L / RQL;                          // requant sub-beats per beat
    localparam int RW  = (R > 1) ? $clog2(R) : 1;
    localparam int HL  = (R > 1) ? L - RQL : RQL;          // lanes held for sub-beats 1..R-1

    typedef enum logic [2:0] {S_IN, S_DR, S_P2, S_RCP, S_P3} state_t;
    state_t st;

    // ---------------- score records ----------------
    logic [31:0] r_bias [NCTX];
    logic [31:0] r_mult [NCTX];
    logic [4:0]  r_shift[NCTX];
    logic [4:0]  r_ls   [NCTX];
    always_ff @(posedge clk) begin
        if (ctx_we && ctx_slot < 8'(NCTX)) begin
            r_bias [ctx_slot[CSW-1:0]] <= ctx_data[31:0];
            r_mult [ctx_slot[CSW-1:0]] <= ctx_data[63:32];
            r_shift[ctx_slot[CSW-1:0]] <= ctx_data[68:64];
            r_ls   [ctx_slot[CSW-1:0]] <= ctx_data[100:96];
        end
    end

    // ---------------- input side ----------------
    logic            inrow;            // a non-last beat of the current row was accepted
    logic            in_fire, rq_in_ready;
    logic            m_rq;             // row mode (latched at the first beat)
    logic [15:0]     m_nkeep;
    logic            cur_rq;
    logic [RW-1:0]   pend;             // requant sub-beats still to issue from the hold register
    assign cur_rq   = inrow ? m_rq : in_rq;
    assign in_ready = (st == S_IN) && (!cur_rq || (rq_in_ready && pend == '0));
    assign in_fire  = in_valid && in_ready;
    assign busy     = (st != S_IN) || inrow;

    // requant front end (int32 scores -> int16), RQL lanes
    logic [L*32-1:0] acc_sh;
    logic [CSW-1:0]  sl;
    assign sl = in_slot[CSW-1:0];
    always_comb begin
        for (int j = 0; j < L; j++) acc_sh[j*32 +: 32] = in_data[j*32 +: 32] << r_ls[sl];
    end
    // sub-beats 1..R-1 of the beat being requantized
    logic [HL*32-1:0] h_acc;
    logic [HL-1:0]    h_keep;
    logic [31:0]      h_bias, h_mult;
    logic [4:0]       h_shift;
    logic             h_last;
    logic             rq_iv, rq_il;
    logic [RQL*32-1:0] rq_acc, rq_bias, rq_mult;
    logic [RQL*5-1:0]  rq_shift;
    logic [RQL-1:0]    rq_ik;
    always_comb begin
        if (pend != '0) begin
            rq_iv  = 1'b1;
            rq_il  = h_last && (pend == RW'(1));
            rq_acc = h_acc[RQL*32-1:0];
            rq_ik  = h_keep[RQL-1:0];
            for (int j = 0; j < RQL; j++) begin
                rq_bias[j*32 +: 32] = h_bias;
                rq_mult[j*32 +: 32] = h_mult;
                rq_shift[j*5 +: 5]  = h_shift;
            end
        end else begin
            rq_iv  = in_fire && cur_rq;
            rq_il  = in_last && (R == 1);
            rq_acc = acc_sh[RQL*32-1:0];
            rq_ik  = in_keep[RQL-1:0];
            for (int j = 0; j < RQL; j++) begin
                rq_bias[j*32 +: 32] = r_bias[sl];
                rq_mult[j*32 +: 32] = r_mult[sl];
                rq_shift[j*5 +: 5]  = r_shift[sl];
            end
        end
    end
    logic              rq_out_valid, rq_out_last;
    logic [RQL*16-1:0] rq_out_data;
    logic [RQL-1:0]    rq_out_keep, rq_out_sat;
    requant #(.L(RQL)) u_rq (
        .clk(clk), .rst_n(rst_n),
        .in_valid(rq_iv), .in_ready(rq_in_ready),
        .in_acc(rq_acc), .in_bias(rq_bias), .in_mult(rq_mult), .in_shift(rq_shift),
        .in_keep(rq_ik), .in_last(rq_il), .in_out16(1'b1),
        .out_valid(rq_out_valid), .out_ready(1'b1), .out_data(rq_out_data),
        .out_keep(rq_out_keep), .out_sat(rq_out_sat), .out_last(rq_out_last),
        .sat_clr(sat_clr), .sat_count(sat_count), .sat_sticky(sat_sticky)
    );
    // reassembly: sub-beats leave the requant pipeline on consecutive cycles (out_ready = 1)
    // and are shifted in from the top, so sub-beat 0 ends in lanes 0..RQL-1
    logic            rqw_fire;
    logic [L*16-1:0] rqw_data;
    logic [L-1:0]    rqw_keep;
    generate
        if (R > 1) begin : g_rqmux
            logic [RW-1:0]        oc;
            logic [HL*16-1:0]     a_data;
            logic [HL-1:0]        a_keep;
            always_ff @(posedge clk) begin
                if (!rst_n) pend <= '0;
                else if (pend != '0)             pend <= pend - 1'b1;
                else if (in_fire && cur_rq)      pend <= RW'(R - 1);
            end
            always_ff @(posedge clk) begin
                if (pend != '0) begin
                    h_acc  <= h_acc >> (RQL*32);
                    h_keep <= h_keep >> RQL;
                end else begin
                    h_acc   <= acc_sh[L*32-1:RQL*32];
                    h_keep  <= in_keep[L-1:RQL];
                    h_bias  <= r_bias[sl];
                    h_mult  <= r_mult[sl];
                    h_shift <= r_shift[sl];
                    h_last  <= in_last;
                end
            end
            always_ff @(posedge clk) begin
                if (!rst_n) oc <= '0;
                else if (rq_out_valid) oc <= (oc == RW'(R - 1)) ? '0 : oc + 1'b1;
            end
            always_ff @(posedge clk) begin
                if (rq_out_valid) begin
                    a_data <= rqw_data[L*16-1:RQL*16];
                    a_keep <= rqw_keep[L-1:RQL];
                end
            end
            assign rqw_fire = rq_out_valid && (oc == RW'(R - 1));
            assign rqw_data = {rq_out_data, a_data};
            assign rqw_keep = {rq_out_keep, a_keep};
        end else begin : g_rqfull
            assign pend   = '0;
            assign h_acc  = '0;
            assign h_keep = '0;
            assign h_bias = '0;
            assign h_mult = '0;
            assign h_shift = '0;
            assign h_last = 1'b0;
            assign rqw_fire = rq_out_valid;
            assign rqw_data = rq_out_data;
            assign rqw_keep = rq_out_keep;
        end
    endgenerate
    // buffer-write source: the input directly (int16 rows) or the requant output
    logic            wr_fire, wr_last;
    logic [L*16-1:0] wr_data;
    logic [L-1:0]    wr_keep, wr_mask;
    logic [CW-1:0]   wcnt, nbeats, rcnt;
    logic [15:0]     nk;            // causal limit of the beat being written
    assign nk = (m_rq || inrow) ? m_nkeep : in_nkeep;
    always_comb begin
        if (m_rq || (cur_rq && !inrow)) begin
            wr_fire = rqw_fire;      wr_last = rq_out_last;
            wr_data = rqw_data;      wr_keep = rqw_keep;
        end else begin
            wr_fire = in_fire;       wr_last = in_last;
            wr_keep = in_keep;
            for (int j = 0; j < L; j++) wr_data[j*16 +: 16] = in_data[j*32 +: 16];
        end
        for (int j = 0; j < L; j++)
            wr_mask[j] = wr_keep[j] &&
                         (nk == 16'd0 || 32'(wcnt) * 32'(L) + 32'(j) < 32'(nk));
    end

    // ---------------- row buffer ----------------
    logic [L*16-1:0] sbuf [NB];
    logic [L-1:0]    kbuf [NB];     // keep as received (returned on out_keep)
    logic [L-1:0]    mbuf [NB];     // effective mask (keep and causal limit)
    logic signed [15:0] mx;
    logic            anyk;
    always_ff @(posedge clk) begin
        if (wr_fire && wcnt < CW'(NB)) begin
            sbuf[wcnt[BAW-1:0]] <= wr_data;
            kbuf[wcnt[BAW-1:0]] <= wr_keep;
            mbuf[wcnt[BAW-1:0]] <= wr_mask;
        end
    end

    // per-beat masked max (combinational over L lanes)
    logic signed [15:0] bmax;
    logic               bany;
    always_comb begin
        bmax = 16'sh8000;
        bany = 1'b0;
        for (int j = 0; j < L; j++) begin
            if (wr_mask[j]) begin
                if (!bany || $signed(wr_data[j*16 +: 16]) > bmax) bmax = $signed(wr_data[j*16 +: 16]);
                bany = 1'b1;
            end
        end
    end

    // ---------------- read pipeline (passes 2 and 3) ----------------
    // A: buffer read | B: d = M - s, EXP2 E1 | C: EXP2 E2 -> e, sum (pass 2) |
    // D: e * 127y product (pass 3) | E: rounding shift, min 127 -> out register
    logic       vA, vB, vC, vD, vE;
    logic       p3A, p3B, p3C;
    logic       lA, lB, lC, lD;
    logic [L-1:0] kA, kB, kC, kD;       // effective masks
    logic [L-1:0] tA, tB, tC, tD;       // keep as received
    logic [L*16-1:0] sA;
    logic       en, issue, issuing;
    logic [SW-1:0] ssum;
    logic [23:0]   y127;
    logic [5:0]    shv;               // 15 + b

    assign en      = !vE || out_ready;
    assign issuing = (st == S_P2 || st == S_P3) && (rcnt < nbeats);
    assign issue   = en && issuing;
    assign out_valid = vE;

    always_ff @(posedge clk) begin
        if (en) begin
            sA  <= sbuf[rcnt[BAW-1:0]];
            kA  <= mbuf[rcnt[BAW-1:0]];
            tA  <= kbuf[rcnt[BAW-1:0]];
            lA  <= (rcnt == nbeats - 1'b1);
            p3A <= (st == S_P3);
            kB <= kA;  tB <= tA;  lB <= lA;  p3B <= p3A;
            kC <= kB;  tC <= tB;  lC <= lB;  p3C <= p3B;
            kD <= kC;  tD <= tC;  lD <= lC;
            out_keep <= tD;
            out_last <= lD;
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            vA <= 1'b0; vB <= 1'b0; vC <= 1'b0; vD <= 1'b0; vE <= 1'b0;
        end else if (en) begin
            vA <= issue;
            vB <= vA;
            vC <= vB;
            vD <= vC && p3C;
            vE <= vD;
        end
    end

    logic [15:0] e_l [L];
    genvar j;
    generate
        for (j = 0; j < L; j++) begin : g_lane
            wire signed [16:0] dd = 17'(mx) - 17'($signed(sA[j*16 +: 16]));
            sfu_exp2 u_exp2 (
                .clk(clk), .en(en), .d(dd[15:0]), .e(e_l[j]),
                .lut_we(lut_we[0]), .lut_waddr(lut_waddr), .lut_wdata(lut_wdata)
            );
            logic [39:0] prodD;
            wire  [15:0] eC = kC[j] ? e_l[j] : 16'd0;
            always_ff @(posedge clk) if (en) prodD <= 40'(eC) * 40'(y127);
            wire  [47:0] rnd = 48'(prodD) + (48'd1 << (shv - 6'd1));
            wire  [47:0] pq  = rnd >> shv;
            always_ff @(posedge clk)
                if (en) out_data[j*32 +: 32] <= (!kD[j]) ? 32'd0 :
                                                (pq > 48'd127) ? 32'd127 : {25'd0, pq[6:0]};
        end
    endgenerate

    // pass-2 lane sum
    logic [SW-1:0] esum;
    always_comb begin
        esum = '0;
        for (int i = 0; i < L; i++) esum = esum + (kC[i] ? SW'(e_l[i]) : SW'(0));
    end

    // ---------------- RECIP (scalar) ----------------
    logic [16:0] ry;
    logic [5:0]  rb;
    logic [2:0]  rc;
    sfu_recip #(.SW(SW)) u_recip (
        .clk(clk), .en(st == S_RCP), .s(ssum), .y(ry), .b(rb),
        .lut_we(lut_we[1]), .lut_waddr(lut_waddr), .lut_wdata(lut_wdata)
    );

    // ---------------- control ----------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IN; wcnt <= '0; anyk <= 1'b0; err_len <= 1'b0; inrow <= 1'b0;
            mx <= 16'sh8000; rcnt <= '0; ssum <= '0; nbeats <= '0; rc <= '0;
            m_rq <= 1'b0; m_nkeep <= '0;
        end else begin
            // input beats: latch the row mode on the first beat
            if (in_fire) begin
                if (!inrow) begin
                    m_rq    <= in_rq;
                    m_nkeep <= in_nkeep;
                end
                inrow <= !in_last;
            end
            // buffer writes (pass 1)
            if (wr_fire && (st == S_IN || st == S_DR)) begin
                if (bany && (!anyk || bmax > mx)) mx <= bmax;
                anyk <= anyk | bany;
                if (wcnt >= CW'(NB)) err_len <= 1'b1;
                if (wr_last) begin
                    nbeats <= wcnt + 1'b1;
                    wcnt   <= '0;
                    rcnt   <= '0;
                    ssum   <= '0;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
            end
            case (st)
                S_IN: begin
                    if (wr_fire && wr_last) st <= S_P2;
                    else if (in_fire && in_last) st <= S_DR;   // requant row: drain the pipe
                end
                S_DR: if (wr_fire && wr_last) st <= S_P2;
                S_P2: begin
                    if (issue) rcnt <= rcnt + 1'b1;
                    if (en && vC && !p3C) ssum <= ssum + esum;
                    // leave as the last e-sum lands (vC accumulates on this edge)
                    if (!issuing && !vA && !vB) begin
                        st <= S_RCP;
                        rc <= '0;
                    end
                end
                S_RCP: begin
                    rc <= rc + 1'b1;
                    if (rc == 3'd4) begin
                        y127 <= 24'(ry) * 24'd127;
                        shv  <= 6'd15 + rb;
                        rcnt <= '0;
                        st   <= S_P3;
                    end
                end
                default: begin // S_P3
                    if (issue) rcnt <= rcnt + 1'b1;
                    // leave as the last output beat transfers
                    if (!issuing && !vA && !vB && !vC && !vD && (!vE || out_ready)) begin
                        st   <= S_IN;
                        anyk <= 1'b0;
                        mx   <= 16'sh8000;
                        m_rq <= 1'b0;
                    end
                end
            endcase
        end
    end
endmodule
