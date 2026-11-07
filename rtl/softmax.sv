// softmax.sv -- C5 integer softmax over one row, exact full-row 3-pass (ADR-002).
//
// Input: a row of int16 logits s_j (Q.8 log2 units, sign-extended in int32 lanes; only the
// low 16 bits are used), streamed as L-lane beats ending with in_last. Lanes with keep = 0
// are masked positions: they are excluded from max and sum and return p = 0.
//   pass 1 (while the row streams in): M = max_j s_j over kept j; the row is buffered
//   pass 2: d_j = M - s_j (uint16); e_j = EXP2(d_j); S = sum_j e_j
//           (y, b) = RECIP(S)
//   pass 3: p_j = min(127, rshr(e_j * (127*y), 15 + b))        (re-reads the buffer)
// Output: one beat per input beat, in order, keep/last unchanged, p in [0,127] (int8 scale
// 1/127) zero-extended in int32 lanes. Bit-exact against nanoxl.quant.ops.softmax_i.
// A row with no kept lane returns all zeros (outside the contract: the reference asserts).
//
// The row buffer holds CTX_MAX elements (ceil(CTX_MAX/L) beats). A longer row sets the
// sticky err_len flag; it still produces one output beat per input beat, but its data is
// unspecified. Tables (exp2, recip) are loaded through the write port while idle.
//
// Timing (docs/sfu.md §4): one row of nb = ceil(len/L) beats takes nb cycles to stream in
// (in_ready = 1 in the input phase), then nb + 3 cycles for pass 2, 5 cycles for RECIP and
// nb + 5 cycles for pass 3 (with out_ready = 1); the next row can stream in after that:
// 3*nb + 13 cycles per row at full rate. Output backpressure stalls pass 3 only.
module softmax #(
    parameter int L       = 4,
    parameter int CTX_MAX = 256
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [L*32-1:0] in_data,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*32-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic            out_last,
    input  logic [1:0]      lut_we,      // [0] exp2, [1] recip
    input  logic [7:0]      lut_waddr,
    input  logic [15:0]     lut_wdata,
    output logic            busy,
    output logic            err_len
);
    localparam int NB  = (CTX_MAX + L - 1) / L;
    localparam int BAW = (NB > 1) ? $clog2(NB) : 1;
    localparam int SW  = 16 + $clog2(CTX_MAX + 1) + 1;     // sum of <= CTX_MAX 16-bit e's
    localparam int CW  = 16;                               // beat counter width

    typedef enum logic [2:0] {S_IN, S_P2, S_RCP, S_P3} state_t;
    state_t st;

    // ---------------- row buffer ----------------
    logic [L*16-1:0] sbuf [NB];
    logic [L-1:0]    kbuf [NB];
    logic [CW-1:0]   wcnt, nbeats, rcnt;
    logic signed [15:0] mx;
    logic            anyk;
    logic            in_fire;

    assign in_ready = (st == S_IN);
    assign in_fire  = in_valid && in_ready;
    assign busy     = (st != S_IN) || (wcnt != '0);

    // per-beat masked max (combinational over L lanes)
    logic signed [15:0] bmax;
    logic               bany;
    always_comb begin
        bmax = 16'sh8000;
        bany = 1'b0;
        for (int j = 0; j < L; j++) begin
            if (in_keep[j]) begin
                if (!bany || $signed(in_data[j*32 +: 16]) > bmax) bmax = $signed(in_data[j*32 +: 16]);
                bany = 1'b1;
            end
        end
    end

    logic [L*16-1:0] in16;
    always_comb for (int j = 0; j < L; j++) in16[j*16 +: 16] = in_data[j*32 +: 16];

    always_ff @(posedge clk) begin
        if (in_fire && wcnt < CW'(NB)) begin
            sbuf[wcnt[BAW-1:0]] <= in16;
            kbuf[wcnt[BAW-1:0]] <= in_keep;
        end
    end

    // ---------------- read pipeline (passes 2 and 3) ----------------
    // A: buffer read | B: d = M - s, EXP2 E1 | C: EXP2 E2 -> e, sum (pass 2) |
    // D: e * 127y product (pass 3) | E: rounding shift, min 127 -> out register
    logic       vA, vB, vC, vD, vE;
    logic       p3A, p3B, p3C;
    logic       lA, lB, lC, lD;
    logic [L-1:0] kA, kB, kC, kD;
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
            kA  <= kbuf[rcnt[BAW-1:0]];
            lA  <= (rcnt == nbeats - 1'b1);
            p3A <= (st == S_P3);
            kB <= kA;  lB <= lA;  p3B <= p3A;
            kC <= kB;  lC <= lB;  p3C <= p3B;
            kD <= kC;  lD <= lC;
            out_keep <= kD;
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
            st <= S_IN; wcnt <= '0; anyk <= 1'b0; err_len <= 1'b0;
            mx <= 16'sh8000; rcnt <= '0; ssum <= '0; nbeats <= '0; rc <= '0;
        end else begin
            case (st)
                S_IN: if (in_fire) begin
                    if (bany && (!anyk || bmax > mx)) mx <= bmax;
                    anyk <= anyk | bany;
                    if (wcnt >= CW'(NB)) err_len <= 1'b1;
                    if (in_last) begin
                        nbeats <= wcnt + 1'b1;
                        wcnt   <= '0;
                        rcnt   <= '0;
                        ssum   <= '0;
                        st     <= S_P2;
                    end else begin
                        wcnt <= wcnt + 1'b1;
                    end
                end
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
                    end
                end
            endcase
        end
    end
endmodule
