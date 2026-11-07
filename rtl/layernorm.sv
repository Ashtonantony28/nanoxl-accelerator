// layernorm.sv -- C5 integer LayerNorm over one row (ADR-002), bit-exact against
// nanoxl.quant.ops.ln_norm_i + ln_affine_i.
//
// Input: int16 residual values x_i (sign-extended in int32 lanes; low 16 bits used),
// L-lane beats ending with in_last; in_slot selects the LayerNorm instance (constants and
// per-channel g/b, written line by line from the ISA LN parameter block on blk_*; L must be
// a power of two). Lanes with keep = 0 are excluded
// from every sum and return 0 (in the model every element of a row is kept).
//   pass 1 (while streaming in): S = sum x                       (int32), row buffered
//            mu = rdivpot(srdhm(S, mu_mult), mu_shift)           (C2 primitives)
//   pass 2: c_i = x_i - mu;  Q = max(sum c_i^2 + eps_q, 1)       (48-bit)
//            (y, k) = RSQRT(Q);  R = y * sqrt_d
//   pass 3: z_i = sat16(rshr(c_i * R, 22 + k))                    [sat event ln_z]
//            out_i = sat8(wrap32(rdivpot(z_i * g_i + b_i, sh)))   [sat event ln_out]
// Output: one beat per input beat, in order, keep/last unchanged, int8 sign-extended.
//
// Widths: c is int18 (C5: int17; holds whenever mu_mult/mu_shift encode 1/d for the row,
// as the bundle does), c^2 36 bits, Q 48 bits (C5 bound 2^46), R kept as int32 (C5: R <
// 2^31), c*R 50 bits (C5: < 2^49), z*g + b 34 bits. The final int32 wrap reproduces the
// reference's astype(int32) after rdivpot (it never triggers for legal sh >= 1).
// Saturation: z_sat_count / out_sat_count count kept lanes that clamp (sat_counter, 32 bit
// saturating, sticky), on output transfers; they match Golden.sat_count["ln_z"] and the
// per-LN "ln_out" counts.
//
// Rows longer than D_MAX set the sticky err_len flag and return unspecified data (beat
// count and keep/last still follow the input). Load tables/parameters only while idle.
// Timing (docs/sfu.md §5): nb = ceil(len/L) beats take nb cycles in, 4 for mu, nb + 3 for
// pass 2, 6 for RSQRT, nb + 6 for pass 3: 3*nb + 19 cycles per row at full rate.
module layernorm #(
    parameter int L     = 4,
    parameter int D_MAX = 384,
    parameter int NSLOT = 16
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [7:0]      in_slot,
    input  logic [L*32-1:0] in_data,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*32-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic            out_last,
    // rsqrt table, one PBUF line (8 uint16 entries) per write
    input  logic            lut_we,
    input  logic [4:0]      lut_waddr,
    input  logic [127:0]    lut_wdata,
    // LN parameter block (docs/isa.md §3, funcsim.ln_block), one 16-byte line per write:
    // line 0 int32 {mu_mult, mu_shift, sqrt_d, sh}; line 1 int64 eps_q; lines 2 .. 2+G-1
    // g int16 (8 per line), then b int32 (4 per line); G = ceil(cols/8). blk_cols = cols.
    input  logic            blk_we,
    input  logic [7:0]      blk_slot,
    input  logic [11:0]     blk_line,
    input  logic [12:0]     blk_cols,
    input  logic [127:0]    blk_data,
    // status
    input  logic            sat_clr,
    output logic [31:0]     z_sat_count,
    output logic [31:0]     out_sat_count,
    output logic            sat_sticky,
    output logic            busy,
    output logic            err_len
);
    localparam int NB  = (D_MAX + L - 1) / L;
    localparam int BAW = (NB > 1) ? $clog2(NB) : 1;
    localparam int SLW = (NSLOT > 1) ? $clog2(NSLOT) : 1;
    localparam int CW  = 16;
    // g/b storage: words of GW g's (NGB banks of one 8-entry line) and BW b's (NBB banks
    // of one 4-entry line); a beat's L lanes always lie inside one word (L a power of two).
    localparam int GW  = (L > 8) ? L : 8;
    localparam int BW  = (L > 4) ? L : 4;
    localparam int NGB = GW / 8;
    localparam int NBB = BW / 4;
    localparam int WG  = (D_MAX + GW - 1) / GW;     // g words per slot
    localparam int WB  = (D_MAX + BW - 1) / BW;     // b words per slot
    localparam int GAW = $clog2(NSLOT * WG + 1);
    localparam int BAW2 = $clog2(NSLOT * WB + 1);

    typedef enum logic [2:0] {S_IN, S_MU, S_P2, S_RS, S_P3} state_t;
    state_t st;

    // ---------------- constants per slot ----------------
    logic [31:0] k_mult  [NSLOT];
    logic [4:0]  k_mush  [NSLOT];
    logic [16:0] k_sqrtd [NSLOT];
    logic [47:0] k_eps   [NSLOT];
    logic [4:0]  k_sh    [NSLOT];
    // block-line decode
    logic [11:0] ng_lines, nb_lines, lg, lb;
    logic        w_ok, w_hdr0, w_hdr1, w_g, w_b;
    always_comb begin
        ng_lines = 12'((blk_cols + 13'd7) >> 3);
        nb_lines = 12'((blk_cols + 13'd3) >> 2);
        lg       = blk_line - 12'd2;
        lb       = blk_line - 12'd2 - ng_lines;
        w_ok     = blk_we && blk_slot < 8'(NSLOT);
        w_hdr0   = w_ok && blk_line == 12'd0;
        w_hdr1   = w_ok && blk_line == 12'd1;
        w_g      = w_ok && blk_line >= 12'd2 && lg < ng_lines && 32'(lg) < 32'(WG * NGB);
        w_b      = w_ok && blk_line >= 12'd2 + ng_lines && lb < nb_lines &&
                   32'(lb) < 32'(WB * NBB);
    end
    always_ff @(posedge clk) begin
        if (w_hdr0) begin
            k_mult [blk_slot[SLW-1:0]] <= blk_data[31:0];
            k_mush [blk_slot[SLW-1:0]] <= blk_data[36:32];
            k_sqrtd[blk_slot[SLW-1:0]] <= blk_data[80:64];
            k_sh   [blk_slot[SLW-1:0]] <= blk_data[100:96];
        end
        if (w_hdr1) k_eps[blk_slot[SLW-1:0]] <= blk_data[47:0];
    end

    // ---------------- input pass ----------------
    logic [L*16-1:0] xbuf [NB];
    logic [L-1:0]    kbuf [NB];
    logic [CW-1:0]   wcnt, nbeats, rcnt;
    logic [SLW-1:0]  slot;
    logic signed [31:0] ssum, mu;
    logic [31:0]     r_mult;
    logic [4:0]      r_mush, r_sh;
    logic [16:0]     r_sqrtd;
    logic [47:0]     r_eps;
    logic            in_fire;

    assign in_ready = (st == S_IN);
    assign in_fire  = in_valid && in_ready;
    assign busy     = (st != S_IN) || (wcnt != '0);

    logic signed [31:0] bsum;
    logic [L*16-1:0]    in16;
    always_comb begin
        bsum = '0;
        for (int j = 0; j < L; j++) begin
            in16[j*16 +: 16] = in_data[j*32 +: 16];
            if (in_keep[j]) bsum = bsum + 32'($signed(in_data[j*32 +: 16]));
        end
    end
    always_ff @(posedge clk) begin
        if (in_fire && wcnt < CW'(NB)) begin
            xbuf[wcnt[BAW-1:0]] <= in16;
            kbuf[wcnt[BAW-1:0]] <= in_keep;
        end
    end

    // ---------------- read pipeline (passes 2 and 3) ----------------
    // A buffer + g/b read | B c = x - mu | C c^2 (p2) or c*R (p3) | D Q accumulate (p2) or
    // z = sat16(rshr(c*R, 22+k)) | E z*g + b | F rdivpot, wrap32, sat8 -> output register
    logic vA, vB, vC, vD, vE, vF;
    logic p3A, p3B, p3C, p3D;
    logic lA, lB, lC, lD, lE;
    logic [L-1:0] kA, kB, kC, kD, kE;
    logic [L*16-1:0] xA;
    logic en, issue, issuing;
    logic [47:0] qacc;
    logic signed [31:0] rr;          // R = y * sqrt_d
    logic [5:0] shz;                 // 22 + k  (15..38)

    assign en        = !vF || out_ready;
    assign issuing   = (st == S_P2 || st == S_P3) && (rcnt < nbeats);
    assign issue     = en && issuing;
    assign out_valid = vF;

    // g/b read address of the beat being issued
    logic [31:0]     e0;
    logic [GAW-1:0]  gaddr_r;
    logic [BAW2-1:0] baddr_r;
    assign e0      = 32'(rcnt) * 32'(L);
    assign gaddr_r = GAW'(slot) * GAW'(WG) + GAW'(e0 / 32'(GW));
    assign baddr_r = BAW2'(slot) * BAW2'(WB) + BAW2'(e0 / 32'(BW));
    logic [127:0] gw_A [NGB];          // stage A: the g / b words of the beat
    logic [127:0] bw_A [NBB];
    logic [4:0]   goffA, boffA;        // lane-0 position inside the words (L < 8 / L < 4)
    genvar k;
    generate
        for (k = 0; k < NGB; k++) begin : g_gbank
            logic [127:0] mem [NSLOT * WG];
            wire  [GAW-1:0] wa = GAW'(blk_slot[SLW-1:0]) * GAW'(WG) + GAW'(lg / 12'(NGB));
            always_ff @(posedge clk) begin
                if (w_g && (lg % 12'(NGB)) == 12'(k)) mem[wa] <= blk_data;
                if (en) gw_A[k] <= mem[gaddr_r];
            end
        end
        for (k = 0; k < NBB; k++) begin : g_bbank
            logic [127:0] mem [NSLOT * WB];
            wire  [BAW2-1:0] wa = BAW2'(blk_slot[SLW-1:0]) * BAW2'(WB) + BAW2'(lb / 12'(NBB));
            always_ff @(posedge clk) begin
                if (w_b && (lb % 12'(NBB)) == 12'(k)) mem[wa] <= blk_data;
                if (en) bw_A[k] <= mem[baddr_r];
            end
        end
    endgenerate
    always_ff @(posedge clk) begin
        if (en) begin
            goffA <= 5'(e0 % 32'(GW));
            boffA <= 5'(e0 % 32'(BW));
        end
    end

    always_ff @(posedge clk) begin
        if (en) begin
            xA  <= xbuf[rcnt[BAW-1:0]];
            kA  <= kbuf[rcnt[BAW-1:0]];
            lA  <= (rcnt == nbeats - 1'b1);
            p3A <= (st == S_P3);
            kB <= kA;  lB <= lA;  p3B <= p3A;
            kC <= kB;  lC <= lB;  p3C <= p3B;
            kD <= kC;  lD <= lC;  p3D <= p3C;
            kE <= kD;  lE <= lD;
            out_keep <= kE;
            out_last <= lE;
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) {vA, vB, vC, vD, vE, vF} <= '0;
        else if (en) begin
            vA <= issue;
            vB <= vA;
            vC <= vB;
            vD <= vC && p3C;
            vE <= vD;
            vF <= vE;
        end
    end

    logic [35:0] sq_l   [L];
    logic [L-1:0] zsatD, zsatE, zsatF, osat_r;

    genvar j;
    generate
        for (j = 0; j < L; j++) begin : g_lane
            // this lane's g and b, selected out of the stage-A words
            // 8-bit lane positions: 5 bits wrapped for j >= 32 (L = 64 fix, SG7b)
            wire [7:0] gp = 8'(goffA) + 8'(j);
            wire [7:0] bp = 8'(boffA) + 8'(j);
            wire [127:0] gwl = gw_A[gp / 8'd8];
            wire [127:0] bwl = bw_A[bp / 8'd4];
            wire signed [15:0] gA = gwl[(gp % 8'd8) * 16 +: 16];
            wire signed [31:0] bA = bwl[(bp % 8'd4) * 32 +: 32];
            logic signed [15:0] gB, gC, gD;
            logic signed [31:0] bB, bC, bD;
            always_ff @(posedge clk) begin
                if (en) begin
                    gB <= gA; bB <= bA; gC <= gB; bC <= bB; gD <= gC; bD <= bC;
                end
            end

            logic signed [17:0] cB;
            logic signed [35:0] sqC;
            logic signed [49:0] prC;
            logic signed [15:0] zD;
            logic signed [33:0] accE;
            wire  signed [15:0] xa = $signed(xA[j*16 +: 16]);
            // stage D: rounding shift and int16 saturation
            wire  signed [49:0] rndD = prC + (50'sd1 <<< (shz - 6'd1));
            wire  signed [49:0] zw   = rndD >>> shz;
            wire  zhi = (zw > 50'sd32767);
            wire  zlo = (zw < -50'sd32768);
            // stage F: rdivpot(acc, sh), int32 wrap, int8 saturation
            wire  [33:0]        msk  = (34'd1 << r_sh) - 34'd1;
            wire  [33:0]        rem  = accE & msk;
            wire  [33:0]        thr  = (msk >> 1) + {33'd0, accE[33]};
            wire  signed [33:0] q34  = (accE >>> r_sh) + ((rem > thr) ? 34'sd1 : 34'sd0);
            wire  signed [31:0] q32  = q34[31:0];
            wire  ohi = (q32 > 32'sd127);
            wire  olo = (q32 < -32'sd128);
            always_ff @(posedge clk) begin
                if (en) begin
                    cB   <= 18'(xa) - 18'(mu);
                    sqC  <= cB * cB;
                    prC  <= 50'(cB) * 50'(rr);
                    zD   <= zhi ? 16'sh7FFF : zlo ? 16'sh8000 : zw[15:0];
                    zsatD[j] <= kC[j] && (zhi || zlo);
                    accE <= 34'(zD) * 34'(gD) + 34'(bD);
                    out_data[j*32 +: 32] <= !kE[j] ? 32'd0 :
                                            ohi ? 32'd127 : olo ? 32'hFFFF_FF80 : 32'(q32);
                    osat_r[j] <= kE[j] && (ohi || olo);
                end
            end
            assign sq_l[j] = kC[j] ? sqC : 36'd0;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (en) begin
            zsatE <= zsatD;
            zsatF <= zsatE;
        end
    end

    logic [47:0] sqsum;
    always_comb begin
        sqsum = '0;
        for (int i = 0; i < L; i++) sqsum = sqsum + 48'(sq_l[i]);
    end

    // ---------------- scalar units ----------------
    logic signed [31:0] sr_y;
    srdhm u_srdhm (.clk(clk), .en(st == S_MU), .a(ssum), .b(r_mult), .y(sr_y));
    // rdivpot(sr_y, mu_shift) (C2b, 32-bit)
    logic [31:0]        mu_msk, mu_rem, mu_thr;
    logic signed [31:0] mu_new;
    assign mu_msk = (32'd1 << r_mush) - 32'd1;
    assign mu_rem = sr_y & mu_msk;
    assign mu_thr = (mu_msk >> 1) + {31'd0, sr_y[31]};
    assign mu_new = (sr_y >>> r_mush) + ((mu_rem > mu_thr) ? 32'sd1 : 32'sd0);

    logic [47:0] qfin;
    logic signed [21:0] rs_y;
    logic signed [7:0]  rs_k;
    always_comb begin
        qfin = qacc + r_eps;
        if (qfin == '0) qfin = 48'd1;
    end
    sfu_rsqrt #(.QW(48)) u_rsqrt (
        .clk(clk), .en(st == S_RS), .q(qfin), .y(rs_y), .k(rs_k),
        .lut_we(lut_we), .lut_waddr(lut_waddr), .lut_wdata(lut_wdata)
    );
    logic signed [39:0] rprod;
    assign rprod = 40'(rs_y) * $signed({23'd0, r_sqrtd});

    // ---------------- control ----------------
    logic [2:0] mc;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IN; wcnt <= '0; err_len <= 1'b0; ssum <= '0; rcnt <= '0;
            nbeats <= '0; mc <= '0; qacc <= '0;
        end else begin
            case (st)
                S_IN: if (in_fire) begin
                    if (wcnt == '0) begin
                        slot    <= in_slot[SLW-1:0];
                        r_mult  <= k_mult [in_slot[SLW-1:0]];
                        r_mush  <= k_mush [in_slot[SLW-1:0]];
                        r_sqrtd <= k_sqrtd[in_slot[SLW-1:0]];
                        r_eps   <= k_eps  [in_slot[SLW-1:0]];
                        r_sh    <= k_sh   [in_slot[SLW-1:0]];
                        ssum    <= bsum;
                    end else begin
                        ssum    <= ssum + bsum;
                    end
                    if (wcnt >= CW'(NB)) err_len <= 1'b1;
                    if (in_last) begin
                        nbeats <= wcnt + 1'b1;
                        wcnt   <= '0;
                        mc     <= '0;
                        st     <= S_MU;
                    end else begin
                        wcnt <= wcnt + 1'b1;
                    end
                end
                S_MU: begin
                    mc <= mc + 1'b1;
                    if (mc == 3'd3) begin
                        mu   <= mu_new;
                        rcnt <= '0;
                        qacc <= '0;
                        st   <= S_P2;
                    end
                end
                S_P2: begin
                    if (issue) rcnt <= rcnt + 1'b1;
                    if (en && vC && !p3C) qacc <= qacc + sqsum;
                    // leave as the last c^2 sum lands
                    if (!issuing && !vA && !vB) begin
                        mc <= '0;
                        st <= S_RS;
                    end
                end
                S_RS: begin
                    mc <= mc + 1'b1;
                    if (mc == 3'd5) begin
                        rr   <= rprod[31:0];
                        shz  <= 6'(8'sd22 + rs_k);
                        rcnt <= '0;
                        st   <= S_P3;
                    end
                end
                default: begin // S_P3
                    if (issue) rcnt <= rcnt + 1'b1;
                    if (!issuing && !vA && !vB && !vC && !vD && !vE && (!vF || out_ready))
                        st <= S_IN;
                end
            endcase
        end
    end

    // ---------------- saturation counters ----------------
    logic xfer, zst, ost;
    assign xfer = out_valid && out_ready;
    sat_counter #(.N(L), .W(32)) u_zc (.clk(clk), .rst_n(rst_n), .clr(sat_clr), .en(xfer),
                                       .inc(zsatF), .count(z_sat_count), .sticky(zst));
    sat_counter #(.N(L), .W(32)) u_oc (.clk(clk), .rst_n(rst_n), .clr(sat_clr), .en(xfer),
                                       .inc(osat_r), .count(out_sat_count), .sticky(ost));
    assign sat_sticky = zst | ost;
endmodule
