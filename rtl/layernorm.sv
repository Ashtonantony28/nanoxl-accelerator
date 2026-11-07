// layernorm.sv -- C5 integer LayerNorm over one row (ADR-002), bit-exact against
// nanoxl.quant.ops.ln_norm_i + ln_affine_i.
//
// Input: int16 residual values x_i (sign-extended in int32 lanes; low 16 bits used),
// L-lane beats ending with in_last; in_slot selects the LayerNorm instance (constants and
// per-channel g/b, loaded through the parameter ports). Lanes with keep = 0 are excluded
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
    // rsqrt table
    input  logic            lut_we,
    input  logic [7:0]      lut_waddr,
    input  logic [15:0]     lut_wdata,
    // per-channel affine parameters: element par_idx of slot par_slot
    input  logic            par_we,
    input  logic [7:0]      par_slot,
    input  logic [15:0]     par_idx,
    input  logic [15:0]     par_g,
    input  logic [31:0]     par_b,
    // constants .c = [mu_mult, mu_shift, sqrt_d, eps_q, sh] (c_idx 0..4) of slot c_slot
    input  logic            c_we,
    input  logic [7:0]      c_slot,
    input  logic [2:0]      c_idx,
    input  logic [47:0]     c_data,
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
    localparam int PAW = $clog2(NSLOT * NB + 1);
    localparam int CW  = 16;

    typedef enum logic [2:0] {S_IN, S_MU, S_P2, S_RS, S_P3} state_t;
    state_t st;

    // ---------------- constants per slot ----------------
    logic [31:0] k_mult  [NSLOT];
    logic [4:0]  k_mush  [NSLOT];
    logic [16:0] k_sqrtd [NSLOT];
    logic [47:0] k_eps   [NSLOT];
    logic [4:0]  k_sh    [NSLOT];
    always_ff @(posedge clk) begin
        if (c_we && c_slot < 8'(NSLOT)) begin
            case (c_idx)
                3'd0: k_mult [c_slot[SLW-1:0]] <= c_data[31:0];
                3'd1: k_mush [c_slot[SLW-1:0]] <= c_data[4:0];
                3'd2: k_sqrtd[c_slot[SLW-1:0]] <= c_data[16:0];
                3'd3: k_eps  [c_slot[SLW-1:0]] <= c_data[47:0];
                3'd4: k_sh   [c_slot[SLW-1:0]] <= c_data[4:0];
                default: ;
            endcase
        end
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

    logic [PAW-1:0] paddr_r;
    assign paddr_r = PAW'(slot) * PAW'(NB) + PAW'(rcnt[BAW-1:0]);

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
            // per-lane g/b memory: entry slot*NB + beat
            logic [15:0] gmem [NSLOT * NB];
            logic [31:0] bmem [NSLOT * NB];
            wire  [PAW-1:0] paddr_w = PAW'(par_slot[SLW-1:0]) * PAW'(NB) + PAW'(par_idx / 16'(L));
            always_ff @(posedge clk) begin
                if (par_we && (par_idx % 16'(L)) == 16'(j) && par_slot < 8'(NSLOT) &&
                    par_idx < 16'(NB * L)) begin
                    gmem[paddr_w] <= par_g;
                    bmem[paddr_w] <= par_b;
                end
            end
            logic signed [15:0] gA, gB, gC, gD;
            logic signed [31:0] bA, bB, bC, bD;
            always_ff @(posedge clk) begin
                if (en) begin
                    gA <= gmem[paddr_r];
                    bA <= bmem[paddr_r];
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
