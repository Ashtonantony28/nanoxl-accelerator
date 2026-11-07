// sfu.sv -- special-function block for the vector unit's SFU hook (docs/vector-unit.md
// §5.6, docs/sfu.md). Implements the C5 nonlinear ops of ADR-002 bit-exactly.
//
//   sub-op (cmd_op = 0x18 + sub-op)   unit        lane in            lane out
//   0 EXP2      elementwise            sfu_elem    d uint16           e Q.15
//   1 RECIP     elementwise            sfu_elem    S uint32           {b, y} packed
//   2 RSQRT     elementwise            sfu_elem    Q uint32           {k, y} packed
//   3 GELU      elementwise            sfu_elem    int8 (imm = table slot = layer)  int8
//   4 SOFTMAX   row (len <= CTX_MAX)   softmax     int16 Q.8 logits   p int8 [0,127]
//   5 LN        row (len <= D_MAX)     layernorm   int16 (imm = LN slot)            int8
//   6, 7        reserved: data passed through unchanged, sticky illegal_op
// Hook rules met: one output beat per input beat, in order, keep/last unchanged. A row is
// the run of beats ending in in_last; in_op and in_imm are constant over a row. Up to
// RECQ rows may be in flight across the three units; the output mux follows a queue of
// the unit of each row, so rows leave in arrival order. in_ready depends combinationally
// on out_ready (through the elementwise pipeline's stall) and never on in_valid.
//
// Configuration write port (tables and LN parameters are model-bundle data, written while
// the unit is idle; address = region << 24 | slot << 16 | index):
//   region 0 lut.exp2[index]   1 lut.recip[index]   2 lut.rsqrt[index]   (wdata[15:0])
//   region 3 h{slot}.gelu.lut[index]                                       (wdata[7:0])
//   region 4 LN slot: g[index] = wdata[15:0], b[index] = wdata[47:16]
//   region 5 LN slot: .c[index] (index 0..4 = mu_mult, mu_shift, sqrt_d, eps_q, sh), wdata[47:0]
// Status: LayerNorm saturation counters (z and out), sticky flags, err_len (a softmax or
// LN row longer than its buffer), busy.
module sfu #(
    parameter int L       = 4,
    parameter int CTX_MAX = 256,
    parameter int D_MAX   = 384,
    parameter int NG      = 4,
    parameter int NSLOT   = 16
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            in_valid,
    output logic            in_ready,
    input  logic [2:0]      in_op,
    input  logic [1:0]      in_fmt,
    input  logic [4:0]      in_shamt,
    input  logic [31:0]     in_imm,
    input  logic [L*32-1:0] in_data,
    input  logic [L-1:0]    in_keep,
    input  logic            in_last,
    output logic            out_valid,
    input  logic            out_ready,
    output logic [L*32-1:0] out_data,
    output logic [L-1:0]    out_keep,
    output logic            out_last,
    input  logic            cfg_we,
    input  logic [31:0]     cfg_addr,
    input  logic [63:0]     cfg_wdata,
    input  logic            stat_clr,
    output logic [31:0]     ln_z_sat_count,
    output logic [31:0]     ln_out_sat_count,
    output logic            sat_sticky,
    output logic            illegal_op,
    output logic            err_len,
    output logic            busy
);
    localparam logic [1:0] U_EL = 2'd0, U_SM = 2'd1, U_LN = 2'd2;
    localparam int RECQ = 4;

    function automatic logic [1:0] unit_of(input logic [2:0] op);
        unit_of = (op == 3'd4) ? U_SM : (op == 3'd5) ? U_LN : U_EL;
    endfunction

    // ---------------- configuration decode (registered) ----------------
    logic [3:0]  cw_region;
    logic        cw_we;
    logic [7:0]  cw_slot;
    logic [15:0] cw_idx;
    logic [63:0] cw_data;
    always_ff @(posedge clk) begin
        if (!rst_n) cw_we <= 1'b0;
        else        cw_we <= cfg_we;
        cw_region <= cfg_addr[27:24];
        cw_slot   <= cfg_addr[23:16];
        cw_idx    <= cfg_addr[15:0];
        cw_data   <= cfg_wdata;
    end
    logic [3:0] el_we;
    logic [1:0] sm_we;
    logic       ln_lut_we, ln_par_we, ln_c_we;
    always_comb begin
        el_we     = '0;
        sm_we     = '0;
        ln_lut_we = 1'b0;
        ln_par_we = cw_we && cw_region == 4'd4;
        ln_c_we   = cw_we && cw_region == 4'd5;
        if (cw_we) begin
            case (cw_region)
                4'd0: begin el_we[0] = 1'b1; sm_we[0] = 1'b1; end
                4'd1: begin el_we[1] = 1'b1; sm_we[1] = 1'b1; end
                4'd2: begin el_we[2] = 1'b1; ln_lut_we = 1'b1; end
                4'd3: el_we[3] = 1'b1;
                default: ;
            endcase
        end
    end
    logic [15:0] el_waddr;
    assign el_waddr = (cw_region == 4'd3) ? {cw_slot, cw_idx[7:0]} : {8'd0, cw_idx[7:0]};

    // ---------------- input routing ----------------
    logic [1:0] sel;
    logic       mid;                      // inside a row (a non-last beat was accepted)
    logic       el_in_ready, sm_in_ready, ln_in_ready, u_rdy;
    logic [1:0] rq_unit [RECQ];
    logic [$clog2(RECQ):0]   rq_cnt;
    logic [$clog2(RECQ)-1:0] rq_wp, rq_rp;
    logic rq_full, rq_empty, push, pop, in_fire;

    assign sel      = unit_of(in_op);
    assign rq_full  = (rq_cnt == ($clog2(RECQ)+1)'(RECQ));
    assign rq_empty = (rq_cnt == '0);
    always_comb begin
        case (sel)
            U_SM:    u_rdy = sm_in_ready;
            U_LN:    u_rdy = ln_in_ready;
            default: u_rdy = el_in_ready;
        endcase
    end
    assign in_ready = u_rdy && (mid || !rq_full);
    assign in_fire  = in_valid && in_ready;
    assign push     = in_fire && !mid;

    // ---------------- units ----------------
    logic el_out_valid, sm_out_valid, ln_out_valid;
    logic el_out_ready, sm_out_ready, ln_out_ready;
    logic [L*32-1:0] el_out_data, sm_out_data, ln_out_data;
    logic [L-1:0] el_out_keep, sm_out_keep, ln_out_keep;
    logic el_out_last, sm_out_last, ln_out_last;
    logic sm_busy, ln_busy, sm_err, ln_err;

    sfu_elem #(.L(L), .NG(NG)) u_elem (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid && in_ready && sel == U_EL), .in_ready(el_in_ready),
        .in_op(in_op), .in_slot(in_imm[7:0]), .in_data(in_data), .in_keep(in_keep),
        .in_last(in_last),
        .out_valid(el_out_valid), .out_ready(el_out_ready), .out_data(el_out_data),
        .out_keep(el_out_keep), .out_last(el_out_last),
        .lut_we(el_we), .lut_waddr(el_waddr), .lut_wdata(cw_data[15:0])
    );

    softmax #(.L(L), .CTX_MAX(CTX_MAX)) u_sm (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid && in_ready && sel == U_SM), .in_ready(sm_in_ready),
        .in_data(in_data), .in_keep(in_keep), .in_last(in_last),
        .out_valid(sm_out_valid), .out_ready(sm_out_ready), .out_data(sm_out_data),
        .out_keep(sm_out_keep), .out_last(sm_out_last),
        .lut_we(sm_we), .lut_waddr(cw_idx[7:0]), .lut_wdata(cw_data[15:0]),
        .busy(sm_busy), .err_len(sm_err)
    );

    logic ln_sticky;
    layernorm #(.L(L), .D_MAX(D_MAX), .NSLOT(NSLOT)) u_ln (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid && in_ready && sel == U_LN), .in_ready(ln_in_ready),
        .in_slot(in_imm[7:0]), .in_data(in_data), .in_keep(in_keep), .in_last(in_last),
        .out_valid(ln_out_valid), .out_ready(ln_out_ready), .out_data(ln_out_data),
        .out_keep(ln_out_keep), .out_last(ln_out_last),
        .lut_we(ln_lut_we), .lut_waddr(cw_idx[7:0]), .lut_wdata(cw_data[15:0]),
        .par_we(ln_par_we), .par_slot(cw_slot), .par_idx(cw_idx), .par_g(cw_data[15:0]),
        .par_b(cw_data[47:16]),
        .c_we(ln_c_we), .c_slot(cw_slot), .c_idx(cw_idx[2:0]), .c_data(cw_data[47:0]),
        .sat_clr(stat_clr), .z_sat_count(ln_z_sat_count), .out_sat_count(ln_out_sat_count),
        .sat_sticky(ln_sticky), .busy(ln_busy), .err_len(ln_err)
    );

    // ---------------- in-order output mux ----------------
    logic [1:0] h;
    assign h = rq_unit[rq_rp];
    always_comb begin
        out_valid = 1'b0; out_data = '0; out_keep = '0; out_last = 1'b0;
        if (!rq_empty) begin
            case (h)
                U_SM: begin
                    out_valid = sm_out_valid; out_data = sm_out_data;
                    out_keep = sm_out_keep; out_last = sm_out_last;
                end
                U_LN: begin
                    out_valid = ln_out_valid; out_data = ln_out_data;
                    out_keep = ln_out_keep; out_last = ln_out_last;
                end
                default: begin
                    out_valid = el_out_valid; out_data = el_out_data;
                    out_keep = el_out_keep; out_last = el_out_last;
                end
            endcase
        end
    end
    assign el_out_ready = out_ready && !rq_empty && h == U_EL;
    assign sm_out_ready = out_ready && !rq_empty && h == U_SM;
    assign ln_out_ready = out_ready && !rq_empty && h == U_LN;
    assign pop = out_valid && out_ready && out_last;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rq_cnt <= '0; rq_wp <= '0; rq_rp <= '0; mid <= 1'b0;
        end else begin
            if (push) begin rq_unit[rq_wp] <= sel; rq_wp <= rq_wp + 1'b1; end
            if (pop) rq_rp <= rq_rp + 1'b1;
            rq_cnt <= rq_cnt + {{$clog2(RECQ){1'b0}}, push} - {{$clog2(RECQ){1'b0}}, pop};
            if (in_fire) mid <= !in_last;
        end
    end

    // ---------------- status ----------------
    always_ff @(posedge clk) begin
        if (!rst_n) illegal_op <= 1'b0;
        else illegal_op <= (stat_clr ? 1'b0 : illegal_op) | (push && in_op[2:1] == 2'b11);
    end
    assign sat_sticky = ln_sticky;
    assign err_len    = sm_err | ln_err;
    assign busy       = !rq_empty || mid || sm_busy || ln_busy;

    // in_fmt / in_shamt are not used by any C5 op (kept for hook compatibility).
    logic unused;
    assign unused = ^{in_fmt, in_shamt, in_imm[31:8]};
endmodule
