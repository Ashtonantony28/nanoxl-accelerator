// cmd_decode.sv -- classify one 128-bit command word (docs/isa.md, contract C4).
//
// Combinational. Field extraction comes from the generated cmd_fields.sv. A word is legal iff
// nanoxl.isa.decode() accepts it (the cocotb test checks this equivalence on random words):
//   ERR_OPCODE  opcode not defined in the ISA (0x00 and every unlisted value; there are no
//               "reserved but ignored" opcodes: a future opcode is illegal until this RTL is
//               regenerated from the new nanoxl/isa.py)
//   ERR_RESV    a reserved bit above the opcode's last field is set
//   ERR_FIELD   isa.validate() rules: WAIT/SIGNAL queue = 3 (CTRL), WAIT count = 0,
//               DMA sp_buf > 4, VEC vop undefined / buffer id > 4 / dtype 3,
//               GEMM int4 with w_trans or with odd N, GEMM_GS odd N, LUT_LOAD table = 3
// route: 0 DMA, 1 ARRAY, 2 VEC queue, 3 CTRL (END/IRQ barriers).
// kind:  0 engine command, 1 WAIT, 2 SIGNAL, 3 IRQ, 4 END.
module cmd_decode (
    input  logic [127:0] w,
    output logic [3:0]   err,
    output logic [1:0]   route,
    output logic [2:0]   kind,
    output logic [4:0]   sem,
    output logic [7:0]   wait_count,
    output logic [15:0]  irq_code
);
    localparam logic [3:0] ERR_NONE = 4'd0, ERR_OPCODE = 4'd1, ERR_RESV = 4'd2, ERR_FIELD = 4'd3;
    localparam logic [2:0] K_ENGINE = 3'd0, K_WAIT = 3'd1, K_SIGNAL = 3'd2, K_IRQ = 3'd3, K_END = 3'd4;

    /* verilator lint_off UNUSEDSIGNAL */
    logic [5:0]  opcode;
    logic        known, resv_ok;
    logic [1:0]  fixed_q;
    logic        is_end, is_irq, is_wait, is_signal, is_dma_load, is_dma_store, is_gemm, is_vec;
    logic [1:0]  wait_queue, signal_queue;
    logic [4:0]  wait_sem, signal_sem;
    logic [7:0]  wait_cnt_f;
    logic [2:0]  ld_buf, st_buf;
    logic [4:0]  vop;
    logic [1:0]  dt_in, dt_out;
    logic [2:0]  b0, b1, bd;
    logic        g_int4, g_wtrans;
    logic [12:0] g_n, gs_n;
    logic        is_gemm_gs, is_lut_load;
    logic [1:0]  lut_table;

    // only the fields used here are connected (additive ISA changes need no edit here)
    /* verilator lint_off PINMISSING */
    cmd_fields u_f (
        .w(w), .opcode(opcode), .known(known), .resv_ok(resv_ok), .fixed_q(fixed_q),
        .is_end(is_end), .is_irq(is_irq), .is_wait(is_wait), .is_signal(is_signal),
        .is_dma_load(is_dma_load), .is_dma_store(is_dma_store), .is_gemm(is_gemm), .is_vec(is_vec),
        .irq_code(irq_code),
        .wait_queue(wait_queue), .wait_sem(wait_sem), .wait_count(wait_cnt_f),
        .signal_queue(signal_queue), .signal_sem(signal_sem),
        .dma_load_sp_buf(ld_buf), .dma_store_sp_buf(st_buf),
        .gemm_int4(g_int4), .gemm_w_trans(g_wtrans), .gemm_n(g_n),
        .is_gemm_gs(is_gemm_gs), .gemm_gs_n(gs_n), .is_lut_load(is_lut_load), .lut_load_table(lut_table),
        .vec_vop(vop), .vec_dt_in(dt_in), .vec_dt_out(dt_out), .vec_src0_buf(b0),
        .vec_src1_buf(b1), .vec_dst_buf(bd)
    );
    /* verilator lint_on PINMISSING */
    /* verilator lint_on UNUSEDSIGNAL */

    logic field_bad;
    always_comb begin
        field_bad = 1'b0;
        if (is_wait   && (wait_queue == 2'd3 || wait_cnt_f == 8'd0)) field_bad = 1'b1;
        if (is_signal && signal_queue == 2'd3)                       field_bad = 1'b1;
        if (is_dma_load  && ld_buf > 3'd4)                           field_bad = 1'b1;
        if (is_dma_store && st_buf > 3'd4)                           field_bad = 1'b1;
        if (is_vec && (vop > 5'd8 || b0 > 3'd4 || b1 > 3'd4 || bd > 3'd4 ||
                       dt_in == 2'd3 || dt_out == 2'd3))             field_bad = 1'b1;
        if (is_gemm && g_int4 && (g_wtrans || g_n[0]))               field_bad = 1'b1;
        if (is_gemm_gs && gs_n[0])                                   field_bad = 1'b1;
        if (is_lut_load && lut_table == 2'd3)                        field_bad = 1'b1;
    end

    always_comb begin
        if (!known)        err = ERR_OPCODE;
        else if (!resv_ok) err = ERR_RESV;
        else if (field_bad) err = ERR_FIELD;
        else               err = ERR_NONE;
        sem        = is_wait ? wait_sem : signal_sem;
        wait_count = wait_cnt_f;
        if (is_wait) begin
            kind = K_WAIT;   route = wait_queue;
        end else if (is_signal) begin
            kind = K_SIGNAL; route = signal_queue;
        end else if (is_irq) begin
            kind = K_IRQ;    route = 2'd3;
        end else if (is_end) begin
            kind = K_END;    route = 2'd3;
        end else begin
            kind = K_ENGINE; route = fixed_q;
        end
    end
endmodule
