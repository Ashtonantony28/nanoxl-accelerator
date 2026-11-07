// ctrl_top.sv -- NanoXL control plane (SG6 / M4): CSRs over AXI4-Lite, command fetcher (AXI4
// read master "mf_axi"), command processor (decode, DMA/ARRAY/VEC queues, semaphores, barriers),
// and two DMA engines selected by the command's chan[0] (SG10a, ADR-005): channel 0 = AXI4 master
// "md_axi" + scratchpad line port "sp_", channel 1 = "md1_axi" + "sp1_".
//
// The ARRAY and VEC engines are external (SG7 wires the array and vector-unit sequencers here;
// the tests drive stubs). Their ports: arr_cmd_valid/ready + arr_cmd_word (128-bit ISA word),
// arr_done (one pulse per accepted command, in order, at least one cycle after acceptance),
// arr_err (level; halts the processor). Same for vec_*. eng_clr pulses on a soft reset.
// See docs/control.md for the register map and the protocols.
module ctrl_top #(
    parameter int AW     = 32,
    parameter int IDW    = 4,
    parameter int PW     = 24,
    parameter int FDEPTH = 16,
    parameter int FBURST = 8,
    parameter int QDEPTH = 8,
    parameter int MAXOUT = 4,
    parameter int MAXB   = 256,
    parameter int LDF    = 512,
    parameter int NCMD   = 4,       // commands in flight per DMA engine (LOAD pipelining)
    parameter int NSAT   = 5
) (
    input  logic               clk,
    input  logic               rst_n,
    output logic               irq,
    output logic               eng_clr,
    output logic               sat_clr,
    input  logic [NSAT*32-1:0] sat_cnt,
    // ---- CSR AXI4-Lite slave ----
    input  logic [11:0]        s_axil_awaddr,
    input  logic [2:0]         s_axil_awprot,
    input  logic               s_axil_awvalid,
    output logic               s_axil_awready,
    input  logic [31:0]        s_axil_wdata,
    input  logic [3:0]         s_axil_wstrb,
    input  logic               s_axil_wvalid,
    output logic               s_axil_wready,
    output logic [1:0]         s_axil_bresp,
    output logic               s_axil_bvalid,
    input  logic               s_axil_bready,
    input  logic [11:0]        s_axil_araddr,
    input  logic [2:0]         s_axil_arprot,
    input  logic               s_axil_arvalid,
    output logic               s_axil_arready,
    output logic [31:0]        s_axil_rdata,
    output logic [1:0]         s_axil_rresp,
    output logic               s_axil_rvalid,
    input  logic               s_axil_rready,
    // ---- command fetch AXI4 master (read only) ----
    output logic [IDW-1:0]     mf_axi_arid,
    output logic [AW-1:0]      mf_axi_araddr,
    output logic [7:0]         mf_axi_arlen,
    output logic [2:0]         mf_axi_arsize,
    output logic [1:0]         mf_axi_arburst,
    output logic               mf_axi_arlock,
    output logic [3:0]         mf_axi_arcache,
    output logic [2:0]         mf_axi_arprot,
    output logic               mf_axi_arvalid,
    input  logic               mf_axi_arready,
    input  logic [IDW-1:0]     mf_axi_rid,
    input  logic [127:0]       mf_axi_rdata,
    input  logic [1:0]         mf_axi_rresp,
    input  logic               mf_axi_rlast,
    input  logic               mf_axi_rvalid,
    output logic               mf_axi_rready,
    // ---- DMA channel 0 AXI4 master ----
    output logic [IDW-1:0]     md_axi_awid,
    output logic [AW-1:0]      md_axi_awaddr,
    output logic [7:0]         md_axi_awlen,
    output logic [2:0]         md_axi_awsize,
    output logic [1:0]         md_axi_awburst,
    output logic               md_axi_awlock,
    output logic [3:0]         md_axi_awcache,
    output logic [2:0]         md_axi_awprot,
    output logic               md_axi_awvalid,
    input  logic               md_axi_awready,
    output logic [127:0]       md_axi_wdata,
    output logic [15:0]        md_axi_wstrb,
    output logic               md_axi_wlast,
    output logic               md_axi_wvalid,
    input  logic               md_axi_wready,
    input  logic [IDW-1:0]     md_axi_bid,
    input  logic [1:0]         md_axi_bresp,
    input  logic               md_axi_bvalid,
    output logic               md_axi_bready,
    output logic [IDW-1:0]     md_axi_arid,
    output logic [AW-1:0]      md_axi_araddr,
    output logic [7:0]         md_axi_arlen,
    output logic [2:0]         md_axi_arsize,
    output logic [1:0]         md_axi_arburst,
    output logic               md_axi_arlock,
    output logic [3:0]         md_axi_arcache,
    output logic [2:0]         md_axi_arprot,
    output logic               md_axi_arvalid,
    input  logic               md_axi_arready,
    input  logic [IDW-1:0]     md_axi_rid,
    input  logic [127:0]       md_axi_rdata,
    input  logic [1:0]         md_axi_rresp,
    input  logic               md_axi_rlast,
    input  logic               md_axi_rvalid,
    output logic               md_axi_rready,
    // ---- DMA channel 1 AXI4 master (SG10a, ADR-005) ----
    output logic [IDW-1:0]     md1_axi_awid,
    output logic [AW-1:0]      md1_axi_awaddr,
    output logic [7:0]         md1_axi_awlen,
    output logic [2:0]         md1_axi_awsize,
    output logic [1:0]         md1_axi_awburst,
    output logic               md1_axi_awlock,
    output logic [3:0]         md1_axi_awcache,
    output logic [2:0]         md1_axi_awprot,
    output logic               md1_axi_awvalid,
    input  logic               md1_axi_awready,
    output logic [127:0]       md1_axi_wdata,
    output logic [15:0]        md1_axi_wstrb,
    output logic               md1_axi_wlast,
    output logic               md1_axi_wvalid,
    input  logic               md1_axi_wready,
    input  logic [IDW-1:0]     md1_axi_bid,
    input  logic [1:0]         md1_axi_bresp,
    input  logic               md1_axi_bvalid,
    output logic               md1_axi_bready,
    output logic [IDW-1:0]     md1_axi_arid,
    output logic [AW-1:0]      md1_axi_araddr,
    output logic [7:0]         md1_axi_arlen,
    output logic [2:0]         md1_axi_arsize,
    output logic [1:0]         md1_axi_arburst,
    output logic               md1_axi_arlock,
    output logic [3:0]         md1_axi_arcache,
    output logic [2:0]         md1_axi_arprot,
    output logic               md1_axi_arvalid,
    input  logic               md1_axi_arready,
    input  logic [IDW-1:0]     md1_axi_rid,
    input  logic [127:0]       md1_axi_rdata,
    input  logic [1:0]         md1_axi_rresp,
    input  logic               md1_axi_rlast,
    input  logic               md1_axi_rvalid,
    output logic               md1_axi_rready,
    // ---- DMA channel 0 scratchpad line port ----
    output logic               sp_wvalid,
    input  logic               sp_wready,
    output logic [2:0]         sp_wbuf,
    output logic [16:0]        sp_waddr,
    output logic [127:0]       sp_wdata,
    output logic [15:0]        sp_wstrb,
    output logic               sp_arvalid,
    input  logic               sp_arready,
    output logic [2:0]         sp_arbuf,
    output logic [16:0]        sp_araddr,
    input  logic               sp_rvalid,
    input  logic [127:0]       sp_rdata,
    // ---- DMA channel 1 scratchpad line port ----
    output logic               sp1_wvalid,
    input  logic               sp1_wready,
    output logic [2:0]         sp1_wbuf,
    output logic [16:0]        sp1_waddr,
    output logic [127:0]       sp1_wdata,
    output logic [15:0]        sp1_wstrb,
    output logic               sp1_arvalid,
    input  logic               sp1_arready,
    output logic [2:0]         sp1_arbuf,
    output logic [16:0]        sp1_araddr,
    input  logic               sp1_rvalid,
    input  logic [127:0]       sp1_rdata,
    // ---- ARRAY / VEC engine ports ----
    output logic               arr_cmd_valid,
    input  logic               arr_cmd_ready,
    output logic [127:0]       arr_cmd_word,
    input  logic               arr_done,
    input  logic               arr_err,
    output logic               vec_cmd_valid,
    input  logic               vec_cmd_ready,
    output logic [127:0]       vec_cmd_word,
    input  logic               vec_done,
    input  logic               vec_err
);
    // ---------------- wires ----------------
    logic [31:0]   ring_base;
    logic [PW-1:0] ring_size, ring_tail, ring_head, err_index, last_index;
    logic          doorbell, step_mode, step_pulse, soft_req, cfg_ok;
    logic [4:0]    sem_sel;
    logic [7:0]    sem_val;
    logic          halted, irq_pulse, end_pulse, all_idle, q_overflow;
    logic [3:0]    err_code;
    logic [127:0]  err_word;
    logic [15:0]   irq_code;
    logic [2:0]    q_idle, eng_busy;
    logic [3:0]    retire_inc;
    logic          f_valid, f_ready, f_rerr, f_idle;
    logic [127:0]  f_word;
    logic [3:0]    e_valid, e_ready, e_done;
    logic [3*128-1:0] e_word;
    logic          dma_err, dma_idle, dma_bytes_valid;
    logic [17:0]   dma_bytes;
    logic [1:0]    d_err, d_idle, d_bv;
    logic [16:0]   d_bytes [2];
    logic          soft_busy, clr, cfg_err;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [1:0]    dma_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    // ---------------- soft reset: stop issuing, drain AXI + scratchpad reads, then clear ----------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            soft_busy <= 1'b0;
            clr       <= 1'b0;
        end else begin
            clr <= 1'b0;
            if (soft_req) soft_busy <= 1'b1;
            if (soft_busy && !clr && f_idle && dma_idle) begin
                clr       <= 1'b1;
                soft_busy <= 1'b0;
            end
        end
    end
    assign eng_clr = clr;
    assign cfg_err = doorbell && !cfg_ok;

    csr_axil #(.PW(PW), .NSAT(NSAT)) u_csr (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awprot(s_axil_awprot), .s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready), .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
        .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready), .s_axil_bresp(s_axil_bresp),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready), .s_axil_araddr(s_axil_araddr),
        .s_axil_arprot(s_axil_arprot), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .ring_base(ring_base), .ring_size(ring_size), .ring_tail(ring_tail), .doorbell(doorbell),
        .step_mode(step_mode), .step_pulse(step_pulse), .soft_req(soft_req), .sem_sel(sem_sel),
        .sat_clr(sat_clr), .irq(irq),
        .clr(clr), .soft_busy(soft_busy), .ring_head(ring_head), .cfg_ok(cfg_ok), .halted(halted),
        .err_code(err_code), .err_index(err_index), .err_word(err_word), .irq_pulse(irq_pulse),
        .irq_code(irq_code), .end_pulse(end_pulse), .all_idle(all_idle), .q_idle(q_idle),
        .eng_busy(eng_busy), .retire_inc(retire_inc), .last_index(last_index),
        .q_overflow(q_overflow), .sem_val(sem_val), .dma_bytes_valid(dma_bytes_valid),
        .dma_bytes(dma_bytes), .sat_cnt(sat_cnt));

    cmd_fetch #(.AW(AW), .IDW(IDW), .PW(PW), .FDEPTH(FDEPTH), .FBURST(FBURST)) u_fetch (
        .clk(clk), .rst_n(rst_n), .clr(clr), .en(!halted && !soft_busy),
        .ring_base(AW'(ring_base)), .ring_size(ring_size), .ring_tail(ring_tail),
        .ring_head(ring_head), .cfg_ok(cfg_ok), .axi_idle(f_idle),
        .m_axi_arid(mf_axi_arid), .m_axi_araddr(mf_axi_araddr), .m_axi_arlen(mf_axi_arlen),
        .m_axi_arsize(mf_axi_arsize), .m_axi_arburst(mf_axi_arburst), .m_axi_arlock(mf_axi_arlock),
        .m_axi_arcache(mf_axi_arcache), .m_axi_arprot(mf_axi_arprot), .m_axi_arvalid(mf_axi_arvalid),
        .m_axi_arready(mf_axi_arready), .m_axi_rid(mf_axi_rid), .m_axi_rdata(mf_axi_rdata),
        .m_axi_rresp(mf_axi_rresp), .m_axi_rlast(mf_axi_rlast), .m_axi_rvalid(mf_axi_rvalid),
        .m_axi_rready(mf_axi_rready),
        .out_valid(f_valid), .out_ready(f_ready), .out_word(f_word), .out_rerr(f_rerr));

    cmd_proc #(.PW(PW), .QDEPTH(QDEPTH), .MAXOUT(MAXOUT)) u_proc (
        .clk(clk), .rst_n(rst_n), .clr(clr), .freeze(soft_busy),
        .in_valid(f_valid), .in_ready(f_ready), .in_word(f_word), .in_rerr(f_rerr), .in_index(ring_head),
        .step_mode(step_mode), .step_pulse(step_pulse), .cfg_err(cfg_err), .dma_err(dma_err),
        .eng_err({vec_err, arr_err}),
        .eng_valid(e_valid), .eng_ready(e_ready), .eng_word(e_word), .eng_done(e_done),
        .halted(halted), .err_code(err_code), .err_index(err_index), .err_word(err_word),
        .irq_pulse(irq_pulse), .irq_code(irq_code), .end_pulse(end_pulse), .all_idle(all_idle),
        .q_idle(q_idle), .eng_busy(eng_busy), .retire_inc(retire_inc), .last_index(last_index),
        .q_overflow(q_overflow), .sem_dbg_sel(sem_sel), .sem_dbg_val(sem_val));

    assign arr_cmd_valid = e_valid[1];
    assign arr_cmd_word  = e_word[1*128 +: 128];
    assign vec_cmd_valid = e_valid[2];
    assign vec_cmd_word  = e_word[2*128 +: 128];
    assign e_ready[1]    = arr_cmd_ready;
    assign e_ready[2]    = vec_cmd_ready;
    assign e_done[1]     = arr_done;
    assign e_done[2]     = vec_done;

    // both DMA channels: errors, idle (soft-reset drain) and moved bytes combine
    assign dma_err         = |d_err;
    assign dma_idle        = &d_idle;
    assign dma_bytes_valid = |d_bv;
    assign dma_bytes       = (d_bv[0] ? 18'(d_bytes[0]) : 18'd0) + (d_bv[1] ? 18'(d_bytes[1]) : 18'd0);

    dma_engine #(.AW(AW), .IDW(IDW), .MAXB(MAXB), .LDF(LDF), .NCMD(NCMD)) u_dma0 (
        .clk(clk), .rst_n(rst_n && !clr), .quiesce(soft_busy),
        .cmd_valid(e_valid[0]), .cmd_ready(e_ready[0]), .cmd_word(e_word[0 +: 128]), .done(e_done[0]),
        .err(d_err[0]), .busy(dma_busy[0]), .axi_idle(d_idle[0]), .bytes_valid(d_bv[0]), .bytes(d_bytes[0]),
        .m_axi_awid(md_axi_awid), .m_axi_awaddr(md_axi_awaddr), .m_axi_awlen(md_axi_awlen),
        .m_axi_awsize(md_axi_awsize), .m_axi_awburst(md_axi_awburst), .m_axi_awlock(md_axi_awlock),
        .m_axi_awcache(md_axi_awcache), .m_axi_awprot(md_axi_awprot), .m_axi_awvalid(md_axi_awvalid),
        .m_axi_awready(md_axi_awready), .m_axi_wdata(md_axi_wdata), .m_axi_wstrb(md_axi_wstrb),
        .m_axi_wlast(md_axi_wlast), .m_axi_wvalid(md_axi_wvalid), .m_axi_wready(md_axi_wready),
        .m_axi_bid(md_axi_bid), .m_axi_bresp(md_axi_bresp), .m_axi_bvalid(md_axi_bvalid),
        .m_axi_bready(md_axi_bready),
        .m_axi_arid(md_axi_arid), .m_axi_araddr(md_axi_araddr), .m_axi_arlen(md_axi_arlen),
        .m_axi_arsize(md_axi_arsize), .m_axi_arburst(md_axi_arburst), .m_axi_arlock(md_axi_arlock),
        .m_axi_arcache(md_axi_arcache), .m_axi_arprot(md_axi_arprot), .m_axi_arvalid(md_axi_arvalid),
        .m_axi_arready(md_axi_arready), .m_axi_rid(md_axi_rid), .m_axi_rresp(md_axi_rresp),
        .m_axi_rdata(md_axi_rdata), .m_axi_rlast(md_axi_rlast), .m_axi_rvalid(md_axi_rvalid),
        .m_axi_rready(md_axi_rready),
        .sp_wvalid(sp_wvalid), .sp_wready(sp_wready), .sp_wbuf(sp_wbuf), .sp_waddr(sp_waddr),
        .sp_wdata(sp_wdata), .sp_wstrb(sp_wstrb), .sp_arvalid(sp_arvalid), .sp_arready(sp_arready),
        .sp_arbuf(sp_arbuf), .sp_araddr(sp_araddr), .sp_rvalid(sp_rvalid), .sp_rdata(sp_rdata));

    dma_engine #(.AW(AW), .IDW(IDW), .MAXB(MAXB), .LDF(LDF), .NCMD(NCMD)) u_dma1 (
        .clk(clk), .rst_n(rst_n && !clr), .quiesce(soft_busy),
        .cmd_valid(e_valid[3]), .cmd_ready(e_ready[3]), .cmd_word(e_word[0 +: 128]), .done(e_done[3]),
        .err(d_err[1]), .busy(dma_busy[1]), .axi_idle(d_idle[1]), .bytes_valid(d_bv[1]), .bytes(d_bytes[1]),
        .m_axi_awid(md1_axi_awid), .m_axi_awaddr(md1_axi_awaddr), .m_axi_awlen(md1_axi_awlen),
        .m_axi_awsize(md1_axi_awsize), .m_axi_awburst(md1_axi_awburst), .m_axi_awlock(md1_axi_awlock),
        .m_axi_awcache(md1_axi_awcache), .m_axi_awprot(md1_axi_awprot), .m_axi_awvalid(md1_axi_awvalid),
        .m_axi_awready(md1_axi_awready), .m_axi_wdata(md1_axi_wdata), .m_axi_wstrb(md1_axi_wstrb),
        .m_axi_wlast(md1_axi_wlast), .m_axi_wvalid(md1_axi_wvalid), .m_axi_wready(md1_axi_wready),
        .m_axi_bid(md1_axi_bid), .m_axi_bresp(md1_axi_bresp), .m_axi_bvalid(md1_axi_bvalid),
        .m_axi_bready(md1_axi_bready),
        .m_axi_arid(md1_axi_arid), .m_axi_araddr(md1_axi_araddr), .m_axi_arlen(md1_axi_arlen),
        .m_axi_arsize(md1_axi_arsize), .m_axi_arburst(md1_axi_arburst), .m_axi_arlock(md1_axi_arlock),
        .m_axi_arcache(md1_axi_arcache), .m_axi_arprot(md1_axi_arprot), .m_axi_arvalid(md1_axi_arvalid),
        .m_axi_arready(md1_axi_arready), .m_axi_rid(md1_axi_rid), .m_axi_rresp(md1_axi_rresp),
        .m_axi_rdata(md1_axi_rdata), .m_axi_rlast(md1_axi_rlast), .m_axi_rvalid(md1_axi_rvalid),
        .m_axi_rready(md1_axi_rready),
        .sp_wvalid(sp1_wvalid), .sp_wready(sp1_wready), .sp_wbuf(sp1_wbuf), .sp_waddr(sp1_waddr),
        .sp_wdata(sp1_wdata), .sp_wstrb(sp1_wstrb), .sp_arvalid(sp1_arvalid), .sp_arready(sp1_arready),
        .sp_arbuf(sp1_arbuf), .sp_araddr(sp1_araddr), .sp_rvalid(sp1_rvalid), .sp_rdata(sp1_rdata));
endmodule
