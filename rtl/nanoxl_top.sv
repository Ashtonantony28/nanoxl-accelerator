// nanoxl_top.sv -- NanoXL accelerator top level (SG7c, docs/top.md).
//
//   host ── AXI4-Lite (s_axil) ──► ctrl_top: CSRs, command fetcher (mf_axi), command processor
//                                   (3 queues, semaphores, barriers), DMA engine (md_axi)
//                                      │ arr_cmd            │ vec_cmd            │ DMA line port
//                                      ▼                    ▼                    ▼
//                                 array_seq            vec_seq (vec_unit+SFU)    │
//                               (array_ctrl +          ra / rb / rp / wd         │
//                                systolic_array)                                 │
//                                 A / W / CR / CW                                │
//                                      └──────────────┬─────────┴────────────────┘
//                                          spad router (buffer id) + 5 x buf_spad
//                                          ABUF  WBUF  ACC  PBUF  KVBUF  (NB banks each)
//
// External ports: AXI4-Lite CSR slave, AXI4 read master `mf_axi` (command fetch), AXI4 masters
// `md_axi` (DMA channel 0) and `md1_axi` (DMA channel 1, SG10a / ADR-005), `irq`. Buffer capacities are parameters in 16-byte lines (powers of two);
// line addresses wrap at the capacity. Buffer ids 5..7 read zeros and drop writes (the
// decoders reject them before they get here).
//
// Wiring contract (docs/top.md §2): DMA writes into PBUF are snooped into the VEC engine's
// parameter residency (pb_snoop_*); ctrl_top eng_clr resets both engines; sat_clr clears the
// VEC engine's counters; SAT_0..4 = {requant, ALU, LN out, LN z, softmax score} (= vec_seq order).
module nanoxl_top #(
    parameter int N          = 8,          // systolic array N x N
    parameter int L          = 4,          // vector lanes
    parameter int M_DEPTH    = 64,         // array_ctrl accumulator rows
    parameter int ABUF_LINES = 4096,       // 64 KiB
    parameter int WBUF_LINES = 2048,       // 32 KiB
    parameter int ACC_LINES  = 4096,       // 64 KiB
    parameter int PBUF_LINES = 2048,       // 32 KiB
    parameter int KV_LINES   = 8192,       // 128 KiB
    parameter int NB         = 0,          // banks per buffer; 0 = 2 x widest port (power of two)
    parameter int VP         = 4,          // VEC engine port beat in 16-byte lines (A, B, P, D); 1 or 4
    parameter int AW         = 32,
    parameter int CTX_MAX    = 256,
    parameter int D_MAX      = 384,
    parameter int NG         = 4,          // GELU table slots (shakespeare: 6)
    parameter int NSLOT      = 16,         // LN parameter slots
    parameter int NCTX       = 8,          // softmax ctx-record slots
    parameter int MAXB       = 256,
    parameter int LDF        = 512
) (
    input  logic               clk,
    input  logic               rst_n,
    output logic               irq,
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
    output logic [3:0]         mf_axi_arid,
    output logic [AW-1:0]      mf_axi_araddr,
    output logic [7:0]         mf_axi_arlen,
    output logic [2:0]         mf_axi_arsize,
    output logic [1:0]         mf_axi_arburst,
    output logic               mf_axi_arlock,
    output logic [3:0]         mf_axi_arcache,
    output logic [2:0]         mf_axi_arprot,
    output logic               mf_axi_arvalid,
    input  logic               mf_axi_arready,
    input  logic [3:0]         mf_axi_rid,
    input  logic [127:0]       mf_axi_rdata,
    input  logic [1:0]         mf_axi_rresp,
    input  logic               mf_axi_rlast,
    input  logic               mf_axi_rvalid,
    output logic               mf_axi_rready,
    // ---- DMA channel 0 AXI4 master ----
    output logic [3:0]         md_axi_awid,
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
    input  logic [3:0]         md_axi_bid,
    input  logic [1:0]         md_axi_bresp,
    input  logic               md_axi_bvalid,
    output logic               md_axi_bready,
    output logic [3:0]         md_axi_arid,
    output logic [AW-1:0]      md_axi_araddr,
    output logic [7:0]         md_axi_arlen,
    output logic [2:0]         md_axi_arsize,
    output logic [1:0]         md_axi_arburst,
    output logic               md_axi_arlock,
    output logic [3:0]         md_axi_arcache,
    output logic [2:0]         md_axi_arprot,
    output logic               md_axi_arvalid,
    input  logic               md_axi_arready,
    input  logic [3:0]         md_axi_rid,
    input  logic [127:0]       md_axi_rdata,
    input  logic [1:0]         md_axi_rresp,
    input  logic               md_axi_rlast,
    input  logic               md_axi_rvalid,
    output logic               md_axi_rready,
    // ---- DMA channel 1 AXI4 master (SG10a, ADR-005) ----
    output logic [3:0]         md1_axi_awid,
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
    input  logic [3:0]         md1_axi_bid,
    input  logic [1:0]         md1_axi_bresp,
    input  logic               md1_axi_bvalid,
    output logic               md1_axi_bready,
    output logic [3:0]         md1_axi_arid,
    output logic [AW-1:0]      md1_axi_araddr,
    output logic [7:0]         md1_axi_arlen,
    output logic [2:0]         md1_axi_arsize,
    output logic [1:0]         md1_axi_arburst,
    output logic               md1_axi_arlock,
    output logic [3:0]         md1_axi_arcache,
    output logic [2:0]         md1_axi_arprot,
    output logic               md1_axi_arvalid,
    input  logic               md1_axi_arready,
    input  logic [3:0]         md1_axi_rid,
    input  logic [127:0]       md1_axi_rdata,
    input  logic [1:0]         md1_axi_rresp,
    input  logic               md1_axi_rlast,
    input  logic               md1_axi_rvalid,
    output logic               md1_axi_rready
);
    localparam int APL  = (N + 15) / 16;              // array A port lines per beat
    localparam int WPL  = (N + 15) / 16;              // array W port lines per beat
    localparam int CPL  = (N >= 4) ? N / 4 : 1;       // array ACC port lines per beat (one int32 row)
    localparam int PM0  = (APL > CPL) ? APL : CPL;
    localparam int PM1  = (PM0 > WPL) ? PM0 : WPL;
    localparam int PMAX = (PM1 > VP) ? PM1 : VP;
    localparam int NBA  = (NB > 0) ? NB : ((PMAX <= 1) ? 2 : (PMAX <= 2) ? 4 : (PMAX <= 4) ? 8 : 16);
    localparam int NSAT = 5;

    // read ports: 0 DMA ch0, 1 array A, 2 array W, 3 array ACC, 4 VEC A, 5 VEC B, 6 VEC P, 7 DMA ch1
    // write ports: 0 DMA ch0, 1 array ACC, 2 VEC D, 3 DMA ch1
    localparam int NR = 8, NW = 4;
    localparam logic [NR*8-1:0] RPL = {8'd1, 8'(VP), 8'(VP), 8'(VP), 8'(CPL), 8'(WPL), 8'(APL), 8'd1};
    localparam logic [NW*8-1:0] WPLV = {8'd1, 8'(VP), 8'(CPL), 8'd1};

    // ---------------------------------------------------------------- control plane
    logic               eng_clr, sat_clr;
    logic [NSAT*32-1:0] sat_cnt;
    logic               sp_wvalid, sp_wready, sp_arvalid, sp_arready, sp_rvalid;
    logic [2:0]         sp_wbuf, sp_arbuf;
    logic [16:0]        sp_waddr, sp_araddr;
    logic [127:0]       sp_wdata, sp_rdata;
    logic [15:0]        sp_wstrb;
    logic               sp1_wvalid, sp1_wready, sp1_arvalid, sp1_arready, sp1_rvalid;   // DMA channel 1
    logic [2:0]         sp1_wbuf, sp1_arbuf;
    logic [16:0]        sp1_waddr, sp1_araddr;
    logic [127:0]       sp1_wdata, sp1_rdata;
    logic [15:0]        sp1_wstrb;
    logic               sp1_wv_raw;
    logic               arr_cmd_valid, arr_cmd_ready, arr_done, arr_err;
    logic               vec_cmd_valid, vec_cmd_ready, vec_done, vec_err;
    logic [127:0]       arr_cmd_word, vec_cmd_word;

    ctrl_top #(.AW(AW), .IDW(4), .MAXB(MAXB), .LDF(LDF), .NSAT(NSAT)) u_ctrl (
        .clk(clk), .rst_n(rst_n), .irq(irq), .eng_clr(eng_clr), .sat_clr(sat_clr), .sat_cnt(sat_cnt),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awprot(s_axil_awprot), .s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready), .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
        .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready), .s_axil_bresp(s_axil_bresp),
        .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready), .s_axil_araddr(s_axil_araddr),
        .s_axil_arprot(s_axil_arprot), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .mf_axi_arid(mf_axi_arid), .mf_axi_araddr(mf_axi_araddr), .mf_axi_arlen(mf_axi_arlen),
        .mf_axi_arsize(mf_axi_arsize), .mf_axi_arburst(mf_axi_arburst), .mf_axi_arlock(mf_axi_arlock),
        .mf_axi_arcache(mf_axi_arcache), .mf_axi_arprot(mf_axi_arprot), .mf_axi_arvalid(mf_axi_arvalid),
        .mf_axi_arready(mf_axi_arready), .mf_axi_rid(mf_axi_rid), .mf_axi_rdata(mf_axi_rdata),
        .mf_axi_rresp(mf_axi_rresp), .mf_axi_rlast(mf_axi_rlast), .mf_axi_rvalid(mf_axi_rvalid),
        .mf_axi_rready(mf_axi_rready),
        .md_axi_awid(md_axi_awid), .md_axi_awaddr(md_axi_awaddr), .md_axi_awlen(md_axi_awlen),
        .md_axi_awsize(md_axi_awsize), .md_axi_awburst(md_axi_awburst), .md_axi_awlock(md_axi_awlock),
        .md_axi_awcache(md_axi_awcache), .md_axi_awprot(md_axi_awprot), .md_axi_awvalid(md_axi_awvalid),
        .md_axi_awready(md_axi_awready), .md_axi_wdata(md_axi_wdata), .md_axi_wstrb(md_axi_wstrb),
        .md_axi_wlast(md_axi_wlast), .md_axi_wvalid(md_axi_wvalid), .md_axi_wready(md_axi_wready),
        .md_axi_bid(md_axi_bid), .md_axi_bresp(md_axi_bresp), .md_axi_bvalid(md_axi_bvalid),
        .md_axi_bready(md_axi_bready),
        .md_axi_arid(md_axi_arid), .md_axi_araddr(md_axi_araddr), .md_axi_arlen(md_axi_arlen),
        .md_axi_arsize(md_axi_arsize), .md_axi_arburst(md_axi_arburst), .md_axi_arlock(md_axi_arlock),
        .md_axi_arcache(md_axi_arcache), .md_axi_arprot(md_axi_arprot), .md_axi_arvalid(md_axi_arvalid),
        .md_axi_arready(md_axi_arready), .md_axi_rid(md_axi_rid), .md_axi_rdata(md_axi_rdata),
        .md_axi_rresp(md_axi_rresp), .md_axi_rlast(md_axi_rlast), .md_axi_rvalid(md_axi_rvalid),
        .md_axi_rready(md_axi_rready),
        .md1_axi_awid(md1_axi_awid), .md1_axi_awaddr(md1_axi_awaddr), .md1_axi_awlen(md1_axi_awlen),
        .md1_axi_awsize(md1_axi_awsize), .md1_axi_awburst(md1_axi_awburst), .md1_axi_awlock(md1_axi_awlock),
        .md1_axi_awcache(md1_axi_awcache), .md1_axi_awprot(md1_axi_awprot), .md1_axi_awvalid(md1_axi_awvalid),
        .md1_axi_awready(md1_axi_awready), .md1_axi_wdata(md1_axi_wdata), .md1_axi_wstrb(md1_axi_wstrb),
        .md1_axi_wlast(md1_axi_wlast), .md1_axi_wvalid(md1_axi_wvalid), .md1_axi_wready(md1_axi_wready),
        .md1_axi_bid(md1_axi_bid), .md1_axi_bresp(md1_axi_bresp), .md1_axi_bvalid(md1_axi_bvalid),
        .md1_axi_bready(md1_axi_bready),
        .md1_axi_arid(md1_axi_arid), .md1_axi_araddr(md1_axi_araddr), .md1_axi_arlen(md1_axi_arlen),
        .md1_axi_arsize(md1_axi_arsize), .md1_axi_arburst(md1_axi_arburst), .md1_axi_arlock(md1_axi_arlock),
        .md1_axi_arcache(md1_axi_arcache), .md1_axi_arprot(md1_axi_arprot), .md1_axi_arvalid(md1_axi_arvalid),
        .md1_axi_arready(md1_axi_arready), .md1_axi_rid(md1_axi_rid), .md1_axi_rdata(md1_axi_rdata),
        .md1_axi_rresp(md1_axi_rresp), .md1_axi_rlast(md1_axi_rlast), .md1_axi_rvalid(md1_axi_rvalid),
        .md1_axi_rready(md1_axi_rready),
        .sp_wvalid(sp_wvalid), .sp_wready(sp_wready), .sp_wbuf(sp_wbuf), .sp_waddr(sp_waddr),
        .sp_wdata(sp_wdata), .sp_wstrb(sp_wstrb), .sp_arvalid(sp_arvalid), .sp_arready(sp_arready),
        .sp_arbuf(sp_arbuf), .sp_araddr(sp_araddr), .sp_rvalid(sp_rvalid), .sp_rdata(sp_rdata),
        .sp1_wvalid(sp1_wv_raw), .sp1_wready(sp1_wready), .sp1_wbuf(sp1_wbuf), .sp1_waddr(sp1_waddr),
        .sp1_wdata(sp1_wdata), .sp1_wstrb(sp1_wstrb), .sp1_arvalid(sp1_arvalid), .sp1_arready(sp1_arready),
        .sp1_arbuf(sp1_arbuf), .sp1_araddr(sp1_araddr), .sp1_rvalid(sp1_rvalid), .sp1_rdata(sp1_rdata),
        .arr_cmd_valid(arr_cmd_valid), .arr_cmd_ready(arr_cmd_ready), .arr_cmd_word(arr_cmd_word),
        .arr_done(arr_done), .arr_err(arr_err),
        .vec_cmd_valid(vec_cmd_valid), .vec_cmd_ready(vec_cmd_ready), .vec_cmd_word(vec_cmd_word),
        .vec_done(vec_done), .vec_err(vec_err));

    // ---------------------------------------------------------------- ARRAY engine
    logic               a_arvalid, a_arready, w_arvalid, w_arready, cr_arvalid, cr_arready;
    logic               a_rvalid, w_rvalid, cr_rvalid, cw_valid, cw_ready;
    logic [2:0]         a_arbuf, w_arbuf, cr_arbuf, cw_buf;
    logic [16:0]        a_araddr, w_araddr, cr_araddr, cw_addr;
    logic [APL*128-1:0] a_rdata;
    logic [WPL*128-1:0] w_rdata;
    logic [N*32-1:0]    cr_rdata, cw_data;
    logic [N*4-1:0]     cw_strb;
    /* verilator lint_off UNUSEDSIGNAL */
    logic               arr_idle, arr_hazard, arr_ovf;
    /* verilator lint_on UNUSEDSIGNAL */

    array_seq #(.N(N), .M_DEPTH(M_DEPTH)) u_arr (
        .clk(clk), .rst_n(rst_n), .clr(eng_clr),
        .cmd_valid(arr_cmd_valid), .cmd_ready(arr_cmd_ready), .cmd_word(arr_cmd_word),
        .done(arr_done), .err(arr_err), .idle(arr_idle),
        .a_arvalid(a_arvalid), .a_arready(a_arready), .a_arbuf(a_arbuf), .a_araddr(a_araddr),
        .a_rvalid(a_rvalid), .a_rdata(a_rdata),
        .w_arvalid(w_arvalid), .w_arready(w_arready), .w_arbuf(w_arbuf), .w_araddr(w_araddr),
        .w_rvalid(w_rvalid), .w_rdata(w_rdata),
        .cr_arvalid(cr_arvalid), .cr_arready(cr_arready), .cr_arbuf(cr_arbuf), .cr_araddr(cr_araddr),
        .cr_rvalid(cr_rvalid), .cr_rdata(cr_rdata),
        .cw_valid(cw_valid), .cw_ready(cw_ready), .cw_buf(cw_buf), .cw_addr(cw_addr),
        .cw_data(cw_data), .cw_strb(cw_strb),
        .hazard_seen(arr_hazard), .overflow_seen(arr_ovf));

    // ---------------------------------------------------------------- VEC engine
    logic               ra_arvalid, ra_arready, rb_arvalid, rb_arready, rp_arvalid, rp_arready;
    logic               ra_rvalid, rb_rvalid, rp_rvalid, wd_valid, wd_ready;
    logic [2:0]         ra_arbuf, rb_arbuf, wd_buf;
    logic [16:0]        ra_araddr, rb_araddr, rp_araddr, wd_addr;
    logic [VP*128-1:0]  ra_rdata, rb_rdata, rp_rdata, wd_data;
    logic [VP*16-1:0]   wd_strb;
    logic               pb_snoop_valid;
    logic [5*32-1:0]    vec_sat;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [3:0]         vec_err_code;
    logic               vec_busy;
    logic [31:0]        st_hits, st_misses, st_lines;
    /* verilator lint_on UNUSEDSIGNAL */

    // every DMA write into PBUF invalidates resident parameter blocks that cover the line. The VEC
    // engine has one snoop port, so DMA channel 1 writes PBUF only in cycles channel 0 does not.
    logic dma0_pb;
    assign dma0_pb        = sp_wvalid && (sp_wbuf == 3'd3);
    assign sp1_wvalid     = sp1_wv_raw && !(dma0_pb && sp1_wbuf == 3'd3);
    assign pb_snoop_valid = (dma0_pb && sp_wready) || (sp1_wvalid && sp1_wready && sp1_wbuf == 3'd3);

    vec_seq #(.L(L), .AW(17), .CTX_MAX(CTX_MAX), .D_MAX(D_MAX), .NG(NG), .NSLOT(NSLOT),
              .NCTX(NCTX), .PL(VP)) u_vec (
        .clk(clk), .rst_n(rst_n), .clr(eng_clr),
        .cmd_valid(vec_cmd_valid), .cmd_ready(vec_cmd_ready), .cmd_word(vec_cmd_word),
        .done(vec_done), .err(vec_err), .err_code(vec_err_code), .busy(vec_busy),
        .ra_arvalid(ra_arvalid), .ra_arready(ra_arready), .ra_arbuf(ra_arbuf), .ra_araddr(ra_araddr),
        .ra_rvalid(ra_rvalid), .ra_rdata(ra_rdata),
        .rb_arvalid(rb_arvalid), .rb_arready(rb_arready), .rb_arbuf(rb_arbuf), .rb_araddr(rb_araddr),
        .rb_rvalid(rb_rvalid), .rb_rdata(rb_rdata),
        .rp_arvalid(rp_arvalid), .rp_arready(rp_arready), .rp_araddr(rp_araddr),
        .rp_rvalid(rp_rvalid), .rp_rdata(rp_rdata),
        .wd_valid(wd_valid), .wd_ready(wd_ready), .wd_buf(wd_buf), .wd_addr(wd_addr),
        .wd_data(wd_data), .wd_strb(wd_strb),
        .pb_snoop_valid(pb_snoop_valid), .pb_snoop_line(dma0_pb ? sp_waddr : sp1_waddr),
        .sat_clr(sat_clr), .sat_cnt(vec_sat),
        .st_hits(st_hits), .st_misses(st_misses), .st_lines(st_lines));

    // saturation counters -> CSR SAT_0..4 in the canonical order of vec_seq.sat_cnt (AUDIT-1 A1-02):
    // 0 requant, 1 ALU, 2 LN out, 3 LN z, 4 softmax score (docs/control.md §5, nanoxl/driver/csr.py)
    assign sat_cnt = vec_sat;

    // ---------------------------------------------------------------- scratchpad router
    logic [NR-1:0]          rq_v, rq_r, rs_v;
    logic [NR*3-1:0]        rq_b;
    logic [NR*17-1:0]       rq_a;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [NR*PMAX*128-1:0] rs_d;                 // narrow ports use the low lines only
    /* verilator lint_on UNUSEDSIGNAL */
    logic [NW-1:0]          wq_v, wq_r;
    logic [NW*3-1:0]        wq_b;
    logic [NW*17-1:0]       wq_a;
    logic [NW*PMAX*128-1:0] wq_d;
    logic [NW*PMAX*16-1:0]  wq_s;

    assign rq_v = {sp1_arvalid, rp_arvalid, rb_arvalid, ra_arvalid, cr_arvalid, w_arvalid, a_arvalid, sp_arvalid};
    assign rq_b = {sp1_arbuf, 3'd3, rb_arbuf, ra_arbuf, cr_arbuf, w_arbuf, a_arbuf, sp_arbuf};
    assign rq_a = {sp1_araddr, rp_araddr, rb_araddr, ra_araddr, cr_araddr, w_araddr, a_araddr, sp_araddr};
    assign {sp1_arready, rp_arready, rb_arready, ra_arready, cr_arready, w_arready, a_arready, sp_arready} = rq_r;
    assign {sp1_rvalid, rp_rvalid, rb_rvalid, ra_rvalid, cr_rvalid, w_rvalid, a_rvalid, sp_rvalid} = rs_v;
    assign sp_rdata = rs_d[0*PMAX*128 +: 128];
    assign a_rdata  = rs_d[1*PMAX*128 +: APL*128];
    assign w_rdata  = rs_d[2*PMAX*128 +: WPL*128];
    assign cr_rdata = rs_d[3*PMAX*128 +: N*32];
    assign ra_rdata = rs_d[4*PMAX*128 +: VP*128];
    assign rb_rdata = rs_d[5*PMAX*128 +: VP*128];
    assign rp_rdata = rs_d[6*PMAX*128 +: VP*128];
    assign sp1_rdata = rs_d[7*PMAX*128 +: 128];

    assign wq_v = {sp1_wvalid, wd_valid, cw_valid, sp_wvalid};
    assign wq_b = {sp1_wbuf, wd_buf, cw_buf, sp_wbuf};
    assign wq_a = {sp1_waddr, wd_addr, cw_addr, sp_waddr};
    assign {sp1_wready, wd_ready, cw_ready, sp_wready} = wq_r;
    always_comb begin
        wq_d = '0;
        wq_s = '0;
        wq_d[0*PMAX*128 +: 128]   = sp_wdata;
        wq_s[0*PMAX*16 +: 16]     = sp_wstrb;
        wq_d[1*PMAX*128 +: N*32]  = cw_data;
        wq_s[1*PMAX*16 +: N*4]    = cw_strb;
        wq_d[2*PMAX*128 +: VP*128] = wd_data;
        wq_s[2*PMAX*16 +: VP*16]   = wd_strb;
        wq_d[3*PMAX*128 +: 128]   = sp1_wdata;
        wq_s[3*PMAX*16 +: 16]     = sp1_wstrb;
    end

    // five buffers; ids 5..7 = null buffer (reads return zeros one cycle later, writes dropped)
    logic [NR-1:0]          b_rv [5];
    logic [NR-1:0]          b_rr [5];
    logic [NR-1:0]          b_dv [5];
    logic [NR*PMAX*128-1:0] b_dd [5];
    logic [NW-1:0]          b_wv [5];
    logic [NW-1:0]          b_wr [5];
    logic [NR-1:0]          nul_r, nul_dv;

    for (genvar b = 0; b < 5; b++) begin : g_sel
        for (genvar p = 0; p < NR; p++) begin : g_r
            assign b_rv[b][p] = rq_v[p] && (rq_b[p*3 +: 3] == 3'(b));
        end
        for (genvar p = 0; p < NW; p++) begin : g_w
            assign b_wv[b][p] = wq_v[p] && (wq_b[p*3 +: 3] == 3'(b));
        end
    end
    for (genvar p = 0; p < NR; p++) begin : g_nul
        assign nul_r[p] = rq_b[p*3 +: 3] > 3'd4;
    end

    always_comb begin
        rq_r = nul_r;
        rs_v = nul_dv;
        rs_d = '0;
        wq_r = '0;
        for (int p = 0; p < NW; p++) wq_r[p] = wq_b[p*3 +: 3] > 3'd4;
        for (int b = 0; b < 5; b++) begin
            rq_r = rq_r | (b_rr[b] & b_rv[b]);
            rs_v = rs_v | b_dv[b];
            wq_r = wq_r | (b_wr[b] & b_wv[b]);
            for (int p = 0; p < NR; p++)
                if (b_dv[b][p]) rs_d[p*PMAX*128 +: PMAX*128] = b_dd[b][p*PMAX*128 +: PMAX*128];
        end
    end
    always_ff @(posedge clk) begin
        if (!rst_n) nul_dv <= '0;
        else        nul_dv <= rq_v & nul_r;
    end

    function automatic int blines(input int b);
        case (b)
            0:       blines = ABUF_LINES;
            1:       blines = WBUF_LINES;
            2:       blines = ACC_LINES;
            3:       blines = PBUF_LINES;
            default: blines = KV_LINES;
        endcase
    endfunction
    for (genvar b = 0; b < 5; b++) begin : g_buf
        buf_spad #(.LINES(blines(b)), .NB(NBA), .NR(NR), .NW(NW), .PMAX(PMAX), .RPL(RPL), .WPL(WPLV)) u_buf (
            .clk(clk), .rst_n(rst_n),
            .r_valid(b_rv[b]), .r_ready(b_rr[b]), .r_addr(rq_a), .d_valid(b_dv[b]), .d_data(b_dd[b]),
            .w_valid(b_wv[b]), .w_ready(b_wr[b]), .w_addr(wq_a), .w_data(wq_d), .w_strb(wq_s));
    end
endmodule
