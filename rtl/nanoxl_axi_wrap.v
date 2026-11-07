// nanoxl_axi_wrap.v -- Verilog-2001 shell around nanoxl_top for Vivado IP integrator (SG9a, docs/fpga.md).
//
// A block design can reference an RTL module only if its top file is Verilog or VHDL, so this file is
// plain Verilog-2001; nanoxl_top and everything below it stay SystemVerilog. The shell adds no logic:
// every port is wired straight through under the AXI names Vivado infers (prefix_SIGNAL), and
// X_INTERFACE_INFO / X_INTERFACE_PARAMETER attributes pin the interfaces explicitly:
//   aclk / aresetn (synchronous, active low = nanoxl_top rst_n), irq (level, active high)
//   s_axi_ctrl  AXI4-Lite slave, 12-bit address (4 KiB CSR window, docs/control.md §5)
//   m_axi_mf    AXI4 read-only master, 128-bit (command fetch)
//   m_axi_md    AXI4 master, 128-bit (DMA)
// tests/fpga checks the names and runs the nanoxl_top smoke test through this shell.
module nanoxl_axi_wrap #(
    parameter integer N          = 8,
    parameter integer L          = 4,
    parameter integer M_DEPTH    = 64,
    parameter integer ABUF_LINES = 4096,
    parameter integer WBUF_LINES = 2048,
    parameter integer ACC_LINES  = 4096,
    parameter integer PBUF_LINES = 2048,
    parameter integer KV_LINES   = 8192,
    parameter integer NB         = 0,
    parameter integer VP         = 4,
    parameter integer AW         = 32,
    parameter integer CTX_MAX    = 256,
    parameter integer D_MAX      = 384,
    parameter integer NG         = 4,
    parameter integer NSLOT      = 16,
    parameter integer NCTX       = 8,
    parameter integer MAXB       = 256,
    parameter integer LDF        = 512
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi_ctrl:m_axi_mf:m_axi_md, ASSOCIATED_RESET aresetn" *)
    input  wire           aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire           aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT", X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output wire           irq,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWADDR", X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12" *)
    input  wire [11:0]    s_axi_ctrl_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWPROT" *)
    input  wire [2:0]     s_axi_ctrl_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWVALID" *)
    input  wire           s_axi_ctrl_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl AWREADY" *)
    output wire           s_axi_ctrl_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl WDATA" *)
    input  wire [31:0]    s_axi_ctrl_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl WSTRB" *)
    input  wire [3:0]     s_axi_ctrl_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl WVALID" *)
    input  wire           s_axi_ctrl_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl WREADY" *)
    output wire           s_axi_ctrl_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl BRESP" *)
    output wire [1:0]     s_axi_ctrl_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl BVALID" *)
    output wire           s_axi_ctrl_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl BREADY" *)
    input  wire           s_axi_ctrl_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARADDR" *)
    input  wire [11:0]    s_axi_ctrl_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARPROT" *)
    input  wire [2:0]     s_axi_ctrl_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARVALID" *)
    input  wire           s_axi_ctrl_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl ARREADY" *)
    output wire           s_axi_ctrl_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl RDATA" *)
    output wire [31:0]    s_axi_ctrl_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl RRESP" *)
    output wire [1:0]     s_axi_ctrl_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl RVALID" *)
    output wire           s_axi_ctrl_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_ctrl RREADY" *)
    input  wire           s_axi_ctrl_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARID", X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 128, ID_WIDTH 4, READ_WRITE_MODE READ_ONLY" *)
    output wire [3:0]     m_axi_mf_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARADDR" *)
    output wire [AW-1:0]  m_axi_mf_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARLEN" *)
    output wire [7:0]     m_axi_mf_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARSIZE" *)
    output wire [2:0]     m_axi_mf_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARBURST" *)
    output wire [1:0]     m_axi_mf_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARLOCK" *)
    output wire           m_axi_mf_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARCACHE" *)
    output wire [3:0]     m_axi_mf_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARPROT" *)
    output wire [2:0]     m_axi_mf_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARVALID" *)
    output wire           m_axi_mf_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf ARREADY" *)
    input  wire           m_axi_mf_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf RID" *)
    input  wire [3:0]     m_axi_mf_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf RDATA" *)
    input  wire [127:0]   m_axi_mf_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf RRESP" *)
    input  wire [1:0]     m_axi_mf_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf RLAST" *)
    input  wire           m_axi_mf_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf RVALID" *)
    input  wire           m_axi_mf_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_mf RREADY" *)
    output wire           m_axi_mf_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWID", X_INTERFACE_PARAMETER = "PROTOCOL AXI4, DATA_WIDTH 128, ID_WIDTH 4, READ_WRITE_MODE READ_WRITE" *)
    output wire [3:0]     m_axi_md_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWADDR" *)
    output wire [AW-1:0]  m_axi_md_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWLEN" *)
    output wire [7:0]     m_axi_md_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWSIZE" *)
    output wire [2:0]     m_axi_md_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWBURST" *)
    output wire [1:0]     m_axi_md_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWLOCK" *)
    output wire           m_axi_md_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWCACHE" *)
    output wire [3:0]     m_axi_md_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWPROT" *)
    output wire [2:0]     m_axi_md_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWVALID" *)
    output wire           m_axi_md_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md AWREADY" *)
    input  wire           m_axi_md_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md WDATA" *)
    output wire [127:0]   m_axi_md_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md WSTRB" *)
    output wire [15:0]    m_axi_md_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md WLAST" *)
    output wire           m_axi_md_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md WVALID" *)
    output wire           m_axi_md_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md WREADY" *)
    input  wire           m_axi_md_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md BID" *)
    input  wire [3:0]     m_axi_md_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md BRESP" *)
    input  wire [1:0]     m_axi_md_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md BVALID" *)
    input  wire           m_axi_md_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md BREADY" *)
    output wire           m_axi_md_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARID" *)
    output wire [3:0]     m_axi_md_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARADDR" *)
    output wire [AW-1:0]  m_axi_md_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARLEN" *)
    output wire [7:0]     m_axi_md_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARSIZE" *)
    output wire [2:0]     m_axi_md_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARBURST" *)
    output wire [1:0]     m_axi_md_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARLOCK" *)
    output wire           m_axi_md_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARCACHE" *)
    output wire [3:0]     m_axi_md_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARPROT" *)
    output wire [2:0]     m_axi_md_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARVALID" *)
    output wire           m_axi_md_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md ARREADY" *)
    input  wire           m_axi_md_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md RID" *)
    input  wire [3:0]     m_axi_md_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md RDATA" *)
    input  wire [127:0]   m_axi_md_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md RRESP" *)
    input  wire [1:0]     m_axi_md_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md RLAST" *)
    input  wire           m_axi_md_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md RVALID" *)
    input  wire           m_axi_md_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 m_axi_md RREADY" *)
    output wire           m_axi_md_rready
);
    nanoxl_top #(
        .N         (N),
        .L         (L),
        .M_DEPTH   (M_DEPTH),
        .ABUF_LINES(ABUF_LINES),
        .WBUF_LINES(WBUF_LINES),
        .ACC_LINES (ACC_LINES),
        .PBUF_LINES(PBUF_LINES),
        .KV_LINES  (KV_LINES),
        .NB        (NB),
        .VP        (VP),
        .AW        (AW),
        .CTX_MAX   (CTX_MAX),
        .D_MAX     (D_MAX),
        .NG        (NG),
        .NSLOT     (NSLOT),
        .NCTX      (NCTX),
        .MAXB      (MAXB),
        .LDF       (LDF)
    ) u_top (
        .clk             (aclk),
        .rst_n           (aresetn),
        .irq             (irq),
        .s_axil_awaddr   (s_axi_ctrl_awaddr),
        .s_axil_awprot   (s_axi_ctrl_awprot),
        .s_axil_awvalid  (s_axi_ctrl_awvalid),
        .s_axil_awready  (s_axi_ctrl_awready),
        .s_axil_wdata    (s_axi_ctrl_wdata),
        .s_axil_wstrb    (s_axi_ctrl_wstrb),
        .s_axil_wvalid   (s_axi_ctrl_wvalid),
        .s_axil_wready   (s_axi_ctrl_wready),
        .s_axil_bresp    (s_axi_ctrl_bresp),
        .s_axil_bvalid   (s_axi_ctrl_bvalid),
        .s_axil_bready   (s_axi_ctrl_bready),
        .s_axil_araddr   (s_axi_ctrl_araddr),
        .s_axil_arprot   (s_axi_ctrl_arprot),
        .s_axil_arvalid  (s_axi_ctrl_arvalid),
        .s_axil_arready  (s_axi_ctrl_arready),
        .s_axil_rdata    (s_axi_ctrl_rdata),
        .s_axil_rresp    (s_axi_ctrl_rresp),
        .s_axil_rvalid   (s_axi_ctrl_rvalid),
        .s_axil_rready   (s_axi_ctrl_rready),
        .mf_axi_arid     (m_axi_mf_arid),
        .mf_axi_araddr   (m_axi_mf_araddr),
        .mf_axi_arlen    (m_axi_mf_arlen),
        .mf_axi_arsize   (m_axi_mf_arsize),
        .mf_axi_arburst  (m_axi_mf_arburst),
        .mf_axi_arlock   (m_axi_mf_arlock),
        .mf_axi_arcache  (m_axi_mf_arcache),
        .mf_axi_arprot   (m_axi_mf_arprot),
        .mf_axi_arvalid  (m_axi_mf_arvalid),
        .mf_axi_arready  (m_axi_mf_arready),
        .mf_axi_rid      (m_axi_mf_rid),
        .mf_axi_rdata    (m_axi_mf_rdata),
        .mf_axi_rresp    (m_axi_mf_rresp),
        .mf_axi_rlast    (m_axi_mf_rlast),
        .mf_axi_rvalid   (m_axi_mf_rvalid),
        .mf_axi_rready   (m_axi_mf_rready),
        .md_axi_awid     (m_axi_md_awid),
        .md_axi_awaddr   (m_axi_md_awaddr),
        .md_axi_awlen    (m_axi_md_awlen),
        .md_axi_awsize   (m_axi_md_awsize),
        .md_axi_awburst  (m_axi_md_awburst),
        .md_axi_awlock   (m_axi_md_awlock),
        .md_axi_awcache  (m_axi_md_awcache),
        .md_axi_awprot   (m_axi_md_awprot),
        .md_axi_awvalid  (m_axi_md_awvalid),
        .md_axi_awready  (m_axi_md_awready),
        .md_axi_wdata    (m_axi_md_wdata),
        .md_axi_wstrb    (m_axi_md_wstrb),
        .md_axi_wlast    (m_axi_md_wlast),
        .md_axi_wvalid   (m_axi_md_wvalid),
        .md_axi_wready   (m_axi_md_wready),
        .md_axi_bid      (m_axi_md_bid),
        .md_axi_bresp    (m_axi_md_bresp),
        .md_axi_bvalid   (m_axi_md_bvalid),
        .md_axi_bready   (m_axi_md_bready),
        .md_axi_arid     (m_axi_md_arid),
        .md_axi_araddr   (m_axi_md_araddr),
        .md_axi_arlen    (m_axi_md_arlen),
        .md_axi_arsize   (m_axi_md_arsize),
        .md_axi_arburst  (m_axi_md_arburst),
        .md_axi_arlock   (m_axi_md_arlock),
        .md_axi_arcache  (m_axi_md_arcache),
        .md_axi_arprot   (m_axi_md_arprot),
        .md_axi_arvalid  (m_axi_md_arvalid),
        .md_axi_arready  (m_axi_md_arready),
        .md_axi_rid      (m_axi_md_rid),
        .md_axi_rdata    (m_axi_md_rdata),
        .md_axi_rresp    (m_axi_md_rresp),
        .md_axi_rlast    (m_axi_md_rlast),
        .md_axi_rvalid   (m_axi_md_rvalid),
        .md_axi_rready   (m_axi_md_rready)
    );
endmodule
