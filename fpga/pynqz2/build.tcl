# NanoXL overlay for the TUL PYNQ-Z2 (Zynq-7020, xc7z020clg400-1).
#
# UNTESTED until run with Vivado (SG9a wrote it without Vivado or hardware; see docs/fpga.md).
# Target tools: Vivado ML Standard 2024.2 (free; covers the 7z020), PYNQ 3.0.1 SD image for PYNQ-Z2.
#
#   cd <repo root>
#   vivado -mode batch -source fpga/pynqz2/build.tcl -tclargs -jobs 8
#   -> fpga/pynqz2/build/overlay/nanoxl.bit, nanoxl.hwh  (+ reports/timing_summary.rpt)
#
# Block design:
#   processing_system7_0 M_AXI_GP0 --smc_ctrl--> nanoxl_0/s_axi_ctrl   0x4000_0000, 4 KiB (CSRs)
#                                  \----------> axi_intc_0/s_axi        0x4180_0000, 4 KiB
#   nanoxl_0/m_axi_mf  (fetch, 128 b) --+
#   nanoxl_0/m_axi_md  (DMA ch 0)     --smc_hp0 (2:1)--> S_AXI_HP0 (64 bit) -> DDR   (ADR-005 §4;
#   nanoxl_0/m_axi_md1 (DMA ch 1)     --smc_md1------> S_AXI_HP1 (64 bit) -> DDR    SmartConnect downsizes)
#   nanoxl_0/irq -> axi_intc_0 -> IRQ_F2P[0]
#   FCLK_CLK0 = 150 MHz requested (the PS PLLs give the nearest value; PYNQ reprograms it from the .hwh).

set script_dir [file normalize [file dirname [info script]]]
source [file join $script_dir .. common nanoxl_build.tcl]

set BOARD    pynq-z2
set PART     xc7z020clg400-1
set BD       nanoxl_bd
set CLK_MHZ  150
set CSR_BASE 0x40000000
set INTC_BASE 0x41800000

# nanoxl_top parameters: keep equal to fpga/boards.py RTL_PARAMS["pynq-z2"] (tests/fpga checks it)
set NX_PARAMS {
    N 16 L 16 M_DEPTH 64
    ABUF_LINES 8192 WBUF_LINES 4096 ACC_LINES 4096 PBUF_LINES 2048 KV_LINES 8192 NB 0 VP 4
    CTX_MAX 256 D_MAX 384 NG 6 NSLOT 16 NCTX 8 MAXB 256 LDF 512
}

set opt [nx_parse_args $argv [dict create out [file join $script_dir build]]]
if {[dict get $opt clk] ne ""} { set CLK_MHZ [dict get $opt clk] }
set NX_PARAMS [dict merge $NX_PARAMS [dict get $opt params]]
set PROJ [dict get $opt out]

nx_create_project nanoxl_pynqz2 $PART $PROJ
set bp [lindex [get_board_parts -quiet -latest_file_version {*pynq-z2*}] 0]
if {$bp ne ""} {
    set_property board_part $bp [current_project]
    puts "NANOXL: board part $bp"
}
nx_add_xdc [file join $script_dir nanoxl_pynqz2.xdc]

create_bd_design $BD

# ---------------------------------------------------------------- PS
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config [list make_external {FIXED_IO, DDR} apply_board_preset [expr {$bp ne "" ? 1 : 0}] \
                  Master Disable Slave Disable] $ps
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0            {1} \
    CONFIG.PCW_USE_S_AXI_HP0            {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH     {64} \
    CONFIG.PCW_USE_S_AXI_HP1            {1} \
    CONFIG.PCW_S_AXI_HP1_DATA_WIDTH     {64} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT     {1} \
    CONFIG.PCW_IRQ_F2P_INTR             {1} \
    CONFIG.PCW_EN_CLK0_PORT             {1} \
    CONFIG.PCW_EN_RST0_PORT             {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $CLK_MHZ \
] $ps

# ---------------------------------------------------------------- core, interconnect, reset, irq
set nx   [nx_add_core $NX_PARAMS]
set rst  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_0]
set smc  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $smc
set shp0 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_hp0]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $shp0
set smd1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_md1]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smd1
set intc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc axi_intc_0]
set_property -dict [list CONFIG.C_IRQ_CONNECTION {1}] $intc

set clk [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net $clk \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP1_ACLK] \
    [get_bd_pins nanoxl_0/aclk] [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins smc_ctrl/aclk] [get_bd_pins smc_hp0/aclk] [get_bd_pins smc_md1/aclk] \
    [get_bd_pins axi_intc_0/s_axi_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/interconnect_aresetn] \
    [get_bd_pins smc_ctrl/aresetn] [get_bd_pins smc_hp0/aresetn] [get_bd_pins smc_md1/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins nanoxl_0/aresetn] [get_bd_pins axi_intc_0/s_axi_aresetn]

connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins nanoxl_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M01_AXI] [get_bd_intf_pins axi_intc_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins nanoxl_0/m_axi_mf] [get_bd_intf_pins smc_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins nanoxl_0/m_axi_md] [get_bd_intf_pins smc_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_hp0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins nanoxl_0/m_axi_md1] [get_bd_intf_pins smc_md1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_md1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP1]

connect_bd_net [get_bd_pins nanoxl_0/irq] [get_bd_pins axi_intc_0/intr]
connect_bd_net [get_bd_pins axi_intc_0/irq] [get_bd_pins processing_system7_0/IRQ_F2P]

# ---------------------------------------------------------------- address map (docs/fpga.md §3)
assign_bd_address -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    -offset $CSR_BASE -range 4K [get_bd_addr_segs nanoxl_0/s_axi_ctrl/*]
assign_bd_address -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    -offset $INTC_BASE -range 4K [get_bd_addr_segs axi_intc_0/S_AXI/*]
nx_map_master nanoxl_0/m_axi_mf  processing_system7_0/S_AXI_HP0 *DDR*
nx_map_master nanoxl_0/m_axi_md  processing_system7_0/S_AXI_HP0 *DDR*
nx_map_master nanoxl_0/m_axi_md1 processing_system7_0/S_AXI_HP1 *DDR*

nx_finish $PROJ $BD $opt
