# NanoXL overlay for the Kria KV260 (K26 SOM, xck26-sfvc784-2LV-c).
#
# UNTESTED until run with Vivado (SG9a wrote it without Vivado or hardware; see docs/fpga.md).
# Target tools: Vivado ML Standard 2024.2 (free; covers the K26), PYNQ for Kria (Kria-PYNQ, PYNQ 3.0.x).
#
#   cd <repo root>
#   vivado -mode batch -source fpga/kv260/build.tcl -tclargs -jobs 8
#   -> fpga/kv260/build/overlay/nanoxl.bit, nanoxl.hwh  (+ reports/timing_summary.rpt)
#
# Block design:
#   zynq_ultra_ps_e_0  M_AXI_HPM0_FPD --smc_ctrl--> nanoxl_0/s_axi_ctrl   0xA000_0000, 4 KiB (CSRs)
#                                     \----------> axi_intc_0/s_axi        0xA001_0000, 4 KiB
#   nanoxl_0/m_axi_mf (fetch) --smc_mf--> S_AXI_HP0_FPD (128 bit) -> DDR_LOW
#   nanoxl_0/m_axi_md (DMA)   --smc_md--> S_AXI_HP1_FPD (128 bit) -> DDR_LOW
#   nanoxl_0/irq -> axi_intc_0 -> pl_ps_irq0     (PYNQ's Interrupt class needs an axi_intc)
#   pl_clk0 = 250 MHz drives everything; proc_sys_reset_0 from pl_resetn0.

set script_dir [file normalize [file dirname [info script]]]
source [file join $script_dir .. common nanoxl_build.tcl]

set BOARD    kv260
set PART     xck26-sfvc784-2LV-c
set BD       nanoxl_bd
set CLK_MHZ  250
set CSR_BASE 0xA0000000
set INTC_BASE 0xA0010000

# nanoxl_top parameters: keep equal to fpga/boards.py RTL_PARAMS["kv260"] (tests/fpga checks it)
set NX_PARAMS {
    N 32 L 64 M_DEPTH 64
    ABUF_LINES 32768 WBUF_LINES 8192 ACC_LINES 16384 PBUF_LINES 4096 KV_LINES 65536 NB 8
    CTX_MAX 256 D_MAX 384 NG 6 NSLOT 16 NCTX 8 MAXB 256 LDF 512
}

set opt [nx_parse_args $argv [dict create out [file join $script_dir build]]]
if {[dict get $opt clk] ne ""} { set CLK_MHZ [dict get $opt clk] }
set NX_PARAMS [dict merge $NX_PARAMS [dict get $opt params]]
set PROJ [dict get $opt out]

nx_create_project nanoxl_$BOARD $PART $PROJ
# Board files are optional (the PS presets only matter for a boot image; PYNQ loads the overlay on a
# booted system). Use them when the Kria board store is installed.
set bp [lindex [get_board_parts -quiet -latest_file_version {*kv260_som*}] 0]
if {$bp ne ""} {
    set_property board_part $bp [current_project]
    puts "NANOXL: board part $bp"
}
nx_add_xdc [file join $script_dir nanoxl_kv260.xdc]

create_bd_design $BD

# ---------------------------------------------------------------- PS
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {$bp ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
}
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0            {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH       {32} \
    CONFIG.PSU__USE__M_AXI_GP1            {0} \
    CONFIG.PSU__USE__M_AXI_GP2            {0} \
    CONFIG.PSU__USE__S_AXI_GP2            {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH       {128} \
    CONFIG.PSU__USE__S_AXI_GP3            {1} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH       {128} \
    CONFIG.PSU__USE__IRQ0                 {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE           {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $CLK_MHZ \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL {IOPLL} \
] $ps

# ---------------------------------------------------------------- core, interconnect, reset, irq
set nx   [nx_add_core $NX_PARAMS]
set rst  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_0]
set smc  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $smc
set smf  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_mf]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smf
set smd  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_md]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smd
set intc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc axi_intc_0]
set_property -dict [list CONFIG.C_IRQ_CONNECTION {1}] $intc

set clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net $clk \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp1_fpd_aclk] \
    [get_bd_pins nanoxl_0/aclk] [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins smc_ctrl/aclk] [get_bd_pins smc_mf/aclk] [get_bd_pins smc_md/aclk] \
    [get_bd_pins axi_intc_0/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/interconnect_aresetn] \
    [get_bd_pins smc_ctrl/aresetn] [get_bd_pins smc_mf/aresetn] [get_bd_pins smc_md/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins nanoxl_0/aresetn] [get_bd_pins axi_intc_0/s_axi_aresetn]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins nanoxl_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M01_AXI] [get_bd_intf_pins axi_intc_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins nanoxl_0/m_axi_mf] [get_bd_intf_pins smc_mf/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_mf/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins nanoxl_0/m_axi_md] [get_bd_intf_pins smc_md/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_md/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]

connect_bd_net [get_bd_pins nanoxl_0/irq] [get_bd_pins axi_intc_0/intr]
connect_bd_net [get_bd_pins axi_intc_0/irq] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

# ---------------------------------------------------------------- address map (docs/fpga.md §3)
assign_bd_address -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    -offset $CSR_BASE -range 4K [get_bd_addr_segs nanoxl_0/s_axi_ctrl/*]
assign_bd_address -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    -offset $INTC_BASE -range 4K [get_bd_addr_segs axi_intc_0/S_AXI/*]
nx_map_master nanoxl_0/m_axi_mf zynq_ultra_ps_e_0/S_AXI_HP0_FPD *DDR_LOW*
nx_map_master nanoxl_0/m_axi_md zynq_ultra_ps_e_0/S_AXI_HP1_FPD *DDR_LOW*

nx_finish $PROJ $BD $opt
