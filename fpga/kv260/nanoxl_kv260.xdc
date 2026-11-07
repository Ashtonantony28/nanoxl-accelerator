# NanoXL on KV260: synthesis + implementation constraints (UNTESTED until run with Vivado, docs/fpga.md).
#
# Clocks: pl_clk0 is created and constrained by the zynq_ultra_ps_e IP (250 MHz by default, -clk overrides),
# and every NanoXL / SmartConnect / intc flop runs on it. No board pins are used: all I/O goes through the PS.
#
# Scratchpad memories: no constraint needed. nanoxl_top RAM_STYLE = 2 (fpga/boards.py) sets the ram_style
# attribute in rtl/spad_bank.sv: ABUF (8 x 4096 x 128 b), ACC (8 x 2048) and KVBUF (8 x 8192) banks go to
# URAM288 (16 + 16 + 32 = 64 of 64), WBUF and PBUF to RAMB36. Check reports/ram_utilization.rpt after the build.
