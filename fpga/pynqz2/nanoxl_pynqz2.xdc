# NanoXL on PYNQ-Z2: synthesis + implementation constraints (UNTESTED until run with Vivado, docs/fpga.md).
#
# Clocks: FCLK_CLK0 is created and constrained by the processing_system7 IP (150 MHz requested, -clk
# overrides). No board pins are used: all I/O goes through the PS (DDR / FIXED_IO are the PS's own pins).
#
# Scratchpad banks (nanoxl_top g_buf[b]: 0 ABUF, 1 WBUF, 2 ACC, 3 PBUF, 4 KVBUF): all in RAMB36. The
# 256-line PBUF banks would otherwise be candidates for LUTRAM (about 500 LUTs each).
set_property RAM_STYLE block [get_cells -quiet -hierarchical -filter {NAME =~ "*u_top/g_buf*u_buf/g_bank*u_bank/mem_reg*"}]
