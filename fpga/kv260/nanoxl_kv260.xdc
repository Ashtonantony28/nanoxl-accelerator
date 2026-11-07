# NanoXL on KV260: synthesis + implementation constraints (UNTESTED until run with Vivado, docs/fpga.md).
#
# Clocks: pl_clk0 is created and constrained by the zynq_ultra_ps_e IP (250 MHz by default, -clk overrides),
# and every NanoXL / SmartConnect / intc flop runs on it. No board pins are used: all I/O goes through the PS.
#
# Scratchpad memories (rtl/buf_spad.sv -> spad_bank; nanoxl_top g_buf[b]: 0 ABUF, 1 WBUF, 2 ACC, 3 PBUF, 4 KVBUF).
# Vivado does not infer UltraRAM unless asked. ABUF (8 x 4096 x 128 b), ACC (8 x 2048 x 128 b) and KVBUF
# (8 x 8192 x 128 b) go to URAM288 (16 + 16 + 32 = 64 of 64); WBUF and PBUF stay in RAMB36.
# `?` matches the literal [ ] of generate names. If a pattern matches nothing, the memory falls back to
# Vivado's default choice: check reports/ram_utilization.rpt after the build.
set_property RAM_STYLE ultra [get_cells -quiet -hierarchical -filter {NAME =~ "*u_top/g_buf?0?.u_buf/g_bank*u_bank/mem_reg*"}]
set_property RAM_STYLE ultra [get_cells -quiet -hierarchical -filter {NAME =~ "*u_top/g_buf?2?.u_buf/g_bank*u_bank/mem_reg*"}]
set_property RAM_STYLE ultra [get_cells -quiet -hierarchical -filter {NAME =~ "*u_top/g_buf?4?.u_buf/g_bank*u_bank/mem_reg*"}]
set_property RAM_STYLE block [get_cells -quiet -hierarchical -filter {NAME =~ "*u_top/g_buf?1?.u_buf/g_bank*u_bank/mem_reg*"}]
set_property RAM_STYLE block [get_cells -quiet -hierarchical -filter {NAME =~ "*u_top/g_buf?3?.u_buf/g_bank*u_bank/mem_reg*"}]
