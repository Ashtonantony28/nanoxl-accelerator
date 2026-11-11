# NanoXL on PYNQ-Z2: synthesis + implementation constraints (UNTESTED until run with Vivado, docs/fpga.md).
#
# Clocks: FCLK_CLK0 is created and constrained by the processing_system7 IP (150 MHz requested, -clk
# overrides). No board pins are used: all I/O goes through the PS (DDR / FIXED_IO are the PS's own pins).
#
# Scratchpad banks: no constraint needed. nanoxl_top RAM_STYLE = 1 (fpga/boards.py) puts every bank in RAMB36
# through the ram_style attribute in rtl/spad_bank.sv (the 256-line PBUF banks would otherwise be LUTRAM).
