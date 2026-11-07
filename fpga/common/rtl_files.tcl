# RTL source set of the NanoXL overlay, relative to the repository root.
# Must equal the files nanoxl_axi_wrap / nanoxl_top elaborate from (tests/fpga/test_fpga_static.py checks
# this against a Verilator dependency run, so a new RTL file shows up as a failing test).
# Order: packages / leaf modules first (Vivado does not care, but it keeps reading logs readable).
set NANOXL_RTL {
    rtl/fifo_sync.sv
    rtl/cmd_fields.sv
    rtl/cmd_decode.sv
    rtl/cmd_ring_ptr.sv
    rtl/cmd_fetch.sv
    rtl/sem_unit.sv
    rtl/cmd_queue.sv
    rtl/cmd_proc.sv
    rtl/dma_realign.sv
    rtl/dma_engine.sv
    rtl/csr_axil.sv
    rtl/ctrl_top.sv
    rtl/pe.sv
    rtl/skew.sv
    rtl/systolic_array.sv
    rtl/scratchpad.sv
    rtl/array_ctrl.sv
    rtl/array_seq_rd.sv
    rtl/array_seq.sv
    rtl/lut_rom.sv
    rtl/sfu_exp2.sv
    rtl/sfu_recip.sv
    rtl/sfu_rsqrt.sv
    rtl/sfu_elem.sv
    rtl/srdhm.sv
    rtl/sat_counter.sv
    rtl/softmax.sv
    rtl/layernorm.sv
    rtl/sfu.sv
    rtl/requant.sv
    rtl/vec_alu.sv
    rtl/vec_reduce.sv
    rtl/vec_unit.sv
    rtl/sfu_vec.sv
    rtl/vec_seq_walk.sv
    rtl/vec_seq_rd.sv
    rtl/vec_seq_wr.sv
    rtl/pbuf_loader.sv
    rtl/vec_seq.sv
    rtl/spad_bank.sv
    rtl/buf_spad.sv
    rtl/nanoxl_top.sv
    rtl/nanoxl_axi_wrap.v
}
