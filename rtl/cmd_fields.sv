// cmd_fields.sv -- GENERATED from nanoxl/isa.py by tests/rtl/test_cmd_isagen.py; do not edit.
// Regenerate: python3 tests/rtl/test_cmd_isagen.py  (test_cmd_isagen fails while stale)
//
// Pure combinational field extraction for a 128-bit command word (docs/isa.md):
//   known      opcode is defined in the ISA (opcode 0 and every unlisted value are illegal)
//   resv_ok    all bits above the opcode's last field are zero
//   is_<op>    one-hot opcode match
//   fixed_q    queue for opcodes with a fixed queue (0 DMA, 1 ARRAY, 2 VEC, 3 CTRL);
//              WAIT/SIGNAL report 3 here and carry their queue in the queue field
//   <op>_<f>   every field, decoded: 'm1' counts have one extra bit and the +1 applied,
//              's' fields are raw two's complement
module cmd_fields (
    input  logic [127:0] w,
    output logic [5:0]   opcode,
    output logic         known,
    output logic         resv_ok,
    output logic [1:0]   fixed_q,
    output logic         is_end,
    output logic         is_irq,
    output logic         is_wait,
    output logic         is_signal,
    output logic         is_dma_load,
    output logic         is_dma_store,
    output logic         is_gemm,
    output logic         is_attn_tile,
    output logic         is_vec,
    output logic         is_requant,
    output logic         is_ln,
    output logic         is_softmax_online,
    output logic         is_gelu,
    output logic         is_kv_append,
    output logic [15:0]  irq_code,
    output logic [1:0]   wait_queue,
    output logic [4:0]   wait_sem,
    output logic [7:0]   wait_count,
    output logic [1:0]   signal_queue,
    output logic [4:0]   signal_sem,
    output logic [1:0]   dma_load_chan,
    output logic [2:0]   dma_load_sp_buf,
    output logic [20:0]  dma_load_sp_addr,
    output logic [31:0]  dma_load_ddr_addr,
    output logic [12:0]  dma_load_rows,
    output logic [16:0]  dma_load_row_bytes,
    output logic [23:0]  dma_load_ddr_stride,
    output logic [11:0]  dma_load_sp_stride,
    output logic [1:0]   dma_store_chan,
    output logic [2:0]   dma_store_sp_buf,
    output logic [20:0]  dma_store_sp_addr,
    output logic [31:0]  dma_store_ddr_addr,
    output logic [12:0]  dma_store_rows,
    output logic [16:0]  dma_store_row_bytes,
    output logic [23:0]  dma_store_ddr_stride,
    output logic [11:0]  dma_store_sp_stride,
    output logic         gemm_int4,
    output logic         gemm_gemv,
    output logic         gemm_w_trans,
    output logic         gemm_accumulate,
    output logic [19:0]  gemm_a_addr,
    output logic         gemm_w_buf,
    output logic [20:0]  gemm_w_addr,
    output logic [19:0]  gemm_c_addr,
    output logic [12:0]  gemm_m,
    output logic [12:0]  gemm_n,
    output logic [12:0]  gemm_k,
    output logic [9:0]   gemm_a_stride,
    output logic [9:0]   gemm_c_stride,
    output logic         attn_tile_first,
    output logic         attn_tile_last,
    output logic [19:0]  attn_tile_q_addr,
    output logic [20:0]  attn_tile_k_addr,
    output logic [20:0]  attn_tile_v_addr,
    output logic [19:0]  attn_tile_o_addr,
    output logic [19:0]  attn_tile_ctx_addr,
    output logic [6:0]   attn_tile_rows,
    output logic [12:0]  attn_tile_len,
    output logic [4:0]   vec_vop,
    output logic [1:0]   vec_dt_in,
    output logic [1:0]   vec_dt_out,
    output logic [2:0]   vec_src0_buf,
    output logic [20:0]  vec_src0_addr,
    output logic [2:0]   vec_src1_buf,
    output logic [20:0]  vec_src1_addr,
    output logic [2:0]   vec_dst_buf,
    output logic [20:0]  vec_dst_addr,
    output logic [20:0]  vec_len,
    output logic [15:0]  vec_imm,
    output logic         requant_out16,
    output logic         requant_per_tensor,
    output logic [19:0]  requant_src_addr,
    output logic [19:0]  requant_dst_addr,
    output logic [19:0]  requant_param_addr,
    output logic [12:0]  requant_m,
    output logic [12:0]  requant_n,
    output logic [11:0]  requant_src_stride,
    output logic [11:0]  requant_dst_stride,
    output logic         ln_in16,
    output logic [19:0]  ln_src_addr,
    output logic [19:0]  ln_dst_addr,
    output logic [19:0]  ln_param_addr,
    output logic [12:0]  ln_rows,
    output logic [12:0]  ln_cols,
    output logic [11:0]  ln_src_stride,
    output logic [11:0]  ln_dst_stride,
    output logic         softmax_online_first,
    output logic         softmax_online_last,
    output logic         softmax_online_causal,
    output logic [19:0]  softmax_online_src_addr,
    output logic [19:0]  softmax_online_dst_addr,
    output logic [19:0]  softmax_online_ctx_addr,
    output logic [12:0]  softmax_online_rows,
    output logic [12:0]  softmax_online_cols,
    output logic [12:0]  softmax_online_diag,
    output logic [9:0]   softmax_online_src_stride,
    output logic [9:0]   softmax_online_dst_stride,
    output logic [19:0]  gelu_src_addr,
    output logic [19:0]  gelu_dst_addr,
    output logic [19:0]  gelu_lut_addr,
    output logic [20:0]  gelu_len,
    output logic [19:0]  kv_append_src_addr,
    output logic [20:0]  kv_append_dst_addr,
    output logic [11:0]  kv_append_pos,
    output logic [6:0]   kv_append_heads,
    output logic [8:0]   kv_append_dh,
    output logic [12:0]  kv_append_ctx,
    output logic [12:0]  kv_append_rows,
    output logic [11:0]  kv_append_src_stride
);
    assign opcode = w[5:0];
    assign is_end = (w[5:0] == 6'h01);
    assign is_irq = (w[5:0] == 6'h02);
    assign is_wait = (w[5:0] == 6'h03);
    assign is_signal = (w[5:0] == 6'h04);
    assign is_dma_load = (w[5:0] == 6'h08);
    assign is_dma_store = (w[5:0] == 6'h09);
    assign is_gemm = (w[5:0] == 6'h10);
    assign is_attn_tile = (w[5:0] == 6'h11);
    assign is_vec = (w[5:0] == 6'h18);
    assign is_requant = (w[5:0] == 6'h19);
    assign is_ln = (w[5:0] == 6'h1a);
    assign is_softmax_online = (w[5:0] == 6'h1b);
    assign is_gelu = (w[5:0] == 6'h1c);
    assign is_kv_append = (w[5:0] == 6'h1d);
    assign known = is_end | is_irq | is_wait | is_signal | is_dma_load | is_dma_store | is_gemm | is_attn_tile | is_vec | is_requant | is_ln | is_softmax_online | is_gelu | is_kv_append;
    assign resv_ok = (is_end & (w[127:6] == '0))
                   | (is_irq & (w[127:22] == '0))
                   | (is_wait & (w[127:21] == '0))
                   | (is_signal & (w[127:13] == '0))
                   | (is_dma_load & 1'b1)
                   | (is_dma_store & 1'b1)
                   | (is_gemm & 1'b1)
                   | (is_attn_tile & 1'b1)
                   | (is_vec & (w[127:123] == '0))
                   | (is_requant & (w[127:116] == '0))
                   | (is_ln & (w[127:115] == '0))
                   | (is_softmax_online & (w[127:126] == '0))
                   | (is_gelu & (w[127:86] == '0))
                   | (is_kv_append & (w[127:109] == '0));
    assign fixed_q = is_end ? 2'd3
                   : is_irq ? 2'd3
                   : is_wait ? 2'd3
                   : is_signal ? 2'd3
                   : is_dma_load ? 2'd0
                   : is_dma_store ? 2'd0
                   : is_gemm ? 2'd1
                   : is_attn_tile ? 2'd1
                   : is_vec ? 2'd2
                   : is_requant ? 2'd2
                   : is_ln ? 2'd2
                   : is_softmax_online ? 2'd2
                   : is_gelu ? 2'd2
                   : is_kv_append ? 2'd2 : 2'd3;
    assign irq_code = w[21:6];
    assign wait_queue = w[7:6];
    assign wait_sem = w[12:8];
    assign wait_count = w[20:13];
    assign signal_queue = w[7:6];
    assign signal_sem = w[12:8];
    assign dma_load_chan = w[7:6];
    assign dma_load_sp_buf = w[10:8];
    assign dma_load_sp_addr = w[31:11];
    assign dma_load_ddr_addr = w[63:32];
    assign dma_load_rows = {1'b0, w[75:64]} + 13'd1;
    assign dma_load_row_bytes = {1'b0, w[91:76]} + 17'd1;
    assign dma_load_ddr_stride = w[115:92];
    assign dma_load_sp_stride = w[127:116];
    assign dma_store_chan = w[7:6];
    assign dma_store_sp_buf = w[10:8];
    assign dma_store_sp_addr = w[31:11];
    assign dma_store_ddr_addr = w[63:32];
    assign dma_store_rows = {1'b0, w[75:64]} + 13'd1;
    assign dma_store_row_bytes = {1'b0, w[91:76]} + 17'd1;
    assign dma_store_ddr_stride = w[115:92];
    assign dma_store_sp_stride = w[127:116];
    assign gemm_int4 = w[6];
    assign gemm_gemv = w[7];
    assign gemm_w_trans = w[8];
    assign gemm_accumulate = w[9];
    assign gemm_a_addr = w[29:10];
    assign gemm_w_buf = w[30];
    assign gemm_w_addr = w[51:31];
    assign gemm_c_addr = w[71:52];
    assign gemm_m = {1'b0, w[83:72]} + 13'd1;
    assign gemm_n = {1'b0, w[95:84]} + 13'd1;
    assign gemm_k = {1'b0, w[107:96]} + 13'd1;
    assign gemm_a_stride = w[117:108];
    assign gemm_c_stride = w[127:118];
    assign attn_tile_first = w[6];
    assign attn_tile_last = w[7];
    assign attn_tile_q_addr = w[27:8];
    assign attn_tile_k_addr = w[48:28];
    assign attn_tile_v_addr = w[69:49];
    assign attn_tile_o_addr = w[89:70];
    assign attn_tile_ctx_addr = w[109:90];
    assign attn_tile_rows = {1'b0, w[115:110]} + 7'd1;
    assign attn_tile_len = {1'b0, w[127:116]} + 13'd1;
    assign vec_vop = w[10:6];
    assign vec_dt_in = w[12:11];
    assign vec_dt_out = w[14:13];
    assign vec_src0_buf = w[17:15];
    assign vec_src0_addr = w[38:18];
    assign vec_src1_buf = w[41:39];
    assign vec_src1_addr = w[62:42];
    assign vec_dst_buf = w[65:63];
    assign vec_dst_addr = w[86:66];
    assign vec_len = {1'b0, w[106:87]} + 21'd1;
    assign vec_imm = w[122:107];
    assign requant_out16 = w[6];
    assign requant_per_tensor = w[7];
    assign requant_src_addr = w[27:8];
    assign requant_dst_addr = w[47:28];
    assign requant_param_addr = w[67:48];
    assign requant_m = {1'b0, w[79:68]} + 13'd1;
    assign requant_n = {1'b0, w[91:80]} + 13'd1;
    assign requant_src_stride = w[103:92];
    assign requant_dst_stride = w[115:104];
    assign ln_in16 = w[6];
    assign ln_src_addr = w[26:7];
    assign ln_dst_addr = w[46:27];
    assign ln_param_addr = w[66:47];
    assign ln_rows = {1'b0, w[78:67]} + 13'd1;
    assign ln_cols = {1'b0, w[90:79]} + 13'd1;
    assign ln_src_stride = w[102:91];
    assign ln_dst_stride = w[114:103];
    assign softmax_online_first = w[6];
    assign softmax_online_last = w[7];
    assign softmax_online_causal = w[8];
    assign softmax_online_src_addr = w[28:9];
    assign softmax_online_dst_addr = w[48:29];
    assign softmax_online_ctx_addr = w[68:49];
    assign softmax_online_rows = {1'b0, w[80:69]} + 13'd1;
    assign softmax_online_cols = {1'b0, w[92:81]} + 13'd1;
    assign softmax_online_diag = w[105:93];
    assign softmax_online_src_stride = w[115:106];
    assign softmax_online_dst_stride = w[125:116];
    assign gelu_src_addr = w[25:6];
    assign gelu_dst_addr = w[45:26];
    assign gelu_lut_addr = w[65:46];
    assign gelu_len = {1'b0, w[85:66]} + 21'd1;
    assign kv_append_src_addr = w[25:6];
    assign kv_append_dst_addr = w[46:26];
    assign kv_append_pos = w[58:47];
    assign kv_append_heads = {1'b0, w[64:59]} + 7'd1;
    assign kv_append_dh = {1'b0, w[72:65]} + 9'd1;
    assign kv_append_ctx = {1'b0, w[84:73]} + 13'd1;
    assign kv_append_rows = {1'b0, w[96:85]} + 13'd1;
    assign kv_append_src_stride = w[108:97];
endmodule
