Critical-path share and engine occupancy per op class (`python3 bench/roofline.py`; measured-RTL cycle model).

**kv260 shakespeare W8 decode, as built** (753k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 92.7 % | 92.7 % |
| dma params | 3.5 % | 3.6 % |
| gemm linear | 0.8 % | 44.2 % |
| gemm attn | 2.6 % | 2.7 % |
| requant | 0.0 % | 2.3 % |

**kv260 shakespeare W8 prefill, as built** (3684k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 37.9 % |
| dma params | 0.0 % | 1.5 % |
| gemm linear | 77.1 % | 72.3 % |
| gemm attn | 6.6 % | 6.0 % |
| requant | 1.0 % | 13.7 % |
| ln | 3.8 % | 3.8 % |
| softmax | 8.0 % | 7.9 % |
| gelu | 1.0 % | 1.0 % |
| vec add | 1.0 % | 1.0 % |
| kv append | 1.0 % | 1.0 % |

**kv260 shakespeare W8 decode, all fixes** (361k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 92.4 % | 92.2 % |
| dma params | 3.2 % | 3.2 % |
| gemm linear | 0.9 % | 46.1 % |
| gemm attn | 2.9 % | 2.8 % |
| requant | 0.1 % | 4.8 % |
| softmax | 0.5 % | 0.5 % |

**kv260 shakespeare W8 prefill, all fixes** (3400k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 19.6 % |
| dma params | 0.0 % | 0.7 % |
| gemm linear | 78.6 % | 78.3 % |
| gemm attn | 6.7 % | 6.5 % |
| requant | 1.1 % | 14.8 % |
| ln | 3.1 % | 3.1 % |
| softmax | 6.9 % | 6.9 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**kv260 shakespeare W4 decode, as built** (421k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 67.0 % | 85.3 % |
| dma params | 6.3 % | 6.4 % |
| gemm linear | 21.5 % | 81.6 % |
| gemm attn | 4.7 % | 4.8 % |
| requant | 0.1 % | 3.1 % |
| ln | 0.0 % | 0.6 % |

**kv260 shakespeare W4 prefill, as built** (3686k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 32.4 % |
| dma params | 0.0 % | 1.5 % |
| gemm linear | 77.6 % | 74.9 % |
| gemm attn | 6.6 % | 6.0 % |
| requant | 0.5 % | 12.7 % |
| ln | 3.8 % | 3.8 % |
| softmax | 7.9 % | 7.9 % |
| gelu | 1.0 % | 1.0 % |
| vec add | 1.0 % | 1.0 % |
| kv append | 1.0 % | 1.0 % |

**kv260 shakespeare W4 decode, all fixes** (206k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 10.5 % | 83.5 % |
| dma params | 3.4 % | 5.6 % |
| gemm linear | 79.4 % | 85.8 % |
| gemm attn | 5.0 % | 4.9 % |
| requant | 0.4 % | 6.3 % |
| softmax | 0.8 % | 0.9 % |

**kv260 shakespeare W4 prefill, all fixes** (3378k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 16.9 % |
| dma params | 0.0 % | 0.7 % |
| gemm linear | 78.9 % | 78.8 % |
| gemm attn | 6.7 % | 6.5 % |
| requant | 0.7 % | 13.9 % |
| ln | 3.1 % | 3.1 % |
| softmax | 6.9 % | 6.9 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**pynq-z2 shakespeare W8 decode, as built** (1518k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.1 % | 91.1 % |
| dma params | 3.2 % | 3.2 % |
| dma act/kv | 5.1 % | 5.5 % |
| gemm linear | 0.4 % | 43.9 % |
| gemm attn | 0.1 % | 2.5 % |
| requant | 0.0 % | 1.7 % |

**pynq-z2 shakespeare W8 prefill, as built** (13221k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 3.6 % | 83.7 % |
| dma params | 0.2 % | 3.0 % |
| dma act/kv | 2.9 % | 7.7 % |
| gemm linear | 80.0 % | 80.5 % |
| gemm attn | 6.0 % | 5.6 % |
| requant | 0.2 % | 5.7 % |
| ln | 2.2 % | 2.4 % |
| softmax | 3.2 % | 3.3 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W8 decode, all fixes** (1167k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.3 % | 91.3 % |
| dma params | 3.1 % | 3.1 % |
| dma act/kv | 5.1 % | 5.1 % |
| gemm linear | 0.3 % | 28.5 % |
| gemm attn | 0.1 % | 1.7 % |
| requant | 0.0 % | 2.3 % |

**pynq-z2 shakespeare W8 prefill, all fixes** (12389k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.1 % | 68.8 % |
| dma params | 0.2 % | 2.4 % |
| dma act/kv | 2.7 % | 6.6 % |
| gemm linear | 84.8 % | 85.9 % |
| gemm attn | 5.7 % | 5.5 % |
| requant | 0.2 % | 6.1 % |
| ln | 1.7 % | 1.8 % |
| softmax | 2.8 % | 2.8 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W4 decode, as built** (859k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 32.9 % | 82.8 % |
| dma params | 5.5 % | 5.7 % |
| dma act/kv | 9.0 % | 9.7 % |
| gemm linear | 52.3 % | 82.3 % |
| gemm attn | 0.2 % | 4.5 % |
| requant | 0.0 % | 2.1 % |

**pynq-z2 shakespeare W4 prefill, as built** (12822k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.2 % | 44.4 % |
| dma params | 0.2 % | 3.1 % |
| dma act/kv | 3.0 % | 8.0 % |
| gemm linear | 82.8 % | 83.0 % |
| gemm attn | 6.2 % | 5.8 % |
| requant | 0.2 % | 4.7 % |
| ln | 2.3 % | 2.5 % |
| softmax | 3.3 % | 3.4 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W4 decode, all fixes** (655k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 83.5 % | 84.1 % |
| dma params | 5.6 % | 5.6 % |
| dma act/kv | 9.1 % | 9.2 % |
| gemm linear | 1.5 % | 57.2 % |
| gemm attn | 0.1 % | 2.9 % |
| requant | 0.0 % | 2.7 % |

**pynq-z2 shakespeare W4 prefill, all fixes** (12314k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.2 % | 35.8 % |
| dma params | 0.2 % | 2.4 % |
| dma act/kv | 0.8 % | 6.6 % |
| gemm linear | 86.6 % | 86.5 % |
| gemm attn | 5.7 % | 5.5 % |
| requant | 0.3 % | 4.9 % |
| ln | 1.7 % | 1.8 % |
| softmax | 2.8 % | 2.8 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |
