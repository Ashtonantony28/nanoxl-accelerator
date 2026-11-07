Critical-path share and engine occupancy per op class (`python3 bench/roofline.py`; measured-RTL cycle model).

**kv260 shakespeare W8 decode, as built** (404k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 63.5 % | 165.2 % |
| dma params | 2.7 % | 5.7 % |
| gemm linear | 28.1 % | 82.4 % |
| gemm attn | 4.7 % | 5.0 % |
| requant | 0.1 % | 4.3 % |
| ln | 0.2 % | 0.7 % |

**kv260 shakespeare W8 prefill, as built** (3502k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 38.1 % |
| dma params | 0.0 % | 1.3 % |
| dma act/kv | 0.5 % | 0.5 % |
| gemm linear | 76.3 % | 76.1 % |
| gemm attn | 6.5 % | 6.3 % |
| requant | 1.0 % | 14.4 % |
| ln | 4.0 % | 4.0 % |
| softmax | 8.4 % | 8.4 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**kv260 shakespeare W8 decode, all fixes** (376k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.9 % | 177.3 % |
| dma params | 3.2 % | 6.2 % |
| gemm linear | 1.5 % | 44.2 % |
| gemm attn | 2.8 % | 2.7 % |
| requant | 0.1 % | 4.6 % |
| softmax | 0.4 % | 0.5 % |

**kv260 shakespeare W8 prefill, all fixes** (3406k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 39.2 % |
| dma params | 0.0 % | 1.4 % |
| dma act/kv | 0.5 % | 0.5 % |
| gemm linear | 78.5 % | 78.2 % |
| gemm attn | 6.7 % | 6.5 % |
| requant | 1.1 % | 14.8 % |
| ln | 3.1 % | 3.1 % |
| softmax | 6.9 % | 6.9 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**kv260 shakespeare W4 decode, as built** (384k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 3.8 % | 89.7 % |
| dma params | 0.1 % | 6.0 % |
| gemm linear | 89.9 % | 89.4 % |
| gemm attn | 4.7 % | 5.2 % |
| requant | 0.3 % | 3.4 % |
| ln | 0.7 % | 0.7 % |

**kv260 shakespeare W4 prefill, as built** (3576k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 32.1 % |
| dma params | 0.0 % | 1.3 % |
| gemm linear | 77.4 % | 77.2 % |
| gemm attn | 6.4 % | 6.2 % |
| requant | 0.5 % | 13.1 % |
| ln | 3.9 % | 3.9 % |
| softmax | 8.2 % | 8.2 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.0 % | 1.0 % |
| kv append | 1.0 % | 1.0 % |

**kv260 shakespeare W4 decode, all fixes** (226k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 53.5 % | 152.3 % |
| dma params | 6.5 % | 10.2 % |
| gemm linear | 34.3 % | 78.1 % |
| gemm attn | 4.6 % | 4.4 % |
| requant | 0.1 % | 5.8 % |
| softmax | 0.7 % | 0.8 % |

**kv260 shakespeare W4 prefill, all fixes** (3384k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 33.9 % |
| dma params | 0.0 % | 1.4 % |
| dma act/kv | 0.5 % | 0.5 % |
| gemm linear | 78.8 % | 78.7 % |
| gemm attn | 6.7 % | 6.5 % |
| requant | 0.7 % | 13.9 % |
| ln | 3.1 % | 3.1 % |
| softmax | 6.9 % | 6.9 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**pynq-z2 shakespeare W8 decode, as built** (1176k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.5 % | 179.8 % |
| dma params | 2.4 % | 4.7 % |
| dma act/kv | 5.0 % | 9.9 % |
| gemm linear | 1.0 % | 56.6 % |
| gemm attn | 0.1 % | 3.3 % |
| requant | 0.0 % | 2.2 % |

**pynq-z2 shakespeare W8 prefill, as built** (12741k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 1.2 % | 126.7 % |
| dma params | 0.0 % | 3.3 % |
| dma act/kv | 3.1 % | 9.4 % |
| gemm linear | 81.8 % | 83.6 % |
| gemm attn | 6.0 % | 5.8 % |
| requant | 0.2 % | 6.0 % |
| ln | 2.5 % | 2.5 % |
| softmax | 3.3 % | 3.4 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W8 decode, all fixes** (1173k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 93.2 % | 180.3 % |
| dma params | 1.2 % | 4.7 % |
| dma act/kv | 5.3 % | 9.9 % |
| gemm linear | 0.1 % | 28.4 % |
| gemm attn | 0.1 % | 1.6 % |
| requant | 0.0 % | 2.3 % |

**pynq-z2 shakespeare W8 prefill, all fixes** (12499k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 1.2 % | 129.3 % |
| dma params | 0.0 % | 3.4 % |
| dma act/kv | 3.2 % | 9.5 % |
| gemm linear | 83.4 % | 85.2 % |
| gemm attn | 5.6 % | 5.4 % |
| requant | 0.2 % | 6.1 % |
| ln | 1.8 % | 1.8 % |
| softmax | 2.8 % | 2.8 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W4 decode, as built** (808k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 7.5 % | 134.9 % |
| dma params | 3.8 % | 7.3 % |
| dma act/kv | 7.4 % | 14.4 % |
| gemm linear | 80.9 % | 87.5 % |
| gemm attn | 0.1 % | 4.8 % |
| requant | 0.1 % | 2.2 % |

**pynq-z2 shakespeare W4 prefill, as built** (12597k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.3 % | 46.2 % |
| dma params | 0.0 % | 3.0 % |
| dma act/kv | 3.2 % | 9.5 % |
| gemm linear | 82.6 % | 84.5 % |
| gemm attn | 6.0 % | 5.9 % |
| requant | 0.2 % | 4.8 % |
| ln | 2.5 % | 2.5 % |
| softmax | 3.3 % | 3.4 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W4 decode, all fixes** (665k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 86.1 % | 164.5 % |
| dma params | 1.5 % | 9.5 % |
| dma act/kv | 9.0 % | 17.5 % |
| gemm linear | 3.2 % | 56.2 % |
| gemm attn | 0.1 % | 2.9 % |
| requant | 0.0 % | 2.7 % |

**pynq-z2 shakespeare W4 prefill, all fixes** (12362k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.8 % | 47.1 % |
| dma params | 0.3 % | 3.1 % |
| dma act/kv | 2.9 % | 9.6 % |
| gemm linear | 84.2 % | 86.1 % |
| gemm attn | 5.3 % | 5.5 % |
| requant | 0.2 % | 4.9 % |
| ln | 1.7 % | 1.8 % |
| softmax | 2.7 % | 2.8 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |
