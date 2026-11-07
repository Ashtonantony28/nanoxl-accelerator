Critical-path share and engine occupancy per op class (`python3 bench/roofline.py`; measured-RTL cycle model).

**kv260 shakespeare W8 decode, as built** (751k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 93.0 % | 93.0 % |
| dma params | 3.5 % | 3.6 % |
| gemm linear | 0.8 % | 44.3 % |
| gemm attn | 2.5 % | 2.7 % |

**kv260 shakespeare W8 prefill, as built** (3469k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 40.2 % |
| dma params | 0.0 % | 1.5 % |
| dma act/kv | 0.5 % | 0.5 % |
| gemm linear | 78.5 % | 76.8 % |
| gemm attn | 6.4 % | 6.4 % |
| requant | 0.4 % | 5.9 % |
| ln | 3.8 % | 3.8 % |
| softmax | 7.7 % | 7.7 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 0.5 % | 0.5 % |
| kv append | 1.1 % | 1.1 % |

**kv260 shakespeare W8 decode, all fixes** (359k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 92.9 % | 92.7 % |
| dma params | 3.2 % | 3.2 % |
| gemm linear | 0.9 % | 46.4 % |
| gemm attn | 2.7 % | 2.8 % |

**kv260 shakespeare W8 prefill, all fixes** (3349k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 19.9 % |
| dma params | 0.0 % | 0.7 % |
| gemm linear | 79.6 % | 79.5 % |
| gemm attn | 6.5 % | 6.6 % |
| requant | 0.5 % | 6.1 % |
| ln | 3.2 % | 3.2 % |
| softmax | 7.2 % | 7.2 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 0.6 % | 0.6 % |
| kv append | 1.1 % | 1.1 % |

**kv260 shakespeare W4 decode, as built** (419k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 85.1 % | 85.8 % |
| dma params | 6.3 % | 6.4 % |
| gemm linear | 3.9 % | 80.8 % |
| gemm attn | 4.5 % | 4.8 % |
| ln | 0.0 % | 0.6 % |

**kv260 shakespeare W4 prefill, as built** (3537k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 33.7 % |
| dma params | 0.0 % | 1.5 % |
| gemm linear | 79.1 % | 77.8 % |
| gemm attn | 6.3 % | 6.3 % |
| requant | 0.2 % | 5.0 % |
| ln | 3.7 % | 3.7 % |
| softmax | 7.6 % | 7.6 % |
| gelu | 1.0 % | 1.0 % |
| vec add | 0.5 % | 0.5 % |
| kv append | 1.0 % | 1.0 % |

**kv260 shakespeare W4 decode, all fixes** (202k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 79.5 % | 85.2 % |
| dma params | 5.5 % | 5.7 % |
| gemm linear | 9.6 % | 85.0 % |
| gemm attn | 4.8 % | 5.0 % |

**kv260 shakespeare W4 prefill, all fixes** (3342k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 17.1 % |
| dma params | 0.0 % | 0.7 % |
| gemm linear | 79.8 % | 79.7 % |
| gemm attn | 6.5 % | 6.6 % |
| requant | 0.2 % | 5.3 % |
| ln | 3.2 % | 3.2 % |
| softmax | 7.2 % | 7.2 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 0.6 % | 0.6 % |
| kv append | 1.1 % | 1.1 % |

**pynq-z2 shakespeare W8 decode, as built** (1517k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.2 % | 91.2 % |
| dma params | 3.2 % | 3.2 % |
| dma act/kv | 5.4 % | 5.5 % |
| gemm linear | 0.1 % | 43.9 % |
| gemm attn | 0.1 % | 2.5 % |

**pynq-z2 shakespeare W8 prefill, as built** (12978k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 37.2 % | 85.3 % |
| dma params | 0.6 % | 3.0 % |
| dma act/kv | 3.0 % | 7.9 % |
| gemm linear | 46.4 % | 82.0 % |
| gemm attn | 5.8 % | 5.7 % |
| requant | 0.1 % | 4.3 % |
| ln | 2.0 % | 2.5 % |
| softmax | 3.3 % | 3.3 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 0.5 % | 0.6 % |

**pynq-z2 shakespeare W8 decode, all fixes** (1166k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.4 % | 91.4 % |
| dma params | 3.1 % | 3.1 % |
| dma act/kv | 5.1 % | 5.2 % |
| gemm linear | 0.3 % | 28.5 % |
| gemm attn | 0.1 % | 1.7 % |

**pynq-z2 shakespeare W8 prefill, all fixes** (12363k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.1 % | 68.9 % |
| dma params | 0.2 % | 2.4 % |
| dma act/kv | 2.7 % | 6.6 % |
| gemm linear | 85.0 % | 86.1 % |
| gemm attn | 5.5 % | 5.5 % |
| requant | 0.1 % | 4.6 % |
| ln | 1.7 % | 1.9 % |
| softmax | 2.8 % | 2.9 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W4 decode, as built** (852k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 82.9 % | 83.5 % |
| dma params | 5.7 % | 5.7 % |
| dma act/kv | 9.1 % | 9.8 % |
| gemm linear | 2.1 % | 79.4 % |
| gemm attn | 0.2 % | 4.5 % |

**pynq-z2 shakespeare W4 prefill, as built** (12674k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.3 % | 44.9 % |
| dma params | 0.3 % | 3.1 % |
| dma act/kv | 3.0 % | 8.1 % |
| gemm linear | 82.8 % | 84.0 % |
| gemm attn | 6.0 % | 5.9 % |
| requant | 0.1 % | 3.8 % |
| ln | 2.3 % | 2.5 % |
| softmax | 3.3 % | 3.4 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |

**pynq-z2 shakespeare W4 decode, all fixes** (653k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 83.7 % | 84.2 % |
| dma params | 5.6 % | 5.6 % |
| dma act/kv | 9.1 % | 9.2 % |
| gemm linear | 1.4 % | 52.5 % |
| gemm attn | 0.1 % | 3.0 % |

**pynq-z2 shakespeare W4 prefill, all fixes** (12284k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.2 % | 35.8 % |
| dma params | 0.2 % | 2.4 % |
| dma act/kv | 0.8 % | 6.6 % |
| gemm linear | 86.7 % | 86.7 % |
| gemm attn | 5.6 % | 5.5 % |
| requant | 0.2 % | 3.9 % |
| ln | 1.7 % | 1.9 % |
| softmax | 2.8 % | 2.9 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |
