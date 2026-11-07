Critical-path share and engine occupancy per op class (`python3 bench/roofline.py`; measured-RTL cycle model).

**kv260 shakespeare W8 decode, as built** (754k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 92.5 % | 92.5 % |
| dma params | 3.5 % | 3.5 % |
| gemm linear | 0.8 % | 44.1 % |
| gemm attn | 2.6 % | 2.7 % |
| requant | 0.1 % | 5.1 % |

**kv260 shakespeare W8 prefill, as built** (4402k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 31.7 % |
| dma params | 0.0 % | 1.2 % |
| gemm linear | 64.4 % | 60.5 % |
| gemm attn | 4.7 % | 5.0 % |
| requant | 4.3 % | 37.4 % |
| ln | 6.2 % | 6.2 % |
| softmax | 11.2 % | 11.2 % |
| gelu | 3.4 % | 3.4 % |
| vec add | 3.4 % | 3.4 % |
| kv append | 2.1 % | 2.1 % |

**kv260 shakespeare W8 decode, all fixes** (362k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 92.2 % | 92.1 % |
| dma params | 3.2 % | 3.2 % |
| gemm linear | 0.9 % | 46.0 % |
| gemm attn | 2.9 % | 2.8 % |
| requant | 0.1 % | 5.1 % |
| softmax | 0.6 % | 0.7 % |

**kv260 shakespeare W8 prefill, all fixes** (3384k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 19.7 % |
| dma params | 0.0 % | 0.7 % |
| gemm linear | 79.0 % | 78.7 % |
| gemm attn | 6.7 % | 6.5 % |
| requant | 1.1 % | 15.0 % |
| ln | 2.8 % | 2.8 % |
| softmax | 6.7 % | 6.7 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**kv260 shakespeare W4 decode, as built** (422k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 57.4 % | 85.0 % |
| dma params | 6.0 % | 6.3 % |
| gemm linear | 30.8 % | 81.3 % |
| gemm attn | 4.7 % | 4.7 % |
| requant | 0.2 % | 8.0 % |
| ln | 0.1 % | 0.8 % |
| softmax | 0.7 % | 0.8 % |

**kv260 shakespeare W4 prefill, as built** (4382k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 27.2 % |
| dma params | 0.0 % | 1.2 % |
| gemm linear | 63.0 % | 63.0 % |
| gemm attn | 4.7 % | 5.0 % |
| requant | 5.6 % | 37.4 % |
| ln | 6.2 % | 6.2 % |
| softmax | 11.3 % | 11.3 % |
| gelu | 3.4 % | 3.4 % |
| vec add | 3.4 % | 3.4 % |
| kv append | 2.1 % | 2.1 % |

**kv260 shakespeare W4 decode, all fixes** (207k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 10.5 % | 83.1 % |
| dma params | 3.3 % | 5.6 % |
| gemm linear | 79.0 % | 85.4 % |
| gemm attn | 5.0 % | 4.8 % |
| requant | 0.4 % | 6.6 % |
| softmax | 1.1 % | 1.3 % |

**kv260 shakespeare W4 prefill, all fixes** (3362k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.0 % | 17.0 % |
| dma params | 0.0 % | 0.7 % |
| gemm linear | 79.3 % | 79.2 % |
| gemm attn | 6.8 % | 6.6 % |
| requant | 0.7 % | 14.0 % |
| ln | 2.8 % | 2.8 % |
| softmax | 6.8 % | 6.8 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |
| kv append | 1.1 % | 1.1 % |

**pynq-z2 shakespeare W8 decode, as built** (1518k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.1 % | 91.1 % |
| dma params | 3.2 % | 3.2 % |
| dma act/kv | 5.1 % | 5.5 % |
| gemm linear | 0.4 % | 43.8 % |
| gemm attn | 0.1 % | 2.5 % |
| requant | 0.0 % | 3.1 % |

**pynq-z2 shakespeare W8 prefill, as built** (13604k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 3.4 % | 81.3 % |
| dma params | 0.2 % | 2.9 % |
| dma act/kv | 2.8 % | 7.5 % |
| gemm linear | 77.7 % | 78.3 % |
| gemm attn | 5.8 % | 5.5 % |
| requant | 0.6 % | 14.7 % |
| ln | 2.7 % | 2.9 % |
| softmax | 4.6 % | 4.6 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |

**pynq-z2 shakespeare W8 decode, all fixes** (1167k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 91.3 % | 91.3 % |
| dma params | 3.1 % | 3.1 % |
| dma act/kv | 5.1 % | 5.1 % |
| gemm linear | 0.3 % | 28.5 % |
| gemm attn | 0.1 % | 1.7 % |
| requant | 0.0 % | 2.3 % |

**pynq-z2 shakespeare W8 prefill, all fixes** (12392k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.1 % | 68.8 % |
| dma params | 0.2 % | 2.4 % |
| dma act/kv | 2.7 % | 6.6 % |
| gemm linear | 84.8 % | 85.9 % |
| gemm attn | 5.7 % | 5.5 % |
| requant | 0.2 % | 6.2 % |
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
| requant | 0.1 % | 4.4 % |

**pynq-z2 shakespeare W4 prefill, as built** (13217k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.2 % | 43.0 % |
| dma params | 0.2 % | 3.0 % |
| dma act/kv | 0.9 % | 7.7 % |
| gemm linear | 82.3 % | 80.6 % |
| gemm attn | 5.9 % | 5.6 % |
| requant | 0.8 % | 14.0 % |
| ln | 2.7 % | 3.0 % |
| softmax | 4.7 % | 4.7 % |
| gelu | 1.1 % | 1.1 % |
| vec add | 1.1 % | 1.1 % |

**pynq-z2 shakespeare W4 decode, all fixes** (655k cycles per token)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 83.5 % | 84.1 % |
| dma params | 5.6 % | 5.6 % |
| dma act/kv | 9.1 % | 9.2 % |
| gemm linear | 1.5 % | 57.1 % |
| gemm attn | 0.1 % | 2.9 % |
| requant | 0.0 % | 2.8 % |

**pynq-z2 shakespeare W4 prefill, all fixes** (12317k cycles, T=256)

| op class | critical path | engine busy |
|---|---|---|
| dma weights | 0.2 % | 35.8 % |
| dma params | 0.2 % | 2.4 % |
| dma act/kv | 0.8 % | 6.6 % |
| gemm linear | 86.6 % | 86.4 % |
| gemm attn | 5.7 % | 5.5 % |
| requant | 0.3 % | 5.0 % |
| ln | 1.7 % | 1.8 % |
| softmax | 2.8 % | 2.9 % |
| gelu | 1.2 % | 1.2 % |
| vec add | 0.6 % | 0.6 % |
