| board | W | fix | decode tok/s | x as built | prefill ms (T=256) | x as built |
|---|---|---|---|---|---|---|
| kv260 | 8 | as built | 333 | 1.00 | 13.88 | 1.00 |
| kv260 | 8 | coalesce contiguous DMA rows | 333 | 1.00 | 13.88 | 1.00 |
| kv260 | 8 | + coalesce contiguous DMA rows | 333 | 1.00 | 13.88 | 1.00 |
| kv260 | 8 | DMA command pipelining | 350 | 1.05 | 13.63 | 1.02 |
| kv260 | 8 | + DMA command pipelining | 350 | 1.05 | 13.63 | 1.02 |
| kv260 | 8 | second DMA master (2 HP ports) | 614 | 1.84 | 13.85 | 1.00 |
| kv260 | 8 | + second DMA master (2 HP ports) | 675 | 2.03 | 13.60 | 1.02 |
| kv260 | 8 | on-chip / prefetched command ring | 333 | 1.00 | 13.88 | 1.00 |
| kv260 | 8 | + on-chip / prefetched command ring | 675 | 2.03 | 13.60 | 1.02 |
| kv260 | 8 | SFU second row buffer | 333 | 1.00 | 13.69 | 1.01 |
| kv260 | 8 | + SFU second row buffer | 675 | 2.03 | 13.41 | 1.03 |
| kv260 | 8 | resident SFU parameter slots | 333 | 1.00 | 13.86 | 1.00 |
| kv260 | 8 | + resident SFU parameter slots | 676 | 2.03 | 13.40 | 1.04 |
| kv260 | 8 | 2-row array weight port | 338 | 1.01 | 13.88 | 1.00 |
| kv260 | 8 | + 2-row array weight port | 696 | 2.09 | 13.40 | 1.04 |
| kv260 | 4 | as built | 597 | 1.00 | 14.15 | 1.00 |
| kv260 | 4 | coalesce contiguous DMA rows | 597 | 1.00 | 14.15 | 1.00 |
| kv260 | 4 | + coalesce contiguous DMA rows | 597 | 1.00 | 14.15 | 1.00 |
| kv260 | 4 | DMA command pipelining | 626 | 1.05 | 13.96 | 1.01 |
| kv260 | 4 | + DMA command pipelining | 626 | 1.05 | 13.96 | 1.01 |
| kv260 | 4 | second DMA master (2 HP ports) | 649 | 1.09 | 14.12 | 1.00 |
| kv260 | 4 | + second DMA master (2 HP ports) | 674 | 1.13 | 13.93 | 1.02 |
| kv260 | 4 | on-chip / prefetched command ring | 597 | 1.00 | 14.15 | 1.00 |
| kv260 | 4 | + on-chip / prefetched command ring | 674 | 1.13 | 13.93 | 1.02 |
| kv260 | 4 | SFU second row buffer | 597 | 1.00 | 13.96 | 1.01 |
| kv260 | 4 | + SFU second row buffer | 674 | 1.13 | 13.74 | 1.03 |
| kv260 | 4 | resident SFU parameter slots | 597 | 1.00 | 14.13 | 1.00 |
| kv260 | 4 | + resident SFU parameter slots | 677 | 1.13 | 13.73 | 1.03 |
| kv260 | 4 | 2-row array weight port | 620 | 1.04 | 13.92 | 1.02 |
| kv260 | 4 | + 2-row array weight port | 1238 | 2.07 | 13.37 | 1.06 |
| pynq-z2 | 8 | as built | 99 | 1.00 | 86.52 | 1.00 |
| pynq-z2 | 8 | coalesce contiguous DMA rows | 99 | 1.00 | 86.52 | 1.00 |
| pynq-z2 | 8 | + coalesce contiguous DMA rows | 99 | 1.00 | 86.52 | 1.00 |
| pynq-z2 | 8 | DMA command pipelining | 103 | 1.04 | 84.26 | 1.03 |
| pynq-z2 | 8 | + DMA command pipelining | 103 | 1.04 | 84.26 | 1.03 |
| pynq-z2 | 8 | second DMA master (2 HP ports) | 122 | 1.24 | 85.92 | 1.01 |
| pynq-z2 | 8 | + second DMA master (2 HP ports) | 128 | 1.30 | 83.98 | 1.03 |
| pynq-z2 | 8 | on-chip / prefetched command ring | 99 | 1.00 | 86.51 | 1.00 |
| pynq-z2 | 8 | + on-chip / prefetched command ring | 128 | 1.30 | 83.98 | 1.03 |
| pynq-z2 | 8 | SFU second row buffer | 99 | 1.00 | 85.82 | 1.01 |
| pynq-z2 | 8 | + SFU second row buffer | 128 | 1.30 | 82.98 | 1.04 |
| pynq-z2 | 8 | resident SFU parameter slots | 99 | 1.00 | 86.44 | 1.00 |
| pynq-z2 | 8 | + resident SFU parameter slots | 128 | 1.30 | 82.88 | 1.04 |
| pynq-z2 | 8 | 2-row array weight port | 99 | 1.00 | 86.09 | 1.01 |
| pynq-z2 | 8 | + 2-row array weight port | 129 | 1.30 | 82.42 | 1.05 |
| pynq-z2 | 4 | as built | 176 | 1.00 | 84.50 | 1.00 |
| pynq-z2 | 4 | coalesce contiguous DMA rows | 176 | 1.00 | 84.50 | 1.00 |
| pynq-z2 | 4 | + coalesce contiguous DMA rows | 176 | 1.00 | 84.50 | 1.00 |
| pynq-z2 | 4 | DMA command pipelining | 182 | 1.04 | 83.64 | 1.01 |
| pynq-z2 | 4 | + DMA command pipelining | 182 | 1.04 | 83.64 | 1.01 |
| pynq-z2 | 4 | second DMA master (2 HP ports) | 191 | 1.09 | 84.31 | 1.00 |
| pynq-z2 | 4 | + second DMA master (2 HP ports) | 196 | 1.11 | 83.46 | 1.01 |
| pynq-z2 | 4 | on-chip / prefetched command ring | 176 | 1.00 | 84.49 | 1.00 |
| pynq-z2 | 4 | + on-chip / prefetched command ring | 196 | 1.11 | 83.46 | 1.01 |
| pynq-z2 | 4 | SFU second row buffer | 176 | 1.00 | 83.50 | 1.01 |
| pynq-z2 | 4 | + SFU second row buffer | 196 | 1.11 | 82.45 | 1.02 |
| pynq-z2 | 4 | resident SFU parameter slots | 176 | 1.00 | 84.40 | 1.00 |
| pynq-z2 | 4 | + resident SFU parameter slots | 196 | 1.11 | 82.36 | 1.03 |
| pynq-z2 | 4 | 2-row array weight port | 178 | 1.01 | 84.04 | 1.01 |
| pynq-z2 | 4 | + 2-row array weight port | 230 | 1.30 | 81.89 | 1.03 |
