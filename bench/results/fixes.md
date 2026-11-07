| board | W | fix | decode tok/s | x as built | prefill ms (T=256) | x as built |
|---|---|---|---|---|---|---|
| kv260 | 8 | as built | 332 | 1.00 | 14.73 | 1.00 |
| kv260 | 8 | coalesce contiguous DMA rows | 332 | 1.00 | 14.73 | 1.00 |
| kv260 | 8 | + coalesce contiguous DMA rows | 332 | 1.00 | 14.73 | 1.00 |
| kv260 | 8 | DMA command pipelining | 349 | 1.05 | 14.01 | 1.05 |
| kv260 | 8 | + DMA command pipelining | 349 | 1.05 | 14.01 | 1.05 |
| kv260 | 8 | second DMA master (2 HP ports) | 611 | 1.84 | 14.71 | 1.00 |
| kv260 | 8 | + second DMA master (2 HP ports) | 672 | 2.02 | 13.98 | 1.05 |
| kv260 | 8 | on-chip / prefetched command ring | 332 | 1.00 | 14.73 | 1.00 |
| kv260 | 8 | + on-chip / prefetched command ring | 672 | 2.02 | 13.98 | 1.05 |
| kv260 | 8 | 4-line (64 B) VEC engine ports | 332 | 1.00 | 14.73 | 1.00 |
| kv260 | 8 | + 4-line (64 B) VEC engine ports | 672 | 2.02 | 13.98 | 1.05 |
| kv260 | 8 | SFU second row buffer | 332 | 1.00 | 14.37 | 1.03 |
| kv260 | 8 | + SFU second row buffer | 672 | 2.02 | 13.61 | 1.08 |
| kv260 | 8 | resident SFU parameter slots | 332 | 1.00 | 14.72 | 1.00 |
| kv260 | 8 | + resident SFU parameter slots | 672 | 2.02 | 13.60 | 1.08 |
| kv260 | 8 | 2-row array weight port | 337 | 1.01 | 14.73 | 1.00 |
| kv260 | 8 | + 2-row array weight port | 693 | 2.09 | 13.60 | 1.08 |
| kv260 | 4 | as built | 594 | 1.00 | 14.75 | 1.00 |
| kv260 | 4 | coalesce contiguous DMA rows | 594 | 1.00 | 14.75 | 1.00 |
| kv260 | 4 | + coalesce contiguous DMA rows | 594 | 1.00 | 14.75 | 1.00 |
| kv260 | 4 | DMA command pipelining | 623 | 1.05 | 14.31 | 1.03 |
| kv260 | 4 | + DMA command pipelining | 623 | 1.05 | 14.31 | 1.03 |
| kv260 | 4 | second DMA master (2 HP ports) | 632 | 1.06 | 14.72 | 1.00 |
| kv260 | 4 | + second DMA master (2 HP ports) | 658 | 1.11 | 14.28 | 1.03 |
| kv260 | 4 | on-chip / prefetched command ring | 594 | 1.00 | 14.74 | 1.00 |
| kv260 | 4 | + on-chip / prefetched command ring | 658 | 1.11 | 14.28 | 1.03 |
| kv260 | 4 | 4-line (64 B) VEC engine ports | 594 | 1.00 | 14.75 | 1.00 |
| kv260 | 4 | + 4-line (64 B) VEC engine ports | 658 | 1.11 | 14.28 | 1.03 |
| kv260 | 4 | SFU second row buffer | 594 | 1.00 | 14.38 | 1.03 |
| kv260 | 4 | + SFU second row buffer | 658 | 1.11 | 13.91 | 1.06 |
| kv260 | 4 | resident SFU parameter slots | 594 | 1.00 | 14.73 | 1.00 |
| kv260 | 4 | + resident SFU parameter slots | 662 | 1.11 | 13.90 | 1.06 |
| kv260 | 4 | 2-row array weight port | 616 | 1.04 | 14.46 | 1.02 |
| kv260 | 4 | + 2-row array weight port | 1213 | 2.04 | 13.51 | 1.09 |
| pynq-z2 | 8 | as built | 99 | 1.00 | 88.14 | 1.00 |
| pynq-z2 | 8 | coalesce contiguous DMA rows | 99 | 1.00 | 88.14 | 1.00 |
| pynq-z2 | 8 | + coalesce contiguous DMA rows | 99 | 1.00 | 88.14 | 1.00 |
| pynq-z2 | 8 | DMA command pipelining | 103 | 1.04 | 84.44 | 1.04 |
| pynq-z2 | 8 | + DMA command pipelining | 103 | 1.04 | 84.44 | 1.04 |
| pynq-z2 | 8 | second DMA master (2 HP ports) | 122 | 1.24 | 87.91 | 1.00 |
| pynq-z2 | 8 | + second DMA master (2 HP ports) | 128 | 1.30 | 84.16 | 1.05 |
| pynq-z2 | 8 | on-chip / prefetched command ring | 99 | 1.00 | 88.14 | 1.00 |
| pynq-z2 | 8 | + on-chip / prefetched command ring | 128 | 1.30 | 84.16 | 1.05 |
| pynq-z2 | 8 | 4-line (64 B) VEC engine ports | 99 | 1.00 | 88.14 | 1.00 |
| pynq-z2 | 8 | + 4-line (64 B) VEC engine ports | 128 | 1.30 | 84.16 | 1.05 |
| pynq-z2 | 8 | SFU second row buffer | 99 | 1.00 | 87.14 | 1.01 |
| pynq-z2 | 8 | + SFU second row buffer | 128 | 1.30 | 83.15 | 1.06 |
| pynq-z2 | 8 | resident SFU parameter slots | 99 | 1.00 | 88.04 | 1.00 |
| pynq-z2 | 8 | + resident SFU parameter slots | 128 | 1.30 | 83.05 | 1.06 |
| pynq-z2 | 8 | 2-row array weight port | 99 | 1.00 | 87.68 | 1.01 |
| pynq-z2 | 8 | + 2-row array weight port | 129 | 1.30 | 82.59 | 1.07 |
| pynq-z2 | 4 | as built | 175 | 1.00 | 85.48 | 1.00 |
| pynq-z2 | 4 | coalesce contiguous DMA rows | 175 | 1.00 | 85.48 | 1.00 |
| pynq-z2 | 4 | + coalesce contiguous DMA rows | 175 | 1.00 | 85.48 | 1.00 |
| pynq-z2 | 4 | DMA command pipelining | 180 | 1.03 | 83.84 | 1.02 |
| pynq-z2 | 4 | + DMA command pipelining | 180 | 1.03 | 83.84 | 1.02 |
| pynq-z2 | 4 | second DMA master (2 HP ports) | 184 | 1.05 | 85.30 | 1.00 |
| pynq-z2 | 4 | + second DMA master (2 HP ports) | 189 | 1.08 | 83.66 | 1.02 |
| pynq-z2 | 4 | on-chip / prefetched command ring | 175 | 1.00 | 85.48 | 1.00 |
| pynq-z2 | 4 | + on-chip / prefetched command ring | 189 | 1.08 | 83.66 | 1.02 |
| pynq-z2 | 4 | 4-line (64 B) VEC engine ports | 175 | 1.00 | 85.48 | 1.00 |
| pynq-z2 | 4 | + 4-line (64 B) VEC engine ports | 189 | 1.08 | 83.66 | 1.02 |
| pynq-z2 | 4 | SFU second row buffer | 175 | 1.00 | 84.48 | 1.01 |
| pynq-z2 | 4 | + SFU second row buffer | 189 | 1.08 | 82.65 | 1.03 |
| pynq-z2 | 4 | resident SFU parameter slots | 175 | 1.00 | 85.38 | 1.00 |
| pynq-z2 | 4 | + resident SFU parameter slots | 189 | 1.08 | 82.55 | 1.04 |
| pynq-z2 | 4 | 2-row array weight port | 178 | 1.02 | 85.02 | 1.01 |
| pynq-z2 | 4 | + 2-row array weight port | 229 | 1.31 | 82.09 | 1.04 |
