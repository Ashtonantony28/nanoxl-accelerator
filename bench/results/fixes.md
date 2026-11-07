| board | W | fix | decode tok/s | x as built | prefill ms (T=256) | x as built |
|---|---|---|---|---|---|---|
| kv260 | 8 | as built | 331 | 1.00 | 17.61 | 1.00 |
| kv260 | 8 | coalesce contiguous DMA rows | 331 | 1.00 | 17.61 | 1.00 |
| kv260 | 8 | + coalesce contiguous DMA rows | 331 | 1.00 | 17.61 | 1.00 |
| kv260 | 8 | DMA command pipelining | 349 | 1.05 | 16.93 | 1.04 |
| kv260 | 8 | + DMA command pipelining | 349 | 1.05 | 16.93 | 1.04 |
| kv260 | 8 | second DMA master (2 HP ports) | 608 | 1.84 | 17.58 | 1.00 |
| kv260 | 8 | + second DMA master (2 HP ports) | 667 | 2.01 | 16.90 | 1.04 |
| kv260 | 8 | on-chip / prefetched command ring | 331 | 1.00 | 17.61 | 1.00 |
| kv260 | 8 | + on-chip / prefetched command ring | 667 | 2.01 | 16.90 | 1.04 |
| kv260 | 8 | 4-line (64 B) VEC engine ports | 332 | 1.00 | 14.67 | 1.20 |
| kv260 | 8 | + 4-line (64 B) VEC engine ports | 670 | 2.02 | 13.92 | 1.27 |
| kv260 | 8 | SFU second row buffer | 331 | 1.00 | 16.45 | 1.07 |
| kv260 | 8 | + SFU second row buffer | 670 | 2.02 | 13.55 | 1.30 |
| kv260 | 8 | resident SFU parameter slots | 331 | 1.00 | 17.59 | 1.00 |
| kv260 | 8 | + resident SFU parameter slots | 671 | 2.02 | 13.53 | 1.30 |
| kv260 | 8 | 2-row array weight port | 336 | 1.01 | 17.61 | 1.00 |
| kv260 | 8 | + 2-row array weight port | 691 | 2.09 | 13.53 | 1.30 |
| kv260 | 4 | as built | 592 | 1.00 | 17.53 | 1.00 |
| kv260 | 4 | coalesce contiguous DMA rows | 592 | 1.00 | 17.53 | 1.00 |
| kv260 | 4 | + coalesce contiguous DMA rows | 592 | 1.00 | 17.53 | 1.00 |
| kv260 | 4 | DMA command pipelining | 620 | 1.05 | 17.17 | 1.02 |
| kv260 | 4 | + DMA command pipelining | 620 | 1.05 | 17.17 | 1.02 |
| kv260 | 4 | second DMA master (2 HP ports) | 625 | 1.06 | 17.50 | 1.00 |
| kv260 | 4 | + second DMA master (2 HP ports) | 651 | 1.10 | 17.14 | 1.02 |
| kv260 | 4 | on-chip / prefetched command ring | 592 | 1.00 | 17.53 | 1.00 |
| kv260 | 4 | + on-chip / prefetched command ring | 651 | 1.10 | 17.14 | 1.02 |
| kv260 | 4 | 4-line (64 B) VEC engine ports | 593 | 1.00 | 14.68 | 1.19 |
| kv260 | 4 | + 4-line (64 B) VEC engine ports | 657 | 1.11 | 14.22 | 1.23 |
| kv260 | 4 | SFU second row buffer | 592 | 1.00 | 16.37 | 1.07 |
| kv260 | 4 | + SFU second row buffer | 657 | 1.11 | 13.85 | 1.27 |
| kv260 | 4 | resident SFU parameter slots | 592 | 1.00 | 17.51 | 1.00 |
| kv260 | 4 | + resident SFU parameter slots | 660 | 1.12 | 13.83 | 1.27 |
| kv260 | 4 | 2-row array weight port | 614 | 1.04 | 17.31 | 1.01 |
| kv260 | 4 | + 2-row array weight port | 1208 | 2.04 | 13.45 | 1.30 |
| pynq-z2 | 8 | as built | 99 | 1.00 | 90.69 | 1.00 |
| pynq-z2 | 8 | coalesce contiguous DMA rows | 99 | 1.00 | 90.69 | 1.00 |
| pynq-z2 | 8 | + coalesce contiguous DMA rows | 99 | 1.00 | 90.69 | 1.00 |
| pynq-z2 | 8 | DMA command pipelining | 103 | 1.04 | 87.02 | 1.04 |
| pynq-z2 | 8 | + DMA command pipelining | 103 | 1.04 | 87.02 | 1.04 |
| pynq-z2 | 8 | second DMA master (2 HP ports) | 122 | 1.24 | 90.51 | 1.00 |
| pynq-z2 | 8 | + second DMA master (2 HP ports) | 128 | 1.30 | 86.79 | 1.04 |
| pynq-z2 | 8 | on-chip / prefetched command ring | 99 | 1.00 | 90.69 | 1.00 |
| pynq-z2 | 8 | + on-chip / prefetched command ring | 128 | 1.30 | 86.79 | 1.04 |
| pynq-z2 | 8 | 4-line (64 B) VEC engine ports | 99 | 1.00 | 88.16 | 1.03 |
| pynq-z2 | 8 | + 4-line (64 B) VEC engine ports | 128 | 1.30 | 84.17 | 1.08 |
| pynq-z2 | 8 | SFU second row buffer | 99 | 1.00 | 88.11 | 1.03 |
| pynq-z2 | 8 | + SFU second row buffer | 128 | 1.30 | 83.17 | 1.09 |
| pynq-z2 | 8 | resident SFU parameter slots | 99 | 1.00 | 90.59 | 1.00 |
| pynq-z2 | 8 | + resident SFU parameter slots | 128 | 1.30 | 83.07 | 1.09 |
| pynq-z2 | 8 | 2-row array weight port | 99 | 1.00 | 90.23 | 1.01 |
| pynq-z2 | 8 | + 2-row array weight port | 129 | 1.30 | 82.61 | 1.10 |
| pynq-z2 | 4 | as built | 175 | 1.00 | 88.12 | 1.00 |
| pynq-z2 | 4 | coalesce contiguous DMA rows | 175 | 1.00 | 88.12 | 1.00 |
| pynq-z2 | 4 | + coalesce contiguous DMA rows | 175 | 1.00 | 88.12 | 1.00 |
| pynq-z2 | 4 | DMA command pipelining | 180 | 1.03 | 86.52 | 1.02 |
| pynq-z2 | 4 | + DMA command pipelining | 180 | 1.03 | 86.52 | 1.02 |
| pynq-z2 | 4 | second DMA master (2 HP ports) | 184 | 1.05 | 87.96 | 1.00 |
| pynq-z2 | 4 | + second DMA master (2 HP ports) | 189 | 1.08 | 86.37 | 1.02 |
| pynq-z2 | 4 | on-chip / prefetched command ring | 175 | 1.00 | 88.11 | 1.00 |
| pynq-z2 | 4 | + on-chip / prefetched command ring | 189 | 1.08 | 86.36 | 1.02 |
| pynq-z2 | 4 | 4-line (64 B) VEC engine ports | 175 | 1.00 | 85.50 | 1.03 |
| pynq-z2 | 4 | + 4-line (64 B) VEC engine ports | 189 | 1.08 | 83.67 | 1.05 |
| pynq-z2 | 4 | SFU second row buffer | 175 | 1.00 | 85.54 | 1.03 |
| pynq-z2 | 4 | + SFU second row buffer | 189 | 1.08 | 82.67 | 1.07 |
| pynq-z2 | 4 | resident SFU parameter slots | 175 | 1.00 | 88.02 | 1.00 |
| pynq-z2 | 4 | + resident SFU parameter slots | 189 | 1.08 | 82.57 | 1.07 |
| pynq-z2 | 4 | 2-row array weight port | 178 | 1.02 | 87.65 | 1.01 |
| pynq-z2 | 4 | + 2-row array weight port | 229 | 1.31 | 82.11 | 1.07 |
