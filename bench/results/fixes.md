| board | W | fix | decode tok/s | x as built | prefill ms (T=256) | x as built |
|---|---|---|---|---|---|---|
| kv260 | 8 | as built | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | coalesce contiguous DMA rows | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | + coalesce contiguous DMA rows | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | on-chip / prefetched command ring | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | + on-chip / prefetched command ring | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | 4-line (64 B) VEC engine ports | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | + 4-line (64 B) VEC engine ports | 619 | 1.00 | 14.01 | 1.00 |
| kv260 | 8 | SFU second row buffer | 619 | 1.00 | 13.64 | 1.03 |
| kv260 | 8 | + SFU second row buffer | 619 | 1.00 | 13.64 | 1.03 |
| kv260 | 8 | resident SFU parameter slots | 620 | 1.00 | 13.99 | 1.00 |
| kv260 | 8 | + resident SFU parameter slots | 620 | 1.00 | 13.62 | 1.03 |
| kv260 | 8 | 2-row array weight port | 664 | 1.07 | 14.01 | 1.00 |
| kv260 | 8 | + 2-row array weight port | 664 | 1.07 | 13.62 | 1.03 |
| kv260 | 4 | as built | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | coalesce contiguous DMA rows | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | + coalesce contiguous DMA rows | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | on-chip / prefetched command ring | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | + on-chip / prefetched command ring | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | 4-line (64 B) VEC engine ports | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | + 4-line (64 B) VEC engine ports | 651 | 1.00 | 14.31 | 1.00 |
| kv260 | 4 | SFU second row buffer | 651 | 1.00 | 13.94 | 1.03 |
| kv260 | 4 | + SFU second row buffer | 651 | 1.00 | 13.94 | 1.03 |
| kv260 | 4 | resident SFU parameter slots | 654 | 1.00 | 14.29 | 1.00 |
| kv260 | 4 | + resident SFU parameter slots | 654 | 1.01 | 13.92 | 1.03 |
| kv260 | 4 | 2-row array weight port | 1104 | 1.70 | 13.92 | 1.03 |
| kv260 | 4 | + 2-row array weight port | 1104 | 1.70 | 13.54 | 1.06 |
| pynq-z2 | 8 | as built | 128 | 1.00 | 84.94 | 1.00 |
| pynq-z2 | 8 | coalesce contiguous DMA rows | 128 | 1.00 | 84.94 | 1.00 |
| pynq-z2 | 8 | + coalesce contiguous DMA rows | 128 | 1.00 | 84.94 | 1.00 |
| pynq-z2 | 8 | on-chip / prefetched command ring | 128 | 1.00 | 84.93 | 1.00 |
| pynq-z2 | 8 | + on-chip / prefetched command ring | 128 | 1.00 | 84.93 | 1.00 |
| pynq-z2 | 8 | 4-line (64 B) VEC engine ports | 128 | 1.00 | 84.94 | 1.00 |
| pynq-z2 | 8 | + 4-line (64 B) VEC engine ports | 128 | 1.00 | 84.93 | 1.00 |
| pynq-z2 | 8 | SFU second row buffer | 128 | 1.00 | 83.90 | 1.01 |
| pynq-z2 | 8 | + SFU second row buffer | 128 | 1.00 | 83.89 | 1.01 |
| pynq-z2 | 8 | resident SFU parameter slots | 128 | 1.00 | 84.83 | 1.00 |
| pynq-z2 | 8 | + resident SFU parameter slots | 128 | 1.00 | 83.78 | 1.01 |
| pynq-z2 | 8 | 2-row array weight port | 128 | 1.00 | 84.48 | 1.01 |
| pynq-z2 | 8 | + 2-row array weight port | 128 | 1.00 | 83.32 | 1.02 |
| pynq-z2 | 4 | as built | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | coalesce contiguous DMA rows | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | + coalesce contiguous DMA rows | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | on-chip / prefetched command ring | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | + on-chip / prefetched command ring | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | 4-line (64 B) VEC engine ports | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | + 4-line (64 B) VEC engine ports | 186 | 1.00 | 83.98 | 1.00 |
| pynq-z2 | 4 | SFU second row buffer | 186 | 1.00 | 82.96 | 1.01 |
| pynq-z2 | 4 | + SFU second row buffer | 186 | 1.00 | 82.96 | 1.01 |
| pynq-z2 | 4 | resident SFU parameter slots | 186 | 1.00 | 83.87 | 1.00 |
| pynq-z2 | 4 | + resident SFU parameter slots | 186 | 1.00 | 82.86 | 1.01 |
| pynq-z2 | 4 | 2-row array weight port | 225 | 1.21 | 83.54 | 1.01 |
| pynq-z2 | 4 | + 2-row array weight port | 225 | 1.21 | 82.41 | 1.02 |
