| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |
|---|---|---|---|---|---|---|---|---|---|
| kv260 | tiny | 8 | onchip | 1 | 0.99 (128, chunk 128) | 0.47 | 3704 | 10066 | 0.41 |
| kv260 | tiny | 8 | onchip | 4 |  |  | 11613 | 40264 | 0.32 |
| kv260 | tiny | 8 | onchip | 8 |  |  | 18029 | 80528 | 0.25 |
| kv260 | tiny | 8 | ddr | 1 | 1.29 (128, chunk 128) | 0.36 | 3495 | 10066 | 0.41 |
| kv260 | tiny | 8 | ddr | 4 |  |  | 10043 | 40264 | 0.36 |
| kv260 | tiny | 8 | ddr | 8 |  |  | 14603 | 80528 | 0.32 |
| kv260 | tiny | 4 | onchip | 1 | 1.02 (128, chunk 128) | 0.45 | 5119 | 19923 | 0.32 |
| kv260 | tiny | 4 | onchip | 4 |  |  | 14756 | 79694 | 0.23 |
| kv260 | tiny | 4 | onchip | 8 |  |  | 21296 | 159388 | 0.17 |
| kv260 | tiny | 4 | ddr | 1 | 1.33 (128, chunk 128) | 0.35 | 4619 | 19923 | 0.33 |
| kv260 | tiny | 4 | ddr | 4 |  |  | 12124 | 79694 | 0.29 |
| kv260 | tiny | 4 | ddr | 8 |  |  | 16503 | 159388 | 0.26 |
| kv260 | shakespeare | 8 | onchip | 1 | 14.73 (256, chunk 128) | 0.78 | 332 | 752 | 0.46 |
| kv260 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | ddr | 1 | 15.80 (256, chunk 128) | 0.73 | 322 | 752 | 0.47 |
| kv260 | shakespeare | 8 | ddr | 4 |  |  | 1083 | 3007 | 0.45 |
| kv260 | shakespeare | 8 | ddr | 8 |  |  | 1785 | 6014 | 0.44 |
| kv260 | shakespeare | 4 | onchip | 1 | 14.75 (256, chunk 128) | 0.78 | 594 | 1500 | 0.44 |
| kv260 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | ddr | 1 | 16.03 (256, chunk 128) | 0.72 | 564 | 1500 | 0.46 |
| kv260 | shakespeare | 4 | ddr | 4 |  |  | 1691 | 6000 | 0.43 |
| kv260 | shakespeare | 4 | ddr | 8 |  |  | 2534 | 12000 | 0.41 |
| pynq-z2 | tiny | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | ddr | 1 | 4.80 (128, chunk 128) | 0.64 | 1136 | 1887 | 0.72 |
| pynq-z2 | tiny | 8 | ddr | 4 |  |  | 3543 | 7550 | 0.68 |
| pynq-z2 | tiny | 8 | ddr | 8 |  |  | 5474 | 15099 | 0.64 |
| pynq-z2 | tiny | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | ddr | 1 | 4.98 (128, chunk 128) | 0.62 | 1690 | 3736 | 0.64 |
| pynq-z2 | tiny | 4 | ddr | 4 |  |  | 4749 | 14943 | 0.60 |
| pynq-z2 | tiny | 4 | ddr | 8 |  |  | 6800 | 29885 | 0.57 |
| pynq-z2 | shakespeare | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | ddr | 1 | 88.14 (256, chunk 32) | 0.86 | 99 | 141 | 0.76 |
| pynq-z2 | shakespeare | 8 | ddr | 4 |  |  | 339 | 564 | 0.76 |
| pynq-z2 | shakespeare | 8 | ddr | 8 |  |  | 570 | 1128 | 0.75 |
| pynq-z2 | shakespeare | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | ddr | 1 | 85.48 (256, chunk 32) | 0.88 | 175 | 281 | 0.75 |
| pynq-z2 | shakespeare | 4 | ddr | 4 |  |  | 540 | 1125 | 0.74 |
| pynq-z2 | shakespeare | 4 | ddr | 8 |  |  | 828 | 2250 | 0.73 |
