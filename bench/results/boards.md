| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |
|---|---|---|---|---|---|---|---|---|---|
| kv260 | tiny | 8 | onchip | 1 | 0.67 (128, chunk 128) | 0.70 | 8765 | 10066 | 0.96 |
| kv260 | tiny | 8 | onchip | 4 |  |  | 26172 | 40264 | 0.72 |
| kv260 | tiny | 8 | onchip | 8 |  |  | 39455 | 80528 | 0.55 |
| kv260 | tiny | 8 | ddr | 1 | 0.67 (128, chunk 128) | 0.69 | 7728 | 10066 | 0.91 |
| kv260 | tiny | 8 | ddr | 4 |  |  | 24287 | 40264 | 0.87 |
| kv260 | tiny | 8 | ddr | 8 |  |  | 37169 | 80528 | 0.81 |
| kv260 | tiny | 4 | onchip | 1 | 0.67 (128, chunk 128) | 0.69 | 14731 | 19923 | 0.92 |
| kv260 | tiny | 4 | onchip | 4 |  |  | 37539 | 79694 | 0.59 |
| kv260 | tiny | 4 | onchip | 8 |  |  | 51041 | 159388 | 0.41 |
| kv260 | tiny | 4 | ddr | 1 | 0.68 (128, chunk 128) | 0.68 | 11979 | 19923 | 0.85 |
| kv260 | tiny | 4 | ddr | 4 |  |  | 33624 | 79694 | 0.79 |
| kv260 | tiny | 4 | ddr | 8 |  |  | 47126 | 159388 | 0.73 |
| kv260 | shakespeare | 8 | onchip | 1 | 12.75 (256, chunk 128) | 0.90 | 712 | 752 | 0.98 |
| kv260 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | ddr | 1 | 12.75 (256, chunk 128) | 0.90 | 676 | 752 | 0.98 |
| kv260 | shakespeare | 8 | ddr | 4 |  |  | 2311 | 3007 | 0.97 |
| kv260 | shakespeare | 8 | ddr | 8 |  |  | 3873 | 6014 | 0.95 |
| kv260 | shakespeare | 4 | onchip | 1 | 12.72 (256, chunk 128) | 0.91 | 1320 | 1500 | 0.97 |
| kv260 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | ddr | 1 | 12.73 (256, chunk 128) | 0.91 | 1201 | 1500 | 0.97 |
| kv260 | shakespeare | 4 | ddr | 4 |  |  | 3693 | 6000 | 0.95 |
| kv260 | shakespeare | 4 | ddr | 8 |  |  | 5643 | 12000 | 0.92 |
| pynq-z2 | tiny | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | ddr | 1 | 4.02 (128, chunk 128) | 0.77 | 1488 | 1887 | 0.94 |
| pynq-z2 | tiny | 8 | ddr | 4 |  |  | 4843 | 7550 | 0.92 |
| pynq-z2 | tiny | 8 | ddr | 8 |  |  | 7745 | 15099 | 0.90 |
| pynq-z2 | tiny | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | ddr | 1 | 4.10 (128, chunk 128) | 0.75 | 2460 | 3736 | 0.93 |
| pynq-z2 | tiny | 4 | ddr | 4 |  |  | 7147 | 14943 | 0.91 |
| pynq-z2 | tiny | 4 | ddr | 8 |  |  | 10447 | 29885 | 0.88 |
| pynq-z2 | shakespeare | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | ddr | 1 | 82.71 (256, chunk 32) | 0.91 | 128 | 141 | 0.99 |
| pynq-z2 | shakespeare | 8 | ddr | 4 |  |  | 441 | 564 | 0.98 |
| pynq-z2 | shakespeare | 8 | ddr | 8 |  |  | 745 | 1128 | 0.98 |
| pynq-z2 | shakespeare | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | ddr | 1 | 82.27 (256, chunk 32) | 0.92 | 229 | 281 | 0.99 |
| pynq-z2 | shakespeare | 4 | ddr | 4 |  |  | 711 | 1125 | 0.98 |
| pynq-z2 | shakespeare | 4 | ddr | 8 |  |  | 1097 | 2250 | 0.96 |
