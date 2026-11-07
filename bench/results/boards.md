| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |
|---|---|---|---|---|---|---|---|---|---|
| kv260 | tiny | 8 | onchip | 1 | 0.91 (128, chunk 128) | 0.51 | 3780 | 10066 | 0.42 |
| kv260 | tiny | 8 | onchip | 4 |  |  | 12275 | 40264 | 0.34 |
| kv260 | tiny | 8 | onchip | 8 |  |  | 19624 | 80528 | 0.27 |
| kv260 | tiny | 8 | ddr | 1 | 1.20 (128, chunk 128) | 0.38 | 3535 | 10066 | 0.42 |
| kv260 | tiny | 8 | ddr | 4 |  |  | 10191 | 40264 | 0.36 |
| kv260 | tiny | 8 | ddr | 8 |  |  | 14850 | 80528 | 0.32 |
| kv260 | tiny | 4 | onchip | 1 | 0.94 (128, chunk 128) | 0.49 | 5338 | 19923 | 0.33 |
| kv260 | tiny | 4 | onchip | 4 |  |  | 16065 | 79694 | 0.25 |
| kv260 | tiny | 4 | onchip | 8 |  |  | 24155 | 159388 | 0.19 |
| kv260 | tiny | 4 | ddr | 1 | 1.25 (128, chunk 128) | 0.37 | 4747 | 19923 | 0.34 |
| kv260 | tiny | 4 | ddr | 4 |  |  | 12476 | 79694 | 0.29 |
| kv260 | tiny | 4 | ddr | 8 |  |  | 17122 | 159388 | 0.27 |
| kv260 | shakespeare | 8 | onchip | 1 | 13.88 (256, chunk 128) | 0.83 | 333 | 752 | 0.46 |
| kv260 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | ddr | 1 | 14.97 (256, chunk 128) | 0.77 | 323 | 752 | 0.47 |
| kv260 | shakespeare | 8 | ddr | 4 |  |  | 1086 | 3007 | 0.45 |
| kv260 | shakespeare | 8 | ddr | 8 |  |  | 1792 | 6014 | 0.44 |
| kv260 | shakespeare | 4 | onchip | 1 | 14.15 (256, chunk 128) | 0.81 | 597 | 1500 | 0.44 |
| kv260 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | ddr | 1 | 15.43 (256, chunk 128) | 0.75 | 565 | 1500 | 0.46 |
| kv260 | shakespeare | 4 | ddr | 4 |  |  | 1700 | 6000 | 0.44 |
| kv260 | shakespeare | 4 | ddr | 8 |  |  | 2553 | 12000 | 0.42 |
| pynq-z2 | tiny | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | ddr | 1 | 4.73 (128, chunk 128) | 0.65 | 1143 | 1887 | 0.72 |
| pynq-z2 | tiny | 8 | ddr | 4 |  |  | 3561 | 7550 | 0.68 |
| pynq-z2 | tiny | 8 | ddr | 8 |  |  | 5500 | 15099 | 0.64 |
| pynq-z2 | tiny | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | ddr | 1 | 4.90 (128, chunk 128) | 0.63 | 1725 | 3736 | 0.65 |
| pynq-z2 | tiny | 4 | ddr | 4 |  |  | 4826 | 14943 | 0.61 |
| pynq-z2 | tiny | 4 | ddr | 8 |  |  | 6884 | 29885 | 0.57 |
| pynq-z2 | shakespeare | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | ddr | 1 | 86.52 (256, chunk 32) | 0.87 | 99 | 141 | 0.77 |
| pynq-z2 | shakespeare | 8 | ddr | 4 |  |  | 339 | 564 | 0.76 |
| pynq-z2 | shakespeare | 8 | ddr | 8 |  |  | 571 | 1128 | 0.75 |
| pynq-z2 | shakespeare | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | ddr | 1 | 84.50 (256, chunk 32) | 0.89 | 176 | 281 | 0.76 |
| pynq-z2 | shakespeare | 4 | ddr | 4 |  |  | 544 | 1125 | 0.75 |
| pynq-z2 | shakespeare | 4 | ddr | 8 |  |  | 835 | 2250 | 0.73 |
