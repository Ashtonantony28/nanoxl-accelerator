| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |
|---|---|---|---|---|---|---|---|---|---|
| kv260 | tiny | 8 | onchip | 1 | 1.60 (128, chunk 128) | 0.29 | 3651 | 10066 | 0.40 |
| kv260 | tiny | 8 | onchip | 4 |  |  | 11252 | 40264 | 0.31 |
| kv260 | tiny | 8 | onchip | 8 |  |  | 17106 | 80528 | 0.24 |
| kv260 | tiny | 8 | ddr | 1 | 1.86 (128, chunk 128) | 0.25 | 3459 | 10066 | 0.41 |
| kv260 | tiny | 8 | ddr | 4 |  |  | 9903 | 40264 | 0.35 |
| kv260 | tiny | 8 | ddr | 8 |  |  | 14275 | 80528 | 0.31 |
| kv260 | tiny | 4 | onchip | 1 | 1.64 (128, chunk 128) | 0.28 | 4783 | 19923 | 0.30 |
| kv260 | tiny | 4 | onchip | 4 |  |  | 13178 | 79694 | 0.21 |
| kv260 | tiny | 4 | onchip | 8 |  |  | 18628 | 159388 | 0.15 |
| kv260 | tiny | 4 | ddr | 1 | 1.93 (128, chunk 128) | 0.24 | 4361 | 19923 | 0.31 |
| kv260 | tiny | 4 | ddr | 4 |  |  | 11206 | 79694 | 0.26 |
| kv260 | tiny | 4 | ddr | 8 |  |  | 15175 | 159388 | 0.24 |
| kv260 | shakespeare | 8 | onchip | 1 | 17.61 (256, chunk 128) | 0.65 | 331 | 752 | 0.46 |
| kv260 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | ddr | 1 | 18.37 (256, chunk 128) | 0.63 | 322 | 752 | 0.47 |
| kv260 | shakespeare | 8 | ddr | 4 |  |  | 1082 | 3007 | 0.45 |
| kv260 | shakespeare | 8 | ddr | 8 |  |  | 1781 | 6014 | 0.44 |
| kv260 | shakespeare | 4 | onchip | 1 | 17.53 (256, chunk 128) | 0.66 | 592 | 1500 | 0.43 |
| kv260 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | ddr | 1 | 18.54 (256, chunk 128) | 0.62 | 563 | 1500 | 0.45 |
| kv260 | shakespeare | 4 | ddr | 4 |  |  | 1685 | 6000 | 0.43 |
| kv260 | shakespeare | 4 | ddr | 8 |  |  | 2511 | 12000 | 0.41 |
| pynq-z2 | tiny | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | ddr | 1 | 5.24 (128, chunk 128) | 0.59 | 1131 | 1887 | 0.71 |
| pynq-z2 | tiny | 8 | ddr | 4 |  |  | 3522 | 7550 | 0.67 |
| pynq-z2 | tiny | 8 | ddr | 8 |  |  | 5432 | 15099 | 0.63 |
| pynq-z2 | tiny | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | ddr | 1 | 5.39 (128, chunk 128) | 0.57 | 1676 | 3736 | 0.63 |
| pynq-z2 | tiny | 4 | ddr | 4 |  |  | 4696 | 14943 | 0.59 |
| pynq-z2 | tiny | 4 | ddr | 8 |  |  | 6667 | 29885 | 0.55 |
| pynq-z2 | shakespeare | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | ddr | 1 | 90.69 (256, chunk 32) | 0.83 | 99 | 141 | 0.76 |
| pynq-z2 | shakespeare | 8 | ddr | 4 |  |  | 339 | 564 | 0.76 |
| pynq-z2 | shakespeare | 8 | ddr | 8 |  |  | 570 | 1128 | 0.75 |
| pynq-z2 | shakespeare | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | ddr | 1 | 88.12 (256, chunk 32) | 0.86 | 175 | 281 | 0.75 |
| pynq-z2 | shakespeare | 4 | ddr | 4 |  |  | 539 | 1125 | 0.74 |
| pynq-z2 | shakespeare | 4 | ddr | 8 |  |  | 826 | 2250 | 0.72 |
