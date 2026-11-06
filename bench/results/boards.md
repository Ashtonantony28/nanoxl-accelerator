| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |
|---|---|---|---|---|---|---|---|---|---|
| kv260 | tiny | 8 | onchip | 1 | 0.61 (128, chunk 128) | 0.76 | 8765 | 10066 | 0.96 |
| kv260 | tiny | 8 | onchip | 4 |  |  | 26292 | 40264 | 0.73 |
| kv260 | tiny | 8 | onchip | 8 |  |  | 39805 | 80528 | 0.55 |
| kv260 | tiny | 8 | ddr | 1 | 0.61 (128, chunk 128) | 0.76 | 7728 | 10066 | 0.91 |
| kv260 | tiny | 8 | ddr | 4 |  |  | 24330 | 40264 | 0.87 |
| kv260 | tiny | 8 | ddr | 8 |  |  | 37390 | 80528 | 0.81 |
| kv260 | tiny | 4 | onchip | 1 | 0.61 (128, chunk 128) | 0.76 | 14993 | 20132 | 0.91 |
| kv260 | tiny | 4 | onchip | 4 |  |  | 38222 | 80528 | 0.59 |
| kv260 | tiny | 4 | onchip | 8 |  |  | 52029 | 161057 | 0.40 |
| kv260 | tiny | 4 | ddr | 1 | 0.62 (128, chunk 128) | 0.74 | 12150 | 20132 | 0.84 |
| kv260 | tiny | 4 | ddr | 4 |  |  | 34038 | 80528 | 0.79 |
| kv260 | tiny | 4 | ddr | 8 |  |  | 47805 | 161057 | 0.73 |
| kv260 | shakespeare | 8 | onchip | 1 | 12.38 (256, chunk 128) | 0.93 | 712 | 752 | 0.98 |
| kv260 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | ddr | 1 | 12.38 (256, chunk 128) | 0.93 | 675 | 752 | 0.98 |
| kv260 | shakespeare | 8 | ddr | 4 |  |  | 2310 | 3007 | 0.97 |
| kv260 | shakespeare | 8 | ddr | 8 |  |  | 3873 | 6014 | 0.95 |
| kv260 | shakespeare | 4 | onchip | 1 | 12.34 (256, chunk 128) | 0.93 | 1360 | 1504 | 0.97 |
| kv260 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | ddr | 1 | 12.35 (256, chunk 128) | 0.93 | 1231 | 1504 | 0.97 |
| kv260 | shakespeare | 4 | ddr | 4 |  |  | 3765 | 6014 | 0.95 |
| kv260 | shakespeare | 4 | ddr | 8 |  |  | 5730 | 12028 | 0.92 |
| pynq-z2 | tiny | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | ddr | 1 | 3.68 (128, chunk 128) | 0.80 | 1488 | 1887 | 0.94 |
| pynq-z2 | tiny | 8 | ddr | 4 |  |  | 4845 | 7550 | 0.92 |
| pynq-z2 | tiny | 8 | ddr | 8 |  |  | 7753 | 15099 | 0.90 |
| pynq-z2 | tiny | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | ddr | 1 | 3.67 (128, chunk 128) | 0.80 | 2496 | 3775 | 0.92 |
| pynq-z2 | tiny | 4 | ddr | 4 |  |  | 7220 | 15099 | 0.90 |
| pynq-z2 | tiny | 4 | ddr | 8 |  |  | 10520 | 30198 | 0.87 |
| pynq-z2 | shakespeare | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | ddr | 1 | 82.26 (256, chunk 32) | 0.92 | 128 | 141 | 0.99 |
| pynq-z2 | shakespeare | 8 | ddr | 4 |  |  | 440 | 564 | 0.98 |
| pynq-z2 | shakespeare | 8 | ddr | 8 |  |  | 744 | 1128 | 0.98 |
| pynq-z2 | shakespeare | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | ddr | 1 | 81.91 (256, chunk 32) | 0.92 | 234 | 282 | 0.98 |
| pynq-z2 | shakespeare | 4 | ddr | 4 |  |  | 724 | 1128 | 0.97 |
| pynq-z2 | shakespeare | 4 | ddr | 8 |  |  | 1113 | 2255 | 0.96 |
