| board | model | W | KV | B | prefill ms (T) | prefill util | decode tok/s | weight bound | DDR util |
|---|---|---|---|---|---|---|---|---|---|
| kv260 | tiny | 8 | onchip | 1 | 0.97 (128, chunk 128) | 0.48 | 5317 | 10066 | 0.59 |
| kv260 | tiny | 8 | onchip | 4 |  |  | 15912 | 40264 | 0.44 |
| kv260 | tiny | 8 | onchip | 8 |  |  | 23730 | 80528 | 0.33 |
| kv260 | tiny | 8 | ddr | 1 | 1.27 (128, chunk 128) | 0.37 | 5092 | 10066 | 0.60 |
| kv260 | tiny | 8 | ddr | 4 |  |  | 14380 | 40264 | 0.51 |
| kv260 | tiny | 8 | ddr | 8 |  |  | 20650 | 80528 | 0.45 |
| kv260 | tiny | 4 | onchip | 1 | 1.00 (128, chunk 128) | 0.46 | 5818 | 19923 | 0.36 |
| kv260 | tiny | 4 | onchip | 4 |  |  | 16629 | 79694 | 0.26 |
| kv260 | tiny | 4 | onchip | 8 |  |  | 23915 | 159388 | 0.19 |
| kv260 | tiny | 4 | ddr | 1 | 1.31 (128, chunk 128) | 0.35 | 5391 | 19923 | 0.38 |
| kv260 | tiny | 4 | ddr | 4 |  |  | 14745 | 79694 | 0.35 |
| kv260 | tiny | 4 | ddr | 8 |  |  | 20741 | 159388 | 0.33 |
| kv260 | shakespeare | 8 | onchip | 1 | 14.01 (256, chunk 128) | 0.82 | 619 | 752 | 0.85 |
| kv260 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 8 | ddr | 1 | 15.09 (256, chunk 128) | 0.76 | 607 | 752 | 0.88 |
| kv260 | shakespeare | 8 | ddr | 4 |  |  | 2000 | 3007 | 0.83 |
| kv260 | shakespeare | 8 | ddr | 8 |  |  | 3247 | 6014 | 0.79 |
| kv260 | shakespeare | 4 | onchip | 1 | 14.31 (256, chunk 128) | 0.81 | 651 | 1500 | 0.48 |
| kv260 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| kv260 | shakespeare | 4 | ddr | 1 | 15.60 (256, chunk 128) | 0.74 | 644 | 1500 | 0.52 |
| kv260 | shakespeare | 4 | ddr | 4 |  |  | 2090 | 6000 | 0.53 |
| kv260 | shakespeare | 4 | ddr | 8 |  |  | 3339 | 12000 | 0.54 |
| pynq-z2 | tiny | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 8 | ddr | 1 | 4.73 (128, chunk 128) | 0.65 | 1329 | 1887 | 0.84 |
| pynq-z2 | tiny | 8 | ddr | 4 |  |  | 4252 | 7550 | 0.81 |
| pynq-z2 | tiny | 8 | ddr | 8 |  |  | 6662 | 15099 | 0.77 |
| pynq-z2 | tiny | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | tiny | 4 | ddr | 1 | 4.90 (128, chunk 128) | 0.63 | 1768 | 3736 | 0.67 |
| pynq-z2 | tiny | 4 | ddr | 4 |  |  | 5191 | 14943 | 0.65 |
| pynq-z2 | tiny | 4 | ddr | 8 |  |  | 7773 | 29885 | 0.64 |
| pynq-z2 | shakespeare | 8 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 8 | ddr | 1 | 84.94 (256, chunk 32) | 0.89 | 128 | 141 | 0.99 |
| pynq-z2 | shakespeare | 8 | ddr | 4 |  |  | 437 | 564 | 0.97 |
| pynq-z2 | shakespeare | 8 | ddr | 8 |  |  | 730 | 1128 | 0.96 |
| pynq-z2 | shakespeare | 4 | onchip | 1 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 4 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | onchip | 8 | KV does not fit | | | | |
| pynq-z2 | shakespeare | 4 | ddr | 1 | 83.98 (256, chunk 32) | 0.90 | 186 | 281 | 0.80 |
| pynq-z2 | shakespeare | 4 | ddr | 4 |  |  | 594 | 1125 | 0.81 |
| pynq-z2 | shakespeare | 4 | ddr | 8 |  |  | 938 | 2250 | 0.81 |
