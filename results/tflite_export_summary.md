# TFLite export — Phase 3 conversion spike

Answers one question: does each model convert to a **self-contained** TFLite graph, or
does it need the Flex delegate (`SELECT_TF_OPS`)? A Flex-only conversion is a **partial
failure** — it will not run in a stock Android interpreter without shipping a slice of the
full TF runtime, which adds tens of MB against the <=100MB NFR.

Fidelity measured on 200 randomly sampled test images (seed 42); latency on 50 of them, single-image, 1-thread dev CPU.
**Latency here is a proxy, not the <=2s NFR verdict** — that still needs a physical device.

| Run | Mode | Converted | Builtin-only | Size (MB) | <=100MB | TFLite acc | Δ vs Keras | Agreement | Latency (ms) |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42 | float32 | yes | yes | 12.1 | yes | 0.9200 | +0.0050 | 0.990 | 47.7 |
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42 | float16 | yes | yes | 6.1 | yes | 0.9250 | +0.0100 | 0.970 | 47.2 |
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42 | dynamic_int8 | yes | yes | 3.3 | yes | 0.9300 | +0.0150 | 0.970 | 114.3 |
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42 | full_int8 | yes | yes | 3.5 | yes | 0.2600 | -0.6550 | 0.265 | 136.7 |
| efficientnetb0_fyp2_blockft_v2_gpu_f50_k3_seed42 | float32 | yes | yes | 15.3 | yes | 0.9100 | +0.0000 | 1.000 | 22.4 |
| efficientnetb0_fyp2_blockft_v2_gpu_f50_k3_seed42 | float16 | yes | yes | 7.7 | yes | 0.9100 | +0.0000 | 1.000 | 23.6 |
| efficientnetb0_fyp2_blockft_v2_gpu_f50_k3_seed42 | dynamic_int8 | yes | yes | 4.3 | yes | 0.9100 | +0.0000 | 0.980 | 30.3 |
| efficientnetb0_fyp2_blockft_v2_gpu_f50_k3_seed42 | full_int8 | yes | yes | 4.7 | yes | 0.6450 | -0.2650 | 0.660 | 20.3 |
| mobilenetv2_fyp2_blockft_v2_gpu_f50_k3_seed42 | float32 | yes | yes | 8.5 | yes | 0.9100 | +0.0050 | 0.995 | 12.1 |
| mobilenetv2_fyp2_blockft_v2_gpu_f50_k3_seed42 | float16 | yes | yes | 4.3 | yes | 0.9050 | +0.0000 | 1.000 | 12.0 |
| mobilenetv2_fyp2_blockft_v2_gpu_f50_k3_seed42 | dynamic_int8 | yes | yes | 2.4 | yes | 0.9200 | +0.0150 | 0.975 | 16.9 |
| mobilenetv2_fyp2_blockft_v2_gpu_f50_k3_seed42 | full_int8 | yes | yes | 2.6 | yes | 0.9150 | +0.0100 | 0.960 | 13.4 |

All requested modes converted builtin-only. No Flex delegate needed.

