# Multi-seed results - Phase 2 architecture comparison

Pooled from `results/{mobilenetv2_*_gpu_seed*, efficientnetb0_*_gpu_seed*, fastvit_*_gpu_k3_seed*}`. Each cell is mean +/- sample std over repeated runs of an identical recipe, varying only `--seed`.

Seeds per architecture: **EfficientNetB0** n=3 (seeds 42, 43, 44); **FastViT_T8** n=3 (seeds 42, 43, 44); **MobileNetV2** n=3 (seeds 42, 43, 44)

Compute backend: GPU (NVIDIA GeForce RTX 4050 Laptop GPU).

| Metric | EfficientNetB0 | FastViT_T8 | MobileNetV2 |
|---|---|---|---|
| Test accuracy (%) | 76.45 +/- 0.90 | 78.72 +/- 0.22 | 73.61 +/- 0.59 |
| ROC-AUC (OvR) | 0.9352 +/- 0.0027 | 0.9425 +/- 0.0006 | 0.9145 +/- 0.0023 |
| Macro F1 | 0.7620 +/- 0.0121 | 0.7860 +/- 0.0026 | 0.7383 +/- 0.0093 |
| Recall - unripe | 0.9270 +/- 0.0053 | 0.8827 +/- 0.0170 | 0.8413 +/- 0.0169 |
| Recall - underripe | 0.4937 +/- 0.0715 | 0.5093 +/- 0.0097 | 0.4823 +/- 0.0588 |
| Recall - ripe | 0.7580 +/- 0.0446 | 0.8630 +/- 0.0173 | 0.7800 +/- 0.0181 |
| Recall - overripe | 0.9127 +/- 0.0081 | 0.9120 +/- 0.0125 | 0.8637 +/- 0.0313 |
| Lighting spread (pts) | 3.06 +/- 0.55 | 1.47 +/- 0.53 | 2.98 +/- 1.28 |
| Params (M) | 4.05 +/- 0.00 | 3.20 +/- 0.00 | 2.26 +/- 0.00 |
| Model size (MB, pre-quant) | 33.9 +/- 0.0 | 30.2 +/- 0.0 | 20.9 +/- 0.0 |
| Inference (ms/img, dev) | 160.3 +/- 2.7 | 150.5 +/- 2.6 | 143.4 +/- 3.4 |

## Is the architecture gap real?

- 3 architecture(s) pooled - the pairwise verdict is emitted only when exactly 2 are present.
