# Multi-seed results - Phase 2 architecture comparison

Pooled from `results/{*_blockft_v2_gpu_f50_k3_seed*}`. Each cell is mean +/- sample std over repeated runs of an identical recipe, varying only `--seed`.

Seeds per architecture: **EfficientNetB0** n=3 (seeds 42, 43, 44); **FastViT_T8** n=3 (seeds 42, 43, 44); **MobileNetV2** n=3 (seeds 42, 43, 44)

Compute backend: GPU (NVIDIA GeForce RTX 4050 Laptop GPU).
Training recipe: v2 (see src/training/schedule.py).

| Metric | EfficientNetB0 | FastViT_T8 | MobileNetV2 |
|---|---|---|---|
| Test accuracy (%) | 91.26 +/- 0.39 | 94.55 +/- 0.31 | 89.62 +/- 0.15 |
| ROC-AUC (OvR) | 0.9861 +/- 0.0012 | 0.9916 +/- 0.0002 | 0.9800 +/- 0.0018 |
| Macro F1 | 0.9137 +/- 0.0042 | 0.9467 +/- 0.0035 | 0.8983 +/- 0.0015 |
| Recall - unripe | 0.9427 +/- 0.0021 | 0.9400 +/- 0.0053 | 0.9397 +/- 0.0107 |
| Recall - underripe | 0.7960 +/- 0.0108 | 0.9033 +/- 0.0161 | 0.7933 +/- 0.0150 |
| Recall - ripe | 0.9553 +/- 0.0068 | 0.9663 +/- 0.0021 | 0.9207 +/- 0.0055 |
| Recall - overripe | 0.9623 +/- 0.0064 | 0.9763 +/- 0.0086 | 0.9363 +/- 0.0095 |
| Lighting spread (pts) | 2.45 +/- 0.37 | 1.02 +/- 0.43 | 5.19 +/- 0.88 |
| Params (M) | 4.05 +/- 0.00 | 3.20 +/- 0.00 | 2.26 +/- 0.00 |
| Model size (MB, pre-quant) | 33.9 +/- 0.0 | 30.2 +/- 0.0 | 20.9 +/- 0.0 |
| Inference (ms/img, dev) | 161.3 +/- 1.5 | 156.6 +/- 7.1 | 141.9 +/- 10.5 |

## Is the architecture gap real?

- 3 architecture(s) pooled - the pairwise verdict is emitted only when exactly 2 are present.
