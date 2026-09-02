# Multi-seed results - Phase 2 architecture comparison

Pooled from `results/{efficientnetb0_*_gpu_k4_seed*}`. Each cell is mean +/- sample std over repeated runs of an identical recipe, varying only `--seed`.

Seeds per architecture: **EfficientNetB0** n=3 (seeds 42, 43, 44)

Compute backend: GPU (NVIDIA GeForce RTX 4050 Laptop GPU).

| Metric | EfficientNetB0 |
|---|---|
| Test accuracy (%) | 76.03 +/- 1.12 |
| ROC-AUC (OvR) | 0.9333 +/- 0.0029 |
| Macro F1 | 0.7550 +/- 0.0106 |
| Recall - unripe | 0.9203 +/- 0.0180 |
| Recall - underripe | 0.4483 +/- 0.0117 |
| Recall - ripe | 0.7873 +/- 0.0142 |
| Recall - overripe | 0.9160 +/- 0.0122 |
| Lighting spread (pts) | 3.70 +/- 0.14 |
| Params (M) | 4.05 +/- 0.00 |
| Model size (MB, pre-quant) | 38.4 +/- 0.0 |
| Inference (ms/img, dev) | 162.2 +/- 4.4 |

## Is the architecture gap real?

- 1 architecture(s) pooled - the pairwise verdict is emitted only when exactly 2 are present.
