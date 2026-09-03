# Multi-seed results - Phase 2 architecture comparison

Pooled from `results/{*_blockft_v2_gpu_f50_k4_seed*}`. Each cell is mean +/- sample std over repeated runs of an identical recipe, varying only `--seed`.

Seeds per architecture: **EfficientNetB0** n=3 (seeds 42, 43, 44)

Compute backend: GPU (NVIDIA GeForce RTX 4050 Laptop GPU).
Training recipe: v2 (see src/training/schedule.py).

| Metric | EfficientNetB0 |
|---|---|
| Test accuracy (%) | 92.73 +/- 0.37 |
| ROC-AUC (OvR) | 0.9898 +/- 0.0007 |
| Macro F1 | 0.9283 +/- 0.0035 |
| Recall - unripe | 0.9507 +/- 0.0051 |
| Recall - underripe | 0.8307 +/- 0.0137 |
| Recall - ripe | 0.9670 +/- 0.0026 |
| Recall - overripe | 0.9640 +/- 0.0020 |
| Lighting spread (pts) | 1.33 +/- 0.15 |
| Params (M) | 4.05 +/- 0.00 |
| Model size (MB, pre-quant) | 38.4 +/- 0.0 |
| Inference (ms/img, dev) | 164.4 +/- 2.5 |

## Is the architecture gap real?

- 1 architecture(s) pooled - the pairwise verdict is emitted only when exactly 2 are present.
