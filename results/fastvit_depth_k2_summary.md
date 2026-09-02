# Multi-seed results - Phase 2 architecture comparison

Pooled from `results/{fastvit_*_gpu_k2_seed*}`. Each cell is mean +/- sample std over repeated runs of an identical recipe, varying only `--seed`.

Seeds per architecture: **FastViT_T8** n=3 (seeds 42, 43, 44)

Compute backend: GPU (NVIDIA GeForce RTX 4050 Laptop GPU).

| Metric | FastViT_T8 |
|---|---|
| Test accuracy (%) | 75.95 +/- 0.43 |
| ROC-AUC (OvR) | 0.9315 +/- 0.0010 |
| Macro F1 | 0.7543 +/- 0.0059 |
| Recall - unripe | 0.8920 +/- 0.0050 |
| Recall - underripe | 0.4217 +/- 0.0204 |
| Recall - ripe | 0.8573 +/- 0.0150 |
| Recall - overripe | 0.8823 +/- 0.0047 |
| Lighting spread (pts) | 1.90 +/- 0.44 |
| Params (M) | 3.20 +/- 0.00 |
| Model size (MB, pre-quant) | 27.0 +/- 0.0 |
| Inference (ms/img, dev) | 160.6 +/- 14.8 |

## Is the architecture gap real?

- 1 architecture(s) pooled - the pairwise verdict is emitted only when exactly 2 are present.
