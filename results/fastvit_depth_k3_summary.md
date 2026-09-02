# Multi-seed results - Phase 2 architecture comparison

Pooled from `results/{fastvit_*_gpu_k3_seed*}`. Each cell is mean +/- sample std over repeated runs of an identical recipe, varying only `--seed`.

Seeds per architecture: **FastViT_T8** n=3 (seeds 42, 43, 44)

Compute backend: GPU (NVIDIA GeForce RTX 4050 Laptop GPU).

| Metric | FastViT_T8 |
|---|---|
| Test accuracy (%) | 78.72 +/- 0.22 |
| ROC-AUC (OvR) | 0.9425 +/- 0.0006 |
| Macro F1 | 0.7860 +/- 0.0026 |
| Recall - unripe | 0.8827 +/- 0.0170 |
| Recall - underripe | 0.5093 +/- 0.0097 |
| Recall - ripe | 0.8630 +/- 0.0173 |
| Recall - overripe | 0.9120 +/- 0.0125 |
| Lighting spread (pts) | 1.47 +/- 0.53 |
| Params (M) | 3.20 +/- 0.00 |
| Model size (MB, pre-quant) | 30.2 +/- 0.0 |
| Inference (ms/img, dev) | 150.5 +/- 2.6 |

## Is the architecture gap real?

- 1 architecture(s) pooled - the pairwise verdict is emitted only when exactly 2 are present.
