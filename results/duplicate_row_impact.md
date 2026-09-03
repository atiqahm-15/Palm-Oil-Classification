# Duplicate-row defect — measured impact on reported test metrics

`data/processed/manifest_full.csv` test split: **2722 rows, 2695 unique images**. 27 rows (0.99%) are exact duplicates on `path`, `class`, `source_photo`, `split`, affecting 27 distinct images. They come from the Bunch Piles crop pipeline and never cross a split boundary, so they are double-counting, not leakage.

Duplicated images by class: unripe 7, underripe 4, ripe 9, overripe 7.

*as-scored* weights each image by its multiplicity in the manifest and reproduces the published figures; *dedup* counts each image once. Predictions are identical in both passes — only the weighting differs.

| Run | Accuracy as-scored | Accuracy dedup | Delta | Macro F1 as-scored | Macro F1 dedup | Delta | Accuracy on the duplicated images |
|---|---:|---:|---:|---:|---:|---:|---:|
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42 | 94.56% | 94.51% | -0.05 pts | 0.9469 | 0.9464 | -0.0005 | 100.00% |
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed43 | 94.23% | 94.17% | -0.06 pts | 0.9433 | 0.9428 | -0.0005 | 100.00% |
| fastvit_fyp2_blockft_v2_gpu_f50_k3_seed44 | 94.86% | 94.84% | -0.01 pts | 0.9498 | 0.9497 | -0.0001 | 96.30% |
| efficientnetb0_fyp2_blockft_v2_gpu_f50_k3_seed42 | 91.55% | 91.47% | -0.08 pts | 0.9170 | 0.9162 | -0.0008 | 100.00% |
| mobilenetv2_fyp2_blockft_v2_gpu_f50_k3_seed42 | 89.64% | 89.61% | -0.03 pts | 0.8981 | 0.8980 | -0.0002 | 92.59% |

**Largest absolute accuracy shift across the runs measured: 0.08 points.**

## What this does and does not establish

This measures the *reporting* error only: the extent to which published test metrics differ from what they would have been had the test split been deduplicated before scoring. It does not measure the *training* error. The training split carries its own duplicate rows, which double-weight some training images; estimating that effect would require retraining on a deduplicated manifest and is out of scope here. The correct claim from this table is about the numbers as reported, not about the models as trained.
