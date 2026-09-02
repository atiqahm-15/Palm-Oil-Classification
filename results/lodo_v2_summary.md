# Leave-one-dataset-out (LODO) results — FastViT_T8

Each source held out entirely in turn: trained on the other 3, tested purely on the held-out source. Cells are mean +/- sample std over seeds of an identical recipe.

Compute backend: GPU.

Macro F1 is computed over the classes actually present in each held-out source, not over all 4 — sklearn's own macro average divides by the full label set and so penalises a fold for a class it could never have been tested on.

| Held-out source | n | Classes | Test accuracy (%) | ROC-AUC (OvR) | Macro F1 |
|---|---:|---:|---|---|---|
| v2_bunch_piles | 3 | 4 | 40.98 +/- 2.33 | 0.7153 +/- 0.0205 | 0.4128 +/- 0.0194 |
| v2_kaggle | 3 | 4 | 50.80 +/- 0.78 | 0.8295 +/- 0.0023 | 0.4972 +/- 0.0051 |
| v2_malaysia_outdoor | 3 | 3 | 60.90 +/- 0.37 | n/a | 0.6503 +/- 0.0125 |
| v2_mendeley_ordinal | 3 | 4 | 55.75 +/- 0.40 | 0.7900 +/- 0.0097 | 0.5474 +/- 0.0073 |
| **Mean across folds** | | | **52.11** | | |

## Per-class recall by fold

A blank cell means the held-out source contains no images of that class (support 0) — undefined, not zero.

| Held-out source | unripe | underripe | ripe | overripe |
|---|---|---|---|---|
| v2_bunch_piles | 0.140 +/- 0.017 | 0.547 +/- 0.128 | 0.459 +/- 0.126 | 0.546 +/- 0.127 |
| v2_kaggle | 0.541 +/- 0.020 | 0.067 +/- 0.012 | 0.921 +/- 0.005 | 0.912 +/- 0.019 |
| v2_malaysia_outdoor | 0.660 +/- 0.036 |  | 0.449 +/- 0.041 | 0.956 +/- 0.015 |
| v2_mendeley_ordinal | 0.732 +/- 0.010 | 0.063 +/- 0.020 | 0.817 +/- 0.003 | 0.823 +/- 0.020 |

## Generalization gap

- Main-split baseline (main split, FastViT_T8 K=3, recipe v2, n=3): **94.55%**
- LODO mean across 4 folds: **52.11%**
- **Gap: 42.44 points.**

The main-split number is measured on unseen *photos* from sources the model also trained on; the LODO number is measured on an entirely unseen source. The gap is the part of main-split accuracy that does not survive a change of camera, site, and labelling convention — the honest estimate of real-world deployment performance.

Fold training-set sizes differ substantially (holding out bunch_piles leaves ~4.7k training images vs ~15.1k for malaysia_outdoor), so a weak bunch_piles fold confounds domain shift with reduced training data. The other folds, which keep most of the corpus, isolate domain shift more cleanly.
