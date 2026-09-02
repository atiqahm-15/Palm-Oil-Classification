# Leave-one-dataset-out (LODO) results — EfficientNetB0

Each source held out entirely in turn: trained on the other 3, tested purely on the held-out source. Cells are mean +/- sample std over seeds of an identical recipe.

Compute backend: GPU.

Macro F1 is computed over the classes actually present in each held-out source, not over all 4 — sklearn's own macro average divides by the full label set and so penalises a fold for a class it could never have been tested on.

| Held-out source | n | Classes | Test accuracy (%) | ROC-AUC (OvR) | Macro F1 |
|---|---:|---:|---|---|---|
| bunch_piles | 3 | 4 | 51.88 +/- 2.91 | 0.7990 +/- 0.0071 | 0.5082 +/- 0.0221 |
| kaggle | 3 | 4 | 56.56 +/- 0.36 | 0.8106 +/- 0.0014 | 0.5901 +/- 0.0096 |
| malaysia_outdoor | 3 | 3 | 62.98 +/- 4.19 | n/a | 0.6774 +/- 0.0350 |
| mendeley_ordinal | 3 | 4 | 52.40 +/- 0.20 | 0.7888 +/- 0.0001 | 0.5181 +/- 0.0051 |
| **Mean across folds** | | | **55.96** | | |

## Per-class recall by fold

A blank cell means the held-out source contains no images of that class (support 0) — undefined, not zero.

| Held-out source | unripe | underripe | ripe | overripe |
|---|---|---|---|---|
| bunch_piles | 0.418 +/- 0.111 | 0.199 +/- 0.135 | 0.738 +/- 0.142 | 0.719 +/- 0.026 |
| kaggle | 0.604 +/- 0.073 | 0.406 +/- 0.031 | 0.659 +/- 0.028 | 0.785 +/- 0.005 |
| malaysia_outdoor | 0.583 +/- 0.010 |  | 0.580 +/- 0.080 | 0.878 +/- 0.017 |
| mendeley_ordinal | 0.824 +/- 0.002 | 0.128 +/- 0.013 | 0.621 +/- 0.009 | 0.802 +/- 0.005 |

## Generalization gap

- Main-split baseline (main split, EfficientNetB0 n=3): **76.45%**
- LODO mean across 4 folds: **55.96%**
- **Gap: 20.49 points.**

The main-split number is measured on unseen *photos* from sources the model also trained on; the LODO number is measured on an entirely unseen source. The gap is the part of main-split accuracy that does not survive a change of camera, site, and labelling convention — the honest estimate of real-world deployment performance.

Fold training-set sizes differ substantially (holding out bunch_piles leaves ~4.7k training images vs ~15.1k for malaysia_outdoor), so a weak bunch_piles fold confounds domain shift with reduced training data. The other folds, which keep most of the corpus, isolate domain shift more cleanly.
