# Per-class threshold calibration

Fitted on the **validation** split (objective: `accuracy`), applied unchanged to test. Test was never used for fitting; the validation gain is optimistic by construction and is shown alongside so any overfitting is visible.

## EfficientNetB0 (n=3)

| Metric | Baseline | Calibrated | Delta |
|---|---|---|---|
| Test accuracy (%) | 91.27 +/- 0.37 | 91.82 +/- 0.15 | **+0.55** |
| Test macro F1 | 0.914 +/- 0.004 | 0.921 +/- 0.001 | **+0.007** |
| Test recall - unripe | 0.943 +/- 0.002 | 0.925 +/- 0.009 | **-0.018** |
| Test recall - underripe | 0.797 +/- 0.010 | 0.870 +/- 0.008 | **+0.074** |
| Test recall - ripe | 0.955 +/- 0.007 | 0.926 +/- 0.004 | **-0.029** |
| Test recall - overripe | 0.962 +/- 0.006 | 0.960 +/- 0.004 | **-0.002** |

Validation gain +0.84 pts vs test gain +0.55 pts — the difference is the amount that did not generalise.

## FastViT_T8 (n=3)

| Metric | Baseline | Calibrated | Delta |
|---|---|---|---|
| Test accuracy (%) | 94.55 +/- 0.31 | 94.76 +/- 0.20 | **+0.21** |
| Test macro F1 | 0.947 +/- 0.003 | 0.949 +/- 0.002 | **+0.002** |
| Test recall - unripe | 0.940 +/- 0.005 | 0.940 +/- 0.005 | **+0.000** |
| Test recall - underripe | 0.904 +/- 0.016 | 0.924 +/- 0.010 | **+0.021** |
| Test recall - ripe | 0.966 +/- 0.002 | 0.962 +/- 0.007 | **-0.004** |
| Test recall - overripe | 0.976 +/- 0.008 | 0.967 +/- 0.003 | **-0.010** |

Validation gain +0.60 pts vs test gain +0.21 pts — the difference is the amount that did not generalise.

## MobileNetV2 (n=3)

| Metric | Baseline | Calibrated | Delta |
|---|---|---|---|
| Test accuracy (%) | 89.62 +/- 0.15 | 89.91 +/- 0.40 | **+0.29** |
| Test macro F1 | 0.898 +/- 0.001 | 0.901 +/- 0.004 | **+0.003** |
| Test recall - unripe | 0.940 +/- 0.011 | 0.919 +/- 0.016 | **-0.021** |
| Test recall - underripe | 0.794 +/- 0.015 | 0.814 +/- 0.026 | **+0.021** |
| Test recall - ripe | 0.921 +/- 0.006 | 0.926 +/- 0.006 | **+0.006** |
| Test recall - overripe | 0.937 +/- 0.010 | 0.943 +/- 0.010 | **+0.006** |

Validation gain +0.46 pts vs test gain +0.29 pts — the difference is the amount that did not generalise.

