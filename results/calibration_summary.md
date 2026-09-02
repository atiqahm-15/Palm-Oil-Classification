# Per-class threshold calibration

> ## ⚠️ SUPERSEDED — recipe v1 numbers (banner added 2026-09-02)
>
> Every figure in this file was fitted on **recipe v1** checkpoints, whose baselines
> (EfficientNetB0 76.45%, FastViT_T8 78.72%) were produced by under-trained runs. The v2
> campaign supersedes them: FastViT_T8 94.55% ± 0.31, EfficientNetB0 91.26% ± 0.39,
> MobileNetV2 89.62% ± 0.15.
>
> Calibration's mechanism here was buying underripe recall at ripe's expense. Under v2 the
> model already reaches 0.903 underripe recall in training, so calibration may now be
> unnecessary or actively harmful — see `PROJECT_PLAN.md` §8. **Refit on the v2 checkpoints
> before quoting any calibrated number.**
>
> This banner is hand-added and will be lost when `calibrate_thresholds.py` regenerates the
> file — which is the intended outcome, since regenerating it is what makes the banner
> unnecessary.

Fitted on the **validation** split (objective: `accuracy`), applied unchanged to test. Test was never used for fitting; the validation gain is optimistic by construction and is shown alongside so any overfitting is visible.

## EfficientNetB0 (n=3)

| Metric | Baseline | Calibrated | Delta |
|---|---|---|---|
| Test accuracy (%) | 76.45 +/- 0.87 | 77.86 +/- 0.69 | **+1.41** |
| Test macro F1 | 0.762 +/- 0.012 | 0.786 +/- 0.006 | **+0.024** |
| Test recall - unripe | 0.926 +/- 0.006 | 0.811 +/- 0.051 | **-0.115** |
| Test recall - underripe | 0.493 +/- 0.071 | 0.649 +/- 0.028 | **+0.156** |
| Test recall - ripe | 0.759 +/- 0.044 | 0.785 +/- 0.022 | **+0.026** |
| Test recall - overripe | 0.913 +/- 0.008 | 0.894 +/- 0.020 | **-0.019** |

Validation gain +1.73 pts vs test gain +1.41 pts — the difference is the amount that did not generalise.

## FastViT_T8 (n=3)

| Metric | Baseline | Calibrated | Delta |
|---|---|---|---|
| Test accuracy (%) | 78.72 +/- 0.22 | 79.17 +/- 0.26 | **+0.45** |
| Test macro F1 | 0.786 +/- 0.003 | 0.799 +/- 0.002 | **+0.013** |
| Test recall - unripe | 0.882 +/- 0.017 | 0.858 +/- 0.016 | **-0.025** |
| Test recall - underripe | 0.510 +/- 0.010 | 0.681 +/- 0.029 | **+0.171** |
| Test recall - ripe | 0.863 +/- 0.018 | 0.743 +/- 0.053 | **-0.120** |
| Test recall - overripe | 0.912 +/- 0.013 | 0.918 +/- 0.008 | **+0.006** |

Validation gain +2.09 pts vs test gain +0.45 pts — the difference is the amount that did not generalise.

## MobileNetV2 (n=3)

| Metric | Baseline | Calibrated | Delta |
|---|---|---|---|
| Test accuracy (%) | 73.63 +/- 0.60 | 73.39 +/- 0.51 | **-0.24** |
| Test macro F1 | 0.739 +/- 0.009 | 0.743 +/- 0.004 | **+0.005** |
| Test recall - unripe | 0.841 +/- 0.017 | 0.780 +/- 0.014 | **-0.061** |
| Test recall - underripe | 0.483 +/- 0.059 | 0.606 +/- 0.038 | **+0.123** |
| Test recall - ripe | 0.779 +/- 0.018 | 0.703 +/- 0.051 | **-0.077** |
| Test recall - overripe | 0.864 +/- 0.030 | 0.882 +/- 0.010 | **+0.018** |

Validation gain +0.39 pts vs test gain -0.24 pts — the difference is the amount that did not generalise.

