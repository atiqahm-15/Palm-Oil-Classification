# Report-ready limitation prose

Draft prose for the FYP2 report's limitations / discussion chapters. Kept separate from
`PROJECT_PLAN.md` §8, which is the decisions *log* (what was decided and why, in engineering
terms); this file is the *report* register — what a reader and examiner should be told.

Every number here traces to an artifact under `results/`. Verify each against the final run
outputs before submission; nothing in this file should be quoted from memory.

---

## L1 — The lightweight ViT comparison: MobileViT could not be evaluated

**Status: write this regardless of the FastViT outcome.** The project plan named MobileViT
specifically, so its absence must be addressed rather than passed over.

> Objective 2 specified a comparison across three lightweight architectures, including a
> Vision Transformer variant, for which MobileViT was proposed. MobileViT could not be
> evaluated in this project. The obstacle was not computational cost or development time but
> the absence of any MobileViT checkpoint loadable under the project's software stack
> (TensorFlow 2.20 / Keras 3.14). Five independent integration routes were attempted, each
> failing for a distinct and reproducible reason: `keras.applications` provides no MobileViT
> implementation; the TensorFlow Hub module is published only in the legacy TF1-Hub format,
> which rejects `trainable=True` and so cannot be fine-tuned; the HuggingFace TensorFlow port
> depends on the `tf_keras` compatibility shim, which had already failed in this project on
> checkpoint reload; and the `keras_cv_attention_models` implementation raises a
> `KerasTensor` type error inside `transformer_pre_process`, where a raw TensorFlow operation
> is applied to a symbolic tensor — an operation Keras 3 forbids. That failure occurs at the
> patch-folding step which is constitutive of the MobileViT architecture, so it cannot be
> worked around without reimplementing the model. The library's own diagnostic suggests
> reverting to Keras 2 via `TF_USE_LEGACY_KERAS=1`, which returns to the already-failed shim.
>
> To establish whether this was specific to MobileViT or general to the library, fifteen
> lightweight transformer variants were swept under identical conditions. Only FastViT built
> successfully; MobileViT V2, LeViT, EfficientViT, EdgeNeXt and TinyViT all failed with the
> identical error, as did the non-transformer GhostNetV2. The limitation is therefore a
> property of the current Keras 3 ecosystem for mobile transformer architectures, not of this
> project's configuration.
>
> This is the second architecture in this project displaced by a pretrained-weights
> availability problem rather than by a research decision; EfficientNet-Lite0 was substituted
> with EfficientNetB0 for the same class of reason. Both substitutions are documented, and
> both were made to preserve comparability of the training recipe across architectures — a
> frozen-only or reimplemented model would confound architecture with training methodology.

**Follow with the substitution** (see L2). If FastViT results are unusable, this paragraph
stands alone and Objective 2 is reported as partially met, with two architectures compared.

---

## L2 — FastViT as the substituted third architecture

> FastViT (Vasu et al., Apple, 2023) was adopted in place of MobileViT. It is a hybrid
> convolution–transformer architecture designed for mobile inference, and therefore satisfies
> the objective's requirement of a lightweight Vision Transformer variant; MobileViT was
> named in the plan as an example rather than as a constraint. The FastViT-T8 variant was
> selected: at 3.20M parameters and 27.0MB it lies between MobileNetV2 (2.26M / 20.9MB) and
> EfficientNetB0 (4.05M / 33.9MB), so all three architectures occupy a comparable cost band
> and the comparison is not confounded by model scale.
>
> Before committing training time, the architecture was validated against six criteria drawn
> from the failures of earlier candidates: construction at 224×224 with ImageNet weights;
> exposure of block-structured layer names, required by the block-level fine-tuning scheme;
> compatibility with the pinned inference-mode BatchNorm used throughout this project;
> successful training steps in both the frozen and fine-tuned stages; and lossless save/reload
> round-trips in both the `.weights.h5` and `.keras` formats. The final criterion is the one
> on which the `tf_keras` route had previously failed.

> ⚠️ **Verify the FastViT citation venue (ICCV vs CVPR 2023) before submission.** It is not
> confirmed in this repository.

---

## L3 — Fine-tuning depth is not exactly matched across the three architectures

This one is easy to overlook and is exactly the kind of thing a viva probes. State it before
being asked.

> Fine-tuning depth in this project is specified in blocks rather than layers, because a fixed
> layer count represents different proportional depths in different architectures. That scheme
> is directly comparable only between architectures with the same block count: MobileNetV2 and
> EfficientNetB0 each expose sixteen blocks, so K=3 unfreezes 18.8% and 19.4% of their layers
> respectively. FastViT-T8 exposes ten blocks, so no integer K reproduces that proportion —
> K=2 unfreezes 15.4% of its layers and K=3 unfreezes 28.0%, bracketing the convolutional
> baselines rather than matching them.
>
> Rather than select one setting and treat the mismatch as negligible, both were run at three
> seeds each. This permits the fine-tuning depth most comparable to the CNN baselines to be
> reported as the primary result, while the second setting quantifies how sensitive FastViT
> actually is to fine-tuning depth — converting a potential confound into a measured effect.

**Measured result, for the discussion chapter:**

> Fine-tuning depth proved to have a larger effect on accuracy than the choice of architecture.
> Increasing FastViT's unfrozen depth from two blocks (15.4% of layers) to three (28.0%)
> improved test accuracy from 75.95% ± 0.43 to 78.72% ± 0.22 — a gain of 2.77 percentage
> points against a combined run-to-run spread of ±0.65, and an improvement on every measured
> axis simultaneously: macro F1 (0.754 → 0.786), underripe recall (0.422 → 0.509), stability
> under the lighting stress test (1.90 → 1.47 points of spread), and inference latency
> (160.6ms → 150.5ms). By comparison, the largest architecture gap measured under a fixed
> recipe was 2.84 points. Fine-tuning depth is therefore not a minor hyperparameter in this
> setting, and reporting an architecture comparison without controlling for it would risk
> attributing a recipe effect to the architecture.

---

## L3b — The strongest result is depth-confounded, and is reported as such

**This is the single most important honesty point in the results chapter.** Write it
immediately after any mention of the 78.72% figure.

> The best single configuration obtained in this project is FastViT-T8 fine-tuned at three
> blocks, reaching 78.72% ± 0.22 test accuracy and exceeding EfficientNetB0's 76.45% ± 0.90 by
> a margin larger than the combined run-to-run spread. This result must be qualified. At that
> setting FastViT is fine-tuned to 28.0% of its layers, whereas both convolutional baselines
> were trained at approximately 19%. The margin may therefore reflect fine-tuning depth rather
> than architecture, particularly given that depth was shown above to produce larger effects
> than architecture in this setting.
>
> A control was therefore run to separate the two: EfficientNetB0 at K=4, unfreezing 25.7% of
> its layers, the closest available match to FastViT's 28.0%. EfficientNetB0 did not benefit
> from the additional depth, reaching 76.03% ± 1.12 against 76.45% ± 0.90 at K=3 — a change of
> −0.42 points, well within run-to-run variation. FastViT's sensitivity to fine-tuning depth is
> therefore not a general property of the training recipe but specific to that architecture, and
> at comparable depth FastViT leads EfficientNetB0 by 2.69 points against a combined spread of
> ±1.34. The advantage survives the control and is attributed to the architecture.
>
> This asymmetry is itself a finding worth reporting: the hybrid transformer improves
> substantially when more of its backbone is unfrozen, while the convolutional baseline is
> already saturated at the shallower setting. A plausible reading is that FastViT's later blocks
> retain more ImageNet-specific representation, and so gain more from adaptation to this domain —
> though this project does not test that explanation directly.
>
> A residual mismatch of approximately 2.3 percentage points of depth remains even in the
> control, because neither architecture's block grid lands on the other's proportion:
> EfficientNetB0 steps 19.4% → 25.7% → 31.6% as K increases, and FastViT steps 15.4% → 28.0%.
> Exact depth matching is not achievable with block-level granularity across architectures with
> different block counts, and this is stated rather than smoothed over.

---

## L6 — Calibration helps the final model, but most of its apparent gain does not generalise

> Per-class threshold calibration was applied to all three architectures, with the weights fitted
> on the validation split alone and applied unchanged to the test split. For the selected model,
> FastViT-T8 at K=3, calibration raised test accuracy from 78.72% ± 0.22 to 79.17% ± 0.26. This
> gain should be interpreted cautiously: the same procedure produced a 2.09-point improvement on
> the validation split it was fitted to, so approximately 1.64 points did not transfer. By
> contrast, EfficientNetB0's calibration gain generalised considerably better (+1.73 on validation
> against +1.41 on test), and MobileNetV2's vanished entirely.
>
> The mechanism is consistent across architectures: calibration substantially recovers underripe
> recall — for FastViT, from 0.510 to 0.681 — but pays for it out of an adjacent class, in this
> case ripe (0.863 to 0.743). For FastViT that exchange is close to break-even, which is why the
> net accuracy gain is small despite a large shift in the confusion structure. Calibration is
> therefore reported as a modest, zero-cost refinement of the operating point rather than as an
> independent contribution to accuracy.

*Sources: `results/calibration_summary.md`, `results/calibration/*.json`.*

**Reproducibility caveat to include:** re-running calibration on identical EfficientNetB0
checkpoints produced +1.41 where an earlier run produced +1.52. Only the inference pass differed,
so this is GPU nondeterminism propagating into the fitted weights. Quote calibration deltas to at
most one decimal place, and regenerate the summary rather than citing an older copy.

---

## L4 — Two non-functional requirements remain unverified on hardware

> Of the five non-functional requirements, model size and inference latency have not been
> verified on a physical device. The figures reported here — 20.9MB / 27.0MB / 33.9MB and
> 143–161 ms per image — are pre-quantization measurements taken on a development machine, and
> the latency figures in particular are not comparable across compute environments: the same
> MobileNetV2 model measures 105 ms natively on Windows and 143 ms under WSL. The TensorFlow
> Lite conversion and on-device benchmark specified in Phase 3 were not carried out, so the
> ≤100MB and ≤2s requirements should be regarded as indicated but not demonstrated. All three
> models sit far enough below the size budget that meeting it is unlikely to be in doubt; the
> latency requirement is the one genuinely awaiting evidence.

---

## L5 — Numerical-reproducibility note (minor, but pre-empts a question)

> During FastViT training the XLA autotuner emitted warnings reporting mismatched results
> between candidate cuDNN convolution algorithms. These warnings did not appear for the two
> convolutional architectures. They are a known artifact of algorithm selection rather than an
> observed training failure, and the multi-seed protocol used throughout this project measures
> total run-to-run variance — which subsumes any such nondeterminism — so the reported spreads
> already account for it. It is noted here for completeness.
