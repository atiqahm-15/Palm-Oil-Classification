# Report-ready limitation prose

> # ⚠️ READ THIS BEFORE DRAFTING FROM THIS FILE (updated 2026-09-03)
>
> **Every section is now current against recipe v2.** The headline figures are FastViT_T8
> **94.55% ± 0.31**, EfficientNetB0 **91.26% ± 0.39**, MobileNetV2 **89.62% ± 0.15**.
>
> Recipe-v1 numbers still appear in two places, both deliberately and both labelled as
> superseded in the surrounding text: the falsified depth control in L3b, and the contrast
> figures in L6 that show what calibration used to be worth. Do not lift either out of its
> context — the point of each is that it *no longer holds*.
>
> **One section reversed rather than merely aged: L3b.** If you drafted anything from an earlier
> copy of this file, that passage now argues the opposite conclusion. Re-derive it.
>
> Section-by-section status:
>
> | Section | Status |
> |---|---|
> | L1 | **Valid.** Argument does not depend on accuracy figures. |
> | L2 | **Valid**, size corrected below. |
> | L3 | **Rewritten 2026-09-03** on the v2 depth measurement (EfficientNetB0 K=3 → K=4). Its v1 measured result (FastViT K=2 vs K=3) is void and has no v2 equivalent — see the note in the section. |
> | L3b | **Rewritten 2026-09-03 — and its earlier conclusion was REVERSED.** The v1 K=4 control that closed this question turned out to be an under-training artifact; re-run under v2 it gives +1.47 rather than −0.42. About 45% of the apparent architecture margin is fine-tuning depth. Do not quote any pre-Sep-2026 version of this section. |
> | L6 | **Rewritten 2026-09-03** on the v2 refit. Calibration is now redundant rather than harmful: +0.21 for the selected model against a ±0.31 seed spread. The headline stays uncalibrated. |
> | L4 | **Rewritten 2026-09-02** — the TFLite conversion has now been carried out. |
> | L5 | **Valid.** |
> | L7 | **New 2026-09-24.** Added when the Android app's parity tests were first run on a device. Independent of every accuracy figure above; it scopes them. |
> | L8 | **New 2026-09-24, updated 2026-09-25.** The largest single effect measured in this project (-45.7 points). Not a modelling limitation — a capture-contract mismatch between the app's UI and the training distribution. **Now corrected in the app** (the capture is cropped to the brackets), but the correction's benefit is predicted by simulation and never confirmed on a real photograph. Draft it as "found and fixed, fix unverified in the field", not as resolved. |
>
> The headline-framing decision is settled (`PROJECT_PLAN.md` §8, 2026-09-02): the thesis
> headlines the main split (94.55%) and presents the 42.44-point LODO gap explicitly. A
> limitation register drafted from this file must carry that scoping sentence.
>
> **L8 is new, and it is the one section describing a defect the project fixed (2026-09-24,
> fix 2026-09-25).** It is the largest single accuracy effect anywhere in this project — bigger
> than the v1→v2 recipe fix, bigger than the LODO gap — and it was found on the app's own capture
> geometry, not in the model. The app now crops the capture to its framing brackets in response.
> Do not soften this into a footnote next to L7, and do not write it up as closed either: the fix
> is verified as geometry, and its accuracy benefit is predicted by the same simulation that found
> the defect, never measured on a photograph taken through the app.

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
> (TensorFlow 2.20 / Keras 3.15). Five independent integration routes were attempted, each
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
> selected: at 3.20M parameters and 30.2MB it lies between MobileNetV2 (2.26M / 20.9MB) and
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

**Measured result, for the discussion chapter.** *Rewritten 2026-09-03.* The previous version of
this passage quantified depth using FastViT at K=2 versus K=3 (+2.77 points). Those runs were
made under recipe v1 and are void; FastViT was not re-run at K=2 under v2, so there is no
like-for-like replacement for that particular measurement. The depth effect is instead quantified
from the EfficientNetB0 K=3 → K=4 control run under v2, which is the measurement that exists.

> Fine-tuning depth proved to have an effect on accuracy comparable to the choice of
> architecture. Increasing EfficientNetB0's unfrozen depth from three blocks (19.4% of its
> layers) to four (25.7%) improved test accuracy from 91.26% ± 0.39 to 92.73% ± 0.37 — a gain of
> 1.47 percentage points against a combined run-to-run spread of ±0.76, with no seed at the
> shallower setting reaching any seed at the deeper one. The gain was not confined to aggregate
> accuracy: recall on underripe, the weakest class throughout this project, rose from 0.796 to
> 0.831, and stability under the lighting stress test tightened from 2.45 to 1.33 points of
> spread. For comparison, the accuracy gap between EfficientNetB0 and MobileNetV2 — two
> architectures that expose the same number of blocks and are therefore directly comparable at
> a shared K — is 1.64 points. A single step of fine-tuning depth is worth about as much as the
> difference between two architectures. Depth is therefore not a minor hyperparameter in this
> setting, and an architecture comparison that does not control for it risks attributing a
> recipe effect to the architecture.

*Sources: `results/multiseed_v2_summary.md`, `results/effnet_depth_k4_v2_summary.md`.*

---

## L3b — Part of the architecture margin is fine-tuning depth, and the earlier control was an artifact

**This is the single most important honesty point in the results chapter.** Write it
immediately after any presentation of 94.55% as an *architecture* result.

*Rewritten 2026-09-03. The previous version of this section reached the opposite conclusion —
that EfficientNetB0 gained nothing from extra depth and that FastViT's advantage therefore
survived the control intact. That conclusion rested on a control run under recipe v1 and did not
survive re-running under v2. Do not quote any earlier version.*

> The best configuration obtained in this project is FastViT-T8 fine-tuned at three blocks,
> reaching 94.55% ± 0.31 test accuracy against EfficientNetB0's 91.26% ± 0.39 and MobileNetV2's
> 89.62% ± 0.15 at the same nominal setting. That comparison overstates the architectural
> difference, because the same block count is not the same proportional depth. FastViT-T8
> exposes ten blocks against sixteen for each convolutional baseline, so K=3 unfreezes 28.0% of
> FastViT's layers but only 19.4% of EfficientNetB0's and 18.8% of MobileNetV2's. The margin may
> therefore reflect fine-tuning depth rather than architecture, particularly given that depth was
> shown above to produce effects comparable in size to the architecture gap itself.
>
> A control was run to separate the two. EfficientNetB0 was retrained at K=4 — 25.7% of its
> layers, the closest available match to FastViT's 28.0% — at three seeds under an otherwise
> identical recipe. It reached 92.73% ± 0.37, an improvement of 1.47 points over the same
> architecture at K=3, with no seed at the shallower setting reaching any seed at the deeper one.
> **At comparable fine-tuning depth, FastViT-T8's advantage over EfficientNetB0 is 1.82 points
> rather than the 3.29 points implied by the same-K comparison. Approximately 45% of the apparent
> architectural advantage is attributable to fine-tuning depth rather than to architecture.** The
> remaining 1.82 points exceed the combined run-to-run spread of ±0.68, and that residual is
> attributed to the architecture.

**The falsified control — include this; it is a methodological contribution, not an admission.**

> This control was run once before, under the project's earlier training schedule, and produced
> the opposite result: EfficientNetB0 at K=4 scored 0.42 points *below* K=3, comfortably within
> run-to-run variation. That measurement supported a conclusion that sensitivity to fine-tuning
> depth was specific to the hybrid transformer and that the architecture margin survived the
> control undiminished. The finding is now known to be an artifact. The earlier schedule halted
> every run at a fixed epoch cap before convergence, so the additional unfrozen layers had no
> remaining training budget in which to adapt; the control could not have detected a depth effect
> had one been present. It is reported here as a superseded measurement.
>
> The episode illustrates a point that generalises beyond this project. An ablation inherits the
> validity of the recipe it is run under: a control executed on an under-trained pipeline measures
> the under-training, not the variable it was designed to isolate. Because the defect suppressed
> the treatment effect rather than producing an implausible number, the result appeared clean and
> was accepted. It was caught only because the recipe was later corrected for unrelated reasons
> and every dependent conclusion was re-derived rather than carried forward.

**Residual limitation — state it; it cuts against the result and pre-empts the obvious question.**

> Depth is still not exactly matched, and cannot be. EfficientNetB0's block grid steps
> 19.4% → 25.7% → 31.6% as K increases, while FastViT's steps 15.4% → 28.0%, so the closest
> available pairing leaves a residual mismatch of roughly 2.3 percentage points — with the
> convolutional baseline on the shallower side. That mismatch is not neutral: EfficientNetB0 was
> still gaining accuracy as depth increased rather than having saturated, so the residual favours
> FastViT. The 1.82-point margin should therefore be read as an upper bound on the architectural
> contribution rather than as a point estimate. Testing EfficientNetB0 at K=5 (31.6%), which would
> bracket FastViT's depth from above, is the natural next control and was not carried out.

*Sources: `results/multiseed_v2_summary.md`, `results/effnet_depth_k4_v2_summary.md`,
`src/training/finetune.py` (the measured layer-proportion table), `PROJECT_PLAN.md` §8.*

---

## L6 — Calibration was a remedy for an under-trained model, and the corrected recipe made it redundant

*Rewritten 2026-09-03 against the v2 refit (`results/calibration_v2_summary.md`). The previous
version reported calibration as a "modest, zero-cost refinement" worth +0.45 points on the
selected model. Those figures were fitted on recipe-v1 checkpoints and are void.*

> Per-class threshold calibration was applied to all three architectures, with weights fitted on
> the validation split alone and applied unchanged to the test split. Under the corrected
> training schedule the procedure yields a gain of 0.21 points for the selected model,
> FastViT-T8, raising test accuracy from 94.55% ± 0.31 to 94.76% ± 0.20. **That gain is smaller
> than the model's own seed-to-seed standard deviation.** Calibration is therefore reported as
> having no effect on the selected model that can be distinguished from run-to-run variance, and
> the model carried forward to deployment is the uncalibrated one.
>
> The reason is visible in the mechanism. Calibration works by trading recall between adjacent
> classes, so its value depends entirely on there being a badly-served class to trade toward.
> Under the earlier training schedule, FastViT-T8's underripe recall was 0.510; calibration
> raised it to 0.681 at a cost of 0.120 in ripe recall — a large change in the confusion
> structure for a net accuracy gain of 0.45 points. Under the corrected schedule underripe recall
> arrives at 0.904 with no calibration at all, and the same fitted procedure moves it by only
> 0.021. There is very little left for the technique to recover.
>
> The pattern across architectures is consistent with that account, though it is not a clean
> ordering. The two convolutional baselines, whose uncalibrated underripe recall sits near 0.79,
> gain more from calibration — 0.55 points for EfficientNetB0 and 0.29 for MobileNetV2 — than
> FastViT-T8 does at 0.904 underripe recall, which gains 0.21. The technique is worth least to
> the model with the least room to recover, which is also the model selected for deployment. The
> difference between the two convolutional baselines is not explained by this account: their
> underripe recalls differ by 0.003 while their calibration gains differ by 0.26 points, so that
> spread is better attributed to the variance of the fitting procedure than to any property of
> the architectures. Notably calibration is no longer harmful to any architecture — under the
> earlier schedule it cost MobileNetV2 0.24 points — so correcting the training schedule did not
> merely shrink the technique's benefit, it removed the conditions under which the technique
> mattered in either direction.
>
> As a methodological note, roughly two-thirds of the fitted gain does not transfer: FastViT-T8
> improves by 0.60 points on the validation split the weights were fitted to, against 0.21 on the
> held-out test split. Both are reported, because the difference between them is the measure of
> how far the fitting procedure overfits its own objective, and quoting only the validation figure
> would misrepresent the technique.

**Why this is reported rather than dropped.** Calibration is a negative result, and it is worth
the space because it corroborates the project's central finding from an independent direction: a
technique introduced specifically to compensate for poor underripe recall becomes superfluous once
the training schedule is fixed. That is evidence that under-training, rather than a modelling or
data limitation, was the binding constraint. Presenting it as a failed improvement would understate
it.

**Do not deploy the calibrated variant.** Beyond the gain being inside the noise, quoting 94.76%
would oblige the deployed pipeline to carry per-class threshold weights through the TensorFlow Lite
export and apply them at inference, adding a step to the mobile path for a benefit that cannot be
measured reliably.

*Sources: `results/calibration_v2_summary.md`, `results/calibration_v2/*.json`. The v1 figures are
retained for contrast in `results/calibration_summary.md` and `results/calibration/*.json`.*

**Reproducibility caveat to include:** re-running calibration on identical EfficientNetB0
checkpoints produced +1.41 where an earlier run produced +1.52. Only the inference pass differed,
so this is GPU nondeterminism propagating into the fitted weights. Quote calibration deltas to at
most one decimal place, and regenerate the summary rather than citing an older copy. This applies
with more force to the v2 figures, whose deltas are several times smaller than the v1 ones.

---

## L4 — Conversion and model size are demonstrated; on-device latency is not

**Rewritten 2026-09-02.** The previous version of this section stated that "the TensorFlow Lite
conversion and on-device benchmark specified in Phase 3 were not carried out." The conversion
half has since been carried out (`src/training/export_tflite.py`,
`results/tflite_export_summary.md`). Only the on-device half remains outstanding.

> Of the five non-functional requirements, the model-size budget is now demonstrated and the
> latency budget is not. All three trained architectures were converted to TensorFlow Lite at
> four quantization settings each. Every one of the twelve conversions succeeded using built-in
> TensorFlow Lite operators alone, with no recourse to the Flex delegate — a result that matters
> because the selected architecture, FastViT-T8, is a hybrid convolution-transformer assembled
> from custom layers, and a Flex-dependent conversion would have required the application to
> embed a portion of the full TensorFlow runtime. The selected deployment artefact, FastViT-T8
> quantized to float16, occupies 6.1MB against the 100MB requirement, and the largest artefact
> produced by any architecture at any setting was 15.3MB. The size requirement is therefore met
> with a margin of more than an order of magnitude, and is reported as demonstrated rather than
> indicated.
>
> The latency requirement remains unverified. Converted single-image inference was measured at
> 47.2ms for FastViT-T8, 23.6ms for EfficientNetB0 and 12.0ms for MobileNetV2 at float16,
> single-threaded on a development CPU. These figures sit three orders of magnitude below the
> 2-second requirement, but they are not measurements of the target hardware and no claim of
> compliance is made on their basis. A physical Android benchmark remains outstanding.
>
> One methodological consequence of the conversion should be recorded. The pre-quantization
> latency figures reported in the architecture comparison — 142ms, 161ms and 157ms for
> MobileNetV2, EfficientNetB0 and FastViT-T8 respectively — proved to be a poor predictor of
> converted performance. After conversion the same three models measure 12.1ms, 22.4ms and
> 47.7ms: a 3.9-fold spread where the pre-conversion figures showed 1.13-fold, and in a
> different rank order. Pre-conversion latency measured in the training framework is therefore
> not a valid proxy for deployed latency, and is reported here as a superseded measurement
> rather than as corroborating evidence.

**Companion finding — quantization robustness is architecture-dependent, and it inverts the
accuracy ranking.** This belongs in the results chapter, not only in the limitations register.

> Post-training full-integer quantization was found to be unusable for the two more accurate
> architectures. Applied to FastViT-T8 it reduced accuracy to 0.260 on a four-class problem —
> indistinguishable from chance — and to EfficientNetB0 it gave 0.645, against 0.910 for the
> unquantized model. MobileNetV2 was unaffected, retaining 0.915. The pattern is consistent
> with the architectures' activation statistics: FastViT's normalization and attention blocks
> and EfficientNet's SiLU activations and squeeze-and-excitation branches both produce dynamic
> ranges that a single per-tensor integer scale cannot represent, whereas MobileNetV2's
> bounded ReLU6 design was explicitly intended to quantize. Quantization robustness therefore
> runs inversely to the accuracy ranking established in this project: the least accurate of the
> three architectures is the most robust to deployment-time compression.
>
> This finding also illustrates a methodological point worth stating. All three fully-quantized
> models converted without error and produced the smallest files of any setting tested. An
> evaluation measuring only conversion success, file size and latency would have selected the
> fully-quantized model for all three architectures, including the one performing at chance.
> The failure is detectable only by comparing the converted model's predictions against those
> of the model it was derived from, which is why every conversion in this project was verified
> against its unquantized parent rather than accepted on conversion success alone.

**Caveats to carry with these figures:** the export covers seed 42 only, not the project's
three-seed protocol; the accuracy column is measured on a 200-image sample sized to compare
converted against unconverted predictions, on which all three architectures fall within one
point of each other and which must not be read as an architecture comparison; and latency is
single-threaded x86, whereas integer arithmetic frequently outperforms float16 on ARM via
XNNPACK or NNAPI, so the float16 recommendation may not survive an on-device measurement.

*Sources: `results/tflite_export_summary.md`, `results/<run>/tflite/export_report.json`.*

---

## L5 — Numerical-reproducibility note (minor, but pre-empts a question)

> During FastViT training the XLA autotuner emitted warnings reporting mismatched results
> between candidate cuDNN convolution algorithms. These warnings did not appear for the two
> convolutional architectures. They are a known artifact of algorithm selection rather than an
> observed training failure, and the multi-seed protocol used throughout this project measures
> total run-to-run variance — which subsumes any such nondeterminism — so the reported spreads
> already account for it. It is noted here for completeness.

---

## L7 — The deployed app cannot decode a JPEG the way training did, and the gap is measured on 20 images only

> Every accuracy figure in this report was produced on images decoded by TensorFlow's
> `tf.io.decode_jpeg`. The Android application cannot call that decoder; it decodes through the
> platform's own libjpeg configuration. The two do not agree exactly. Measured over the twenty
> stratified test-split fixtures bundled with the application, an independent decoder disagrees
> with TensorFlow's by up to 16 levels out of 255 per pixel (mean 1.8), and the model's output
> probabilities move by up to 0.042 as a result, with a mean shift of 0.006. **The predicted class
> did not change on any of the twenty images.**
>
> This is not an implementation defect, and the instrumented test suite is what separates the two
> possibilities. Fed the exact input tensors the Python pipeline fed, the application reproduces
> the reference probabilities to within 1e-4, which establishes that the interpreter wiring,
> tensor layout and class ordering are correct. A separate unit test establishes that the
> application's bilinear resize matches `tf.image.resize` under half-pixel centres to better than
> 1e-3 across 800 sampled positions. With those two fixed, the residual difference is attributable
> to image decoding alone — and the same worst case, on the same image, is reproduced to seven
> decimal places by a desktop decoder that is not TensorFlow's, which is the evidence that the
> decoder rather than the application is responsible.
>
> The limitation is one of scope, not of direction. Twenty images bound the effect anecdotally;
> they do not quantify what decoding through a different library does to accuracy across the
> full 2,722-image test set. That measurement is inexpensive — re-scoring the test split with a
> second decoder — and until it is carried out, the reported test accuracy should be read as the
> accuracy of the model on TensorFlow-decoded images, with the on-device figure expected to be
> very close but not proven identical.

---

## L8 — The camera's framing guide feeds the model a distribution it never trained on, and the cost is 45.7 points

> Every training, validation and test image in this project is a tight crop around a single
> bunch. The application's camera screen instead draws a framing guide over roughly a quarter to
> a third of the preview and then feeds the *entire* preview frame, guide and all, to the model —
> so a user who obeys the on-screen hint hands the model a small bunch surrounded by background
> at a scale and context nothing in training resembled. This was identified qualitatively during
> Phase 4 screen-building and is now measured: re-scoring the full 2,722-image test split through
> the application's actual capture geometry (a 9:20 aspect crop covering ~25% of the frame area,
> matching the drawn guide) drops accuracy from a same-model baseline of 94.16% to **48.46%**, a
> **45.7-point** loss. This is the single largest accuracy effect measured anywhere in this
> project — larger than the entire under-trained-recipe correction (78.69%→93.98%) and larger
> than the 42.44-point leave-one-dataset-out generalization gap (L1-adjacent, see the headline
> scoping note above).
>
> The loss is not evenly distributed. Unripe and underripe recall — already this project's
> weakest boundary — collapse hardest (0.938→0.390 and 0.904→0.312 respectively), while ripe and
> overripe are comparatively robust (0.958→0.673 and 0.971→0.554). A user photographing an
> ambiguous bunch through the current framing guide is disproportionately likely to get the
> classes the model was already least confident about.
>
> Isolating the two effects the capture geometry combines — the frame being stretched to a
> non-square aspect, and the frame covering only a fraction of the model's field of view — shows
> scale dominates. Holding the crop at the full frame and varying only aspect costs between -0.6
> points (a 1:1 guide) and -12.7 points (the app's actual 9:20 preview), a real but comparatively
> minor effect. But once the crop is narrowed to the app's ~25% area fraction, aspect stops
> mattering: a square guide at that area still loses 47.8 points, worse than the app's own 9:20
> guide at the same area. **A squarer viewfinder alone would not fix this** — any fix has to give
> the model a materially larger fraction of the frame, not just a better-shaped one.
>
> This measurement is a synthetic bound, not a field result, and should be reported as one. The
> background behind the simulated crop is either a blurred continuation of the crop itself or
> flat grey — a stand-in for real plantation clutter, not a photograph of it — so the true
> on-device cost could be somewhat better or somewhat worse than 45.7 points; the honest claim is
> that the scale/aspect mismatch alone, independent of background realism, costs roughly this
> much.
>
> **The application was changed in response to this measurement rather than merely documenting
> it.** The capture is now cropped to the framing brackets before preprocessing: the brackets are
> measured against the camera preview at capture time and the decoded frame is cut to them, so the
> model receives the region the user framed rather than the whole field of view, and the review
> screen displays that same cropped region. Cropping was preferred over the alternative remedy —
> widening the brackets and rewriting the on-screen instruction — because it does not depend on the
> user complying with a hint. The measurement also rules out the intuitive remedy of simply making
> the viewfinder squarer: at the application's framing, a 1:1 guide loses 47.8 points, marginally
> worse than the tall one, because the dominant term is how much of the frame the bunch occupies
> rather than the shape it is stretched into.
>
> **The recovery was measured at the geometry actually shipped, and it is a simulated recovery.**
> The brackets are 280 x 360 dp — an aspect ratio of 7:9 — so that condition was re-scored over the
> full test split rather than interpolated from neighbouring ones. With the bunch filling the
> brackets, accuracy is **92.47%** against the 94.16% baseline, a residual loss of 1.69 points:
> the correction recovers 44.0 of the 45.7 points the uncorrected framing cost. The residual falls
> almost entirely on the underripe class, whose recall goes from 0.904 to 0.834 while unripe, ripe
> and overripe are within a point of baseline, which is consistent with every other result in this
> project locating the residual difficulty at the underripe boundary. Mean confidence in the top
> class also falls, from 0.805 to 0.761. **This remains a simulation**: it measures the scale and
> aspect change the crop makes to test-split images, using the same synthetic backgrounds that
> found the defect, and the correctness of the crop geometry itself is covered by unit tests rather
> than by photographs. **No image taken through the corrected application has been scored, because
> field photographs could not be obtained for this project.**
> Two risks consequently remain open and should be stated alongside any claim about field
> performance: the synthetic backgrounds used throughout may not behave as real plantation clutter
> does, and a bunch that overflows the brackets is now clipped by the crop where previously it was
> merely diluted by the surrounding frame. The honest summary is that a quantified defect was found
> by simulation and corrected by construction, with the correction's benefit predicted by the same
> simulation and not independently confirmed.
