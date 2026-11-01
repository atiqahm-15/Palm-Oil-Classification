# FYP2 Project Plan

**Smartphone-Based Palm Oil Ripeness Detection Using Lightweight Vision Models**

Nur Atiqah Maisarah Binti Mohamad Rasi (22003055)
Supervisor: Dr. Muhammad Shahreeza Safiruz Kassim · Co-Supervisor: Ahmad Syazwan Ramli (MPOB)
Prepared: August 2026 — Semester break, ahead of formal FYP2 supervision
*Revision 2: updated after the decision to stop using the original Roboflow dataset and source a new, merged dataset instead.*

## 0. Current Status — updated 2026-09-03

| Phase | Status |
|---|---|
| 0 — Environment & data sourcing | Complete |
| 1 — Unified dataset construction | Complete |
| 2 — Model development & comparison | **Complete under recipe v2** — FastViT_T8 94.55% ± 0.31, see end of §8 |
| 3 — Mobile optimization (TFLite) | **Conversion + lighting done; one item left.** 12/12 exports builtin-only, no Flex (2026-09-02); FastViT_T8 float16 (6.1MB) is the recommended artefact. Lighting re-measured through the converted model 2026-09-25 — spread 1.36 pts, full-split TFLite accuracy 94.12% (-0.44 vs Keras). **Only the on-device latency benchmark remains**; it is written and compiling (`DeviceLatencyBenchmark`, `docs/device_session_checklist.md`). See end of §8 |
| 4 — Android app | **All four screens built 2026-09-24** — end-to-end capture/upload → on-device inference → result runs on an emulator; app reproduces Python to within the JPEG-decoder difference. Camera framing was measured as a -45.7-point risk and is **fixed 2026-09-25 by cropping the capture to the brackets** (verified geometrically, not in the field). Not yet seen on a phone. See end of §8 |
| 5 — Extended evaluation & ablation | **v2 LODO complete** — FastViT_T8, 4 folds × 3 seeds, mean **52.11%** vs 94.55% main split (see end of §8). Ablations not started |
| 6 — Report writing & viva prep | Not started |

**The one thing to know before quoting any number in this document.** In Sep 2026 every
training run in the project was found to have been stopped by an epoch cap before it
converged — fine-tuning early-stopped in 0 of 15 GPU runs. A corrected recipe
(`src/training/schedule.py`, "v2") lifted a single FastViT seed from 78.69% to **93.98%**
test accuracy on unchanged data and unchanged splits, since confirmed across all three
architectures and three seeds (FastViT_T8 **94.55% ± 0.31**, EfficientNetB0 91.26% ± 0.39,
MobileNetV2 89.62% ± 0.15). **Every accuracy figure in §8 dated Aug 2026 is therefore a
floor, not a ceiling.** All three architectures now clear the ≥85% NFR on the main split.
Read the final entry of §8 first.

**The second thing to know, added 2026-09-02.** The v2 leave-one-dataset-out campaign has
now finished, and **cross-source accuracy did not follow the main split up**. LODO mean is
**52.11%** against a 94.55% main split — a **42.44-point generalization gap**, wider than the
20.49 points measured under v1. The corrected recipe bought ~18 points on held-out *photos*
from sources the model trained on, and gave back ~4 points on an entirely held-out *source*.
Critically this is **not** another under-training artifact: 10 of 12 fine-tuning stages
early-stopped, versus 0 of 15 under v1. The honest reading is that a properly-trained model
fits each training source's labelling convention harder rather than generalizing past it.
See the final entry of §8 for the full result and its confound.

**The third thing to know, decided 2026-09-02.** The reporting question this forced is now
settled: **the thesis headlines the main split (94.55% ± 0.31) and presents the 42.44-point
gap as a quantified part of the result, not a footnote.** Every appearance of the headline
must carry the scoping sentence given in the final entry of §8 — the ≥85% NFR is met on the
held-out test split and is *not* claimed to hold under source shift. Consequence for the
critical path: Phase 3/4 only have to demonstrate the main-split model on-device.

**The fourth thing to know, added 2026-09-03 — the report can now be drafted from
`results/`.** Two things that would have forced a rewrite mid-draft are closed. (1) Every
figure in `results/figures/` was a recipe-v1 figure, because the figure scripts *defaulted* to
the v1 runs and CSVs; all eleven are now regenerated on the v2 campaign, and four
silent-wrong-figure bugs in those scripts were fixed. (2) The duplicate-row defect is measured
rather than merely bounded: it moves the headline by **-0.04 points** (94.55% → 94.51%),
at most 0.08 across any run, always upward-biasing, and no conclusion changes — so the campaign
is **not** being re-run for it. See the last two entries of §8.

Calibration is now refit on v2 (see §8): the gain shrinks to +0.21 points for the selected
model against a ±0.31 seed spread, so the headline stays **uncalibrated** and figure 10 is
regenerated. **Every figure in `results/figures/` is now a v2 figure.**

The fine-tuning-depth confound is now controlled (see §8), and the answer changes the
discussion chapter: EfficientNetB0 gains **+1.47 points** from the extra depth under v2, where
under v1 it gained nothing. At comparable depth FastViT_T8 still leads, but by **1.82 points
rather than 3.29** — roughly **45% of the apparent architecture gap was depth**. The headline
is unaffected; the *attribution* is. `docs/report_limitations.md` **L3b is now wrong, not just
void, and must be rewritten** — its central claim is the falsified v1 control.
Note that the v2 campaign being uniformly K=3 does *not* resolve it: K counts blocks, and
FastViT_T8 has ten blocks against sixteen for both CNNs, so K=3 unfreezes 28.0% of FastViT's
layers against ~19% for MobileNetV2 and EfficientNetB0. That is precisely the mismatch

**The fifth thing to know, added 2026-09-24.** The camera-framing question flagged at the end
of the previous entry — that the app feeds the model a whole, background-heavy frame while
every training image is a tight bunch crop — is no longer a hunch. `src/training/
check_framing_shift.py` re-scored the full 2,722-image test split through the app's actual
capture geometry (9:20 aspect, ~25% area fraction, matching the drawn framing guide) and
measured **48.46% accuracy against a 94.16% same-model baseline — a 45.7-point drop**, larger
than every other single effect measured in this project, including the entire v1→v2 recipe
fix and the LODO generalization gap. Unripe and underripe recall are hit hardest (0.94→0.39,
0.90→0.31); ripe and overripe are comparatively robust (0.96→0.67, 0.97→0.55).

**Fixed 2026-09-25**: the app now crops the capture to the framing brackets, so the model is
given what the user framed rather than the whole frame. Scored at the shipped 7:9 bracket
geometry, that recovers **44.0 of the 45.7 points** (92.47% against the 94.16% baseline,
`results/framing_shift_brackets/`). The fix is unit-tested and builds, but **no photograph taken
through the shipped app has been scored and field photographs could not be obtained** — so it is
verified geometrically and in simulation, not in a plantation, and the report must say exactly
that. See the last two entries of §8 and `docs/report_limitations.md` L8.
`docs/report_limitations.md` L3b describes, and the control that previously closed it
(EfficientNetB0 at K=4) was measured under v1 and is void. Either re-run that control under v2
— three seeds — or report the confound as unresolved.

**Reading order:** §0 for status, the last entry of §8 for what is in flight, §9 for what
needs a supervisor decision.

## 1. Purpose

FYP1 is complete: literature review, problem statement, objectives, system design, and a MobileNetV2 baseline are done. This document restarts the planning for FYP2 with precise phases and milestones, grounded in the actual FYP1 results and the GitHub repository, rather than a generic template. This revision reflects a second restart decision: the original training dataset turned out to be reused, leaky, and thin (see Section 3), so Phases 0–1 now cover sourcing and building a new, merged dataset from independent public sources before any retraining happens.

## 2. Where FYP1 Left Off

### 2.1 Approved objectives

- Develop a smartphone-based oil palm FFB ripeness detection system using lightweight image classification models.
- Analyse and compare lightweight architectures — MobileNet, EfficientNet-Lite, and Vision Transformers — for ripeness classification.
- Evaluate the models under varying lighting conditions and optimize the best-performing one for smartphone deployment.

### 2.2 Baseline results (MobileNetV2, transfer learning)

- Two-stage training: frozen feature extraction, then fine-tuning the last 30 layers at lr = 1e-5, with label smoothing 0.15.
- Top-1 accuracy: 68.14% · Top-2 accuracy: 89.06% (test set, 521 images).
- Accuracy held reasonably steady across lighting shifts (original 0.681, darkened 0.678, brightened 0.655), but off a low base.
- Per-class recall is very uneven: overripe 0.72, ripe 0.65, underripe 0.87, unripe 0.38 — the model is strongly biased toward predicting "underripe."

### 2.3 Non-functional requirements (still to be met)

- ≥ 85% classification accuracy on the test set
- ≤ 2 seconds inference per image on-device
- ≤ 100 MB model size for mobile deployment
- Stable performance across daylight, shadow, and low-light conditions
- Support for input resolutions as low as 224 × 224

## 3. Diagnosis: What Needed to Change

| Issue | Evidence from FYP1 | Why it matters |
|---|---|---|
| Accuracy well below target | Top-1 accuracy 68.14% vs the ≥85% NFR — a 17-point gap | Biggest blocker to FYP2 sign-off; every later phase depends on closing it |
| Unripe class barely learned | Unripe recall = 0.38; 54 of 105 unripe test images predicted as underripe | Class imbalance in training data, not a modelling limitation alone |
| Ripe/underripe boundary confusion | Ripe has the lowest ROC-AUC (0.826); 37 ripe images misclassified as underripe | Matches a limitation flagged in the literature review — a known hard problem |
| Only 1 of 3 architectures tested | Repo and slides show MobileNetV2 results only | Objective 2 requires comparing MobileNet, EfficientNet-Lite, and ViT — not done yet |
| Mobile app not started | Repo contains only the training notebook, README, and results/ | Large, separate body of work still needed |
| Dataset turned out to be reused | The "new" Roboflow zip had identical per-class test counts to the FYP1 baseline | Confirmed it was the same public dataset already trained on |
| Severe train/valid/test leakage | 194 of 205 unique source photos had crops split across train, valid, *and* test | The 68.14% accuracy is likely an overestimate of real generalization |
| Thin, single-source data | Only 205 unique source photographs, all shot over 2 days (3–4 Aug 2022) | Explains low diversity and the unripe-class weakness (only 45 unique source photos) |

## 4. New Dataset Sourcing Decision

Instead of retraining on the same public dataset again, FYP2 merges independent public sources:

- **Bunch Piles dataset** (Suharjito & Junior, *Scientific Data*, Science Data Bank) — 7,171 images, 6 classes incl. empty bunch/abnormal, CC BY 4.0. Collected at palm oil mills with a smartphone, expert-validated. Ships as YOLO detection boxes, not pre-cropped classification images.
- **Outdoor-Tenera Malaysia set** (MunirahRosbi, GitHub) — collected in Johor, Negeri Sembilan, and Perak under outdoor lighting; pre-split train/test.
- **Mendeley Ordinal dataset** (Kamal et al., 2024) — 4,728 high-res images, 5 classes, Central Kalimantan, industry-funded, stratified splits.
- **Kaggle supplement** (hadeemadadum) — 857 images, exact 4-class match to the current taxonomy; used as a targeted top-up, not a primary source.

The original Roboflow `palm-fruit-ripeness-classificationcnn` set is retired as a training source and kept only as a documented negative result in the report.

## 5. Phased Plan

Timeline is in relative weeks starting from when you begin executing this plan — anchor Week 1 to your actual start date once your FYP2 supervision schedule is confirmed.

### Phase 0 — Weeks 1–2: Restart: Environment & Data Sourcing ✅ COMPLETE

- Restructure GitHub repo: `src/`, `data/` (gitignored), `models/`, `results/<run-name>/`, `app/`, `docs/`; drop the project `CLAUDE.md` into the repo root.
- Set up VS Code + Claude Code extension (or the Claude Code CLI) against the restructured repo, plus a pinned Python venv/requirements.txt.
- Download and spot-check each candidate dataset before committing to it (Bunch Piles, Outdoor-Tenera, Mendeley, Kaggle).
- Visually sample ~10 images per class per source — catch label/domain problems before merging, not after training.
- Discard the old Roboflow set as a training source given its leakage and 205-photo ceiling; keep it only as a documented negative result.

**Milestone:** All candidate datasets downloaded, licensed for use (CC BY / CC BY-NC as applicable), and spot-checked; go/no-go decision recorded per source.

### Phase 1 — Weeks 3–6: Unified Dataset Construction ✅ COMPLETE

- Crop classification patches from the Bunch Piles YOLO boxes per class.
- Map every source onto the final 4-class taxonomy (unripe/underripe/ripe/overripe); exclude Bunch Piles' empty-bunch/abnormal-fruit images and Mendeley's Decayed category, which fall outside this scope.
- Merge the chosen sources into one pool; keep a source column in the manifest so each image is traceable back to its dataset (needed for the leave-one-dataset-out check in Phase 2).
- Split train/valid/test by source photo/bunch, never by crop — enforce this programmatically and verify zero leakage.
- Rebalance so unripe (the weakest class previously) has genuine source-photo diversity, not just duplicated crops.
- MPOB field images via co-supervisor Ahmad Syazwan Ramli are not a dependency for this timeline — fold them in opportunistically if they arrive, otherwise proceed without them.

**Milestone:** One unified, leakage-checked dataset with a written data-quality report: per-class counts, per-source counts, split methodology, and a zero-leakage verification output.

### Phase 2 — Weeks 7–9: Model Development & Comparison ▶ RE-OPENED (recipe v2)

- Train MobileNetV2 on the new unified dataset as the updated baseline, with class-weighted or focal loss.
- Train EfficientNet-Lite0/Lite2 as the second architecture required by Objective 2.
- Train a lightweight ViT variant (e.g. MobileViT) as the third comparison point — time-box this; literature suggests it needs more data/compute than MobileNet or EfficientNet-Lite for a mobile budget.
- Leave-one-dataset-out (LODO) generalization check on the winning architecture only: 4 extra runs, holding out Bunch Piles, then Outdoor-Tenera, then Mendeley, then Kaggle in turn, training on the other 3 each time. This is the project's primary generalization evidence, not a backup — the plan doesn't depend on MPOB field data arriving in time.
- Document hyperparameters (learning rate, frozen vs fine-tuned layers, label smoothing) per run.
- Score every model on accuracy, per-class F1, ROC-AUC, params, and raw inference time.

**Milestone:** Comparison table across ≥3 architectures plus LODO results across all 4 sources; best model selected against the ≥85% accuracy target with a written justification if the target isn't fully met.

### Phase 3 — Weeks 10–11: Mobile Optimization ◐ ONE ITEM LEFT (on-device latency)

- ✅ Convert the selected model to TensorFlow Lite; apply float16 and int8 quantization.
- ◐ Benchmark actual file size (≤100MB) and on-device inference latency (≤2s) — not Colab
  estimates. **Size demonstrated (6.1MB); latency still host-side only.**
- ✅ Re-run the lighting-robustness test (original/darkened/brightened) on the optimized model.
- ✅ Test on real 224×224 input images per NFR-5.

**Milestone:** Final `.tflite` export meeting all 5 non-functional requirements, with a benchmark report on a real Android device.

### Phase 4 — Weeks 12–14: Mobile App Development ◐ IN PROGRESS (all four screens built; phone testing outstanding)

- Build the Android app in Kotlin per the existing mockups (Home / Preview / Result pages).
- Implement capture-or-upload flow, on-device TFLite inference, and confidence-score display with the per-class bar shown in the mockup.
- Wire up the use-case flows already designed: Upload Image, Capture Image, Detect Ripeness, View Result.
- Internal functional testing and bug fixing.

**Milestone:** Working APK — end-to-end capture → on-device inference → result screen.

### Phase 5 — Weeks 15–16: Extended Evaluation & Ablation ◐ LODO DONE (v1 + v2), ABLATIONS NOT STARTED

- Deeper error analysis on the LODO results: which held-out source hurts accuracy most, and does it line up with a specific class confusion (e.g. ripe/underripe) or a specific source's collection conditions.
- Optional ablation study if time allows: with/without augmentation, with/without label smoothing, with/without segmentation preprocessing (see sub-plan below) — genuine research-style evidence for the report, not just a single accuracy number.
- Final evaluation against every functional and non-functional requirement.
- Fold in MPOB field images as a bonus real-world check only if they've arrived by this point — not a dependency for this phase.
- Write up limitations honestly (e.g. residual ripe/underripe confusion if it persists).

**Milestone:** LODO results and error analysis written up; limitations documented; MPOB test included if available.

#### Phase 5 sub-plan: segmentation-preprocessing ablation (optional, cost-gated)

Motivated by Harjanto et al. (BINUS, 2025), "Lightweight YOLOv8n Segmentation and ShuffleNetV2 Transfer Learning Integration for Oil Palm Fruit Ripeness Classification," which reports up to +6.58% accuracy from segmenting the fruit bunch out of the background before classification. Their setup differs from this project's in ways that matter and mean the finding must be re-verified here, not assumed to transfer: single source (Mendeley only), one object per image after manual curation, 2,400 images hand-annotated with segmentation masks to train YOLOv8n-Seg, and no on-device latency/size constraint (a desktop study). This project's merged corpus is 4 heterogeneous sources, ~18k images, some multi-bunch, and any added segmentation step still has to clear this project's ≤2s/≤100MB NFRs on top of classification.

Two stages, in order — do not skip to stage 2 without stage 1 showing a real lift:

1. **Cheap test, zero new annotation.** Run an off-the-shelf segmenter (e.g. SAM, or classical GrabCut) over the existing clean test set, crop to the largest foreground blob, and re-evaluate the already-trained winning Phase 2 classifier on the cropped images. If accuracy doesn't meaningfully improve, write this up as a negative/non-transferring result and stop here — no annotation investment needed.
2. **Only if stage 1 shows a real lift:** annotate a stratified subset (~300-500 images across all 4 sources and 4 classes, not the full corpus) with segmentation masks via Roboflow's polygon annotator (smart-polygon assist speeds this up), train a YOLOv8n-Seg on that subset, and re-run the Phase 2 comparison with segmentation as a preprocessing step. Bunch Piles already ships bounding boxes from its Roboflow export — a GrabCut-from-box pass can seed rough masks there instead of drawing from scratch, so manual polygon time only needs to go to the other 3 (unboxed) sources.
3. Whichever stage this stops at, benchmark the added segmentation model's own latency/size contribution against the ≤2s/≤100MB NFRs — an accuracy gain that breaks either NFR is a trade-off, not a free win, and must be reported as such.

### Phase 6 — Weeks 17–19: Report Writing & Viva Prep ⬜ NOT STARTED

- Write FYP2 report chapters: results, discussion, conclusion, future work.
- Update the viva deck with FYP2 results (replace the FYP1 preliminary MobileNetV2-only results).
- Rehearse viva, anticipate questions on the accuracy gap and how it was closed or explained.

**Milestone:** Submitted thesis draft + final viva presentation.

## 6. Tooling & Workflow

For the ML pipeline (Phases 0–3): VS Code with the Claude Code extension gives inline diffs, Jupyter notebook support, and Python debugging in one place — well suited to iterating on training scripts and experiment notebooks.

For the app (Phase 4): Android Studio remains the right tool for the Kotlin/TFLite integration; Claude Code (CLI or VS Code extension) can still help scaffold and debug alongside it.

Keep the GitHub repo as the source of truth: commit experiment configs and results per run so the architecture comparison in Phase 2 is reproducible and citable in the report.

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| MPOB field data never arrives | Low | Already priced in — the plan doesn't depend on it. LODO across all 4 public sources is the primary generalization evidence; MPOB is a bonus in Phase 5 only |
| ViT variant underperforms or overruns its time budget | Low–Medium | Time-box ViT training to a fixed window in Phase 2; report a clear underperformance as a negative result rather than chasing it |
| App development starts too late and eats into writing time | High | Begin the app skeleton in parallel during Phase 2–3 using a placeholder model, so UI work isn't blocked on final model selection |
| 85% accuracy target still not met after Phase 2 | High | Document the gap honestly with error analysis; a well-justified 78–82% with clear discussion of the ripe/underripe ambiguity is defensible in a viva — silently missing the NFR is not |

## 8. Decisions Log

> **⚠ Read §0 first.** This log is append-only and chronological. Entries dated Aug 2026
> were all measured under training recipe v1, which stopped every run before it converged
> (see the final entry). Their *relative* comparisons may survive; their *absolute*
> accuracies do not. Individual entries below carry a **[SUPERSEDED]** marker where the v2
> finding directly overturns them.


- **Taxonomy locked to 4 classes** (unripe/underripe/ripe/overripe). Bunch Piles' empty/abnormal and Mendeley's Decayed categories are excluded during merge.
- **MPOB field data is not a dependency.** The leave-one-dataset-out check across all 4 public sources is the project's primary generalization evidence; MPOB images are folded into Phase 5 opportunistically if they arrive.
- **merge_and_split.py source-imbalance bug found and fixed (Aug 2026).** The first MobileNetV2 baseline run (test accuracy 63.71%) turned out to be confounded: `assign_splits()` balanced group assignment by class count only, not by `(source, class)`. Because Bunch Piles' source photos are large groups (many crops per pile frame) while the other 3 sources are all single-image groups, and the greedy assignment sorts by group size descending and picks whichever split is furthest below target, Bunch Piles was silently funneled almost entirely into train (82-94% of every class's train split) while valid/test ended up dominated by the other sources (as little as 14% Bunch Piles) — the model was being evaluated on a different source mix than it trained on, for every class. Fixed by computing targets per `(source_dataset, class)` instead of per class, so each source keeps its own ~70/15/15 split independent of its group sizes; verified post-fix (e.g. Bunch Piles now 63%/63%/63% of underripe across train/valid/test). Zero source-photo leakage re-verified after the fix. See `data/README.md` for full detail. **All results below are from the corrected split** — the original 63.71% run and its buggy artifacts were discarded, not kept for comparison, since they were never a valid measurement.
- **MobileNetV2 FYP2 baseline, corrected split (Aug 2026).** `results/mobilenetv2_fyp2_baseline/`: test accuracy **73.73%**, ROC-AUC (OvR) 0.920 — a ~10-point jump from the confounded run, confirming the split bug was a real and substantial confound, not noise. Still an ~11-point gap to the 85% NFR. Findings:
  - **Unripe recall 0.87** (vs FYP1's 0.38) — the dataset rebuild still resolves FYP1's original diagnosed problem (SS3) even after the split fix.
  - **Underripe remains the weak point** (recall 0.39, up from 0.30 pre-fix), confused mainly with Ripe (underripe->ripe 235/698 = 34%, ripe->underripe 80/782 = 10%) — improved by the split fix but not eliminated, consistent with a genuine residual ambiguity: Mendeley's "PartiallyRipe" class dominates underripe's test images (69%) [**stale — the current `manifest_full.csv` has underripe's test images at 63% bunch_piles / 30% mendeley; the manifest was rebuilt after this entry was written (18,179 → 18,167 rows). Re-derive before citing.**] and is drawn from an explicitly *ordinal* dataset (fuzzy graduated boundary by the source's own design), not a crisp category.
  - **Lighting stability NFR holds**: clean/darkened/brightened/shadow accuracy within 2.2 points of each other (71.5-73.7%).
  - **Early, non-final signal** on the other 2 NFRs (dev machine, pre-quantization): 20.9MB model file, 108ms/image.
  - Fine-tuning (Stage 2) now clearly beats the frozen-stage result (74.4% vs 66.6% best val accuracy) — the FYP1-like fine-tuning gain that was absent in the confounded run reappeared once the split was fixed.
- **CORAL ordinal-loss experiment (Aug 2026) — negative result.** Hypothesis: since ripeness is ordinal (unripe<underripe<ripe<overripe), an ordinal-aware head/loss (CORAL: K-1 rank-consistent binary thresholds, `src/training/ordinal.py`) should reduce the underripe<->ripe confusion better than plain categorical cross-entropy. Same backbone, same two-stage recipe, same class weighting as the baseline — only the loss/head changed. Result: `results/mobilenetv2_fyp2_ordinal/`, test accuracy **66.90%** (7 points below the categorical baseline). It *did* reduce underripe->ripe confusion (34% -> 21%) and improved underripe recall (0.39 -> 0.47), directionally confirming the hypothesis — but overcorrected: ripe->underripe confusion nearly quadrupled (10% -> 34%, worse than the original problem in reverse), ripe recall collapsed (0.82 -> 0.53), and lighting stability got worse (6.1-point spread vs baseline's 2.2). **Conclusion: don't adopt CORAL as-is for EfficientNet-Lite/MobileViT runs** — the categorical baseline with class weighting + label smoothing remains the better default. Worth revisiting later with per-threshold class weighting or calibrated (non-0.5) decision thresholds if there's time, but not pursued further this session.
  - MobileViT (Phase 2's remaining required architecture) and the LODO check are not yet run.
  - Report figures for both runs (class distribution, source composition, confusion matrices, per-class metrics, lighting stress-test, training curves, underripe/ripe confusion by source) are in `results/figures/`.
- **EfficientNet-Lite0 substituted with EfficientNetB0 (Aug 2026).** Objective 2's second architecture was planned as "EfficientNet-Lite0/Lite2," but its only available pretrained weights are TensorFlow Hub's `efficientnet/lite0/feature-vector` module, which is exported in **TF1 Hub format only** — confirmed both by its Kaggle Models path (`tensorFlow1/lite0-feature-vector`) and empirically: constructing it with `trainable=True` raises `ValueError: Setting hub.KerasLayer.trainable = True is unsupported when loading from the TF1 Hub format`. It was also tested directly against this project's TF 2.20/Keras 3 stack (`hub.KerasLayer` in a Keras 3 Functional model, a custom layer wrapping `hub.load()`, and the legacy `tf_keras` package) — all three hit real Keras-3/TF-Hub incompatibilities independent of the fine-tuning limitation (e.g. `tf_keras` itself failed to reload its own saved checkpoint). Since every other Phase 2 architecture uses two-stage (frozen → fine-tune) training, a frozen-only Lite0 would make the comparison table unfair — any accuracy gap would be confounded with training methodology, not architecture. Decision: train `tf.keras.applications.EfficientNetB0` instead (`src/training/train_efficientnetb0.py`) — Keras-3 native, supports the identical two-stage recipe, same lightweight-CNN family, no plumbing risk. One implementation detail: Keras's built-in EfficientNetB0 bakes its own Rescaling+Normalization internally and expects raw `[0, 255]` input, unlike MobileNetV2's `[-1, 1]` — handled locally in the training script (undoes the shared pipeline's `[-1, 1]` scaling before the EfficientNetB0 base) rather than changing `dataset.py`, which other runs depend on. Report/thesis should refer to this architecture as "EfficientNetB0," not "EfficientNet-Lite0."
- **EfficientNetB0 FYP2 baseline (Aug 2026).** `results/efficientnetb0_fyp2_baseline/`: test accuracy **73.18%**, ROC-AUC (OvR) 0.912 — statistically on par with the MobileNetV2 baseline (73.73%, ROC-AUC 0.920), not a clear improvement despite the identical recipe. Per-class comparison against MobileNetV2:
  - Unripe and overripe recall are near-identical to MobileNetV2 (0.873 vs 0.874, 0.888 vs 0.882).
  - Ripe recall is slightly better (0.838 vs 0.823), but **underripe recall is worse (0.350 vs 0.391)** — EfficientNetB0 does not resolve the underripe/ripe ambiguity that's this project's main residual weakness; if anything it's marginally more confused there.
  - Cost profile is worse on every axis that matters for mobile deployment: 4.05M params vs 2.26M, 27.8MB vs 20.9MB (pre-quantization), 129.6ms vs 108ms/image (dev CPU) — nearly double the params for no accuracy gain.
  - Lighting stability holds: 2.06-point spread across brightened/darkened/shadow, comparable to MobileNetV2's 2.2-point spread.
  - **Conclusion: MobileNetV2 remains the front-runner** pending MobileViT and the LODO check — EfficientNetB0 doesn't earn its extra size/latency cost with this recipe.
- **Block-level fine-tuning + run-to-run variance finding (Aug 2026).** Fine-tuning depth was respecified from a raw layer count to a block count (`src/training/finetune.py`, `--finetune-blocks`, default K=3 following Suharjito et al.). Motivation: `--finetune-layers 30` meant ~19.5% of MobileNetV2 (154 layers) but only ~12.6% of EfficientNetB0 (237 layers), so the two baselines were fine-tuned to different proportional depths while appearing to share a hyperparameter. Both architectures expose exactly 16 blocks, so K is directly comparable. Applied to both architectures simultaneously — changing it for one only would have confounded architecture with recipe.
  - **EfficientNetB0 (`results/efficientnetb0_fyp2_blockft/`): 73.07%** vs 73.18% under the layer scheme, despite unfreezing 46 layers instead of 30. Underripe recall 0.361 vs 0.350. **The depth-asymmetry hypothesis — that EfficientNetB0 underperformed because it was fine-tuned proportionally shallower — is not supported.**
  - **MobileNetV2 (`results/mobilenetv2_fyp2_blockft/`): 74.17%** vs 73.73%, macro F1 0.751 vs 0.731, underripe recall **0.596 vs 0.391**.
  - **The MobileNetV2 pair differs by exactly one unfrozen layer** (`block_13_project_BN`; old = `layers[-30:]`, new = `layers[-29:]`, verified by set difference). A one-layer change cannot cause a 20-point underripe-recall swing, so this is **run-to-run variance, not an effect of the scheme**: the tf.data pipeline uses `num_parallel_calls=AUTOTUNE` with reshuffling, so runs are not bit-reproducible even at a fixed `seed=42`, and the unstable underripe/ripe boundary lets runs settle at materially different per-class trade-offs at near-identical overall accuracy.
  - **Methodological consequence: every single-run comparison in this project so far is within noise**, including the MobileNetV2-vs-EfficientNetB0 architecture comparison (73.73% vs 73.18%) and the apparent block-vs-layer improvement. Architecture selection and any accuracy claim in the report need multi-seed runs with mean ± std, not single runs. The CORAL result (66.90%, ~7 points down) is large enough to likely survive this, but the ~0.5-point differences are not interpretable.
- **BatchNorm `training=False` confound found and corrected (Aug 2026).** The first attempt at the block-level refactor also rewrote `base(inputs, training=finetune)` (with `finetune=False` at build time) into plain `base(inputs)`, un-pinning the flag and letting BatchNorm re-estimate statistics from training batches during Stage 2 instead of using frozen ImageNet moving statistics. MobileNetV2 dropped to 72.30% test accuracy while **validation accuracy stayed essentially unchanged (0.7391 vs 0.7398)** — the validation metric did not surface the problem at all. Both training scripts now pin `training=False` with an explanatory comment. The confounded run is preserved as `results/mobilenetv2_fyp2_bn_unfrozen/` with a `run_note` in its `config.json`; since its unfreeze depth was within one layer of the baseline, it doubles as an incidental ablation suggesting BN-unfreezing costs ~1.4 points — though per the variance finding above, that too is a single run and should not be quoted as a firm number.
- **All 4 candidate datasets merged (Aug 2026).** `data/processed/manifest_full.csv`: 18,179 images, 9,485 unique source photos, zero source-photo leakage verified across train/valid/test (70/15/15) — Phase 0's sourcing milestone and Phase 1's unified-dataset milestone are both satisfied. Three of the four source datasets turned out to have their own baked-in leakage or duplication that would have silently reintroduced the FYP1 bug if merged naively: Bunch Piles' Roboflow re-export duplicated 552 frames across its train/valid/test folders (and reordered its class ids relative to the source paper); Outdoor-Tenera had 2 byte-identical image pairs crossing its own FFBtrain/FFBtest split; the Kaggle supplement had 6 duplicate pairs within-class. All neutralized by grouping/deduping on true content identity (MD5 hash or de-suffixed frame id) before splitting; all 4 sources were also cross-checked against each other and share zero images. See `data/README.md` for full per-source detail.

- **[SUPERSEDED by recipe v2 — see the final entry] Multi-seed replication campaign (Aug 2026) — the architecture comparison is now decided, and it reverses the earlier call.** Motivated by the variance finding above: every prior single-run comparison was within noise, so nothing could be concluded from them. Ran 3 seeds (42/43/44) per architecture under an identical recipe (block-level fine-tuning K=3, `src/training/run_multiseed.py`), all 6 runs on one consistent backend, aggregated by `src/training/aggregate_seeds.py` into `results/multiseed_summary.md` + `results/multiseed_per_run.csv`.

  | Metric | EfficientNetB0 | MobileNetV2 |
  |---|---|---|
  | Test accuracy (%) | **76.45 ± 0.90** | 73.61 ± 0.59 |
  | ROC-AUC (OvR) | 0.9352 ± 0.0027 | 0.9145 ± 0.0023 |
  | Macro F1 | 0.7620 ± 0.0121 | 0.7383 ± 0.0093 |
  | Params / size / latency | 4.05M / 33.9MB / 160ms | 2.26M / 20.9MB / 143ms |

  - **The 2.84-point gap exceeds the combined run-to-run spread (±1.07)** — Welch's t p=0.014, indicative only at n=3. This is the project's first architecture claim that survives its own variance check.
  - **Supersedes the "MobileNetV2 remains the front-runner" conclusion** from the EfficientNetB0 baseline entry above. That call rested on single runs (73.73% vs 73.18%) which are now known to be uninterpretable. EfficientNetB0 costs ~1.8x the params and ~13MB more, but both sit far under the 100MB NFR, so the cost is affordable and it now earns it.
  - **Underripe remains the weak point for *both* architectures and is not architecture-specific**: recall 0.494 ± 0.072 (EfficientNetB0) vs 0.482 ± 0.059 (MobileNetV2) — statistically indistinguishable, and with by far the largest per-class standard deviation. Its instability across seeds (up to a 17-point swing) is strong evidence the underripe/ripe boundary is a genuine labelling/visual ambiguity rather than a modelling failure — directly relevant to the §9 open question.
  - **Accuracy gap to the 85% NFR is now ~8.6 points** (from ~11), the best position the project has reached.
  - **Methodological rule going forward: no accuracy claim from a single run.** Report mean ± std over >= 3 seeds. Per-class recalls on the underripe/ripe boundary are especially unsafe to quote from one run.

- **GPU training environment (Aug 2026).** TF 2.20 has no Windows-native CUDA support (TensorFlow dropped it after 2.10), so the RTX 4050 was unusable from the Windows env despite a working driver. Set up WSL2 (Ubuntu 24.04 via `wsl --import`; the Store install path stalled) with `tensorflow[and-cuda]==2.20.0` and every other library version pinned to match the Windows environment, so the two differ only in compute backend. Measured 128ms/step vs 400ms/step on CPU (**3.1x**; capped well below GPU potential because JPEG decode is CPU-bound in the tf.data pipeline — deliberately *not* "fixed" with caching, since changing the pipeline would alter the recipe). Wall-clock ~9-13 min/run vs ~58 min. Runs now record their backend in `config.json` (`src/training/runtime_info.py`) and `aggregate_seeds.py` refuses to pool across backends unless explicitly overridden.
  - `evaluate.py`'s latency measurement is now pinned to CPU (loading its own CPU-resident copy of the model), so the cost column stays comparable once runs come from different backends. Note WSL CPU latency runs higher than Windows-native for the same model (143ms vs 105ms for MobileNetV2) — compare latency only within a campaign, and treat the Phase 3 on-device benchmark as the real NFR gate.

- **[SUPERSEDED by recipe v2 — refit required, see the final entry] Per-class threshold calibration (Aug 2026) — modest real gain for EfficientNetB0, none for MobileNetV2.** `src/training/calibrate_thresholds.py` rescales predicted class probabilities by per-class weights (argmax over p*w, standard prior correction), with **w fitted by coordinate ascent on the validation split only and applied unchanged to test**. No retraining. Outputs in `results/calibration/` + `results/calibration_summary.md`; original run dirs left untouched.

  | Test metric (n=3) | EfficientNetB0 base -> calibrated | MobileNetV2 base -> calibrated |
  |---|---|---|
  | Accuracy (%) | 76.44 ± 0.82 -> **77.96 ± 0.82** (**+1.52**) | 73.65 ± 0.59 -> 73.61 ± 0.17 (**-0.04**) |
  | Macro F1 | 0.762 -> **0.787** (+0.025) | 0.739 -> 0.740 (+0.002) |
  | Recall underripe | 0.494 ± 0.070 -> **0.652 ± 0.022** (+0.159) | 0.484 -> 0.518 (+0.034) |
  | Recall unripe | 0.926 -> 0.808 (-0.118) | 0.841 -> 0.826 (-0.015) |

  - **The gain generalises for EfficientNetB0**: validation gain +1.73 pts vs test gain +1.52 pts, so only ~0.2 pts was overfitting to the fitting split. For MobileNetV2 the validation gain (+0.43) vanished entirely on test (-0.04) — its probability estimates carry less usable ranking information, a genuine architecture difference beyond raw accuracy.
  - **Underripe recall improves substantially and becomes far more stable** (std 0.070 -> 0.022): repositioning the decision boundary fixes some of what the loss function could not, which is the deliberate, calibrated version of what the CORAL experiment attempted and overshot.
  - **Corrects an earlier projection.** The estimate that lifting underripe recall to ~0.80 would yield ~84% assumed the other classes held constant. They do not: the recovered underripe recall is paid for mostly out of **unripe** (-0.118), revealing an unripe<->underripe confusion in addition to the known underripe<->ripe one. Repositioning the operating point alone buys ~1.5 pts, not ~8. **The remaining gap needs better features, not a better threshold.**
  - Widens the architecture gap: EfficientNetB0 calibrated 77.96% vs MobileNetV2 73.61%, ~4.35 pts. Gap to the 85% NFR now ~7 pts.
  - Calibration is a post-processing step on the softmax output — it costs no parameters and no latency, so it carries through TFLite export in Phase 3 as a stored weight vector applied after inference.

- **LODO generalization check complete (Aug 2026) — the project's primary generalization evidence.** Run on EfficientNetB0 (the multi-seed winner, not MobileNetV2 as originally planned), 4 folds x 3 seeds = 12 runs, `src/training/run_lodo.py` + `src/data/make_lodo_manifests.py`, aggregated by `src/training/aggregate_lodo.py` into `results/lodo_summary.md` + `results/lodo_per_run.csv`. Fold manifests verified for zero leakage: held-out source appears only in test, no path overlap, no source photo straddling train/valid.

  | Held out | n | Classes | Test accuracy (%) | Macro F1 |
  |---|---:|---:|---|---|
  | bunch_piles | 3 | 4 | 51.88 ± 2.91 | 0.508 ± 0.022 |
  | kaggle | 3 | 4 | 56.56 ± 0.36 | 0.590 ± 0.010 |
  | malaysia_outdoor | 3 | 3 | 62.98 ± 4.19 | 0.677 ± 0.035 |
  | mendeley_ordinal | 3 | 4 | 52.40 ± 0.20 | 0.518 ± 0.005 |
  | **Mean** | | | **55.96** | |

  - **Generalization gap: 20.49 points** (76.45% main split -> 55.96% LODO mean). The main-split number is measured on unseen *photos* from sources the model also trained on; LODO measures an entirely unseen source. **The gap is the honest estimate of what does not survive a change of camera, site, and labelling convention — and it should be the headline of the generalization chapter, not the 76.45%.**
  - **The underripe failure is a cross-source labelling-convention mismatch, not a modelling failure.** Held-out Mendeley underripe recall is **0.128 ± 0.013** — trained on the other 3 sources, the model recognises Mendeley's "PartiallyRipe" images ~13% of the time, reproducibly across all 3 seeds. The same fold simultaneously produces the *highest* validation accuracy on the other 3 sources (0.831-0.841, above the main split's 0.777): removing Mendeley makes everything else easier, testing on Mendeley fails. Underripe is the worst class in every fold that contains it (0.199 / 0.406 / 0.128). **This is direct evidence for the §9 open question** — the ambiguity is a labelling-convention difference between sources, now with a number attached.
  - **Domain shift dominates, not training-set size.** The kaggle and malaysia folds keep ~14.7-15.1k training images (vs 4.7k for bunch_piles) and still land at 56-63%, so the confound flagged in `run_lodo.py` is real but secondary.
  - **Per-fold variance differs enormously and single-fold numbers must not be quoted bare**: kaggle ± 0.36 and mendeley ± 0.20 are near-deterministic, while malaysia_outdoor is ± 4.19 (only 416 test images, and with no underripe class every underripe prediction is automatically wrong, so small shifts in that tendency swing accuracy hard) and bunch_piles ± 2.91.
  - Two evaluation bugs were found and fixed while setting this up, both of which would have silently corrupted the numbers: `classification_report`/`confusion_matrix` now pin `labels=` to all 4 classes (malaysia_outdoor has no underripe, which otherwise produced a 3x3 matrix against 4 tick labels and crashed the plot), and `aggregate_lodo.py` treats zero-support classes as undefined rather than as recall 0.000 — sklearn's own macro average divides by the full label set, reporting macro F1 0.478 for the malaysia fold where the true macro over present classes is 0.637.

- **MobileViT is not usable on this stack; FastViT substituted (Sep 2026).** Objective 2's third
  architecture — "a lightweight ViT variant (e.g. MobileViT)" — was blocked not by time or effort but
  by a pretrained-weights ecosystem gap. Five independent routes were tried and each failed for its
  own distinct, reproducible reason:

  | Route | Failure |
  |---|---|
  | `keras.applications` | Ships no MobileViT at all |
  | TensorFlow Hub | TF1-Hub-format only; rejects `trainable=True` (the same trap that forced EfficientNet-Lite0 → EfficientNetB0) |
  | HuggingFace `transformers` (TF port) | Requires `tf_keras`, which already failed here on checkpoint reload |
  | `keras_cv_attention_models` (kecam) | `transformer_pre_process` applies a raw TF op to a KerasTensor — illegal in Keras 3 — at the patch-folding step that *defines* MobileViT |
  | kecam's own workaround | Its error hint advises `TF_USE_LEGACY_KERAS=1`, which lands back on route 3 |

  - **The failure is broad, not MobileViT-specific.** A sweep of 15 lightweight transformers in kecam
    (`spike_lightweight_vit.py`) found **only FastViT_T8 and FastViT_T12 build at all** under Keras 3.
    MobileViT V2, LeViT, EfficientViT (M0/M1/B0), EdgeNeXt, TinyViT and even GhostNetV2 all die with
    the identical KerasTensor error. This is a package-wide Keras-3 incompatibility, not a local
    misconfiguration — worth stating plainly in the report rather than as an excuse.
  - **FastViT_T8 adopted as the third architecture.** FastViT (Vasu et al., Apple, 2023 — *verify the
    exact venue before citing; it is ICCV or CVPR 2023, not confirmed here*) is a mobile-targeted
    hybrid CNN-transformer, so it answers "a lightweight ViT variant" — MobileViT was the plan's
    *example*, not its requirement. T8's 3.20M params / 27.0MB sits between MobileNetV2 (2.26M /
    20.9MB) and EfficientNetB0 (4.05M / 33.9MB), keeping all three inside one comparable cost band.
    It cleared a six-gate feasibility spike before any training time was committed: build with
    ImageNet weights, block-structured layer names, pinned `training=False` BN, `train_on_batch` in
    both stages, and **both `.weights.h5` and `.keras` save/reload round-trips at max delta 0.00e+00** —
    the last being precisely the gate that killed `tf_keras`.
  - **`evaluate.py` could not load FastViT models — found and fixed before the campaign, not after.**
    kecam's custom layers (`ChannelAffine` et al.) only enter Keras's serialization registry when kecam
    is imported, so `load_model()` failed with `Could not locate class 'ChannelAffine'`. A 1-epoch smoke
    run surfaced this; without it, all six runs would have trained successfully and then failed at
    evaluation. Fixed by `src/training/custom_layers.py`, a guarded import mirroring the existing
    `import ordinal` trick — guarded because kecam is installed in the WSL GPU venv but not the Windows
    env, and MobileNetV2/EfficientNetB0 evaluation must not depend on it.
  - **K is no longer comparable across architectures, so depth is run as an ablation.** The block-level
    scheme worked because MobileNetV2 and EfficientNetB0 both expose exactly 16 blocks. FastViT_T8
    exposes 10 (4 stacks of depth 2/2/4/2), so the same K is a different proportional depth. Measured
    through `unfreeze_last_blocks()` itself:

    | Architecture | Blocks | K | Layers unfrozen | % of layers |
    |---|---:|---:|---:|---:|
    | MobileNetV2 | 16 | 3 | 29/154 | 18.8% |
    | EfficientNetB0 | 16 | 3 | 46/237 | 19.4% |
    | FastViT_T8 | 10 | 2 | 38/246 | 15.4% |
    | FastViT_T8 | 10 | 3 | 69/246 | 28.0% |

    **No integer K matches the CNNs' ~19% — it falls between FastViT's K=2 and K=3.** Rather than pick
    one and hope, both are run at 3 seeds each (6 runs, `--tag gpu_k2` / `gpu_k3`), so the CNN depth is
    *bracketed* and FastViT's sensitivity to fine-tuning depth is measured rather than assumed. An
    initial estimate of 20.3% for K=2 was wrong: stack transitions add downsample layers belonging to
    no `stackN_blockM` group, so K=2 unfreezes 38 layers, not stack4's 50.
  - **`aggregate_seeds.py` hardened accordingly.** Both FastViT campaigns report
    `architecture: "FastViT_T8"`, so a glob catching both would have averaged two recipes into one row
    and silently destroyed the ablation. It now refuses to pool one architecture's runs across
    differing `finetune_blocks` (override: `--allow-mixed-depth`), mirroring the existing compute-backend
    guard, and `--pattern` now accepts multiple globs so the 3-architecture table can combine
    `*_blockft_gpu_seed*` with `fastvit_*_gpu_k2_seed*` while leaving the k3 runs out.
  - **Result: Objective 2's three-architecture comparison, at FastViT's nearer-matched depth (K=2).**
    `results/multiseed_summary.md`, 3 seeds each, one GPU backend:

    | Metric | EfficientNetB0 | FastViT_T8 (K=2) | MobileNetV2 |
    |---|---|---|---|
    | Test accuracy (%) | 76.45 ± 0.90 | 75.95 ± 0.43 | 73.61 ± 0.59 |
    | ROC-AUC (OvR) | 0.9352 ± 0.0027 | 0.9315 ± 0.0010 | 0.9145 ± 0.0023 |
    | Macro F1 | 0.7620 ± 0.0121 | 0.7543 ± 0.0059 | 0.7383 ± 0.0093 |
    | Lighting spread (pts) | 3.06 ± 0.55 | **1.90 ± 0.44** | 2.98 ± 1.28 |
    | Params / size / latency | 4.05M / 33.9MB / 160.3ms | 3.20M / 27.0MB / 160.6ms | 2.26M / 20.9MB / 143.4ms |

    At this depth FastViT and EfficientNetB0 are **not distinguishable** (0.50 pts apart against a
    combined spread of ±1.33), while both beat MobileNetV2 beyond its noise. FastViT reaches parity
    with the incumbent winner at 79% of its parameters and 80% of its file size, and has the most
    stable behaviour under the lighting stress test of any architecture tried.
  - **Result: fine-tuning depth matters more than architecture choice.** `results/fastvit_depth_k2_*`
    vs `results/fastvit_depth_k3_*`:

    | FastViT_T8 | Accuracy (%) | Macro F1 | Recall underripe | Lighting spread | Latency (ms) |
    |---|---|---|---|---|---|
    | K=2 (15.4% depth) | 75.95 ± 0.43 | 0.7543 ± 0.0059 | 0.4217 ± 0.0204 | 1.90 ± 0.44 | 160.6 ± 14.8 |
    | K=3 (28.0% depth) | **78.72 ± 0.22** | **0.7860 ± 0.0026** | **0.5093 ± 0.0097** | **1.47 ± 0.53** | **150.5 ± 2.6** |

    **+2.77 pts from one extra block**, against a combined spread of ±0.65 — larger than any
    architecture gap this project has measured, and an improvement on every axis at once (accuracy,
    macro F1, underripe recall, lighting stability, and latency). 78.72% is the best result the project
    has produced, narrowing the gap to the 85% NFR to ~6.3 points before calibration.
  - **Depth control (Sep 2026): FastViT's advantage is architectural, not a depth artifact.** Because
    FastViT's K=3 sits at 28.0% depth while both CNNs were only ever run at ~19%, its 2.27-point margin
    could have been a recipe effect. EfficientNetB0 was therefore re-run at K=4 (25.7%, the nearest
    available match to 28.0%; its grid steps 19.4% → 25.7% → 31.6%), 3 seeds,
    `results/effnet_depth_k4_*`:

    | Architecture | K | Depth | Test accuracy (%) | Δ vs its own K=3 |
    |---|---:|---:|---|---|
    | EfficientNetB0 | 3 | 19.4% | 76.45 ± 0.90 | — |
    | EfficientNetB0 | 4 | 25.7% | 76.03 ± 1.12 | **−0.42 (flat, within noise)** |
    | FastViT_T8 | 2 | 15.4% | 75.95 ± 0.43 | — |
    | FastViT_T8 | 3 | 28.0% | **78.72 ± 0.22** | **+2.77 (real)** |

    **EfficientNetB0 does not benefit from deeper fine-tuning; FastViT does.** At comparable depth
    (25.7% vs 28.0%) FastViT leads EfficientNetB0 by 2.69 points against a combined spread of ±1.34,
    so the margin survives the control. The finding is itself interesting and worth discussing: the
    hybrid transformer gains substantially from unfreezing more of its backbone while the CNN is
    saturated, consistent with its later blocks carrying more ImageNet-specific representation that
    benefits from adaptation. A residual 2.3-point depth mismatch remains and should be stated —
    exact matching is not achievable across architectures with different block counts.
  - **Calibration transfers to FastViT, but weakly — and it is now the project's best model.**
    `results/calibration_summary.md`, refitted across all three architectures (validation-only fit,
    applied unchanged to test):

    | Architecture | Baseline | Calibrated | Δ test | Δ validation |
    |---|---|---|---|---|
    | FastViT_T8 (K=3) | 78.72 ± 0.22 | **79.17 ± 0.26** | +0.45 | +2.09 |
    | EfficientNetB0 (K=3) | 76.45 ± 0.87 | 77.86 ± 0.69 | +1.41 | +1.73 |
    | MobileNetV2 (K=3) | 73.63 ± 0.60 | 73.39 ± 0.51 | −0.24 | +0.39 |

    **79.17% is the best result the project has produced, and the gap to the 85% NFR is now ~5.8
    points** (from ~8.6 before this campaign). But note the caveat: FastViT's calibration gain is
    mostly *not* generalising — +2.09 on the split it was fitted on versus +0.45 on test, so 1.64
    points were fitted noise. EfficientNetB0's gain generalises far better (+1.73 → +1.41). The
    mechanism is the familiar one: underripe recall rises sharply (0.510 → 0.681) but is paid for out
    of ripe (0.863 → 0.743), and for FastViT that trade is close to break-even. Calibration should
    therefore be reported as a genuine but small benefit for FastViT, not as a second headline.
  - **HEADLINE TABLE for the report — each architecture at its best setting.** Now defensible because
    the depth control above shows EfficientNetB0 does not improve with more depth, so this is not
    flattering FastViT with a recipe the CNNs were denied. `results/multiseed_best_summary.md`:

    | Metric | FastViT_T8 (K=3) | EfficientNetB0 (K=3) | MobileNetV2 (K=3) |
    |---|---|---|---|
    | Test accuracy (%) | **78.72 ± 0.22** | 76.45 ± 0.90 | 73.61 ± 0.59 |
    | ROC-AUC (OvR) | **0.9425 ± 0.0006** | 0.9352 ± 0.0027 | 0.9145 ± 0.0023 |
    | Macro F1 | **0.7860 ± 0.0026** | 0.7620 ± 0.0121 | 0.7383 ± 0.0093 |
    | Recall - underripe | **0.509 ± 0.010** | 0.494 ± 0.072 | 0.482 ± 0.059 |
    | Recall - ripe | **0.863 ± 0.017** | 0.758 ± 0.045 | 0.780 ± 0.018 |
    | Recall - unripe | 0.883 ± 0.017 | **0.927 ± 0.005** | 0.841 ± 0.017 |
    | Lighting spread (pts) | **1.47 ± 0.53** | 3.06 ± 0.55 | 2.98 ± 1.28 |
    | Params / size / latency | 3.20M / 30.2MB / **150.5ms** | 4.05M / 33.9MB / 160.3ms | 2.26M / 20.9MB / 143.4ms |

    FastViT_T8 wins on every metric except unripe recall, where EfficientNetB0 keeps a clear lead
    (0.927 vs 0.883). Two results deserve emphasis beyond raw accuracy: its **underripe recall is not
    only best but an order of magnitude more stable across seeds** (±0.010 vs ±0.072 and ±0.059), which
    matters because underripe instability has been this project's most persistent measurement problem;
    and its **lighting spread is roughly half** either CNN's, which is the NFR most directly tied to
    field deployment. Note FastViT's saved file grows 27.0MB → 30.2MB between K=2 and K=3 (more
    trainable variables carry more optimizer state into the checkpoint); the deployment-relevant figure
    is the Phase 3 TFLite export, not this.
  - **Note on reproducibility.** Re-running calibration on the *same* EfficientNetB0 checkpoints gave
    +1.41 where the August run gave +1.52 (76.44 → 77.96 vs 76.45 → 77.86). Nothing changed but the
    inference pass, so this is GPU nondeterminism in prediction propagating through the fitted weights.
    It is small relative to the reported spreads, but it means calibration deltas should be quoted to
    one decimal at most, and the summary regenerated rather than cited from an older copy.
  - **Reporting note.** Even with FastViT delivered, the report should state that *MobileViT specifically*
    was not evaluated and why, since the plan named it. The evidence above makes that a documented
    ecosystem constraint rather than a gap in the work.


- **Recipe v1 → v2: every prior run was under-trained (Sep 2026). CONFIRMED by a 9-run
  campaign.** This is the largest single result the project has
  produced, and it invalidates more of the log above than any earlier correction.

  **Diagnosis.** Reading the saved `history.json` of all 15 GPU runs (3 architectures plus
  the K=2 and K=4 depth arms):

  | Indicator | Count |
  |---|---|
  | Stage 2 (fine-tuning) early-stopped | **0 / 15** |
  | Stage 1 (frozen) early-stopped | 6 / 15 |
  | `val_loss` still falling at the final epoch | 13 / 15 |
  | best `val_accuracy` **at** the final epoch | 11 / 15 |

  Fine-tuning ran to its 10-epoch cap in every single run. That `EarlyStopping(patience=4)`
  *did* fire in Stage 1 for 6 runs is what makes this conclusive rather than a suspected dead
  callback: the frozen head genuinely plateaus and the callback correctly ends it, while the
  fine-tuning stage never got the chance. Cause: the `10 + 10` epoch cap and the `1e-5`
  fine-tuning LR were both carried over unexamined from the FYP1 MobileNetV2 notebook.

  **Fix.** `src/training/schedule.py` now holds the recipe in one place, and all three
  training scripts drive their two stages through its `run_two_stage()` — so "all three
  architectures share an identical recipe" is enforced by construction rather than by keeping
  three copies of the training loop in sync. Recipe v2: epoch caps 30/40 (50/40 for the
  campaign), EarlyStopping patience 4 → 8, fine-tuning LR 1e-5 → 1e-4 under cosine decay.
  Runs record `recipe_version`, and `aggregate_seeds.py` refuses to pool across versions for
  the same reason it refuses to pool across compute backends.

  **Probe result** (`results/fastvit_fyp2_v2_gpu_k3_seed42/`) — single seed, FastViT_T8 K=3,
  same seed / data / splits as its v1 counterpart, only the recipe differs:

  | Metric | v1 | v2 | Δ |
  |---|---|---|---|
  | Test accuracy | 78.69% | **93.98%** | **+15.29** |
  | ROC-AUC (OvR) | 0.9429 | 0.9915 | +0.049 |
  | Macro F1 | 0.784 | 0.941 | +0.157 |
  | Recall — underripe | 0.501 | **0.880** | **+0.379** |
  | Recall — ripe | 0.844 | 0.972 | +0.128 |
  | Lighting spread | 0.88 pts | 0.48 pts | halved |

  Stage 2 early-stopped at epoch 32/40 (`converged_finetune: true`) — the first fine-tuning
  stage in this project to end by converging rather than by running out of budget.

  **Verified before acceptance**, because a +15-point jump makes split integrity load-bearing
  in a way it was not at 78%: zero source-photo groups span more than one split (0 of 9,473);
  no image path appears in more than one split; no path carries conflicting class labels; test
  accuracy sits just *below* validation (93.98 vs 94.72), the healthy ordering.

  **Data defect found in passing — not leakage, but fix it.** 1,481 manifest rows are exact
  duplicates of another row (same path, class, source photo, and split), all from the
  Bunch Piles crop pipeline. They never cross splits, so they cannot leak, but ~27 test images
  are double-counted (~1% of the test set) and some training images are double-weighted.
  Fix in `src/data/crop_bunch_piles.py` before the final numbers are quoted.

  **CONFIRMED (2026-09-01, 23:21).** The 9-run campaign completed — 3 architectures × 3
  seeds, K=3, frozen cap 50, `results/multiseed_v2_summary.md`, zero failures in 347 min.
  **This is the report's headline table.**

  | Metric | **FastViT_T8** | EfficientNetB0 | MobileNetV2 |
  |---|---|---|---|
  | Test accuracy (%) | **94.55 ± 0.31** | 91.26 ± 0.39 | 89.62 ± 0.15 |
  | *(recipe v1, for contrast)* | *78.72 ± 0.22* | *76.45 ± 0.90* | *73.61 ± 0.59* |
  | ROC-AUC (OvR) | **0.9916 ± 0.0002** | 0.9861 ± 0.0012 | 0.9800 ± 0.0018 |
  | Macro F1 | **0.9467 ± 0.0035** | 0.9137 ± 0.0042 | 0.8983 ± 0.0015 |
  | Recall — unripe | 0.940 ± 0.005 | **0.943 ± 0.002** | 0.940 ± 0.011 |
  | Recall — underripe | **0.903 ± 0.016** | 0.796 ± 0.011 | 0.793 ± 0.015 |
  | Recall — ripe | **0.966 ± 0.002** | 0.955 ± 0.007 | 0.921 ± 0.006 |
  | Recall — overripe | **0.976 ± 0.009** | 0.962 ± 0.006 | 0.936 ± 0.010 |
  | Lighting spread (pts) | **1.02 ± 0.43** | 2.45 ± 0.37 | 5.19 ± 0.88 |
  | Params / size / latency | 3.20M / 30.2MB / 157ms | 4.05M / 33.9MB / 161ms | 2.26M / 20.9MB / 142ms |

  - **The gains are uniform: +15.83, +14.81, +16.01 points.** A training-budget defect
    lifting three structurally different backbones by the same amount is the signature of a
    real recipe effect, not an artifact peculiar to one model. **The v1 ranking is preserved**,
    so the architecture conclusion never depended on the broken recipe — only the absolute
    numbers moved.
  - **Both gaps clear their noise check.** FastViT − EfficientNetB0 = 3.29 pts against a
    combined spread of ±0.50 (Welch t=−11.44, p=0.000); EfficientNetB0 − MobileNetV2 = 1.64
    pts against ±0.42 (t=6.83, p=0.010). Both p-values are badly underpowered at n=3 and must
    be quoted as indicative only, but the margins clear the spreads comfortably.
  - **All three architectures now clear the ≥85% accuracy NFR on the main split**, and all
    clear the ≤100MB and ≤2s budgets (pre-quantization, dev CPU).
  - **FastViT wins on every metric except unripe recall**, where the three are tied within
    noise. Its underripe recall (0.903) is 10 points clear of both CNNs, and its lighting
    spread (1.02 pts) is less than half EfficientNetB0's and a fifth of MobileNetV2's.
  - **MobileNetV2's lighting spread got *worse* under v2** (2.98 → 5.19 pts) — the only metric
    that regressed for any architecture. It is the weakest on the NFR most directly tied to
    field deployment, which further weakens its case as the deployment candidate despite being
    the smallest and fastest.
  - **Residual caveat: 3 of the 9 runs hit the 40-epoch fine-tuning cap** rather than
    early-stopping (`efficientnetb0_seed43`, `fastvit_seed43`, `mobilenetv2_seed42`) — one per
    architecture, so the shortfall is symmetric and does not favour any of them. Runs are
    finishing at 29–40 epochs, so the cap is marginal for all three; a cap of ~80 would let
    every run converge and would likely add a little to all three equally. Separately, FastViT
    hit the frozen cap 50/50 in **all three** seeds — systematic for that architecture, though
    its Stage 2 converged in 2 of 3. Both are worth stating in the report as known residual
    limits rather than being quietly ignored.

  **Campaign provenance.** Tag `v2_gpu_f50_k3`, launched and completed 2026-09-01: 9 runs,
  347 min, 0 failures. The single-seed probe that triggered this work is kept separately as
  `results/fastvit_fyp2_v2_gpu_k3_seed42/` — it used a frozen cap of 30, so it is a different
  config and is deliberately *not* pooled with the campaign. No single-seed result enters the
  report; the project's multi-seed rule stands.

  **What this supersedes, and what it does not:**

  1. **Every absolute accuracy in §8 dated Aug 2026** — floors, not ceilings.
  2. **The headline three-architecture table** — void until the v2 campaign re-runs it.
  3. **"Underripe is a genuine labelling/visual ambiguity rather than a modelling failure"** —
     on the *main split* this is substantially falsified: recall went 0.501 → 0.880 with no
     data change at all. **But the LODO evidence is a different claim and still stands**: the
     held-out-Mendeley underripe recall of 0.128 is about a labelling-convention mismatch
     *between sources*, which no amount of training on the other three sources can fix. Keep
     these two apart in the report — the main-split story was under-training, the cross-source
     story is convention mismatch.
  4. **The K=4 depth control** ("EfficientNetB0 does not benefit from deeper fine-tuning") —
     measured entirely under v1, where unfreezing more layers meant spreading more parameters
     across the same 10 epochs at 1e-5. Extra capacity with no budget to adapt it will look
     flat whether or not it is useless. Must be re-measured before it goes in the report.
  5. **Per-class threshold calibration** — its mechanism was buying underripe recall at ripe's
     expense. If v2 delivers that recall in training, calibration may be unnecessary or
     actively harmful. Refit on the v2 checkpoints before reporting any calibrated number.
  6. **The 20.49-point LODO generalization gap** — the main-split figure it is measured
     against moved from 76.45% to (provisionally) ~94%. **Re-measuring LODO under v2 is now
     the single most important open question in the project**: whether a properly-trained
     model generalizes better across sources, or merely fits the training sources' quirks
     harder, is genuinely unknown and is exactly where an examiner will probe a 94% headline.
     **→ ANSWERED 2026-09-02 — see the next entry. It is the second reading: the model fits
     the training sources harder. The gap widened to 42.44 points.**

- **LODO re-measured under recipe v2 (Sep 2026) — this is the project's headline
  generalization result, and it is a negative one.** Run on FastViT_T8 (the v2 multi-seed
  winner), 4 folds x 3 seeds = 12 runs, tag `lodo_v2`, `src/training/run_lodo.py` aggregated
  by `src/training/aggregate_lodo.py` into `results/lodo_v2_summary.md` +
  `results/lodo_v2_per_run.csv`. Same fold manifests as the v1 campaign, so the splits are
  unchanged and zero-leakage-verified; only the training recipe and the architecture differ.

  | Held-out source | n | Classes | Test accuracy (%) | Macro F1 (present classes) |
  |---|---:|---:|---|---|
  | bunch_piles | 3 | 4 | 40.98 ± 2.33 | 0.4128 ± 0.0194 |
  | kaggle | 3 | 4 | 50.80 ± 0.78 | 0.4972 ± 0.0051 |
  | malaysia_outdoor | 3 | 3 | 60.90 ± 0.37 | 0.6503 ± 0.0125 |
  | mendeley_ordinal | 3 | 4 | 55.75 ± 0.40 | 0.5474 ± 0.0073 |
  | **Mean across folds** | | | **52.11** | |

  - **The generalization gap widened from 20.49 to 42.44 points.** Main split went
    76.45% -> 94.55% (+18.10) while the LODO mean went 55.96% -> 52.11% (-3.85). Recipe v2
    bought accuracy on unseen *photos* from sources in the training pool and gave a little
    back on an entirely unseen *source*.
  - **This is not another under-training artifact, and that matters.** Under v1, 0 of 15 GPU
    runs early-stopped. Under v2, **10 of 12 fine-tuning stages early-stopped**; only
    `kaggle_seed43` and `malaysia_outdoor_seed43` hit the 40-epoch cap, and 3 of 12 hit the
    frozen cap of 50 (`bunch_piles_seed43/44`, `mendeley_ordinal_seed44`). The low
    cross-source number therefore cannot be explained away by training budget the way the v1
    main-split numbers could.
  - **Confound to state explicitly in the report: v1 LODO ran EfficientNetB0, v2 LODO runs
    FastViT_T8.** Recipe and architecture changed together, so the -3.85 delta is not
    attributable to the recipe alone. What *is* safe to claim is the direction — cross-source
    accuracy did not improve, under any attribution. A clean decomposition would need
    EfficientNetB0 under v2 LODO (12 more runs, ~6h GPU); cost-gate it against Phase 3/4,
    which are on the critical path and not started.
  - **The failure mode is convention-fitting, not uniform degradation.** Under v2 the model
    concentrates its cross-source predictions on ripe/overripe: on held-out kaggle, ripe
    recall 0.659 -> 0.921 and overripe 0.785 -> 0.912 while underripe **collapsed 0.406 ->
    0.067**; on held-out mendeley, ripe 0.621 -> 0.817 and overripe 0.802 -> 0.823 while
    underripe went 0.128 -> 0.063. Held-out bunch_piles moves the other way (underripe
    0.199 -> 0.547, unripe 0.418 -> 0.140), which is consistent with the same mechanism
    rather than against it — each fold snaps to whichever convention the remaining three
    sources agree on.
  - **This confirms, rather than overturns, the split diagnosis in the previous entry.**
    Main-split underripe weakness was under-training and v2 fixed it (0.501 -> 0.903).
    Cross-source underripe weakness is a labelling-convention mismatch between sources, and
    v2 made it worse. Keep the two claims apart in the report — they are now backed by
    different campaigns pointing in opposite directions.
  - **Reporting consequence.** The thesis cannot present 94.55% unqualified. Either LODO
    becomes the headline with the main split as a ceiling, or both are presented as bounding
    a range. This is now a live decision for supervisors, not a caveat — see §9.

- **DECIDED (2026-09-02) — the thesis headlines the main split and reports the gap
  explicitly.** Resolving the reporting question raised in the entry above and in §9. Of the
  three framings on the table, the chosen one is closest to (a) but carries (c)'s substance:
  **94.55% ± 0.31 is the headline result, and the 42.44-point generalization gap is presented
  as a quantified part of the result rather than as a footnote.** Decision taken by the
  student; to be put to Dr. Shahreeza as a recorded decision with its reasoning, not as an
  open question.

  **Required wording — the NFR claim must be scoped, every time the headline appears.**
  The examiner's question is "your requirement is ≥85% and your field estimate is 52%, so is
  the requirement met?" and the report must answer it rather than leave it standing:

  > The ≥85% accuracy requirement is met on the held-out test split (94.55% ± 0.31). It is
  > **not** claimed to hold under source shift: leave-one-dataset-out accuracy is 52.11%, a
  > 42.44-point generalization gap, and that figure is the honest estimate of performance on
  > a mill this model has never seen.

  Stating the limit unprompted is what makes the 94.55% headline credible under examination;
  burying it is what makes it look evasive. The NFR was written against a held-out test
  split, so 94.55% clears it legitimately — the scoping sentence is what keeps that claim
  honest rather than what weakens it.

  **What this settles downstream.** Phase 3/4 now only have to demonstrate on-device
  performance of the main-split model; no extra artefact is needed to evidence cross-source
  robustness. This is the cheaper path and it protects the Phase 3/4/6 calendar. It also
  keeps the duplicate-row defect (below) a measure-and-document item rather than a
  load-bearing one.

- **Phase 3 TFLite conversion spike (2026-09-02) — the deployability risk is CLOSED, and
  full-integer quantization is ruled out for the selected model.** All three v2 architectures
  at seed 42, four quantization modes each, `src/training/export_tflite.py`, results in
  `results/tflite_export_summary.md` + per-run `results/<run>/tflite/`. Run in the `fyp-gpu`
  WSL venv (TF 2.20 / Keras 3.15.1 / Python 3.12.3).

  **Headline: 12 of 12 conversions succeeded builtin-only. Not one needed the Flex delegate.**
  The concern that FastViT_T8 — a kecam hybrid CNN-transformer built from custom layers — might
  only convert via `SELECT_TF_OPS`, and so drag a slice of the TF runtime into the APK, does not
  materialise. The architecture selected in the v2 campaign is deployable as-is, and Phase 4 can
  target a stock TFLite interpreter.

  | Model | Mode | Size (MB) | Acc (n=200) | Agreement | Latency (ms) |
  |---|---|---:|---:|---:|---:|
  | FastViT_T8 | float32 | 12.1 | 0.920 | 0.990 | 47.7 |
  | FastViT_T8 | **float16** | **6.1** | **0.925** | 0.970 | **47.2** |
  | FastViT_T8 | dynamic_int8 | 3.3 | 0.930 | 0.970 | 114.3 |
  | FastViT_T8 | full_int8 | 3.5 | **0.260** | 0.265 | 136.7 |
  | EfficientNetB0 | float32 | 15.3 | 0.910 | 1.000 | 22.4 |
  | EfficientNetB0 | float16 | 7.7 | 0.910 | 1.000 | 23.6 |
  | EfficientNetB0 | dynamic_int8 | 4.3 | 0.910 | 0.980 | 30.3 |
  | EfficientNetB0 | full_int8 | 4.7 | **0.645** | 0.660 | 20.3 |
  | MobileNetV2 | float32 | 8.5 | 0.910 | 0.995 | 12.1 |
  | MobileNetV2 | float16 | 4.3 | 0.905 | 1.000 | 12.0 |
  | MobileNetV2 | dynamic_int8 | 2.4 | 0.920 | 0.975 | 16.9 |
  | MobileNetV2 | full_int8 | 2.6 | 0.915 | 0.960 | 13.4 |

  - **Post-training full-integer int8 is not usable for the two stronger architectures, and the
    failure tracks architecture, not size.** FastViT collapses to 0.260 — chance on 4 classes —
    and EfficientNetB0 to 0.645, while MobileNetV2 is untouched at 0.915. This is the textbook
    pattern: FastViT's normalization/attention activations and EfficientNet's SiLU + squeeze-excite
    blocks both have dynamic ranges that per-tensor int8 scales cannot represent, whereas
    MobileNetV2's ReLU6 design was built to quantize. **Quantization robustness runs *inversely*
    to the accuracy ranking** — the least accurate architecture is the most deployable. Worth a
    paragraph in the results chapter; it is a real architecture-level trade-off, not a footnote.
  - **These modes converted cleanly and are the smallest files produced.** A benchmark measuring
    only size and latency would have selected `full_int8` as the winner for all three. It is the
    fidelity check that catches it, which is why `export_tflite.py` compares every converted model
    against its Keras parent rather than treating conversion success as a pass.
  - **Recommended deployment artefact: FastViT_T8 float16 — 6.1 MB, 47.2 ms, no accuracy cost.**
    Half the float32 size for no latency penalty. Size is nowhere near binding (6.1 MB against a
    100 MB budget), so there is no reason to accept dynamic_int8's 2.4x latency penalty to save a
    further 2.8 MB.
  - **The pre-quantization dev-CPU latency column in the v2 headline table does not survive
    conversion and should be dropped rather than corrected.** It reported 142 / 161 / 157 ms for
    MobileNetV2 / EfficientNetB0 / FastViT — a 1.13x spread. Converted, the same three measure
    12.1 / 22.4 / 47.7 ms — a **3.9x spread in a different order**. FastViT is the most accurate
    and the slowest by a wide margin once deployed; the Keras figures concealed that entirely.
  - **Environment finding worth recording.** Both commonly-documented conversion entry points fail
    on this stack: `from_saved_model` over a `model.export()` archive dies in the trackable walk
    ("this `__dict__` descriptor does not support `_DictWrapper` objects", a Python 3.12
    `inspect.getattr_static` interaction) and `from_keras_model` surfaces the same fault as
    "'NoneType' object is not callable". Only `from_concrete_functions` works, for all three
    architectures — so this is a TF/Keras/Python constraint, not a kecam or FastViT problem. The
    script probes all three routes and records which one worked.
  - **Separately: `.keras` checkpoints are not portable between this project's two environments.**
    WSL has Keras 3.15.1, Windows 3.14.1, and the newer one serializes `GlorotUniform` with
    `input_axes`/`output_axes` that the older rejects. Every checkpoint-loading script is therefore
    WSL-only. "TF 2.20" alone does not pin the environment; `requirements.txt` still does not pin
    anything, and does not list `keras_cv_attention_models` at all.

  **What this does NOT establish.** Seed 42 only, not the 3-seed protocol. Fidelity is measured on
  a 200-image sample sized to compare TFLite against Keras — on that sample the three architectures
  sit within one point of each other while the full test set separates them by five, so **the
  accuracy column here must not be read as an architecture comparison**. Latency is single-threaded
  x86 dev CPU: int8 commonly wins on ARM via XNNPACK/NNAPI, so the float16-over-dynamic_int8 call
  could reverse on a phone. **The <=2s NFR is still not demonstrated on device** — that remains the
  open half of Phase 3, and `docs/report_limitations.md` L4 needs rewriting now that the conversion
  half is done.

- **Duplicate-row defect measured (2026-09-03) — the headline moves by 0.04 points, and the
  item is CLOSED as documented rather than fixed.** `src/training/measure_duplicate_impact.py`,
  results in `results/duplicate_row_impact.md` + `.csv`. Run in the `fyp-gpu` WSL venv.

  The defect was recorded above as a measure-and-document item on the strength of "~1% of the
  test set is double-counted." That bounded the exposure but did not say which way or how far
  the numbers move, which is not enough to put in front of an examiner. It is now measured.

  Method: duplicate rows point at the same image, so inference runs once per unique test image
  and each metric is computed twice over identical predictions — once with rows weighted by
  their manifest multiplicity (reproducing what was published), once with each image counted
  one. **The as-scored pass returns 94.55% ± 0.31 for FastViT_T8, exactly the published
  headline**, which is what certifies the measurement rather than merely asserting it.

  | Quantity | As published | Deduplicated | Delta |
  |---|---:|---:|---:|
  | FastViT_T8 test accuracy (n=3) | **94.55% ± 0.31** | **94.51% ± 0.33** | **-0.04 pts** |
  | EfficientNetB0 (seed 42) | 91.55% | 91.47% | -0.08 pts |
  | MobileNetV2 (seed 42) | 89.64% | 89.61% | -0.03 pts |

  - **Largest shift across any run measured: 0.08 points** — an order of magnitude below the
    ±0.31 seed-to-seed spread the headline already carries, and far below the 9.55-point margin
    over the ≥85% NFR. No reported conclusion changes.
  - **The bias direction is knowable, not a coin flip.** Every delta is negative because the
    27 duplicated test images are ones the models classify well (92-100% accuracy on that
    subset, against 90-95% overall), so double-counting inflates rather than deflates. The
    report should state the direction, not just the magnitude.
  - **Test-split exposure is smaller than the raw 1,481 figure suggests.** Only 27 of those
    rows are in test (0.99% of 2,722, affecting 27 distinct images, spread across all four
    classes: unripe 7, underripe 4, ripe 9, overripe 7). The other 1,454 are in train (1,422)
    and valid (32).
  - **What is measured is the reporting error, not the training error.** The 1,422 duplicated
    training rows double-weight some training images; quantifying that would require retraining
    on a deduplicated manifest. Out of scope, and stated as such in the output — the claim this
    supports is about the numbers as reported, not the models as trained.

  **Consequence: do not re-run the campaign for this.** Fixing `src/data/crop_bunch_piles.py`
  and retraining would cost the full 9-run campaign to move the headline by 0.04 points. The
  defect is instead reported as a measured, bounded, direction-known limitation. The fix still
  belongs in the pipeline before any *future* campaign, and the earlier instruction to fix it
  "before the final numbers are quoted" is superseded by this measurement.

- **Report figures regenerated onto v2 (2026-09-03).** Every figure in `results/figures/` was
  a recipe-v1 figure until this date, while the report headlines v2 numbers. Figures 01-08 had
  been generated from `mobilenetv2_fyp2_baseline` + `mobilenetv2_fyp2_ordinal` (retired v1 CPU
  runs) and figures 09/11 from `multiseed_per_run.csv` / `lodo_per_run.csv` (the v1 CSVs), in
  both cases because those were the *script defaults*. Regenerated against the v2 campaign;
  three defects in the figure scripts were fixed in the process, each of which had produced a
  wrong-but-plausible figure rather than an error:

  1. `make_campaign_figures.py` hardcoded **EfficientNetB0** as the source of figure 11's
     main-split reference line — correct for v1, where EfficientNetB0 was the architecture
     LODO ran on, and silently wrong for v2, where it is FastViT_T8. It would have drawn a
     94.55%-vs-52.11% gap as ~39 points against the wrong baseline. The architecture is now
     derived from the LODO CSV's own `architecture` column and printed at run time.
  2. The same script keyed folds on bare source names (`bunch_piles`), while v2 fold names
     carry a recipe prefix (`v2_bunch_piles`). Because the lookup was a `defaultdict`, the
     mismatch would have rendered figure 11 with empty bars instead of raising. Prefixes are
     now stripped on load and missing folds raise.
  3. `make_report_figures.py` keyed colour and label on a `kind` of "baseline" vs "ordinal" —
     the FYP1-era categorical-vs-CORAL question. All three v2 architectures are categorical, so
     all three collapsed onto one key: figure 04 would have plotted one architecture's metrics
     three times, all labelled "Categorical (baseline)". Runs are now keyed by run name and
     labelled from their own `config.json`. Bar widths, fixed at 0.35 for exactly two runs, now
     scale with run count.

  Figure 10 (calibration) was held back at this point, because the only calibration artifacts on
  disk had been fitted on v1 checkpoints. It is now opt-in via an explicit `--calibration` flag
  rather than a default, so it cannot be silently rebuilt from stale inputs, and it was
  regenerated later the same day once the v2 refit landed — see the entry below. A fourth latent
  bug surfaced there: panel (a)'s y-limit was hardcoded to 92, sized for v1's best calibrated run
  of 79.17%, and clipped every v2 bar and its value label.

- **Calibration refit onto v2 (2026-09-03) — the technique is now redundant, and the headline
  stays uncalibrated. Item CLOSED.** `results/calibration_v2_summary.md`,
  `results/calibration_v2/`, figure 10 regenerated. Inference only; nothing retrained.

  The v1 calibration numbers were fitted on under-trained checkpoints and were void. Refitting
  on the v2 checkpoints answers whether per-class threshold calibration still earns a place in
  the report. **The baselines it recovers — 91.27 / 94.55 / 89.62 — reproduce the published v2
  campaign, which is what validates the refit.**

  | Architecture | Test gain, v1 | Test gain, v2 | Validation gain, v2 |
  |---|---:|---:|---:|
  | EfficientNetB0 | +1.41 | **+0.55** | +0.84 |
  | FastViT_T8 (selected) | +0.45 | **+0.21** | +0.60 |
  | MobileNetV2 | **-0.24** | **+0.29** | +0.46 |

  - **For the selected model the gain is smaller than the noise it is measured against.**
    FastViT_T8 gains +0.21 points against a seed-to-seed spread of ±0.31. Calibration can no
    longer be distinguished from run-to-run variance on the model actually being deployed.
  - **The mechanism has gone quiet because the problem it treated is gone.** Under v1 it bought
    underripe recall violently — FastViT +0.171 underripe paid for with -0.120 ripe. Under v2
    the same procedure moves underripe by +0.021 and ripe by -0.004, because underripe already
    sits at 0.904 out of training rather than 0.510. There is almost nothing left to buy.
  - **It is no longer *harmful* anywhere**, which is itself a change: MobileNetV2 went from
    -0.24 under v1 to +0.29 under v2. The earlier concern that calibration might now actively
    hurt did not materialise. It simply stopped mattering.
  - **Roughly two-thirds of the fitted gain still fails to transfer** (FastViT +0.60 validation
    against +0.21 test), the same qualitative finding as v1 at a smaller scale.

  **Decision: the headline stays 94.55% ± 0.31, uncalibrated.** Quoting the calibrated 94.76%
  would claim a gain inside the seed spread, and would oblige the deployed pipeline to carry
  per-class threshold weights through the TFLite export for a benefit that cannot be
  distinguished from noise. Calibration is reported in the discussion as a v1-era remedy that
  the corrected training recipe made redundant — which supports the project's central finding
  rather than competing with it. `docs/report_limitations.md` L6 needs rewriting on this basis;
  its v1 numbers are void.

- **Depth control re-run under v2 (2026-09-03) — the v1 control was an under-training
  artifact, and roughly 45% of the apparent architecture gap is fine-tuning depth. L3b must be
  rewritten.** `results/effnet_depth_k4_v2_summary.md`, runs
  `efficientnetb0_fyp2_blockft_v2_gpu_f50_k4_seed{42,43,44}`. 121 min, 3/3 succeeded.

  EfficientNetB0 at K=4 (25.7% of layers unfrozen), the closest available match to FastViT_T8's
  K=3 (28.0%), everything else identical to the v2 campaign.

  | Configuration | Layers unfrozen | Test accuracy | Per-seed |
  |---|---:|---:|---|
  | EfficientNetB0 K=3 | 19.4% | 91.26 ± 0.39 | 91.55 / 91.40 / 90.82 |
  | **EfficientNetB0 K=4** | **25.7%** | **92.73 ± 0.37** | 92.73 / 92.36 / 93.09 |
  | FastViT_T8 K=3 (selected) | 28.0% | 94.55 ± 0.31 | — |

  - **The v1 finding is falsified.** Under v1, EfficientNetB0 at K=4 scored -0.42 points against
    K=3 — within noise — which is what licensed L3b's claim that depth sensitivity was specific
    to FastViT and that the architecture advantage survived the control. Under v2 the same
    control gives **+1.47 points**, with per-seed ranges that do not overlap at all
    (90.82-91.55 vs 92.36-93.09). The v1 result was an artifact of stopping every run before
    convergence: extra unfrozen layers had no epoch budget in which to adapt, so of course they
    bought nothing. Re-running the control was the right call.
  - **The architecture conclusion survives, but the margin nearly halves.** At comparable depth
    FastViT_T8 leads EfficientNetB0 by **1.82 points** (94.55 vs 92.73) against a combined
    spread of ±0.68 — still outside the noise, so FastViT is still genuinely the better
    architecture here. But the naive same-K comparison put the gap at 3.29 points, so
    **about 45% of the apparent architecture advantage was fine-tuning depth, not architecture.**
    Any sentence in the report attributing the full 3.29 points to the architecture is wrong.
  - **Depth also improves what the report cares about most.** EfficientNetB0's underripe recall
    — the project's persistent weak class — rises 0.796 → 0.831 at K=4, and its lighting spread
    tightens from 2.45 to 1.33 points. Depth is not a pure accuracy knob.
  - **Cost side, for the trade-off table:** K=4 raises pre-quantization size 33.9 → 38.4 MB and
    training time 31.9 → 40.4 min/run. Both irrelevant against the 100 MB NFR.

  **Residual limitation, to state rather than smooth over.** Depth is still not exactly matched:
  25.7% against 28.0%, about 2.3 points apart, and EfficientNetB0 is evidently still gaining as
  depth increases rather than having saturated. A K=5 run (31.6%) would bracket FastViT from
  above and is the natural next control if time allows — without it, the honest claim is that
  FastViT leads at the closest matchable depth, not at an exactly matched one.


- **Phase 4 started (2026-09-24) — the Android project exists, builds, and its preprocessing is
  now proven against Python rather than asserted.** `app/` was a placeholder README; it is now a
  Gradle module at the repository root (`settings.gradle.kts`, wrapper committed) with the model
  layer, the theme from `app/DESIGN_HANDOFF.md`, the Home screen, and both parity test suites.
  What is built and what is not is listed in `app/README.md`; Camera, Review and Results are
  routed but still land on a placeholder. Three build decisions worth recording because they are
  not recoverable from the code:
  - **Jetpack Compose, not XML views.** The handoff is a dp/sp spec with six custom components and
    deliberately no Material look; Compose expresses it directly, and the buttons are built from
    primitives so no Material default leaks into a screen the report shows.
  - **`minSdk` 26.** Google Fonts publishes Fraunces and Instrument Sans only as variable fonts,
    and setting a weight on the `wght` axis needs API 26. Below that, Fraunces falls back to its
    default instance, which is Thin — every heading in the app would render hairline. Both fonts
    are bundled (OFL, licences in `app/licenses/`) because the manifest requests no `INTERNET`
    permission, so a downloadable-font provider could never resolve. That missing permission is
    deliberate and is the checkable form of the report's offline claim.
  - **Gradle must run on JDK 17 on this machine.** Android Studio 2026.1.4 ships JBR 25, which
    Gradle 8.10.2 refuses to run on. Set the Gradle JDK to Temurin 17 in Studio, or export
    `JAVA_HOME` for command-line builds. AGP 8.7.3 / Gradle 8.10.2 / Kotlin 2.0.21 are pinned in
    `gradle/libs.versions.toml` so the APK the report benchmarks can be rebuilt later.

  **The result that matters, and it is not the screens.** `ModelParityTest` runs on a device and
  compares the app's outputs to Python's on the same 20 stratified test-split fixtures. Feeding
  the exact float32 tensors Python fed, the app reproduces its probabilities to **within 1e-4** —
  the interpreter wiring, buffer order, output shape and label order are all correct. Starting
  from the same JPEGs and running the app's own decode → resize → scale, the probabilities move
  by up to **0.0424**, with the **predicted class unchanged on all 20**. The JVM test
  `ResizeParityTest` separately confirms the Kotlin bilinear resize matches `tf.image.resize`
  (half-pixel centres, no antialias) to <1e-3 at 800 sampled positions, so the resize is not the
  source of that 0.0424.

  **The source is the JPEG decoder, and it is now measured rather than argued.** Android cannot
  call `tf.io.decode_jpeg`. Decoding the same fixtures with Pillow on the desktop — an
  independent libjpeg configuration — gives pixels up to **16 of 255** apart from TensorFlow's
  (mean 1.8) and a worst probability shift of **0.042418**, which is the *same* worst case the
  device produced, on the same image, to seven decimal places. So the remaining difference is the
  decoder, not the app. `make_app_bundle.py` now performs this comparison as part of bundling and
  writes it into `reference.json` (`decoder_spread`), deriving the instrumented test's jpg
  tolerance from the measurement (currently 0.064) instead of the 0.02 that was previously
  assumed. The argmax assertion is the one that must not bend; wrong preprocessing — wrong value
  range, a centre crop, swapped channels — is off by whole units, not hundredths.

  **What this does not yet establish.** Decoder agreement was measured on 20 images, not on the
  2,722-image test split, so the accuracy cost of deploying with a non-TensorFlow decoder is
  bounded anecdotally rather than quantified. Re-scoring the test split with Pillow decoding is a
  few minutes of compute and would turn it into a number, in the same spirit as the duplicate-row
  measurement; it has not been run. Nothing on-device is a latency measurement either: the
  instrumented run was an x86_64 emulator, which says nothing about a phone, and the ≤2 s
  requirement still needs the real device benchmark that Phase 3 owes (see L4).

- **Phase 4 screens complete (2026-09-24), same day as the scaffold.** Camera, Review and Results
  are built on top of the morning's model layer, and the flow in `app/DESIGN_HANDOFF.md` runs end
  to end on an Android 35 emulator: capture or upload, then on-device inference, then the verdict
  with its per-stage chart. Picking the bundled fixture `10_ripe.jpg` through the upload path gives
  **Ripe, 88.9%** where Python gives **88.7%** on the same file — the 0.2-point gap is the decoder
  difference documented above, reaching the screen exactly as predicted. The close-call notice was
  observed firing on its own on a nonsense photograph whose top two scores landed 0.3 points apart.
  Three changes to the drawn design, all in `app/DESIGN_NOTES.md` with reasoning:
  - **The Analysing screen loses its determinate progress bar.** It was specified as 0-100 with a
    percentage; inference is tens of milliseconds, so the percentage would be fiction. It is now an
    indeterminate spinner held for a **400 ms floor** so the step registers rather than flickers.
  - **Two layouts were made proportional rather than absolute.** The handoff gives dp from a
    360 x 800 reference. Review's 640 dp photo area left a dead band on a 952 dp-tall phone and now
    takes the space above the buttons; the camera hint pill is anchored 20 dp below the framing
    guide, which is where its absolute 530 dp puts it on the reference and nowhere else.
  - **Review's title gained a gradient scrim.** White text on a sunlit bunch is otherwise illegible,
    which only became visible once a real photograph was behind it.

  **One finding that is not a UI matter and should reach the supervisor.** The framing guide covers
  roughly a third of the previewed frame, and the contract feeds the model the **whole** frame
  squashed to 224 x 224 — while every training image is a tight crop around a single bunch. A user
  who obeys the hint therefore hands the model a smaller, more background-heavy bunch than anything
  in training, which is a scale and context shift the parity fixtures cannot reveal, because they
  are already crops. Three ways out: leave it as specified, crop the capture to the brackets before
  the resize, or widen the brackets to the full frame and change the hint copy. Choosing between
  them needs field photographs taken both ways on a phone and scored, not an argument from first
  principles. **This is now the most likely reason for a gap between the 94.55% test accuracy and
  what the app does in a plantation.**

  Still outstanding for Phase 4: nothing has been seen on real hardware. The torch cannot be tested
  on an emulator at all (no flash unit is reported, so the toggle hides itself), the low-light
  thresholds in `LumaMeter` are provisional and uncalibrated, and the emulator's latency figure is
  meaningless for the ≤ 2 s requirement. All three need one session with a phone.

- **Camera-framing accuracy cost measured (2026-09-24) — the app's own capture geometry costs
  45.7 points on the full test split.** `src/training/check_framing_shift.py` closes the finding
  flagged in the previous entry rather than leaving it as prose. Method: take each of the 2,722
  test-split images, simulate the crop-then-resize the app actually performs — a viewfinder box
  at a given aspect ratio and area fraction, background filled two ways (a blurred continuation
  of the crop, or flat grey, as a synthetic stand-in for real plantation background) — and re-run
  the exported `model.tflite` on the result. `baseline` (unmodified crops) reproduces **94.16%**,
  close to but not identical to the 94.55% ± 0.31 multi-seed headline (this is one exported
  float16 model, not the seed average — the ~0.4-point gap is expected and not itself a finding).

  | Condition | Aspect | Area fraction | Accuracy | Δ vs baseline |
  |---|---|---:|---:|---:|
  | Baseline (full crop) | — | 1.00 | 94.16% | — |
  | App's actual geometry | 9:20 | 0.25 | **48.46%** | **-45.7 pts** |
  | Same aspect, full frame | 9:20 | 1.00 | 81.41% | -12.7 pts |
  | Squarer viewfinder, full frame | 3:4 | 1.00 | 91.62% | -2.5 pts |
  | Square viewfinder, full frame | 1:1 | 1.00 | 93.57% | -0.6 pts |
  | Square viewfinder, app's area | 1:1 | 0.25 | 46.33% | -47.8 pts |

  **Scale, not aspect, is the dominant cost.** Holding area fraction at 1.0 and varying aspect
  alone costs between -0.6 and -12.7 points depending how far from square the box is — a real
  but comparatively small effect a squarer viewfinder would fix. But at the app's actual ~25%
  area fraction, aspect stops mattering: 9:20 and 1:1 both land around -46 to -48 points. A
  squarer framing guide alone would *not* fix this; the guide would still need to cover most of
  the frame. **Unripe and underripe recall collapse hardest** (0.938→0.390, 0.904→0.312) while
  ripe and overripe are comparatively robust (0.958→0.673, 0.971→0.554) — the effect lands
  squarest on the project's already-weakest class boundary, not evenly across classes.

  **What this settles and what it doesn't.** It rules out "leave the framing guide as specified"
  as an option among the three named in the previous entry — the cost is too large to ship
  un-mitigated. It does not yet choose between the other two (crop the capture to the brackets
  before resizing, vs. widen the brackets to most of the frame and change the hint copy), because
  both would converge toward the "full frame" rows above and this experiment doesn't distinguish
  their remaining difference. The background in every non-baseline row is synthetic — a blurred
  self-continuation or flat grey, not a real plantation backdrop — so the measured drop is a
  faithful bound on the scale/aspect effect alone; real background clutter could make it worse
  or the model could be more robust to it than to the synthetic fill. Field photographs, taken
  both ways on a phone, remain the confirmation step this was meant to justify running.

- **Fix chosen and built (2026-09-25): the capture is cropped to the framing brackets.** Of the
  three options the previous entry left open, **crop the capture to the brackets before the
  resize** was taken, and the deciding argument is that it does not depend on the user obeying a
  hint: whatever they frame, the app cuts to the box they were aiming through, so the model's
  input is fixed by code rather than by photographic discipline. Widening the brackets and
  rewriting the hint was the alternative; it fails exactly when a user ignores the instruction,
  which is the case the field version of this problem is made of. The measurement also killed the
  intuition that a squarer viewfinder was the answer — at the app's ~25% area fraction a 1:1
  framing loses **47.8** points, marginally *worse* than the tall one, because scale rather than
  shape is what breaks it.

  **How it works.** `CameraScreen` measures the `FramingGuide` composable and the preview surface
  in the same coordinate space and passes the brackets' bounds up with the JPEG as a
  `FramingRegion` (`app/src/main/java/.../model/FramingRegion.kt`); `MainViewModel.acceptCapture`
  crops the decoded frame to it before anything else sees the photo, so **Review shows exactly
  what the model is given** — the invariant that docstring already claimed is now also true of the
  framing. The geometry does one non-obvious thing worth recording: it first cuts the decoded frame
  to the preview's *aspect* and only then applies the brackets, because `PreviewView` is
  FILL_CENTER and an in-memory `takePicture` may hand back the full sensor frame with its viewport
  attached only as metadata. That makes the mapping correct whether or not CameraX already applied
  the viewport crop, instead of resting on an assumption about which. The activity is
  portrait-locked, so no rotation case arises. Gallery uploads are deliberately **not** cropped:
  the app did not frame them and has nothing to say about where the subject sits.

  **Verification.** `app/src/test/java/.../model/FramingRegionTest.kt` covers the mapping on the
  JVM (reference geometry, a wider-than-preview sensor frame, aspect preservation, an unmeasured
  preview, a degenerate region, and brackets overhanging a short preview); `:app:testDebugUnitTest`
  and `:app:assembleDebug` both pass. What that does *not* establish is the accuracy recovery
  itself — see the caveat below.

  **What it is worth, measured at the shipped geometry rather than interpolated.** The brackets are
  280 x 360 dp, i.e. **7:9**, which is not one of the aspects in the table above, so that exact
  condition was scored on the full test split rather than read off the neighbouring rows
  (`results/framing_shift_brackets/`, 2026-09-25):

  | Condition | Accuracy | Δ vs baseline |
  |---|---:|---:|
  | Baseline (unmodified crops) | 94.16% | — |
  | **After the crop (7:9, bunch filling the brackets)** | **92.47%** | **-1.69 pts** |
  | Before the crop (9:20 @ 0.25, as built) | 48.46% | -45.70 pts |

  So the crop recovers **44.0 of the 45.7 points** under simulation. Per-class, the residual cost
  falls almost entirely on underripe (recall 0.904 → 0.834); unripe and ripe are unchanged or
  marginally better (0.938 → 0.948, 0.958 → 0.960) and overripe slips 0.971 → 0.960. Mean top
  probability drops 0.805 → 0.761, i.e. the model is slightly less certain even where it stays
  right — worth knowing because the close-call notice keys off that margin. One detail to quote
  accurately: the achieved area fraction is **0.841**, not 1.0, because a 7:9 canvas cannot hold a
  roughly square bunch crop at full area. That is arguably the more realistic figure anyway — it
  represents a user who fills the brackets well but not perfectly.

  **What is still not established, and the phrasing matters.** This is the same simulator that
  found the defect, with the same synthetic backgrounds: it measures the *scale and aspect* change
  the crop makes, not the app. No photograph taken through the shipped build has been scored, and
  **field photographs could not be obtained**, so the confirmation step the previous entry called
  for is unavailable and the report must say so rather than imply end-to-end verification. Two
  residual risks follow and are written into `docs/report_limitations.md` L8: the synthetic
  backgrounds may not behave like real plantation clutter, and a bunch that overflows the brackets
  is now *clipped* where previously it was merely diluted.

- **Lighting robustness re-measured through the converted model (2026-09-25) — NFR-4 now holds
  for the artefact the app ships, and the conversion's true accuracy cost is -0.44 points, not
  the +1.0 the spike's sample suggested.** `src/training/check_tflite_lighting.py`, results in
  `results/tflite_lighting_summary.md` + `results/<run>/tflite/lighting_metrics__float16.csv`.
  FastViT_T8 float16, seed 42, full test split (2,722 images per condition).

  Phase 3's milestone asked for the lighting test to be re-run "on the optimized model", and it
  never had been: every robustness number in `results/` came from a Keras checkpoint, while the
  app ships a float16 export. The concern was specific rather than procedural — float16 halves
  every weight's mantissa, and `darkened` pushes activations into the bottom of the value range
  where relative precision loss is largest.

  | Condition | TFLite float16 | Keras, same seed | Δ (points) |
  |---|---:|---:|---:|
  | clean | 0.9412 | 0.9456 | **-0.44** |
  | darkened | 0.9379 | 0.9372 | +0.07 |
  | brightened | 0.9276 | 0.9306 | -0.29 |
  | shadow | 0.9357 | 0.9390 | -0.33 |

  - **The precision concern does not materialise, and `darkened` is the condition that moves
    *least*.** TFLite spread is 1.36 points against the same seed's Keras 1.51 — the converted
    model is, if anything, marginally flatter across conditions. Every condition clears the >=85%
    NFR with more than seven points of margin. NFR-4 is now demonstrated on the deployed artefact
    rather than on its Keras parent.
  - **The conversion's accuracy cost is now a full-split number and it points the other way from
    the spike.** `export_tflite.py` measured float16 at 0.925 against Keras 0.915 on 200 images —
    a one-point *gain*. On all 2,722 it is 0.9412 against 0.9456, a **0.44-point loss**. This is
    exactly the small-sample caveat recorded in the spike's entry, now closed with a real figure.
    The direction matters for the report: the deployed model is slightly *worse* than the trained
    one, and that is the honest way to write it.
  - **The headline does not move.** It remains the three-seed campaign figure (94.55% ± 0.31) for
    the trained model; 94.12% is a property of the deployment artefact at one seed, and belongs in
    the Phase 3 / deployment discussion, not in place of the headline.
  - Per-class recall under conversion tracks the Keras pattern — underripe stays the weak class
    (0.877-0.903 across conditions) and no class collapses in any lighting condition.

  **What this does not establish.** Seed 42 only, float16 only, and host-side: it is the *model's*
  robustness through the converted graph, not the app's, and it says nothing about latency. The
  on-device benchmark remains the open half of Phase 3 (`DeviceLatencyBenchmark`, staged and
  compiling; see `docs/device_session_checklist.md`).

## 9. Open Questions for Supervisors

> **Note (updated 2026-09-02):** the headline-framing question below is **DECIDED** — the
> thesis headlines the main split (94.55%) and reports the 42.44-point gap explicitly. See the
> final entry of §8 for the decision and the required scoping wording. It is left in this
> section, struck through, so the reasoning stays visible; raise it with Dr. Shahreeza as a
> recorded decision, not as an open question.
>
> **Note (updated 2026-09-02):** the v2 campaign landed and the first question below is now
> **moot for the main split** — all three architectures clear ≥85% there (FastViT_T8 94.55%).
> It is **live and sharper for the cross-source result**, where the LODO mean is 52.11%,
> i.e. 33 points *below* the NFR rather than 6. Re-frame it that way before raising it.


- Is a documented, well-justified accuracy below 85% acceptable if the ripe/underripe ambiguity is shown to be a genuine labeling/visual limitation rather than a modelling failure?
- Why did EfficientNetB0 gain ~3.3 points moving from the CPU runs (73.07-73.18%, n=2) to the GPU runs (76.45% ± 0.90, n=3), while MobileNetV2 barely moved (73.73-74.17% CPU vs 73.61% ± 0.59 GPU)? With no multi-seed CPU baseline, a genuine backend/numerics interaction cannot be separated from two unlucky CPU runs. It does not affect the architecture conclusion, which is drawn entirely within the GPU campaign — but it should be stated as a limitation rather than glossed over. Cheap resolution if time allows: 3 CPU seeds of EfficientNetB0.
- **~~(Sep 2026, now concrete — the highest-priority question for supervisors.)~~ DECIDED
  2026-09-02 — headline the main split, report the gap explicitly; see the final entry of §8.**
  Main split is **94.55%**; the leave-one-dataset-out mean is **52.11%**. Which is *the* result
  in the thesis? The NFR (≥85%) was written against a held-out test split, which 94.55% clears
  comfortably — but LODO is the honest estimate of field performance and misses it by 33
  points. Three defensible framings: (a) headline the main split, report LODO as a
  limitation; (b) headline LODO, present the main split as an upper bound under matched
  capture conditions; (c) present both as a bounding range and argue the deployment case sits
  between them, closer to LODO. **This must be decided before the results chapter is drafted,
  and it changes what Phase 3/4 need to demonstrate on-device.** Recommendation to put to
  Dr. Shahreeza: (c), because it is defensible under examination and does not require
  discarding a legitimately-earned 94.55%.
- **(Sep 2026)** Given that recipe v1 under-trained every run, should the FYP1 MobileNetV2
  baseline (68.14%) be re-run under v2 for a fair like-for-like comparison, or reported as-is
  with the confound documented? Re-running is cheap; the argument against is that FYP1's
  result is a historical record, not a live claim.
- **~~(2026-09-24, new — priority for supervisors given how large the number is.)~~ PARTLY
  DECIDED 2026-09-25 — the fix is chosen and built; what remains is whether it can be accepted
  without field confirmation.** The camera framing measurement put a **45.7-point** synthetic cost
  on the as-built capture geometry, which ruled out shipping the guide as drawn. The app now crops
  the capture to the brackets (§8, final entry), which simulation expects to recover most of it
  (~92-94%). **The live question for Dr. Shahreeza is evidentiary, not a design choice:** field
  photographs taken through the app could not be obtained, so the fix is verified geometrically —
  unit-tested mapping, emulator run — and its accuracy benefit is inferred from simulated
  framings with synthetic backgrounds, never measured on a real photograph. Is that acceptable for
  the deployment claim, stated plainly as such in L8, or does the thesis need to scale back what it
  claims the app achieves in the field? A related residual: a bunch overflowing the brackets is now
  clipped rather than diluted, and no measurement covers that case either.
