# Smartphone-Based Palm Oil Ripeness Detection Using Lightweight Vision Models

**Nur Atiqah Maisarah Binti Mohamad Rasi (22003055)**
Supervisor: Dr. Muhammad Shahreeza Safiruz Kassim
Co-Supervisor: Ahmad Syazwan Ramli (MPOB)
Faculty of Computer Science and Information Technology, Universiti Malaya

---

> **Drafting notes — delete before submission.**
>
> - Every figure reference points at a real file in `results/figures/`. Every number traces to a
>   named artifact under `results/`; the source is given in italics under each table.
> - Passages marked **[FILL]** need material this repository does not contain — mostly FYP1
>   citations and the mockup images. Nothing in those passages is invented; the placeholder marks
>   exactly what is missing.
> - Chapter 5 draws on `docs/report_limitations.md`, which holds the fuller, examiner-facing prose
>   for each limitation. Where this draft compresses a limitation to a paragraph, that file has the
>   long version — lift from there rather than re-deriving.
> - **The scoping sentence in §4.1 must accompany every appearance of the 94.55% headline.** This
>   is a recorded project decision (`PROJECT_PLAN.md` §8, 2026-09-02), not a stylistic preference.

---

## 1. Introduction

### 1.1 Background

Oil palm (*Elaeis guineensis*) is Malaysia's most economically significant agricultural commodity,
and the oil yield extracted from a fresh fruit bunch (FFB) depends heavily on when that bunch is
harvested. A bunch cut before maturity carries a low oil-to-bunch ratio; a bunch left too long
accumulates free fatty acids that degrade the oil's commercial grade. The window in which both
quantity and quality are maximised is narrow, and the decision to cut is made in the field, bunch by
bunch, by a human grader.

That decision is currently made by eye. Graders judge ripeness from the colour of the fruitlets and
the number of loose fruits detached from the bunch, following criteria issued by the Malaysian Palm
Oil Board (MPOB). Visual grading is fast and requires no equipment, but it is subjective: judgement
varies between graders, degrades with fatigue over a working day, and shifts with ambient lighting —
the same bunch photographed in direct noon sun and in plantation shade presents noticeably different
colour. Inconsistent grading propagates directly into mill intake quality and into the price a
smallholder receives.

Automating this judgement is therefore a well-motivated computer-vision problem, and one that has
attracted substantial published attention [FILL: cite the 3–5 strongest works from your FYP1
review here]. The constraint that shapes this project is not accuracy alone but *where the
computation happens*. Malaysian oil palm smallholdings are frequently beyond reliable mobile data
coverage, and the graders who would use such a tool carry mid-range Android phones, not
workstations. A system that requires a server round-trip is unusable in the setting it is meant to
serve. This project therefore targets a model that runs entirely on-device, offline, within the
memory and latency budget of a commodity smartphone.

### 1.2 Problem statement

Manual visual grading of oil palm FFB ripeness is subjective, inconsistent between graders, and
sensitive to field lighting conditions, which introduces avoidable variance into harvest timing and
mill intake quality. Existing automated approaches in the literature largely report accuracy on a
single curated dataset evaluated under laboratory or desktop conditions, and are seldom constrained
by the memory, latency, and offline-operation limits of the smartphone hardware a field grader
actually carries.

This project's own preliminary work (FYP1) illustrates the gap concretely. A MobileNetV2
transfer-learning baseline trained on a public Roboflow dataset reached 68.14% top-1 accuracy — 17
points short of the project's ≥85% acceptance bar. Diagnosis found the shortfall was not primarily a
modelling failure: 194 of the dataset's 205 unique source photographs had crops distributed across
the training, validation, *and* test splits, so the reported accuracy was itself measured under
train/test leakage and overstated true generalization. The dataset contained only 205 unique source
photographs, all captured over two days, and the unripe class was represented by just 45 of them —
which explains that model's 0.38 recall on that class. The problem is therefore twofold: build an
evaluation setting in which an accuracy number means what it claims to mean, and within that setting
produce a model that meets a smartphone deployment budget.

### 1.3 Research objectives

The three objectives approved at FYP1 are carried forward unchanged:

1. **RO1** — To develop a smartphone-based oil palm FFB ripeness detection system using lightweight
   image classification models.
2. **RO2** — To analyse and compare lightweight architectures — MobileNet, EfficientNet-Lite, and
   Vision Transformers — for ripeness classification.
3. **RO3** — To evaluate the models under varying lighting conditions and optimize the
   best-performing model for smartphone deployment.

Two substitutions were made within RO2 during execution, both forced by pretrained-weight
availability rather than by a research decision, and both documented in §3.4.3: EfficientNet-Lite0
was replaced by EfficientNetB0, and MobileViT by FastViT-T8. The objective's requirement — a
lightweight CNN of each family plus a lightweight Vision Transformer variant — is preserved by both
substitutions.

### 1.4 Research questions

- **RQ1** — Can a lightweight vision model reach ≥85% four-class FFB ripeness accuracy on a
  leakage-free, multi-source test split?
- **RQ2** — Among a lightweight CNN (MobileNetV2), a scaled CNN (EfficientNetB0), and a hybrid
  convolution–transformer (FastViT-T8) trained under an identical recipe, which performs best, and
  is the margin larger than run-to-run variance?
- **RQ3** — Does accuracy measured on held-out *photographs* from sources present in training
  transfer to an entirely held-out *source* — a different camera, site, and labelling convention?
- **RQ4** — Does the selected model survive conversion and quantization to TensorFlow Lite within
  the ≤100MB size and ≤2s latency budget, and does it remain stable across daylight, shadow, and
  low-light conditions?

### 1.5 Hypotheses

- **H1** — Replacing the leaky single-source FYP1 dataset with a leakage-free merged corpus drawn
  from four independent sources will raise achievable accuracy above the 85% bar, and will
  specifically repair the unripe-class weakness (recall 0.38 at FYP1), because that weakness was a
  data-diversity artifact rather than a modelling limitation.
- **H2** — A hybrid convolution–transformer designed for mobile inference will outperform the two
  convolutional baselines at comparable parameter cost, because global attention captures the
  gradual, whole-bunch colour transition distinguishing adjacent ripeness stages better than purely
  local convolution.
- **H3** — Accuracy on a held-out source will be materially lower than accuracy on the main split,
  because the four source datasets encode different labelling conventions for the same physical
  ripeness continuum.
- **H4** — The selected model will meet the ≤100MB and ≤2s budgets after quantization without a
  material accuracy penalty.

H1 and H3 are supported by the results in Chapter 4. H2 is supported *with a substantial correction
to its magnitude* (§4.3). H4 is supported for model size and unresolved for on-device latency
(§4.5). One further result was anticipated by no hypothesis and became the project's central
methodological finding: for most of this project's duration the binding constraint on accuracy was
neither the architecture nor the data but the training schedule (§4.2).

### 1.6 Scope and limitations

In scope: four ripeness classes (unripe, underripe, ripe, overripe); still images at 224×224
resolution; on-device inference with no server component; four public datasets as the training
corpus.

Out of scope, and stated here rather than left for the reader to discover: the empty-bunch and
abnormal-fruit categories present in one source and the "decayed" category present in another are
excluded rather than remapped, as they fall outside the four-class taxonomy; bunch detection and
counting; and yield estimation. Field images from MPOB via the co-supervisor were treated as an
opportunistic bonus rather than a dependency and did not arrive within the project timeline —
cross-source generalization is therefore evidenced by the leave-one-dataset-out protocol of §3.6
rather than by a field trial. The Android application (§3.2) was specified, designed, and
implemented; it runs the exported model end to end on an emulator and reproduces the Python
pipeline's probabilities to within the difference attributable to the platform's JPEG decoder (§4.6).
What it has not had is a physical device: no latency figure, and no photograph taken through the
application in a plantation, exists. Field images could not be obtained within the project period,
which bounds what the deployment claims in this report can rest on and is stated as such wherever
they appear.

### 1.7 Contributions

1. A leakage-free merged corpus of 18,167 labelled FFB images from four independently collected
   public sources, split by source photograph rather than by crop, with per-`(source, class)` split
   balancing so that every source retains its own 70/15/15 proportions.
2. A three-architecture comparison under a recipe held identical by construction, at three seeds
   each, reporting run-to-run variance rather than single-run point estimates.
3. A quantified cross-source generalization gap — 42.44 points — obtained by holding out each of the
   four sources entirely in turn, together with an analysis identifying the failure mode as
   labelling-convention fitting rather than uniform degradation.
4. A methodological finding of general applicability: an ablation inherits the validity of the
   recipe it is run under. A fine-tuning-depth control run under this project's original schedule
   produced a clean but false negative result, which reversed when re-run under the corrected
   schedule (§4.3).
5. A TensorFlow Lite conversion study across three architectures and four quantization modes,
   establishing that post-training full-integer quantization is unusable for the two more accurate
   architectures — quantization robustness runs *inversely* to the accuracy ranking (§4.5).

---

## 2. Literature Review

> **[FILL]** This chapter is drafted as a scaffold carrying the arguments and the works actually
> anchored in this project's artifacts. Citations marked [cite] must be supplied from your FYP1
> literature review, which is the source of record for them and is not held in this repository. Do
> not submit with [cite] markers in place.

### 2.1 Oil palm ripeness grading: the task and its criteria

MPOB's grading standard defines ripeness by fruitlet colour and by the count of loose fruits
detached from the bunch [cite]. The criteria are ordinal — unripe, underripe, ripe, overripe form a
progression rather than a set of unrelated categories — and the boundaries between adjacent stages
are conventions imposed on a continuous biological process rather than natural discontinuities. This
matters more than it first appears, and §4.4 returns to it: two datasets can label the same physical
bunch differently without either being wrong, and a model trained across several such datasets must
reconcile conventions that do not agree.

### 2.2 Classical computer-vision approaches

Early automated grading work extracted hand-designed colour and texture features — RGB and HSV
channel statistics, colour histograms, GLCM texture descriptors — and classified them with
conventional learners such as SVM, k-NN, or shallow neural networks [cite]. These methods are
computationally cheap, which suits edge deployment, but they are brittle under illumination change
precisely because their features are colour statistics and illumination change *is* a colour
transformation [cite]. That brittleness motivates the lighting-robustness requirement carried
through this project as NFR-4 and evaluated in §4.7.

### 2.3 Deep learning for FFB ripeness classification

Convolutional networks displaced hand-designed features for this task, with reported accuracies
commonly in the high eighties to mid nineties [cite]. Two published works bear directly on this
project's design decisions and are treated in detail rather than merely listed.

**Suharjito et al.** — the research group whose annotated bunch-pile dataset forms the largest
component of this project's corpus — published a companion study applying ImageNet transfer learning
to four lightweight CNNs for mobile FFB ripeness classification. EfficientNetB0 was their best
performer at **89.8% accuracy, 89.3% after float16 TensorFlow Lite quantization, at 96ms per
image**; MobileNetV1 was less accurate at 81.1% but fastest [cite: full bibliographic entry]. Two
specifics of their recipe are relevant here. First, they fine-tune at **block-level granularity —
three unfrozen convolutional blocks** — rather than by a fixed layer count. This project adopted that
scheme (§3.4.4), and §4.3 shows the choice to be considerably more consequential than it appears.
Second, they employ a fixed nine-angle multi-crop augmentation scheme, which this project did *not*
adopt; it remains a candidate for future work (§5.5).

**Harjanto et al. (2025)** report up to **+6.58% accuracy** from segmenting the fruit bunch out of
the background with a lightweight YOLOv8n-Seg model before classifying with ShuffleNetV2 [cite: full
bibliographic entry]. Their setting differs from this project's in ways that prevent the finding
from being assumed to transfer: a single source dataset, one object per image after manual curation,
2,400 hand-annotated segmentation masks, and no on-device latency or size constraint. This project
scoped a two-stage, cost-gated replication of the finding (§5.5) but did not execute it within the
project period.

### 2.4 Lightweight architectures for mobile inference

**MobileNetV2** [cite: Sandler et al., 2018] introduced inverted residual blocks with linear
bottlenecks and bounded ReLU6 activations, targeting mobile CPU inference directly. The bounded
activation range is a design decision made *for* quantization, and §4.5 observes its consequence
empirically: MobileNetV2 is the only one of the three architectures tested here that survives
post-training full-integer quantization intact.

**EfficientNet** [cite: Tan & Le, 2019] derives its family by compound scaling of depth, width and
resolution from a mobile-sized base network. The EfficientNet-**Lite** variants were designed
specifically for edge deployment by removing squeeze-and-excitation blocks and replacing swish with
ReLU6 — both changes made for quantization compatibility [cite]. This project intended to use
EfficientNet-Lite0 and could not (§3.4.3); the substitution to EfficientNetB0 retains the SE blocks
and SiLU activations that Lite removed, which §4.5 shows is exactly why EfficientNetB0's
full-integer quantization degrades. The substitution is thus not merely a limitation but an
informative one.

**Vision Transformers on mobile.** ViTs [cite: Dosovitskiy et al., 2021] model global relationships
between image patches through self-attention, at a computational cost and data appetite that are
poor fits for edge hardware. Mobile-oriented hybrids interleave convolution with attention to
recover locality and parameter efficiency: **MobileViT** [cite: Mehta & Rastegari, 2022] and
**FastViT** [cite: Vasu et al., Apple, 2023 — ⚠️ verify venue, ICCV vs CVPR, before submission] are
the two considered here. MobileViT was named in this project's plan; §3.4.3 documents why it could
not be evaluated under the project's software stack and why FastViT-T8 replaced it. The
architectural argument motivating H2 is that the whole-bunch colour gradient distinguishing adjacent
ripeness stages is a global property of the image, which attention is better positioned to capture
than a purely local convolutional receptive field.

### 2.5 Ordinal classification

Because ripeness is ordinal, treating the four classes as unrelated categories discards structure:
confusing unripe with overripe is a worse error than confusing ripe with underripe, and categorical
cross-entropy is indifferent between them. Rank-consistent ordinal formulations such as **CORAL**
[cite: Cao et al., 2020] decompose a K-class ordinal problem into K−1 rank-monotonic binary
thresholds. This project tested the approach directly; §4.8 reports it as a negative result with the
mechanism identified.

### 2.6 Datasets and the reproducibility problem

Published FFB ripeness work is fragmented across small, independently collected datasets, and
several public datasets are re-exports or derivatives of one another. This project encountered the
problem in a form worth reporting. The "new" dataset sourced at the start of FYP2 proved to be the
same public dataset already used in FYP1, identified by identical per-class test counts. The
archival copy of the bunch-piles dataset turned out to be a Roboflow re-export whose class-id
ordering differed from the ordering stated in its own source publication, and in which 552 of
roughly 4,160 unique frames appeared in more than one split folder — the same leakage defect that
compromised FYP1, baked into the distribution. Both defects are silent: neither raises an error, and
both produce plausible-looking results. §3.3 documents how each was detected and handled.

### 2.7 Research gap

Three gaps follow and define this project's contribution. First, accuracy figures in this literature
are predominantly single-source and single-run; the effect of a change of camera, site and labelling
convention is rarely quantified, and seed-to-seed variance is rarely reported at all. Second,
deployment claims are commonly made from framework-side latency measurements rather than from
converted, quantized models — §4.5 shows those to be a poor predictor. Third, lightweight Vision
Transformers are under-represented in FFB ripeness work relative to their prominence in general
mobile vision. This project addresses the first two directly and the third within the constraint
documented in §3.4.3.

---

## 3. Methodology

### 3.1 Overall system architecture

The system is organised as four layers, each layer's output forming the next layer's input. There is
no server component at any point: the trained model is embedded in the application and all inference
happens on the device.

```
┌─ DATA LAYER ────────────────────────────────────────────────────────────┐
│  4 public sources ──▶ Merge & Split ──▶ Augmentation (train split only) │
│                       (by source photo)   lighting + geometric, on-the-fly│
│                             │                                            │
│                             └──▶ Lighting Stress-Test set (deterministic)│
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼  training-ready dataset
┌─ TRAINING LAYER ────────────────────────────────────────────────────────┐
│  Training Orchestrator ──▶ Evaluator                                     │
│  one identical recipe      accuracy / F1 / ROC-AUC / lighting / LODO      │
│  × 3 architectures × 3 seeds                                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼  selected model
┌─ EXPORT LAYER ──────────────────────────────────────────────────────────┐
│  TFLite Converter ──▶ Fidelity check vs Keras parent ──▶ Device benchmark │
│  float32 / float16 / dynamic-int8 / full-int8                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼  .tflite + labels.txt
┌─ APP LAYER (on-device, offline) ────────────────────────────────────────┐
│  Capture/Upload UI ──▶ TFLite Inference Engine ──▶ Result Display        │
└─────────────────────────────────────────────────────────────────────────┘
```

*Figure 3.1 — System architecture. Reproduce as a proper diagram for submission; the Mermaid source
is in `SYSTEM_ARCHITECTURE.md` §2.*

**Data Layer.** The four raw sources are mapped onto the locked four-class taxonomy and merged into a
single pool that retains a `source` column on every row, which is what makes the leave-one-dataset-out
protocol of §3.6 possible. Splitting is by source photograph, never by crop. Augmentation is applied
on the fly to the training split only, so validation, test, and the stress-test set remain clean and
reproducible measures.

**Training Layer.** One recipe, defined in a single module (`src/training/schedule.py`) through which
all three architectures run, so "all three architectures share an identical recipe" is enforced by
construction rather than by copy-paste discipline across three scripts. Every run writes its
configuration, per-epoch history, and metrics to its own directory under `results/<run-name>/`; no run
overwrites another.

**Export Layer.** The selected checkpoint is converted to TensorFlow Lite at four quantization
settings, and — critically — each converted model's predictions are compared against its unquantized
Keras parent rather than the conversion being accepted on success alone. §4.5 shows why that check is
load-bearing.

**App Layer.** Capture or upload, on-device inference through a stock TFLite interpreter, and a result
screen with per-class confidence. Designed in §3.2 and implemented; §4.6 reports what it does with the
model, including the framing defect that layer turned out to contain.

### 3.2 App design

> **Status.** Implemented. All four screens are built in Jetpack Compose and the flow runs end to
> end on an Android 35 emulator; §4.6 reports what the built application does with the model. This
> section presents the design; two deliberate departures from it are noted at the end.
>
> **[FILL]** Insert the three mockup images from the FYP1 deck (Home / Preview / Result), and
> consider replacing them with screenshots of the built application, which now exist. The
> wireframes below describe them but do not replace them.

**Screen 1 — Home / Capture.** Entry point. Two primary actions, deliberately only two: *Take Photo*
opens the device camera; *Upload Image* opens the gallery picker. A single-line status indicator
confirms the model is loaded and that the app works offline — the offline guarantee is a selling
point for the intended user and is surfaced rather than left implicit.

```
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│  Palm Ripeness           │   │  Preview                 │   │  Result                  │
│                          │   │                          │   │                          │
│   ┌──────────────────┐   │   │   ┌──────────────────┐   │   │   ┌──────────────────┐   │
│   │                  │   │   │   │                  │   │   │   │   [thumbnail]    │   │
│   │   [app icon /    │   │   │   │  captured image  │   │   │   └──────────────────┘   │
│   │    illustration] │   │   │   │                  │   │   │                          │
│   │                  │   │   │   │                  │   │   │        RIPE              │
│   └──────────────────┘   │   │   └──────────────────┘   │   │      96.4% confident     │
│                          │   │                          │   │                          │
│  ┌────────────────────┐  │   │  ┌────────┐ ┌─────────┐  │   │  unripe    ▇ 1.2%        │
│  │   TAKE PHOTO       │  │   │  │ RETAKE │ │ DETECT  │  │   │  underripe ▇▇ 2.1%       │
│  └────────────────────┘  │   │  └────────┘ └─────────┘  │   │  ripe      ▇▇▇▇▇▇ 96.4%  │
│  ┌────────────────────┐  │   │                          │   │  overripe  ▇ 0.3%        │
│  │   UPLOAD IMAGE     │  │   │  Tip: fill the frame     │   │                          │
│  └────────────────────┘  │   │  with a single bunch     │   │  ┌────────────────────┐  │
│                          │   │                          │   │  │   NEW PHOTO        │  │
│  ● Model ready · offline │   │                          │   │  └────────────────────┘  │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

*Figure 3.2 — Screen flow: Home → Preview → Result. Replace with the FYP1 mockup images.*

**Screen 2 — Preview.** The captured or selected image is shown before inference, with *Retake* and
*Detect Ripeness*. The confirmation step exists because field capture conditions are poor and a
blurred or badly framed photo should be rejected by the user rather than silently classified. A
one-line framing hint sets the capture convention the model was trained on.

**Screen 3 — Result.** The predicted class in large type, the model's confidence in that class, and —
importantly — a per-class confidence bar for all four classes. Showing the full distribution rather
than the winning label alone is a deliberate design response to a measured property of the model:
the underripe/ripe boundary is genuinely ambiguous (§4.4), and a bunch scoring 51% ripe against 45%
underripe is a materially different situation for the grader than one scoring 96% ripe. Collapsing
both to the word "Ripe" would hide exactly the uncertainty the grader needs to see.

**Interaction and non-visual design decisions.**

- **Offline by construction.** The `.tflite` model and `labels.txt` are bundled into the APK at build
  time. The application requests no network permission at all, which makes the offline guarantee
  verifiable from the manifest rather than a claim.
- **No account, no upload, no history sync.** Nothing leaves the device. This removes the privacy
  question entirely and keeps the app usable with no connectivity.
- **Inference on a background thread** with the result screen showing a determinate progress state,
  so the UI cannot appear frozen even if a low-end device approaches the latency budget.
- **Failure is a state, not a crash.** Model-load failure and unreadable-image failure each have a
  defined user-visible message.

**Two departures from this design in the built application, both deliberate.** The determinate
progress state was replaced by an indeterminate one: inference takes tens of milliseconds, so a
percentage would have been fiction, and the indicator is instead held for a 400 ms floor so the step
registers as a step. More consequentially, the camera's framing brackets were specified as advisory
decoration over a preview whose *whole* frame was to be classified; they are instead **load-bearing**,
and the capture is cropped to them before preprocessing. §4.6 gives the measurement that forced that
change. The offline property is enforced rather than asserted: the manifest requests no network
permission at all, so the bundled model and labels are the only possible source of a prediction.

### 3.3 Requirements

#### 3.3.1 Functional requirements — what the system must do

| ID | Requirement | Verification | Status |
|---|---|---|---|
| FR-1 | The user can capture an FFB image using the device camera | App integration test | **Implemented** — CameraX preview, shutter, torch; emulator only (§4.6) |
| FR-2 | The user can select an existing FFB image from device storage | App integration test | **Implemented** — system photo picker, no storage permission |
| FR-3 | The system preprocesses any accepted image to 224×224 RGB before inference | Unit test on preprocessing path | **Implemented** — matches Python to 1e-4 on tensors, decoder difference aside (§4.6) |
| FR-4 | The system classifies an image into exactly one of unripe / underripe / ripe / overripe | Test-set evaluation | **Met** (§4.1) |
| FR-5 | The system displays the predicted class together with a confidence score for **all four** classes | UI test | **Implemented** — per-stage chart plus a close-call notice |
| FR-6 | All inference runs on-device with no network call | Absence of network permission in the manifest | **Met** — no network permission is requested at all |
| FR-7 | The user can discard a preview and re-capture before classifying | UI test | **Implemented** — Review screen, Retake / Detect |
| FR-8 | The system reports a defined error state when the model fails to load or an image cannot be decoded | UI test | **Implemented** — model-load and unreadable-image states |

#### 3.3.2 Non-functional requirements — how well it must do it

In plain terms: it has to be *right* often enough to be trusted, *fast* enough not to interrupt
fieldwork, *small* enough to install on an ordinary phone, *steady* across the lighting a plantation
actually offers, and *free* to run.

| ID | Requirement | Target | Measured | Status |
|---|---|---|---|---|
| NFR-1 | Classification accuracy on the held-out test split | ≥ 85% | **94.55% ± 0.31** (FastViT-T8, n=3) | **Met** — with the scoping in §4.1 |
| NFR-2 | On-device inference latency per image | ≤ 2 s | 47.2 ms (converted, dev CPU, single-thread) | **Not demonstrated** — no physical-device benchmark (§4.5) |
| NFR-3 | Deployed model size | ≤ 100 MB | **6.1 MB** (FastViT-T8, float16 TFLite) | **Met**, by more than an order of magnitude |
| NFR-4 | Stability across daylight / shadow / low-light | Stable | **1.02 ± 0.43 pts** spread across four lighting conditions | **Met** (§4.7) |
| NFR-5 | Supports 224×224 input | Required | Entire pipeline is 224×224 | **Met** |
| NFR-6 | Operates with no network connectivity and no per-use cost | Required | Fully on-device; conversion needs no Flex delegate (§4.5) | **Met at model level**; app-level pending implementation |

*Sources: `results/multiseed_v2_summary.md`, `results/tflite_export_summary.md`,
`results/fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42/lighting_stress_test_metrics.csv`.*

NFR-1 and NFR-2 stand in tension with NFR-3 in principle, and the tension did not bind in practice:
the most accurate architecture also produced the smallest float16 artifact of the three (6.1 MB
against EfficientNetB0's 7.7 MB), because parameter count rather than accuracy drives file size.
A genuine trade-off does appear at the quantization level and is reported in §4.5: the smallest and
fastest setting is unusable for the selected architecture.

### 3.4 Dataset construction

#### 3.4.1 Sources

Four independently collected public datasets were merged. The FYP1 Roboflow dataset was retired as a
training source and retained only as a documented negative result.

| Source | Origin | Rows contributed | Notes |
|---|---|---:|---|
| Bunch Piles (Suharjito & Junior) | Mill-collected video frames, Indonesia; CC BY 4.0 | 12,583 | Ships as YOLO detection boxes; cropped to classification patches |
| Mendeley Ordinal (Kamal et al., 2024) | Central Kalimantan, industry-funded | 4,317 | 5 classes; "Decayed" dropped as out of scope |
| Kaggle (hadeemadadum) | Public supplement; CC BY 4.0 | 851 | Exact 4-class taxonomy match |
| Outdoor-Tenera Malaysia (MunirahRosbi) | Johor, N. Sembilan, Perak; outdoor lighting | 416 | **No underripe class**; damaged/empty dropped |
| **Total** | | **18,167** | 9,473 unique source photographs |

*Source: `data/README.md`; counts re-derived from `data/processed/manifest_full.csv`.*

Per-class totals by source are given in §4 and plotted in `results/figures/01_class_distribution.png`
and `02_source_composition.png`. The corpus is dominated by Bunch Piles (69% of rows), and the
Malaysia outdoor set — the only Malaysian-collected source, and therefore the closest proxy to the
deployment setting — contributes 2.3%. Both facts bear on the interpretation of §4.4.

#### 3.4.2 Merge, taxonomy mapping, and split discipline

All sources are mapped onto the locked four-class taxonomy; anything outside it is dropped rather
than remapped. Three defects were found in the source data during this process and are reported
because each was silent:

1. **Bunch Piles class-id order does not match its own publication.** The Roboflow re-export's
   `_darknet.labels` gives a different 0–5 ordering than the paper states. The cropping script
   text-matches each label in Malay and English rather than hardcoding ids, and refuses to run if no
   labels file is present.
2. **Bunch Piles frames appear in multiple split folders.** 552 of ~4,160 unique frames occur in more
   than one of the export's own `train`/`valid`/`test` directories. Every source's own split is
   therefore ignored and the split rebuilt from scratch across the merged corpus.
3. **Byte-identical duplicate images exist within sources**, including two pairs crossing the Malaysia
   set's own train/test boundary. Every image is MD5-hashed and exact-content duplicates dropped (11
   from Malaysia, 6 pairs from Kaggle). All whole-photo sources were also cross-checked against each
   other by MD5: **zero images are shared across datasets.**

Splitting groups every row by `(source_dataset, source_photo)` and assigns whole groups to
train/valid/test at 70/15/15, so every crop from one bunch-pile frame stays in one split. **Zero
source-photo leakage is verified programmatically** — every group's rows land in exactly one split.

| Split | overripe | ripe | underripe | unripe | Total |
|---|---:|---:|---:|---:|---:|
| train | 2,580 | 3,648 | 3,259 | 3,229 | 12,716 |
| valid | 553 | 784 | 699 | 693 | 2,729 |
| test | 552 | 780 | 698 | 692 | 2,722 |

**A split-balancing bug found and fixed.** The first version of the split assigned groups against a
global per-class target. Because Bunch Piles' groups are large (many crops per frame) while every
other source contributes single-image groups, and the greedy assignment sorts by group size
descending and fills whichever split is furthest below target, Bunch Piles was funnelled into train
— 82–94% of every class's train split against as little as 14% of valid/test. Models were being
evaluated on a different source mix than they trained on, for every class. The fix computes targets
per `(source_dataset, class)`, so each source retains its own 70/15/15 proportions regardless of
group size. The correction was worth roughly ten points of measured accuracy on the baseline model
(63.71% → 73.73%), and every result in this report is from the corrected split.

#### 3.4.3 Architecture substitutions

Both substitutions were forced by pretrained-weight availability under the project's software stack
(TensorFlow 2.20 / Keras 3.15), not chosen.

**EfficientNet-Lite0 → EfficientNetB0.** The only available pretrained Lite0 weights are published in
the legacy TF1-Hub format, which raises `ValueError` on `trainable=True` and therefore cannot be
fine-tuned. Three integration routes were tested and all failed for reasons independent of the
fine-tuning limitation. Since every other architecture in the comparison uses two-stage
frozen-then-fine-tuned training, a frozen-only Lite0 would confound architecture with training
methodology. EfficientNetB0 is Keras-3 native, supports the identical recipe, and belongs to the same
lightweight CNN family.

**MobileViT → FastViT-T8.** Five independent integration routes for MobileViT each failed for a
distinct, reproducible reason, the most fundamental being a `KerasTensor` type error inside
`transformer_pre_process` at the patch-folding step — an operation constitutive of the MobileViT
architecture, so not workable around without reimplementing the model. To establish whether the
obstacle was specific to MobileViT, fifteen lightweight transformer variants were swept under
identical conditions: **only FastViT built successfully**; MobileViT V2, LeViT, EfficientViT,
EdgeNeXt and TinyViT all failed identically. The limitation is a property of the current Keras 3
ecosystem for mobile transformers, not of this project's configuration. FastViT-T8 was then validated
against six criteria before any training time was committed — including lossless save/reload
round-trips, the criterion on which an earlier candidate route had failed. At 3.20M parameters it
sits between MobileNetV2 (2.26M) and EfficientNetB0 (4.05M), so the comparison is not confounded by
model scale.

#### 3.4.4 Training recipe

All three architectures are driven through one shared two-stage schedule.

| Element | Setting |
|---|---|
| Input | 224×224×3, batch size 32 |
| Head | GlobalAveragePooling → Dropout(0.2) → Dense(4, softmax) |
| Loss | Categorical cross-entropy, label smoothing 0.15 |
| Class imbalance | Balanced class weights applied as per-sample weights |
| Stage 1 (frozen) | Backbone frozen, head only, lr 1e-3, cap 30–50 epochs |
| Stage 2 (fine-tune) | Last **K=3 blocks** unfrozen, lr 1e-4 under cosine decay to ~0, cap 40 epochs |
| Early stopping | Patience 8 on validation loss, best weights restored |
| BatchNorm | Pinned to inference mode throughout, including Stage 2 |
| Augmentation | Train split only: horizontal flip, ±10% rotation / zoom / translation, ±20% brightness and contrast, applied on-the-fly in [0,255] space before rescaling |
| Seeds | 42, 43, 44 for every reported configuration |

Fine-tuning depth is specified in **blocks rather than layers**, following Suharjito et al., because a
fixed layer count represents different proportional depths in different architectures. §4.3 shows
this choice to be far more consequential than a routine hyperparameter, and §5.2 records the residual
confound it leaves.

Label smoothing at 0.15 is retained from FYP1 for a task-specific reason: the ripe/underripe boundary
is genuinely subjective, and label smoothing makes the model appropriately less confident on
borderline cases rather than forcing a hard call.

### 3.5 Evaluation protocol

Two splits are used and must not be confused.

| | **Main split** | **Leave-one-dataset-out (LODO)** |
|---|---|---|
| Construction | All four sources pooled, split by source photograph; every source appears in train, valid and test | Each source held out *entirely* in turn; the other three form the training pool. Four runs. |
| Question answered | "How accurate is the model on an unseen photograph from these sources?" | "Does the model work on a camera, site and labelling convention it has never encountered?" |
| Role | Headline figure against NFR-1; selects the winning architecture | Primary generalization evidence; the honest estimate of field performance |
| Cost | All three architectures × 3 seeds | Winner only: 4 folds × 3 seeds |

Metrics reported: overall accuracy, per-class precision/recall/F1, macro F1, one-vs-rest ROC-AUC,
confusion matrices, parameter count, artifact size, and inference latency. For LODO folds, macro F1
is computed over the classes actually present in the held-out source — the Malaysia set contains no
underripe images, and averaging over the full label set would penalise that fold for a class it could
not have been tested on.

### 3.6 Lighting robustness protocol

The lighting stress-test set is generated once from the clean test split and versioned as its own
artifact, never randomly regenerated. Three conditions are applied to every one of the 2,722 test
images:

- **darkened** — brightness ×0.7, simulating low light;
- **brightened** — brightness ×1.3, simulating harsh daylight and overexposure;
- **shadow** — a soft-edged elliptical patch at 50% brightness, its position and size seeded
  deterministically from a hash of the image path, so the effect varies image-to-image like a real
  cast shadow but reproduces exactly on every run.

The shadow condition is an addition over FYP1's brightness-only test, made so that all three lighting
conditions named in NFR-4 are actually covered. The cost is that these figures are not numerically
comparable with FYP1's lighting numbers, and are not presented as such.

### 3.7 Reproducibility measures

Every run writes `config.json` (including a `recipe_version` field), `history.json`, `metrics.csv`, a
classification report, a confusion matrix, and lighting stress-test metrics to its own directory. The
aggregation scripts refuse to pool runs across recipe versions or across compute backends, on the
grounds that a v1 run and a v2 run are not replicates of the same process — nor are a CPU run and a
GPU run. All reported configurations use three seeds, and every headline figure is quoted as
mean ± sample standard deviation rather than as a single run.

---

## 4. Results and Analysis

All results in this chapter were produced under training recipe **v2** on a single NVIDIA RTX 4050
Laptop GPU (TensorFlow 2.20 / Keras 3.15 / Python 3.12). Where an earlier v1 figure is quoted it is
explicitly labelled as superseded; §4.2 explains why the distinction is load-bearing.

### 4.1 Headline result

**FastViT-T8, fine-tuned at three blocks, reaches 94.55% ± 0.31 test accuracy on the held-out test
split (n=3 seeds), with macro F1 0.9467 ± 0.0035 and one-vs-rest ROC-AUC 0.9916 ± 0.0002.**

> **Required scoping — reproduce this wherever the headline appears.** The ≥85% accuracy requirement
> is met on the held-out test split (94.55% ± 0.31). It is **not** claimed to hold under source
> shift: leave-one-dataset-out accuracy is 52.11%, a 42.44-point generalization gap, and that figure
> is the honest estimate of performance on a mill this model has never seen.

All three architectures clear the 85% bar on the main split, so **RQ1 is answered affirmatively and
H1 is supported.** The FYP1 unripe-class failure is fully repaired: recall on unripe rises from 0.38
to 0.940 ± 0.005, confirming that the weakness was a data-diversity artifact rather than a modelling
limitation.

### 4.2 Architecture comparison

| Metric | MobileNetV2 | EfficientNetB0 | **FastViT-T8** |
|---|---|---|---|
| Test accuracy (%) | 89.62 ± 0.15 | 91.26 ± 0.39 | **94.55 ± 0.31** |
| ROC-AUC (OvR) | 0.9800 ± 0.0018 | 0.9861 ± 0.0012 | **0.9916 ± 0.0002** |
| Macro F1 | 0.8983 ± 0.0015 | 0.9137 ± 0.0042 | **0.9467 ± 0.0035** |
| Recall — unripe | 0.9397 ± 0.0107 | 0.9427 ± 0.0021 | 0.9400 ± 0.0053 |
| Recall — underripe | 0.7933 ± 0.0150 | 0.7960 ± 0.0108 | **0.9033 ± 0.0161** |
| Recall — ripe | 0.9207 ± 0.0055 | 0.9553 ± 0.0068 | **0.9663 ± 0.0021** |
| Recall — overripe | 0.9363 ± 0.0095 | 0.9623 ± 0.0064 | **0.9763 ± 0.0086** |
| Lighting spread (pts) | 5.19 ± 0.88 | 2.45 ± 0.37 | **1.02 ± 0.43** |
| Parameters (M) | 2.26 | 4.05 | 3.20 |
| Size, pre-quantization (MB) | 20.9 | 33.9 | 30.2 |

*Source: `results/multiseed_v2_summary.md`. Figure: `results/figures/09_multiseed_architecture_comparison.png`.
Per-class metrics: `04_per_class_metrics.png`. Confusion matrices: `03_confusion_matrices.png`.*

FastViT-T8 leads on every accuracy metric by margins several times the seed-to-seed spread, so **RQ2
is answered: the ranking is real, not noise.** The margin over EfficientNetB0 at the same nominal
K is 3.29 points — but §4.3 shows that figure to overstate the architectural contribution
substantially, and it must not be quoted without that correction.

The largest single per-class gain is on **underripe**, the class that had been this project's
persistent weak point: 0.79 for both CNNs against 0.903 for FastViT. That the hybrid architecture's
advantage concentrates on the hardest ordinal boundary is consistent with the mechanism proposed in
H2 — global attention capturing a whole-bunch colour gradient — though this report does not claim to
have isolated that mechanism.

**Note on the pre-quantization latency column.** The v2 campaign also recorded framework-side
inference times of 141.9 / 161.3 / 156.6 ms for MobileNetV2 / EfficientNetB0 / FastViT-T8. §4.5
establishes those figures to be a poor predictor of deployed latency, and they are reported here only
to be superseded — the converted models measure 12.1 / 22.4 / 47.7 ms, a 3.9× spread where the
framework figures showed 1.13×, and in a different rank order.

#### 4.2.1 The training-schedule correction — the project's central methodological finding

For most of this project's duration every reported accuracy was an artifact of an under-trained
pipeline, and the defect was invisible in the results.

The original schedule (recipe v1) capped training at 10 frozen plus 10 fine-tuning epochs with a
constant 1e-5 fine-tuning learning rate. Reading back the saved per-epoch history of all fifteen GPU
runs afterwards showed:

| Diagnostic | Count |
|---|---|
| Stage 2 (fine-tuning) early-stopped | **0 of 15** |
| Stage 1 (frozen) early-stopped | 6 of 15 |
| Validation loss still falling at the final epoch | 13 of 15 |
| Best validation accuracy occurring *at* the final epoch | 11 of 15 |

Fine-tuning hit its epoch cap in **every single run** — the early-stopping callback never once fired
during Stage 2. That it *did* fire in Stage 1 for six of those runs is what makes the diagnosis
conclusive rather than a suspected dead callback: the frozen head genuinely plateaus and the callback
correctly ends it, while the fine-tuning stage never got the chance.

Recipe v2 raises the caps well clear of where runs actually stop (30–50 frozen, 40 fine-tuning),
raises early-stopping patience from 4 to 8, and raises the fine-tuning learning rate to 1e-4 under
cosine decay. On **unchanged data and unchanged splits**, a single FastViT seed moved from 78.69% to
93.98%, since confirmed across all three architectures and three seeds each.

| Architecture | Recipe v1 | Recipe v2 | Δ |
|---|---:|---:|---:|
| MobileNetV2 | 73.61 ± 0.59 | 89.62 ± 0.15 | +16.01 |
| EfficientNetB0 | 76.45 ± 0.90 | 91.26 ± 0.39 | +14.81 |
| FastViT-T8 | ~78.7 (single seed) | 94.55 ± 0.31 | ≈ +15.9 |

Two variables move together in v2 — training budget and learning rate — so this is an upgraded recipe
applied uniformly, not a controlled single-variable ablation. The *architecture* comparison remains
fair because every architecture receives the same v2. Runs record their recipe version, and the
aggregation scripts refuse to pool across versions.

The finding that matters beyond this project is in §4.3: an ablation run under v1 inherited v1's
invalidity, and produced a clean result that was false.

### 4.3 Fine-tuning depth: how much of the architecture margin is really architecture?

FastViT-T8 exposes ten blocks; MobileNetV2 and EfficientNetB0 expose sixteen each. K=3 therefore
unfreezes **28.0%** of FastViT's layers but only **19.4%** of EfficientNetB0's and 18.8% of
MobileNetV2's. The same nominal setting is not the same proportional depth, so the 3.29-point margin
may reflect fine-tuning depth rather than architecture.

A control was run to separate them: EfficientNetB0 at K=4 — 25.7% of its layers, the closest
available match to FastViT's 28.0% — at three seeds, everything else identical.

| Configuration | Layers unfrozen | Test accuracy | Per-seed |
|---|---:|---:|---|
| EfficientNetB0, K=3 | 19.4% | 91.26 ± 0.39 | 91.55 / 91.40 / 90.82 |
| **EfficientNetB0, K=4** | **25.7%** | **92.73 ± 0.37** | 92.73 / 92.36 / 93.09 |
| FastViT-T8, K=3 (selected) | 28.0% | 94.55 ± 0.31 | — |

*Source: `results/effnet_depth_k4_v2_summary.md`.*

**At comparable fine-tuning depth, FastViT-T8's advantage over EfficientNetB0 is 1.82 points rather
than the 3.29 points implied by the same-K comparison. Approximately 45% of the apparent
architectural advantage is attributable to fine-tuning depth rather than to architecture.** The
residual 1.82 points exceeds the combined run-to-run spread of ±0.68, so the architecture conclusion
survives — H2 is supported, at roughly half the magnitude it first appeared to have. Any sentence in
this report attributing the full 3.29 points to architecture would be wrong.

The depth effect is not confined to aggregate accuracy: at K=4, EfficientNetB0's underripe recall
rises 0.796 → 0.831 and its lighting spread tightens from 2.45 to 1.33 points. For scale, the
accuracy gap between EfficientNetB0 and MobileNetV2 — two architectures with identical block counts,
and therefore directly comparable at a shared K — is 1.64 points. **A single step of fine-tuning
depth is worth about as much as the difference between two architectures.**

#### 4.3.1 The falsified control — a methodological result, not an admission

This same control was run once before, under recipe v1, and produced the opposite result:
EfficientNetB0 at K=4 scored **0.42 points below** K=3, comfortably within run-to-run variation. That
measurement supported a conclusion — that depth sensitivity was specific to the hybrid transformer
and the architecture margin survived the control undiminished — which is now known to be an artifact.
The v1 schedule halted every run before convergence, so the additional unfrozen layers had no
remaining budget in which to adapt. **The control could not have detected a depth effect had one been
present.**

The episode generalises beyond this project. An ablation inherits the validity of the recipe it is
run under: a control executed on an under-trained pipeline measures the under-training, not the
variable it was designed to isolate. Because the defect *suppressed* the treatment effect rather than
producing an implausible number, the result looked clean and was accepted. It was caught only because
the recipe was later corrected for unrelated reasons and every dependent conclusion was re-derived
rather than carried forward.

**Residual limitation.** Depth is still not exactly matched and cannot be: EfficientNetB0's block
grid steps 19.4% → 25.7% → 31.6% while FastViT's steps 15.4% → 28.0%, leaving a residual mismatch of
about 2.3 points with the CNN on the shallower side. That mismatch is not neutral — EfficientNetB0
was still gaining as depth increased rather than having saturated — so **the 1.82-point margin should
be read as an upper bound on the architectural contribution, not a point estimate.** A K=5 run
(31.6%), which would bracket FastViT from above, is the natural next control and was not carried out.

### 4.4 Cross-source generalization (LODO)

Each source was held out entirely in turn, FastViT-T8 trained on the remaining three, and evaluated
purely on the held-out source. Four folds × three seeds.

| Held-out source | Classes | Test accuracy (%) | ROC-AUC | Macro F1 (present classes) |
|---|---:|---:|---|---|
| Bunch Piles | 4 | 40.98 ± 2.33 | 0.7153 ± 0.0205 | 0.4128 ± 0.0194 |
| Kaggle | 4 | 50.80 ± 0.78 | 0.8295 ± 0.0023 | 0.4972 ± 0.0051 |
| Mendeley Ordinal | 4 | 55.75 ± 0.40 | 0.7900 ± 0.0097 | 0.5474 ± 0.0073 |
| Malaysia Outdoor | 3 | 60.90 ± 0.37 | n/a | 0.6503 ± 0.0125 |
| **Mean across folds** | | **52.11** | | |

*Source: `results/lodo_v2_summary.md`. Figure: `results/figures/11_lodo_generalization_gap.png`.*

**The generalization gap is 42.44 points** (94.55% main split against 52.11% LODO mean). **RQ3 is
answered and H3 is supported, at a magnitude larger than anticipated.**

Three properties of this result require statement rather than summary.

**It is not another under-training artifact.** Under v1, 0 of 15 runs early-stopped; under v2, **10
of 12 fine-tuning stages early-stopped.** The low cross-source number cannot be explained away by
training budget the way the v1 main-split numbers could. The honest reading is that a properly
trained model fits each training source's labelling convention *harder* rather than generalizing past
it — correcting the recipe bought roughly 18 points on held-out photographs and gave back roughly 4
points on an entirely held-out source (the v1 LODO mean was 55.96%, against a 76.45% main split: a
20.49-point gap that has now widened to 42.44).

**The failure mode is convention-fitting, not uniform degradation.** Under v2 the model concentrates
its cross-source predictions on ripe and overripe. On held-out Kaggle, ripe recall rose 0.659 → 0.921
and overripe 0.785 → 0.912 between v1 and v2, while **underripe collapsed from 0.406 to 0.067**. On
held-out Mendeley the same pattern appears (ripe 0.621 → 0.817, underripe 0.128 → 0.063). Held-out
Bunch Piles moves the other way (underripe 0.199 → 0.547, unripe 0.418 → 0.140), which is consistent
with the same mechanism rather than against it: **each fold snaps to whichever convention the
remaining three sources agree on.** Where those three agree against the held-out source, the held-out
source's convention is simply overwritten.

**Two claims about underripe must be kept apart.** Main-split underripe weakness was under-training,
and v2 fixed it (≈0.50 → 0.903). Cross-source underripe weakness is a labelling-convention mismatch
between sources, and v2 made it *worse*. These are backed by different campaigns pointing in opposite
directions and must not be conflated in the discussion.

**Confounds to state.** First, fold training-set sizes differ substantially — holding out Bunch Piles
leaves ~4.7k training images against ~15.1k for Malaysia Outdoor — so the weak Bunch Piles fold
confounds domain shift with reduced training data. The other three folds, which retain most of the
corpus, isolate domain shift more cleanly. Second, the v1 LODO campaign ran EfficientNetB0 while the
v2 campaign runs FastViT-T8, so recipe and architecture changed together and the −3.85 delta is not
attributable to the recipe alone. What is safe to claim under any attribution is the **direction**:
cross-source accuracy did not improve.

### 4.5 Mobile deployment: conversion, quantization, and size

All three v2 architectures at seed 42 were converted to TensorFlow Lite at four quantization
settings, with each converted model's predictions compared against its unquantized Keras parent on a
200-image sample.

| Model | Mode | Size (MB) | Accuracy (n=200) | Agreement with parent | Latency (ms) |
|---|---|---:|---:|---:|---:|
| FastViT-T8 | float32 | 12.1 | 0.920 | 0.990 | 47.7 |
| **FastViT-T8** | **float16** | **6.1** | **0.925** | 0.970 | **47.2** |
| FastViT-T8 | dynamic int8 | 3.3 | 0.930 | 0.970 | 114.3 |
| FastViT-T8 | full int8 | 3.5 | **0.260** | 0.265 | 136.7 |
| EfficientNetB0 | float16 | 7.7 | 0.910 | 1.000 | 23.6 |
| EfficientNetB0 | full int8 | 4.7 | **0.645** | 0.660 | 20.3 |
| MobileNetV2 | float16 | 4.3 | 0.905 | 1.000 | 12.0 |
| MobileNetV2 | full int8 | 2.6 | 0.915 | 0.960 | 13.4 |

*Source: `results/tflite_export_summary.md` (full 12-row table). Latency is single-threaded x86
development CPU, not a device measurement.*

**All twelve conversions succeeded using built-in TensorFlow Lite operators alone — none required the
Flex delegate.** This matters because FastViT-T8 is a hybrid convolution–transformer assembled from
custom layers, and a Flex-dependent conversion would have obliged the application to embed a slice of
the full TensorFlow runtime, costing tens of megabytes. The selected architecture is deployable
against a stock TFLite interpreter, and the deployability risk is closed.

**Recommended artifact: FastViT-T8 at float16 — 6.1 MB, 47.2 ms, no accuracy cost.** Half the float32
size for no latency penalty. Size is nowhere near binding (6.1 MB against 100 MB), so there is no
reason to accept dynamic int8's 2.4× latency penalty to save a further 2.8 MB. **NFR-3 is met with a
margin of more than an order of magnitude.**

**Post-training full-integer quantization is unusable for the two more accurate architectures, and
the failure tracks architecture rather than size.** FastViT collapses to 0.260 — chance on four
classes — and EfficientNetB0 to 0.645, while MobileNetV2 is untouched at 0.915. This is the textbook
pattern: FastViT's normalization and attention activations and EfficientNet's SiLU plus
squeeze-and-excitation branches both produce dynamic ranges that a single per-tensor integer scale
cannot represent, whereas MobileNetV2's bounded ReLU6 design was built to quantize. **Quantization
robustness runs inversely to the accuracy ranking: the least accurate architecture is the most
deployable.** This is a genuine architecture-level trade-off and is one of this report's substantive
results, not a footnote.

It also carries a methodological point. All three fully-quantized models converted without error and
produced the smallest files of any setting tested. **A benchmark measuring only conversion success,
file size and latency would have selected the fully-quantized model for all three architectures,
including the one performing at chance.** The failure is detectable only by comparing the converted
model's predictions against those of the model it was derived from.

**What this does not establish.** Seed 42 only, not the three-seed protocol. The accuracy column is
measured on a 200-image sample sized to compare converted against unconverted predictions — on that
sample the three architectures sit within one point of each other while the full test set separates
them by five, so **it must not be read as an architecture comparison.** Latency is single-threaded
x86; integer arithmetic frequently outperforms float16 on ARM via XNNPACK or NNAPI, so the float16
recommendation could reverse on a phone. **NFR-2 remains undemonstrated: a physical Android benchmark
was not carried out.**

### 4.6 On-device implementation and input fidelity

The application described in §3.2 was built and runs the float16 artifact end to end. This section
reports what that exercise established about the model's *input*, which turned out to matter more than
anything it established about its output: **the largest single accuracy effect measured anywhere in
this project was found here, in the camera's framing, and not in the model.**

**The application reproduces the Python pipeline, and the residual difference is the JPEG decoder.**
Three measurements separate the possible causes rather than asserting a conclusion. First, fed the
exact float32 tensors the Python pipeline was fed, on twenty stratified test-split fixtures bundled
with the application, the on-device interpreter reproduces the reference probabilities to within
**1e-4** — establishing that interpreter wiring, buffer layout, output shape and label order are all
correct. Second, a JVM unit test confirms the Kotlin bilinear resize matches `tf.image.resize` under
half-pixel centres without antialiasing to better than 1e-3 at 800 sampled positions, removing the
resize as a candidate. Third, starting instead from the same JPEG files and running the application's
own decode → resize → scale path, probabilities move by up to **0.0424**, with the **predicted class
unchanged on all twenty**. With the first two fixed, the third is attributable to image decoding
alone: Android cannot call `tf.io.decode_jpeg`, and decoding the same fixtures on the desktop with
Pillow — an independent libjpeg configuration — disagrees with TensorFlow's decoder by up to 16 levels
of 255 per pixel (mean 1.8) and reproduces the *same* worst-case probability shift, on the same image,
to seven decimal places (0.042418). The difference is the decoder, not the application.
`make_app_bundle.py` now performs this comparison as part of building the app bundle and derives the
instrumented test's tolerance from it (0.064) rather than from an assumed value.

*Source: `app/src/androidTest/.../ModelParityTest.kt`, `app/src/test/.../ResizeParityTest.kt`,
`app/src/main/assets/reference.json` (`decoder_spread`).*

**The framing defect.** Every image in the corpus of §3.4 is a tight crop around a single bunch. The
application as first specified drew advisory framing brackets over roughly a quarter of a full-bleed
preview and classified the *whole* frame, so a user obeying the on-screen hint would hand the model a
small, background-heavy bunch at a scale and context nothing in training resembled. The parity
fixtures cannot reveal this, because they are already crops. Re-scoring the full 2,722-image test
split through simulated capture framings quantifies it.

| Capture framing | Aspect | Area fraction | Accuracy | Δ (pts) |
|---|---:|---:|---:|---:|
| Unmodified test crops (baseline) | — | 1.00 | 94.16% | — |
| *As specified:* whole frame classified | 9:20 | 0.25 | **48.46%** | −45.70 |
| Squarer viewfinder, same area | 1:1 | 0.25 | 46.33% | −47.83 |
| Aspect distortion alone | 9:20 | 0.64 | 81.41% | −12.75 |
| Aspect distortion alone | 1:1 | 0.93 | 93.57% | −0.59 |
| *As built:* cropped to the brackets | 7:9 | 0.84 | **92.47%** | **−1.69** |

*Source: `results/framing_shift/framing_shift.json` and
`results/framing_shift_brackets/framing_shift.json`. Seed-42 float16 TFLite artifact, Pillow decoding,
synthetic backgrounds; area fraction as achieved rather than as requested. Note the 94.16% baseline is
this artifact on the full test split under Pillow decoding, not the 94.55% ± 0.31 three-seed Keras
headline of §4.1 — the comparison in this table is internal to the row set.*

At the framing as specified the model scores **48.46%**, a **45.70-point** loss against the same
artifact's 94.16% on unmodified crops. That is larger than the training-schedule correction of §4.2.1
and larger than the cross-source gap of §4.4: the largest single effect in this report, and one
produced entirely by user-interface geometry. It is also not evenly distributed — unripe and underripe
recall collapse to 0.390 and 0.312 from 0.938 and 0.904, while ripe and overripe degrade comparatively
gently — so the loss concentrates on the boundary this project has found difficult everywhere else.

**Scale, not shape, is the binding term, which rules out the intuitive remedy.** Holding the bunch at
full frame and varying only the aspect ratio costs between 0.59 and 12.75 points; narrowing the bunch
to a quarter of the frame costs upward of 45 regardless of shape, and a square viewfinder at that area
(46.33%) is marginally *worse* than the tall one. Simply making the viewfinder squarer would therefore
not have addressed the defect.

**The remedy adopted, and what it recovers.** Of the available options — leave the specification
alone, widen the brackets to most of the frame and rewrite the hint, or crop the capture to the
brackets before preprocessing — the third was implemented. It was preferred over widening because it
does not depend on the user complying with an instruction: whatever is framed, the application cuts to
the box the user was aiming through, so the model's input is determined by code rather than by
photographic discipline. The brackets are measured against the camera preview at capture time and the
decoded frame is cropped to them, with the review screen displaying that same cropped region, so what
the user approves is what is classified. Scored at the shipped brackets' 7:9 aspect, the corrected
framing reaches **92.47%** — recovering **44.0 of the 45.7 points**. The residual 1.69 points falls
almost entirely on underripe (recall 0.904 → 0.834), with the other three classes within a point of
baseline; mean confidence in the predicted class also falls, from 0.805 to 0.761, which matters
because the application's close-call notice keys off that margin. Images supplied through the gallery
are deliberately not cropped: the application did not frame them and has no basis for locating the
subject within them.

**What this does not establish.** The framing conditions are simulated. Backgrounds are synthetic — a
blurred continuation of the crop, or flat grey as a control — rather than photographs of plantation
clutter, so these figures isolate the scale and aspect change faithfully while approximating
everything else. The correction's geometry is covered by unit tests, but **no photograph taken through
the corrected application has been scored, because field images could not be obtained**, and the
recovery figure therefore comes from the same simulator that found the defect rather than from an
independent observation. Two risks follow and are carried into §5.3: real background clutter may not
behave as the synthetic fill does, and a bunch overflowing the brackets is now clipped by the crop
where previously it was merely diluted by the surrounding frame. Finally, nothing here is a latency
measurement — the instrumented run was an x86_64 emulator, which says nothing about a phone.

### 4.7 Lighting robustness

The lighting stress-test set applies three deterministic transforms to all 2,722 clean test images.

| Condition | FastViT-T8 accuracy |
|---|---:|
| Clean test baseline | 94.56% |
| Shadow | 93.90% |
| Darkened (×0.7) | 93.72% |
| Brightened (×1.3) | 93.06% |
| **Spread across conditions** | **1.02 ± 0.43 pts** (n=3 seeds) |

*Source: `results/fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42/lighting_stress_test_metrics.csv`,
`results/multiseed_v2_summary.md`. Figure: `results/figures/05_lighting_stress_test.png`.*

**NFR-4 is met.** The spread across four lighting conditions is roughly three times the seed-to-seed
standard deviation and no condition costs more than 1.5 points. The ranking across architectures is
notable in its own right: FastViT-T8 1.02 pts, EfficientNetB0 2.45 pts, MobileNetV2 5.19 pts — the
selected architecture is not only the most accurate but by a clear margin the most stable, and
MobileNetV2's five-point spread is the widest measured. Brightening is consistently the harshest
condition, consistent with overexposure destroying the colour saturation cues on which the task
depends.

### 4.8 Secondary experiments and negative results

**Ordinal loss (CORAL) — negative result, mechanism identified.** Hypothesis: since ripeness is
ordinal, a rank-consistent ordinal head should reduce underripe↔ripe confusion better than
categorical cross-entropy. Same backbone, same recipe, same class weighting; only the loss and head
changed. Result: 66.90% against the categorical baseline's 73.73% under the same (v1) conditions. It
*did* reduce underripe→ripe confusion from 34% to 21% and raised underripe recall from 0.39 to 0.47,
directionally confirming the hypothesis — but it overcorrected: ripe→underripe confusion nearly
quadrupled from 10% to 34%, ripe recall collapsed from 0.82 to 0.53, and lighting stability worsened
from a 2.2-point spread to 6.1. The technique traded one direction of the confusion for the other
rather than resolving it. Not adopted. *Caveat: this experiment was run under recipe v1 and was not
re-run under v2; the comparison is internally consistent but its absolute numbers are superseded.*

**Per-class threshold calibration — a v1-era remedy the corrected recipe made redundant.** Per-class
decision thresholds were fitted on the validation split alone and applied unchanged to test.

| Architecture | Test gain, v1 | Test gain, v2 | Validation gain, v2 |
|---|---:|---:|---:|
| EfficientNetB0 | +1.41 | +0.55 | +0.84 |
| **FastViT-T8 (selected)** | +0.45 | **+0.21** | +0.60 |
| MobileNetV2 | −0.24 | +0.29 | +0.46 |

*Source: `results/calibration_v2_summary.md`. Figure: `results/figures/10_calibration_effect.png`.*

For the selected model **the gain (+0.21) is smaller than its own seed-to-seed spread (±0.31)** and
cannot be distinguished from run-to-run variance. The mechanism explains why: calibration works by
trading recall between adjacent classes, so its value depends on there being a badly served class to
trade toward. Under v1, FastViT's underripe recall was 0.510 and calibration raised it to 0.681 at a
cost of 0.120 in ripe recall. Under v2 underripe arrives at 0.904 untouched, and the same procedure
moves it by 0.021. There is almost nothing left to recover. Roughly two-thirds of the fitted gain
also fails to transfer (+0.60 validation against +0.21 test), which is the measure of how far the
fitting procedure overfits its own objective. **Decision: the headline stays uncalibrated at 94.55%.**
Quoting the calibrated 94.76% would claim a gain inside the noise and would oblige the deployed
pipeline to carry threshold weights through the TFLite export for an unmeasurable benefit.

This negative result earns its place because it corroborates §4.2.1 from an independent direction: a
technique introduced specifically to compensate for poor underripe recall becomes superfluous once
the training schedule is fixed. That is evidence that under-training, rather than a modelling or data
limitation, was the binding constraint.

### 4.9 Measurement integrity: the duplicate-row defect

The merged manifest's test split contains 2,722 rows over 2,695 unique images: 27 rows (0.99%) are
exact duplicates originating in the bunch-piles crop pipeline. They never cross a split boundary, so
this is double-counting rather than leakage. Rather than bound the exposure and leave it, the effect
was measured: inference was run once per unique image and every metric computed twice over identical
predictions — once weighted by manifest multiplicity (reproducing what was published) and once
counting each image one.

| Quantity | As published | Deduplicated | Δ |
|---|---:|---:|---:|
| FastViT-T8 test accuracy (n=3) | **94.55% ± 0.31** | 94.51% ± 0.33 | **−0.04 pts** |
| EfficientNetB0 (seed 42) | 91.55% | 91.47% | −0.08 pts |
| MobileNetV2 (seed 42) | 89.64% | 89.61% | −0.03 pts |

*Source: `results/duplicate_row_impact.md`.*

The largest shift across any run measured is **0.08 points** — an order of magnitude below the ±0.31
seed spread the headline already carries, and far below the 9.55-point margin over NFR-1. **No
reported conclusion changes.** The bias direction is knowable rather than a coin flip: every delta is
negative because the 27 duplicated images are ones the models classify well (92–100% on that subset
against 90–95% overall), so double-counting inflates rather than deflates. What is measured is the
*reporting* error, not the *training* error — the training split carries 1,422 duplicated rows whose
effect would require retraining on a deduplicated manifest to quantify, which was out of scope.

---

## 5. Discussion and Conclusion

### 5.1 Revisiting the research questions

| | Question | Answer |
|---|---|---|
| RQ1 | ≥85% on a leakage-free multi-source split? | **Yes** — all three architectures clear it; FastViT-T8 at 94.55% ± 0.31 |
| RQ2 | Which architecture, and is the margin real? | **FastViT-T8**, and yes — but the margin over EfficientNetB0 is 1.82 points at matched depth, not the 3.29 implied by a naive comparison |
| RQ3 | Does it transfer to an unseen source? | **No** — 52.11% LODO mean, a 42.44-point gap, with a convention-fitting failure mode |
| RQ4 | Does it deploy within budget? | **Size yes** (6.1 MB / 100 MB), **lighting yes** (1.02 pts spread), **application built** and faithful to the Python pipeline within decoder tolerance, **on-device latency undemonstrated** |

H1 and H3 are supported. H2 is supported at roughly half its apparent magnitude. H4 is half-resolved.
The project's most consequential finding was hypothesised by nobody: the binding constraint on
accuracy was the training schedule (§4.2.1), and correcting it moved every architecture by 15–16
points on unchanged data.

### 5.2 What the generalization gap means

The 42.44-point gap is the single most important number in this report for anyone considering field
deployment, and it deserves interpretation rather than apology.

The main-split figure is measured on unseen *photographs* from sources the model also trained on. The
LODO figure is measured on an entirely unseen *source* — a different camera, a different site, and,
decisively, a different labelling convention. The gap is the part of main-split accuracy that does
not survive that change. Because ripeness is a continuous biological process cut into four categories
by human convention (§2.1), two datasets can label the same physical bunch differently without either
being wrong, and a model trained across several such datasets learns to reconcile them in whichever
way the training majority agrees on. §4.4 shows exactly that: each held-out fold snaps to the
convention shared by the remaining three sources, and the class most sensitive to convention —
underripe, the boundary case — is the one that collapses.

Two things follow. First, **the gap is not primarily a capacity or a data-volume problem**, so it
would not obviously be closed by a larger model or by more images from the same four sources. It is a
label-semantics problem, and the appropriate remedies are label harmonisation against a single
standard (MPOB's, for a Malaysian deployment), or fine-tuning on target-site data before use, or
both. Second, **it explains why the corrected recipe widened rather than narrowed the gap**: a
properly trained model fits its training sources' conventions harder. Under-training was
inadvertently acting as regularisation against convention-fitting — an observation worth stating
because it is counterintuitive and is directly evidenced here.

The project's reporting decision follows from this analysis rather than from convenience. The
NFR was written against a held-out test split, which 94.55% clears legitimately; the honest estimate
of performance at an unseen mill is 52.11%; and both are stated wherever the headline appears. Stating
the limit unprompted is what makes the headline credible under examination.

### 5.3 Limitations

Compressed here; `docs/report_limitations.md` holds the full examiner-facing prose for each.

- **L1 — The intended ViT could not be evaluated.** MobileViT was named in the plan and proved
  unloadable under Keras 3 across five integration routes; a sweep of fifteen lightweight transformer
  variants found only FastViT buildable. The limitation is a property of the current Keras 3
  ecosystem, not of this project's configuration, and the substitution is documented rather than
  quiet. This is the second architecture in this project displaced by weight availability rather than
  by a research decision.
- **L2 — Fine-tuning depth is not exactly matched, and cannot be** (§4.3). The residual 2.3-point
  mismatch favours FastViT, so 1.82 points is an upper bound on its architectural advantage. A K=5
  control would bracket it and was not run.
- **L3 — On-device latency is undemonstrated** (§4.5). Converted latency is three orders of magnitude
  inside the 2s budget, but on a development x86 CPU. No compliance claim is made on that basis.
- **L4 — Cross-source generalization is weak and the LODO folds carry confounds** (§4.4): unequal
  fold training-set sizes, and a v1→v2 comparison in which architecture and recipe changed together.
- **L5 — The corpus is geographically skewed.** 69% of rows come from one Indonesian mill-collected
  source; the only Malaysian source contributes 2.3% and contains no underripe class at all. For a
  Malaysian deployment target this is a material limitation of the training distribution, and the
  Malaysia fold's 60.90% — the *best* LODO fold — is the closest thing this project has to a
  field estimate.
- **L6 — Measured defects retained by decision, not oversight.** The duplicate-row defect moves the
  headline by 0.04 points and is reported rather than fixed (§4.9); re-running the nine-run campaign
  to recover 0.04 points against a ±0.31 spread was not justified. The fix belongs in the pipeline
  before any future campaign.
- **L7 — CORAL and the FYP1 baseline were not re-run under recipe v2**, so their absolute numbers are
  superseded even though their internal comparisons hold.
- **L8 — Numerical reproducibility.** The XLA autotuner emitted cuDNN algorithm-mismatch warnings
  during FastViT training and not for the CNNs. The three-seed protocol measures total run-to-run
  variance, which subsumes such nondeterminism, so the reported spreads already account for it.
- **L9 — The deployed application cannot decode a JPEG the way training did** (§4.6). Every accuracy
  figure in this report was measured on TensorFlow-decoded images; Android decodes through the
  platform's own libjpeg configuration, which disagrees by up to 16 levels of 255 per pixel and shifts
  probabilities by up to 0.042. The predicted class did not change on any of the twenty fixtures
  measured, but twenty images bound the effect anecdotally rather than quantifying it over the test
  set. The reported accuracies should be read as accuracies on TensorFlow-decoded images.
- **L10 — The framing correction is verified in simulation, not in the field** (§4.6). The
  45.70-point framing defect and the 44.0 points the crop recovers are both measured by re-scoring
  the test split through simulated capture geometries with synthetic backgrounds. The correction's
  geometry is unit-tested, but no photograph taken through the corrected application has been scored,
  because field images could not be obtained within the project period. Two consequences remain
  unmeasured: whether real plantation background behaves as the synthetic fill does, and the new
  failure mode in which a bunch overflowing the brackets is clipped rather than diluted. **This is the
  limitation most likely to separate this report's figures from field behaviour**, and it is stated as
  an open risk rather than a closed one.

### 5.4 Conclusion

This project set out to determine whether a lightweight vision model could grade oil palm FFB
ripeness accurately enough, and cheaply enough, to run offline on a smartphone. On the terms the
requirement was written in, it can: FastViT-T8 reaches 94.55% ± 0.31 on a leakage-free four-source
test split, converts to a 6.1 MB TensorFlow Lite artifact needing no non-standard runtime, and varies
by only 1.02 points across daylight, shadow and low-light conditions. Four of the five non-functional
requirements are met; the fifth, on-device latency, is undemonstrated rather than failed, with
converted inference measuring 47 ms against a 2,000 ms budget on development hardware.

That is not the whole finding, and the report would be dishonest if it stopped there. The same model
scores 52.11% when tested on a source it has never seen — a 42.44-point gap that is the honest
estimate of performance at an unfamiliar mill. The analysis in §5.2 locates the cause not in model
capacity but in the disagreement between datasets about where one ripeness category ends and the next
begins, which is a labelling problem that more training on the same corpus will not solve.

Three methodological results are, in the author's assessment, of wider value than the accuracy
figures. The first is that this project spent most of its duration reporting accuracies that were 15
points too low because every run was stopped before convergence, and the defect was invisible in the
results — visible only in the saved per-epoch histories, which is why they were saved. The second is
that a controlled ablation run on that broken pipeline returned a clean, plausible, and false
result, and reversed when re-run correctly. An ablation inherits the validity of the recipe it is run
under. The third emerged only once the model was put behind a user interface: a framing decision in
the camera screen, invisible to every model-side metric in this report and to the on-device parity
tests alike, cost more accuracy than any modelling choice examined here (§4.6). A deployment path is
not verified by checking that the model was ported faithfully; it is verified by checking what the
application actually feeds it. All three were caught only because every dependent conclusion was
re-derived rather than carried forward, and all three are reported here as findings rather than as
confessions.

### 5.5 Future work

1. **Complete the deployment path** — benchmark the float16 artifact on a physical Android device
   against NFR-2. The application exists (§4.6); what it has never run on is real hardware, which
   also leaves the torch control and the low-light thresholds untested. On ARM, integer quantization
   may outperform float16 via XNNPACK or NNAPI, which could reverse the float16 recommendation.
2. **Photograph bunches through the application and score them**, which is the one measurement that
   would convert the framing correction of §4.6 from a simulated result into an observed one, and
   would simultaneously test the two risks that simulation cannot reach: real plantation background,
   and bunches that overflow the framing brackets. It would also bound the JPEG-decoder effect of L9
   over more than twenty images.
3. **Attack the generalization gap at its cause.** Re-label a stratified subset of all four sources
   against a single standard (MPOB's) and re-measure LODO; if the gap narrows materially, the
   convention-mismatch diagnosis is confirmed and label harmonisation becomes the project's most
   valuable next investment.
4. **Site fine-tuning.** Evaluate how many target-site images are needed to recover main-split
   accuracy at a new mill — a directly actionable number for anyone deploying this.
5. **The K=5 depth control** to bracket FastViT's fine-tuning depth from above and convert the
   1.82-point upper bound into a point estimate.
6. **Segmentation preprocessing**, cost-gated in two stages: first crop the existing test set with an
   off-the-shelf segmenter and re-evaluate the trained classifier — if there is no lift, report a
   non-transferring result and stop; only on a real lift annotate 300–500 masks and train a
   YOLOv8n-Seg stage. Any accuracy gain must then be charged against the ≤2s and ≤100MB budgets.
7. **Nine-angle crop augmentation** per Suharjito et al., which this project did not adopt — with
   source-photo split discipline preserved, since derived crops crossing splits is precisely the
   FYP1 leakage bug.
8. **Re-run the FYP1 MobileNetV2 baseline and the CORAL experiment under recipe v2** for a
   like-for-like comparison, both being cheap.

---

## References

> **[FILL]** Populate from the FYP1 literature review in the faculty's required citation style. The
> entries below are the works actually load-bearing in this report and must all appear; verify each
> against the original before submission.

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted
   Residuals and Linear Bottlenecks. *CVPR*.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks. *ICML*.
3. Vasu, P. K. A., Gabriel, J., Zhu, J., Tuzel, O., & Ranjan, A. (2023). FastViT: A Fast Hybrid
   Vision Transformer using Structural Reparameterization. ⚠️ **Verify venue (ICCV vs CVPR).**
4. Mehta, S., & Rastegari, M. (2022). MobileViT: Light-weight, General-purpose, and Mobile-friendly
   Vision Transformer. *ICLR*.
5. Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition
   at Scale. *ICLR*.
6. Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural
   networks with application to age estimation. *Pattern Recognition Letters*.
7. Suharjito, & Junior, F. A. Annotated Video Dataset of Oil Palm Fruit Bunch Piles for Ripeness
   Grading. *Scientific Data* / Science Data Bank, DOI 10.57760/sciencedb.j00001.00713. [FILL:
   complete entry]
8. Suharjito, et al. Transfer learning of lightweight CNNs for mobile oil palm FFB ripeness
   classification. [FILL: complete entry — this is the EfficientNetB0 89.8% / 96ms comparable]
9. Kamal, et al. (2024). Ordinal Dataset for Ripeness Level Classification of Oil Palm Fresh Fruit
   Bunches. *Mendeley Data*, DOI 10.17632/424y96m6sw. [FILL: complete entry]
10. Harjanto, et al. (2025). Lightweight YOLOv8n Segmentation and ShuffleNetV2 Transfer Learning
    Integration for Oil Palm Fruit Ripeness Classification. BINUS. [FILL: complete entry]
11. Rosbi, M. Outdoor-Tenera Oil Palm Fruit Image dataset. GitHub. [FILL: complete entry]
12. hadeemadadum. Oil Palm Fruit Ripeness Classification Dataset. Kaggle. [FILL: complete entry]
13. Malaysian Palm Oil Board. Oil palm fresh fruit bunch grading standard. [FILL]

---

## Appendix A — Figure index

All figures are regenerated on the recipe-v2 campaign and live in `results/figures/`.

| File | Content | Belongs in |
|---|---|---|
| `01_class_distribution.png` | Per-class counts across the merged corpus | §3.4.1 |
| `02_source_composition.png` | Per-source composition of each class | §3.4.1 |
| `03_confusion_matrices.png` | Confusion matrices per architecture | §4.2 |
| `04_per_class_metrics.png` | Per-class precision / recall / F1 | §4.2 |
| `05_lighting_stress_test.png` | Accuracy across the four lighting conditions | §4.7 |
| `06_overall_summary.png` | Headline metric summary | §4.1 |
| `07_training_curves.png` | Per-epoch training and validation curves | §4.2.1 |
| `08_underripe_ripe_confusion_by_source.png` | The hard boundary, broken down by source | §4.4 |
| `09_multiseed_architecture_comparison.png` | Three-architecture comparison with seed spread | §4.2 |
| `10_calibration_effect.png` | Calibration gain, v2 refit | §4.8 |
| `framing_shift/framing_shift.png` | Accuracy against camera framing, by aspect and area fraction | §4.6 |
| `11_lodo_generalization_gap.png` | Main split vs LODO, per fold | §4.4 |

## Appendix B — Run index

| Result | Run directory / artifact |
|---|---|
| Headline, FastViT-T8 | `results/fastvit_fyp2_blockft_v2_gpu_f50_k3_seed{42,43,44}/` |
| Architecture comparison | `results/multiseed_v2_summary.md` |
| Depth control | `results/effnet_depth_k4_v2_summary.md` |
| LODO campaign | `results/lodo_v2_summary.md` |
| TFLite export | `results/tflite_export_summary.md` |
| Calibration refit | `results/calibration_v2_summary.md` |
| Duplicate-row measurement | `results/duplicate_row_impact.md` |
| Framing defect (as specified) | `results/framing_shift/framing_shift.json` |
| Framing correction (as built) | `results/framing_shift_brackets/framing_shift.json` |
| On-device parity | `app/src/androidTest/.../ModelParityTest.kt`, `app/src/main/assets/reference.json` |
| Decisions log (full reasoning) | `PROJECT_PLAN.md` §8 |
| Limitation prose (full) | `docs/report_limitations.md` |
