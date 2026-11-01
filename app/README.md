# Palm Ripeness — Android app

Phase 4 of the FYP2 plan: the Kotlin app that runs the exported TFLite model on-device, with no
network. Design spec is `DESIGN_HANDOFF.md` in this folder; the decisions behind it are in
`DESIGN_NOTES.md`; the model's input contract is `src/main/assets/model_contract.json`.

## Opening it

The Gradle project root is the **repository root**, not this folder — open
`Palm-Oil-Classification` in Android Studio, not `app/`.

One machine-local setting is needed the first time:

- **Gradle JDK must be 17.** Android Studio 2026.1.4 ships JBR 25, which Gradle 8.10.2 will not
  run on. Set *Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK* to
  the Temurin 17 install. From the command line, export `JAVA_HOME` to it instead.
- `local.properties` (gitignored) holds `sdk.dir`. Android Studio writes it; it is already
  written on this machine.

```bash
JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-17.0.12.7-hotspot" ./gradlew :app:assembleDebug
```

## Build setup

| Setting | Value | Why |
| --- | --- | --- |
| UI toolkit | Jetpack Compose | The handoff is a dp/sp spec with custom components and no Material look; Compose expresses it directly. |
| `minSdk` | 26 (Android 8.0) | Both fonts ship only as variable fonts, and setting a weight on the `wght` axis needs API 26. Fraunces' default instance is Thin, so a lower `minSdk` would render hairline headings. |
| `compileSdk` / `targetSdk` | 35 | Latest platform installed in the SDK. |
| AGP / Gradle / Kotlin | 8.7.3 / 8.10.2 / 2.0.21 | Pinned in `gradle/libs.versions.toml` so the APK the report benchmarks can be rebuilt later. |
| Inference | `org.tensorflow:tensorflow-lite` | Built-in ops only — the conversion spike confirmed no Flex delegate is needed. |
| Permissions | `CAMERA` only | No `INTERNET`, deliberately: the offline claim in the report is checkable in the manifest. |

### A note on APK size before it reaches the report

The debug APK is ~34 MB, and almost none of that is the model (6.1 MB). The bulk is
`libtensorflowlite_jni.so` shipped for all four ABIs plus un-minified debug code. The number
that belongs in the report against the ≤ 100 MB requirement is a **release** APK for the ABI
of the test phone, not this one — measure it with `assembleRelease` (R8 is already on for
release) and, if a single-ABI figure is wanted, an `abi` split.

## Assets, and where they come from

Everything in `src/main/assets/` and the test fixtures are generated, never hand-edited. Run from
the repo root:

```bash
python src/training/make_app_bundle.py
```

That writes the model, `labels.txt`, the contract, the JVM resize fixture, the instrumented test
fixtures, and the quantization variants the device latency sweep needs
(`src/androidTest/assets/benchmark/`). `model.tflite`, the fixture `.jpg` files, the `.input.bin`
tensors and the staged variants are gitignored (derived weights and dataset images stay out of
git), so a fresh clone must run it before the instrumented tests will pass.

Current bundle: FastViT_T8 float16 from
`fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42`, 6.1 MB, generated 2026-09-25 (see
`model_contract.json` for the sha256).

## Tests

The preprocessing is the part that fails silently — feed `[0, 255]` instead of `[-1, 1]`, or
centre-crop instead of squashing, and the model still returns four confident, wrong scores. Two
test layers exist for exactly that:

```bash
./gradlew :app:testDebugUnitTest          # JVM: Kotlin resize == tf.image.resize
./gradlew :app:connectedDebugAndroidTest  # device: app numbers == Python numbers
```

- `ResizeParityTest` (JVM, no emulator) regenerates the test pattern from its formula and checks
  the Kotlin bilinear resize against sampled `tf.image.resize` outputs.
- `DeviceLatencyBenchmark` (device) sweeps every staged quantization mode × {1, 4} threads, 15
  warm-up runs and 50 timed runs per cell, and writes `device_benchmark.md` to the app's external
  files dir. This is the evidence for the ≤ 2 s requirement; pull it with

  ```bash
  adb pull /sdcard/Android/data/my.edu.um.palmripeness/files/device_benchmark.md results/
  ```

  It sweeps rather than timing only the shipped model because the float16-over-dynamic_int8 choice
  was made on a dev CPU, where int8 has no specialised path. On ARM it goes through XNNPACK and may
  win, in which case the bundled artefact should be reconsidered — against
  `results/tflite_export_summary.md`, since a latency win only counts if fidelity holds. `full_int8`
  is deliberately not staged: it is the fastest and smallest and FastViT scores 0.260 under it.
- `ModelParityTest` (device) checks the interpreter wiring from the exact float tensors Python fed
  (tolerance 1e-4), then the whole decode → resize → scale path from the same JPEGs, and records
  median inference time against the ≤ 2 s requirement.

The jpg-path tolerance is **measured, not assumed**. Android cannot use `tf.io.decode_jpeg`, and
two libjpeg configurations legitimately disagree; `make_app_bundle.py` re-decodes every fixture
with Pillow and writes the resulting spread into `reference.json`, and the tolerance is that
number with headroom. Currently: worst 0.0424 probability difference, mean 0.0059, **0 of 20
argmax flips**, so the tolerance is 0.064. The device run reproduced Pillow's worst case to seven
decimal places, which is the evidence that the remaining difference is the decoder and not the
app. The failure this test exists to catch — wrong value range, a centre crop, swapped channels —
is off by whole units, and the argmax assertion is the one that must never bend.

## State

All four screens are built and the flow runs end to end: Home, Camera (CameraX preview, framing
guide, torch, shutter, low-light hint, permission-off state), Review, and Results with its
analysing, verdict, close-call and unreadable states. Upload goes through the system photo picker,
which needs no storage permission.

Verified on an Android 35 emulator by driving the app: photographing the emulator's scene runs the
whole capture path, and picking the bundled fixture `10_ripe.jpg` through the upload path produces
**Ripe, 88.9%** against Python's **88.7%** for the same file — the 0.2-point gap is the JPEG
decoder difference measured below, not a preprocessing error.

Not done:

- No screen has been seen on real hardware. The torch in particular is untestable on an emulator,
  which reports no flash unit, so `FlashToggle` has only been exercised through a Compose preview.
- Process death drops the pending photo; the flow restarts at Home rather than restoring it.

## Deviations from the handoff, and open questions

Full reasoning for each is in `DESIGN_NOTES.md` under "Decisions made during implementation".

- **The Analysing screen has no percentage.** The handoff called for a determinate 0-100 bar;
  inference is tens of milliseconds, so it is an indeterminate spinner on a 400 ms floor instead.
- **Review's photo area takes the space above the buttons** rather than a flat 640 dp, which is the
  same thing on the 360 x 800 reference but avoids a dead band on a taller phone. Its title also
  gained a gradient scrim, without which white text is lost on a sunlit bunch.
- **The camera hint pill is anchored 20 dp under the framing guide**, which is where the handoff's
  absolute 530 dp puts it on the reference device and where it stays on every other one.
- The Home body copy under the H1 was not specified; the text in `HomeScreen.kt` is a first draft.
- Stage strip images are tinted silhouettes, not photos — the placeholder the handoff expects.
- "Tabular figures" for the confidence line is not applied; Instrument Sans' `tnum` feature is not
  wired up.

- **The capture is cropped to the framing brackets before it reaches the model** (2026-09-25).
  The brackets were advisory: they cover 280 x 360 dp of a full-bleed preview while the whole
  frame went to the model, so a user obeying the hint handed it a small, background-heavy bunch
  where every training image is a tight crop. Simulated over the full test split that framing
  costs **45.7 points** (`results/framing_shift/`, `PROJECT_PLAN.md` §8). `CameraScreen` now
  measures the brackets against the preview, hands the bounds up as a `FramingRegion`, and
  `MainViewModel` crops the decoded frame to them — so Review shows, and the model receives, what
  the brackets held. Geometry is `model/FramingRegion.kt` with `test/.../FramingRegionTest.kt`
  covering it. The hint reads "Fill the brackets with one bunch" for the same reason.
  **The gallery path is deliberately not cropped**: nothing says where the subject sits in a photo
  the app did not frame.

Still open:

- **The crop is verified geometrically and in simulation, not in a plantation.** Scored at the
  brackets' own 7:9 aspect it recovers 44.0 of the 45.7 points — 92.47% against a 94.16% baseline
  (`results/framing_shift_brackets/`) — but no photograph taken through the shipped app has been
  scored, because field images could not be obtained. Two things it therefore cannot rule out: real
  plantation background behaving differently from the synthetic fill, and users framing bunches
  that overflow the brackets, which the crop now clips rather than shrinks.
- **The low-light thresholds are provisional** (mean preview luma, enter below 50, clear above 70)
  and have not been checked against real photographs.
- The close-call threshold ships as the design's assumed 15 points (`Prediction.MARGIN_POINTS`),
  which is still not derived from the validation results.

## Fonts

Fraunces and Instrument Sans, bundled in `src/main/res/font/` as the variable fonts Google Fonts
publishes. Both are SIL Open Font License 1.1; the licences are in `licenses/`.
