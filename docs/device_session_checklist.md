# Device session checklist — the one thing that needs a real phone

Phase 3 and Phase 4 each have exactly one item left that no amount of simulation closes, and they
close in the same sitting. This is the run sheet for it, written before the session so nothing is
decided while a phone is plugged in and the clock is running.

Everything else in both phases is done or closeable at the desk — see `PROJECT_PLAN.md` §0.

## No phone available (2026-11-01): run the automated half on Firebase Test Lab

There is no Android phone for this project. Firebase Test Lab runs the two instrumented tests below
on real ARM phones in a Google data centre, which covers everything in the table that follows
**except** the torch, the low-light thresholds, and the camera-path checks — those need a person
holding the phone and stay open.

**Build (desk, ~2 min).** Assets must exist first (`make_app_bundle.py`, see below), then:

```bash
JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-17.0.12.7-hotspot" \
  ./gradlew :app:assembleDebug :app:assembleDebugAndroidTest
```

That gives two files to upload:

- app: `app/build/outputs/apk/debug/app-debug.apk` (~34 MB)
- test: `app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk` (~37 MB; carries the
  three staged model variants and the 20 fixtures — check it is not ~2 MB, which means the
  gitignored assets were missing)

**Run (Firebase console, no install needed).**

1. <https://console.firebase.google.com> → create a project (Analytics can be off). The free Spark
   plan includes a small daily quota of physical-device runs; check the current limit there, since
   it changes. One or two runs are enough.
2. Test Lab → *Run a test* → *Instrumentation*. Upload the app APK, then the test APK.
3. Pick **physical** devices, not virtual ones — a virtual device is the same x86 emulator problem
   again. Prefer the **lowest-spec phone on offer** (a mid-range Samsung Galaxy A-series or similar):
   the user this app is for carries a mid-range phone, and the slowest device is the conservative
   bound for NFR-2. A second, newer device (e.g. a Pixel) is useful context, not a requirement.
4. Optional, under advanced options if offered: directories to pull =
   `/sdcard/Android/data/my.edu.um.palmripeness/files` (fetches `device_benchmark.md`). Not required —
   the same report is also in logcat.
5. When it finishes: open the device's result → *Logs* → filter by **`PalmBench`**. That is the full
   benchmark table plus three `parity/...` lines (tensor parity, JPEG parity with argmax count, and a
   1-thread median). Download the logcat and save it under `results/device_benchmark_testlab/`.

Equivalent CLI, if `gcloud` is ever installed:

```bash
gcloud firebase test android run --type instrumentation \
  --app app/build/outputs/apk/debug/app-debug.apk \
  --test app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk \
  --device model=<MODEL_ID>,version=<API> \
  --directories-to-pull /sdcard/Android/data/my.edu.um.palmripeness/files
```

(`gcloud firebase test android models list` gives the model IDs.)

**What uploading means.** Both APKs go to Google: the app, the 6.1 MB model and its variants, and
the 20 fixture images (crops from the public CC BY datasets). Nothing else from the repo is sent.

**How to report it.** A Test Lab phone is real target hardware, so it does evidence NFR-2. Two things
should be said in the same sentence: it is one device model in a data centre (on mains power and
air-cooled, so it may throttle less than a phone in a plantation), and the figure is interpreter time
only, excluding decode and resize. Name the device, SoC and Android version, all of which the report
records.

**Then update** `PROJECT_PLAN.md` §0/§8, `docs/report_limitations.md` L4, `report.md` §4.5 and the
`.tex` NFR table from "not demonstrated" to the measured median — and if dynamic_int8 wins, follow
point 4 of "What to write down afterwards" below rather than swapping the artefact on latency alone.

## Why a phone, specifically

| Item | Why the emulator/host cannot answer it |
|---|---|
| NFR-2 (≤ 2 s inference) | The requirement says on-device. Every figure so far is x86 dev CPU; `docs/report_limitations.md` L4 declines to claim compliance from it. |
| float16 vs dynamic_int8 | The artefact was chosen on a 47.2 ms vs 114.3 ms host gap. int8 has no specialised x86 path but goes through XNNPACK on ARM, so the ordering can reverse. |
| Torch / flash | The emulator reports no flash unit. `FlashToggle` has only ever run in a Compose preview. |
| Low-light thresholds | Mean preview luma, enter below 50, clear above 70 — provisional numbers that only real dim light can tune. |
| Camera path end to end | The framing-bracket crop has never met an actual sensor. |
| Release APK size | The ≤ 100 MB figure for the report is a release APK for the test phone's ABI, not the ~34 MB debug build. |

## Before the phone arrives (desk work, no hardware)

- [ ] `python src/training/make_app_bundle.py` — regenerates the model, fixtures and the staged
      quantization variants. All gitignored, so this is not optional on a fresh clone.
- [ ] `JAVA_HOME=<Temurin 17> ./gradlew :app:testDebugUnitTest` — JVM tests, catches a broken
      toolchain now rather than tomorrow. Android Studio's bundled JBR 25 will not run Gradle 8.10.2.
- [ ] `./gradlew :app:assembleDebugAndroidTest` — proves the test APK compiles.
- [ ] `./gradlew :app:assembleRelease` — the APK whose size goes in the report.
- [ ] On the phone: Developer Options → USB debugging on; `adb devices` shows it as `device`, not
      `unauthorized`.

## With the phone connected

```bash
JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-17.0.12.7-hotspot" \
  ./gradlew :app:connectedDebugAndroidTest
adb pull /sdcard/Android/data/my.edu.um.palmripeness/files/device_benchmark.md results/
```

That one command runs both instrumented tests:

- `ModelParityTest` — the app's numbers against Python's, on real hardware, plus the ≤ 2 s median.
- `DeviceLatencyBenchmark` — every staged mode × {1, 4} threads, and writes the report to pull.

Run the benchmark **from cold**, before the manual UI checks warm the SoC. The sweep runs its cells
back to back, so a throttling phone shows up as max drifting away from median; if anything looks
borderline, re-run cold rather than explaining it away.

## Then, by hand (nothing automates these)

- [ ] Torch on/off in real dim light — first execution of that code path ever.
- [ ] Walk into shade / dim room: does the low-light hint appear and clear at sensible points, or do
      50/70 need moving? Write down the luma values where it feels wrong, not just "it felt wrong".
- [ ] Capture a bunch filling the brackets; confirm Review shows the *cropped* region, not the full
      frame. This is the visual check on the 2026-09-25 fix.
- [ ] Capture one bunch deliberately overflowing the brackets — the crop clips rather than shrinks,
      and that limitation is currently theoretical.
- [ ] Upload path through the system picker (gallery images are deliberately not cropped).
- [ ] Note the phone: manufacturer, model, SoC, Android version, ABI. The benchmark records these
      automatically, but the report needs them in prose too.

## Fixes made 2026-09-25 that this session is the first chance to confirm

These came out of a code review and a UI audit on the same day, and every one of them was
reasoned about rather than observed. The phone is what turns them into evidence.

- [ ] **Tab order**: `focusRing` no longer adds its own `focusable()`. Tab through Home with a
      keyboard and confirm each button is **one** stop, not two, and that the ring draws on it.
- [ ] **Framing guide on a short screen**: the guide block is centred above the controls instead
      of pinned 150dp down. Confirm the brackets and the hint pill clear the shutter row.
- [ ] **Status bar on the dark screens**: Camera and Review should now show *white* status-bar
      icons. On Home and Results they stay dark.
- [ ] **Camera failure state**: hard to trigger deliberately. If another app holds the camera,
      the screen should say "The camera did not start" and offer the gallery, not sit black.
- [ ] **Large gallery photo**: pick the biggest photo on the phone. It should classify without an
      OOM; the decode cap came down from 24M to 8M pixels (`ImageInput.MAX_SOURCE_PIXELS`).
- [ ] **Review scrim**: photograph something bright and confirm the title and subtitle stay
      readable over it.

## What to write down afterwards

1. `PROJECT_PLAN.md` §0 — Phase 3 to Complete if the sweep passes, Phase 4's "not yet seen on a
   phone" struck.
2. `docs/report_limitations.md` **L4** — currently says the latency half is outstanding. If the
   benchmark passes, this section is rewritten from "unverified" to a measurement, the same way its
   conversion half was on 2026-09-02.
3. `docs/report_limitations.md` **L7** — the decoder limitation. Its full-split accuracy effect is
   already measured (0.04 points, 2026-11-01); add one line saying Android's decoder on a real phone
   reproduced the emulator's fixture parity, or flag it if it did not.
4. If dynamic_int8 wins on ARM: this is a **decision, not a fix**. Log it in §8 with the accuracy
   column from `results/tflite_export_summary.md` beside the latency, and re-bundle with
   `--mode dynamic_int8` only if fidelity holds. Do not let "it was faster" carry it alone.

## What stays open regardless

No photograph taken through the shipped app has been scored against ground truth, and field
photographs could not be obtained. A successful device session evidences latency, UI behaviour and
parity — it does **not** convert the framing fix's 44.0-point simulated recovery into a field
result. L8 stays worded as "found and fixed, fix unverified in the field".
