# Device session checklist — the one thing that needs a real phone

Phase 3 and Phase 4 each have exactly one item left that no amount of simulation closes, and they
close in the same sitting. This is the run sheet for it, written before the session so nothing is
decided while a phone is plugged in and the clock is running.

Everything else in both phases is done or closeable at the desk — see `PROJECT_PLAN.md` §0.

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
      OOM; the decode cap came down from 24M to 4M pixels.
- [ ] **Review scrim**: photograph something bright and confirm the title and subtitle stay
      readable over it.

## What to write down afterwards

1. `PROJECT_PLAN.md` §0 — Phase 3 to Complete if the sweep passes, Phase 4's "not yet seen on a
   phone" struck.
2. `docs/report_limitations.md` **L4** — currently says the latency half is outstanding. If the
   benchmark passes, this section is rewritten from "unverified" to a measurement, the same way its
   conversion half was on 2026-09-02.
3. `docs/report_limitations.md` **L7** — scopes the accuracy figures to emulator-only testing.
4. If dynamic_int8 wins on ARM: this is a **decision, not a fix**. Log it in §8 with the accuracy
   column from `results/tflite_export_summary.md` beside the latency, and re-bundle with
   `--mode dynamic_int8` only if fidelity holds. Do not let "it was faster" carry it alone.

## What stays open regardless

No photograph taken through the shipped app has been scored against ground truth, and field
photographs could not be obtained. A successful device session evidences latency, UI behaviour and
parity — it does **not** convert the framing fix's 44.0-point simulated recovery into a field
result. L8 stays worded as "found and fixed, fix unverified in the field".
