# On-device latency and parity — two physical phones via Firebase Test Lab (NFR-2)

Run 2026-11-01 on Firebase Test Lab (physical devices, Spark plan), project `palm-ripeness-fyp`,
from the APKs built at commit `75f5b87`. Raw output, as copied from the Test Lab Logs tab:
`logcat_PalmBench_SM-A125F.txt` and `logcat_PalmBench_Pixel5.txt` in this folder.

| | Samsung Galaxy A12 | Google Pixel 5 |
|---|---|---|
| Model / codename | SM-A125F / `a12` | `redfin` |
| Class | 2020 budget | 2020 upper-mid-range |
| SoC | MediaTek MT6765 Helio P35 (8 × Cortex-A53) | Qualcomm Snapdragon 765G |
| ABI | arm64-v8a | arm64-v8a |
| Android | 12 (API 31) | 11 (API 30) |

Model `fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42`, bundled mode float16. 15 warm-up + 50 timed
runs per cell, batch 1, interpreter time only. (The Pixel's SoC is from its published spec; the
benchmark records `Build.HARDWARE` below API 31, which is the codename.)

## Latency (DeviceLatencyBenchmark), milliseconds

| Mode | Size (MB) | A12, 1 thr | A12, 4 thr | A12 max | Pixel 5, 1 thr | Pixel 5, 4 thr | Pixel 5 max | Host x86 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float32 | 12.1 | 292.4 | 298.7 | 319.5 | 60.4 | 54.5 | 78.4 | 47.7 |
| **float16 (shipped)** | 6.1 | 289.3 | **283.0** | 395.8 | 56.3 | **53.5** | 71.7 | 47.2 |
| dynamic_int8 | 3.3 | 950.5 | 939.0 | 1018.1 | 81.0 | 83.5 | 91.3 | 114.3 |

"Max" is the 4-thread cell. p95 at 4 threads, float16: A12 328.2, Pixel 5 67.5.

**NFR-2 (≤ 2 s) is met on both phones**: shipped median 53.5–283.0 ms, worst single run 395.8 ms
(budget phone) — at least 5× inside the budget.

- **Hardware spread is 5.3×** between the two phones for the same model and configuration. The
  budget phone is the binding case.
- **The float16 choice survives on ARM, on both devices.** dynamic_int8 is slower on both — 3.3×
  on the A12, 1.6× on the Pixel 5. The concern that int8 would win via XNNPACK does not materialise.
- **float16 buys size, not speed**: within 2–6% of float32 on both phones.
- **Four threads barely help** (≤ 5% on the A12, ~5% on the Pixel). Not investigated.
- **Host x86 latency is a proxy for an upper-mid phone, not for a budget one.** 47 ms on the dev CPU
  is close to the Pixel 5's 54 ms, but understates the A12 by 6×. The ordering between modes held
  on every device; the magnitude depends on the phone.

## Parity (ModelParityTest)

| Check | Galaxy A12 | Pixel 5 | Emulator / Pillow reference |
|---|---|---|---|
| Exact tensor → interpreter, worst diff (tol 1e-4) | 1.9e-6 | 1.9e-6 | within 1e-4 |
| JPEG → app path: argmax agreement | 20/20 | 20/20 | 20/20 |
| JPEG → app path: worst prob diff (tol 0.064) | **0.0629** (01_unripe) | 0.0424 (11_ripe) | 0.042418 (Pillow) |
| Latency, 1 thread, 10 runs, median | 274 ms | 55 ms | — |

- The interpreter reproduces Python to 2e-6 on both phones.
- **The JPEG-decoder difference depends on the phone.** The Pixel 5's decoder reproduces Pillow's
  (and the emulator's) worst case to six significant figures. The Galaxy A12's disagrees with
  TensorFlow's about half as much again (0.063), on a different fixture, passing the 0.064
  tolerance narrowly. Predicted class unchanged on all 20 on both. So the 0.04-point full-split
  decoder figure (Pillow vs TensorFlow, `docs/report_limitations.md` L7) describes phones whose
  decoder behaves like Pillow's, and indicates rather than bounds the effect on the Samsung.

## What this does not establish

- **Two device models, in a data centre** — on mains power in a cooled rack; a phone held in
  plantation sun may throttle further. Within each run the max/median ratio does not rise across the
  sweep (A12: 1.21, 1.07, 1.14, 1.40, 1.06, 1.08; Pixel 5: 1.37, 1.44, 1.78, 1.34, 1.09, 1.09 in run
  order), so there is no sign of throttling during the measurement.
- **Interpreter time only.** JPEG decode, EXIF rotation, the bracket crop and the resize are
  excluded; user-visible time is larger and was not measured. The Analysing screen holds a 400 ms
  floor regardless.
- **No hands-on checks.** Torch, low-light thresholds, and the framing crop on a real camera sensor
  were not exercised — nobody held either phone.
