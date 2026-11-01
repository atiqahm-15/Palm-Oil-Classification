"""Bundle an exported .tflite for the Android app, plus the fixtures that prove the app matches Python.

Phase 4, step 1. The export (export_tflite.py) answers "does the model convert and stay accurate";
this answers "what exactly must the Kotlin side do to feed it", and writes that down as data the app
reads and tests against, rather than as prose someone re-types into Kotlin.

Why this exists. The model's input contract is easy to get silently wrong on Android and nothing
raises when it is: feed [0, 255] or [0, 1] instead of [-1, 1], apply ImageNet mean/std a second time
(FastViT's normalisation is already inside the graph — see train_fastvit.py), center-crop instead of
squashing, or read the classes in the wrong order, and the model still returns four confident
softmax scores. The only defence is comparing the app's numbers to Python's on the same inputs.

Outputs (paths relative to the repo root):

  app/src/main/assets/model.tflite          copy of the chosen export (gitignored, *.tflite)
  app/src/main/assets/labels.txt            class names, one per line, in output-index order
  app/src/main/assets/model_contract.json   input/output spec, preprocessing, provenance, sha256

  app/src/test/resources/resize_parity.json
      JVM unit-test fixture for the Kotlin resize. The input is a deterministic pattern the test
      regenerates from a formula (see pattern_pixel), so no image is stored and the file can be
      committed. Expected values are tf.image.resize outputs at sampled positions.

  app/src/androidTest/assets/fixtures/
      <id>.jpg          stratified test-split crops (gitignored — dataset images stay out of git)
      <id>.input.bin    the exact float32 [-1, 1] tensor Python fed the model (gitignored)
      reference.json    expected probabilities per image (tracked — numbers, not images)

  app/src/androidTest/assets/benchmark/
      <mode>.tflite     the other quantization modes, for the on-device latency sweep (gitignored)
      manifest.json     which modes are present, their sizes, and the host latency to beat (tracked)

  The benchmark assets exist because the deployment artefact was chosen on x86 evidence. The spike
  measured float16 at 47.2ms against dynamic_int8's 114.3ms on a dev CPU and picked float16 — but
  int8 commonly wins on ARM via XNNPACK/NNAPI, so that ordering is exactly the kind that reverses
  on a phone. DeviceLatencyBenchmark re-runs the comparison on the target hardware.

  Two checks the instrumented test can then make, separating the two failure sources:
    - input.bin -> interpreter -> probs must match to ~1e-4: proves the interpreter wiring.
    - jpg -> Kotlin decode/resize/scale -> interpreter -> probs: proves the preprocessing. JPEG
      decoders differ slightly between TF and Android, so this one is checked with a tolerance and
      on argmax agreement rather than bit-exactness.

Usage (from the repo root; Windows env is fine — only the TFLite interpreter is needed, not kecam):
    python src/training/make_app_bundle.py
        [--run-name fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42] [--mode float16]
        [--per-class 5] [--seed 42]
"""
import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from dataset import CLASS_NAMES, IMG_SIZE, _to_model_input, load_manifest_split

APP_DIR = Path("app")
ASSETS = APP_DIR / "src/main/assets"
UNIT_RES = APP_DIR / "src/test/resources"
FIXTURES = APP_DIR / "src/androidTest/assets/fixtures"
BENCHMARK = APP_DIR / "src/androidTest/assets/benchmark"

# Sizes for the resize-parity fixture: one downscale and one upscale, both non-square and with
# odd dimensions, since those are where half-pixel-centre off-by-ones show up.
PARITY_CASES = [(389, 517), (61, 97)]  # (height, width)
PARITY_SAMPLES = 400


def pattern_pixel(y: np.ndarray, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Deterministic uint8 test pattern. Mirrored exactly in the Kotlin unit test.

    Kept to small-integer arithmetic so it cannot overflow a 32-bit Kotlin Int for any size used
    here. The (x * y) % 7 term breaks up the pure gradient so interpolation errors are visible.
    """
    return (x * 31 + y * 17 + c * 101 + (x * y) % 7 * 23) % 256


def reference_preprocess(rgb_uint8: np.ndarray) -> np.ndarray:
    """The training pipeline's preprocessing, applied to an already-decoded HxWx3 uint8 image.

    Mirrors dataset._decode_and_resize + _to_model_input: bilinear resize of the whole frame to
    224x224 (aspect ratio NOT preserved, no crop), then x / 127.5 - 1.
    """
    img = tf.image.resize(tf.convert_to_tensor(rgb_uint8), [IMG_SIZE, IMG_SIZE])
    img, _ = _to_model_input(img, None)
    return img.numpy().astype(np.float32)


def write_resize_parity(seed: int) -> Path:
    rng = np.random.default_rng(seed)
    cases = []
    for h, w in PARITY_CASES:
        yy, xx, cc = np.meshgrid(np.arange(h), np.arange(w), np.arange(3), indexing="ij")
        src = pattern_pixel(yy, xx, cc).astype(np.uint8)
        resized = tf.image.resize(src, [IMG_SIZE, IMG_SIZE]).numpy()  # [0, 255] float, pre-scale
        idx = rng.integers(0, [IMG_SIZE, IMG_SIZE, 3], size=(PARITY_SAMPLES, 3))
        cases.append({
            "src_height": h,
            "src_width": w,
            "samples": [[int(y), int(x), int(c), round(float(resized[y, x, c]), 5)] for y, x, c in idx],
        })
    out = UNIT_RES / "resize_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "about": "tf.image.resize (bilinear, half_pixel_centers, no antialias) of pattern_pixel "
                 "images, sampled as [y, x, channel, value in 0..255 before the [-1,1] rescale]. "
                 "Generated by src/training/make_app_bundle.py.",
        "pattern": "(x*31 + y*17 + c*101 + (x*y)%7*23) % 256",
        "target_size": IMG_SIZE,
        "cases": cases,
    }, indent=1), encoding="utf-8")
    return out


def run_tflite(blob: bytes, batch: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_content=blob, num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    probs = []
    for row in batch:
        interp.set_tensor(inp["index"], row[None, ...])
        interp.invoke()
        probs.append(interp.get_tensor(out["index"])[0].copy())
    return np.stack(probs)


def check_io_shape(blob: bytes) -> tuple[dict, dict]:
    interp = tf.lite.Interpreter(model_content=blob)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    expected_in = [1, IMG_SIZE, IMG_SIZE, 3]
    if list(inp["shape"]) != expected_in or inp["dtype"] != np.float32:
        raise SystemExit(f"Unexpected input {inp['shape']} {inp['dtype']}; the contract assumes "
                         f"{expected_in} float32.")
    if list(out["shape"]) != [1, len(CLASS_NAMES)] or out["dtype"] != np.float32:
        raise SystemExit(f"Unexpected output {out['shape']} {out['dtype']}.")
    return inp, out


def write_fixtures(blob: bytes, manifest: Path, per_class: int, seed: int) -> tuple[Path, float]:
    """Stratified test-split sample -> jpg + exact input tensor + expected probabilities."""
    paths, labels = load_manifest_split(manifest, "test")
    rng = np.random.default_rng(seed)
    picks = []
    for k in range(len(CLASS_NAMES)):
        members = [i for i, lab in enumerate(labels) if lab == k]
        picks += sorted(rng.choice(members, size=min(per_class, len(members)), replace=False).tolist())

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for stale in FIXTURES.glob("*"):
        stale.unlink()

    tensors = []
    for i in picks:
        rgb = tf.io.decode_jpeg(tf.io.read_file(paths[i]), channels=3).numpy()
        tensors.append(reference_preprocess(rgb))
    batch = np.stack(tensors)
    probs = run_tflite(blob, batch)

    images = []
    for n, (i, x, p) in enumerate(zip(picks, batch, probs)):
        fid = f"{n:02d}_{CLASS_NAMES[labels[i]]}"
        shutil.copyfile(paths[i], FIXTURES / f"{fid}.jpg")
        # Little-endian float32, NHWC without the batch dim — what a Java/Kotlin
        # ByteBuffer.order(LITTLE_ENDIAN).asFloatBuffer() reads back directly.
        x.astype("<f4").tofile(FIXTURES / f"{fid}.input.bin")
        images.append({
            "id": fid,
            "source_path": paths[i],
            "true_class": CLASS_NAMES[labels[i]],
            "predicted_class": CLASS_NAMES[int(np.argmax(p))],
            "probabilities": [round(float(v), 6) for v in p],
        })

    acc = float(np.mean([im["true_class"] == im["predicted_class"] for im in images]))
    spread = measure_decoder_spread(blob, [FIXTURES / f"{im['id']}.jpg" for im in images], probs)

    # The jpg-path tolerance is measured, not assumed. Android's JPEG decoder is not TensorFlow's,
    # and two libjpeg configurations legitimately disagree by a few levels per pixel, which the
    # model then amplifies. Pillow stands in for "some other decoder" to put a number on that,
    # and the tolerance is that number with headroom. The check this test actually has to fail is
    # wrong preprocessing — wrong value range, a centre crop, swapped channels — and every one of
    # those is off by whole units, not hundredths.
    tolerance_from_jpg = round(max(0.02, spread["max_abs_prob_diff"] * 1.5), 3)

    (FIXTURES / "reference.json").write_text(json.dumps({
        "about": "Expected TFLite outputs for the bundled model on these inputs, from Python. "
                 "Generated by src/training/make_app_bundle.py; *.jpg and *.input.bin are "
                 "gitignored and must be regenerated locally.",
        "class_names": CLASS_NAMES,
        "input_bin_format": f"float32 little-endian, {IMG_SIZE}x{IMG_SIZE}x3 HWC, values in [-1, 1]",
        "tolerance_from_input_bin": 1e-4,
        "tolerance_from_jpg": tolerance_from_jpg,
        "decoder_spread": spread,
        "images": images,
    }, indent=1), encoding="utf-8")
    return FIXTURES / "reference.json", acc


def measure_decoder_spread(blob: bytes, jpg_paths: list[Path], tf_probs: np.ndarray) -> dict:
    """Re-run the fixtures through a second JPEG decoder and report how far the outputs move.

    The app cannot use tf.io.decode_jpeg, so some decoder disagreement is unavoidable rather than
    a defect. Pillow is an independent libjpeg configuration, so the spread it produces bounds the
    order of magnitude the Android decoder can be expected to land in — and, being measured here,
    it does not have to be guessed in the instrumented test.
    """
    pil_probs = run_tflite(blob, np.stack([
        reference_preprocess(np.array(Image.open(p).convert("RGB"))) for p in jpg_paths
    ]))
    diff = np.abs(pil_probs - tf_probs)
    return {
        "about": "Same JPEGs, decoded with Pillow instead of tf.io.decode_jpeg, through the same "
                 "model. Bounds how much a non-TensorFlow decoder (such as Android's) moves the "
                 "probabilities; the predicted class is what must not move.",
        "comparison_decoder": f"Pillow {Image.__version__}",
        "max_abs_prob_diff": round(float(diff.max()), 6),
        "mean_abs_prob_diff": round(float(diff.mean()), 6),
        "argmax_flips": int((pil_probs.argmax(axis=1) != tf_probs.argmax(axis=1)).sum()),
        "images": len(jpg_paths),
    }


def write_benchmark_assets(run_name: str, modes: list[str], bundled_mode: str) -> tuple[Path, list[str]]:
    """Stage the quantization variants the device benchmark sweeps, plus the host numbers to beat.

    Only latency and size are at stake here, so the variants are staged as-is — their *accuracy*
    was already settled by export_tflite.py, and full_int8 in particular is excluded by default
    because FastViT collapses to 0.260 under it. A mode that is not worth shipping is not worth
    benchmarking, however fast it is.

    The host latency is carried across from `export_report.json` so the device report can print
    the x86 figure beside the ARM one rather than asking the reader to hold two documents open.
    """
    tflite_dir = Path("results") / run_name / "tflite"
    report_path = tflite_dir / "export_report.json"
    host = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        host = {m["mode"]: m for m in report.get("modes", []) if m.get("converted")}

    BENCHMARK.mkdir(parents=True, exist_ok=True)
    for stale in BENCHMARK.glob("*"):
        stale.unlink()

    entries, staged = [], []
    for mode in modes:
        src = tflite_dir / f"{run_name}__{mode}.tflite"
        if not src.exists():
            print(f"  skipping {mode}: {src} not found")
            continue
        blob = src.read_bytes()
        shutil.copyfile(src, BENCHMARK / f"{mode}.tflite")
        staged.append(mode)
        entry = {
            "mode": mode,
            "asset": f"benchmark/{mode}.tflite",
            "size_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "is_bundled_model": mode == bundled_mode,
        }
        if mode in host:
            entry["host_latency_ms_devcpu"] = host[mode].get("latency_ms_devcpu")
            entry["host_num_threads"] = host[mode].get("num_threads")
            entry["tflite_accuracy_sample"] = host[mode].get("tflite_accuracy")
        entries.append(entry)

    (BENCHMARK / "manifest.json").write_text(json.dumps({
        "about": "Quantization variants staged for the on-device latency sweep "
                 "(DeviceLatencyBenchmark). Generated by src/training/make_app_bundle.py; the "
                 "*.tflite files are gitignored and must be regenerated locally.",
        "source_run": run_name,
        "bundled_mode": bundled_mode,
        "generated": date.today().isoformat(),
        "nfr_latency_ms": 2000,
        "host_latency_caveat": "host_latency_ms_devcpu is single-threaded x86 dev CPU from the "
                               "conversion spike. It is the number the device run exists to "
                               "replace, not a target.",
        "accuracy_caveat": "tflite_accuracy_sample is the spike's 200-image figure — a "
                           "conversion-fidelity check, not an architecture comparison.",
        "modes": entries,
    }, indent=1), encoding="utf-8")
    return BENCHMARK / "manifest.json", staged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42")
    parser.add_argument("--mode", default="float16")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-modes", nargs="*",
                        default=["float32", "float16", "dynamic_int8"],
                        help="quantization variants to stage for the on-device latency sweep. "
                             "full_int8 is excluded on purpose: it is the fastest and smallest "
                             "and it is unusable (FastViT 0.260). Pass an empty list to skip.")
    args = parser.parse_args()

    src = Path("results") / args.run_name / "tflite" / f"{args.run_name}__{args.mode}.tflite"
    if not src.exists():
        raise SystemExit(f"{src} not found — run export_tflite.py for that run first.")
    blob = src.read_bytes()
    inp, out = check_io_shape(blob)

    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ASSETS / "model.tflite")
    (ASSETS / "labels.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    contract = {
        "model_file": "model.tflite",
        "labels_file": "labels.txt",
        "source_run": args.run_name,
        "source_mode": args.mode,
        "source_file": src.as_posix(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "generated": date.today().isoformat(),
        "input": {
            "name": inp["name"],
            "shape": [int(v) for v in inp["shape"]],
            "dtype": "float32",
            "layout": "NHWC, RGB channel order",
            "value_range": [-1.0, 1.0],
            "scaling": "x / 127.5 - 1, applied to 0..255 pixel values after resizing",
            "resize": {
                "target": [IMG_SIZE, IMG_SIZE],
                "method": "bilinear",
                "half_pixel_centers": True,
                "antialias": False,
                "aspect_ratio": "not preserved: the whole frame is squashed to 224x224, no crop",
            },
            "orientation": "apply EXIF orientation before resizing so the bunch is upright; the "
                           "training crops carry no EXIF rotation",
            "do_not": "apply ImageNet mean/std — FastViT's normalisation is inside the graph",
        },
        "output": {
            "name": out["name"],
            "shape": [int(v) for v in out["shape"]],
            "dtype": "float32",
            "meaning": "softmax probabilities, one per class, in labels.txt order; sum to 1",
            "calibration": "uncalibrated — report as 'model confidence', not probability of correctness",
        },
        "class_names": CLASS_NAMES,
    }
    (ASSETS / "model_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Bundled {src} -> {ASSETS / 'model.tflite'} ({len(blob) / 2**20:.1f} MB)")

    parity = write_resize_parity(args.seed)
    print(f"Wrote {parity}")
    ref, acc = write_fixtures(blob, args.manifest, args.per_class, args.seed)
    print(f"Wrote {ref} ({args.per_class * len(CLASS_NAMES)} images, fixture accuracy {acc:.2f})")

    if args.benchmark_modes:
        bench, staged = write_benchmark_assets(args.run_name, args.benchmark_modes, args.mode)
        print(f"Wrote {bench} (modes staged: {', '.join(staged) or 'none'})")


if __name__ == "__main__":
    main()
