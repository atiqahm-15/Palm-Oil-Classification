"""Convert a trained checkpoint to TensorFlow Lite and check it is actually deployable (Phase 3 gate).

This is the Phase 3 conversion spike. Its job is NOT to produce a polished mobile benchmark —
that still needs a physical Android device. Its job is to answer one question before any more
work is spent on the selected architecture:

    Does the winning model convert to a *self-contained* TFLite graph, or does it need the
    Flex delegate?

Why that question gates the project. FastViT_T8 comes from keras_cv_attention_models and is
built from custom layers (ChannelAffine et al. — see custom_layers.py). Hybrid CNN-transformer
blocks are exactly where TFLite conversion tends to fall back to `SELECT_TF_OPS`, i.e. the Flex
delegate, which embeds a slice of the full TensorFlow runtime. A model that "converts" via Flex
does NOT run in a stock Android interpreter: the app has to ship `tensorflow-lite-select-tf-ops`,
which adds tens of MB and puts real pressure on the <=100MB NFR. So a Flex-only conversion is a
PARTIAL FAILURE here, not a pass, and is reported as such.

How the Flex verdict is established. Each mode is converted twice at most:

  1. Strict pass — `target_spec.supported_ops = [TFLITE_BUILTINS]` only. If this succeeds, the
     graph is definitively builtin-only and no op introspection is needed to prove it.
  2. Flex pass — only attempted if the strict pass raised. Retried with
     `[TFLITE_BUILTINS, SELECT_TF_OPS]`, and the offending ops are listed best-effort so the
     report can say *which* ops forced it.

If both fail the mode is recorded as a hard conversion failure with the converter's own error.

Conversion-route note — measured, not assumed. On this project's stack (TF 2.20, Keras 3,
Python 3.12) BOTH commonly-documented entry points fail before they reach the graph:
`from_saved_model` over a `model.export()` archive dies walking the trackable graph with
"this __dict__ descriptor does not support '_DictWrapper' objects" (a Python 3.12
`inspect.getattr_static` interaction), and `from_keras_model` surfaces the same fault as
"'NoneType' object is not callable". What works is tracing the model into a concrete function
and using `from_concrete_functions`, which skips the trackable walk. All three are probed once
per run in CONVERT_ROUTES order and the working one is recorded — "how it was converted" is a
methodology detail the report needs, and this is a genuine environment finding rather than a
detail to bury.

Preprocessing note. All three architectures take model input in [-1, 1] (dataset.py's
`_to_model_input`). FastViT's torch-mode rescale happens *inside* its graph, not in the pipeline
(see train_fastvit.py), so the converted model's input contract is identical across the three and
the int8 representative dataset feeds [-1, 1] for all of them.

Outputs land in `results/<run-name>/tflite/` — a new subdirectory, so nothing a training or
evaluation run already wrote is touched. Re-running the export overwrites only its own previous
output. A combined comparison table is written to `results/tflite_export_summary.md`.

Usage (from the repo root, in the WSL GPU venv where kecam is installed):
    python src/training/export_tflite.py \
        --run-names fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42 \
                    efficientnetb0_fyp2_blockft_v2_gpu_f50_k3_seed42
        [--modes float32 float16 dynamic_int8 full_int8]
        [--manifest data/processed/manifest_full.csv]
        [--fidelity-samples 200] [--latency-samples 50] [--num-threads 1]
"""
import argparse
import contextlib
import csv
import io
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import custom_layers  # noqa: F401 — registers kecam's ChannelAffine et al. so load_model() can
                      # deserialize FastViT runs. Guarded inside; kecam lives only in the WSL venv.
import ordinal  # noqa: F401 — same trick for CoralLayer, so ordinal checkpoints stay loadable.
from dataset import CLASS_NAMES, IMG_SIZE, _to_model_input, load_manifest_split

# NFR bar from CLAUDE.md / PROJECT_PLAN.md 2.3. Size is the one this script can genuinely test;
# latency here is a dev-CPU proxy and is NOT the <=2s on-device verdict.
NFR_MAX_SIZE_MB = 100.0

ALL_MODES = ("float32", "float16", "dynamic_int8", "full_int8")

# Ordered by what works on TF 2.20 / Keras 3 / Python 3.12 -- see _new_converter.
CONVERT_ROUTES = ("concrete_function", "saved_model", "from_keras_model")

MODE_NOTES = {
    "float32": "no optimization — the conversion-viability baseline",
    "float16": "float16 weights, float32 compute — usually the safest real win",
    "dynamic_int8": "dynamic-range int8 weights, float activations — no calibration data needed",
    "full_int8": "full-integer via representative dataset, float I/O so it stays drop-in",
}


def load_images_as_batch(paths: list[str]) -> np.ndarray:
    """Decode + resize + rescale to the model's [-1, 1] input contract.

    Deliberately a local copy of evaluate.py's helper rather than an import: importing
    evaluate pulls in matplotlib and seaborn for its plotting, which this script never needs.
    """
    images = []
    for p in paths:
        raw = tf.io.read_file(p)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        images.append(img)
    batch = tf.stack(images)
    batch, _ = _to_model_input(batch, None)
    return batch.numpy().astype(np.float32)


def make_representative_dataset(manifest: Path, n_samples: int, seed: int = 42):
    """Calibration data for full-integer quantization, drawn from *train* — never test.

    Full-int8 quantization fits activation ranges from the data it is shown. Calibrating on the
    test split would leak test statistics into the deployed model and make the quantized test
    accuracy an overestimate — the same class of mistake as fitting thresholds on test.
    """
    paths, _ = load_manifest_split(manifest, "train")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(paths), size=min(n_samples, len(paths)), replace=False)
    sample_paths = [paths[i] for i in picks]

    def gen():
        for i in range(0, len(sample_paths), 8):
            batch = load_images_as_batch(sample_paths[i : i + 8])
            for row in batch:
                yield [row[None, ...]]

    return gen


def _new_converter(model, route: str, exporter):
    """Build a fresh converter via `route`. Converters are single-use — every attempt needs its own.

    `concrete_function` is first in CONVERT_ROUTES because it is the only one that works on this
    project's stack. Both documented alternatives fail on TF 2.20 + Keras 3 + Python 3.12:
    `from_saved_model` (over a `model.export()` archive) dies walking the trackable graph with
    "this __dict__ descriptor does not support '_DictWrapper' objects" — a Python 3.12
    `inspect.getattr_static` interaction — and `from_keras_model` surfaces the same fault as
    "'NoneType' object is not callable". Tracing the model into a concrete function sidesteps the
    trackable walk entirely. The other two are kept as fallbacks so a different TF/Python
    combination is not locked out.

    The batch dimension is pinned to 1: that is what the phone actually does, and it is what the
    interpreter is fed in tflite_predict / measure_tflite_latency.
    """
    if route == "concrete_function":
        run = tf.function(lambda x: model(x, training=False))
        cf = run.get_concrete_function(
            tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="input")
        )
        return tf.lite.TFLiteConverter.from_concrete_functions([cf], run)
    if route == "saved_model":
        return tf.lite.TFLiteConverter.from_saved_model(str(exporter()))
    if route == "from_keras_model":
        return tf.lite.TFLiteConverter.from_keras_model(model)
    raise ValueError(f"unknown route {route!r}")


def pick_route(model, exporter) -> str:
    """Find a conversion entry point that actually works on this machine, once per run.

    Probed with Flex *allowed*, deliberately: this step is only choosing a working API path, and
    a model that genuinely needs Flex must still get past it so that convert_mode's own
    strict-then-Flex logic is what decides the deployability verdict.
    """
    errors = {}
    for route in CONVERT_ROUTES:
        try:
            converter = _new_converter(model, route, exporter)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.convert()
            print(f"Conversion route: {route}")
            return route
        except Exception as exc:
            errors[route] = _truncate(f"{type(exc).__name__}: {exc}", 300)
            print(f"   route {route} unavailable: {errors[route][:120]}")
    raise SystemExit(
        "No TFLite conversion route works in this environment. Tried:\n"
        + "\n".join(f"  {k}: {v}" for k, v in errors.items())
    )


def _configure(converter, mode: str, rep_ds):
    if mode == "float32":
        pass
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "full_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_ds
        # I/O deliberately left float32: an int8-in/int8-out model needs the Android side to
        # replicate the quantization params by hand, which is an easy place to introduce a
        # silent preprocessing bug. Float I/O keeps the app's contract identical to the
        # float model's while still quantizing the weights and activations internally.
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _list_flex_ops(tflite_bytes: bytes) -> list[str]:
    """Best-effort: which ops forced the Flex delegate. Diagnostic only — never load-bearing."""
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tf.lite.experimental.Analyzer.analyze(model_content=tflite_bytes)
        # Analyzer prints ops as e.g. "Op#3 FlexErf(T#1) -> [T#2]", so pull the op name out
        # with a regex — splitting on whitespace leaves the trailing "(T#1" attached.
        found = set(re.findall(r"\bFlex[A-Za-z0-9_]+", buf.getvalue()))
        if found:
            return sorted(found)
    except Exception:
        pass
    try:
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        return sorted({d["op_name"] for d in interp._get_ops_details()
                       if str(d["op_name"]).startswith("Flex")})
    except Exception:
        return []


def convert_mode(model, route: str, exporter, mode: str, rep_ds) -> dict:
    """Strict conversion first; Flex only as a fallback. Returns a result record."""
    result = {"mode": mode, "note": MODE_NOTES[mode]}

    try:
        converter = _new_converter(model, route, exporter)
        _configure(converter, mode, rep_ds)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        blob = converter.convert()
        result.update(converted=True, requires_flex=False, flex_ops=[],
                      convert_route=route, strict_error=None)
        return result | {"_bytes": blob}
    except Exception as strict_exc:
        strict_error = f"{type(strict_exc).__name__}: {strict_exc}"

    try:
        converter = _new_converter(model, route, exporter)
        _configure(converter, mode, rep_ds)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        blob = converter.convert()
        result.update(converted=True, requires_flex=True, flex_ops=_list_flex_ops(blob),
                      convert_route=route, strict_error=_truncate(strict_error))
        return result | {"_bytes": blob}
    except Exception as flex_exc:
        result.update(converted=False, requires_flex=None, flex_ops=[], convert_route=None,
                      strict_error=_truncate(strict_error),
                      flex_error=_truncate(f"{type(flex_exc).__name__}: {flex_exc}"))
        return result


def _truncate(msg: str, limit: int = 1200) -> str:
    msg = " ".join(msg.split())
    return msg if len(msg) <= limit else msg[:limit] + " …[truncated]"


def tflite_predict(tflite_bytes: bytes, batch: np.ndarray, num_threads: int) -> np.ndarray:
    """Run the converted model image-by-image (batch 1 — what the phone actually does)."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=num_threads)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    probs = []
    for row in batch:
        x = row[None, ...].astype(in_detail["dtype"])
        interp.set_tensor(in_detail["index"], x)
        interp.invoke()
        probs.append(interp.get_tensor(out_detail["index"])[0].copy())
    return np.stack(probs)


def measure_tflite_latency(tflite_bytes: bytes, batch: np.ndarray, num_threads: int) -> float:
    """Single-image dev-CPU latency, single-threaded by default.

    A conservative proxy, and explicitly NOT the <=2s NFR verdict — that needs the real device.
    Kept single-threaded so the number is comparable across models and machines rather than
    silently reflecting however many cores happened to be free.
    """
    interp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=num_threads)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    idx = in_detail["index"]

    interp.set_tensor(idx, batch[:1].astype(in_detail["dtype"]))
    interp.invoke()  # warm up

    t0 = time.time()
    for row in batch:
        interp.set_tensor(idx, row[None, ...].astype(in_detail["dtype"]))
        interp.invoke()
    return (time.time() - t0) / len(batch)


def export_run(run_name: str, args) -> list[dict]:
    run_dir = Path("results") / run_name
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found — train or fetch that run first.")

    out_dir = run_dir / "tflite"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}\n{run_name}\n{'=' * 78}")
    model = tf.keras.models.load_model(model_path)
    keras_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Loaded  params={model.count_params():,}  keras file={keras_size_mb:.1f} MB")

    paths, labels = load_manifest_split(args.manifest, "test")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(paths), size=min(args.fidelity_samples, len(paths)), replace=False)
    fid_paths = [paths[i] for i in picks]
    fid_labels = np.array([labels[i] for i in picks])
    fid_batch = load_images_as_batch(fid_paths)

    keras_probs = model.predict(fid_batch, verbose=0)
    keras_preds = np.argmax(keras_probs, axis=1)
    keras_acc = float((keras_preds == fid_labels).mean())
    print(f"Keras reference accuracy on {len(fid_paths)} sampled test images: {keras_acc:.4f}")

    rep_ds = make_representative_dataset(args.manifest, args.calibration_samples, args.seed)

    # The SavedModel is exported lazily and cached: only the `saved_model` route needs it, and
    # on this stack that route never wins, so exporting eagerly would just burn time (and, for
    # FastViT, risk failing on a path nothing ends up using).
    tmp = tempfile.TemporaryDirectory()
    _cache: dict[str, Path] = {}

    def exporter() -> Path:
        if "dir" not in _cache:
            model.export(f"{tmp.name}/saved_model")
            _cache["dir"] = Path(tmp.name) / "saved_model"
        return _cache["dir"]

    route = pick_route(model, exporter)

    records = []
    for mode in args.modes:
        print(f"\n-- {mode} ({MODE_NOTES[mode]})")
        res = convert_mode(model, route, exporter, mode, rep_ds)
        blob = res.pop("_bytes", None)
        res["run_name"] = run_name

        if not res["converted"]:
            print(f"   FAILED to convert, even with Flex.\n   strict: {res['strict_error'][:200]}")
            records.append(res)
            continue

        tflite_path = out_dir / f"{run_name}__{mode}.tflite"
        tflite_path.write_bytes(blob)
        size_mb = len(blob) / (1024 * 1024)

        probs = tflite_predict(blob, fid_batch, args.num_threads)
        preds = np.argmax(probs, axis=1)
        agreement = float((preds == keras_preds).mean())
        tflite_acc = float((preds == fid_labels).mean())
        max_prob_delta = float(np.abs(probs - keras_probs).max())
        latency_s = measure_tflite_latency(blob, fid_batch[: args.latency_samples], args.num_threads)

        res.update(
            file=str(tflite_path),
            size_mb=round(size_mb, 2),
            size_within_nfr=bool(size_mb <= NFR_MAX_SIZE_MB),
            keras_accuracy=round(keras_acc, 4),
            tflite_accuracy=round(tflite_acc, 4),
            accuracy_delta=round(tflite_acc - keras_acc, 4),
            agreement_with_keras=round(agreement, 4),
            max_prob_delta=round(max_prob_delta, 5),
            latency_ms_devcpu=round(latency_s * 1000, 1),
            num_threads=args.num_threads,
        )
        flex = " REQUIRES FLEX" if res["requires_flex"] else " builtin-only"
        print(f"   converted{flex}  {size_mb:.1f} MB  acc {tflite_acc:.4f} "
              f"({res['accuracy_delta']:+.4f})  agree {agreement:.3f}  "
              f"{res['latency_ms_devcpu']:.1f} ms/img")
        if res["requires_flex"] and res["flex_ops"]:
            print(f"   flex ops: {', '.join(res['flex_ops'])}")
        records.append(res)

    tmp.cleanup()

    (out_dir / "export_report.json").write_text(json.dumps(
        {
            "run_name": run_name,
            "keras_model_size_mb": round(keras_size_mb, 2),
            "n_params": int(model.count_params()),
            "fidelity_samples": len(fid_paths),
            "latency_samples": min(args.latency_samples, len(fid_paths)),
            "num_threads": args.num_threads,
            "manifest": str(args.manifest),
            "nfr_max_size_mb": NFR_MAX_SIZE_MB,
            "latency_caveat": "dev CPU, single-threaded — NOT the Phase 3 on-device <=2s verdict",
            "modes": records,
        },
        indent=2,
    ), encoding="utf-8")
    return records


def write_summary(records: list[dict], out_path: Path, args) -> None:
    header = ["run_name", "mode", "converted", "requires_flex", "size_mb", "size_within_nfr",
              "keras_accuracy", "tflite_accuracy", "accuracy_delta", "agreement_with_keras",
              "max_prob_delta", "latency_ms_devcpu"]

    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    lines = [
        "# TFLite export — Phase 3 conversion spike",
        "",
        "Answers one question: does each model convert to a **self-contained** TFLite graph, or",
        "does it need the Flex delegate (`SELECT_TF_OPS`)? A Flex-only conversion is a **partial",
        "failure** — it will not run in a stock Android interpreter without shipping a slice of the",
        "full TF runtime, which adds tens of MB against the <=100MB NFR.",
        "",
        f"Fidelity measured on {args.fidelity_samples} randomly sampled test images (seed {args.seed}); "
        f"latency on {args.latency_samples} of them, single-image, {args.num_threads}-thread dev CPU.",
        "**Latency here is a proxy, not the <=2s NFR verdict** — that still needs a physical device.",
        "",
        "| Run | Mode | Converted | Builtin-only | Size (MB) | <=100MB | TFLite acc | Δ vs Keras | Agreement | Latency (ms) |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in records:
        if not r.get("converted"):
            lines.append(f"| {r['run_name']} | {r['mode']} | **NO** | — | — | — | — | — | — | — |")
            continue
        builtin = "yes" if not r["requires_flex"] else "**NO — Flex**"
        lines.append(
            f"| {r['run_name']} | {r['mode']} | yes | {builtin} | {r['size_mb']:.1f} | "
            f"{'yes' if r['size_within_nfr'] else '**NO**'} | {r['tflite_accuracy']:.4f} | "
            f"{r['accuracy_delta']:+.4f} | {r['agreement_with_keras']:.3f} | {r['latency_ms_devcpu']:.1f} |"
        )

    failures = [r for r in records if not r.get("converted")]
    flexed = [r for r in records if r.get("converted") and r.get("requires_flex")]
    if failures or flexed:
        lines += ["", "## Problems", ""]
        for r in failures:
            lines.append(f"- **{r['run_name']} / {r['mode']} — did not convert.** "
                         f"Strict: `{r.get('strict_error', '')[:400]}`")
        for r in flexed:
            ops = ", ".join(r["flex_ops"]) or "not identified"
            lines.append(f"- **{r['run_name']} / {r['mode']} — converted only with Flex.** "
                         f"Ops forcing it: {ops}")
    else:
        lines += ["", "All requested modes converted builtin-only. No Flex delegate needed.", ""]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path} and {out_path.with_suffix('.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-names", nargs="+", required=True,
                        help="one or more results/<run-name> directories containing model.keras")
    parser.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=ALL_MODES)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--fidelity-samples", type=int, default=200,
                        help="test images used to compare TFLite vs Keras predictions")
    parser.add_argument("--latency-samples", type=int, default=50)
    parser.add_argument("--calibration-samples", type=int, default=200,
                        help="train images used to calibrate full_int8 quantization")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="TFLite interpreter threads; 1 keeps latency comparable across machines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", type=Path, default=Path("results/tflite_export_summary.md"))
    args = parser.parse_args()

    all_records = []
    for run_name in args.run_names:
        all_records += export_run(run_name, args)

    write_summary(all_records, args.summary, args)

    hard_fail = [r for r in all_records if not r.get("converted")]
    flexed = [r for r in all_records if r.get("converted") and r.get("requires_flex")]
    print("\n=== verdict ===")
    print(f"{len(all_records) - len(hard_fail)}/{len(all_records)} modes converted; "
          f"{len(flexed)} needed the Flex delegate; {len(hard_fail)} failed outright.")
    if not hard_fail and not flexed:
        print("All builtin-only — the selected architecture is deployable as-is.")


if __name__ == "__main__":
    main()
