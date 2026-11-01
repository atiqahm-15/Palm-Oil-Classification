"""Re-run the Lighting Stress-Test through the *converted* TFLite model (Phase 3, NFR-4).

Phase 3's milestone asks for the lighting-robustness test to be re-run "on the optimized model".
Until now it had only ever been run on the Keras checkpoints (evaluate.py ->
`lighting_stress_test_metrics.csv`), so the published lighting spread — FastViT_T8 1.02 +/- 0.43
points, `results/multiseed_v2_summary.md` — describes a model that is not the one the app ships.

Why that gap is worth closing rather than assuming. float16 conversion halves the mantissa of
every weight, and the conditions this test applies are exactly the ones that push activations
around: `darkened` compresses the input into the bottom of the value range, where relative
precision loss is largest. Whether that matters is an empirical question, and it is cheap to
answer — but it cannot be answered by the conversion spike, which scored 200 clean images only.

What this measures, and how it differs from `export_tflite.py`:

  - The **full** test split (2,722 images), not a 200-image sample. The spike's sample was sized
    to compare TFLite against Keras, and `PROJECT_PLAN.md` §8 warns its accuracy column must not
    be read as an architecture comparison. The number this script writes for `clean` can be, and
    doubles as the full-split TFLite fidelity figure the spike did not produce.
  - All four lighting conditions (clean baseline + darkened / brightened / shadow), from
    `manifest_lighting_stress_test.csv` — the same materialised JPEGs evaluate.py scores, so the
    Keras and TFLite columns are directly comparable.
  - Per-class recall per condition, because the lighting spread is an average and the FYP1
    baseline's failure was a per-class one (unripe recall 0.38 behind a 68% headline).

Duplicate rows: the test split double-counts ~1% of images (`results/duplicate_row_impact.md`,
measured at -0.04 points on the headline). This script scores rows as they appear in the manifest,
which is what evaluate.py does, so its clean column is comparable to every other accuracy in
`results/`. It is not re-deduplicated here; that decision is logged and closed.

Preprocessing is `dataset.py`'s, identical to training and to evaluate.py: decode, bilinear resize
to 224x224 with no crop, then x / 127.5 - 1.

Outputs:
  results/<run-name>/tflite/lighting_metrics__<mode>.csv   per-condition accuracy + per-class recall
  results/tflite_lighting_summary.md                       combined table across every mode scored

Usage (repo root; Windows env is fine — only the TFLite interpreter is needed, not kecam):
    python src/training/check_tflite_lighting.py
        [--run-name fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42]
        [--modes float16 float32 dynamic_int8]
        [--manifest data/processed/manifest_full.csv]
        [--stress-test-manifest data/processed/manifest_lighting_stress_test.csv]
        [--limit N]       # smoke-test on a subset; refuses to overwrite full-run output
        [--force]         # let a --limit run overwrite it anyway
        [--render-only]   # rebuild the markdown from the cached JSON, no re-scoring
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import CLASS_NAMES, IMG_SIZE, _to_model_input, load_manifest_split

CONDITION_ORDER = ["clean", "darkened", "brightened", "shadow"]


def preprocess(path: str) -> np.ndarray:
    """dataset.py's eval-time path, one image at a time (the exports take a batch of 1)."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image, _ = _to_model_input(image, None)
    return image.numpy().astype(np.float32)


def score(interpreter: tf.lite.Interpreter, paths: list[str], labels: list[int]) -> dict:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    preds = np.empty(len(paths), dtype=int)
    for i, path in enumerate(paths):
        interpreter.set_tensor(inp["index"], preprocess(path)[None, ...])
        interpreter.invoke()
        preds[i] = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
        if (i + 1) % 250 == 0:
            print(f"    {i + 1}/{len(paths)}", flush=True)

    truth = np.asarray(labels)
    recalls = {}
    for k, name in enumerate(CLASS_NAMES):
        members = truth == k
        recalls[name] = float((preds[members] == k).mean()) if members.any() else float("nan")
    return {
        "accuracy": float((preds == truth).mean()),
        "n_images": len(paths),
        "recalls": recalls,
    }


def load_conditions(stress_manifest: Path) -> dict[str, tuple[list[str], list[int]]]:
    with stress_manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    conditions: dict[str, tuple[list[str], list[int]]] = {}
    for condition in sorted({r["condition"] for r in rows}):
        members = [r for r in rows if r["condition"] == condition]
        conditions[condition] = (
            [r["path"] for r in members],
            [CLASS_NAMES.index(r["class"]) for r in members],
        )
    return conditions


def write_mode_csv(dest: Path, mode: str, results: dict[str, dict], partial: bool) -> None:
    with dest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "accuracy", "n_images"] + [f"recall_{c}" for c in CLASS_NAMES])
        for condition in CONDITION_ORDER:
            if condition not in results:
                continue
            r = results[condition]
            writer.writerow(
                [condition, r["accuracy"], r["n_images"]]
                + [r["recalls"][c] for c in CLASS_NAMES]
            )
        if partial:
            writer.writerow(["# PARTIAL — --limit was used; not a reportable figure", "", ""])


def spread_points(results: dict[str, dict]) -> float:
    """Worst-case gap in accuracy points across every condition, clean included.

    Matches how the campaign summaries report "lighting spread": one number, max minus min, so
    the TFLite column can be put straight next to the Keras one.
    """
    accs = [r["accuracy"] for r in results.values()]
    return (max(accs) - min(accs)) * 100


def load_keras_lighting(run_name: str) -> dict[str, float]:
    """The same run's Keras numbers, so the conversion cost is a column rather than an exercise.

    `evaluate.py` wrote these for the identical images and the identical seed, which matters: the
    campaign summaries pool three seeds, and comparing a one-seed TFLite figure against a
    three-seed Keras mean would attribute seed noise to the conversion.
    """
    path = Path("results") / run_name / "lighting_stress_test_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        condition = "clean" if row["condition"] == "clean_test_baseline" else row["condition"]
        out[condition] = float(row["accuracy"])
    return out


def render_summary(entries: list[dict], partial: bool) -> str:
    lines = [
        "# Lighting Stress-Test through the converted TFLite models — Phase 3 (NFR-4)",
        "",
        "Closes the half of Phase 3's lighting bullet that `export_tflite.py` did not cover: the",
        "robustness numbers quoted elsewhere in `results/` are Keras-model numbers, and the app",
        "ships a converted model. Full test split (2,722 images per condition), preprocessing",
        "identical to `evaluate.py`, so these columns sit directly beside the Keras ones.",
        "",
        "`clean` is also the full-split TFLite accuracy — the conversion spike measured fidelity on",
        "a 200-image sample only, which `PROJECT_PLAN.md` §8 warns must not be read as a headline.",
        "",
    ]
    if partial:
        lines += [
            "> **PARTIAL RUN — `--limit` was used. These numbers are a smoke test, not reportable.**",
            "",
        ]
    lines += [
        "| Run | Mode | Condition | TFLite acc | Keras acc (same seed) | Δ (points) | unripe | underripe | ripe | overripe |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        keras = load_keras_lighting(entry["run_name"])
        for condition in CONDITION_ORDER:
            if condition not in entry["results"]:
                continue
            r = entry["results"][condition]
            rec = r["recalls"]
            k = keras.get(condition)
            k_cell = f"{k:.4f}" if k is not None else "—"
            d_cell = f"{(r['accuracy'] - k) * 100:+.2f}" if k is not None else "—"
            lines.append(
                f"| {entry['run_name']} | {entry['mode']} | {condition} | {r['accuracy']:.4f} | "
                f"{k_cell} | {d_cell} | "
                + " | ".join(f"{rec[c]:.3f}" for c in CLASS_NAMES)
                + " |"
            )
    lines += ["", "## Lighting spread (max - min accuracy across all four conditions)", "",
              "| Run | Mode | TFLite spread (pts) | Keras spread, same seed (pts) | Clean accuracy |",
              "|---|---|---:|---:|---:|"]
    for entry in entries:
        keras = load_keras_lighting(entry["run_name"])
        k_spread = (
            f"{(max(keras.values()) - min(keras.values())) * 100:.2f}" if keras else "—"
        )
        lines.append(
            f"| {entry['run_name']} | {entry['mode']} | {entry['spread_points']:.2f} | {k_spread} | "
            f"{entry['results']['clean']['accuracy']:.4f} |"
        )
    lines += [
        "",
        "The Keras column is the *same seed*, from that run's `lighting_stress_test_metrics.csv`.",
        "Do not compare these against the campaign figures in `results/multiseed_v2_summary.md`",
        "(FastViT_T8 1.02 +/- 0.43 points, EfficientNetB0 2.45 +/- 0.37, MobileNetV2 5.19 +/- 0.88):",
        "those pool three seeds, and a one-seed-versus-three-seed difference attributes seed noise",
        "to the conversion.",
        "",
        f"Generated by `src/training/check_tflite_lighting.py` on {date.today().isoformat()}.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", default="fastvit_fyp2_blockft_v2_gpu_f50_k3_seed42")
    parser.add_argument("--modes", nargs="+", default=["float16"])
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument(
        "--stress-test-manifest",
        type=Path,
        default=Path("data/processed/manifest_lighting_stress_test.csv"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N images per condition (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="allow a --limit run to overwrite existing full-run output")
    parser.add_argument("--render-only", action="store_true",
                        help="rebuild the markdown from tflite_lighting_summary.json without "
                             "re-scoring. For presentation changes only — it cannot change a number.")
    args = parser.parse_args()

    summary = Path("results") / "tflite_lighting_summary.md"
    if args.render_only:
        cached = json.loads(summary.with_suffix(".json").read_text(encoding="utf-8"))
        summary.write_text(render_summary(cached["entries"], cached["partial"]), encoding="utf-8")
        print(f"Re-rendered {summary} from cached scores")
        return

    if not args.stress_test_manifest.exists():
        raise SystemExit(
            f"{args.stress_test_manifest} not found — run src/data/make_lighting_stress_test.py first."
        )

    clean_paths, clean_labels = load_manifest_split(args.manifest, "test")
    conditions = {"clean": (clean_paths, clean_labels)}
    conditions.update(load_conditions(args.stress_test_manifest))

    partial = args.limit is not None
    if partial:
        conditions = {k: (p[: args.limit], l[: args.limit]) for k, (p, l) in conditions.items()}

    tflite_dir = Path("results") / args.run_name / "tflite"

    # A --limit run produces numbers that are not reportable, and it must not be able to replace
    # numbers that are. The repo convention is that a run never overwrites a previous one; the
    # sibling check_framing_shift.py gates its outputs behind --force and this now matches.
    if partial and not args.force:
        existing = [summary, summary.with_suffix(".json")] + [
            tflite_dir / f"lighting_metrics__{mode}.csv" for mode in args.modes
        ]
        clash = [p for p in existing if p.exists()]
        if clash:
            raise SystemExit(
                "Refusing to overwrite full-run output with partial (--limit) results:\n  "
                + "\n  ".join(str(p) for p in clash)
                + "\nRe-run without --limit, or pass --force if the partial numbers are wanted."
            )
    entries = []
    for mode in args.modes:
        model_path = tflite_dir / f"{args.run_name}__{mode}.tflite"
        if not model_path.exists():
            raise SystemExit(f"{model_path} not found — run export_tflite.py for that run/mode first.")

        print(f"\n=== {args.run_name} / {mode} ===", flush=True)
        interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
        interpreter.allocate_tensors()

        results = {}
        for condition in CONDITION_ORDER:
            if condition not in conditions:
                continue
            paths, labels = conditions[condition]
            print(f"  {condition} (n={len(paths)})", flush=True)
            results[condition] = score(interpreter, paths, labels)
            print(f"  {condition}: accuracy={results[condition]['accuracy']:.4f}", flush=True)

        dest = tflite_dir / f"lighting_metrics__{mode}.csv"
        write_mode_csv(dest, mode, results, partial)
        print(f"  Wrote {dest}")

        entries.append({
            "run_name": args.run_name,
            "mode": mode,
            "results": results,
            "spread_points": spread_points(results),
        })

    summary.write_text(render_summary(entries, partial), encoding="utf-8")
    (summary.with_suffix(".json")).write_text(
        json.dumps({"partial": partial, "entries": entries}, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {summary}")
    for entry in entries:
        print(f"  {entry['mode']}: clean {entry['results']['clean']['accuracy']:.4f}, "
              f"spread {entry['spread_points']:.2f} points")


if __name__ == "__main__":
    main()
