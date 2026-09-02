"""Per-class decision-threshold calibration on already-trained checkpoints.

Motivation: with EfficientNetB0 at 76.45% +/- 0.90, the per-class recalls reproduce that
number exactly - and underripe (0.494) accounts for essentially the whole remaining gap to
the 85% NFR. Every other class already sits at 0.91-0.93. Holding the others fixed, lifting
underripe recall to 0.80 would put accuracy near 84%. So "close the accuracy gap" and "fix
the underripe/ripe boundary" are the same task.

The CORAL ordinal experiment (PROJECT_PLAN.md section 8) showed that trade-off is movable -
underripe recall 0.39 -> 0.47 - but it overshot, wrecking ripe recall, because the loss
function had to guess the operating point during training. Calibration picks that operating
point explicitly, after training, at zero retraining cost.

Method: rescale predicted class probabilities by per-class weights w (argmax over p * w),
the standard prior-correction / threshold-adjustment form. w is fitted by coordinate ascent
over a geometric grid.

METHODOLOGICAL DISCIPLINE - w is fitted on the VALIDATION split only and then applied
unchanged to test. Test is never used for fitting. Both the validation gain (which is
optimistic, being what was optimised) and the test gain (the honest number) are reported,
so any overfitting to validation is visible rather than hidden.

Run directories are treated as immutable (CLAUDE.md convention): output goes to
results/calibration/ and results/calibration_summary.md, not into the original run dirs.

Usage:
    python src/training/calibrate_thresholds.py [--pattern "*_blockft_gpu_seed*"]
        [--objective accuracy|macro_f1]
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

from dataset import CLASS_NAMES, load_manifest_split
from evaluate import load_images_as_batch


def predict_probs(model, paths: list[str], batch_size: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch_size):
        out.append(model.predict(load_images_as_batch(paths[i : i + batch_size]), verbose=0))
    return np.concatenate(out, axis=0)


def score(probs: np.ndarray, y_true: np.ndarray, w: np.ndarray, objective: str) -> float:
    preds = np.argmax(probs * w, axis=1)
    if objective == "macro_f1":
        return float(f1_score(y_true, preds, average="macro", zero_division=0))
    return float((preds == y_true).mean())


def fit_weights(probs, y_true, objective: str, rounds: int = 10) -> tuple[np.ndarray, float]:
    """Coordinate ascent over per-class weights. argmax is invariant to global scaling,
    so the weight vector is only meaningful up to a constant factor."""
    grid = np.exp(np.linspace(-2.0, 2.0, 81))
    w = np.ones(len(CLASS_NAMES))
    best = score(probs, y_true, w, objective)
    for _ in range(rounds):
        improved = False
        for c in range(len(CLASS_NAMES)):
            for cand in grid:
                trial = w.copy()
                trial[c] = cand
                s = score(probs, y_true, trial, objective)
                if s > best + 1e-9:
                    best, w, improved = s, trial, True
        if not improved:
            break
    return w / w.max(), best


def metrics_for(probs, y_true, w) -> dict:
    preds = np.argmax(probs * w, axis=1)
    recalls = recall_score(y_true, preds, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
    return {
        "accuracy": float((preds == y_true).mean()),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        **{f"recall_{c}": float(r) for c, r in zip(CLASS_NAMES, recalls)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", nargs="+", default=["*_blockft_gpu_seed*"],
                        help="one or more globs over results/. Several are accepted because "
                             "results/calibration_summary.md is rewritten from whatever matches, "
                             "so calibrating one architecture alone would drop the others from "
                             "the summary. Runs are grouped by the `architecture` field, so pass "
                             "only ONE fine-tuning depth per architecture — e.g. "
                             "\"*_blockft_gpu_seed*\" plus \"fastvit_*_gpu_k3_seed*\", never both "
                             "FastViT depths at once, or K=2 and K=3 would be averaged together.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--objective", default="accuracy", choices=["accuracy", "macro_f1"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/calibration"))
    args = parser.parse_args()

    pattern_desc = ", ".join(args.pattern)
    run_dirs = sorted(dict.fromkeys(
        d for p in args.pattern for d in Path("results").glob(p)
        if d.is_dir() and (d / "model.keras").exists()
    ))
    if not run_dirs:
        raise SystemExit(f"No runs with a model.keras matched results/{{{pattern_desc}}}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    valid_paths, valid_labels = load_manifest_split(args.manifest, "valid")
    test_paths, test_labels = load_manifest_split(args.manifest, "test")
    y_valid, y_test = np.array(valid_labels), np.array(test_labels)
    print(f"valid={len(valid_paths)} test={len(test_paths)} | objective={args.objective}")

    rows = []
    for run_dir in run_dirs:
        print(f"\n=== {run_dir.name} ===", flush=True)
        config = json.loads((run_dir / "config.json").read_text())
        model = tf.keras.models.load_model(run_dir / "model.keras")

        p_valid = predict_probs(model, valid_paths)
        p_test = predict_probs(model, test_paths)

        w, val_best = fit_weights(p_valid, y_valid, args.objective)
        ones = np.ones(len(CLASS_NAMES))

        rec = {
            "run_name": run_dir.name,
            "architecture": config.get("architecture", "?"),
            "seed": config.get("seed"),
            "objective": args.objective,
            "weights": {c: float(x) for c, x in zip(CLASS_NAMES, w)},
            "valid_baseline": metrics_for(p_valid, y_valid, ones),
            "valid_calibrated": metrics_for(p_valid, y_valid, w),
            "test_baseline": metrics_for(p_test, y_test, ones),
            "test_calibrated": metrics_for(p_test, y_test, w),
        }
        (args.out_dir / f"{run_dir.name}.json").write_text(json.dumps(rec, indent=2))
        rows.append(rec)

        vb, vc = rec["valid_baseline"], rec["valid_calibrated"]
        tb, tc = rec["test_baseline"], rec["test_calibrated"]
        print(f"  weights: { {c: round(float(x), 3) for c, x in zip(CLASS_NAMES, w)} }")
        print(f"  valid acc {vb['accuracy']:.4f} -> {vc['accuracy']:.4f} (fitted here, optimistic)")
        print(f"  test  acc {tb['accuracy']:.4f} -> {tc['accuracy']:.4f} (honest)")
        print(f"  test  underripe recall {tb['recall_underripe']:.3f} -> {tc['recall_underripe']:.3f}"
              f" | ripe {tb['recall_ripe']:.3f} -> {tc['recall_ripe']:.3f}")

    # --- aggregate per architecture, mean +/- std (project rule: never a single run) ---
    archs = list(dict.fromkeys(r["architecture"] for r in rows))
    lines = ["# Per-class threshold calibration", "",
             f"Fitted on the **validation** split (objective: `{args.objective}`), applied unchanged "
             "to test. Test was never used for fitting; the validation gain is optimistic by "
             "construction and is shown alongside so any overfitting is visible.", ""]

    def ms(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    for arch in archs:
        g = [r for r in rows if r["architecture"] == arch]
        lines += [f"## {arch} (n={len(g)})", "",
                  "| Metric | Baseline | Calibrated | Delta |", "|---|---|---|---|"]
        keys = [("Test accuracy (%)", "accuracy", 100),
                ("Test macro F1", "macro_f1", 1)] + \
               [(f"Test recall - {c}", f"recall_{c}", 1) for c in CLASS_NAMES]
        for label, key, scale in keys:
            bm, bs = ms([r["test_baseline"][key] * scale for r in g])
            cm, cs = ms([r["test_calibrated"][key] * scale for r in g])
            d = cm - bm
            digits = 2 if scale == 100 else 3
            lines.append(f"| {label} | {bm:.{digits}f} +/- {bs:.{digits}f} | "
                         f"{cm:.{digits}f} +/- {cs:.{digits}f} | **{d:+.{digits}f}** |")
        vbm, _ = ms([r["valid_baseline"]["accuracy"] * 100 for r in g])
        vcm, _ = ms([r["valid_calibrated"]["accuracy"] * 100 for r in g])
        tbm, _ = ms([r["test_baseline"]["accuracy"] * 100 for r in g])
        tcm, _ = ms([r["test_calibrated"]["accuracy"] * 100 for r in g])
        lines += ["", f"Validation gain {vcm - vbm:+.2f} pts vs test gain {tcm - tbm:+.2f} pts — "
                      f"the difference is the amount that did not generalise.", ""]

    out_md = Path("results/calibration_summary.md")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {args.out_dir}/ and {out_md}")


if __name__ == "__main__":
    main()
