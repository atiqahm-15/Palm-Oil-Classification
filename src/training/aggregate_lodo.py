"""Aggregate the leave-one-dataset-out runs into per-fold mean +/- std.

LODO is the project's primary generalization evidence (SYSTEM_ARCHITECTURE.md section 5):
each source held out entirely in turn, tested purely on the held-out source. This script
turns the per-fold replicate runs into the table the report needs, and states the
generalization gap against the main-split baseline.

Two traps this handles explicitly, both of which would silently corrupt the summary:

1. **Zero-support classes.** malaysia_outdoor contains no underripe images, so sklearn
   reports recall 0.000 for it with support 0. That is "undefined", not "the model scored
   zero" - averaging it in would understate that fold and the overall mean. Rows with
   support 0 are parsed as None and excluded from every aggregate.
2. **ROC-AUC that is legitimately nan.** One-vs-rest ROC-AUC is undefined when a class is
   missing from y_true, so the malaysia fold has no AUC. nan is dropped rather than
   propagated through the mean.

Usage:
    python src/training/aggregate_lodo.py [--pattern "*_lodo_*_seed*"]
        [--baseline-accuracy 76.45] [--baseline-label "main split, EfficientNetB0 n=3"]
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

from dataset import CLASS_NAMES


def parse_report_with_support(path: Path) -> dict:
    """Per-class precision/recall/f1, but None where support == 0 (class absent from the fold)."""
    out: dict[str, float | None] = {}
    if not path.exists():
        return out
    wanted = {c: c for c in CLASS_NAMES}
    wanted["macro avg"] = "macro"
    for line in path.read_text().splitlines():
        for label, key in wanted.items():
            m = re.match(r"\s*" + re.escape(label) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
            if not m:
                continue
            support = int(m.group(4))
            absent = support == 0 and key != "macro"
            out[f"precision_{key}"] = None if absent else float(m.group(1))
            out[f"recall_{key}"] = None if absent else float(m.group(2))
            out[f"f1_{key}"] = None if absent else float(m.group(3))
            out[f"support_{key}"] = support
    return out


def load_run(run_dir: Path) -> dict | None:
    config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.csv"
    if not (config_path.exists() and metrics_path.exists()):
        return None
    config = json.loads(config_path.read_text())
    with metrics_path.open(newline="") as f:
        metrics = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}

    name = run_dir.name
    m = re.match(r"^(?P<arch>.+?)_lodo_(?P<fold>.+)_seed(?P<seed>\d+)$", name)
    if not m:
        return None

    auc = metrics.get("roc_auc_ovr")
    row = {
        "run_name": name,
        "architecture": config.get("architecture", m.group("arch")),
        "fold": m.group("fold"),
        "seed": int(m.group("seed")),
        "accuracy": metrics.get("accuracy"),
        # nan is a legitimate outcome (class missing from the held-out source), not an error.
        "roc_auc_ovr": None if auc is None or math.isnan(auc) else auc,
        "compute_platform": (config.get("compute_platform") or {}).get("device", "unknown"),
    }
    report = parse_report_with_support(run_dir / "classification_report.txt")
    row.update(report)

    # sklearn's own "macro avg" divides by the full label set, so a fold whose held-out
    # source lacks a class (malaysia_outdoor has no underripe) gets a macro dragged toward
    # zero by a class that was never testable: 0.478 reported vs 0.637 over the 3 classes
    # actually present. Recompute macro over present classes so folds stay comparable.
    present_f1 = [report[f"f1_{c}"] for c in CLASS_NAMES if report.get(f"f1_{c}") is not None]
    row["n_classes_present"] = len(present_f1)
    row["f1_macro_present"] = statistics.mean(present_f1) if present_f1 else None
    return row


def mean_std(values) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt(vals, scale=100, digits=2) -> str:
    m, s, n = mean_std(vals)
    if n == 0:
        return "n/a"
    if n == 1:
        return f"{m * scale:.{digits}f} (n=1)"
    return f"{m * scale:.{digits}f} +/- {s * scale:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="*_lodo_*_seed*")
    parser.add_argument("--baseline-accuracy", type=float, default=76.45,
                        help="main-split accuracy (%%) to measure the generalization gap against")
    parser.add_argument("--baseline-label", default="main split, EfficientNetB0 n=3")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/lodo"))
    args = parser.parse_args()

    runs = [r for r in (load_run(d) for d in sorted(Path("results").glob(args.pattern)) if d.is_dir()) if r]
    if not runs:
        raise SystemExit(f"No complete LODO runs matched results/{args.pattern}")

    platforms = sorted({r["compute_platform"] for r in runs})
    if len(platforms) > 1:
        raise SystemExit(f"REFUSING: runs span backends {platforms}; narrow --pattern to one.")

    folds = sorted({r["fold"] for r in runs})
    print(f"Pooling {len(runs)} run(s) across {len(folds)} folds: {', '.join(folds)}")

    fieldnames = list(dict.fromkeys(k for r in runs for k in r))
    per_run = args.out_prefix.with_name(args.out_prefix.name + "_per_run.csv")
    per_run.parent.mkdir(parents=True, exist_ok=True)
    with per_run.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(runs)

    arch = runs[0]["architecture"]
    lines = [f"# Leave-one-dataset-out (LODO) results — {arch}", "",
             "Each source held out entirely in turn: trained on the other 3, tested purely on the "
             "held-out source. Cells are mean +/- sample std over seeds of an identical recipe.", "",
             f"Compute backend: {platforms[0]}.", "",
             "Macro F1 is computed over the classes actually present in each held-out source, not "
             "over all 4 — sklearn's own macro average divides by the full label set and so "
             "penalises a fold for a class it could never have been tested on.", "",
             "| Held-out source | n | Classes | Test accuracy (%) | ROC-AUC (OvR) | Macro F1 |",
             "|---|---:|---:|---|---|---|"]

    fold_means = []
    for fold in folds:
        g = [r for r in runs if r["fold"] == fold]
        acc_m, _, _ = mean_std([r["accuracy"] for r in g])
        fold_means.append(acc_m)
        n_cls = g[0].get("n_classes_present", len(CLASS_NAMES))
        lines.append(f"| {fold} | {len(g)} | {n_cls} | {fmt([r['accuracy'] for r in g])} | "
                     f"{fmt([r['roc_auc_ovr'] for r in g], 1, 4)} | "
                     f"{fmt([r.get('f1_macro_present') for r in g], 1, 4)} |")

    overall = statistics.mean([m for m in fold_means if not math.isnan(m)]) * 100
    lines.append(f"| **Mean across folds** | | | **{overall:.2f}** | | |")

    lines += ["", "## Per-class recall by fold", "",
              "A blank cell means the held-out source contains no images of that class "
              "(support 0) — undefined, not zero.", "",
              "| Held-out source | " + " | ".join(CLASS_NAMES) + " |",
              "|---|" + "---|" * len(CLASS_NAMES)]
    for fold in folds:
        g = [r for r in runs if r["fold"] == fold]
        cells = [fmt([r.get(f"recall_{c}") for r in g], 1, 3) for c in CLASS_NAMES]
        cells = ["" if c == "n/a" else c for c in cells]
        lines.append(f"| {fold} | " + " | ".join(cells) + " |")

    gap = args.baseline_accuracy - overall
    lines += ["", "## Generalization gap", "",
              f"- Main-split baseline ({args.baseline_label}): **{args.baseline_accuracy:.2f}%**",
              f"- LODO mean across {len(folds)} folds: **{overall:.2f}%**",
              f"- **Gap: {gap:.2f} points.**", "",
              "The main-split number is measured on unseen *photos* from sources the model also "
              "trained on; the LODO number is measured on an entirely unseen source. The gap is the "
              "part of main-split accuracy that does not survive a change of camera, site, and "
              "labelling convention — the honest estimate of real-world deployment performance.", "",
              "Fold training-set sizes differ substantially (holding out bunch_piles leaves ~4.7k "
              "training images vs ~15.1k for malaysia_outdoor), so a weak bunch_piles fold "
              "confounds domain shift with reduced training data. The other folds, which keep most "
              "of the corpus, isolate domain shift more cleanly."]

    out_md = args.out_prefix.with_name(args.out_prefix.name + "_summary.md")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {per_run} and {out_md}")


if __name__ == "__main__":
    main()
