"""Aggregate repeated training runs into mean +/- std, per architecture.

Why: PROJECT_PLAN.md section 8 records that two MobileNetV2 runs differing by exactly one
unfrozen layer produced a 20-point underripe-recall swing - run-to-run variance, not an
effect. Every single-run comparison in this project is therefore uninterpretable on its
own, including MobileNetV2 (73.73%) vs EfficientNetB0 (73.18%). This script turns a set of
replicate runs into the mean +/- std table the report needs, and states plainly whether an
observed architecture gap is larger than the noise it sits in.

Reads each run's existing artifacts (config.json, metrics.csv, classification_report.txt,
lighting_stress_test_metrics.csv) - it does not retrain or re-evaluate anything.

Usage:
    python src/training/aggregate_seeds.py [--pattern "*_fyp2_blockft*"]
        [--out-prefix results/multiseed]
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from dataset import CLASS_NAMES

# Metrics pulled straight from metrics.csv, in report order.
SCALAR_METRICS = ["accuracy", "roc_auc_ovr"]
COST_METRICS = ["n_params", "model_size_mb_prequant", "avg_inference_s_devcpu"]


def parse_classification_report(path: Path) -> dict[str, float]:
    """Pull per-class recall/precision/f1 and macro avg out of sklearn's text report."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    wanted = {c: c for c in CLASS_NAMES}
    wanted["macro avg"] = "macro"
    for line in path.read_text().splitlines():
        for label, key in wanted.items():
            # Anchor on the label followed by exactly 3 float columns (precision/recall/f1)
            # and the integer support, so "macro avg" can't also match "weighted avg".
            m = re.match(r"\s*" + re.escape(label) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s*$", line)
            if m:
                out["precision_" + key] = float(m.group(1))
                out["recall_" + key] = float(m.group(2))
                out["f1_" + key] = float(m.group(3))
    return out


def parse_lighting(path: Path) -> dict[str, float]:
    """Per-condition accuracy plus the spread the 'stable across lighting' NFR cares about."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {"lighting_" + r["condition"]: float(r["accuracy"]) for r in rows}
    accs = [float(r["accuracy"]) for r in rows]
    if accs:
        out["lighting_spread_pts"] = (max(accs) - min(accs)) * 100
    return out


def load_run(run_dir: Path) -> dict | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    if not (config_path.exists() and metrics_path.exists()):
        return None  # incomplete/aborted run - config.json isn't written until training finishes
    config = json.loads(config_path.read_text())
    with metrics_path.open(newline="") as f:
        metrics = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}

    row = {
        "run_name": run_dir.name,
        "architecture": config.get("architecture", "?"),
        "seed": config.get("seed"),
        "finetune_blocks": config.get("finetune_blocks"),
        # Runs predating schedule.py carry no recipe_version; they are all v1 (10+10
        # epochs, constant 1e-5 fine-tuning LR), so default rather than guess.
        "recipe_version": config.get("recipe_version", 1),
        "best_val_accuracy": config.get("best_val_accuracy"),
        "train_seconds": config.get("train_seconds"),
        # Runs predating runtime_info.py have no compute_platform block; they were all CPU,
        # but record them as "unknown" rather than assuming - the guard in main() then makes
        # the ambiguity visible instead of silently pooling them.
        "compute_platform": (config.get("compute_platform") or {}).get("device", "unknown"),
        "gpu": ", ".join((config.get("compute_platform") or {}).get("gpu_names", [])),
    }
    for k in SCALAR_METRICS + COST_METRICS:
        row[k] = metrics.get(k)
    row.update(parse_classification_report(run_dir / "classification_report.txt"))
    row.update(parse_lighting(run_dir / "lighting_stress_test_metrics.csv"))
    return row


def mean_std(values: list) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    # Sample std (ddof=1) - these runs are a sample of the training process, not a population.
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.mean(vals), std, len(vals)


def fmt(mean: float, std: float, n: int, scale: float = 100, digits: int = 2) -> str:
    if n == 0:
        return "n/a"
    if n == 1:
        return f"{mean * scale:.{digits}f} (n=1)"
    return f"{mean * scale:.{digits}f} +/- {std * scale:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", nargs="+", default=["*_fyp2_blockft*"],
                        help="one or more globs over results/ selecting the replicate runs to "
                             "pool. Several are accepted because architectures do not always "
                             "share a run-name namespace: the FastViT campaign is tagged by "
                             "fine-tuning depth (gpu_k2/gpu_k3), so building the 3-architecture "
                             "table needs e.g. \"*_blockft_gpu_seed*\" plus "
                             "\"fastvit_*_gpu_k2_seed*\" — which also keeps the k3 runs out.")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/multiseed"))
    parser.add_argument("--force", action="store_true",
                        help="overwrite an --out-prefix whose existing summary was "
                             "built from a DIFFERENT --pattern. Without this the "
                             "aggregate refuses, because that combination means one "
                             "campaign is about to be written over another's.")
    parser.add_argument("--allow-mixed-platform", action="store_true",
                        help="pool runs from different compute backends anyway (confounded - "
                             "only for deliberate CPU-vs-GPU comparisons)")
    parser.add_argument("--allow-mixed-recipe", action="store_true",
                        help="pool runs trained under different schedule.py recipe "
                             "versions anyway (confounded - v1 stopped at a 10+10 epoch "
                             "cap before converging, v2 trains to convergence, so their "
                             "accuracies are not samples of one process)")
    parser.add_argument("--allow-mixed-depth", action="store_true",
                        help="pool runs of one architecture that were fine-tuned to different "
                             "block depths anyway (confounded - the depth ablation is meant to "
                             "be reported as separate groups, not averaged together)")
    args = parser.parse_args()

    # This script already refuses to POOL runs across backends, recipes and depths.
    # Its OUTPUT path had no equivalent guard: --out-prefix defaults to
    # results/multiseed, so aggregating any campaign without passing it silently
    # overwrites whichever campaign wrote there last. That is the same class of
    # error the pooling guards exist to prevent, one step further down the pipeline.
    # Compare the pattern recorded in an existing summary against this run's.

    pattern_desc = ", ".join(args.pattern)
    existing_md = args.out_prefix.with_name(args.out_prefix.name + "_summary.md")
    if existing_md.exists() and not args.force:
        prior = existing_md.read_text(encoding="utf-8")
        marker = f"Pooled from `results/{{{pattern_desc}}}`"
        if "Pooled from `results/" in prior and marker not in prior:
            prior_line = next(l for l in prior.splitlines() if l.startswith("Pooled from"))
            raise SystemExit("\n".join([
                f"{existing_md} was built from a different pattern:",
                f"  existing: {prior_line}",
                f"  this run: Pooled from `results/{{{pattern_desc}}}`",
                "Writing here would overwrite another campaign's aggregate. Pass a distinct",
                "--out-prefix (e.g. results/<arch>_depth_k4_v2), or --force if replacing it",
                "is genuinely intended.",
            ]))
    # dict.fromkeys over the globs de-duplicates a directory matched by more than one pattern,
    # which would otherwise be counted twice in the mean.
    run_dirs = sorted(dict.fromkeys(
        d for p in args.pattern for d in Path("results").glob(p) if d.is_dir()
    ))
    runs = [r for r in (load_run(d) for d in run_dirs) if r]
    if not runs:
        raise SystemExit(f"No complete runs matched results/{{{pattern_desc}}}")

    skipped = len(run_dirs) - len(runs)
    note = f" ({skipped} incomplete, skipped)" if skipped else ""
    print(f"Pooling {len(runs)} run(s) from results/{{{pattern_desc}}}{note}")

    # Guard: replicates must share a compute backend. cuDNN and oneDNN pick different
    # convolution algorithms, so a CPU run and a GPU run are not samples of the same
    # process - averaging them would bury a platform change inside the "run-to-run
    # variance" this script exists to measure.
    platforms = sorted({r["compute_platform"] for r in runs})
    if len(platforms) > 1:
        breakdown = "\n".join(
            f"    {r['run_name']}: {r['compute_platform']}" + (f" ({r['gpu']})" if r["gpu"] else "")
            for r in runs
        )
        msg = (f"Runs span {len(platforms)} compute backends ({', '.join(platforms)}):\n{breakdown}\n"
               "Pooling these into one mean +/- std would confound the backend with the "
               "variance being measured. Narrow --pattern to one backend, or pass "
               "--allow-mixed-platform to override deliberately.")
        if not args.allow_mixed_platform:
            raise SystemExit("REFUSING: " + msg)
        print("WARNING (--allow-mixed-platform): " + msg)

    # Guard: replicates must share a training recipe. Recipe v1 capped every run at
    # 10+10 epochs and stopped it before convergence; v2 trains until EarlyStopping
    # actually fires. Averaging the two would fold a recipe upgrade into the run-to-run
    # variance this script exists to measure - the same failure mode as pooling backends.
    recipes = sorted({r["recipe_version"] for r in runs})
    if len(recipes) > 1:
        breakdown = "\n".join(
            f"    {r['run_name']}: recipe v{r['recipe_version']}" for r in runs
        )
        msg = (f"Runs span {len(recipes)} training recipes "
               f"({', '.join(f'v{v}' for v in recipes)}):\n{breakdown}\n"
               "Pooling these into one mean +/- std would confound the recipe change with "
               "the variance being measured. Narrow --pattern to one recipe, or pass "
               "--allow-mixed-recipe to override deliberately.")
        if not args.allow_mixed_recipe:
            raise SystemExit("REFUSING: " + msg)
        print("WARNING (--allow-mixed-recipe): " + msg)

    # Guard: replicates of one architecture must share a fine-tuning depth. Runs are grouped
    # by the `architecture` field, and FastViT's K=2 and K=3 campaigns both report
    # "FastViT_T8" - so a glob catching both would average two different recipes into one
    # "architecture" row and silently bury the depth ablation this campaign exists to measure.
    depths_by_arch = {}
    for r in runs:
        depths_by_arch.setdefault(r["architecture"], {}).setdefault(r["finetune_blocks"], []).append(
            r["run_name"]
        )
    mixed = {a: d for a, d in depths_by_arch.items() if len(d) > 1}
    if mixed:
        breakdown = "\n".join(
            f"    {a}: " + "; ".join(f"K={k} -> {', '.join(names)}" for k, names in sorted(
                d.items(), key=lambda kv: (kv[0] is None, kv[0])))
            for a, d in mixed.items()
        )
        msg = (f"{len(mixed)} architecture(s) span multiple fine-tuning depths:\n{breakdown}\n"
               "Averaging these into one mean +/- std would confound fine-tuning depth with the "
               "run-to-run variance being measured. Narrow --pattern to one depth per "
               "architecture, or pass --allow-mixed-depth to override deliberately.")
        if not args.allow_mixed_depth:
            raise SystemExit("REFUSING: " + msg)
        print("WARNING (--allow-mixed-depth): " + msg)

    # --- per-run CSV (long form) ---
    fieldnames = list(dict.fromkeys(k for r in runs for k in r))
    per_run = args.out_prefix.with_name(args.out_prefix.name + "_per_run.csv")
    per_run.parent.mkdir(parents=True, exist_ok=True)
    with per_run.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(runs)

    # --- group by architecture ---
    archs = list(dict.fromkeys(r["architecture"] for r in runs))
    groups = {a: [r for r in runs if r["architecture"] == a] for a in archs}

    report_rows = [
        ("Test accuracy (%)", "accuracy", 100, 2),
        ("ROC-AUC (OvR)", "roc_auc_ovr", 1, 4),
        ("Macro F1", "f1_macro", 1, 4),
    ]
    for c in CLASS_NAMES:
        report_rows.append((f"Recall - {c}", "recall_" + c, 1, 4))
    report_rows += [
        ("Lighting spread (pts)", "lighting_spread_pts", 1, 2),
        ("Params (M)", "n_params", 1e-6, 2),
        ("Model size (MB, pre-quant)", "model_size_mb_prequant", 1, 1),
        ("Inference (ms/img, dev)", "avg_inference_s_devcpu", 1000, 1),
    ]

    lines = ["# Multi-seed results - Phase 2 architecture comparison", ""]
    lines.append(f"Pooled from `results/{{{pattern_desc}}}`. Each cell is mean +/- sample std "
                 "over repeated runs of an identical recipe, varying only `--seed`.")
    lines.append("")
    seed_desc = []
    for a, g in groups.items():
        seeds = ", ".join(str(r["seed"]) for r in sorted(g, key=lambda x: x["seed"] or 0))
        seed_desc.append(f"**{a}** n={len(g)} (seeds {seeds})")
    lines.append("Seeds per architecture: " + "; ".join(seed_desc))
    lines.append("")
    gpus = sorted({r["gpu"] for r in runs if r["gpu"]})
    hw = ", ".join(platforms) + (f" ({'; '.join(gpus)})" if gpus else "")
    lines.append(f"Compute backend: {hw}.")
    rv = sorted({r["recipe_version"] for r in runs})
    lines.append("Training recipe: " + ", ".join(f"v{v}" for v in rv)
                 + " (see src/training/schedule.py).")
    lines.append("")
    lines.append("| Metric | " + " | ".join(archs) + " |")
    lines.append("|---|" + "---|" * len(archs))
    for label, key, scale, digits in report_rows:
        cells = []
        for a in archs:
            m, s, n = mean_std([r.get(key) for r in groups[a]])
            cells.append(fmt(m, s, n, scale, digits))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # --- the verdict this whole campaign exists to produce ---
    lines += ["", "## Is the architecture gap real?", ""]
    if len(archs) == 2:
        a, b = archs
        ma, sa, na = mean_std([r.get("accuracy") for r in groups[a]])
        mb, sb, nb = mean_std([r.get("accuracy") for r in groups[b]])
        delta = (ma - mb) * 100
        combined = ((sa ** 2 + sb ** 2) ** 0.5) * 100
        lines.append(f"- {a}: {ma * 100:.2f}% +/- {sa * 100:.2f} (n={na})")
        lines.append(f"- {b}: {mb * 100:.2f}% +/- {sb * 100:.2f} (n={nb})")
        lines.append(f"- Difference: **{delta:+.2f} pts**, combined spread +/-{combined:.2f} pts.")
        if na > 1 and nb > 1:
            verdict = ("larger than" if abs(delta) > combined else "within") + " the run-to-run noise"
            lines.append(f"- Verdict: the gap is **{verdict}**.")
            if abs(delta) <= combined:
                lines.append("  At this sample size the two architectures are **not distinguishable** "
                             "on accuracy; select on cost (params/size/latency) or run more seeds.")
            try:
                from scipy import stats
                t, p = stats.ttest_ind([r["accuracy"] for r in groups[a]],
                                       [r["accuracy"] for r in groups[b]], equal_var=False)
                lines.append(f"- Welch's t-test: t={t:.3f}, p={p:.3f} (n={na} vs {nb} - "
                             "indicative only, badly underpowered at this sample size).")
            except ImportError:
                pass
    else:
        lines.append(f"- {len(archs)} architecture(s) pooled - the pairwise verdict is emitted "
                     "only when exactly 2 are present.")

    summary_md = args.out_prefix.with_name(args.out_prefix.name + "_summary.md")
    summary_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {per_run} and {summary_md}")


if __name__ == "__main__":
    main()
