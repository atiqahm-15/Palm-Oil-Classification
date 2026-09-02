"""Generate all report-ready evaluation figures into results/figures/.

Compares the categorical MobileNetV2 baseline against the CORAL ordinal experiment
(train_mobilenetv2.py vs train_mobilenetv2_ordinal.py) wherever both runs exist, using
the validated palette from the dataviz skill: the sequential blue ramp (light->dark) for
the 4 ripeness classes, since ripeness is an ordinal progression, not an unordered
category; the fixed categorical blue/orange pair for baseline-vs-ordinal comparisons and
per-source breakdowns.

Every figure is saved once per run of this script (overwritten on rerun — these are
derived/regenerable, unlike results/<run-name>/, which is never overwritten).

Usage:
    python src/training/make_report_figures.py
        [--runs mobilenetv2_fyp2_baseline mobilenetv2_fyp2_ordinal]
        [--manifest data/processed/manifest_full.csv] [--out results/figures]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

import custom_layers  # noqa: F401 — registers kecam's ChannelAffine et al. for FastViT runs
import ordinal  # noqa: F401 — registers CoralLayer/CoralLogitsToProbs for model loading
from dataset import CLASS_NAMES, load_manifest_split
from evaluate import evaluate_split

# --- Palette (validated defaults, dataviz skill references/palette.md) ---------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
MODEL_COLOR = {"baseline": CATEGORICAL[0], "ordinal": CATEGORICAL[1]}
MODEL_LABEL = {"baseline": "Categorical (baseline)", "ordinal": "CORAL (ordinal)"}

# Ordinal ramp steps 250/400/550/700 (light->dark), unripe->overripe: less ripe = lighter.
CLASS_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CLASS_COLOR = dict(zip(CLASS_NAMES, CLASS_RAMP))

SEQUENTIAL_STEPS = [  # full 100->700 blue ramp, for heatmap gradation
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_sequential", SEQUENTIAL_STEPS)

SOURCE_ORDER = ["bunch_piles", "kaggle", "malaysia_outdoor", "mendeley_ordinal"]
SOURCE_LABEL = {
    "bunch_piles": "Bunch Piles", "kaggle": "Kaggle",
    "malaysia_outdoor": "Outdoor-Tenera", "mendeley_ordinal": "Mendeley Ordinal",
}
CONDITION_ORDER = ["clean_test_baseline", "darkened", "brightened", "shadow"]
CONDITION_LABEL = {
    "clean_test_baseline": "Clean (daylight)", "darkened": "Darkened (low-light)",
    "brightened": "Brightened", "shadow": "Shadow",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def savefig(fig, out_dir: Path, name: str):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = out_dir / name
    # bbox_inches="tight": tight_layout() only reserves space for axes content, so a
    # fig.legend() placed outside the axes (bbox_to_anchor below the last subplot) was
    # getting silently clipped out of the saved image. bbox_inches="tight" recomputes
    # the actual bounding box from every visible artist, legend included.
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --- Data loading ----------------------------------------------------------------------

def load_run(run_name: str, manifest_path: Path) -> dict | None:
    run_dir = Path("results") / run_name
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        print(f"  skipping {run_name}: {model_path} not found")
        return None

    with (run_dir / "config.json").open() as f:
        config = json.load(f)
    with (run_dir / "history.json").open() as f:
        history = json.load(f)

    kind = "ordinal" if "ordinal" in config["architecture"].lower() else "baseline"

    model = tf.keras.models.load_model(model_path)
    test_rows = [r for r in csv.DictReader(open(manifest_path, newline="")) if r["split"] == "test"]
    paths = [r["path"] for r in test_rows]
    labels = [CLASS_NAMES.index(r["class"]) for r in test_rows]
    accuracy, report, cm, roc_auc, preds, probs = evaluate_split(model, paths, labels)

    lst_path = run_dir / "lighting_stress_test_metrics.csv"
    lighting = {}
    if lst_path.exists():
        for row in csv.DictReader(open(lst_path, newline="")):
            lighting[row["condition"]] = float(row["accuracy"])

    return {
        "run_name": run_name, "kind": kind, "config": config, "history": history,
        "test_rows": test_rows, "labels": labels, "preds": preds, "probs": probs,
        "accuracy": accuracy, "roc_auc": roc_auc, "cm": cm, "lighting": lighting,
    }


# --- Figures -----------------------------------------------------------------------------

def fig_class_distribution(manifest_path: Path, out_dir: Path):
    rows = list(csv.DictReader(open(manifest_path, newline="")))
    splits = ["train", "valid", "test"]
    counts = {s: Counter(r["class"] for r in rows if r["split"] == s) for s in splits}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(splits))
    n_classes = len(CLASS_NAMES)
    width = 0.8 / n_classes
    for i, cls in enumerate(CLASS_NAMES):
        vals = [counts[s][cls] for s in splits]
        offset = (i - (n_classes - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=cls, color=CLASS_COLOR[cls],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.bar_label(bars, fontsize=7, color=INK_SECONDARY, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Images")
    ax.set_title("Class distribution by split", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, ncols=n_classes, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axes(ax)
    savefig(fig, out_dir, "01_class_distribution.png")


def fig_source_composition(manifest_path: Path, out_dir: Path):
    rows = list(csv.DictReader(open(manifest_path, newline="")))
    comp = {cls: Counter(r["source_dataset"] for r in rows if r["class"] == cls) for cls in CLASS_NAMES}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CLASS_NAMES))
    bottoms = np.zeros(len(CLASS_NAMES))
    for i, src in enumerate(SOURCE_ORDER):
        vals = np.array([comp[cls][src] for cls in CLASS_NAMES])
        ax.bar(x, vals, bottom=bottoms, label=SOURCE_LABEL[src], color=CATEGORICAL[i],
               edgecolor=SURFACE, linewidth=2, zorder=3, width=0.6)
        bottoms += vals
    for i, cls in enumerate(CLASS_NAMES):
        ax.text(i, bottoms[i] + max(bottoms) * 0.015, f"{int(bottoms[i])}", ha="center",
                 fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Images (all splits)")
    ax.set_title("Per-class source composition", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axes(ax)
    savefig(fig, out_dir, "02_source_composition.png")


def fig_confusion_matrices(runs: list[dict], out_dir: Path):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8))
    if n == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        cm = run["cm"]
        im = ax.imshow(cm, cmap=BLUE_CMAP)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        vmax = cm.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "#ffffff" if cm[i, j] > vmax * 0.6 else INK_PRIMARY
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODEL_LABEL[run['kind']]}\naccuracy={run['accuracy']:.3f}",
                      color=INK_PRIMARY, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Confusion matrix — clean test set", color=INK_PRIMARY, fontsize=13, x=0.02, ha="left")
    savefig(fig, out_dir, "03_confusion_matrices.png")


def fig_per_class_metrics(runs: list[dict], out_dir: Path):
    from sklearn.metrics import precision_recall_fscore_support

    metrics_by_run = {}
    for run in runs:
        p, r, f1, _ = precision_recall_fscore_support(
            run["labels"], run["preds"], labels=range(len(CLASS_NAMES)), zero_division=0
        )
        metrics_by_run[run["kind"]] = {"precision": p, "recall": r, "f1": f1}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for i, run in enumerate(runs):
            vals = metrics_by_run[run["kind"]][metric]
            offset = (i - (len(runs) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=MODEL_LABEL[run["kind"]],
                           color=MODEL_COLOR[run["kind"]], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_SECONDARY, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.capitalize(), color=INK_PRIMARY, fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=len(runs), loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Per-class precision / recall / F1", color=INK_PRIMARY, fontsize=13, x=0.02, ha="left")
    savefig(fig, out_dir, "04_per_class_metrics.png")


def fig_lighting_stress_test(runs: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.35
    for i, run in enumerate(runs):
        vals = [run["lighting"].get(c, np.nan) for c in CONDITION_ORDER]
        offset = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_LABEL[run["kind"]],
                       color=MODEL_COLOR[run["kind"]], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, color=INK_SECONDARY, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, max(0.05 + max(v for r in runs for v in r["lighting"].values() if v == v), 0.8))
    ax.set_title("Lighting Stress-Test accuracy", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, ncols=len(runs), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    style_axes(ax)
    savefig(fig, out_dir, "05_lighting_stress_test.png")


def fig_overall_summary(runs: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    metrics = ["accuracy", "roc_auc"]
    metric_labels = ["Test accuracy", "ROC-AUC (OvR)"]
    x = np.arange(len(metrics))
    width = 0.35
    for i, run in enumerate(runs):
        vals = [run["accuracy"], run["roc_auc"]]
        offset = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_LABEL[run["kind"]],
                       color=MODEL_COLOR[run["kind"]], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, color=INK_SECONDARY, padding=3)
    ax.axhline(0.85, color=CATEGORICAL[3], linestyle="--", linewidth=1.5, zorder=2)
    ax.text(len(metrics) - 0.5, 0.86, "85% NFR target", fontsize=8, color=INK_SECONDARY, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall test-set performance", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, ncols=1, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axes(ax)
    savefig(fig, out_dir, "06_overall_summary.png")


def fig_training_curves(runs: list[dict], out_dir: Path):
    fig, axes = plt.subplots(2, len(runs), figsize=(6 * len(runs), 7), squeeze=False)
    for col, run in enumerate(runs):
        history = run["history"]
        acc_key = "ordinal_accuracy" if run["kind"] == "ordinal" else "accuracy"
        epochs = np.arange(1, len(history["loss"]) + 1)
        stage_split = run["config"]["final_epochs_run_frozen"]

        for row, (keys, ylabel) in enumerate([
            ((acc_key, f"val_{acc_key}"), "Accuracy"),
            (("loss", "val_loss"), "Loss"),
        ]):
            ax = axes[row][col]
            ax.plot(epochs, history[keys[0]], color=CATEGORICAL[0], linewidth=2, label="Train", zorder=3)
            ax.plot(epochs, history[keys[1]], color=CATEGORICAL[1], linewidth=2, label="Validation", zorder=3)
            ax.axvline(stage_split + 0.5, color=AXIS, linestyle="--", linewidth=1, zorder=2)
            if row == 0:
                ax.set_title(MODEL_LABEL[run["kind"]], color=INK_PRIMARY, fontsize=12)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            style_axes(ax)
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Training curves (dashed line = frozen -> fine-tune transition)",
                  color=INK_PRIMARY, fontsize=13, x=0.02, ha="left")
    savefig(fig, out_dir, "07_training_curves.png")


def fig_underripe_ripe_confusion_by_source(runs: list[dict], out_dir: Path):
    underripe_idx, ripe_idx = CLASS_NAMES.index("underripe"), CLASS_NAMES.index("ripe")
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        source_stats = defaultdict(lambda: Counter())
        for row, true_idx, pred_idx in zip(run["test_rows"], run["labels"], run["preds"]):
            src = row["source_dataset"]
            if true_idx == underripe_idx:
                source_stats[src]["underripe_total"] += 1
                if pred_idx == ripe_idx:
                    source_stats[src]["underripe_as_ripe"] += 1
            if true_idx == ripe_idx:
                source_stats[src]["ripe_total"] += 1
                if pred_idx == underripe_idx:
                    source_stats[src]["ripe_as_underripe"] += 1

        sources = [s for s in SOURCE_ORDER if source_stats[s]["underripe_total"] or source_stats[s]["ripe_total"]]
        x = np.arange(len(sources))
        width = 0.35
        ur_rate = [source_stats[s]["underripe_as_ripe"] / max(source_stats[s]["underripe_total"], 1) for s in sources]
        r_rate = [source_stats[s]["ripe_as_underripe"] / max(source_stats[s]["ripe_total"], 1) for s in sources]
        b1 = ax.bar(x - width / 2, ur_rate, width, label="Underripe -> Ripe", color=CATEGORICAL[0],
                     edgecolor=SURFACE, linewidth=1.5, zorder=3)
        b2 = ax.bar(x + width / 2, r_rate, width, label="Ripe -> Underripe", color=CATEGORICAL[1],
                     edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.bar_label(b1, labels=[f"{v*100:.0f}%" for v in ur_rate], fontsize=7, color=INK_SECONDARY, padding=2)
        ax.bar_label(b2, labels=[f"{v*100:.0f}%" for v in r_rate], fontsize=7, color=INK_SECONDARY, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([SOURCE_LABEL[s] for s in sources], rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(MODEL_LABEL[run["kind"]], color=INK_PRIMARY, fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("Confusion rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Underripe <-> Ripe confusion by source dataset", color=INK_PRIMARY, fontsize=13, x=0.02, ha="left")
    savefig(fig, out_dir, "08_underripe_ripe_confusion_by_source.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["mobilenetv2_fyp2_baseline", "mobilenetv2_fyp2_ordinal"])
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print("Dataset-level figures (don't need a trained model):")
    fig_class_distribution(args.manifest, args.out)
    fig_source_composition(args.manifest, args.out)

    print("\nLoading runs (re-running inference on the test set for each)...")
    runs = [r for r in (load_run(name, args.manifest) for name in args.runs) if r is not None]
    if not runs:
        print("No trained runs found — skipping model-comparison figures.")
        return

    print(f"\nModel-comparison figures ({len(runs)} run(s): {[r['run_name'] for r in runs]}):")
    fig_confusion_matrices(runs, args.out)
    fig_per_class_metrics(runs, args.out)
    fig_overall_summary(runs, args.out)
    if all(r["lighting"] for r in runs):
        fig_lighting_stress_test(runs, args.out)
    else:
        print("  skipping lighting stress-test figure: missing lighting_stress_test_metrics.csv for a run")
    fig_training_curves(runs, args.out)
    fig_underripe_ripe_confusion_by_source(runs, args.out)

    print(f"\nAll figures written to {args.out}")


if __name__ == "__main__":
    main()
