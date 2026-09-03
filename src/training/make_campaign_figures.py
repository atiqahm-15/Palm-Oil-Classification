"""Generate the campaign-level report figures (09-11) into results/figures/.

These cover the three Aug-2026 campaigns that `make_report_figures.py` does not:
the multi-seed architecture comparison, per-class threshold calibration, and the
leave-one-dataset-out generalization check. Unlike that script, this one reads only
the aggregated CSV/JSON artifacts already on disk -- no model loading, no TensorFlow --
so it runs in seconds and cannot perturb any run directory.

Palette constants mirror make_report_figures.py (validated dataviz defaults). They are
duplicated rather than imported because importing that module pulls in TensorFlow and
the dataset/evaluate stack, which this script does not need.

All text is ASCII on purpose: results/lodo_summary.md already carries mojibake from an
earlier encoding slip, so non-ASCII glyphs are avoided end to end.

Usage:
    python src/training/make_campaign_figures.py [--out results/figures]
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Palette (mirrors make_report_figures.py) -----------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASS_NAMES = ["unripe", "underripe", "ripe", "overripe"]

SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_sequential", SEQUENTIAL_STEPS)

# Preferred display order; only architectures actually present in the data are drawn, so the
# same script works before and after the FastViT campaign lands, and works for the calibration
# figure (which covers only the two CNNs) without a separate code path.
ARCH_ORDER = ["EfficientNetB0", "MobileNetV2", "FastViT_T8"]
ARCH_COLOR = {
    "EfficientNetB0": CATEGORICAL[0],
    "MobileNetV2": CATEGORICAL[1],
    "FastViT_T8": CATEGORICAL[2],
}


def present_archs(by_arch: dict) -> list[str]:
    """Architectures present in `by_arch`, in ARCH_ORDER; unknown ones appended alphabetically."""
    known = [a for a in ARCH_ORDER if by_arch.get(a)]
    extra = sorted(a for a in by_arch if a not in ARCH_ORDER and by_arch.get(a))
    return known + extra

FOLD_ORDER = ["bunch_piles", "kaggle", "malaysia_outdoor", "mendeley_ordinal"]
FOLD_LABEL = {
    "bunch_piles": "Bunch Piles", "kaggle": "Kaggle",
    "malaysia_outdoor": "Outdoor-Tenera", "mendeley_ordinal": "Mendeley Ordinal",
}

NFR_ACCURACY = 85.0  # CLAUDE.md acceptance bar


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
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    # ddof=1 -> sample std, matching aggregate_seeds.py and the summary tables.
    return arr.mean(), (arr.std(ddof=1) if arr.size > 1 else 0.0)


# --- Data loading ----------------------------------------------------------------------

def load_multiseed(path: Path) -> dict:
    by_arch = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_arch[row["architecture"]].append(row)
    return by_arch


def load_calibration(cal_dir: Path) -> dict:
    by_arch = defaultdict(list)
    for jf in sorted(cal_dir.glob("*.json")):
        with jf.open() as f:
            d = json.load(f)
        by_arch[d["architecture"]].append(d)
    return by_arch


# Fold names carry a recipe-version prefix from v2 onward ("v2_bunch_piles"), while the v1
# campaign wrote them bare ("bunch_piles"). FOLD_ORDER/FOLD_LABEL are keyed on the bare source
# name so one set of labels serves every campaign; stripping the prefix here is what lets that
# hold. Without it, by_fold is a defaultdict, so an unmatched key yields an empty list rather
# than a KeyError -- figure 11 would render with blank bars instead of failing.
FOLD_PREFIX = re.compile(r"^v\d+_")


def load_lodo(path: Path) -> dict:
    by_fold = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_fold[FOLD_PREFIX.sub("", row["fold"])].append(row)
    missing = [f for f in FOLD_ORDER if not by_fold.get(f)]
    if missing:
        raise SystemExit(f"{path}: no rows for fold(s) {missing}; expected {FOLD_ORDER}.")
    return by_fold


# --- Figure 09: multi-seed architecture comparison ---------------------------------------

def fig_multiseed(by_arch: dict, out_dir: Path):
    archs = present_archs(by_arch)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2))

    # (a) Bar = mean, whisker = sample std, dots = the individual seed runs.
    label_tops, n_seeds = [], set()
    for i, arch in enumerate(archs):
        accs = [float(r["accuracy"]) * 100 for r in by_arch[arch]]
        n_seeds.add(len(accs))
        m, s = mean_std(accs)
        ax1.bar(i, m, 0.55, color=ARCH_COLOR[arch], zorder=2)
        ax1.errorbar(i, m, yerr=s, fmt="none", ecolor=INK_PRIMARY, capsize=6, lw=1.6, zorder=4)
        ax1.scatter([i] * len(accs), accs, color=SURFACE, edgecolor=INK_PRIMARY,
                    s=34, zorder=5, linewidth=1.2)
        label_tops.append(m + s + 2.5)
        ax1.text(i, m + s + 2.5, f"{m:.2f} +/- {s:.2f}", ha="center",
                 color=INK_PRIMARY, fontsize=10, fontweight="bold")

    ax1.axhline(NFR_ACCURACY, color=CATEGORICAL[3], ls="--", lw=1.5, zorder=3)
    ax1.text(len(archs) - 0.54, NFR_ACCURACY + 0.8, "NFR >= 85%", ha="right",
             color=CATEGORICAL[3], fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels(archs, color=INK_SECONDARY, fontsize=9.5)
    # Ceiling follows the tallest value label rather than a fixed 95. Hardcoding it was safe
    # while the best run was 78.72% (recipe v1) and clipped the 94.55 +/- 0.31 label into the
    # title once v2 landed. Bars stay zero-based -- this is a bar chart of a percentage.
    ax1.set_ylim(0, max(max(label_tops) + 3.0, NFR_ACCURACY + 8.0))
    ax1.set_ylabel("Test accuracy (%)")
    seeds = f"{n_seeds.pop()} seeds" if len(n_seeds) == 1 else "varying seed counts"
    ax1.set_title(f"(a) Test accuracy, {seeds} per architecture",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    style_axes(ax1)

    # (b) Underripe carries by far the largest spread, for every architecture.
    # Bar width adapts to however many architectures are present, keeping each cluster 0.8 wide.
    width = 0.8 / len(archs)
    x = np.arange(len(CLASS_NAMES))
    for i, arch in enumerate(archs):
        offset = (i - (len(archs) - 1) / 2) * width
        means, stds = [], []
        for cls in CLASS_NAMES:
            m, s = mean_std([float(r[f"recall_{cls}"]) for r in by_arch[arch]])
            means.append(m)
            stds.append(s)
        ax2.bar(x + offset, means, width, yerr=stds, capsize=3,
                color=ARCH_COLOR.get(arch, INK_MUTED), label=arch, zorder=2,
                error_kw={"ecolor": INK_PRIMARY, "lw": 1.2})
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in CLASS_NAMES], color=INK_SECONDARY)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Recall (mean +/- std over seeds)")
    ax2.set_title("(b) Per-class recall - underripe is weak for all",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    ax2.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9.5)
    style_axes(ax2)

    # (c) Does the winner earn its cost? Accuracy against pre-quantization size.
    # Points are annotated in size order with the label alternating below/above, because
    # neighbouring architectures (FastViT 27.0MB, EfficientNetB0 33.9MB) sit close enough on
    # the x axis that same-side labels overlap.
    by_size = sorted(archs, key=lambda a: float(by_arch[a][0]["model_size_mb_prequant"]))
    sizes, accs_all = [], []
    for rank, arch in enumerate(by_size):
        accs = [float(r["accuracy"]) * 100 for r in by_arch[arch]]
        m, s = mean_std(accs)
        size = float(by_arch[arch][0]["model_size_mb_prequant"])
        params = float(by_arch[arch][0]["n_params"]) / 1e6
        sizes.append(size)
        accs_all.append(m)
        ax3.errorbar(size, m, yerr=s, fmt="o", ms=11, color=ARCH_COLOR.get(arch, INK_MUTED),
                     ecolor=INK_PRIMARY, capsize=5, lw=1.6, zorder=4)
        below = rank % 2 == 0
        ax3.annotate(f"{arch}\n{params:.2f}M params",
                     (size, m), textcoords="offset points",
                     xytext=(0, -40 if below else 26),
                     ha="center", va="top" if below else "bottom",
                     color=INK_SECONDARY, fontsize=9)
    # Pad the limits around the data rather than hardcoding, so a third architecture landing
    # outside the original 10-45MB / 70-80% window is not silently clipped off the axes.
    ax3.set_xlim(min(sizes) - 12, max(sizes) + 12)
    ax3.set_ylim(min(accs_all) - 5.5, max(accs_all) + 4.0)
    ax3.set_xlabel("Model size (MB, pre-quantization)")
    ax3.set_ylabel("Test accuracy (%)")
    ax3.set_title("(c) Accuracy vs cost - all far under the 100MB NFR",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    style_axes(ax3)

    savefig(fig, out_dir, "09_multiseed_architecture_comparison.png")


# --- Figure 10: per-class threshold calibration ------------------------------------------

def fig_calibration(by_arch: dict, out_dir: Path):
    # Calibration was only run for the two CNNs, so this resolves to those; it will pick up a
    # third architecture automatically if calibrate_thresholds.py is ever run for one.
    archs = present_archs(by_arch)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2))
    width = 0.36
    x = np.arange(len(archs))

    # (a) The gain is real for EfficientNetB0 and absent for MobileNetV2.
    for i, (key, label, color) in enumerate([
        ("test_baseline", "Baseline (argmax)", CATEGORICAL[0]),
        ("test_calibrated", "Calibrated", CATEGORICAL[2]),
    ]):
        means, stds = [], []
        for arch in archs:
            m, s = mean_std([d[key]["accuracy"] * 100 for d in by_arch[arch]])
            means.append(m)
            stds.append(s)
        ax1.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=label,
                color=color, zorder=2, error_kw={"ecolor": INK_PRIMARY, "lw": 1.4})

    cal_label_tops = []
    for i, arch in enumerate(archs):
        base, _ = mean_std([d["test_baseline"]["accuracy"] * 100 for d in by_arch[arch]])
        cal, _ = mean_std([d["test_calibrated"]["accuracy"] * 100 for d in by_arch[arch]])
        delta = cal - base
        cal_label_tops.append(max(base, cal) + 2.6)
        ax1.text(i, max(base, cal) + 2.6, f"{delta:+.2f} pts", ha="center",
                 fontsize=10.5, fontweight="bold",
                 color=CATEGORICAL[2] if delta > 0.1 else INK_MUTED)

    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, color=INK_SECONDARY)
    # Ceiling follows the tallest value label rather than a fixed 92, which was sized
    # for recipe v1 (best calibrated run 79.17%) and clipped every v2 bar and its label.
    ax1.set_ylim(0, max(cal_label_tops) + 9.0)
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("(a) Calibration gain on test",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9.5, loc="upper left")
    style_axes(ax1)

    # (b) Where the gain comes from, for whichever architecture ends up best after calibration --
    # that is the model actually deployed, so its trade is the one the reader needs. Chosen from
    # the data rather than hardcoded, so this keeps tracking the winner as runs are added.
    focus = max(archs, key=lambda a: mean_std([d["test_calibrated"]["accuracy"] for d in by_arch[a]])[0])
    xc = np.arange(len(CLASS_NAMES))
    for i, (key, label, color) in enumerate([
        ("test_baseline", "Baseline", CATEGORICAL[0]),
        ("test_calibrated", "Calibrated", CATEGORICAL[2]),
    ]):
        means, stds = [], []
        for cls in CLASS_NAMES:
            m, s = mean_std([d[key][f"recall_{cls}"] for d in by_arch[focus]])
            means.append(m)
            stds.append(s)
        ax2.bar(xc + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=label,
                color=color, zorder=2, error_kw={"ecolor": INK_PRIMARY, "lw": 1.4})

    for j, cls in enumerate(CLASS_NAMES):
        b, _ = mean_std([d["test_baseline"][f"recall_{cls}"] for d in by_arch[focus]])
        c, _ = mean_std([d["test_calibrated"][f"recall_{cls}"] for d in by_arch[focus]])
        delta = c - b
        ax2.text(j, max(b, c) + 0.08, f"{delta:+.3f}", ha="center", fontsize=9.5,
                 fontweight="bold",
                 color=CATEGORICAL[2] if delta > 0 else CATEGORICAL[1])

    ax2.set_xticks(xc)
    ax2.set_xticklabels([c.capitalize() for c in CLASS_NAMES], color=INK_SECONDARY)
    ax2.set_ylim(0, 1.34)
    ax2.set_ylabel("Recall (mean +/- std over seeds)")
    ax2.set_title(f"(b) {focus} per-class recall - the trade",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    ax2.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9.5, loc="upper right")
    style_axes(ax2)

    # (c) Weights were fitted on validation only; the test bar is the honest one.
    all_gains = []
    for i, (split, label, color) in enumerate([
        ("valid", "Validation (fitted on)", CATEGORICAL[3]),
        ("test", "Test (held out)", CATEGORICAL[2]),
    ]):
        gains = []
        for arch in archs:
            b, _ = mean_std([d[f"{split}_baseline"]["accuracy"] * 100 for d in by_arch[arch]])
            c, _ = mean_std([d[f"{split}_calibrated"]["accuracy"] * 100 for d in by_arch[arch]])
            gains.append(c - b)
        all_gains.extend(gains)
        ax3.bar(x + (i - 0.5) * width, gains, width, label=label, color=color, zorder=2)
        for j, g in enumerate(gains):
            ax3.text(j + (i - 0.5) * width, g + (0.07 if g >= 0 else -0.20), f"{g:+.2f}",
                     ha="center", va="bottom" if g >= 0 else "top",
                     fontsize=9.5, color=INK_SECONDARY)

    ax3.axhline(0, color=AXIS, lw=1.2, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(archs, color=INK_SECONDARY)
    # Explicit headroom below zero, so the negative MobileNetV2 label stays inside the
    # axes instead of colliding with the x tick labels drawn under the spine.
    # Pad around the actual gains: a fixed ceiling clipped FastViT's +2.09 validation bar.
    lo, hi = min(all_gains), max(all_gains)
    ax3.set_ylim(min(lo - 0.35, -0.5), hi + 0.45)
    ax3.set_ylabel("Accuracy gain from calibration (pts)")
    ax3.set_title("(c) Fitted on validation, held up on test",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    ax3.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9.5, loc="lower left")
    style_axes(ax3)

    savefig(fig, out_dir, "10_calibration_effect.png")


# --- Figure 11: leave-one-dataset-out generalization -------------------------------------

def fig_lodo(by_fold: dict, main_split_acc: float, arch: str, n_seeds: int, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) Every fold sits far below the main split - that distance is the headline.
    fold_means = []
    for i, fold in enumerate(FOLD_ORDER):
        accs = [float(r["accuracy"]) * 100 for r in by_fold[fold]]
        m, s = mean_std(accs)
        fold_means.append(m)
        ax1.bar(i, m, 0.55, color=CATEGORICAL[0], zorder=2)
        ax1.errorbar(i, m, yerr=s, fmt="none", ecolor=INK_PRIMARY, capsize=6, lw=1.6, zorder=4)
        ax1.scatter([i] * len(accs), accs, color=SURFACE, edgecolor=INK_PRIMARY,
                    s=34, zorder=5, linewidth=1.2)
        ax1.text(i, m + s + 1.6, f"{m:.2f}\n+/- {s:.2f}", ha="center",
                 color=INK_PRIMARY, fontsize=9, fontweight="bold")

    lodo_mean = float(np.mean(fold_means))
    # The two reference levels go in a legend rather than as in-plot text: every strip of
    # space beside these lines is already taken by a bar or its value label.
    ax1.axhline(main_split_acc, color=CATEGORICAL[2], ls="-", lw=1.8, zorder=3,
                label=f"Main split ({arch}, n={n_seeds}): {main_split_acc:.2f}%")
    ax1.axhline(lodo_mean, color=CATEGORICAL[1], ls="--", lw=1.6, zorder=3,
                label=f"LODO mean across 4 folds: {lodo_mean:.2f}%")

    # Annotate the gap itself, between the two reference lines. Shown to 1 dp: computing
    # it at full precision gives 20.50, while lodo_summary.md/PROJECT_PLAN.md quote 20.49
    # from differencing already-rounded means. One decimal is true under both.
    ax1.annotate(
        "", xy=(3.42, main_split_acc), xytext=(3.42, lodo_mean),
        arrowprops={"arrowstyle": "<->", "color": INK_PRIMARY, "lw": 1.6},
    )
    ax1.text(3.32, (main_split_acc + lodo_mean) / 2,
             f"gap\n{main_split_acc - lodo_mean:.1f} pts", ha="right", va="center",
             fontsize=10, fontweight="bold", color=INK_PRIMARY)

    ax1.set_xticks(range(len(FOLD_ORDER)))
    ax1.set_xticklabels([FOLD_LABEL[f] for f in FOLD_ORDER], color=INK_SECONDARY, fontsize=9.5)
    ax1.set_xlim(-0.7, 3.7)
    # Headroom above the main-split line so the legend has clear space of its own.
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Test accuracy on held-out source (%)")
    ax1.set_title("(a) Held-out source accuracy vs the main split",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    ax1.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9.5, loc="upper left")
    style_axes(ax1)

    # (b) Per-class recall by fold. A source with no images of a class is left blank --
    # support 0 means undefined, not zero, which is the bug aggregate_lodo.py fixed.
    grid = np.full((len(FOLD_ORDER), len(CLASS_NAMES)), np.nan)
    for i, fold in enumerate(FOLD_ORDER):
        for j, cls in enumerate(CLASS_NAMES):
            supports = [float(r[f"support_{cls}"]) for r in by_fold[fold]]
            if max(supports) == 0:
                continue
            grid[i, j] = np.mean([float(r[f"recall_{cls}"]) for r in by_fold[fold]])

    masked = np.ma.masked_invalid(grid)
    cmap = BLUE_CMAP.copy()
    cmap.set_bad(GRIDLINE)
    im = ax2.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(FOLD_ORDER)):
        for j in range(len(CLASS_NAMES)):
            if np.isnan(grid[i, j]):
                ax2.text(j, i, "n/a", ha="center", va="center",
                         color=INK_MUTED, fontsize=9, style="italic")
            else:
                val = grid[i, j]
                ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                         color=SURFACE if val > 0.55 else INK_PRIMARY,
                         fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(CLASS_NAMES)))
    ax2.set_xticklabels([c.capitalize() for c in CLASS_NAMES], color=INK_SECONDARY)
    ax2.set_yticks(range(len(FOLD_ORDER)))
    ax2.set_yticklabels([FOLD_LABEL[f] for f in FOLD_ORDER], color=INK_SECONDARY, fontsize=9.5)
    ax2.set_title("(b) Per-class recall by held-out source",
                  color=INK_PRIMARY, fontsize=11.5, fontweight="bold", loc="left")
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors=INK_MUTED, length=0)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cbar.set_label("Recall", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)
    cbar.outline.set_visible(False)

    savefig(fig, out_dir, "11_lodo_generalization_gap.png")


def lodo_architecture(by_fold: dict) -> str:
    """The architecture the LODO campaign was actually run on, read from its own rows.

    This used to be hardcoded to EfficientNetB0 -- correct for the v1 campaign, where
    EfficientNetB0 was the winner LODO was run on, and silently wrong for v2, where the
    winner is FastViT_T8. Hardcoding it meant figure 11 could draw its main-split
    reference line from one architecture and its folds from another, understating or
    overstating the generalization gap with nothing on the figure to show it. Deriving it
    from the LODO rows makes that class of mismatch impossible.
    """
    archs = {row["architecture"] for rows in by_fold.values() for row in rows}
    if len(archs) != 1:
        raise SystemExit(f"LODO CSV mixes architectures {sorted(archs)}; expected exactly one.")
    return archs.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults point at the v2 campaign, which is the live one. They pointed at the v1 CSVs
    # until 2026-09-03, which is how results/figures/ came to hold v1 figures for a report
    # headlining v2 numbers. v1 remains reproducible by passing the v1 paths explicitly.
    parser.add_argument("--multiseed", type=Path, default=Path("results/multiseed_v2_per_run.csv"))
    parser.add_argument("--lodo", type=Path, default=Path("results/lodo_v2_per_run.csv"))
    # No default: the only calibration artifacts on disk were fitted on v1 checkpoints and
    # are void under v2 (docs/report_limitations.md L6). Figure 10 is therefore opt-in --
    # passing --calibration is an assertion that the directory matches the campaign.
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Multi-seed architecture comparison:")
    by_arch = load_multiseed(args.multiseed)
    fig_multiseed(by_arch, args.out)

    if args.calibration is None:
        print("Threshold calibration: SKIPPED (no --calibration given; figure 10 left as-is).")
    else:
        print("Threshold calibration:")
        by_arch_cal = load_calibration(args.calibration)
        fig_calibration(by_arch_cal, args.out)

    print("LODO generalization:")
    by_fold = load_lodo(args.lodo)
    # Reference line comes from the same multi-seed campaign the LODO winner was picked
    # from, rather than a hardcoded number, so the two figures can never disagree.
    arch = lodo_architecture(by_fold)
    if arch not in by_arch:
        raise SystemExit(
            f"LODO ran on {arch}, absent from {args.multiseed}. The main-split reference "
            f"line must come from the same architecture and campaign as the folds."
        )
    ref_rows = by_arch[arch]
    main_acc, _ = mean_std([float(r["accuracy"]) * 100 for r in ref_rows])
    print(f"  reference: {arch} main split {main_acc:.2f}% "
          f"(n={len(ref_rows)}, from {args.multiseed.name})")
    fig_lodo(by_fold, main_acc, arch, len(ref_rows), args.out)


if __name__ == "__main__":
    main()
