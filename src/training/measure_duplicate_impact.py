"""Quantify how much the Bunch Piles duplicate-row defect moves the reported test metrics.

WHAT THE DEFECT IS. `data/processed/manifest_full.csv` contains rows that are exact
duplicates of another row -- identical path, class, source photo, and split -- introduced by
the Bunch Piles crop pipeline (`src/data/crop_bunch_piles.py`). They never cross a split
boundary, so they are NOT leakage: no test image was trained on because of them. What they do
is double-count a subset of test images, so every reported test metric is a weighted average
in which the duplicated images count twice.

WHY MEASURE RATHER THAN JUST FIX. PROJECT_PLAN.md §8 records the decision to treat this as a
measure-and-document item rather than a blocker. That decision is only defensible with a
number attached: "~1% of the test set is double-counted" bounds the exposure but does not say
which way, or how far, the headline moves. This script produces that number, so the report can
state the effect instead of the mechanism.

METHOD. Predictions depend only on the image, and duplicate rows point at the same image, so
inference runs once per unique test path. Each metric is then computed twice over those same
predictions:

  as-scored     -- rows weighted by their multiplicity in the manifest. Reproduces exactly
                   what evaluate.py measured and what results/multiseed_v2_summary.md quotes.
  deduplicated  -- one row per unique image. What the metric would have been without the defect.

The difference between the two is the defect's contribution. Nothing is re-trained: the models
were trained on a training split that also contains duplicates (1,422 of them, double-weighting
some training images), and this script does not attempt to estimate that effect -- it measures
the reporting error, not the training error. That distinction is stated in the output.

Usage (WSL GPU venv only -- .keras checkpoints do not load under the Windows Keras version):
    python src/training/measure_duplicate_impact.py --runs <run-name> [<run-name> ...]
        [--manifest data/processed/manifest_full.csv]
        [--out-prefix results/duplicate_row_impact]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score

import custom_layers  # noqa: F401 — registers kecam's ChannelAffine et al. for FastViT runs
import ordinal  # noqa: F401 — registers CoralLayer/CoralLogitsToProbs for ordinal checkpoints
from dataset import CLASS_NAMES
from evaluate import evaluate_split

# The columns that define row identity. A row duplicating another across all four is the
# defect; two different crops of the same source photo are legitimate and differ in `path`.
IDENTITY = ("path", "class", "source_photo", "split")


def load_test_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def multiplicities(rows: list[dict]) -> Counter:
    return Counter(tuple(r[c] for c in IDENTITY) for r in rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict:
    """Accuracy / macro-F1 / per-class recall over rows carrying integer weights.

    sklearn's sample_weight is what makes the two passes strictly comparable: the as-scored
    pass is the deduplicated pass with the duplicated images weighted 2 (or more), rather than
    a separately-computed quantity that might differ for an unrelated reason.
    """
    labels = list(range(len(CLASS_NAMES)))
    out = {
        "accuracy": float(np.average(y_true == y_pred, weights=weights)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0, sample_weight=weights)),
    }
    per_class = recall_score(y_true, y_pred, labels=labels, average=None,
                             zero_division=0, sample_weight=weights)
    for name, value in zip(CLASS_NAMES, per_class):
        out[f"recall_{name}"] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--out-prefix", type=Path, default=Path("results/duplicate_row_impact"))
    args = parser.parse_args()

    rows = load_test_rows(args.manifest)
    counts = multiplicities(rows)
    unique_keys = list(counts)
    weights = np.array([counts[k] for k in unique_keys], dtype=float)
    ones = np.ones_like(weights)

    key_index = {c: i for i, c in enumerate(IDENTITY)}
    paths = [k[key_index["path"]] for k in unique_keys]
    y_true = np.array([CLASS_NAMES.index(k[key_index["class"]]) for k in unique_keys])

    n_rows, n_unique = len(rows), len(unique_keys)
    n_dup_rows = n_rows - n_unique
    n_images_affected = sum(1 for k in unique_keys if counts[k] > 1)
    print(f"Test split: {n_rows} manifest rows, {n_unique} unique images.")
    print(f"  {n_dup_rows} duplicate rows ({n_dup_rows / n_rows:.2%} of rows), "
          f"affecting {n_images_affected} distinct images "
          f"(max multiplicity {int(max(counts.values()))}).")
    if n_dup_rows == 0:
        print("No duplicate rows -- the defect is fixed; nothing to measure.")
        return

    # The duplicated images are not a random sample: they all come from one source's crop
    # pipeline. Reporting their class mix shows which class the double-counting favours.
    dup_classes = Counter(k[key_index["class"]] for k in unique_keys if counts[k] > 1)
    print("  duplicated images by class: "
          + ", ".join(f"{c}={dup_classes.get(c, 0)}" for c in CLASS_NAMES))

    records = []
    for run_name in args.runs:
        model_path = Path("results") / run_name / "model.keras"
        if not model_path.exists():
            print(f"  skipping {run_name}: {model_path} not found")
            continue
        print(f"\n{run_name}: predicting {n_unique} unique test images...")
        model = tf.keras.models.load_model(model_path)
        _, _, _, _, preds, _ = evaluate_split(model, paths, list(y_true))

        as_scored = metrics(y_true, preds, weights)
        dedup = metrics(y_true, preds, ones)
        # Accuracy on the duplicated images alone: the mechanism behind whichever way the
        # headline moves. Above the overall figure, double-counting inflates it; below, deflates.
        dup_mask = weights > 1
        dup_acc = float(np.mean(y_true[dup_mask] == preds[dup_mask]))

        rec = {"run_name": run_name, "accuracy_on_duplicated_images": dup_acc}
        for key in as_scored:
            rec[f"as_scored_{key}"] = as_scored[key]
            rec[f"dedup_{key}"] = dedup[key]
            rec[f"delta_{key}"] = dedup[key] - as_scored[key]
        records.append(rec)
        print(f"  accuracy  as-scored {as_scored['accuracy']:.4f} -> "
              f"dedup {dedup['accuracy']:.4f}  (delta {rec['delta_accuracy']:+.4f})")
        print(f"  macro F1  as-scored {as_scored['macro_f1']:.4f} -> "
              f"dedup {dedup['macro_f1']:.4f}  (delta {rec['delta_macro_f1']:+.4f})")

    if not records:
        print("\nNo runs evaluated.")
        return

    csv_path = args.out_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nwrote {csv_path}")

    deltas = [abs(r["delta_accuracy"]) for r in records]
    worst = max(deltas)
    md_path = args.out_prefix.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Duplicate-row defect — measured impact on reported test metrics\n\n")
        f.write(
            f"`{args.manifest}` test split: **{n_rows} rows, {n_unique} unique images**. "
            f"{n_dup_rows} rows ({n_dup_rows / n_rows:.2%}) are exact duplicates on "
            f"`{'`, `'.join(IDENTITY)}`, affecting {n_images_affected} distinct images. "
            "They come from the Bunch Piles crop pipeline and never cross a split boundary, "
            "so they are double-counting, not leakage.\n\n"
        )
        f.write("Duplicated images by class: "
                + ", ".join(f"{c} {dup_classes.get(c, 0)}" for c in CLASS_NAMES) + ".\n\n")
        f.write(
            "*as-scored* weights each image by its multiplicity in the manifest and reproduces "
            "the published figures; *dedup* counts each image once. Predictions are identical "
            "in both passes — only the weighting differs.\n\n"
        )
        f.write("| Run | Accuracy as-scored | Accuracy dedup | Delta | Macro F1 as-scored | "
                "Macro F1 dedup | Delta | Accuracy on the duplicated images |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in records:
            f.write(
                f"| {r['run_name']} | {r['as_scored_accuracy'] * 100:.2f}% | "
                f"{r['dedup_accuracy'] * 100:.2f}% | {r['delta_accuracy'] * 100:+.2f} pts | "
                f"{r['as_scored_macro_f1']:.4f} | {r['dedup_macro_f1']:.4f} | "
                f"{r['delta_macro_f1']:+.4f} | {r['accuracy_on_duplicated_images'] * 100:.2f}% |\n"
            )
        f.write(
            f"\n**Largest absolute accuracy shift across the runs measured: "
            f"{worst * 100:.2f} points.**\n\n"
            "## What this does and does not establish\n\n"
            "This measures the *reporting* error only: the extent to which published test "
            "metrics differ from what they would have been had the test split been "
            "deduplicated before scoring. It does not measure the *training* error. The "
            "training split carries its own duplicate rows, which double-weight some training "
            "images; estimating that effect would require retraining on a deduplicated "
            "manifest and is out of scope here. The correct claim from this table is about "
            "the numbers as reported, not about the models as trained.\n"
        )
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
