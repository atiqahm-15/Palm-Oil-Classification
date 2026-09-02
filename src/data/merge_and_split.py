"""Merge all per-source manifests and split train/valid/test by source photo.

Reads every data/processed/manifest_*.csv (one row per image/patch, produced by
crop_bunch_piles.py and catalog_malaysia.py) and assigns each row to a split.

Splitting unit is the *source photo*, not the crop: every patch cropped from the same
bunch-pile frame is kept in a single split. This is the rule from CLAUDE.md — the
FYP1 baseline leaked 194/205 source photos across splits by splitting on crops
instead, which inflated validation accuracy. A group (source_dataset, source_photo)
is assigned as a whole using a greedy balanced heuristic: for each group, in
descending size order, assign it to whichever split is currently furthest below its
(source_dataset, class) target.

Targets are per (source_dataset, class), not just per class — an earlier version
balanced only by class, which silently let source composition skew hard between
splits: Bunch Piles' source photos are large groups (many crops per pile frame),
while the other 3 sources are all size-1 groups (one image = one photo). Sorting by
group size descending and always picking the split "furthest below target" means big
groups get funneled into train first, simply because train's raw target number is
largest, not because of anything about the source itself — found empirically as
Bunch Piles being 82-94% of every class's train split but only 14-48% of valid/test.
Balancing per (source, class) instead of per class makes each source keep its own
~70/15/15 split independent of how big its groups happen to be.

Usage:
    python src/data/merge_and_split.py [--dst data/processed] [--seed 42]
        [--train 0.7] [--valid 0.15] [--test 0.15]
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

FIELDNAMES = ["path", "class", "source_dataset", "source_photo", "split"]
OUTPUT_NAME = "manifest_full.csv"
# Not a per-source input manifest — different schema (condition/orig_path, no split),
# produced by make_lighting_stress_test.py *from* manifest_full.csv, so including it here
# would be circular as well as schema-incompatible.
NON_SOURCE_MANIFESTS = {OUTPUT_NAME, "manifest_lighting_stress_test.csv"}


def load_manifests(dst: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(dst.glob("manifest_*.csv")):
        if manifest_path.name in NON_SOURCE_MANIFESTS:
            continue
        with manifest_path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def assign_splits(rows: list[dict], ratios: dict[str, float], seed: int) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source_dataset"], row["source_photo"])].append(row)

    # Targets per (source_dataset, class) — see module docstring for why per-class alone
    # let one source's group-size distribution skew composition between splits.
    per_source_class_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_source_class_counts[row["source_dataset"]][row["class"]] += 1

    targets = {
        split: {
            src: {cls: count * ratio for cls, count in classes.items()}
            for src, classes in per_source_class_counts.items()
        }
        for split, ratio in ratios.items()
    }
    current = {split: defaultdict(Counter) for split in ratios}

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    for (source, _photo), group_rows in group_items:
        group_counts = Counter(row["class"] for row in group_rows)
        scores = {
            split: sum(
                group_counts[c] * max(0.0, targets[split][source][c] - current[split][source][c])
                for c in group_counts
            )
            for split in ratios
        }
        if all(score == 0 for score in scores.values()):
            chosen = min(ratios, key=lambda s: sum(current[s][source].values()))
        else:
            chosen = max(scores, key=scores.get)
        for row in group_rows:
            row["split"] = chosen
        current[chosen][source].update(group_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--valid", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    ratios = {"train": args.train, "valid": args.valid, "test": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit(f"--train/--valid/--test must sum to 1.0, got {sum(ratios.values())}")

    rows = load_manifests(args.dst)
    if not rows:
        raise SystemExit(
            f"No manifest_*.csv files found under {args.dst}. "
            f"Run crop_bunch_piles.py and/or catalog_malaysia.py first."
        )

    assign_splits(rows, ratios, args.seed)

    output_path = args.dst / OUTPUT_NAME
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}\n")
    print(f"{'split':<8}{'class':<12}{'count':>8}")
    for split in ratios:
        split_rows = [r for r in rows if r["split"] == split]
        counts = Counter(r["class"] for r in split_rows)
        for cls in sorted(counts):
            print(f"{split:<8}{cls:<12}{counts[cls]:>8}")
        print(f"{split:<8}{'TOTAL':<12}{len(split_rows):>8}")

    n_photos = len({(r["source_dataset"], r["source_photo"]) for r in rows})
    n_photos_per_split = {
        split: len({(r["source_dataset"], r["source_photo"]) for r in rows if r["split"] == split})
        for split in ratios
    }
    print(f"\n{n_photos} unique source photos total: {n_photos_per_split}")

    print(f"\nPer-source share of each class's split (sanity check — each should land near "
          f"{', '.join(f'{s}={r:.0%}' for s, r in ratios.items())}):")
    sources = sorted({r["source_dataset"] for r in rows})
    for cls in sorted({r["class"] for r in rows}):
        parts = []
        for split in ratios:
            split_rows = [r for r in rows if r["split"] == split and r["class"] == cls]
            total = len(split_rows)
            src_counts = Counter(r["source_dataset"] for r in split_rows)
            worst_src = max(sources, key=lambda s: src_counts[s] / total if total else 0)
            parts.append(f"{split}: {worst_src}={src_counts[worst_src]}/{total}"
                         f"({src_counts[worst_src]/total:.0%})" if total else f"{split}: (empty)")
        print(f"  {cls:<10} " + "  ".join(parts))


if __name__ == "__main__":
    main()
