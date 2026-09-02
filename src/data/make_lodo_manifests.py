"""Build leave-one-dataset-out (LODO) manifests from the unified manifest.

Per SYSTEM_ARCHITECTURE.md section 5, LODO is the project's primary generalization evidence:
each of the 4 source datasets is held out entirely in turn, the model trains on the other 3,
and is tested purely on the held-out source. It answers "does this still work on a
camera/location it has never encountered at all?" - a different question from the main split.

Split rule per fold (held-out source S):
  test  = every row from S, regardless of its split in the main manifest
  train = rows from the other 3 sources whose main split is train OR test
  valid = rows from the other 3 sources whose main split is valid

Two choices worth stating explicitly, since both are the kind of thing a viva examiner asks:

1. A validation split is still needed - early stopping and checkpoint selection have to
   happen somewhere - and it is drawn only from the 3 training sources. The held-out source
   is never seen during training or model selection, which is the whole point of LODO;
   validating on it would leak the thing being measured.
2. The other 3 sources' main-split *test* rows are folded into train. They are not being
   evaluated in this fold, so withholding them would just discard usable training data.
   This does mean each fold's training set differs in size from the main split's - but that
   is unavoidable anyway once a whole source is removed, so LODO accuracy is never directly
   comparable to the main-split number regardless.

Source-photo grouping is inherited from the main manifest and is not re-derived here: groups
never straddle train/valid because the main split already guaranteed that, and test is a
whole source so no group can straddle the train/test boundary either.

Usage:
    python src/data/make_lodo_manifests.py [--manifest data/processed/manifest_full.csv]
        [--out-dir data/processed/lodo]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/lodo"))
    args = parser.parse_args()

    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    sources = sorted({r["source_dataset"] for r in rows})
    print(f"{len(rows)} rows, {len(sources)} sources: {', '.join(sources)}\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for held_out in sources:
        out_rows = []
        for r in rows:
            new = dict(r)
            if r["source_dataset"] == held_out:
                new["split"] = "test"
            elif r["split"] == "valid":
                new["split"] = "valid"
            else:  # main-split train and test rows from the other 3 sources both train here
                new["split"] = "train"
            out_rows.append(new)

        out_path = args.out_dir / f"manifest_lodo_{held_out}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

        # --- verification, printed per fold so the guarantees are auditable ---
        by_split = {s: [r for r in out_rows if r["split"] == s] for s in ("train", "valid", "test")}
        test_sources = {r["source_dataset"] for r in by_split["test"]}
        trainval_sources = {r["source_dataset"] for r in by_split["train"] + by_split["valid"]}
        assert test_sources == {held_out}, f"test contains {test_sources}, expected only {held_out}"
        assert held_out not in trainval_sources, f"{held_out} leaked into train/valid"

        test_paths = {r["path"] for r in by_split["test"]}
        trainval_paths = {r["path"] for r in by_split["train"] + by_split["valid"]}
        assert not (test_paths & trainval_paths), "path overlap between test and train/valid"

        groups_tv = {(r["source_dataset"], r["source_photo"]) for r in by_split["train"]}
        groups_v = {(r["source_dataset"], r["source_photo"]) for r in by_split["valid"]}
        straddle = groups_tv & groups_v
        assert not straddle, f"{len(straddle)} source photos straddle train/valid"

        classes = Counter(r["class"] for r in by_split["test"])
        missing = [c for c in ("unripe", "underripe", "ripe", "overripe") if c not in classes]
        print(f"hold out {held_out:<18} train={len(by_split['train']):>6} "
              f"valid={len(by_split['valid']):>5} test={len(by_split['test']):>6}")
        print(f"    test classes: { dict(classes) }")
        if missing:
            # Not an error: malaysia_outdoor genuinely has no underripe images. Recorded so
            # the fold's per-class metrics are read with that in mind.
            print(f"    NOTE: held-out source has no {', '.join(missing)} images — "
                  f"that class is undefined in this fold's recall/ROC-AUC.")
        print(f"    -> {out_path}")

    print("\nAll folds verified: held-out source appears only in test, zero path overlap, "
          "no source photo straddling train/valid.")


if __name__ == "__main__":
    main()
