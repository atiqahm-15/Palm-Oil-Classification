"""Run the leave-one-dataset-out (LODO) generalization check.

LODO is the project's primary generalization evidence (SYSTEM_ARCHITECTURE.md section 5):
each source is held out entirely in turn, the model trains on the other 3, and is tested
purely on the held-out source - "does this still work on a camera/location it has never
encountered at all?"

Runs on the winning architecture only, and that winner has changed twice. The Aug-2026
multi-seed campaign put EfficientNetB0 (76.45%) over MobileNetV2 (73.61%); the Sep-2026
recipe-v2 campaign then put FastViT_T8 (94.55% +/- 0.31) clearly ahead of both
(EfficientNetB0 91.26% +/- 0.39, MobileNetV2 89.62% +/- 0.15). LODO therefore runs on
FastViT now. The v1 LODO results (EfficientNetB0, mean 55.96% across folds) were measured
under the under-trained recipe and cannot be compared against a v2 main split - use --tag
to keep the two campaigns in separate namespaces.

Keep the LODO recipe identical to the main-split campaign it will be compared against.
The generalization gap is the quantity of interest, so the two sides of the subtraction
must share a recipe: pass the same --finetune-blocks and epoch caps used for the main run.

Multiple seeds per fold, because this project has established that single runs are within
noise. Jobs are ordered seed-major (all 4 folds at seed 42, then all 4 at seed 43, ...) so
an interruption leaves a complete single-seed sweep rather than a partial one.

Interpretation caveat worth carrying into the report: fold training-set sizes differ a lot
(holding out bunch_piles leaves only ~4.7k training images, versus ~15.1k when holding out
malaysia_outdoor) because bunch_piles is 69% of the corpus. A weak bunch_piles fold
therefore confounds domain shift with reduced training data; the two cannot be separated
from this experiment alone.

The Lighting Stress-Test is skipped for these folds - they measure cross-source domain
shift, and lighting robustness is already measured on the main split.

Usage:
    python src/training/run_lodo.py --arch fastvit --tag v2 --finetune-blocks 3 \
        --epochs-frozen 50 [--seeds 42 43 44] [--dry-run]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

TRAIN_SCRIPTS = {
    "mobilenetv2": "src/training/train_mobilenetv2.py",
    "efficientnetb0": "src/training/train_efficientnetb0.py",
    "fastvit": "src/training/train_fastvit.py",
}
LODO_DIR = Path("data/processed/lodo")


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--arch", default="efficientnetb0", choices=list(TRAIN_SCRIPTS))
    parser.add_argument("--tag", default=None,
                        help="namespace for this campaign's run dirs, e.g. 'v2'. Required "
                             "whenever the recipe differs from an earlier LODO campaign of "
                             "the same architecture, so the two cannot collide or be pooled.")
    parser.add_argument("--finetune-blocks", type=int, default=None,
                        help="fine-tuning depth in blocks, passed through to the training "
                             "script. FastViT defaults to K=2; the v2 campaign used K=3.")
    parser.add_argument("--epochs-frozen", type=int, default=None,
                        help="Stage 1 epoch cap, passed through. Match the main-split run.")
    parser.add_argument("--epochs-finetune", type=int, default=None,
                        help="Stage 2 epoch cap, passed through. Match the main-split run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifests = sorted(LODO_DIR.glob("manifest_lodo_*.csv"))
    if not manifests:
        raise SystemExit(f"No fold manifests in {LODO_DIR}. Run src/data/make_lodo_manifests.py first.")
    folds = [(m.stem.replace("manifest_lodo_", ""), m) for m in manifests]

    # Seed-major: a full single-seed sweep across all folds completes before seed 2 starts.
    jobs = [(fold, manifest, seed) for seed in args.seeds for fold, manifest in folds]

    print(f"LODO campaign: {args.arch}, {len(folds)} folds x {len(args.seeds)} seeds = {len(jobs)} runs")
    print(f"Folds: {', '.join(f for f, _ in folds)}")

    t0 = time.time()
    completed, skipped, failed = 0, 0, []
    for i, (fold, manifest, seed) in enumerate(jobs, 1):
        name = (f"{args.arch}_lodo_{args.tag}_{fold}_seed{seed}" if args.tag
                else f"{args.arch}_lodo_{fold}_seed{seed}")
        run_dir = Path("results") / name
        print(f"\n{'=' * 70}\n[{i}/{len(jobs)}] {name}  ({time.time() - t0:.0f}s elapsed)\n{'=' * 70}", flush=True)
        if run_dir.exists():
            print(f"{run_dir} already exists — skipping (campaign is resumable).", flush=True)
            skipped += 1
            continue
        try:
            train_cmd = [sys.executable, TRAIN_SCRIPTS[args.arch], "--run-name", name,
                         "--seed", str(seed), "--manifest", str(manifest)]
            if args.finetune_blocks is not None:
                train_cmd += ["--finetune-blocks", str(args.finetune_blocks)]
            if args.epochs_frozen is not None:
                train_cmd += ["--epochs-frozen", str(args.epochs_frozen)]
            if args.epochs_finetune is not None:
                train_cmd += ["--epochs-finetune", str(args.epochs_finetune)]
            run(train_cmd, args.dry_run)
            run([sys.executable, "src/training/evaluate.py", "--run-name", name,
                 "--manifest", str(manifest), "--skip-stress-test"], args.dry_run)
            completed += 1
        except subprocess.CalledProcessError as e:
            print(f"\n!!! {name} FAILED (exit {e.returncode}): {' '.join(e.cmd)}\n"
                  f"    Continuing with the remaining runs; re-run this one afterwards.", flush=True)
            failed.append(name)

    print(f"\nLODO done in {(time.time() - t0) / 60:.1f} min "
          f"({completed} run, {skipped} skipped, {len(failed)} failed).")
    if failed:
        print("FAILED runs (their results/ dirs may be incomplete — delete before retrying):")
        for name in failed:
            print(f"  - {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
