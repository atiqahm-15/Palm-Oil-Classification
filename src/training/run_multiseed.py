"""Run the multi-seed replication campaign for the Phase 2 architecture comparison.

Why this exists: the block-vs-layer fine-tuning comparison (PROJECT_PLAN.md section 8)
showed a 20-point underripe-recall swing between two MobileNetV2 runs that differed by
exactly one unfrozen layer - i.e. run-to-run variance, not an effect. That makes every
single-run comparison in this project uninterpretable on its own, including MobileNetV2 vs
EfficientNetB0 (73.73% vs 73.18%). Architecture selection needs mean +/- std over repeated
runs, which is what this campaign produces.

What the spread actually measures: `--seed` is wired through weight init, the train-split
shuffle, and the augmentation layers, but the tf.data pipeline uses
num_parallel_calls=AUTOTUNE, so runs are not bit-reproducible even at a fixed seed. The
spread is therefore total run-to-run variance (seed + nondeterminism) - exactly the
quantity needed to decide whether an accuracy gap is real.

Compute backend matters. CPU (oneDNN) and GPU (cuDNN) select different convolution
algorithms, so runs from the two are not replicates of the same process and must not be
pooled - see runtime_info.py and the guard in aggregate_seeds.py. Use `--tag` to keep each
backend's runs in their own namespace:

    # CPU (Windows, TF 2.20 has no Windows-native CUDA):
    python src/training/run_multiseed.py --seeds 43 44

    # GPU (WSL2), all three seeds re-run so the set is internally consistent:
    /opt/tfgpu/bin/python src/training/run_multiseed.py --tag gpu --seeds 42 43 44

    # FastViT depth ablation: it exposes 10 blocks vs 16 for both CNNs, so K=3 is NOT the
    # same proportional depth (see finetune.py). K=2 is the proportional match; both are run
    # so the difference is measured rather than assumed. Separate tags keep them unpooled:
    ... run_multiseed.py --archs fastvit --tag gpu_k2 --finetune-blocks 2 --seeds 42 43 44
    ... run_multiseed.py --archs fastvit --tag gpu_k3 --finetune-blocks 3 --seeds 42 43 44

Resumable: a run whose results/ directory already exists is skipped, so re-running after
an interruption picks up where it stopped.
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
# Untagged seed 42 is the pre-existing results/<arch>_fyp2_blockft/ run, kept under its
# original name so the earlier CPU results stay addressable.
BASELINE_SEED = 42


def run_name_for(arch: str, seed: int, tag: str | None) -> str:
    if tag:
        return f"{arch}_fyp2_blockft_{tag}_seed{seed}"
    if seed == BASELINE_SEED:
        return f"{arch}_fyp2_blockft"
    return f"{arch}_fyp2_blockft_seed{seed}"


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--archs", nargs="+", default=list(TRAIN_SCRIPTS), choices=list(TRAIN_SCRIPTS))
    parser.add_argument("--tag", default=None,
                        help="namespace for this campaign's run directories, e.g. 'gpu'. Keeps "
                             "runs from different compute backends from colliding or being pooled.")
    parser.add_argument("--finetune-blocks", type=int, default=None,
                        help="fine-tuning depth in blocks, passed through to the training script. "
                             "Omit to use its default (K=3). Vary this only with a matching --tag, "
                             "so runs at different depths land in separate namespaces.")
    parser.add_argument("--epochs-frozen", type=int, default=None,
                        help="Stage 1 epoch cap, passed through to the training script. "
                             "Omit to use schedule.py's default. Vary only with a "
                             "matching --tag: a different cap is a different recipe, and "
                             "aggregate_seeds.py cannot detect that from recipe_version "
                             "alone.")
    parser.add_argument("--epochs-finetune", type=int, default=None,
                        help="Stage 2 epoch cap, passed through to the training script. "
                             "Same tagging caveat as --epochs-frozen.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Interleave architectures so an interruption still leaves both with equal coverage,
    # rather than 3 seeds of one and 1 of the other.
    jobs = [(arch, seed) for seed in args.seeds for arch in args.archs]

    print(f"Campaign: {len(jobs)} run(s) — {', '.join(f'{a}@{s}' for a, s in jobs)}"
          + (f" [tag={args.tag}]" if args.tag else ""))
    t0 = time.time()
    completed, skipped, failed = 0, 0, []
    for i, (arch, seed) in enumerate(jobs, 1):
        name = run_name_for(arch, seed, args.tag)
        run_dir = Path("results") / name
        print(f"\n{'=' * 70}\n[{i}/{len(jobs)}] {name}  ({time.time() - t0:.0f}s elapsed)\n{'=' * 70}", flush=True)
        if run_dir.exists():
            print(f"{run_dir} already exists — skipping (campaign is resumable).", flush=True)
            skipped += 1
            continue
        # One failing run must not abandon the rest of the campaign: these are hours of
        # compute, and a fault in a later stage (e.g. evaluation) leaves earlier runs
        # perfectly usable. Failures are collected and reported at the end instead.
        train_cmd = [sys.executable, TRAIN_SCRIPTS[arch], "--run-name", name, "--seed", str(seed)]
        if args.finetune_blocks is not None:
            train_cmd += ["--finetune-blocks", str(args.finetune_blocks)]
        if args.epochs_frozen is not None:
            train_cmd += ["--epochs-frozen", str(args.epochs_frozen)]
        if args.epochs_finetune is not None:
            train_cmd += ["--epochs-finetune", str(args.epochs_finetune)]
        try:
            run(train_cmd, args.dry_run)
            run([sys.executable, "src/training/evaluate.py", "--run-name", name], args.dry_run)
            completed += 1
        except subprocess.CalledProcessError as e:
            print(f"\n!!! {name} FAILED (exit {e.returncode}): {' '.join(e.cmd)}\n"
                  f"    Continuing with the remaining runs; re-run this one afterwards.", flush=True)
            failed.append(name)

    print(f"\nCampaign done in {(time.time() - t0) / 60:.1f} min "
          f"({completed} run, {skipped} skipped, {len(failed)} failed).")
    if failed:
        print("FAILED runs (their results/ dirs may be incomplete — delete before retrying):")
        for name in failed:
            print(f"  - {name}")
    pattern = f"*_blockft_{args.tag}_seed*" if args.tag else "*_fyp2_blockft*"
    print(f'Next: python src/training/aggregate_seeds.py --pattern "{pattern}"')
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
