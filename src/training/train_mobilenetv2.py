"""Train the MobileNetV2 baseline on the unified 4-source dataset (Phase 2, Training Orchestrator).

Two-stage transfer learning, carried over from the FYP1 notebook methodology:
  Stage 1 (feature extraction): MobileNetV2 base frozen, only the classification head trains.
  Stage 2 (fine-tuning): last K base blocks unfrozen, trained at a low learning rate.

Unlike FYP1, this run adds class-weighted loss (inverse-frequency, computed from the
actual train-split counts) on top of label smoothing, since the unified dataset's classes
aren't perfectly balanced and unripe recall was FYP1's weakest point (recall 0.38).

Every run's config and metrics are written to results/<run-name>/ and never overwritten
(CLAUDE.md convention) — the script refuses to run if that directory already exists.

Usage:
    python src/training/train_mobilenetv2.py --run-name mobilenetv2_fyp2_baseline
        [--manifest data/processed/manifest_full.csv] [--epochs-frozen 10] [--epochs-finetune 10]
        [--finetune-blocks 3] [--batch-size 32] [--seed 42]

Fine-tuning depth is specified in blocks, not layers (see finetune.py) — a fixed layer
count meant different proportional depths across architectures and made the Phase 2
comparison unfair.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from dataset import CLASS_NAMES, IMG_SIZE, attach_sample_weights, build_dataset, load_manifest_split
from schedule import add_training_args, run_two_stage
from runtime_info import compute_platform, describe

LABEL_SMOOTHING = 0.15


def compute_train_class_weights(manifest_path: Path, seed: int) -> dict[int, float]:
    _, labels = load_manifest_split(manifest_path, "train")
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # training=False is pinned into the graph deliberately: it keeps BatchNorm in inference
    # mode (frozen ImageNet moving statistics) even during Stage 2 fine-tuning, which is the
    # standard transfer-learning recipe and what every earlier baseline here was trained with.
    # Dropping it lets BN re-estimate statistics from our batches and measurably changes
    # results, so it must not be removed when varying fine-tuning depth.
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    add_training_args(parser, default_finetune_blocks=3)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    # Recorded into config.json: CPU and GPU runs are not poolable as replicates.
    platform_info = compute_platform()
    print(f"Compute platform: {describe(platform_info)}")

    run_dir = Path("results") / args.run_name
    if run_dir.exists():
        raise SystemExit(
            f"{run_dir} already exists. Every run gets its own directory (CLAUDE.md convention) "
            f"— pick a different --run-name instead of overwriting."
        )
    run_dir.mkdir(parents=True)

    class_weights = compute_train_class_weights(args.manifest, args.seed)
    print(f"Class weights (balanced, by train-split frequency): "
          f"{ {CLASS_NAMES[k]: round(v, 3) for k, v in class_weights.items()} }")

    train_ds = build_dataset(args.manifest, "train", batch_size=args.batch_size, seed=args.seed)
    train_ds = attach_sample_weights(train_ds, class_weights)
    valid_ds = build_dataset(args.manifest, "valid", batch_size=args.batch_size, seed=args.seed)

    model, base = build_model()
    ckpt_path = run_dir / "best.weights.h5"
    n_train = len(load_manifest_split(args.manifest, "train")[0])

    # The recipe itself lives in schedule.py so all three architectures share it by
    # construction -- see that module's docstring for the v1 -> v2 rationale.
    t0 = time.time()
    history_frozen, history_finetune, finetune_info, schedule_info = run_two_stage(
        model=model,
        base=base,
        train_ds=train_ds,
        valid_ds=valid_ds,
        ckpt_path=ckpt_path,
        args=args,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        n_train=n_train,
    )
    train_seconds = time.time() - t0

    model.load_weights(ckpt_path)
    model_path = run_dir / "model.keras"
    model.save(model_path)

    history = {}
    for h in (history_frozen, history_finetune):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)
    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    config = {
        "architecture": "MobileNetV2",
        "run_name": args.run_name,
        "manifest": str(args.manifest),
        "img_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "epochs_frozen": args.epochs_frozen,
        "epochs_finetune": args.epochs_finetune,
        **schedule_info,
        "finetune_blocks": args.finetune_blocks,
        "finetune_detail": finetune_info,
        "label_smoothing": LABEL_SMOOTHING,
        "class_weights": {CLASS_NAMES[k]: v for k, v in class_weights.items()},
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "compute_platform": platform_info,
        "train_seconds": round(train_seconds, 1),
        "final_epochs_run_frozen": len(history_frozen.history["loss"]),
        "final_epochs_run_finetune": len(history_finetune.history["loss"]),
        "best_val_accuracy": max(history["val_accuracy"]),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone in {train_seconds/60:.1f} min. Best val_accuracy={config['best_val_accuracy']:.4f}")
    print(f"Model saved to {model_path}")
    print(f"Run evaluate.py --run-name {args.run_name} next for test-set + Lighting Stress-Test metrics.")


if __name__ == "__main__":
    main()
