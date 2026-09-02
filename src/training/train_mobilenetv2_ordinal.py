"""Train MobileNetV2 with a CORAL ordinal head, as a comparison against the categorical
baseline (train_mobilenetv2.py) — see ordinal.py for why and how.

Same backbone, same two-stage transfer learning recipe, same class weighting as the
baseline, so the *only* thing this experiment changes is the loss/output head — an
apples-to-apples comparison of "does encoding rank order help the underripe<->ripe
confusion found in the baseline's error analysis" (PROJECT_PLAN.md decisions log).

The saved model outputs plain (batch, 4) per-class probabilities (via CoralLogitsToProbs
baked into the model graph), so evaluate.py works on it completely unchanged.

Usage:
    python src/training/train_mobilenetv2_ordinal.py --run-name mobilenetv2_fyp2_ordinal
        [--manifest data/processed/manifest_full.csv] [--epochs-frozen 10] [--epochs-finetune 10]
        [--finetune-layers 30] [--batch-size 32] [--seed 42]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from dataset import CLASS_NAMES, IMG_SIZE, attach_sample_weights, build_dataset, load_manifest_split
from ordinal import CoralLayer, CoralLogitsToProbs, OrdinalAccuracy, coral_loss, onehot_to_cumulative

LABEL_SMOOTHING = None  # not meaningful for CORAL's binary threshold tasks; omitted


def compute_train_class_weights(manifest_path: Path, seed: int) -> dict[int, float]:
    _, labels = load_manifest_split(manifest_path, "train")
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def to_cumulative_labels(ds: tf.data.Dataset) -> tf.data.Dataset:
    def _convert(image, label, *rest):
        return (image, onehot_to_cumulative(label), *rest)

    return ds.map(_convert, num_parallel_calls=tf.data.AUTOTUNE)


def build_model(finetune: bool, finetune_layers: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = finetune
    if finetune:
        for layer in base.layers[:-finetune_layers]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=finetune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    logits = CoralLayer(len(CLASS_NAMES))(x)

    train_model = tf.keras.Model(inputs, logits)
    inference_model = tf.keras.Model(inputs, CoralLogitsToProbs()(logits))
    return train_model, inference_model, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--epochs-frozen", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--finetune-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

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
    train_ds = to_cumulative_labels(train_ds)
    valid_ds = build_dataset(args.manifest, "valid", batch_size=args.batch_size, seed=args.seed)
    valid_ds = to_cumulative_labels(valid_ds)

    train_model, inference_model, base = build_model(finetune=False, finetune_layers=args.finetune_layers)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=coral_loss,
        metrics=[OrdinalAccuracy()],
    )

    ckpt_path = run_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ordinal_accuracy", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_ordinal_accuracy", mode="max", save_best_only=True, save_weights_only=True
        ),
    ]

    t0 = time.time()
    print(f"\n=== Stage 1: frozen feature extraction ({args.epochs_frozen} epochs) ===")
    history_frozen = train_model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs_frozen,
        callbacks=callbacks,
    )

    print(f"\n=== Stage 2: fine-tuning last {args.finetune_layers} base layers ({args.epochs_finetune} epochs) ===")
    base.trainable = True
    for layer in base.layers[:-args.finetune_layers]:
        layer.trainable = False
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=coral_loss,
        metrics=[OrdinalAccuracy()],
    )
    history_finetune = train_model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs_finetune,
        callbacks=callbacks,
    )
    train_seconds = time.time() - t0

    train_model.load_weights(ckpt_path)  # inference_model shares the same layer objects/weights
    model_path = run_dir / "model.keras"
    inference_model.save(model_path)

    history = {}
    for h in (history_frozen, history_finetune):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)
    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    config = {
        "architecture": "MobileNetV2-CORAL-ordinal",
        "run_name": args.run_name,
        "manifest": str(args.manifest),
        "img_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "epochs_frozen": args.epochs_frozen,
        "epochs_finetune": args.epochs_finetune,
        "finetune_layers": args.finetune_layers,
        "loss": "CORAL (ordinal, rank-consistent binary thresholds)",
        "class_weights": {CLASS_NAMES[k]: v for k, v in class_weights.items()},
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "train_seconds": round(train_seconds, 1),
        "final_epochs_run_frozen": len(history_frozen.history["loss"]),
        "final_epochs_run_finetune": len(history_finetune.history["loss"]),
        "best_val_accuracy": max(history["val_ordinal_accuracy"]),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone in {train_seconds/60:.1f} min. Best val_ordinal_accuracy={config['best_val_accuracy']:.4f}")
    print(f"Model saved to {model_path}")
    print(f"Run evaluate.py --run-name {args.run_name} next for test-set + Lighting Stress-Test metrics.")


if __name__ == "__main__":
    main()
