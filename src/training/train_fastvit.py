"""Train FastViT on the unified 4-source dataset (Phase 2, Training Orchestrator).

Third of the 3 architectures required by Objective 2 -- the "lightweight ViT variant".
Uses the identical recipe as train_mobilenetv2.py / train_efficientnetb0.py (two-stage
frozen->fine-tune, class-weighted loss, label smoothing) so all three runs are a fair,
matched comparison -- same manifest, same splits, same hyperparameters, only the backbone
differs.

Substituted for "MobileViT" per PROJECT_PLAN.md's decisions log. MobileViT has no pretrained
weights that load under this project's Keras 3 stack: keras.applications has no MobileViT;
TF-Hub's is TF1-Hub-format (the same trap that forced EfficientNet-Lite0 -> EfficientNetB0);
HuggingFace's TF port needs tf_keras, which already failed here on checkpoint reload; and
keras_cv_attention_models' MobileViT dies in transformer_pre_process applying a raw TF op to
a KerasTensor, which Keras 3 forbids. A sweep of 15 lightweight transformers in that package
found only FastViT_T8/T12 build at all -- MobileViT V2, LeViT, EfficientViT, EdgeNeXt and
TinyViT all fail identically. FastViT (Apple, CVPR 2023) is a mobile-targeted hybrid
CNN-transformer, so it answers Objective 2's "lightweight ViT variant" (MobileViT was the
plan's example, not its requirement), and T8's 3.20M params sits between MobileNetV2's 2.26M
and EfficientNetB0's 4.05M, keeping the three architectures in one comparable cost band.

Note: FastViT expects `torch`-mode scaling -- (x - mean) / std over [0, 255] pixels with the
ImageNet constants below. kecam does NOT bake this into the graph (add_pre_post_process only
attaches a `model.preprocess_input` helper attribute), so this script must apply it itself.
Beware that kecam's own docstring states the constants as [0.485, 0.456, 0.406] /
[0.229, 0.224, 0.225]; init_mean_std_by_rescale_mode actually multiplies both by 255. Using
the docstring values would mis-scale the input ~255x with no error raised -- silently
producing a garbage third data point. Since dataset.py's shared pipeline always emits
[-1, 1], this script converts locally rather than changing the pipeline other runs depend on.

Fine-tuning depth: FastViT_T8 exposes 10 blocks vs 16 for both CNNs, so K is NOT directly
comparable across architectures (see finetune.py). Measured: K=2 unfreezes 38/246 layers
(15.4%) and K=3 unfreezes 69/246 (28.0%), while the CNNs at K=3 sit at 18.8% (MobileNetV2)
and 19.4% (EfficientNetB0). The CNN depth therefore falls *between* FastViT's two settings
and no integer K matches it. Default here is 2, the nearer of the two; the campaign runs
both so the CNNs are bracketed and depth sensitivity is measured rather than assumed.

Usage:
    python src/training/train_fastvit.py --run-name fastvit_fyp2_blockft_gpu_k2_seed42
        [--manifest data/processed/manifest_full.csv] [--epochs-frozen 10] [--epochs-finetune 10]
        [--finetune-blocks 2] [--batch-size 32] [--seed 42] [--variant FastViT_T8]
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

# kecam rescale_mode="torch", as resolved by init_mean_std_by_rescale_mode: the [0,1]-scale
# ImageNet constants multiplied by 255, applied to [0, 255] pixels.
TORCH_MEAN = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]  # [123.675, 116.28,  103.53 ]
TORCH_STD = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]   # [ 58.395,  57.12,   57.375]


def compute_train_class_weights(manifest_path: Path, seed: int) -> dict[int, float]:
    _, labels = load_manifest_split(manifest_path, "train")
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_model(variant: str) -> tuple[tf.keras.Model, tf.keras.Model]:
    from keras_cv_attention_models import fastvit

    base = getattr(fastvit, variant)(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=0, pretrained="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # dataset.py's pipeline emits [-1, 1]; FastViT wants torch-mode normalised [0, 255],
    # and kecam does not apply it inside the model, so do both steps here.
    x = (inputs + 1.0) * 127.5
    x = (x - tf.constant(TORCH_MEAN, dtype=tf.float32)) / tf.constant(TORCH_STD, dtype=tf.float32)
    # training=False is pinned into the graph deliberately: it keeps BatchNorm in inference
    # mode (frozen ImageNet moving statistics) even during Stage 2 fine-tuning, which is the
    # standard transfer-learning recipe and what every earlier baseline here was trained with.
    # Dropping it lets BN re-estimate statistics from our batches and measurably changes
    # results, so it must not be removed when varying fine-tuning depth.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    add_training_args(parser, default_finetune_blocks=2)
    parser.add_argument("--variant", default="FastViT_T8")
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

    model, base = build_model(args.variant)
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
        "architecture": args.variant,
        "architecture_note": (
            "Substituted for MobileViT — no MobileViT pretrained weights load under Keras 3 "
            "(keras.applications lacks it; TF-Hub is TF1-format; HuggingFace needs tf_keras, "
            "which failed here on checkpoint reload; keras_cv_attention_models' MobileViT "
            "applies a raw TF op to a KerasTensor). FastViT was the only lightweight "
            "transformer of 15 swept in that package that builds under Keras 3. "
            "See PROJECT_PLAN.md decisions log."
        ),
        "finetune_blocks_note": (
            f"{args.variant} exposes {finetune_info['total_blocks']} blocks vs 16 for both CNNs, "
            f"so K is not directly comparable across architectures. The CNNs at K=3 unfreeze "
            f"18.8% (MobileNetV2) and 19.4% (EfficientNetB0) of their layers, which falls "
            f"between FastViT's K=2 (15.4%) and K=3 (28.0%) — no integer K matches them. "
            f"This run used K={args.finetune_blocks}: "
            f"{finetune_info['layers_unfrozen']}/{finetune_info['total_layers']} layers "
            f"({finetune_info['layers_unfrozen'] / finetune_info['total_layers'] * 100:.1f}%)."
        ),
        "rescale_mode": "torch",
        "rescale_note": (
            "kecam rescale_mode='torch' applied in-script: [-1,1] -> [0,255] -> (x-mean)/std "
            f"with mean={[round(v, 3) for v in TORCH_MEAN]}, std={[round(v, 3) for v in TORCH_STD]}. "
            "kecam does not bake this into the graph, and its docstring understates the "
            "constants by a factor of 255."
        ),
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
