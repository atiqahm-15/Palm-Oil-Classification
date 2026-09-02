"""Evaluate a trained checkpoint: clean test set + Lighting Stress-Test set (Evaluator).

Reports against CLAUDE.md's NFR bar:
  - accuracy (>= 85% target) on the clean test set
  - per-condition accuracy (darkened/brightened/shadow) on the Lighting Stress-Test set,
    for "stable across daylight/shadow/low-light" — clean test acts as the daylight baseline
  - raw inference time and model size, as an early signal only. These are measured on this
    dev machine, not a phone, and pre-quantization — NOT the Phase 3 mobile benchmark that
    actually gates the <=2s / <=100MB NFRs.

Writes classification_report.txt, confusion_matrix.png, metrics.csv, and
lighting_stress_test_metrics.csv into the same results/<run-name>/ directory the
training run wrote its checkpoint to.

Usage:
    python src/training/evaluate.py --run-name mobilenetv2_fyp2_baseline
        [--manifest data/processed/manifest_full.csv]
        [--stress-test-manifest data/processed/manifest_lighting_stress_test.csv]
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import custom_layers  # noqa: F401 — same trick for FastViT: registers kecam's ChannelAffine
                      # et al. so load_model() can deserialize FastViT runs. Guarded inside,
                      # since kecam is only installed in the WSL GPU venv.
import ordinal  # noqa: F401 — unused directly, but importing registers CoralLayer/CoralLogitsToProbs
                 # with Keras's serialization registry so load_model() can deserialize ordinal
                 # checkpoints too. Without this import, loading an ordinal model raises
                 # "Could not locate class 'CoralLayer'" even though it's saved as registered.
from dataset import CLASS_NAMES, IMG_SIZE, _to_model_input, load_manifest_split


def load_images_as_batch(paths: list[str]) -> np.ndarray:
    images = []
    for p in paths:
        raw = tf.io.read_file(p)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        images.append(img)
    batch = tf.stack(images)
    batch, _ = _to_model_input(batch, None)
    return batch.numpy()


def evaluate_split(model, paths: list[str], labels: list[int], batch_size: int = 32):
    probs = []
    for i in range(0, len(paths), batch_size):
        batch = load_images_as_batch(paths[i : i + batch_size])
        probs.append(model.predict(batch, verbose=0))
    probs = np.concatenate(probs, axis=0)
    preds = np.argmax(probs, axis=1)
    y_true = np.array(labels)

    accuracy = float((preds == y_true).mean())
    # labels= is pinned so the report and matrix always span all 4 classes. A LODO fold
    # can hold out a source that contains no examples of a class at all (malaysia_outdoor
    # has no underripe), and without this the matrix collapses to the classes present and
    # no longer matches CLASS_NAMES when plotted.
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(y_true, preds, labels=labels, target_names=CLASS_NAMES,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=labels)
    try:
        roc_auc = roc_auc_score(tf.one_hot(y_true, len(CLASS_NAMES)).numpy(), probs, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")
    return accuracy, report, cm, roc_auc, preds, probs


def measure_inference_time(model_path: Path, paths: list[str], n_samples: int = 50) -> float:
    """Single-image CPU latency - an early, pre-quantization signal only.

    Measured on the CPU deliberately. Once some runs train on GPU (WSL2/cuDNN) and earlier
    ones on CPU, an unpinned measurement would report GPU latency for some rows of the
    architecture comparison table and CPU latency for others - all under a column named
    avg_inference_s_devcpu. A CPU number is also the better proxy for the phone, which is
    what the <=2s NFR actually gates; the Phase 3 on-device benchmark remains the real test.

    Loads its own CPU-resident copy of the model rather than reusing the caller's. The
    evaluation model's variables are placed on the GPU when one is visible, and predicting
    from a CPU device scope against GPU-placed variables fails with a cross-device resource
    access error - so reusing it here would break every GPU run.
    """
    sample_paths = paths[:n_samples]
    with tf.device("/CPU:0"):
        cpu_model = tf.keras.models.load_model(model_path)
        batch = load_images_as_batch(sample_paths)
        cpu_model.predict(batch[:1], verbose=0)  # warm up
        t0 = time.time()
        for i in range(len(sample_paths)):
            cpu_model.predict(batch[i : i + 1], verbose=0)
        elapsed = time.time() - t0
    return elapsed / len(sample_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument(
        "--stress-test-manifest", type=Path, default=Path("data/processed/manifest_lighting_stress_test.csv")
    )
    parser.add_argument("--skip-stress-test", action="store_true",
                        help="skip the Lighting Stress-Test pass. Used by LODO folds, which measure "
                             "cross-source domain shift; lighting robustness is measured on the main split.")
    args = parser.parse_args()

    run_dir = Path("results") / args.run_name
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found. Run train_mobilenetv2.py --run-name {args.run_name} first.")

    model = tf.keras.models.load_model(model_path)

    print("=== Clean test set ===")
    paths, labels = load_manifest_split(args.manifest, "test")
    accuracy, report, cm, roc_auc, preds, probs = evaluate_split(model, paths, labels)
    print(f"Accuracy: {accuracy:.4f}  ROC-AUC (OvR): {roc_auc:.4f}")
    print(report)

    (run_dir / "classification_report.txt").write_text(
        f"Accuracy: {accuracy:.4f}\nROC-AUC (OvR): {roc_auc:.4f}\n\n{report}"
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{args.run_name} — clean test set")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    n_params = model.count_params()
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    avg_inference_s = measure_inference_time(model_path, paths)
    print(f"\nParams: {n_params:,}  Model file size: {model_size_mb:.1f} MB (pre-quantization, dev machine)")
    print(f"Avg inference time: {avg_inference_s*1000:.1f} ms/image (dev CPU, NOT the Phase 3 mobile benchmark)")

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", accuracy])
        writer.writerow(["roc_auc_ovr", roc_auc])
        writer.writerow(["n_params", n_params])
        writer.writerow(["model_size_mb_prequant", model_size_mb])
        writer.writerow(["avg_inference_s_devcpu", avg_inference_s])

    if args.skip_stress_test:
        print("Skipping Lighting Stress-Test (--skip-stress-test).")
    elif args.stress_test_manifest.exists():
        print("\n=== Lighting Stress-Test set ===")
        with args.stress_test_manifest.open(newline="") as f:
            rows = list(csv.DictReader(f))
        conditions = sorted({r["condition"] for r in rows})

        with (run_dir / "lighting_stress_test_metrics.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "accuracy", "n_images"])
            writer.writerow(["clean_test_baseline", accuracy, len(paths)])
            for condition in conditions:
                cond_rows = [r for r in rows if r["condition"] == condition]
                cond_paths = [r["path"] for r in cond_rows]
                cond_labels = [CLASS_NAMES.index(r["class"]) for r in cond_rows]
                cond_acc, _, _, _, _, _ = evaluate_split(model, cond_paths, cond_labels)
                print(f"{condition}: accuracy={cond_acc:.4f} (n={len(cond_rows)})")
                writer.writerow([condition, cond_acc, len(cond_rows)])
    else:
        print(f"\n{args.stress_test_manifest} not found — skipping Lighting Stress-Test evaluation. "
              f"Run make_lighting_stress_test.py first.")

    print(f"\nWrote results to {run_dir}")


if __name__ == "__main__":
    main()
