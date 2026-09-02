"""tf.data pipeline over data/processed/manifest_full.csv.

Train split gets on-the-fly augmentation (augmentation.py, batch-time, never pre-baked);
valid/test stay clean. Images are resized to 224x224 (NFR-5: "supports 224x224 input")
and scaled to [-1, 1] to match MobileNetV2/EfficientNet-style preprocessing.

Class order is fixed here as the single source of truth (CLASS_NAMES) — every training
and evaluation script imports it from this module so label indices never drift between
scripts.
"""
import csv
from pathlib import Path

import tensorflow as tf

from augmentation import build_augmentation

CLASS_NAMES = ["unripe", "underripe", "ripe", "overripe"]
IMG_SIZE = 224


def load_manifest_split(manifest_path: Path, split: str) -> tuple[list[str], list[int]]:
    with open(manifest_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise ValueError(f"No rows with split == {split!r} in {manifest_path}")
    paths = [r["path"] for r in rows]
    labels = [CLASS_NAMES.index(r["class"]) for r in rows]
    return paths, labels


def _decode_and_resize(path, label):
    # Stays in [0, 255] float32 here — augmentation (if any) runs in this space too.
    # The final [-1, 1] model-input rescale happens last, in _to_model_input, so it's
    # applied uniformly whether or not augmentation ran. See augmentation.py for why
    # this ordering matters (RandomContrast clips to [0, 255] internally).
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    label = tf.one_hot(label, depth=len(CLASS_NAMES))
    return image, label


def _to_model_input(image, label):
    return (image / 127.5) - 1.0, label


def attach_sample_weights(ds: tf.data.Dataset, class_weights: dict[int, float]) -> tf.data.Dataset:
    """Turn a (image, one_hot_label) dataset into (image, one_hot_label, sample_weight).

    Keras' `model.fit(..., class_weight=...)` does NOT reliably resolve one-hot targets
    back to a class index in this TF/Keras version — verified empirically: it produces a
    different loss than an explicitly-computed per-sample weight array for the same
    data/weights (1.37 vs 1.59 in a controlled smoke test), so it silently doesn't apply
    the intended weighting. This is the reliable alternative: Keras natively supports
    (x, y, sample_weight) triples yielded from a tf.data.Dataset.
    """
    weights_tensor = tf.constant([class_weights[i] for i in range(len(CLASS_NAMES))], dtype=tf.float32)

    def _add_weight(image, label):
        class_idx = tf.argmax(label, axis=-1)
        weight = tf.gather(weights_tensor, class_idx)
        return image, label, weight

    return ds.map(_add_weight, num_parallel_calls=tf.data.AUTOTUNE)


def build_dataset(
    manifest_path: Path,
    split: str,
    batch_size: int = 32,
    augment: bool | None = None,
    seed: int = 42,
    shuffle_buffer: int = 2000,
) -> tf.data.Dataset:
    """augment defaults to True for split == 'train', False otherwise."""
    if augment is None:
        augment = split == "train"

    paths, labels = load_manifest_split(manifest_path, split)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = build_augmentation(seed=seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_to_model_input, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    # Quick manual sanity check: python src/training/dataset.py
    manifest = Path("data/processed/manifest_full.csv")
    for split in ("train", "valid", "test"):
        paths, labels = load_manifest_split(manifest, split)
        print(f"{split}: {len(paths)} images")
    ds = build_dataset(manifest, "train", batch_size=8)
    images, labels = next(iter(ds))
    print(f"batch images shape={images.shape} dtype={images.dtype} range=[{tf.reduce_min(images):.2f}, {tf.reduce_max(images):.2f}]")
    print(f"batch labels shape={labels.shape}, class names={CLASS_NAMES}")
