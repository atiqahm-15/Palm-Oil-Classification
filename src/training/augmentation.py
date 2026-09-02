"""On-the-fly training augmentation — geometric + lighting.

Per SYSTEM_ARCHITECTURE.md's Data Layer: applied only to the training split, at batch
time, never pre-baked into saved files. Validation, test, and the Lighting Stress-Test
set (see make_lighting_stress_test.py) stay untouched so they remain a clean, reproducible
measure of performance.

Recipe carried over from the FYP1 notebook (src/palm_oil_ripeness.ipynb, cell 7): flip,
rotation, zoom, translation, brightness, contrast. Assumes input pixels are in [0, 255]
float32 (decoded, resized, NOT yet rescaled to the model's [-1, 1] input range) — fill
values and RandomBrightness's value_range are set accordingly, and augmentation must run
*before* the final [-1, 1] rescale (see dataset.py). This isn't just a style choice:
`RandomContrast` in this TF/Keras version hardcodes an internal clip to [0, 255]
regardless of value_range, so running it on already-[-1, 1]-scaled images silently
corrupts the pixel range (verified empirically — output range came out as [0, ~1.17]
instead of [-1, 1]). Augmenting in [0, 255] space sidesteps that entirely, since it's
what the layers actually assume internally.
"""
import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation(seed: int = 42) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.10, seed=seed, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.10, seed=seed, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.1, 0.1, seed=seed, fill_mode="constant", fill_value=0.0),
            layers.RandomBrightness(factor=0.2, value_range=(0.0, 255.0), seed=seed),
            layers.RandomContrast(factor=0.2, seed=seed),
        ],
        name="train_augmentation",
    )
