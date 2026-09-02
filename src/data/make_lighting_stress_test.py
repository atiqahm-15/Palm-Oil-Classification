"""Generate the deterministic Lighting Stress-Test set from the clean test split.

Per SYSTEM_ARCHITECTURE.md's Data Layer: fixed brightness/shadow transforms applied to
the clean test images *once*, versioned as its own artifact under
data/processed/lighting_stress_test/ — used for robustness evaluation (the "stable across
daylight/shadow/low-light" NFR), never randomly re-generated. This is a fresh baseline for
the new merged dataset, not a reproduction of FYP1's old +/-30%-brightness-only notebook
test — it adds a shadow condition to actually cover all 3 lighting conditions named in the
NFR, at the cost of not being numerically comparable to the old FYP1 lighting numbers.

Three conditions, each applied to every image in the clean test split:
  - darkened:   PIL brightness x0.7  (~30% dimmer, simulates low-light)
  - brightened: PIL brightness x1.3  (~30% brighter, simulates harsh daylight/overexposure)
  - shadow:     a soft-edged elliptical patch darkened to 50% brightness, position/size
                seeded deterministically per image (hash of its relative source path,
                combined with --seed) so the effect varies image-to-image like a real
                cast shadow would, but is exactly reproducible on every run.

Usage:
    python src/data/make_lighting_stress_test.py [--manifest data/processed/manifest_full.csv]
        [--dst data/processed] [--seed 42]
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

CONDITIONS = ("darkened", "brightened", "shadow")
DARKEN_FACTOR = 0.7
BRIGHTEN_FACTOR = 1.3
SHADOW_STRENGTH = 0.5  # brightness multiplier inside the shadow core
SHADOW_MIN_FRAC, SHADOW_MAX_FRAC = 0.4, 0.7  # shadow ellipse size, as a fraction of image dims


def seeded_rng(relative_path: str, seed: int) -> random.Random:
    digest = hashlib.md5(f"{seed}:{relative_path}".encode()).hexdigest()
    return random.Random(int(digest, 16))


def apply_darkened(image: Image.Image, rng: random.Random) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(DARKEN_FACTOR)


def apply_brightened(image: Image.Image, rng: random.Random) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(BRIGHTEN_FACTOR)


def apply_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    w, h = image.size
    ew = rng.uniform(SHADOW_MIN_FRAC, SHADOW_MAX_FRAC) * w
    eh = rng.uniform(SHADOW_MIN_FRAC, SHADOW_MAX_FRAC) * h
    cx = rng.uniform(0.0, 1.0) * w
    cy = rng.uniform(0.0, 1.0) * h

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2], fill=255)
    feather_radius = max(w, h) * 0.08
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0  # 0..1, 1 = full shadow
    brightness_mult = 1.0 - mask_arr[..., None] * (1.0 - SHADOW_STRENGTH)

    img_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    out_arr = np.clip(img_arr * brightness_mult, 0, 255).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")


TRANSFORMS = {"darkened": apply_darkened, "brightened": apply_brightened, "shadow": apply_shadow}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"{args.manifest} not found. Run merge_and_split.py first.")

    with args.manifest.open(newline="") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if not test_rows:
        raise SystemExit(f"No rows with split == 'test' in {args.manifest}.")

    out_root = args.dst / "lighting_stress_test"
    out_rows = []

    for i, row in enumerate(test_rows, 1):
        src_path = Path(row["path"])
        rel_id = f"{row['source_dataset']}/{row['source_photo']}"
        rng = seeded_rng(rel_id, args.seed)
        image = Image.open(src_path).convert("RGB")

        for condition in CONDITIONS:
            transformed = TRANSFORMS[condition](image, rng)
            class_dir = out_root / condition / row["class"]
            class_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{row['source_dataset']}__{row['source_photo'].replace('/', '_')}__{src_path.stem}.jpg"
            out_path = class_dir / out_name
            transformed.save(out_path, quality=95)

            out_rows.append(
                {
                    "path": out_path.as_posix(),
                    "class": row["class"],
                    "source_dataset": row["source_dataset"],
                    "source_photo": row["source_photo"],
                    "condition": condition,
                    "orig_path": row["path"],
                }
            )

        if i % 500 == 0:
            print(f"  processed {i}/{len(test_rows)} test images")

    manifest_out = args.dst / "manifest_lighting_stress_test.csv"
    with manifest_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "class", "source_dataset", "source_photo", "condition", "orig_path"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} stress-test images ({len(test_rows)} test images x {len(CONDITIONS)} conditions)")
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
