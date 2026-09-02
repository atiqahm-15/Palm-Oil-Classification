"""Catalog the Malaysia outdoor oil palm FFB dataset into the shared manifest format.

Source: MunirahRosbi/Outdoor-Tenera-Oil-Palm-Fruit-Image (GitHub tag v1, CC BY 4.0).
Images are already single/few-bunch photos (not piles), so no cropping is needed —
each image is its own source photo. The repo's own FFBtrain/FFBtest split is ignored;
splitting is rebuilt from scratch across the merged corpus in merge_and_split.py.

Only Unripe/Ripe/Overripe are kept. This dataset has no Underripe class, and its
Damaged/Empty categories are outside this project's 4-class scope (see CLAUDE.md) —
both are dropped rather than remapped.

The dataset ships with byte-identical duplicate images, including at least one pair
that crosses the repo's own FFBtrain/FFBtest split (e.g. TestUnripe(13).jpg ==
TrainUnripe(126).jpg) — the same leakage bug this project is trying to avoid, just
baked into the source data instead of introduced by our own splitting. Every image
is hashed (MD5) and exact-content duplicates are dropped, keeping one copy per
distinct photo, so no image can end up double-counted or split across train/valid/test.

Usage:
    python src/data/catalog_malaysia.py [--src data/raw/malaysia_outdoor] [--dst data/processed]
"""
import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image

KEEP_CLASSES = {"unripe", "ripe", "overripe"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_decodable(path: Path) -> bool:
    """MD5 hashing alone doesn't catch truncation — a half-downloaded JPEG hashes fine.
    Found empirically: 23 of this dataset's files looked complete (real byte counts, no
    0-byte files) but were truncated mid-download and fail to actually decode."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_to_class(folder_name: str) -> str | None:
    name = folder_name.lower()
    for prefix in ("train", "test"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name if name in KEEP_CLASSES else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/malaysia_outdoor"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"{args.src} does not exist. Extract the dataset there first.")

    manifest_path = args.dst / "manifest_malaysia.csv"
    args.dst.mkdir(parents=True, exist_ok=True)

    rows = []
    n_dropped = 0
    n_corrupt = 0
    seen_hashes: dict[str, Path] = {}
    n_dupe_same_split, n_dupe_cross_split = 0, 0

    for image_path in sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        class_name = folder_to_class(image_path.parent.name)
        if class_name is None:
            n_dropped += 1
            continue

        if not is_decodable(image_path):
            n_corrupt += 1
            print(f"  DROPPED corrupt/truncated: {image_path.relative_to(args.src).as_posix()}")
            continue

        digest = file_md5(image_path)
        rel = image_path.relative_to(args.src)
        if digest in seen_hashes:
            original = seen_hashes[digest]
            orig_split = original.parts[0]
            dup_split = rel.parts[0]
            if orig_split != dup_split:
                n_dupe_cross_split += 1
                print(f"  DROPPED cross-split duplicate: {rel.as_posix()} == {original.as_posix()}")
            else:
                n_dupe_same_split += 1
            continue
        seen_hashes[digest] = rel

        rows.append(
            {
                "path": image_path.as_posix(),
                "class": class_name,
                "source_dataset": "malaysia_outdoor",
                "source_photo": rel.with_suffix("").as_posix(),
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "source_dataset", "source_photo"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} images to {manifest_path} ({n_dropped} dropped: damaged/empty/other, {n_corrupt} dropped: corrupt/truncated)")
    print(
        f"Deduplicated {n_dupe_same_split + n_dupe_cross_split} exact-content duplicates "
        f"({n_dupe_cross_split} crossed the source repo's own train/test split, "
        f"{n_dupe_same_split} were within the same split)"
    )


if __name__ == "__main__":
    main()
