"""Catalog the Kaggle oil palm ripeness supplement dataset into the shared manifest format.

Source: "Oil Palm Fruit Ripeness Classification Dataset" (hadeemadadum, Kaggle), CC BY 4.0.
857 images, exact 4-class match to this project's taxonomy (Over-ripe/Ripe/Under-ripe/
Unripe) — no remapping or exclusions needed. Images are already single/few-bunch photos,
so no cropping is needed — each image is its own source photo. The dataset ships its own
stratified train/valid/test CSVs; those are ignored here, same as every other source —
splitting is rebuilt from scratch across the merged corpus in merge_and_split.py.

Like every other source merged so far, this one was checked for exact-content duplicates
(MD5) before trusting it — found 6 duplicate pairs (12 files), all within the same class,
so no mislabeling risk, but still deduplicated so our own re-split can't scatter copies
of the same photo across train/valid/test.

Usage:
    python src/data/catalog_kaggle.py [--src data/raw/kaggle] [--dst data/processed]
"""
import argparse
import csv
import hashlib
from pathlib import Path

CLASS_MAP = {
    "over-ripe": "overripe",
    "ripe": "ripe",
    "under-ripe": "underripe",
    "unripe": "unripe",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_to_class(folder_name: str) -> str | None:
    return CLASS_MAP.get(folder_name.lower())


def find_images_root(src: Path) -> Path:
    if (src / "Images").is_dir():
        return src / "Images"
    matches = list(src.rglob("Images"))
    if not matches:
        raise SystemExit(f"No 'Images' folder found anywhere under {src}.")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/kaggle"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"{args.src} does not exist. Extract the dataset there first.")

    images_root = find_images_root(args.src)
    manifest_path = args.dst / "manifest_kaggle.csv"
    args.dst.mkdir(parents=True, exist_ok=True)

    rows = []
    n_dropped = 0
    seen_hashes: dict[str, Path] = {}
    n_dupes = 0

    for image_path in sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        class_name = folder_to_class(image_path.parent.name)
        if class_name is None:
            n_dropped += 1
            continue

        digest = file_md5(image_path)
        rel = image_path.relative_to(images_root)
        if digest in seen_hashes:
            n_dupes += 1
            print(f"  DROPPED duplicate: {rel.as_posix()} == {seen_hashes[digest].as_posix()}")
            continue
        seen_hashes[digest] = rel

        rows.append(
            {
                "path": image_path.as_posix(),
                "class": class_name,
                "source_dataset": "kaggle",
                "source_photo": rel.with_suffix("").as_posix(),
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "source_dataset", "source_photo"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} images to {manifest_path} ({n_dropped} dropped: unrecognized class folder)")
    print(f"Deduplicated {n_dupes} exact-content duplicates")


if __name__ == "__main__":
    main()
