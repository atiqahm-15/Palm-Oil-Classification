"""Catalog the Mendeley Ordinal oil palm ripeness dataset into the shared manifest format.

Source: "An Ordinal Dataset for Ripeness Level Classification in Oil Palm Fruit Quality
Grading" (Kamal, Mochammad, Nurahmi, Natawijaya & Kalili, 2024), Mendeley Data,
DOI 10.17632/424y96m6sw.1, CC BY 4.0. Images are already single/few-bunch photos (not
piles), so no cropping is needed — each image is its own source photo. The dataset ships
its own stratified Training/Validation/Testing split (`Train_val_test_split/*.txt`); that
is ignored here, same as every other source — splitting is rebuilt from scratch across
the merged corpus in merge_and_split.py.

5 classes ship in `Images/`: 0Immature, 1PartiallyRipe, 2FullyRipe, 3OverRipe, 4Decayed.
Only the first 4 map onto this project's taxonomy (unripe/underripe/ripe/overripe).
Decayed is outside the 4-class scope (see PROJECT_PLAN.md SS8) and is dropped rather
than remapped.

Like every other source merged so far, this one was checked for exact-content
duplicates (MD5) before trusting it — found exactly 1 duplicate pair, both in the
out-of-scope Decayed class, so it has no effect on the merged corpus. The same
dedup logic is applied anyway for consistency and in case future re-downloads differ.

Usage:
    python src/data/catalog_mendeley.py [--src data/raw/mendeley_ordinal] [--dst data/processed]
"""
import argparse
import csv
import hashlib
from pathlib import Path

CLASS_MAP = {
    "immature": "unripe",
    "partiallyripe": "underripe",
    "fullyripe": "ripe",
    "overripe": "overripe",
    "decayed": None,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_to_class(folder_name: str) -> str | None:
    name = folder_name.lower().lstrip("0123456789")
    return CLASS_MAP.get(name)


def find_images_root(src: Path) -> Path:
    if (src / "Images").is_dir():
        return src / "Images"
    matches = list(src.rglob("Images"))
    if not matches:
        raise SystemExit(f"No 'Images' folder found anywhere under {src}.")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/mendeley_ordinal"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"{args.src} does not exist. Extract the dataset there first.")

    images_root = find_images_root(args.src)
    manifest_path = args.dst / "manifest_mendeley.csv"
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
                "source_dataset": "mendeley_ordinal",
                "source_photo": rel.with_suffix("").as_posix(),
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "source_dataset", "source_photo"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} images to {manifest_path} ({n_dropped} dropped: Decayed/out of scope)")
    print(f"Deduplicated {n_dupes} exact-content duplicates")


if __name__ == "__main__":
    main()
