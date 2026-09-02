"""Crop YOLO-annotated fruit bunches out of bunch-pile frames into per-class patches.

Source: "Annotated Video Dataset of Oil Palm Fruit Bunch Piles for Ripeness Grading"
(Suharjito & Junior, ScienceDB, DOI 10.57760/sciencedb.j00001.00713). Each frame has a
same-named .txt file with one YOLO row per box: "class x_center y_center width height"
(all normalized 0-1, relative to image dims).

Class-id-to-name mapping is NOT assumed to be the paper's stated 0-5 order — Roboflow
re-exports commonly reorder classes. Instead it's read from whatever labels file ships
with the export (`_darknet.labels`, `classes.txt`, or `data.yaml`'s `names:` list) and
each label text is matched (Malay + English keywords) to this project's 4-class taxonomy.
Labels that don't match a known ripeness keyword (e.g. "Janjang kosong" / empty bunch,
"TBS abnormal") are treated as out of scope and skipped, same as classes 4-5 in the
original paper's taxonomy.

If the export is a Roboflow drop with train/valid/test folders, the same source frame
can appear more than once across those folders (Roboflow's own split leaks frames the
same way the FYP1 Roboflow dataset did). Detected via the `_<ext>.rf.<hash>` filename
suffix Roboflow adds: when present, the split folder and hash are stripped so all copies
of one frame collapse to the same `source_photo` key — this project's own
merge_and_split.py then keeps every copy in a single split, regardless of which Roboflow
folder it happened to land in.

Usage:
    python src/data/crop_bunch_piles.py [--src data/raw/bunch_piles] [--dst data/processed]
"""
import argparse
import csv
import re
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_FILENAMES = ("_darknet.labels", "classes.txt", "data.yaml")
ROBOFLOW_SUFFIX_RE = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-fA-F]+$", re.IGNORECASE)
KNOWN_SPLIT_DIRS = {"train", "valid", "test"}

# Ordered so more specific Malay phrases are checked before the generic "masak" (ripe)
# substring they contain, e.g. "kurang masak" (underripe) and "terlalu masak" (overripe).
LABEL_KEYWORDS = [
    ("underripe", ("kurang", "underripe")),
    ("overripe", ("terlalu", "overripe")),
    ("unripe", ("mentah", "unripe")),
    ("ripe", ("masak", "ripe")),
]


def normalize_label(text: str) -> str | None:
    t = text.strip().lower()
    for class_name, keywords in LABEL_KEYWORDS:
        if any(kw in t for kw in keywords):
            return class_name
    return None  # out of scope: empty bunch, abnormal FFB, or unrecognized label


def find_label_file(src: Path) -> Path | None:
    for name in LABEL_FILENAMES:
        matches = list(src.rglob(name))
        if matches:
            return matches[0]
    return None


def load_class_map(label_path: Path) -> dict[int, str]:
    if label_path.name == "data.yaml":
        lines = label_path.read_text().splitlines()
        in_names = False
        names = []
        for line in lines:
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                if line.strip().startswith("-"):
                    names.append(line.strip().lstrip("-").strip().strip("'\""))
                else:
                    break
    else:
        names = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]

    class_map = {}
    print(f"Class mapping (from {label_path}):")
    for class_id, label_text in enumerate(names):
        mapped = normalize_label(label_text)
        class_map[class_id] = mapped
        print(f"  {class_id}: {label_text!r} -> {mapped or 'OUT OF SCOPE'}")
    return class_map


def parse_yolo_annotation(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
        boxes.append((cls_id, x, y, w, h))
    return boxes


def crop_box(image: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image | None:
    img_w, img_h = image.size
    left = max(0, int((x - w / 2) * img_w))
    top = max(0, int((y - h / 2) * img_h))
    right = min(img_w, int((x + w / 2) * img_w))
    bottom = min(img_h, int((y + h / 2) * img_h))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def compute_source_photo(image_path: Path, src_root: Path) -> str:
    rel = image_path.relative_to(src_root)
    match = ROBOFLOW_SUFFIX_RE.search(rel.stem)
    if not match:
        return rel.with_suffix("").as_posix()
    stem = rel.stem[: match.start()]
    dir_parts = tuple(p for p in rel.parts[:-1] if p.lower() not in KNOWN_SPLIT_DIRS)
    return "/".join((*dir_parts, stem))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/bunch_piles"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(
            f"{args.src} does not exist. Download the ScienceDB dataset "
            f"(DOI 10.57760/sciencedb.j00001.00713) and extract it there first."
        )

    label_path = find_label_file(args.src)
    if label_path is None:
        raise SystemExit(
            f"No {LABEL_FILENAMES} found under {args.src}. Refusing to guess the "
            f"class-id-to-name mapping — a wrong guess mislabels the whole dataset silently."
        )
    class_map = load_class_map(label_path)

    crops_dir = args.dst / "crops"
    manifest_path = args.dst / "manifest_bunch_piles.csv"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    image_paths = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    n_skipped_no_ann, n_boxes_out_of_scope, n_boxes_degenerate = 0, 0, 0

    for i, image_path in enumerate(image_paths, 1):
        ann_path = image_path.with_suffix(".txt")
        if not ann_path.exists():
            n_skipped_no_ann += 1
            continue

        source_photo = compute_source_photo(image_path, args.src)
        image = Image.open(image_path).convert("RGB")

        for box_idx, (cls_id, x, y, w, h) in enumerate(parse_yolo_annotation(ann_path)):
            class_name = class_map.get(cls_id)
            if class_name is None:
                n_boxes_out_of_scope += 1
                continue
            patch = crop_box(image, x, y, w, h)
            if patch is None:
                n_boxes_degenerate += 1
                continue

            class_dir = crops_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            safe_stem = source_photo.replace("/", "_")
            patch_name = f"bunch_piles__{safe_stem}__{image_path.stem}__box{box_idx}.jpg"
            patch_path = class_dir / patch_name
            patch.save(patch_path, quality=95)

            rows.append(
                {
                    "path": patch_path.as_posix(),
                    "class": class_name,
                    "source_dataset": "bunch_piles",
                    "source_photo": source_photo,
                }
            )

        if i % 500 == 0:
            print(f"  processed {i}/{len(image_paths)} frames, {len(rows)} patches so far")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "source_dataset", "source_photo"])
        writer.writeheader()
        writer.writerows(rows)

    n_unique_photos = len({row["source_photo"] for row in rows})
    print(f"Wrote {len(rows)} patches from {n_unique_photos} unique source photos ({len(image_paths)} files) to {manifest_path}")
    print(
        f"Skipped: {n_skipped_no_ann} frames with no annotation file, "
        f"{n_boxes_out_of_scope} boxes out of scope, "
        f"{n_boxes_degenerate} degenerate boxes"
    )


if __name__ == "__main__":
    main()
