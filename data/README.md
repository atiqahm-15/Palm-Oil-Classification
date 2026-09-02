# Data

Raw datasets are not committed to this repo — `data/raw/` is gitignored. This file tracks
what's being sourced and why.

## Why FYP1's dataset isn't enough on its own

The FYP1 MobileNetV2 baseline trained on the Roboflow "palm-fruit-ripeness-classificationcnn"
set reached only 68.14% top-1 accuracy, well under the 85% target. Root cause: severe
train/valid/test leakage (194 of 205 unique source photos had crops split across multiple
sets), only 205 unique source photos in the whole dataset, and class imbalance (unripe was
most underrepresented, recall 0.38). See `CLAUDE.md` for the full status.

**Rule for any new/merged dataset:** split by source photo (or source tree/bunch), never by
crop. Splitting by crop is what caused the FYP1 leakage bug — it must not repeat.

## Public datasets merged (all 4, as of Aug 2026)

| Dataset | Size | Classes | Notes |
|---|---|---|---|
| Annotated Video Dataset of Oil Palm Fruit Bunch Piles for Ripeness Grading (Suharjito & Junior) | 7,171 images | 6 (incl. empty bunch/abnormal) | CC BY 4.0, Science Data Bank. Best taxonomy match; needs cropping from YOLO boxes for classification use. |
| Outdoor-Tenera-Oil-Palm-Fruit-Image (MunirahRosbi, GitHub) | — | — | Malaysia-collected (Johor/N.Sembilan/Perak), outdoor lighting focus, pre-split train/test. |
| Mendeley "Ordinal Dataset for Ripeness Level Classification" (Kamal et al., 2024) | 4,728 high-res images | 5 | Central Kalimantan, industry-funded, stratified splits. |
| Kaggle "Oil Palm Fruit Ripeness Classification Dataset" (hadeemadadum) | 857 images | 4 | Exact 4-class match to our taxonomy — useful as a small supplement. |

## Merge pipeline (bunch-piles + Malaysia outdoor) — done, Aug 2026

Merged the bunch-piles dataset (cropped to patches) with the Malaysia outdoor dataset.
Both feed the same locked 4-class taxonomy (unripe/underripe/ripe/overripe; see
`PROJECT_PLAN.md` §8); anything outside that scope is dropped rather than remapped.

1. **Bunch-piles — manual download, done.** ScienceDB (scidb.cn) serves this dataset
   through a login-gated JS app with no scriptable download link, so it was downloaded
   by hand (DOI `10.57760/sciencedb.j00001.00713`) and extracted to
   `data/raw/bunch_piles/`. The actual drop turned out to be a **Roboflow re-export**
   (`train/`, `valid/`, `test/` folders; filenames carry a `_<ext>.rf.<hash>` suffix), which
   introduced two gotchas `crop_bunch_piles.py` now handles rather than assumes away:
   - **Class-id order is not the paper's stated 0-5 order.** Roboflow's own
     `_darknet.labels` in this export reads `0 Janjang kosong (empty), 1 Kurang masak
     (underripe), 2 TBS abnormal (abnormal), 3 TBS masak (ripe), 4 TBS mentah (unripe),
     5 Terlalu masak (overripe)` — completely different from the paper's 0=unripe...5=abnormal.
     The script reads this file and text-matches each label (Malay + English keywords)
     rather than hardcoding ids; it refuses to run if no labels file is found.
   - **The same source frame can appear in more than one Roboflow split folder**
     (552 of ~4,160 unique frames did, e.g. `frame1--1-` in both `train/` and `valid/`) —
     the same leakage bug as the original FYP1 dataset, just baked into this export. The
     script strips the split folder and `.rf.<hash>` suffix to recover the true frame
     identity, so `merge_and_split.py` keeps every copy of a frame in one split.

   Run:
   ```
   python src/data/crop_bunch_piles.py
   ```
   Crops each in-scope box into its own patch under `data/processed/crops/<class>/`.
   Result: 12,583 patches from 3,889 unique source frames (1 frame had no annotation file;
   3,857 boxes were out of scope). Output: `data/processed/manifest_bunch_piles.csv`.

2. **Malaysia outdoor — scripted, done.** Downloaded from GitHub
   (`MunirahRosbi/Outdoor-Tenera-Oil-Palm-Fruit-Image`, tag `v1`, CC BY 4.0) into
   `data/raw/malaysia_outdoor/` (466 files; bulk zip download kept getting truncated
   mid-transfer, so it was pulled file-by-file via `raw.githubusercontent.com` with
   per-file retries instead). Images are already single/few-bunch photos, so no cropping
   is needed. Cataloged with:
   ```
   python src/data/catalog_malaysia.py
   ```
   This dataset has **no Underripe class**, and its Damaged/Empty categories are dropped
   (out of the 4-class scope) rather than remapped. It also ships **byte-identical
   duplicate images**, including 2 pairs that cross the repo's own `FFBtrain`/`FFBtest`
   split (e.g. `TestUnripe(13).jpg` == `TrainUnripe(126).jpg`) — leakage baked into the
   source data itself. The script MD5-hashes every image and drops exact-content
   duplicates (11 dropped: 2 cross-split, 9 within the same split), keeping one copy each.
   Result: 428 images. Output: `data/processed/manifest_malaysia.csv`. The repo's own
   `FFBtrain`/`FFBtest` split is ignored regardless; the split below is rebuilt from
   scratch across the merged corpus.

3. **Mendeley Ordinal — scripted, done.** Downloaded via Mendeley's public API
   (`data.mendeley.com/public-api/datasets/424y96m6sw`, which exposes a direct file
   download URL — no login needed), extracted to `data/raw/mendeley_ordinal/`. Images
   are single/few-bunch photos, no cropping needed. Cataloged with:
   ```
   python src/data/catalog_mendeley.py
   ```
   Ships 5 classes (`0Immature`, `1PartiallyRipe`, `2FullyRipe`, `3OverRipe`, `4Decayed`);
   the first 4 map onto unripe/underripe/ripe/overripe, `Decayed` is dropped (out of the
   4-class scope). Its own `Train_val_test_split/*.txt` stratified split is ignored, same
   as every other source. 1 exact-content duplicate found, both copies in the already-out-
   of-scope `Decayed` class — no effect on the merged corpus. Result: 4,317 images. Output:
   `data/processed/manifest_mendeley.csv`.

4. **Kaggle supplement — manual download, done.** Kaggle requires an API token or a
   logged-in browser session with no public direct-download URL, so it was downloaded by
   hand from `hadeemadadum/oil-palm-fruit-ripeness-classification-dataset` (CC BY 4.0)
   into `data/raw/kaggle/`. Exact 4-class match to this project's taxonomy
   (`Over-ripe`/`Ripe`/`Under-ripe`/`Unripe`) — no remapping or exclusions needed.
   Cataloged with:
   ```
   python src/data/catalog_kaggle.py
   ```
   Its own stratified train/valid/test CSVs are ignored, same as every other source.
   6 exact-content duplicate pairs found (12 files, all within the same class — no
   mislabeling risk, but still deduplicated). Result: 851 images. Output:
   `data/processed/manifest_kaggle.csv`.

   Before merging, all four sources' whole-photo images (Malaysia, Mendeley, Kaggle —
   bunch-piles crops don't apply here) were also cross-checked against each other by MD5:
   **zero images shared across datasets.**

5. **Merge + split — done, all 4 sources.**
   ```
   python src/data/merge_and_split.py
   ```
   Groups every row by `(source_dataset, source_photo)` — so every crop from the same
   bunch-pile frame, and every deduplicated photo from any source, stays in one split —
   then greedily assigns whole groups to train/valid/test (70/15/15) to keep counts
   close to target ratios. Output: `data/processed/manifest_full.csv`, 18,167 rows,
   9,473 unique source photos, **zero source-photo leakage verified** (every group's rows
   land in exactly one split):

   | split | overripe | ripe | underripe | unripe | total |
   |---|---:|---:|---:|---:|---:|
   | train | 2,580 | 3,648 | 3,259 | 3,229 | 12,716 |
   | valid | 553 | 784 | 699 | 693 | 2,729 |
   | test | 552 | 780 | 698 | 692 | 2,722 |

   **Bug found and fixed (Aug 2026): targets must be per `(source, class)`, not just
   per class.** The original version balanced group assignment against a global
   per-class target only. Bunch Piles' source photos are large groups (many crops per
   pile frame); the other 3 sources are all single-image groups. Sorting groups by size
   descending and always picking whichever split is furthest below target means large
   groups get funneled into train first, simply because train's raw target number is
   the biggest — nothing to do with the source itself, but the effect looks like it:
   Bunch Piles ended up 82-94% of every class's *train* split but only 14-48% of
   valid/test, so models were being evaluated on a different source mix than they
   trained on. Fixed by computing targets per `(source_dataset, class)` — each source
   now keeps its own ~70/15/15 split regardless of its group sizes (e.g. Bunch Piles is
   63%/63%/63% of underripe across train/valid/test post-fix). This changed the
   downstream model results substantially — see `PROJECT_PLAN.md` §8.

All 4 candidate datasets from `PROJECT_PLAN.md` §4 are now merged — Phase 0's sourcing
milestone and Phase 1's unified-dataset milestone are both satisfied.

## Layout

```
data/
├── README.md            # this file (tracked)
├── raw/                  # gitignored — place downloaded/extracted datasets here
│   ├── bunch_piles/       # manual download from ScienceDB (see above)
│   └── malaysia_outdoor/  # scripted download from GitHub
└── processed/             # gitignored — derived, regenerable from raw/
    ├── crops/              # patches cropped from bunch_piles
    ├── manifest_bunch_piles.csv
    ├── manifest_malaysia.csv
    └── manifest_full.csv   # merged + split, output of merge_and_split.py
```
