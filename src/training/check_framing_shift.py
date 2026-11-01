"""How much accuracy does the app's camera framing cost, before anyone goes to a plantation?

The problem this measures. Every image the model was trained and tested on is a tight crop around
a single bunch. The Android app feeds it the *whole* previewed frame squashed to 224x224, as
`app/src/main/assets/model_contract.json` specifies, and the framing guide drawn on the camera
screen covers only about a quarter to a third of that frame. A user who obeys the hint therefore
hands the model something the model has never seen: a smaller bunch surrounded by background.

There are two distinct effects and this script measures them together, because the app applies
them together:

  1. **Scale and context.** The bunch occupies a fraction of the frame rather than all of it, so
     the model sees it at lower effective resolution with unfamiliar surroundings.
  2. **Aspect distortion.** The capture is bound to the preview, which is full-bleed on a tall
     phone, so the frame is roughly 9:20. Squashing that to a square stretches the bunch
     horizontally by more than 2x. A squarer viewfinder box would not.

The parity fixtures cannot reveal any of this, because they are already crops. So this re-scores
the real test split through simulated framings and reports what each one costs.

What it does NOT settle. The background here is synthetic — a blurred continuation of the crop
itself, or flat grey as a control — not real plantation background. The *scale* and *aspect* shifts
are faithful; the background is an approximation, and the spread between the two background modes
is the honest measure of how much that approximation matters. Field photographs remain the
confirmation step. This is the cheap experiment that says whether they are worth taking, and which
design option they should be taken to compare.

Reading the output. `baseline` re-scores the unmodified crops and must reproduce the run's known
test accuracy; if it does not, distrust everything else in the file. Each other condition is an
(aspect, area fraction) pair. The app as built today sits at roughly aspect 9:20, fraction 0.25 on
a modern phone and 0.35 on the 360x800 reference.

Usage (from the repo root):
    python src/training/check_framing_shift.py
        [--model app/src/main/assets/model.tflite]
        [--aspects 9:20,3:4,1:1] [--fractions 1.0,0.6,0.35,0.25]
        [--background edge|gray] [--decoder pil|tf]
        [--limit-per-class 0] [--out-dir results/framing_shift]

Full test split x 3 aspects x 4 fractions is ~33k inferences, on the order of fifteen minutes on a
laptop CPU. Use --limit-per-class for a quick look first.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from dataset import CLASS_NAMES, IMG_SIZE, _to_model_input, load_manifest_split

DEFAULT_MODEL = Path("app/src/main/assets/model.tflite")

# Flat grey for the control background: mid-grey is the conventional letterbox fill and carries
# no texture the model could read as foliage or ground.
GRAY = (114, 114, 114)


def parse_aspect(text: str) -> tuple[int, int]:
    """'9:20' -> (9, 20), width first. Portrait phone framings have width < height."""
    width, _, height = text.partition(":")
    return int(width), int(height)


def frame_crop(
    crop: Image.Image,
    aspect: tuple[int, int],
    fraction: float,
    background: str,
) -> tuple[Image.Image, float]:
    """Place [crop] into a canvas of the given aspect so it covers [fraction] of the canvas area.

    The crop is pasted at its native size and never resampled here, so the only interpolation in
    the whole pipeline is the final resize to 224 — the same one the training pipeline applies.
    When the requested fraction is too large for the canvas aspect to contain the crop, the canvas
    shrinks to the smallest one that fits and the achieved fraction is returned instead.
    """
    w, h = crop.size
    ratio = aspect[0] / aspect[1]

    # Canvas area = crop area / fraction, at the requested aspect.
    canvas_h = int(round(((w * h) / (fraction * ratio)) ** 0.5))
    canvas_w = int(round(ratio * canvas_h))

    # The crop must fit. If it does not, this fraction is unreachable at this aspect.
    canvas_w = max(canvas_w, w)
    canvas_h = max(canvas_h, h)
    achieved = (w * h) / (canvas_w * canvas_h)

    if background == "gray":
        canvas = Image.new("RGB", (canvas_w, canvas_h), GRAY)
    else:
        # A blurred, stretched copy of the crop: colour-matched and textureless, which is closer to
        # the out-of-focus ground and foliage around a real bunch than flat grey is.
        canvas = crop.resize((canvas_w, canvas_h), Image.BILINEAR).filter(
            ImageFilter.GaussianBlur(radius=max(2, canvas_w // 25))
        )
    canvas.paste(crop, ((canvas_w - w) // 2, (canvas_h - h) // 2))
    return canvas, achieved


def load_rgb(path: str, decoder: str) -> Image.Image:
    """Decode one JPEG.

    Which decoder is used is a real experimental variable, not plumbing: the Android app cannot
    call tf.io.decode_jpeg, and two libjpeg configurations disagree by a few levels per pixel.
    Pillow stands in for the app; TensorFlow is what every accuracy figure in the report was
    measured with. Running the same condition both ways isolates that cost from everything else.
    """
    if decoder == "tf":
        rgb = tf.io.decode_jpeg(tf.io.read_file(path), channels=3).numpy()
        return Image.fromarray(rgb, mode="RGB")
    with Image.open(path) as handle:
        return handle.convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    """The training pipeline's preprocessing: bilinear squash to 224, then x / 127.5 - 1."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized = tf.image.resize(tf.convert_to_tensor(rgb), [IMG_SIZE, IMG_SIZE])
    tensor, _ = _to_model_input(resized, None)
    return tensor.numpy().astype(np.float32)


class Model:
    def __init__(self, path: Path, num_threads: int):
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        if list(self.inp["shape"]) != [1, IMG_SIZE, IMG_SIZE, 3]:
            raise SystemExit(f"unexpected model input {self.inp['shape']}")

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.inp["index"], tensor[None, ...])
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out["index"])[0].copy()


def score(
    model: Model,
    paths: list[str],
    labels: list[int],
    aspect: tuple[int, int] | None,
    fraction: float | None,
    background: str,
    decoder: str,
) -> dict:
    """Run every image through one framing condition and summarise the outcome."""
    n_classes = len(CLASS_NAMES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    top_probs: list[float] = []
    margins: list[float] = []
    achieved: list[float] = []

    for path, label in zip(paths, labels):
        crop = load_rgb(path, decoder)
        if aspect is None:
            framed, got = crop, 1.0
        else:
            framed, got = frame_crop(crop, aspect, fraction, background)
        probs = model(preprocess(framed))

        order = np.argsort(probs)[::-1]
        confusion[label, int(order[0])] += 1
        top_probs.append(float(probs[order[0]]))
        margins.append(float(probs[order[0]] - probs[order[1]]))
        achieved.append(got)

    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    support = confusion.sum(axis=1)
    return {
        "accuracy": correct / total,
        "images": total,
        "achieved_area_fraction": float(np.mean(achieved)),
        "mean_top_probability": float(np.mean(top_probs)),
        "mean_margin": float(np.mean(margins)),
        "per_class_recall": {
            name: (float(confusion[k, k] / support[k]) if support[k] else None)
            for k, name in enumerate(CLASS_NAMES)
        },
        "predicted_distribution": {
            name: int(confusion[:, k].sum()) for k, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist(),
    }


def stratified_subset(
    paths: list[str], labels: list[int], per_class: int, seed: int
) -> tuple[list[str], list[int]]:
    if per_class <= 0:
        return paths, labels
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for k in range(len(CLASS_NAMES)):
        members = [i for i, lab in enumerate(labels) if lab == k]
        take = min(per_class, len(members))
        keep += sorted(rng.choice(members, size=take, replace=False).tolist())
    return [paths[i] for i in keep], [labels[i] for i in keep]


def write_figure(results: list[dict], baseline_accuracy: float, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_aspect: dict[str, list[dict]] = {}
    for row in results:
        by_aspect.setdefault(row["aspect"], []).append(row)

    for aspect, rows in by_aspect.items():
        rows = sorted(rows, key=lambda r: r["achieved_area_fraction"])
        ax.plot(
            [r["achieved_area_fraction"] for r in rows],
            [100 * r["accuracy"] for r in rows],
            marker="o",
            label=f"capture aspect {aspect}",
        )

    ax.axhline(
        100 * baseline_accuracy,
        linestyle="--",
        linewidth=1,
        color="grey",
        label="unmodified crops (baseline)",
    )
    # Where the app as built actually sits.
    ax.axvspan(0.24, 0.36, alpha=0.12, color="tab:red")
    ax.annotate(
        "app as built",
        xy=(0.30, ax.get_ylim()[1]),
        xytext=(0.30, ax.get_ylim()[1] - 3),
        ha="center",
        fontsize=9,
        color="tab:red",
    )

    # Plotted against the fraction actually achieved, not the one requested: a 9:20 frame cannot
    # hold a roughly square bunch at more than about 0.64 of its area, and hiding that behind the
    # requested value would make the tall frame look better than it is.
    ax.set_xlabel("Fraction of the frame the bunch occupies (achieved)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy against camera framing")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="the .tflite to score; defaults to the one bundled in the app")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_full.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--aspects", default="9:20,3:4,1:1",
                        help="capture aspects as width:height, comma separated")
    parser.add_argument("--fractions", default="1.0,0.6,0.35,0.25",
                        help="fractions of the frame area the bunch should occupy")
    parser.add_argument("--background", choices=["edge", "gray"], default="edge")
    parser.add_argument("--decoder", choices=["pil", "tf"], default="pil",
                        help="pil matches what the Android app does; tf matches how every "
                             "accuracy figure in the report was measured")
    parser.add_argument("--limit-per-class", type=int, default=0,
                        help="subsample for a quick look; 0 uses the whole split")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("results/framing_shift"))
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output directory")
    parser.add_argument("--redraw", type=Path, default=None,
                        help="redraw the figure from an existing framing_shift.json and exit")
    args = parser.parse_args()

    if args.redraw is not None:
        payload = json.loads(args.redraw.read_text(encoding="utf-8"))
        out = write_figure(
            payload["conditions"], payload["baseline"]["accuracy"],
            args.redraw.with_name("framing_shift.png"),
        )
        print(f"Wrote {out}" if out else "matplotlib unavailable")
        return

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"{args.out_dir} already has results in it. Results are never overwritten — point "
            f"--out-dir somewhere new, or pass --force if you mean to discard them."
        )
    if not args.model.exists():
        raise SystemExit(f"{args.model} not found — run src/training/make_app_bundle.py first.")

    aspects = [parse_aspect(a) for a in args.aspects.split(",")]
    fractions = [float(f) for f in args.fractions.split(",")]

    paths, labels = load_manifest_split(args.manifest, args.split)
    paths, labels = stratified_subset(paths, labels, args.limit_per_class, args.seed)
    conditions = 1 + len(aspects) * len(fractions)
    print(f"{len(paths)} images x {conditions} conditions = "
          f"{len(paths) * conditions} inferences")

    model = Model(args.model, args.num_threads)

    print("scoring baseline (unmodified crops)...", flush=True)
    baseline = score(model, paths, labels, None, None, args.background, args.decoder)
    print(f"  baseline accuracy {baseline['accuracy']:.4f}")

    results = []
    for aspect in aspects:
        for fraction in fractions:
            name = f"{aspect[0]}:{aspect[1]} @ {fraction:g}"
            print(f"scoring {name}...", flush=True)
            summary = score(model, paths, labels, aspect, fraction, args.background,
                            args.decoder)
            summary.update(
                aspect=f"{aspect[0]}:{aspect[1]}",
                requested_area_fraction=fraction,
                delta_vs_baseline=summary["accuracy"] - baseline["accuracy"],
            )
            results.append(summary)
            print(f"  accuracy {summary['accuracy']:.4f} "
                  f"({100 * summary['delta_vs_baseline']:+.2f} points), "
                  f"achieved fraction {summary['achieved_area_fraction']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "about": "Test-split accuracy under simulated camera framings. Generated by "
                 "src/training/check_framing_shift.py. The background is synthetic; see the "
                 "module docstring for what this does and does not establish.",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "manifest": str(args.manifest),
        "split": args.split,
        "images": len(paths),
        "class_names": CLASS_NAMES,
        "background": args.background,
        "decoder": args.decoder,
        "num_threads": args.num_threads,
        "limit_per_class": args.limit_per_class,
        "baseline": baseline,
        "conditions": results,
    }
    (args.out_dir / "framing_shift.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    with (args.out_dir / "framing_shift.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("aspect,requested_fraction,achieved_fraction,accuracy,delta_points,"
                     "mean_top_probability,mean_margin,"
                     + ",".join(f"recall_{c}" for c in CLASS_NAMES) + "\n")
        handle.write(
            f"baseline,1.0,1.0,{baseline['accuracy']:.6f},0.0,"
            f"{baseline['mean_top_probability']:.6f},{baseline['mean_margin']:.6f},"
            + ",".join(f"{baseline['per_class_recall'][c]:.6f}" for c in CLASS_NAMES) + "\n"
        )
        for row in results:
            handle.write(
                f"{row['aspect']},{row['requested_area_fraction']:g},"
                f"{row['achieved_area_fraction']:.6f},{row['accuracy']:.6f},"
                f"{100 * row['delta_vs_baseline']:.4f},"
                f"{row['mean_top_probability']:.6f},{row['mean_margin']:.6f},"
                + ",".join(f"{row['per_class_recall'][c]:.6f}" for c in CLASS_NAMES) + "\n"
            )

    figure = write_figure(results, baseline["accuracy"], args.out_dir / "framing_shift.png")

    print(f"\nWrote {args.out_dir / 'framing_shift.json'}")
    print(f"Wrote {args.out_dir / 'framing_shift.csv'}")
    if figure:
        print(f"Wrote {figure}")

    worst = min(results, key=lambda r: r["accuracy"])
    print(f"\nBaseline {100 * baseline['accuracy']:.2f}%. "
          f"Worst framing {worst['aspect']} @ {worst['requested_area_fraction']:g} "
          f"gives {100 * worst['accuracy']:.2f}% "
          f"({100 * worst['delta_vs_baseline']:+.2f} points).")
    print("The predicted-class distribution per condition is in the JSON: a collapse onto one "
          "class means the model is reading the background, not the bunch.")


if __name__ == "__main__":
    main()
