"""Check whether the model relies on the background, by editing the input images.

The trained model (no retraining) is run on three versions of every image:
  original         the image as it is
  mosquito_only    background replaced by one solid color (keep_mosquito)
  background_only  mosquito erased and filled in with the surrounding background (keep_background)

How to read the results:
  - mosquito_only stays high      -> the model does not need the background
  - background_only near chance   -> the background carries no species information for the model
    (accuracy ~0.5 and AUC ~0.5). AUC is checked as well because a model shown images with no
    mosquito may put everything into one class, which hides any ranking in the scores.
  - The mask leaves small leg tips and a smooth filled silhouette in background_only, so its
    numbers are an upper bound on what the background alone gives.

How the mask is made (bright background, dark mosquito):
  1. Estimate the slowly varying background brightness with a large blur.
  2. Pixels clearly darker than their local background locate the mosquito (largest piece).
  3. The mask then grows into fainter pixels touching it (wings, pale legs), and gets a margin.

Writes into <run dir>/background_test_<split>/:
  mask_check.jpg   masks and edited images for some examples, to check the segmentation
  predictions.csv  one row per image per condition
  metrics.txt      accuracy, per-class recall, AUC, confusion matrix per condition
  scores.png       distribution of the albopictus score per condition and true class

Usage:  python background_test.py runs/<run>/best.pt [--split val]
"""
import argparse
import csv
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from dataset import IMAGE_DIR, SPLITS_CSV, eval_transform
from train import load_checkpoint

# Darkness = how much darker a pixel is than its local background (0-255).
# Tuned by looking at mask_check.jpg. For the background-only test, missing part of the
# mosquito is worse than removing extra background, so these settings lean towards a larger mask.
STRONG_THRESHOLD = 30  # clearly mosquito (body, dark leg segments): used to find the mosquito
WEAK_THRESHOLD = 12    # faint parts (wings, pale legs): kept only if connected to the strong part
BRIDGE_RADIUS = 4      # pixels; joins strong pieces of one mosquito
GAP_RADIUS = 6         # pixels; lets the growth jump over white leg bands
MARGIN_RADIUS = 8      # pixels; extra safety border around the final mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # image filters run faster on a GPU


def to_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None].to(DEVICE)


def dilate(mask, radius):
    """Grow a boolean mask by `radius` pixels (a max filter over a square window)."""
    grown = F.max_pool2d(to_tensor(mask), 2 * radius + 1, stride=1, padding=radius)
    return grown[0, 0].cpu().numpy() > 0


def largest_component(mask):
    """Keep only the largest group of connected True pixels (breadth-first search)."""
    height, width = mask.shape
    seen = np.zeros_like(mask)
    best = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue, component = deque([start]), []
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(mask)
    if best:
        ys, xs = zip(*best)
        result[list(ys), list(xs)] = True
    return result


def grow_within(seed, allowed):
    """Grow `seed` one pixel at a time, but only into `allowed` pixels, until it stops changing."""
    current, allowed = to_tensor(seed), to_tensor(allowed)
    while True:
        grown = F.max_pool2d(current, 3, stride=1, padding=1) * allowed
        if torch.equal(grown, current):
            return current[0, 0].cpu().numpy() > 0
        current = grown


def mosquito_mask(image):
    """Return a boolean array (height, width) that is True on the mosquito.

    Two thresholds ("hysteresis"): the clearly dark pixels locate the mosquito, then the mask
    grows into faint pixels touching it. Faint debris elsewhere in the frame is not touched.
    """
    gray = image.convert("L")
    background = np.asarray(gray.filter(ImageFilter.GaussianBlur(40)), dtype=np.float32)
    darkness = background - np.asarray(gray, dtype=np.float32)
    seed = largest_component(dilate(darkness > STRONG_THRESHOLD, BRIDGE_RADIUS))
    allowed = dilate(darkness > WEAK_THRESHOLD, GAP_RADIUS) | seed
    return dilate(grow_within(seed, allowed), MARGIN_RADIUS)


# One fixed fill color for every image (about the median background color of the dataset).
# A per-image color would leak information: background color alone separates the species a little.
FILL_COLOR = (99, 126, 212)


def keep_mosquito(image, mask):
    """Replace everything outside the mask with the same solid color."""
    pixels = np.asarray(image).copy()
    pixels[~mask] = FILL_COLOR
    return Image.fromarray(pixels)


def keep_background(image, mask, blur_radius=25):
    """Erase the mosquito, filling the hole with a blur of the pixels around it."""
    pixels = np.asarray(image, dtype=np.float32)
    outside = (~mask).astype(np.float32)

    def blur(channel):
        # Three box blurs in a row look almost like a Gaussian blur. Each box blur is done as a
        # horizontal pass then a vertical pass, which is much faster than one square window.
        size = 2 * blur_radius + 1
        tensor = to_tensor(channel)
        for _ in range(3):
            tensor = F.avg_pool2d(tensor, (1, size), stride=1, padding=(0, blur_radius),
                                  count_include_pad=False)
            tensor = F.avg_pool2d(tensor, (size, 1), stride=1, padding=(blur_radius, 0),
                                  count_include_pad=False)
        return tensor[0, 0].cpu().numpy()

    # Blur only the background pixels, then divide by how much background each spot received
    weight = np.maximum(blur(outside), 1e-3)
    filled = pixels.copy()
    for c in range(3):
        smooth = blur(pixels[..., c] * outside) / weight
        filled[..., c][mask] = smooth[mask]
    return Image.fromarray(filled.clip(0, 255).astype(np.uint8))


def save_preview(rows, masks, path):
    """One figure row per image: original, mask in red, mosquito only, background only."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, len(rows) * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax_row, (r, area) in zip(axes, rows):
        image = Image.open(IMAGE_DIR / r["path"]).convert("RGB")
        mask = masks[r["path"]]
        overlay = np.asarray(image).copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        ax_row[0].imshow(image)
        ax_row[0].set_title(f"{Path(r['path']).name} | mask {area:.1%}", fontsize=9)
        ax_row[1].imshow(overlay)
        ax_row[2].imshow(keep_mosquito(image, mask))
        ax_row[3].imshow(keep_background(image, mask))
    for ax, title in zip(axes[0], ["", "mask", "A: mosquito only", "B: background only"]):
        if title:
            ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


CONDITIONS = {
    "original": lambda image, mask: image,
    "mosquito_only": keep_mosquito,
    "background_only": keep_background,
}


def predict_conditions(rows, masks, model, transform, classes, batch_size=32):
    """Run the model on every image in every condition. Returns one dict per (condition, image)."""
    results = []
    for condition, edit in CONDITIONS.items():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            images = [edit(Image.open(IMAGE_DIR / r["path"]).convert("RGB"), masks[r["path"]]) for r in chunk]
            batch = torch.stack([transform(image) for image in images]).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1).cpu()
            for r, p in zip(chunk, probs):
                results.append({"condition": condition, "path": r["path"], "specimen_id": r["specimen_id"],
                                "true": r["species"], "pred": classes[int(p.argmax())],
                                **{f"score_{c}": float(p[i]) for i, c in enumerate(classes)}})
    return results


def auc(results, classes):
    """Chance that a random image of classes[1] gets a higher classes[1] score than a random
    image of classes[0]. 0.5 = the scores don't separate the classes at all, 1.0 = perfectly."""
    key = f"score_{classes[1]}"
    positives = [r[key] for r in results if r["true"] == classes[1]]
    negatives = [r[key] for r in results if r["true"] == classes[0]]
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def summarize(results, classes):
    lines = []
    for condition in CONDITIONS:
        sel = [r for r in results if r["condition"] == condition]
        accuracy = sum(r["true"] == r["pred"] for r in sel) / len(sel)
        lines += [f"== {condition} ==",
                  f"accuracy {accuracy:.3f} | AUC {auc(sel, classes):.3f}"]
        for c in classes:
            of_class = [r for r in sel if r["true"] == c]
            counts = [sum(r["pred"] == p for r in of_class) for p in classes]
            lines.append(f"  true {c:<11} recall {counts[classes.index(c)] / len(of_class):.3f} | predicted as "
                         + ", ".join(f"{p} {n}" for p, n in zip(classes, counts)))
        lines.append("")
    return lines


def save_score_plot(results, classes, path):
    """Histogram of the classes[1] score, split by true class, one panel per condition."""
    key = f"score_{classes[1]}"
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 3.5), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, condition in zip(axes, CONDITIONS):
        for c in classes:
            scores = [r[key] for r in results if r["condition"] == condition and r["true"] == c]
            ax.hist(scores, bins=bins, alpha=0.6, label=f"true {c}")
        ax.set_title(condition)
        ax.set_xlabel(f"{classes[1]} score")
    axes[0].set_ylabel("images")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4, help="random images to preview per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(SPLITS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    out_dir = args.checkpoint.parent / f"background_test_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Segment every image and save a preview to check the masks
    masks, areas = {}, {}
    for r in rows:
        masks[r["path"]] = mosquito_mask(Image.open(IMAGE_DIR / r["path"]).convert("RGB"))
        areas[r["path"]] = float(masks[r["path"]].mean())
    values = np.array(list(areas.values()))
    print(f"{args.split}: {len(rows)} images | mask area of frame: min {values.min():.1%}, "
          f"median {np.median(values):.1%}, max {values.max():.1%}")

    # Random examples of each species, plus the most extreme masks, where failures show up first
    rng = random.Random(args.seed)
    picked = []
    for species in ["aegypti", "albopictus"]:
        picked += rng.sample([r for r in rows if r["species"] == species], args.n)
    by_area = sorted(rows, key=lambda r: areas[r["path"]])
    picked += by_area[:2] + by_area[-2:]
    save_preview([(r, areas[r["path"]]) for r in picked], masks, out_dir / "mask_check.jpg")

    # 2. Run the trained model on the original and edited images
    model, ckpt = load_checkpoint(args.checkpoint, DEVICE)
    classes = ckpt["classes"]
    transform = eval_transform(ckpt["input_size"], ckpt["mean"], ckpt["std"])
    results = predict_conditions(rows, masks, model, transform, classes)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = "\n".join([f"checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})",
                        f"split: {args.split} | {len(rows)} images", "",
                        *summarize(results, classes)])
    (out_dir / "metrics.txt").write_text(report + "\n")
    print(report)
    save_score_plot(results, classes, out_dir / "scores.png")
    print(f"results saved in {out_dir}")


if __name__ == "__main__":
    main()
