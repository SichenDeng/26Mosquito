"""Evaluate a trained checkpoint on one split (default: test) of the public dataset.

Writes into <run dir>/eval_<split>/:
  predictions.csv       one row per image: true label, predicted label, scores
  metrics.txt           accuracy, per-class precision/recall/F1, confusion matrix, per-specimen accuracy
  confusion_matrix.png
  misclassified.jpg     the wrongly predicted images, to look at what the model gets wrong

NOTE: these numbers describe performance on the PUBLIC dataset (Ong 2021, same camera/background
as training). They do NOT tell how well the model works on our own device's photos.

Usage:  python evaluate.py runs/<run>/best.pt [--split test]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import ROOT, MosquitoDataset, eval_transform
from train import get_device, load_checkpoint


def per_class_metrics(confusion, classes):
    """confusion[t][p] = number of images with true class t predicted as p."""
    lines = []
    for c, name in enumerate(classes):
        tp = confusion[c][c]                                   # true c, predicted c
        fp = sum(confusion[t][c] for t in range(len(classes))) - tp  # predicted c, but wrong
        fn = sum(confusion[c]) - tp                            # true c, but predicted something else
        precision = tp / (tp + fp) if tp + fp else 0.0         # "when it says c, how often is it right?"
        recall = tp / (tp + fn) if tp + fn else 0.0            # "of all real c, how many did it find?"
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(f"{name:<12} precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}  "
                     f"support {sum(confusion[c])}")
    return lines


def save_confusion_plot(confusion, classes, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(confusion, cmap="Blues")
    for t in range(len(classes)):
        for p in range(len(classes)):
            ax.text(p, t, confusion[t][p], ha="center", va="center", fontsize=14)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified(rows, image_dir, path, max_images=16):
    wrong = [r for r in rows if r["true"] != r["pred"]][:max_images]
    if not wrong:
        return 0
    cols = 4
    n_rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 4, n_rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, wrong):
        ax.imshow(Image.open(image_dir / r["path"]))
        ax.set_title(f"{Path(r['path']).name}\ntrue {r['true']} | pred {r['pred']} ({r['pred_score']:.2f})",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return len(wrong)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-dir", help="image folder relative to this script "
                                            "(default: the folder the checkpoint was trained on)")
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = ckpt["classes"]
    # Older checkpoints don't store image_dir: they were trained on the original images
    image_dir = args.image_dir or ckpt.get("image_dir", "data/processed/ong2021_800")
    dataset = MosquitoDataset(args.split, eval_transform(ckpt["input_size"], ckpt["mean"], ckpt["std"]),
                              ROOT / image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run the model on every image. softmax turns the 2 raw scores into "probabilities" summing to 1.
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            all_probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
    probs = torch.cat(all_probs)

    rows = []
    for meta, p in zip(dataset.rows, probs):
        pred = int(p.argmax())
        rows.append({"path": meta["path"], "specimen_id": meta["specimen_id"], "true": meta["species"],
                     "pred": classes[pred], "pred_score": float(p[pred]),
                     **{f"score_{c}": float(p[i]) for i, c in enumerate(classes)}})

    out_dir = args.checkpoint.parent / f"eval_{args.split}"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Image-level metrics
    confusion = [[0] * len(classes) for _ in classes]
    for r in rows:
        confusion[classes.index(r["true"])][classes.index(r["pred"])] += 1
    accuracy = sum(r["true"] == r["pred"] for r in rows) / len(rows)

    # Specimen-level accuracy: average the scores of all photos of one mosquito, then decide.
    # Shows whether errors are spread out or concentrated on a few hard specimens.
    by_specimen = defaultdict(list)
    for r in rows:
        by_specimen[r["specimen_id"]].append(r)
    specimen_correct = 0
    for spec_rows in by_specimen.values():
        mean_scores = [sum(r[f"score_{c}"] for r in spec_rows) / len(spec_rows) for c in classes]
        specimen_correct += classes[mean_scores.index(max(mean_scores))] == spec_rows[0]["true"]

    lines = [f"checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})",
             f"split: {args.split} | {len(rows)} images | {len(by_specimen)} specimens | images: {image_dir}",
             "NOTE: public-dataset performance only, not device performance.", "",
             f"accuracy (image level):    {accuracy:.3f}",
             f"accuracy (specimen level): {specimen_correct / len(by_specimen):.3f}", "",
             *per_class_metrics(confusion, classes), "",
             "confusion matrix (rows = true, cols = predicted):",
             "             " + "  ".join(f"{c:>11}" for c in classes),
             *[f"{c:<12} " + "  ".join(f"{n:>11}" for n in confusion[i]) for i, c in enumerate(classes)]]
    report = "\n".join(lines)
    (out_dir / "metrics.txt").write_text(report + "\n")
    print(report)

    save_confusion_plot(confusion, classes, out_dir / "confusion_matrix.png")
    n_wrong = save_misclassified(rows, dataset.image_dir, out_dir / "misclassified.jpg")
    print(f"\n{n_wrong} misclassified images shown in misclassified.jpg" if n_wrong else "\nno misclassified images")
    print(f"results saved in {out_dir}")


if __name__ == "__main__":
    main()
