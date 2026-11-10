"""Show which image regions drive the model's decision, using Grad-CAM.

How Grad-CAM works:
  1. Run the image through the model and keep the activations of a late convolutional layer.
     Each of its channels is a feature map ("this feature is present here").
  2. Backpropagate the score of the predicted class to get the gradient of every channel.
     The average gradient of a channel says how much that feature pushes the score up.
  3. Add up the channels using those averages as weights, and keep only the positive part.
     The result is a coarse heatmap of the regions that support the prediction.

Limits when reading the heatmaps:
  - layer4 output is 1/32 of the input size (16 x 12 cells for 512 x 384), so one cell covers
    32 x 32 input pixels. It can separate mosquito from background, but not fine scale patterns.
    --layer layer3 gives a finer map (32 x 24), usually noisier.
  - Each heatmap is scaled to its own maximum, so colors can't be compared between images.
  - A heatmap shows where the evidence is, not which visual feature the model is using.

Needs eval_<split>/predictions.csv, so run evaluate.py first.
Writes into <run dir>/gradcam_<split>_<layer>/:
  misclassified.jpg, correct_aegypti.jpg, correct_albopictus.jpg

Usage:  python gradcam.py runs/<run>/best.pt [--split test] [--n 8] [--layer layer4]
"""
import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

from dataset import ROOT, eval_transform
from train import get_device, load_checkpoint


def grad_cam(model, layer, image, target_class):
    """Return a heatmap (2D array, values 0-1, same height/width as `image`) for one image."""
    store = {}

    def save_activations(_module, _inputs, output):
        output.retain_grad()               # intermediate tensors drop their gradient by default
        store["activations"] = output

    handle = layer.register_forward_hook(save_activations)
    logits = model(image)
    handle.remove()

    # Use the raw score, not softmax: at a probability near 1.0 the gradients almost vanish.
    model.zero_grad()
    logits[0, target_class].backward()

    activations = store["activations"]                                # (1, channels, h, w)
    weights = activations.grad.mean(dim=(2, 3), keepdim=True)         # one weight per channel
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))    # (1, 1, h, w)
    cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
    cam = cam / (cam.max() + 1e-8)
    return cam[0, 0].detach().cpu().numpy()


def save_grid(rows, path, model, layer, transform, classes, device, input_size, image_dir):
    """Save one original + heatmap pair per image, two pairs per figure row."""
    height, width = input_size
    n_rows = (len(rows) + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in enumerate(rows):
        image = Image.open(image_dir / r["path"]).convert("RGB")
        batch = transform(image).unsqueeze(0).to(device)
        cam = grad_cam(model, layer, batch, classes.index(r["pred"]))
        shown = image.resize((width, height))
        original_ax, heatmap_ax = axes.flat[2 * i], axes.flat[2 * i + 1]
        original_ax.imshow(shown)
        original_ax.set_title(f"{Path(r['path']).name} | true {r['true']}", fontsize=9)
        heatmap_ax.imshow(shown)
        heatmap_ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        heatmap_ax.set_title(f"pred {r['pred']} ({float(r['pred_score']):.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=8, help="correct images to show per class")
    parser.add_argument("--layer", default="layer4", choices=["layer3", "layer4"])
    parser.add_argument("--out", type=Path, help="output folder (default: inside the run folder)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions = args.checkpoint.parent / f"eval_{args.split}" / "predictions.csv"
    if not predictions.exists():
        raise SystemExit(f"{predictions} not found. Run evaluate.py with --split {args.split} first.")
    with open(predictions) as f:
        rows = list(csv.DictReader(f))

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = ckpt["classes"]
    transform = eval_transform(ckpt["input_size"], ckpt["mean"], ckpt["std"])
    layer = getattr(model, args.layer)
    # Same images the model was trained on (older checkpoints: the original images)
    image_dir = ROOT / ckpt.get("image_dir", "data/processed/ong2021_800")

    out_dir = args.out or args.checkpoint.parent / f"gradcam_{args.split}_{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every wrong image, plus a random sample of correct ones per class
    groups = {"misclassified": [r for r in rows if r["true"] != r["pred"]]}
    rng = random.Random(args.seed)
    for c in classes:
        correct = [r for r in rows if r["true"] == c and r["pred"] == c]
        groups[f"correct_{c}"] = rng.sample(correct, min(args.n, len(correct)))

    for name, group in groups.items():
        if group:
            save_grid(group, out_dir / f"{name}.jpg", model, layer, transform, classes, device,
                      ckpt["input_size"], image_dir)
            print(f"{name}: {len(group)} images")
        else:
            print(f"{name}: no images, skipped")
    print(f"results saved in {out_dir}")


if __name__ == "__main__":
    main()
