"""Predict the species for one image, several images, or every image in a folder.

Images are prepared exactly as during training (size, normalization read from the checkpoint).
Input should be a photo of ONE mosquito, ideally 4:3 like the training photos
(other aspect ratios get stretched to 4:3).

The scores only compare aegypti vs albopictus: the model has never seen other species or
non-mosquito objects, so a high score does NOT mean "this is surely one of the two species".

Usage:
  python predict.py runs/<run>/best.pt some_image.jpg
  python predict.py runs/<run>/best.pt some_folder/ --out predictions.csv
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from dataset import eval_transform
from train import get_device, load_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(inputs):
    paths = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("--out", type=Path, help="optional CSV file to save the results")
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = ckpt["classes"]
    transform = eval_transform(ckpt["input_size"], ckpt["mean"], ckpt["std"])

    results = []
    with torch.no_grad():
        for path in collect_images(args.inputs):
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0)  # add batch dimension
            probs = torch.softmax(model(image.to(device)), dim=1)[0].cpu()
            pred = int(probs.argmax())
            results.append({"path": str(path), "pred": classes[pred], "score": float(probs[pred]),
                            **{f"score_{c}": float(probs[i]) for i, c in enumerate(classes)}})
            print(f"{path.name:<40} {classes[pred]:<11} {float(probs[pred]):.3f}")

    if args.out and results:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"saved {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
