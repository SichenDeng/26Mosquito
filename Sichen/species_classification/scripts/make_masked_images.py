"""Make a copy of the dataset where only the mosquito is left and the background is one solid color.

Why: the baseline model learned to tell the species apart from the background (see
background_test.py). Training on these images forces it to use the mosquito itself.
Uses the same segmentation as background_test.py.

data/processed/ong2021_800/aegypti_0/aegypti1b.jpg -> data/processed/ong2021_800_masked_tight/aegypti_0/aegypti1b.jpg

Output folders:
  ong2021_800_masked        older version (8 px background border), used by run 20261109_203751
  ong2021_800_masked_tight  current version (mosquito pixels + 2 px), see MARGIN_RADIUS

Usage:  python scripts/make_masked_images.py
"""
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so the modules in the project folder can be imported
from background_test import keep_mosquito, mosquito_mask  # noqa: E402

IN_DIR = ROOT / "data" / "processed" / "ong2021_800"
OUT_DIR = ROOT / "data" / "processed" / "ong2021_800_masked_tight"
SPLITS_CSV = ROOT / "data" / "splits.csv"


def main():
    with open(SPLITS_CSV) as f:
        paths = [row["path"] for row in csv.DictReader(f)]

    for i, rel in enumerate(paths, 1):
        dest = OUT_DIR / rel
        if not dest.exists():
            image = Image.open(IN_DIR / rel).convert("RGB")
            dest.parent.mkdir(parents=True, exist_ok=True)
            keep_mosquito(image, mosquito_mask(image)).save(dest, quality=95)
        if i % 200 == 0 or i == len(paths):
            print(f"{i}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
