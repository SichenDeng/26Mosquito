"""Make smaller copies of the raw images (3280x2464, ~4 MB each) so training doesn't have to
decode huge JPEGs every epoch. Keeps the 4:3 aspect ratio. Raw images are left untouched.

data/raw/ong2021/aegypti_0/aegypti1b.jpg -> data/processed/ong2021_800/aegypti_0/aegypti1b.jpg

Usage:  python resize_images.py
"""
import csv
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "ong2021"
OUT_DIR = ROOT / "data" / "processed" / "ong2021_800"
SPLITS_CSV = ROOT / "data" / "splits.csv"
SIZE = (800, 600)  # (width, height), same 4:3 ratio as the raw images


def main():
    with open(SPLITS_CSV) as f:
        paths = [row["path"] for row in csv.DictReader(f)]  # only images that are in the split

    for i, rel in enumerate(paths, 1):
        dest = OUT_DIR / rel
        if dest.exists():
            continue
        with Image.open(RAW_DIR / rel) as im:
            im.draft("RGB", (SIZE[0] * 2, SIZE[1] * 2))  # lets PIL decode the JPEG at reduced size: much faster
            im = im.convert("RGB").resize(SIZE, Image.BICUBIC)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=95)
        if i % 200 == 0 or i == len(paths):
            print(f"{i}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
