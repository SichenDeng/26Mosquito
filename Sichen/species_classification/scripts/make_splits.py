"""Split the Ong 2021 images into train/val/test BY SPECIMEN, and write data/splits.csv.

Why by specimen: files like aegypti12a/b/c/d.jpg are 4 photos of the SAME mosquito.
If 12a is in train and 12d in test, the model is tested on a mosquito it has already seen,
and test accuracy looks better than it really is (data leakage). So all photos of one
specimen must go to the same split. The official *_testing folders are ignored for this reason.

Usage:  python make_splits.py
"""
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "ong2021"
OUT_CSV = ROOT / "data" / "splits.csv"
SEED = 42                                   # fixed seed -> same split on Mac and Windows
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# aegypti12b.jpg -> species "aegypti", specimen number "12", view "b"
NAME_RE = re.compile(r"^(aegypti|albopictus)(\d+)([a-d])\.jpg$")


def main():
    # 1. Group every image by specimen: {(species, number): [relative paths]}
    specimens = defaultdict(list)
    for path in sorted(RAW_DIR.glob("*/*.jpg")):
        # The dataset has 3 "<name> - Copy.jpg" files, byte-identical to <name>.jpg (checked with md5)
        if path.name.endswith(" - Copy.jpg"):
            print(f"Skipping duplicate: {path.name}")
            continue
        m = NAME_RE.match(path.name)
        if m is None:
            raise ValueError(f"Unexpected file name: {path}")
        species, number, _view = m.groups()
        # Store paths relative to RAW_DIR with "/" so the CSV also works on Windows
        specimens[(species, number)].append(path.relative_to(RAW_DIR).as_posix())

    # 2. For each species separately (stratified), shuffle specimens and cut into train/val/test
    rng = random.Random(SEED)
    split_of = {}
    for species in ["aegypti", "albopictus"]:
        keys = sorted(k for k in specimens if k[0] == species)
        rng.shuffle(keys)
        n_train = round(len(keys) * FRACTIONS["train"])
        n_val = round(len(keys) * FRACTIONS["val"])
        for i, key in enumerate(keys):
            split_of[key] = "train" if i < n_train else "val" if i < n_train + n_val else "test"

    # 3. Write one row per image
    rows = []
    for (species, number), paths in sorted(specimens.items()):
        for p in paths:
            rows.append({"path": p, "species": species,
                         "specimen_id": f"{species}_{number}", "split": split_of[(species, number)]})
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "species", "specimen_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # 4. Sanity checks and summary
    splits_per_specimen = defaultdict(set)
    for r in rows:
        splits_per_specimen[r["specimen_id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_per_specimen.values()), "a specimen appears in two splits!"

    print(f"Wrote {len(rows)} rows to {OUT_CSV}\n")
    print(f"{'split':<6} {'species':<11} {'specimens':>9} {'images':>7}")
    for split in FRACTIONS:
        for species in ["aegypti", "albopictus"]:
            sel = [r for r in rows if r["split"] == split and r["species"] == species]
            n_spec = len({r["specimen_id"] for r in sel})
            print(f"{split:<6} {species:<11} {n_spec:>9} {len(sel):>7}")
    print("\nPhotos per specimen:", dict(Counter(len(p) for p in specimens.values())))


if __name__ == "__main__":
    main()
