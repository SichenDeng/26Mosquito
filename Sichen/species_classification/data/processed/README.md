# Processed images: `ong2021_800/`

Downscaled copies (800×600 JPEG, quality 95) of the unrotated images from the
**Aedes Mosquitos Dataset** by Pradeep Isawasan, published on Kaggle:
https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos

**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
These derived images are redistributed under the same license, for non-commercial
academic use only.

**Please cite:**
Ong, S.-Q., Ahmad, H., Nair, G., et al. (2021).
Implementation of a deep learning model for automated classification of *Aedes aegypti* (Linnaeus)
and *Aedes albopictus* (Skuse) in real time. *Scientific Reports*, 11, 9908.
https://doi.org/10.1038/s41598-021-89365-3

## How these files were made

1. `scripts/download_ong2021.py` downloads only the `aegypti_0`, `aegypti_0_testing`, `albopictus_0`
   and `albopictus_0_testing` folders. The `_90/_180/_270` folders are rotated copies, so they are skipped.
2. `scripts/make_splits.py` drops 3 byte-identical `"<name> - Copy.jpg"` duplicates and
   writes `../splits.csv`. Train/val/test are split by specimen.
3. `scripts/resize_images.py` resizes each 3280×2464 original to 800×600.

Changes from the original: resized, and duplicates removed. Folder and file names are unchanged.
