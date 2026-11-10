# Species classification results (Ong 2021 public dataset)

Selected outputs copied from `runs/` (which is not tracked by git). Model checkpoints (`best.pt`,
~94 MB each) are **not** included; they stay on the training laptop.

All numbers describe the **public dataset only** (Raspberry Pi NoIR images of dead, unmarked
mosquitoes on a fixed background). They say nothing yet about images from our device.

Images in these folders are derived from Ong et al. 2021 (Sci Rep 11:9908), Kaggle
`pradeepisawasan/aedes-mosquitos`, licensed CC BY-NC-SA 4.0.

## Runs

Both runs: ResNet-50 (ImageNet weights), input 512×384, 5 epochs head-only + 15 epochs fine-tuning,
checkpoint chosen by lowest val loss. Split by specimen (see `data/splits.csv`). The only difference
between the two runs is the training images.

| Folder | Training images | Best epoch |
|---|---|---|
| `20261109_181015_baseline_full_image/` | full images (`data/processed/ong2021_800`) | 11 |
| `20261109_203751_mosquito_only/` | background replaced by one solid color (`data/processed/ong2021_800_masked`, made by `scripts/make_masked_images.py`) | 18 |

## Files in each folder

| File | Made by | Content |
|---|---|---|
| `history.csv` | `train.py` | loss and accuracy per epoch |
| `eval_<split>/` | `evaluate.py` | metrics, confusion matrix, misclassified images, per-image predictions |
| `gradcam_<split>_<layer>/` | `gradcam.py` | Grad-CAM heatmaps for misclassified and sampled correct images |
| `background_test_val/` | `background_test.py` | the model run on edited val images (see below) |

## Background test (val, 180 images)

Each val image is edited and given to the trained model without retraining. Accuracy alone can
hide information when a model puts everything into one class, so AUC (0.5 = no separation) is
reported too.

| Condition | What is left in the image | Baseline acc / AUC | Mosquito-only model acc / AUC |
|---|---|---|---|
| original | full image | 1.000 / 1.000 | 0.944 / 0.999 |
| mosquito_only | mosquito + ~8 px border of real background | 0.644 / 0.948 | 0.972 / 0.998 |
| background_only | background, mosquito erased | 0.939 / 1.000 | 0.789 / 0.953 |
| halo_only | only the ~8 px border (pure background) | 0.600 / 0.736 | 0.544 / 0.858 |
| halo_shape_only | the border's shape in one flat color | 0.561 / 0.615 | 0.517 / 0.614 |
| mosquito_no_halo | mosquito without the outer border | 0.550 / 0.924 | 0.928 / 0.990 |

## Findings so far

1. **The baseline separates the species mostly by background.** With the mosquito erased it still
   reaches AUC 1.000; with the background removed, albopictus recall drops to ~0.30. Grad-CAM heatmaps
   lie mostly on the background. Its test accuracy (0.978) is therefore **not** a measure of species
   recognition. Splitting by specimen does not prevent this, because the background difference is
   shared by all images of a species (likely photographed in separate sessions).
2. **Training on mosquito-only images moves the model onto the mosquito.** Val accuracy 0.983
   (specimen level 0.978; all 3 errors from one specimen, aegypti_139). Grad-CAM heatmaps of all
   16 sampled correct images lie on the mosquito.
3. **Some leak remains through the border of real background kept around the mosquito.** The
   border alone gives AUC 0.858, against 0.614 for its shape alone, so its texture carries species
   information. Removing it lowers accuracy from 0.972 to 0.928 (albopictus recall 1.000 → 0.902),
   part of which may just be the unfamiliar sharper edges.
4. The test split has **not** been used for the mosquito-only model yet; it will be used once, after
   the remaining leak is dealt with.

Lesson for collecting our own data: photograph both species in the same sessions on the same
background, alternating, or a model can learn the session instead of the species.
