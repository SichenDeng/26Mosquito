# Species classification results (Ong 2021 public dataset)

Selected outputs copied from `runs/` (which is not tracked by git). Model checkpoints (`best.pt`,
~94 MB each) are **not** included; they stay on the training laptop.

All numbers describe the **public dataset only** (Raspberry Pi NoIR images of dead, unmarked
mosquitoes on a fixed background). They say nothing yet about images from our device.

Images in these folders are derived from Ong et al. 2021 (Sci Rep 11:9908), Kaggle
`pradeepisawasan/aedes-mosquitos`, licensed CC BY-NC-SA 4.0.

## Runs

All runs: ResNet-50 (ImageNet weights), input 512×384, 5 epochs head-only + 15 epochs fine-tuning,
checkpoint chosen by lowest val loss. Split by specimen (see `data/splits.csv`). The only difference
between the runs is the training images.

| Folder | Training images | Best epoch | Val acc (image / specimen) |
|---|---|---|---|
| `20261109_181015_baseline_full_image/` | full images (`data/processed/ong2021_800`) | 11 | 1.000 / 1.000 |
| `20261109_203751_mosquito_only/` | background replaced by one solid color, ~8 px border of real background kept (`ong2021_800_masked`) | 18 | 0.983 / 0.978 |
| `20261110_160414_mosquito_only_tight/` | same, but only the mosquito pixels + 2 px kept (`ong2021_800_masked_tight`, current `scripts/make_masked_images.py`) | 18 | 0.972 / 0.978 |

Val has 180 images / 45 specimens, so one image is ~0.6 percentage points; the two mosquito-only
models are within noise of each other.

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

The masks differ between the first two models and the tight model, so the conditions are run with
the mask each model was trained with (first two: ~8 px border; tight: 2 px border).

| Condition | What is left in the image | Baseline acc / AUC | Mosquito-only (8 px) acc / AUC | Mosquito-only tight (2 px) acc / AUC |
|---|---|---|---|---|
| original | full image | 1.000 / 1.000 | 0.944 / 0.999 | 0.594 / 0.975 |
| mosquito_only | mosquito + border of real background | 0.644 / 0.948 | 0.972 / 0.998 | 0.967 / 0.994 |
| background_only | background, mosquito erased | 0.939 / 1.000 | 0.789 / 0.953 | **0.467 / 0.584** |
| halo_only | only the border | 0.600 / 0.736 | 0.544 / 0.858 | 0.794 / 0.918 * |
| halo_shape_only | the border's shape in one flat color | 0.561 / 0.615 | 0.517 / 0.614 | 0.528 / 0.778 * |
| mosquito_no_halo | mosquito without the border | 0.550 / 0.924 | 0.928 / 0.990 | 0.900 / 0.991 |

\* With a 2 px border hugging the mosquito, the "border" is mostly the mosquito's own edges and
outline, so these two rows no longer measure background; `background_only` is the relevant test.

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
4. **Keeping only the mosquito pixels + 2 px removes the background leak.** With the mosquito erased,
   the tight model is near chance (AUC 0.584, against 0.953 for the 8 px model), while val accuracy
   stays similar (0.972, specimen level 0.978). Grad-CAM heatmaps lie on the mosquito (body or legs).
   5 val errors: aegypti_139 (2 images, also the hardest specimen for the 8 px model), aegypti57d,
   albopictus26c, albopictus47a.
5. **The tight model needs segmented input.** On full images it predicts almost everything as
   aegypti (accuracy 0.594), so any use on device images requires the same segmentation step.
6. The test split has **not** been used for the mosquito-only models yet; it will be used once, on
   the chosen final model.

Lesson for collecting our own data: photograph both species in the same sessions on the same
background, alternating, or a model can learn the session instead of the species.
