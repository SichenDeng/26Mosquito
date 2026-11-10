# Weekly Summary — Species Classification (week of 2026-11-10)

**Owner:** Sichen (imaging + CV) · **Task:** *Ae. aegypti* vs *Ae. albopictus*, one mosquito per image
**Data:** Ong et al. 2021 public dataset (Raspberry Pi NoIR camera, dead unmarked females, fixed
background), 1,200 images / 301 mosquitoes, split by mosquito into train / val / test (70/15/15).

> All numbers below are on the **public dataset**, not on images from our device (none exist yet).
> Val and test each have only 45 mosquitoes (180 images): one image ≈ 0.6 percentage points.

## 1. Summary

- A first model looked excellent (**97.8%** test accuracy) but was mostly **recognizing the
  background**, not the mosquito.
- After cutting the mosquito out of each image and training only on that, the final model reaches
  **97.2%** on val (44 of 45 mosquitoes correct). When it is shown only background, it is near chance,
  and its attention maps sit on the mosquito.
- **Lesson for our own data collection:** both species must be photographed in the same sessions on
  the same background. Otherwise a model learns the photo session instead of the species.

## 2. What happened

| Step | Result |
|---|---|
| Baseline: ResNet-50 on full images | Test accuracy 97.8% (176/180) |
| Check where the model looks (Grad-CAM heatmaps) | Heat mostly on the **background**, especially for *albopictus* |
| Test: erase the mosquito, keep only the background | Still **93.9%** accuracy, AUC 1.00, so the background alone identifies the species |
| Test: keep only the mosquito, remove the background | Accuracy drops to 64%; most *albopictus* are called *aegypti* |
| Fix 1: train on cut-out mosquitoes (8 px of background kept around them) | Val 98.3%; heatmaps move onto the mosquito, but background-only AUC still 0.95 |
| Find the remaining leak | The 8 px strip of real background around the mosquito still carried species information (AUC 0.86) |
| Fix 2: keep only the mosquito + 2 px | Val **97.2%**; background-only AUC **0.58** (≈ chance). Leak removed |

Why the split by mosquito did not catch this: the background difference is shared by *all* images
of a species (likely photographed in separate batches), so it is present in train, val and test alike.

## 3. Final model so far

| | Baseline (full image) | Final (mosquito only, 2 px) |
|---|---|---|
| Val accuracy, image / mosquito level | 100% / 100% | 97.2% / 97.8% |
| Background only (mosquito erased): accuracy / AUC | 93.9% / 1.00 | 46.7% / 0.58 |
| Grad-CAM heat on the mosquito (16 sampled correct images) | 5 / 16 | 16 / 16 |

*AUC: how well the model's scores rank the two species. 0.5 = no information, 1.0 = perfect.*

The final model gets 5 val images wrong: 2 from one hard specimen, plus 1 image each from three other
specimens. Errors go both ways.

## 4. Limitations

- **Not tested on our device.** The device images will differ in camera setup, background, live
  mosquitoes and fluorescent powder. SR-5 (≥70% species accuracy) must be verified on device images.
- **The model needs a segmentation step.** It was trained on cut-out mosquitoes and fails on raw
  full images (59%). Our current cut-out method assumes a bright, even background.
- **Small evaluation sets.** Differences of 1–2 points are noise. The test set is held back and will
  be used **once**, on the final model.
- **Two classes only.** The model always picks one of the two species; it cannot say "other".

## 5. Next steps

1. Run the final model once on the held-out test set.
2. External check on a different public dataset (mosquitoes on human skin) to see how well it holds up
   on new backgrounds.
3. Controlled experiments, one change at a time: crop and enlarge the mosquito, color correction,
   higher input resolution.

## 6. Input needed from the team

- **Data collection protocol:** photograph both species alternately in the same session, with the
  same background and lighting.
- **Imaging stage background:** a plain, bright, uniform background behind the mosquito makes the
  cut-out step reliable. This also supports IA-01 / IA-03 in the requirements document.

## 7. Figures for slides

Paths are relative to this folder (`Sichen/species_classification/results/`).

| What it shows | File |
|---|---|
| Baseline heatmaps on the background | `20261109_181015_baseline_full_image/gradcam_test_layer4/correct_albopictus.jpg` |
| Original / cut-out / background-only images side by side | `20261109_181015_baseline_full_image/background_test_val/mask_check.jpg` |
| Baseline scores: background-only still separates the species | `20261109_181015_baseline_full_image/background_test_val/scores.png` |
| Final model heatmaps on the mosquito | `20261110_160414_mosquito_only_tight/gradcam_val_layer4/correct_aegypti.jpg` (and `correct_albopictus.jpg`) |
| Final model scores: background-only is mixed | `20261110_160414_mosquito_only_tight/background_test_val/scores.png` |
| Final model confusion matrix (val) | `20261110_160414_mosquito_only_tight/eval_val/confusion_matrix.png` |
| Final model errors | `20261110_160414_mosquito_only_tight/eval_val/misclassified.jpg` |

Full numbers for all three models: `README.md` in this folder.

Data: Ong et al. 2021, Sci Rep 11:9908 (Kaggle `pradeepisawasan/aedes-mosquitos`, CC BY-NC-SA 4.0).
