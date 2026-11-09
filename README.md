# JHUSPH27 — Improved Mosquito Marking and Trap Device

**Sponsor:** Dr. Derek Cummings, JHU Bloomberg School of Public Health (Epidemiology) / WSE (BME)
**Program:** JHU Mechanical Engineering Senior Design, 2026–2027
**Confidentiality:** NDA required; IP owned by sponsor.

> This document is the working foundation for the project. Section 4 (CV requirements) is a
> **draft for the Week 6 System Requirements Review (SRR)**. Values marked *Proposed* are team
> proposals; values marked *TBD* need sponsor input.

---

## 1. Project Overview

### 1.1 Problem
Vector-borne diseases (dengue, chikungunya, Zika) are hard to control in part because we cannot
estimate **absolute** mosquito population sizes, only imperfect relative numbers. Dr. Cummings'
team is running a *Wolbachia* intervention trial in Belo Horizonte, Brazil, and needs better
measurements of:
- absolute abundance,
- mobility / dispersal (re-colonization from surrounding areas),
- age structure (older mosquitoes are more likely to transmit pathogens).

Mark-release-recapture (MRR) is the standard ecological method for these measurements.

### 1.2 Goal
Design a field-deployable trap that runs the MRR loop automatically:

```
Attract ──► Capture ──► Image & Identify (CV) ──► Mark (fluorescent powder) ──► Release
                              │
                              └──► Log data (count, species, mark color) ──► Export
```

The starting point is the Cummings lab's existing prototype (3D-printed body, optics,
Raspberry Pi, AI-based analysis, external 12 V + 6 V DC power).

---

## 2. Sponsor Requirements (from project brief)

| ID    | Requirement (paraphrased)                                                              | CV relevance |
|-------|-----------------------------------------------------------------------------------------|--------------|
| SR-1  | Attract mosquitoes; capture rate ≥ current device (reared-and-released, enclosed test)  | Indirect (counting is used to measure it) |
| SR-2  | Consistent fluorescent marking without harm; marking index ↑, injury rate not ↑          | **Direct**: mark detection / scoring |
| SR-3  | Release mosquitoes safely after marking                                                  | Indirect |
| SR-4  | **Count mosquitoes with 90% accuracy**                                                   | **Direct** |
| SR-5  | **Speciate mosquitoes with 70% accuracy**                                                | **Direct** |
| SR-6  | Collect and store data, easy extraction (stretch: Wi-Fi)                                  | **Direct** |
| SR-7  | 7-day deployment without human intervention                                              | Power / storage budget |
| SR-8  | One-person portable, ground platform                                                      | Size / mass of optics + compute |
| SR-9  | Onboard power supply (stretch: rechargeable solar)                                       | Compute power budget |
| SR-10 | Rain, 40–120 °F (4–49 °C), direct equatorial sun, debris resistant                        | Optics fogging, thermal limits of compute |

---

## 3. CV Software Scope

This section covers the **software** side of CV (data, models, inference, logging). Camera,
optics, lighting, power, and enclosure design belong to the hardware team; CV only states the
input assumptions it depends on (Section 4.3).

**In scope**
- Detecting and counting each mosquito that passes through the imaging stage
- Species classification (2 target species, see 3.1)
- Fluorescent mark color classification (6 colors + unmarked, see 3.1)
- Automated marking-coverage score to support SR-2 (stretch)
- On-device data logging and export

**Out of scope**
- Sex classification (not reliably labelable by the team at this time; revisit if the sponsor requires it)
- Camera / lens / illumination hardware selection
- Attractant design, powder delivery, release mechanism, power hardware

### 3.1 Label Schema

| Field        | Values | Notes | Labeling phase |
|--------------|--------|-------|----------------|
| `bbox`       | box around the whole mosquito | For detection / counting | Phase 1 |
| `mark_color` | `orange`, `yellow`, `green`, `blue`, `violet`, `magenta`, `unmarked` | 7 classes | Phase 1 |
| `species`    | `aedes_aegypti` (yellow fever mosquito), `aedes_albopictus` (Asian tiger mosquito), `other` | `other` = non-target mosquito or non-mosquito object | Phase 2 (deferred) |

**Phase 1 (current): color detection.** Label `bbox` and `mark_color` only. Species is not
labeled in this round because the team cannot yet identify species reliably by eye.

**Phase 2: species.** Label `species` once a reliable ground-truth source is available (see below).

### 3.2 Ground-Truth Source

Wherever possible, labels should come from the **experiment setup, not from visual judgment**:

- **`mark_color`:** mark each batch with one known powder color and photograph it separately,
  so every image in that batch inherits the color label. Visual labeling is used only for
  field images where the batch is unknown.
- **`species`:** use lab-reared colonies kept separately by species, so species is known
  before imaging. Field images without known species need review by the sponsor's lab.

This avoids training the model on the labeler's own confusion (e.g. `violet` vs. `magenta`).

### 3.3 Known Visual Challenges (from sample images)

- **Powder hides species features.** The main cue separating the two species is the thorax
  pattern (*Ae. aegypti*: lyre-shaped white markings; *Ae. albopictus*: single white median
  stripe). Heavy powder (e.g. blue, violet) can cover it. Recaptured mosquitoes are always
  marked, so the species model **must be trained and tested on marked mosquitoes too**.
- **Similar colors.** `violet` vs. `magenta`, and `blue` vs. `unmarked` (both dark), are the
  likely confusion pairs.
- **Sparse marking.** Some mosquitoes carry only a little powder (e.g. the yellow sample), which
  risks being classified as `unmarked`.

---

## 4. Derived CV Requirements (DRAFT for SRR)

### 4.1 Definitions
- **Event:** one mosquito passing through the imaging stage.
- **Count accuracy:** `1 − |N_pred − N_true| / N_true`, computed per collection interval
  (*Proposed*: 24 h), where `N_true` comes from known release numbers or manual counts.
- **Species accuracy:** top-1 accuracy on a held-out test set imaged **in the trap**
  (not lab-microscope images), reported with a full confusion matrix.

### 4.2 Requirements

| ID    | Requirement | Target | Verification | Traces to | Status |
|-------|-------------|--------|--------------|-----------|--------|
| CV-01 | Count mosquitoes per interval | ≥ 90% count accuracy | Enclosed release tests with known N, ≥ *TBD* trials, densities up to *TBD*/h | SR-4 | Sponsor value; definition *Proposed* |
| CV-02 | Do not double-count one individual within an event | ≤ *TBD*% duplicates | Video review of test runs | SR-4 | *Proposed* |
| CV-03 | Classify species (`aedes_aegypti`, `aedes_albopictus`, `other`) | ≥ 70% overall top-1 accuracy | Held-out in-trap test set | SR-5 | Sponsor value |
| CV-04 | Minimum per-class performance (so a model can't reach 70% by predicting the majority class) | Recall ≥ *TBD* (e.g. 60%) for each target species | Confusion matrix | SR-5 | *Proposed* |
| CV-05 | Species accuracy holds on marked mosquitoes | CV-03 target met separately on the `unmarked` subset and the marked subset | Split test set by `mark_color` | SR-5 | *Proposed* |
| CV-06 | Detect fluorescent mark (marked vs. unmarked) | Sensitivity ≥ *TBD* (e.g. 95%), specificity ≥ *TBD* (e.g. 98%) | Known-marked / unmarked batches under trap illumination | SR-2 | *Proposed* |
| CV-07 | Classify mark color (7 classes incl. `unmarked`) | Overall ≥ *TBD* (e.g. 90%); report `violet`/`magenta` and `blue`/`unmarked` confusion explicitly | Batches marked with each color | SR-2 | *Proposed* |
| CV-08 | Automated marking-coverage index (per body region) | Agreement with manual scoring: *TBD* (e.g. Cohen's κ ≥ 0.7) | Compare against human scorers on the same images | SR-2 | Stretch |
| CV-09 | Low-confidence handling | Predictions below a confidence threshold are tagged "uncertain" and their images saved for human review | Inspect logs | SR-5, SR-6 | *Proposed* |
| CV-10 | On-device latency | Each event classified before release, ≤ *TBD* s on the onboard computer | Bench timing | SR-3 | *Proposed* |
| CV-11 | Per-event data record | timestamp, trap ID, image crop, species + confidence, mark color + confidence, count | Inspect logs | SR-6 | *Proposed* |
| CV-12 | Storage and export | 7 days of records + crops on local storage; export via USB / SD (stretch: Wi-Fi) | 7-day bench run | SR-6, SR-7 | *Proposed* |

### 4.3 Input Assumptions (CV → hardware team)

The models only work if deployment images look like training images. CV does not choose the
camera, but requires the following:

| ID    | Assumption | Why it matters for the software |
|-------|------------|---------------------------------|
| IA-01 | Training / test data is captured with the **same camera, lens, lighting, and background** as the deployed trap | Pretrained models (ImageNet, lab-microscope datasets) lose accuracy under domain shift; we fine-tune on in-trap images |
| IA-02 | Resolution sufficient to resolve thorax markings: ≥ *TBD* px/mm at the subject | The species decision depends on the thorax pattern |
| IA-03 | Fixed illumination and **locked white balance / exposure** (no auto modes) | Color classification depends on stable color rendering |
| IA-04 | Ambient sunlight does not reach the imaging stage | Same as IA-03 |
| IA-05 | Event-triggered capture (e.g. IR beam-break) is available | Saves power over the 7-day deployment (SR-7, SR-9); defines one "event" for counting |
| IA-06 | Compute stays below thermal throttling at 49 °C ambient | Latency target CV-10 |

---

## 5. Open Questions for Sponsor

1. Expected ratio of *Ae. aegypti* to *Ae. albopictus* in field traps (class imbalance)? How often do non-target insects enter?
2. Sex classification is currently out of scope. Does the sponsor need it? If so, can the lab provide sex-labeled samples?
3. What does each of the 6 marking colors encode (release batch, date, site)?
4. How is "count accuracy" scored today, and at what mosquito densities?
5. What do the current device's camera, model, and accuracy look like? Is there an existing labeled dataset?
6. Is the image taken **before** marking (speciation), **after** marking (mark verification), or both?
7. Is *Wolbachia* status, age (e.g. wing wear), or blood-fed status of interest? These are mostly not visually detectable and should be explicitly out of scope unless requested.

---

## 6. Revision History

| Date       | Change                           | Author |
|------------|----------------------------------|--------|
| 2026-10-07 | Initial draft from project brief | Team   |
| 2026-10-08 | Fixed species (2) and mark colors (6 + unmarked); added label schema; moved hardware items to input assumptions | Team |
| 2026-10-08 | Removed sex classification from scope and label schema; renumbered CV-06 to CV-12 | Team |
| 2026-10-08 | Added labeling phases (Phase 1: color; Phase 2: species, deferred) and ground-truth source | Team |
