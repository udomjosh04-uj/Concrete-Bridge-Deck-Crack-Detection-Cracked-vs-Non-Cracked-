# Concrete-Bridge-Deck-Crack-Detection-Cracked-vs-Non-Cracked-

# Contributors 
## | Udom, Joshua Sunday| 22/EG/CO/1729
## | Obot,Datoiyoabasi Samuel|22/EG/CO/1699
## | Ibokette, Mfoniso N.| 22/EG/CO/1967 
## | Umanah, Ananambiet pius| 22/EG/CO/1709 
## | Akang, Iniabasi Anietie| 22/EG/CO/1809 
## | Ufot Prince-aniekan Emmanuel  | 22/EG/CO/1749 
## | promise aniekanabasi udoekpo | 22/EG/CO/1799
## | Edet, Donald Anthony | 22/EG/CO/1789
## | Adeniyi,Taiwo Okikiola | 22/EG/CO/1719
## | Abraham, Unwana Emmanuel| 22/EG/CO/1669
## | Ushie Divine kipeh| 22/EG/CO/1769
## | Iwok, Emmanuel Aniedi| 22/EG/CO/1739
## | Friday, Success Essien| 22/EG/CO/1819
## | George, Victor Solomon| 22/EG/CO/1639 
## | Etim,Blessing Edo    |22/EG/CO/1759


# Concrete Bridge Deck Crack Detection

Binary image classifier (Cracked vs Non-cracked) for concrete bridge deck surfaces, trained on the
Deck subset of SDNET2018, deployed as a Streamlit app loading the model from Hugging Face Hub.



---

## 1. Dataset

- Source: [SDNET2018 (aniruddhsharma mirror)](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images) — Deck subset only (Pavement and Wall excluded).
- Raw Deck counts (verified by walking the actual folder, not assumed):
  - Non-cracked: 11,595
  - Cracked: 2,025
- Images are patches cropped from a small number of source bridge-deck photographs — **only 66 unique
  source-photo groups** across the entire 13,620-image Deck subset. This is the single most important
  fact about this dataset and drives most of the design decisions below.

## 2. Key design decisions (and the bug that shaped them)

### Class imbalance: 2:1 group-aware undersampling
Non-cracked was undersampled to 2× the Cracked count (~4,110 : 2,025) to reduce imbalance without
fully discarding it. Undersampling was done by **dropping whole source-photo groups**, never individual
images, so no group ends up partially represented.

This is a compromise, not a proven-optimal choice — the notebook also computes `class_weight`s for
training on the full, un-undersampled dataset as an alternative. That comparison has not been run yet;
it should be, before treating the undersampled model as final.

### Group-leakage prevention — and a bug found during real testing
Because patches are crops from the same ~66 source photos, a naive random or per-class train/val/test
split lets patches from the same photo appear in multiple splits, inflating test metrics artificially.

**First implementation split each class's groups independently** (train/val/test computed separately for
Cracked and Non-cracked, then unioned). This silently broke on real data: because the same group ID can
hold both Cracked and Non-cracked patches (same photo, different crops), a group could land in train for
one class and val for the other — a real leak, caught by a hard `assert` that intentionally halts
execution rather than producing a misleadingly good result.

**Fix:** switched to `StratifiedGroupKFold` applied once across the *whole* balanced dataset (both
classes together), so a group is assigned to exactly one split regardless of which class its patches
belong to. Verified leak-free on the real split output.

**Known limitation, unresolved:** 66 groups total means a single 70/15/15 split puts test-set
performance at the mercy of which ~9 source photos happen to land there — this is real sampling
variance, not something the leakage fix solves. A single run's test metrics should not be reported as
a precise number. Recommended: repeat the split with multiple seeds, or run proper k-fold CV, and report
mean ± std.

## 3. Models

Two models are trained and evaluated identically for comparison:

| Model | Purpose |
|---|---|
| Custom CNN (4 conv blocks, GAP, dense head) | Baseline — establishes a floor so the transfer-learning result means something |
| EfficientNetV2-B0 (ImageNet weights, frozen → fine-tuned) | Primary model |

Both take 224×224×3 inputs normalized to [0,1]; `preprocess_input` for EfficientNetV2 is applied
**inside** the model graph itself (not in the data pipeline), so downstream inference code must feed
`[0,1]`-normalized images and must not apply `preprocess_input` a second time.

Augmentation (random flip, rotation, brightness, contrast) is a Keras `Sequential` block inside the
model, active only when `training=True` — a no-op at inference by default.

## 4. Training

- Loss: binary cross-entropy
- Metrics tracked: accuracy, precision, recall, AUC (ROC), AUC (PR)
- Callbacks: `EarlyStopping` (on `val_auc`), `ReduceLROnPlateau`, `ModelCheckpoint` saving the best model
- EfficientNet training is two-phase: frozen-base warmup, then full fine-tune at a lower learning rate

## 5. Evaluation

Reported on the held-out test split: accuracy, balanced accuracy, precision/recall/F1 (with particular
attention to **recall on the Cracked class**, since a missed crack is a safety-relevant false negative
and matters more than a false positive here), ROC curve/AUC, precision-recall curve/AP, confusion
matrix, and Grad-CAM visualizations to sanity-check the model is attending to crack regions rather than
incidental texture or lighting artifacts.

### Results

```
=== Custom CNN ===
              precision    recall  f1-score   support

 Non-cracked   |    0.51   |   1.00   |   0.67    |  226
     Cracked   |    0.00   |   0.00    |  0.00    |  221

    accuracy                           0.51       447
   macro avg       0.25      0.50      0.34       447
weighted avg       0.26      0.51      0.34       447

```

```
=== EfficientNetV2-B0 (fine-tuned) ===
              precision    recall  f1-score   support

 Non-cracked       0.62      0.97      0.76       226
     Cracked       0.94      0.40      0.56       221

    accuracy                           0.69       447
   macro avg       0.78      0.69      0.66       447
weighted avg       0.78      0.69      0.66       447

```
- Trained EfficientNetV2-B0 model (~71MB) hosted on Hugging Face Hub: `Abasiofon001/Concrete_Crack_Screening`.
- Inference served through a Streamlit app (`app.py`) that downloads the model via `hf_hub_download` and
  caches it with `st.cache_resource` so it's fetched once per session, not on every rerun.
- **Filename note:** the uploaded model file is named `best_model .keras` (literal space before the
  extension) — this was a typo carried over from the original save/upload and the app currently matches
  it. Recommended fix: rename the file on the Hugging Face repo to `best_model.keras` and update
  `HF_FILENAME` in `app.py` — a space in a hosted artifact's filename is fragile and worth cleaning up
  once, rather than working around indefinitely.
- The decision threshold for flagging "Cracked" is exposed as a UI slider, **not hardcoded at 0.5** —
  the actual production default should come from whatever threshold on the validation PR curve
  maximizes recall on the Cracked class, not an unexamined default.

## 7. Repo contents

| File | Purpose |
|---|---|
| `concrete_deck_crack_detection.ipynb` | Full pipeline: data loading, group-aware undersampling, leakage-checked split, model definitions, training, evaluation, Grad-CAM |
| `app.py` | Streamlit inference app, loads model from Hugging Face Hub |
| `requirements.txt` | Streamlit app dependencies |
| `debug_hf_repo.py` | Utility to list actual files/repo_type in a Hugging Face repo when a download 404s |

## 8. How to run

**Notebook:**
1. Set up Kaggle credentials (`kaggle.json`) or manually download the dataset and point `DATASET_ROOT` at it.
2. Run cells top to bottom. Section 3 prints raw counts and a filename→group-ID sample — verify these
   match expectations before trusting anything downstream.
3. Section 5's leakage assertion must pass silently. If it doesn't, do not proceed to training.

**Streamlit app:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Set `HF_REPO_ID` in `app.py` to your Hugging Face repo before running.

