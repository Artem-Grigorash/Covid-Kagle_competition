### COVID CT Segmentation — Kaggle-style Pipeline

This repository contains a compact PyTorch pipeline for segmenting COVID-19 CT slices and preparing a Kaggle-ready submission. It includes:
- `train.py` — baseline training on MedSeg data with strong augmentations
- `finetune.py` — fine‑tuning on a combined dataset and staged unfreezing of the encoder
- `predict.py` — inference on test CT slices and CSV export for submission


---

### Key Features & Approaches

- Task: 2‑class semantic segmentation of lung findings (ground‑glass opacities and consolidation).
- Model: `UnetPlusPlus` (segmentation_models_pytorch) with `resnet34` encoder.
- Input channels: 1 (grayscale CT); Output classes: 2.
- Device selection: automatic — Apple Metal (MPS) → CUDA → CPU.
- Data format: `.npy` tensors shaped like `(N, H, W, 1)` for images and `(N, H, W, 2)` for masks.
- CT pre‑processing: Hounsfield Units windowing to [-1000, 400] and per‑slice min‑max normalization to [0, 1].
- Augmentations:
  - In `train.py`: Albumentations (random crop to 512×512, flips, 90° rotations, shift/scale/rotate, brightness/contrast).
  - In `finetune.py`: "Safe" tensor‑only flips and `rot90` (no OpenCV dependency).
- Loss: Dice + BCE hybrid (sigmoid probabilities) — balances overlap quality and per‑pixel classification.
- Optimizer: AdamW with weight decay; LR schedule: CosineAnnealingLR.
- Metric: Dice coefficient computed on sigmoid‑thresholded predictions (default threshold = 0.35).
- Model selection: save best checkpoint by validation Dice.
- Inference: batched per‑slice prediction, thresholding, packing predictions to a flat vector, CSV export `(Id, Predicted)` for Kaggle.

---

### Repository Structure

- `train.py` — Baseline training on MedSeg
  - Loads `covid_data/images_medseg.npy` and `covid_data/masks_medseg.npy`.
  - Uses Albumentations and `ToTensorV2` for heavy spatial and color augmentations.
  - Implements `CovidSegDataset` that ensures correct channel orders: images `[1, H, W]`, masks `[2, H, W]`.
  - Trains `UnetPlusPlus(resnet34, in_channels=1, classes=2)` for 50 epochs (default).
  - Loss: custom `DiceBCELoss` (BCE over sigmoid + soft Dice term).
  - Saves best weights to `unetpp_covid.pth`.

- `finetune.py` — Fine‑tuning with staged unfreezing
  - Loads and concatenates MedSeg and Radiopedia data:
    - `images_medseg.npy`, `masks_medseg.npy` (only two mask channels are used)
    - `images_radiopedia.npy`, `masks_radiopedia.npy` (filters to positive slices only)
  - Splits data 85/15, uses a "safe" PyTorch‑only augmentation dataset (flips and `rot90`).
  - Initializes `UnetPlusPlus` and loads baseline weights from `unetpp_covid.pth`.
  - Freezes encoder parameters for the first 5 epochs, then unfreezes (staged training) and restarts AdamW.
  - Loss/metric same as baseline; saves best to `unetpp_covid_plus.pth`.

- `predict.py` — Inference & submission
  - Loads `unetpp_covid_plus.pth` (by default).
  - Loads `covid_data/test_images_medseg.npy`.
  - Applies the same CT normalization, runs the model, thresholds at 0.35, stacks predictions.
  - Produces `submission_final.csv` with rows equal to `num_images * H * W * 2` (for 2 classes).

---

### Training

Baseline training on MedSeg:
```
python train.py
```
- Hyperparameters (defaults in `train.py`):
  - `EPOCHS=50`, `BATCH_SIZE=4`, `LR=1e-4`, `IMG_SIZE=512`, `THRESHOLD=0.35`
- Output: best weights saved to `unetpp_covid.pth`.

Fine‑tuning on combined dataset with encoder unfreezing:
```
python finetune.py
```
- Hyperparameters (defaults in `finetune.py`):
  - `EPOCHS=30`, `BATCH_SIZE=6`, `LR=1e-5`, `WEIGHT_DECAY=1e-5`, `THRESH=0.35`
- Loads `unetpp_covid.pth`, freezes encoder for 5 epochs, then unfreezes.
- Output: best weights saved to `unetpp_covid_plus.pth`.

---

### Inference and Submission

```
python predict.py
```
- Loads `unetpp_covid_plus.pth` by default (change `WEIGHTS_PATH` if needed).
- Reads `covid_data/test_images_medseg.npy`.
- Writes `submission_final.csv` with columns `(Id, Predicted)`.

Tip: Ensure the total rows printed match `T * 512 * 512 * 2`.

---

### Design Choices & Rationale

- Unet++ with ResNet‑34: strong segmentation baseline with skip‑connection refinement.
- Two‑class setup: focuses on clinically relevant COVID patterns (GGO + consolidation); simplifies training vs. multi‑class schemes.
- HU windowing + per‑slice min‑max: standardizes intensity distribution across scans while preserving lesion contrast.
- Hybrid Dice + BCE: improves stability and overlap vs. BCE alone; Dice helps with class imbalance typical in lesion segmentation.
- Cosine LR + AdamW: widely used, good convergence properties; weight decay helps regularization.
- Staged unfreezing during fine‑tuning: preserves learned generic features early, then adapts encoder later for domain shift.
- Safe augmentations option: removes OpenCV dependency for environments where Albumentations/OpenCV may be problematic.

