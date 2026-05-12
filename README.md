# Dental OPG Xray Project Setup

This repository is prepared for the 4-class Dental OPG classification capstone project using only these folders:

- `BDC-BDR`
- `Caries`
- `Healthy Teeth`
- `Impacted teeth`

These folders remain present in the source dataset but are excluded from training:

- `Fractured Teeth`
- `Infection`

## Repository Contents

- `Dental OPG XRAY Dataset/Dental OPG XRAY Dataset/Dental OPG (Classification)/`
- `artifacts/splits/`
- `artifacts/configs/`
- `notebooks/`
- `scripts/`
- `src/`

The extracted image dataset is intended to be committed to GitHub with the repo. The compressed file `Dental OPG XRAY Dataset.zip` stays ignored because it is too large and not needed once the folder is extracted.

## Split Summary

- Total used images: `481`
- Train: `289`
- Validation: `96`
- Test: `96`
- Random seed: `42`

Per-class split counts:

- Train: `BDC-BDR=31`, `Caries=71`, `Healthy Teeth=134`, `Impacted teeth=53`
- Validation: `BDC-BDR=11`, `Caries=24`, `Healthy Teeth=44`, `Impacted teeth=17`
- Test: `BDC-BDR=10`, `Caries=24`, `Healthy Teeth=45`, `Impacted teeth=17`

## Experiment Layout

- Architectures: `base_cnn`, `vgg16`, `resnet34`, `densenet121`
- Dropout values: `0.2`, `0.5`, `0.7`
- Batch sizes: `16`, `32`, `64`, `128`
- Dense units: `16`, `32`, `64`
- Total runs: `144`

Parts are assigned by dense units:

- Part 1: `dense_units = 16`
- Part 2: `dense_units = 32`
- Part 3: `dense_units = 64`

## Notebooks

- [01_shared_config.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/01_shared_config.ipynb)
- [02_dental_opg_xray_part_1.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/02_dental_opg_xray_part_1.ipynb)
- [03_dental_opg_xray_part_2.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/03_dental_opg_xray_part_2.ipynb)
- [04_dental_opg_xray_part_3.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/04_dental_opg_xray_part_3.ipynb)
- [05_results.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/05_results.ipynb)

Each notebook now includes an optional Colab bootstrap cell for collaborators who open the notebook directly from GitHub.

## Local Setup

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/generate_study_assets.py
```

## Google Colab Workflow

Use the notebook bootstrap cell if you open the notebook from GitHub in Colab:

1. Set `GITHUB_REPO_URL` inside the bootstrap cell.
2. Change `RUN_COLAB_BOOTSTRAP` to `True`.
3. Run the bootstrap cell to clone the repo into Colab or Google Drive.
4. Run the remaining notebook cells top to bottom.

Detailed instructions are in [GOOGLE_COLAB_WORKFLOW.md](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/GOOGLE_COLAB_WORKFLOW.md).

## Expected Outputs Per Run

Each config saves to `outputs/part_X/config_id/`:

- `loss_accuracy_curves.png`
- `auc_curves.png`
- `history.csv`
- `metrics.json`
- `metrics.csv`
- `auc_summary.json`
- `class_weights.json`
- `config.json`

## Notes

- Input size is fixed to `640 x 640`
- Base CNN uses `3` convolutional layers with filters `8`, `16`, `32`
- Optimizer is Adam with learning rate `0.0001`
- Epochs are fixed at `50`
- Class weights use inverse proportional frequency based on the training split
- Output layer is multiclass softmax for `4` classes
- Augmentation is applied to training and validation generators
- Test generator uses normalization only
