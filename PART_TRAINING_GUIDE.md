# Part Training Guide

This guide explains how each team member should run the assigned Dental OPG training notebook after the project is pushed to GitHub.

## Before Training

Make sure the repo already contains:

- the notebooks
- the `artifacts` folder
- the extracted `Dental OPG (Classification)` image folders

The zip file is not needed for training.

## Shared Setup

Before any part starts training, one person should run [01_shared_config.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/01_shared_config.ipynb). This regenerates the shared split manifests and configuration tables from the dataset stored in the repo.

## Google Colab Use

If a teammate opens a notebook directly from GitHub in Colab:

1. Open the assigned notebook.
2. In the bootstrap cell, set `GITHUB_REPO_URL`.
3. Change `RUN_COLAB_BOOTSTRAP` to `True`.
4. Run the bootstrap cell.
5. Run the rest of the notebook top to bottom.

If the notebook is already running inside a cloned repo folder, the bootstrap cell can stay disabled.

## Notebook Assignments

- Part 1 runs [02_dental_opg_xray_part_1.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/02_dental_opg_xray_part_1.ipynb)
- Part 2 runs [03_dental_opg_xray_part_2.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/03_dental_opg_xray_part_2.ipynb)
- Part 3 runs [04_dental_opg_xray_part_3.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/04_dental_opg_xray_part_3.ipynb)

These correspond to:

- Part 1: `dense_units = 16`
- Part 2: `dense_units = 32`
- Part 3: `dense_units = 64`

Each part trains `48` configurations.

## Notebook Settings

Keep these defaults unless you have a reason to change them:

```python
PART_ID = 1 or 2 or 3
OVERWRITE_EXISTING = False
ONLY_CONFIG_IDS = []
STOP_ON_ERROR = False
```

If you want a quick test run first:

```python
ONLY_CONFIG_IDS = ["base_cnn_drop_0p2_batch_16_dense_64"]
```

## Where Outputs Go

By default, outputs are written under `outputs/part_X/` inside the repo clone.

In Colab, you can optionally set `OUTPUTS_DIR` in the notebook to a Google Drive folder if you want results to persist outside the repo working tree.

## After Training

Once all three parts finish, run [05_results.ipynb](D:/road%20to%20summa/4th%20year%20last%20sem/hi%20192/hi-192-dental-xray/notebooks/05_results.ipynb) to combine the saved `metrics.json` files into one summary table.
