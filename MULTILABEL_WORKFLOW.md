# Multilabel Classification Workflow

Use this workflow when treating the classification folders as overlapping image labels instead of mutually exclusive classes.

## What Changed

The original CNN experiment specification is preserved where it still applies:

- Architectures: `base_cnn`, `vgg16`, `resnet34`, `densenet121`
- Input size: `640 x 640`
- Epochs: `50`
- Optimizer: Adam with learning rate `0.0001`
- Dropout grid: `0.2`, `0.5`, `0.7`
- Batch-size grid: `16`, `32`, `64`, `128`
- Dense-unit grid: `16`, `32`, `64`
- Training augmentation only; validation/test preprocessing only

The necessary multilabel changes are:

- One manifest row per original radiograph ID
- Six binary label columns: `BDC-BDR`, `Caries`, `Fractured Teeth`, `Healthy Teeth`, `Impacted teeth`, `Infection`
- Sigmoid output units instead of softmax
- Weighted binary cross-entropy instead of categorical cross-entropy
- Multilabel metrics: subset accuracy, label accuracy, macro/micro precision, recall, F1, specificity, MCC, and AUC

## Dataset Summary

Generated from `Dental OPG XRAY Dataset/Dental OPG XRAY Dataset/Dental OPG (Classification)`:

- Unique radiographs: `231`
- Total folder memberships: `517`
- Images with multiple positive labels: `178`
- Train/validation/test split: `139 / 46 / 46`

The split artifacts live in:

```text
artifacts/multilabel_splits/
```

The experiment configs live in:

```text
artifacts/multilabel_configs/
```

## Notebooks

Run these in order:

1. `notebooks/14_multilabel_setup.ipynb`
2. `notebooks/15_multilabel_part_1.ipynb`
3. `notebooks/16_multilabel_part_2.ipynb`
4. `notebooks/17_multilabel_part_3.ipynb`
5. `notebooks/18_multilabel_results.ipynb`

Each part notebook can be run independently by a collaborator. Completed runs are skipped unless `OVERWRITE_EXISTING = True`.

## Outputs

Each completed config saves under:

```text
outputs_multilabel/part_X/config_id/
```

Expected files include:

- `loss_accuracy_curves.png`
- `auc_curves.png`
- `history.csv`
- `metrics.json`
- `metrics.csv`
- `auc_summary.json`
- `positive_class_weights.json`
- `test_predictions.csv`
- `config.json`

## Colab Notes

In the bootstrap cell:

- Set `GITHUB_REPO_URL = "https://github.com/xyldxal/hi-192-dental-xray.git"`
- Set `RUN_COLAB_BOOTSTRAP = True`
- Use Google Drive only if you want outputs persisted across Colab runtime resets

If dependencies are missing, run the optional `%pip install -r requirements.txt` cell after the project import cell. The notebook changes into `PROJECT_ROOT`, so it should find `requirements.txt`.
