# Google Colab Workflow

This project is set up so collaborators can train from Google Colab after cloning the GitHub repository.

## Recommended Setup

1. Push this whole project to GitHub, including the extracted image dataset folder.
2. Open the notebook you need in Google Colab from the GitHub repo.
3. In the notebook bootstrap cell:
   Set `GITHUB_REPO_URL` to your repository URL.
4. Change `RUN_COLAB_BOOTSTRAP` to `True`.
5. Run the bootstrap cell.

The bootstrap cell can clone into Google Drive so the repo and outputs remain available across Colab sessions.

## What Gets Tracked

Tracked in GitHub:

- source code
- notebooks
- manifests and config CSVs
- extracted dataset images under `Dental OPG (Classification)`

Not tracked in GitHub:

- `Dental OPG XRAY Dataset.zip`
- `outputs/`
- local virtual environments

## Colab Notes

- If you keep `DATASET_ROOT = None`, the notebooks use the dataset inside the cloned repo.
- If you keep `OUTPUTS_DIR = None`, the notebooks save to `outputs/` inside the cloned repo.
- If you want outputs somewhere else, set `OUTPUTS_DIR` to a Google Drive path in the notebook.

## Suggested Team Flow

1. One person runs `01_shared_config.ipynb`.
2. Each collaborator opens the part notebook assigned to them.
3. Everyone keeps `OVERWRITE_EXISTING = False` so reruns skip finished configs.
4. After all parts finish, one person runs `05_results.ipynb`.
