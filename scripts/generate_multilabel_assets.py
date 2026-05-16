from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dental_opg_experiment as doe


NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in textwrap.dedent(source).strip().splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in textwrap.dedent(source).strip().splitlines()],
    }


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def colab_bootstrap_markdown() -> dict:
    return markdown_cell(
        """
        ## Optional Google Colab Bootstrap

        If you open this notebook directly from GitHub in Google Colab, run the next cell after:

        1. Setting `GITHUB_REPO_URL`
        2. Changing `RUN_COLAB_BOOTSTRAP` to `True`

        Leave the cell as-is for local Jupyter use.
        """
    )


def colab_bootstrap_code() -> dict:
    return code_cell(
        """
        import os
        import subprocess
        from pathlib import Path

        RUN_COLAB_BOOTSTRAP = False
        GITHUB_REPO_URL = ""
        USE_GOOGLE_DRIVE = False
        COLAB_REPO_DIR = "/content/hi-192-dental-xray"

        try:
            from google.colab import drive  # type: ignore
            IN_COLAB = True
        except ImportError:
            IN_COLAB = False

        if IN_COLAB and RUN_COLAB_BOOTSTRAP:
            if USE_GOOGLE_DRIVE:
                drive.mount("/content/drive")

            repo_root = Path(COLAB_REPO_DIR)
            repo_root.parent.mkdir(parents=True, exist_ok=True)

            if not (repo_root / ".git").exists():
                if not GITHUB_REPO_URL.strip():
                    raise ValueError("Set GITHUB_REPO_URL before running the Colab bootstrap cell.")
                subprocess.run(["git", "clone", GITHUB_REPO_URL, str(repo_root)], check=True)

            os.chdir(repo_root)
            print(f"Colab working directory set to: {repo_root}")
        else:
            print(f"Current working directory: {Path.cwd()}")
        """
    )


def project_import_code() -> dict:
    return code_cell(
        """
        from pathlib import Path
        import os
        import sys

        def find_project_root():
            for candidate in [Path.cwd(), *Path.cwd().parents]:
                if (candidate / "src" / "dental_opg_experiment.py").exists():
                    return candidate
            raise FileNotFoundError(
                "Could not locate the project root. In Colab, run the bootstrap cell first or clone the repo manually."
            )

        PROJECT_ROOT = find_project_root()
        os.chdir(PROJECT_ROOT)

        SRC_DIR = PROJECT_ROOT / "src"
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        import dental_opg_experiment as doe
        """
    )


def multilabel_runtime_paths_code() -> dict:
    return code_cell(
        """
        DATASET_ROOT = None
        MULTILABEL_SPLITS_DIR = PROJECT_ROOT / "artifacts" / "multilabel_splits"
        MULTILABEL_CONFIGS_DIR = PROJECT_ROOT / "artifacts" / "multilabel_configs"
        OUTPUTS_DIR = PROJECT_ROOT / "outputs_multilabel"

        print(f"Dataset root: {doe.resolve_dataset_root(DATASET_ROOT)}")
        print(f"Multilabel splits dir: {MULTILABEL_SPLITS_DIR}")
        print(f"Multilabel configs dir: {MULTILABEL_CONFIGS_DIR}")
        print(f"Outputs dir: {OUTPUTS_DIR}")
        """
    )


def optional_requirements_cell() -> dict:
    return code_cell(
        """
        # Run this if the environment is missing TensorFlow, scikit-learn, or image-classifiers.
        # Because the previous cell changes into PROJECT_ROOT, this will find requirements.txt.
        # %pip install -r requirements.txt
        """
    )


def build_multilabel_setup_notebook() -> None:
    path = NOTEBOOKS_DIR / "14_multilabel_setup.ipynb"
    cells = [
        markdown_cell(
            """
            # Multilabel Classification Setup

            This notebook creates a one-row-per-radiograph manifest from the classification folders. Unlike the original single-label workflow, duplicated folder memberships are preserved as independent binary labels, so one image can have multiple positive classes.

            The CNN experiment grid stays the same: `base_cnn`, `vgg16`, `resnet34`, `densenet121`; dropout `0.2/0.5/0.7`; batch size `16/32/64/128`; dense units `16/32/64`; image size `640 x 640`; epochs `50`; Adam learning rate `0.0001`.
            """
        ),
        colab_bootstrap_markdown(),
        colab_bootstrap_code(),
        project_import_code(),
        optional_requirements_cell(),
        multilabel_runtime_paths_code(),
        code_cell(
            """
            import json

            split_summary = doe.write_multilabel_split_artifacts(
                dataset_root=DATASET_ROOT,
                out_dir=MULTILABEL_SPLITS_DIR,
                seed=42,
            )
            part_summary = doe.write_config_artifacts(out_dir=MULTILABEL_CONFIGS_DIR)

            print(json.dumps(split_summary, indent=2))
            print(json.dumps(part_summary, indent=2))
            """
        ),
        code_cell(
            """
            print("Multilabel split files:")
            for path in sorted(MULTILABEL_SPLITS_DIR.glob("*")):
                print(path.name)

            print("\\nMultilabel config files:")
            for path in sorted(MULTILABEL_CONFIGS_DIR.glob("*")):
                print(path.name)
            """
        ),
    ]
    write_notebook(path, cells)


def build_multilabel_part_notebook(part_id: int) -> None:
    path = NOTEBOOKS_DIR / f"{14 + part_id:02d}_multilabel_part_{part_id}.ipynb"
    cells = [
        markdown_cell(
            f"""
            # Multilabel Classification Part {part_id}

            This notebook trains the same CNN experiment grid on multilabel image targets. The only required modeling change is a sigmoid output head with weighted binary cross-entropy instead of a softmax head with categorical cross-entropy.
            """
        ),
        colab_bootstrap_markdown(),
        colab_bootstrap_code(),
        project_import_code(),
        optional_requirements_cell(),
        multilabel_runtime_paths_code(),
        code_cell(
            f"""
            PART_ID = {part_id}
            OVERWRITE_EXISTING = False
            ONLY_CONFIG_IDS = []  # Example: ["base_cnn_drop_0p2_batch_16_dense_64"]
            STOP_ON_ERROR = False
            THRESHOLD = 0.5
            """
        ),
        code_cell(
            """
            configs = doe.load_part_configs(PART_ID, configs_dir=MULTILABEL_CONFIGS_DIR)
            print(f"Multilabel part {PART_ID} has {len(configs)} configs.")

            for config in configs[:5]:
                print(config)
            """
        ),
        code_cell(
            """
            import json

            output_root = doe.resolve_outputs_dir(OUTPUTS_DIR) / f"part_{PART_ID}"
            output_root.mkdir(parents=True, exist_ok=True)

            selected_configs = [cfg for cfg in configs if not ONLY_CONFIG_IDS or cfg["config_id"] in ONLY_CONFIG_IDS]
            results = []

            for config in selected_configs:
                run_dir = output_root / config["config_id"]
                metrics_path = run_dir / "metrics.json"

                if metrics_path.exists() and not OVERWRITE_EXISTING:
                    print(f"Skipping completed run: {config['config_id']}")
                    results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    continue

                print(
                    f"Running {config['config_id']} -> {config['architecture']}, "
                    f"dropout={config['dropout']}, batch={config['batch_size']}, dense={config['dense_units']}"
                )

                try:
                    metrics = doe.train_single_multilabel_config(
                        config=config,
                        part_id=PART_ID,
                        outputs_dir=OUTPUTS_DIR,
                        splits_dir=MULTILABEL_SPLITS_DIR,
                        dataset_root=DATASET_ROOT,
                        threshold=THRESHOLD,
                    )
                    results.append(metrics)
                except Exception as exc:
                    print(f"Failed on {config['config_id']}: {exc}")
                    if STOP_ON_ERROR:
                        raise

            print(f"Collected {len(results)} result rows.")
            """
        ),
        code_cell(
            """
            summary_df = doe.aggregate_part_results(outputs_dir=OUTPUTS_DIR, part_id=PART_ID)
            summary_path = doe.resolve_outputs_dir(OUTPUTS_DIR) / f"part_{PART_ID}" / "part_summary.csv"

            if not summary_df.empty:
                summary_df.to_csv(summary_path, index=False)
                display(summary_df.sort_values("f1_macro", ascending=False).head(10))
                print(f"Saved summary to: {summary_path}")
            else:
                print("No completed runs yet.")
            """
        ),
    ]
    write_notebook(path, cells)


def build_multilabel_results_notebook() -> None:
    path = NOTEBOOKS_DIR / "18_multilabel_results.ipynb"
    cells = [
        markdown_cell(
            """
            # Multilabel Classification Results

            This notebook consolidates all completed multilabel experiment outputs.
            """
        ),
        colab_bootstrap_markdown(),
        colab_bootstrap_code(),
        project_import_code(),
        optional_requirements_cell(),
        multilabel_runtime_paths_code(),
        code_cell(
            """
            all_results = doe.aggregate_part_results(outputs_dir=OUTPUTS_DIR)
            all_results
            """
        ),
        code_cell(
            """
            if not all_results.empty:
                output_path = doe.resolve_outputs_dir(OUTPUTS_DIR) / "all_results_summary.csv"
                all_results.to_csv(output_path, index=False)
                display(all_results.sort_values("f1_macro", ascending=False).head(20))
                print(f"Saved combined results to: {output_path}")
            else:
                print("No metrics were found yet.")
            """
        ),
    ]
    write_notebook(path, cells)


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    split_summary = doe.write_multilabel_split_artifacts(
        out_dir=doe.MULTILABEL_SPLITS_DIR,
        seed=42,
    )
    part_summary = doe.write_config_artifacts(out_dir=doe.MULTILABEL_CONFIGS_DIR)

    build_multilabel_setup_notebook()
    build_multilabel_part_notebook(1)
    build_multilabel_part_notebook(2)
    build_multilabel_part_notebook(3)
    build_multilabel_results_notebook()

    print("Generated multilabel split and config artifacts.")
    print(json.dumps(split_summary, indent=2))
    print(json.dumps(part_summary, indent=2))
    print(f"Multilabel notebooks written to: {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
