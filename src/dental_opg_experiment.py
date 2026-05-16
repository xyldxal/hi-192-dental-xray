from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dental OPG XRAY Dataset" / "Dental OPG XRAY Dataset" / "Dental OPG (Classification)"
OBJECT_DETECTION_ROOT = PROJECT_ROOT / "Dental OPG XRAY Dataset" / "Dental OPG XRAY Dataset" / "Dental OPG (Object Detection)"
OBJECT_DETECTION_ORIGINAL_DIR = PROJECT_ROOT / "Dental OPG XRAY Dataset" / "Dental OPG XRAY Dataset" / "Dental OPG (Object Detection)" / "Original Dataset"
OBJECT_DETECTION_AUGMENTED_DIR = PROJECT_ROOT / "Dental OPG XRAY Dataset" / "Dental OPG XRAY Dataset" / "Dental OPG (Object Detection)" / "Augmented Dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
CONFIGS_DIR = ARTIFACTS_DIR / "configs"
PRIMARY_CONDITION_SPLITS_DIR = ARTIFACTS_DIR / "primary_condition_splits"
PRIMARY_CONDITION_CONFIGS_DIR = ARTIFACTS_DIR / "primary_condition_configs"
MULTILABEL_SPLITS_DIR = ARTIFACTS_DIR / "multilabel_splits"
MULTILABEL_CONFIGS_DIR = ARTIFACTS_DIR / "multilabel_configs"
YOLO_DETECTION_SPLITS_DIR = ARTIFACTS_DIR / "yolo_detection_splits"
YOLO_DETECTION_DATASET_DIR = PROJECT_ROOT / "yolo_detection_dataset"
YOLO_DETECTION_OUTPUTS_DIR = PROJECT_ROOT / "outputs_yolo_detection"
MULTILABEL_OUTPUTS_DIR = PROJECT_ROOT / "outputs_multilabel"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_ROOT_ENV_VAR = "DENTAL_OPG_DATASET_ROOT"
OUTPUTS_DIR_ENV_VAR = "DENTAL_OPG_OUTPUTS_DIR"

INCLUDED_CLASS_FOLDERS = {
    "BDC-BDR": "Broken down crown/roots",
    "Caries": "Caries",
    "Healthy Teeth": "Healthy teeth",
    "Impacted teeth": "Impacted teeth",
}

EXCLUDED_CLASS_FOLDERS = {
    "Fractured Teeth",
    "Infection",
}

ALL_CLASS_FOLDERS = {
    "BDC-BDR": "Broken down crown/roots",
    "Caries": "Caries",
    "Fractured Teeth": "Fractured teeth",
    "Healthy Teeth": "Healthy teeth",
    "Impacted teeth": "Impacted teeth",
    "Infection": "Infection",
}

MULTILABEL_CLASS_FOLDERS = tuple(sorted(INCLUDED_CLASS_FOLDERS))

YOLO_CLASS_ID_TO_FOLDER = {
    0: "Caries",
    1: "Infection",
    2: "Impacted teeth",
    3: "Fractured Teeth",
    4: "BDC-BDR",
    5: "Healthy Teeth",
}
YOLO_CLASS_NAMES = {
    0: "Caries",
    1: "Infection",
    2: "Impacted teeth",
    3: "Fractured Teeth",
    4: "BDC-BDR",
    5: "Healthy Teeth",
}
PRIMARY_CONDITION_YOLO_IDS = {
    0,
    2,
    4,
}
HEALTHY_TEETH_YOLO_ID = 5

ARCHITECTURES = (
    "base_cnn",
    "vgg16",
    "resnet34",
    "densenet121",
)

DROPOUT_VALUES = (0.2, 0.5, 0.7)
BATCH_SIZES = (16, 32, 64, 128)
DENSE_UNITS = (16, 32, 64)
IMAGE_SIZE = (640, 640)
INPUT_SHAPE = (640, 640, 3)
EPOCHS = 50
LEARNING_RATE = 0.0001
DEFAULT_SEED = 42

TRAIN_RATIO = 289 / 481
VAL_RATIO = 96 / 481
TEST_RATIO = 96 / 481


@dataclass(frozen=True)
class SampleRecord:
    class_folder: str
    class_label: str
    class_index: int
    filename: str
    filepath: str
    relative_path: str


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_path(
    supplied_path: Path | str | None,
    env_var_name: str,
    default_path: Path,
    *,
    must_exist: bool,
    label: str,
) -> Path:
    if supplied_path:
        resolved = Path(supplied_path)
    else:
        env_value = os.environ.get(env_var_name, "").strip()
        resolved = Path(env_value) if env_value else default_path

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")

    return resolved


def resolve_dataset_root(dataset_root: Path | str | None = None) -> Path:
    return _resolve_path(
        dataset_root,
        DATASET_ROOT_ENV_VAR,
        DATASET_ROOT,
        must_exist=True,
        label="Dataset root",
    )


def resolve_object_detection_original_dir(
    object_detection_original_dir: Path | str | None = None,
) -> Path:
    path = Path(object_detection_original_dir) if object_detection_original_dir else OBJECT_DETECTION_ORIGINAL_DIR
    if not path.exists():
        raise FileNotFoundError(f"Object-detection original dataset directory does not exist: {path}")
    return path


def resolve_object_detection_augmented_dir(
    object_detection_augmented_dir: Path | str | None = None,
) -> Path:
    path = Path(object_detection_augmented_dir) if object_detection_augmented_dir else OBJECT_DETECTION_AUGMENTED_DIR
    if not path.exists():
        raise FileNotFoundError(f"Object-detection augmented dataset directory does not exist: {path}")
    return path


def resolve_outputs_dir(outputs_dir: Path | str | None = None) -> Path:
    return _resolve_path(
        outputs_dir,
        OUTPUTS_DIR_ENV_VAR,
        OUTPUTS_DIR,
        must_exist=False,
        label="Outputs directory",
    )


def get_class_index_mapping() -> Dict[str, int]:
    ordered_folders = sorted(INCLUDED_CLASS_FOLDERS)
    return {folder: index for index, folder in enumerate(ordered_folders)}


def _source_id_sort_key(value: str) -> Tuple[int, int | str]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _multilabel_column_name(class_folder: str) -> str:
    normalized = class_folder.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return f"label_{normalized}"


def _is_multilabel_target_column(column_name: str) -> bool:
    return column_name.startswith("label_") and column_name != "label_count"


def get_multilabel_class_folders(class_folders: Sequence[str] | None = None) -> Tuple[str, ...]:
    folders = tuple(class_folders) if class_folders else MULTILABEL_CLASS_FOLDERS
    unknown = sorted(set(folders) - set(ALL_CLASS_FOLDERS))
    if unknown:
        raise ValueError(f"Unknown multilabel class folders: {unknown}")
    return folders


def get_multilabel_label_columns(class_folders: Sequence[str] | None = None) -> List[str]:
    return [_multilabel_column_name(folder) for folder in get_multilabel_class_folders(class_folders)]


def list_included_samples(dataset_root: Path | str | None = None) -> List[SampleRecord]:
    root = resolve_dataset_root(dataset_root)
    class_index_mapping = get_class_index_mapping()
    samples: List[SampleRecord] = []

    for class_folder in sorted(INCLUDED_CLASS_FOLDERS):
        class_dir = root / class_folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder is missing: {class_dir}")

        for image_path in sorted(class_dir.glob("*.jpg")):
            samples.append(
                SampleRecord(
                    class_folder=class_folder,
                    class_label=INCLUDED_CLASS_FOLDERS[class_folder],
                    class_index=class_index_mapping[class_folder],
                    filename=image_path.name,
                    filepath=str(image_path.resolve()),
                    relative_path=image_path.relative_to(root).as_posix(),
                )
            )

    return samples


def _largest_remainder_allocation(class_counts: Dict[str, int], total_target: int) -> Dict[str, int]:
    total_count = sum(class_counts.values())
    if total_target > total_count:
        raise ValueError("Allocation target cannot exceed the available sample count.")

    raw_allocations = {
        class_name: (count / total_count) * total_target
        for class_name, count in class_counts.items()
    }
    floors = {class_name: math.floor(value) for class_name, value in raw_allocations.items()}
    allocated = sum(floors.values())
    remaining = total_target - allocated

    remainders = sorted(
        (
            raw_allocations[class_name] - floors[class_name],
            class_name,
        )
        for class_name in class_counts
    )
    remainders.reverse()

    allocation = floors.copy()
    for _, class_name in remainders[:remaining]:
        allocation[class_name] += 1

    return allocation


def build_multilabel_rows(
    dataset_root: Path | str | None = None,
    class_folders: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    root = resolve_dataset_root(dataset_root)
    selected_folders = get_multilabel_class_folders(class_folders)
    label_columns = get_multilabel_label_columns(selected_folders)

    memberships: Dict[str, set[str]] = defaultdict(set)
    relative_paths_by_id: Dict[str, Dict[str, str]] = defaultdict(dict)

    for class_folder in selected_folders:
        class_dir = root / class_folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder is missing: {class_dir}")

        for image_path in sorted(class_dir.glob("*.jpg"), key=lambda path: _source_id_sort_key(path.stem)):
            source_image_id = image_path.stem
            memberships[source_image_id].add(class_folder)
            relative_paths_by_id[source_image_id][class_folder] = image_path.relative_to(root).as_posix()

    rows: List[Dict[str, object]] = []
    for source_image_id in sorted(memberships, key=_source_id_sort_key):
        present_folders = sorted(memberships[source_image_id])
        present_labels = [ALL_CLASS_FOLDERS[folder] for folder in present_folders]

        # The classification folders contain duplicated copies of the same
        # radiograph. Use one physical file per image ID, then keep all folder
        # memberships as independent labels.
        representative_folder = next(
            folder for folder in selected_folders if folder in relative_paths_by_id[source_image_id]
        )

        row: Dict[str, object] = {
            "source_image_id": source_image_id,
            "filename": f"{source_image_id}.jpg",
            "relative_path": relative_paths_by_id[source_image_id][representative_folder],
            "representative_class_folder": representative_folder,
            "class_folders": json.dumps(present_folders),
            "class_labels": json.dumps(present_labels),
            "label_count": len(present_folders),
        }

        for class_folder, label_column in zip(selected_folders, label_columns):
            row[label_column] = 1 if class_folder in memberships[source_image_id] else 0

        rows.append(row)

    return rows


def _split_multilabel_rows(
    rows: Sequence[Dict[str, object]],
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[Dict[str, object]]]:
    total_rows = len(rows)
    test_total = round(total_rows * TEST_RATIO)
    val_total = round(total_rows * VAL_RATIO)

    shuffled = [dict(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    test_rows = shuffled[:test_total]
    val_rows = shuffled[test_total : test_total + val_total]
    train_rows = shuffled[test_total + val_total :]

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def write_multilabel_split_artifacts(
    dataset_root: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
    class_folders: Sequence[str] | None = None,
) -> Dict[str, object]:
    selected_folders = get_multilabel_class_folders(class_folders)
    label_columns = get_multilabel_label_columns(selected_folders)
    rows = build_multilabel_rows(dataset_root=dataset_root, class_folders=selected_folders)
    split_rows = _split_multilabel_rows(rows, seed=seed)
    output_dir = _ensure_directory(Path(out_dir) if out_dir else MULTILABEL_SPLITS_DIR)

    manifest_fieldnames = [
        "split",
        "source_image_id",
        "filename",
        "relative_path",
        "representative_class_folder",
        "class_folders",
        "class_labels",
        "label_count",
        *label_columns,
    ]

    class_to_column = {
        class_folder: label_column
        for class_folder, label_column in zip(selected_folders, label_columns)
    }
    summary = {
        "seed": seed,
        "image_size": list(IMAGE_SIZE),
        "label_task": "multilabel_classification",
        "label_rule": (
            "One row per original radiograph ID. Every classification-folder membership becomes "
            "an independent binary target; no folder is forced to be the single winning class."
        ),
        "class_folders": list(selected_folders),
        "class_folder_labels": {folder: ALL_CLASS_FOLDERS[folder] for folder in selected_folders},
        "class_folder_to_label_column": class_to_column,
        "total_unique_images": len(rows),
        "total_folder_memberships": int(sum(int(row["label_count"]) for row in rows)),
        "multi_positive_images": int(sum(1 for row in rows if int(row["label_count"]) > 1)),
        "splits": {},
    }

    all_rows: List[Dict[str, object]] = []
    for split_name, records in split_rows.items():
        per_label_positive = {
            class_folder: int(sum(int(record[label_column]) for record in records))
            for class_folder, label_column in class_to_column.items()
        }
        label_count_distribution = Counter(int(record["label_count"]) for record in records)

        summary["splits"][split_name] = {
            "total": len(records),
            "per_label_positive": dict(sorted(per_label_positive.items())),
            "label_count_distribution": dict(sorted((str(key), value) for key, value in label_count_distribution.items())),
        }

        split_records = []
        for record in sorted(records, key=lambda item: _source_id_sort_key(str(item["source_image_id"]))):
            row = {"split": split_name, **record}
            split_records.append(row)
            all_rows.append(row)

        split_path = output_dir / f"{split_name}_manifest.csv"
        with split_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
            writer.writeheader()
            writer.writerows(split_records)

    master_manifest_path = output_dir / "master_manifest.csv"
    with master_manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "multilabel_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_split_records(
    dataset_root: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[SampleRecord]]:
    samples = list_included_samples(dataset_root)
    grouped: Dict[str, List[SampleRecord]] = defaultdict(list)
    for sample in samples:
        grouped[sample.class_folder].append(sample)

    total_samples = len(samples)
    test_total = round(total_samples * TEST_RATIO)
    val_total = round(total_samples * VAL_RATIO)
    train_total = total_samples - test_total - val_total

    class_counts = {class_folder: len(group) for class_folder, group in grouped.items()}
    test_targets = _largest_remainder_allocation(class_counts, test_total)

    remaining_after_test = {
        class_folder: class_counts[class_folder] - test_targets[class_folder]
        for class_folder in class_counts
    }
    val_targets = _largest_remainder_allocation(remaining_after_test, val_total)
    train_targets = {
        class_folder: remaining_after_test[class_folder] - val_targets[class_folder]
        for class_folder in class_counts
    }

    split_records: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for class_folder, records in sorted(grouped.items()):
        shuffled = list(records)
        rng.shuffle(shuffled)

        test_cutoff = test_targets[class_folder]
        val_cutoff = test_cutoff + val_targets[class_folder]

        split_records["test"].extend(shuffled[:test_cutoff])
        split_records["val"].extend(shuffled[test_cutoff:val_cutoff])
        split_records["train"].extend(shuffled[val_cutoff:])

        if len(shuffled[val_cutoff:]) != train_targets[class_folder]:
            raise ValueError(f"Unexpected train allocation for class {class_folder}")

    if len(split_records["train"]) != train_total:
        raise ValueError("Train split size does not match the requested total.")
    if len(split_records["val"]) != val_total:
        raise ValueError("Validation split size does not match the requested total.")
    if len(split_records["test"]) != test_total:
        raise ValueError("Test split size does not match the requested total.")

    return split_records


def write_split_artifacts(
    dataset_root: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    split_records = build_split_records(dataset_root=dataset_root, seed=seed)
    output_dir = _ensure_directory(Path(out_dir) if out_dir else SPLITS_DIR)

    summary = {
        "seed": seed,
        "image_size": list(IMAGE_SIZE),
        "class_mapping": get_class_index_mapping(),
        "included_class_folders": INCLUDED_CLASS_FOLDERS,
        "excluded_class_folders": sorted(EXCLUDED_CLASS_FOLDERS),
        "splits": {},
    }

    manifest_fieldnames = [
        "split",
        "class_folder",
        "class_label",
        "class_index",
        "filename",
        "relative_path",
    ]

    all_rows: List[Dict[str, object]] = []
    for split_name, records in split_records.items():
        counts = Counter(record.class_folder for record in records)
        summary["splits"][split_name] = {
            "total": len(records),
            "per_class": dict(sorted(counts.items())),
        }

        rows = [
            {
                "split": split_name,
                "class_folder": record.class_folder,
                "class_label": record.class_label,
                "class_index": record.class_index,
                "filename": record.filename,
                "relative_path": record.relative_path,
            }
            for record in sorted(records, key=lambda item: (item.class_folder, item.filename))
        ]
        all_rows.extend(rows)

        split_path = output_dir / f"{split_name}_manifest.csv"
        with split_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    master_manifest_path = output_dir / "master_manifest.csv"
    with master_manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _split_rows_by_class(
    rows: Sequence[Dict[str, object]],
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_folder"])].append(dict(row))

    total_rows = len(rows)
    test_total = round(total_rows * TEST_RATIO)
    val_total = round(total_rows * VAL_RATIO)
    train_total = total_rows - test_total - val_total

    class_counts = {class_folder: len(group) for class_folder, group in grouped.items()}
    test_targets = _largest_remainder_allocation(class_counts, test_total)
    remaining_after_test = {
        class_folder: class_counts[class_folder] - test_targets[class_folder]
        for class_folder in class_counts
    }
    val_targets = _largest_remainder_allocation(remaining_after_test, val_total)

    split_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for class_folder, class_rows in sorted(grouped.items()):
        shuffled = list(class_rows)
        rng.shuffle(shuffled)

        test_cutoff = test_targets[class_folder]
        val_cutoff = test_cutoff + val_targets[class_folder]

        split_rows["test"].extend(shuffled[:test_cutoff])
        split_rows["val"].extend(shuffled[test_cutoff:val_cutoff])
        split_rows["train"].extend(shuffled[val_cutoff:])

    if len(split_rows["train"]) != train_total:
        raise ValueError("Train split size does not match the requested total.")
    if len(split_rows["val"]) != val_total:
        raise ValueError("Validation split size does not match the requested total.")
    if len(split_rows["test"]) != test_total:
        raise ValueError("Test split size does not match the requested total.")

    return split_rows


def _read_yolo_annotation(label_path: Path) -> List[Dict[str, float]]:
    annotations: List[Dict[str, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        class_id, x_center, y_center, width, height = line.split()
        annotations.append(
            {
                "class_id": int(class_id),
                "x_center": float(x_center),
                "y_center": float(y_center),
                "width": float(width),
                "height": float(height),
                "area": float(width) * float(height),
            }
        )

    return annotations


def build_primary_condition_rows(
    dataset_root: Path | str | None = None,
    object_detection_original_dir: Path | str | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    classification_root = resolve_dataset_root(dataset_root)
    od_original_dir = resolve_object_detection_original_dir(object_detection_original_dir)
    class_index_mapping = get_class_index_mapping()

    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    label_paths = sorted(
        od_original_dir.glob("*.txt"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )

    for label_path in label_paths:
        annotations = _read_yolo_annotation(label_path)
        present_class_ids = sorted({int(item["class_id"]) for item in annotations})
        selected_areas: Dict[int, float] = defaultdict(float)

        for annotation in annotations:
            class_id = int(annotation["class_id"])
            if class_id in PRIMARY_CONDITION_YOLO_IDS:
                selected_areas[class_id] += float(annotation["area"])

        if selected_areas:
            primary_yolo_class_id = max(
                selected_areas,
                key=lambda class_id: (selected_areas[class_id], -class_id),
            )
            primary_rule = "largest_selected_disease_box_area"
        elif set(present_class_ids) <= {HEALTHY_TEETH_YOLO_ID}:
            primary_yolo_class_id = HEALTHY_TEETH_YOLO_ID
            primary_rule = "healthy_only"
        else:
            skipped.append(
                {
                    "source_image_id": label_path.stem,
                    "reason": "only_excluded_disease_classes_present",
                    "present_yolo_class_ids": present_class_ids,
                }
            )
            continue

        class_folder = YOLO_CLASS_ID_TO_FOLDER[primary_yolo_class_id]
        relative_path = f"{class_folder}/{label_path.stem}.jpg"
        image_path = classification_root / relative_path
        if not image_path.exists():
            skipped.append(
                {
                    "source_image_id": label_path.stem,
                    "reason": "classification_image_copy_missing",
                    "primary_class_folder": class_folder,
                    "relative_path": relative_path,
                    "present_yolo_class_ids": present_class_ids,
                }
            )
            continue

        area_by_class = {
            YOLO_CLASS_ID_TO_FOLDER[class_id]: round(area, 10)
            for class_id, area in sorted(selected_areas.items())
        }

        rows.append(
            {
                "class_folder": class_folder,
                "class_label": INCLUDED_CLASS_FOLDERS[class_folder],
                "class_index": class_index_mapping[class_folder],
                "filename": image_path.name,
                "relative_path": relative_path,
                "source_image_id": label_path.stem,
                "primary_yolo_class_id": primary_yolo_class_id,
                "primary_rule": primary_rule,
                "primary_area": round(float(selected_areas.get(primary_yolo_class_id, 0.0)), 10),
                "selected_yolo_area_by_class": json.dumps(area_by_class, sort_keys=True),
                "present_yolo_class_ids": json.dumps(present_class_ids),
            }
        )

    return rows, skipped


def write_primary_condition_split_artifacts(
    dataset_root: Path | str | None = None,
    object_detection_original_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    rows, skipped = build_primary_condition_rows(
        dataset_root=dataset_root,
        object_detection_original_dir=object_detection_original_dir,
    )
    split_rows = _split_rows_by_class(rows, seed=seed)
    output_dir = _ensure_directory(Path(out_dir) if out_dir else PRIMARY_CONDITION_SPLITS_DIR)

    manifest_fieldnames = [
        "split",
        "class_folder",
        "class_label",
        "class_index",
        "filename",
        "relative_path",
        "source_image_id",
        "primary_yolo_class_id",
        "primary_rule",
        "primary_area",
        "selected_yolo_area_by_class",
        "present_yolo_class_ids",
    ]

    summary = {
        "seed": seed,
        "image_size": list(IMAGE_SIZE),
        "primary_label_rule": (
            "Choose the included disease class with the largest total YOLO bounding-box area; "
            "use Healthy Teeth only when the image has healthy-teeth annotations and no disease annotations."
        ),
        "yolo_class_id_to_folder": YOLO_CLASS_ID_TO_FOLDER,
        "included_class_folders": INCLUDED_CLASS_FOLDERS,
        "excluded_class_folders": sorted(EXCLUDED_CLASS_FOLDERS),
        "total_used_images": len(rows),
        "skipped_images": skipped,
        "splits": {},
    }

    all_rows: List[Dict[str, object]] = []
    for split_name, records in split_rows.items():
        counts = Counter(str(record["class_folder"]) for record in records)
        summary["splits"][split_name] = {
            "total": len(records),
            "per_class": dict(sorted(counts.items())),
        }

        split_records = []
        for record in sorted(records, key=lambda item: (str(item["class_folder"]), str(item["filename"]))):
            row = {"split": split_name, **record}
            split_records.append(row)
            all_rows.append(row)

        split_path = output_dir / f"{split_name}_manifest.csv"
        with split_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
            writer.writeheader()
            writer.writerows(split_records)

    master_manifest_path = output_dir / "master_manifest.csv"
    with master_manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "primary_condition_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_augmented_source_id(image_path: Path) -> str:
    return image_path.name.split("_jpg.rf.", 1)[0]


def _build_authored_yolo_split_ids(
    object_detection_augmented_dir: Path | str | None = None,
) -> Dict[str, set[str]]:
    augmented_dir = resolve_object_detection_augmented_dir(object_detection_augmented_dir)
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }
    split_ids: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for source_split_name, target_split_name in split_map.items():
        image_dir = augmented_dir / source_split_name / "images"
        if not image_dir.exists():
            raise FileNotFoundError(f"Expected YOLO augmented image directory is missing: {image_dir}")

        split_ids[target_split_name] = {
            _parse_augmented_source_id(image_path)
            for image_path in image_dir.glob("*.jpg")
        }

    return split_ids


def _build_random_yolo_split_ids(
    source_ids: Sequence[str],
    seed: int = DEFAULT_SEED,
) -> Dict[str, set[str]]:
    shuffled = sorted(source_ids, key=lambda value: int(value) if value.isdigit() else value)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_total = round(total * 0.10)
    val_total = round(total * 0.10)

    test_ids = set(shuffled[:test_total])
    val_ids = set(shuffled[test_total : test_total + val_total])
    train_ids = set(shuffled[test_total + val_total :])

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_yolo_detection_split_rows(
    object_detection_original_dir: Path | str | None = None,
    object_detection_augmented_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]], str]:
    original_dir = resolve_object_detection_original_dir(object_detection_original_dir)

    available_ids = {
        image_path.stem
        for image_path in original_dir.glob("*.jpg")
        if (original_dir / f"{image_path.stem}.txt").exists()
    }
    skipped: List[Dict[str, object]] = []

    for image_path in sorted(original_dir.glob("*.jpg")):
        if not (original_dir / f"{image_path.stem}.txt").exists():
            skipped.append(
                {
                    "source_image_id": image_path.stem,
                    "reason": "image_without_matching_label",
                    "image_path": image_path.name,
                }
            )

    for label_path in sorted(original_dir.glob("*.txt")):
        if not (original_dir / f"{label_path.stem}.jpg").exists():
            skipped.append(
                {
                    "source_image_id": label_path.stem,
                    "reason": "label_without_matching_image",
                    "label_path": label_path.name,
                }
            )

    split_source = "random_80_10_10"
    try:
        split_ids = _build_authored_yolo_split_ids(object_detection_augmented_dir)
        split_source = "authored_augmented_dataset_split"
    except FileNotFoundError:
        split_ids = _build_random_yolo_split_ids(sorted(available_ids), seed=seed)

    split_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    assigned_ids: set[str] = set()

    for split_name, source_ids in split_ids.items():
        for source_id in sorted(source_ids, key=lambda value: int(value) if value.isdigit() else value):
            if source_id not in available_ids:
                skipped.append(
                    {
                        "source_image_id": source_id,
                        "reason": "split_source_id_missing_original_image_or_label",
                        "split": split_name,
                    }
                )
                continue

            label_path = original_dir / f"{source_id}.txt"
            annotations = _read_yolo_annotation(label_path)
            class_ids = [int(annotation["class_id"]) for annotation in annotations]
            class_counts = Counter(class_ids)

            split_rows[split_name].append(
                {
                    "split": split_name,
                    "source_image_id": source_id,
                    "image_filename": f"{source_id}.jpg",
                    "label_filename": f"{source_id}.txt",
                    "image_relative_path": f"Original Dataset/{source_id}.jpg",
                    "label_relative_path": f"Original Dataset/{source_id}.txt",
                    "box_count": len(annotations),
                    "present_yolo_class_ids": json.dumps(sorted(set(class_ids))),
                    "box_count_by_class_id": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
                }
            )
            assigned_ids.add(source_id)

    for source_id in sorted(available_ids - assigned_ids, key=lambda value: int(value) if value.isdigit() else value):
        skipped.append(
            {
                "source_image_id": source_id,
                "reason": "available_original_pair_not_assigned_to_split",
            }
        )

    return split_rows, skipped, split_source


def write_yolo_detection_split_artifacts(
    object_detection_original_dir: Path | str | None = None,
    object_detection_augmented_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    output_dir = _ensure_directory(Path(out_dir) if out_dir else YOLO_DETECTION_SPLITS_DIR)
    split_rows, skipped, split_source = build_yolo_detection_split_rows(
        object_detection_original_dir=object_detection_original_dir,
        object_detection_augmented_dir=object_detection_augmented_dir,
        seed=seed,
    )

    manifest_fieldnames = [
        "split",
        "source_image_id",
        "image_filename",
        "label_filename",
        "image_relative_path",
        "label_relative_path",
        "box_count",
        "present_yolo_class_ids",
        "box_count_by_class_id",
    ]

    summary = {
        "seed": seed,
        "image_size": list(IMAGE_SIZE),
        "split_source": split_source,
        "yolo_class_names": YOLO_CLASS_NAMES,
        "skipped_items": skipped,
        "splits": {},
    }

    all_rows: List[Dict[str, object]] = []
    for split_name, rows in split_rows.items():
        class_counts: Counter[int] = Counter()
        for row in rows:
            class_counts.update(json.loads(str(row["box_count_by_class_id"])))

        summary["splits"][split_name] = {
            "total_images": len(rows),
            "total_boxes": sum(int(row["box_count"]) for row in rows),
            "box_counts_by_class_id": dict(sorted((str(key), value) for key, value in class_counts.items())),
        }

        split_path = output_dir / f"{split_name}_manifest.csv"
        with split_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        all_rows.extend(rows)

    master_manifest_path = output_dir / "master_manifest.csv"
    with master_manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir / "yolo_detection_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_yolo_detection_dataset(
    dataset_dir: Path | str | None = None,
    split_dir: Path | str | None = None,
    object_detection_original_dir: Path | str | None = None,
) -> Path:
    output_dir = Path(dataset_dir) if dataset_dir else YOLO_DETECTION_DATASET_DIR
    split_directory = Path(split_dir) if split_dir else YOLO_DETECTION_SPLITS_DIR
    original_dir = resolve_object_detection_original_dir(object_detection_original_dir)

    for split_name in ("train", "val", "test"):
        rows = read_csv_rows(split_directory / f"{split_name}_manifest.csv")
        images_dir = _ensure_directory(output_dir / "images" / split_name)
        labels_dir = _ensure_directory(output_dir / "labels" / split_name)

        for row in rows:
            source_image = original_dir / str(row["image_filename"])
            source_label = original_dir / str(row["label_filename"])
            target_image = images_dir / str(row["image_filename"])
            target_label = labels_dir / str(row["label_filename"])
            shutil.copy2(source_image, target_image)
            shutil.copy2(source_label, target_label)

    data_yaml = output_dir / "data.yaml"
    names = [YOLO_CLASS_NAMES[index] for index in sorted(YOLO_CLASS_NAMES)]
    yaml_text = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(names)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(names)],
            "",
        ]
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml


def build_config_id(architecture: str, dropout: float, batch_size: int, dense_units: int) -> str:
    dropout_text = str(dropout).replace(".", "p")
    return f"{architecture}_drop_{dropout_text}_batch_{batch_size}_dense_{dense_units}"


def build_experiment_grid() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for architecture in ARCHITECTURES:
        architecture_rows: List[Dict[str, object]] = []
        for dropout in DROPOUT_VALUES:
            for batch_size in BATCH_SIZES:
                for dense_units in DENSE_UNITS:
                    architecture_rows.append(
                        {
                            "config_id": build_config_id(
                                architecture=architecture,
                                dropout=dropout,
                                batch_size=batch_size,
                                dense_units=dense_units,
                            ),
                            "architecture": architecture,
                            "dropout": dropout,
                            "batch_size": batch_size,
                            "dense_units": dense_units,
                            "epochs": EPOCHS,
                            "learning_rate": LEARNING_RATE,
                            "image_width": IMAGE_SIZE[0],
                            "image_height": IMAGE_SIZE[1],
                        }
                    )

        for index, row in enumerate(architecture_rows):
            row["part_id"] = (index % 3) + 1
            rows.append(row)

    return rows


def write_config_artifacts(out_dir: Path | str | None = None) -> Dict[str, object]:
    output_dir = _ensure_directory(Path(out_dir) if out_dir else CONFIGS_DIR)
    rows = build_experiment_grid()

    configs_path = output_dir / "experiment_configs.csv"
    with configs_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "config_id",
                "part_id",
                "architecture",
                "dropout",
                "batch_size",
                "dense_units",
                "epochs",
                "learning_rate",
                "image_width",
                "image_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    part_summary: Dict[str, object] = {
        "total_configs": len(rows),
        "parts": {},
    }
    for part_id in (1, 2, 3):
        part_rows = [row for row in rows if row["part_id"] == part_id]
        part_summary["parts"][str(part_id)] = {
            "total_configs": len(part_rows),
            "per_architecture": dict(Counter(row["architecture"] for row in part_rows)),
        }

        part_path = output_dir / f"part_{part_id}_configs.csv"
        with part_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(part_rows)

    summary_path = output_dir / "part_assignment_summary.json"
    summary_path.write_text(json.dumps(part_summary, indent=2), encoding="utf-8")
    return part_summary


def read_csv_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_split_rows(split_name: str, splits_dir: Path | str | None = None) -> List[Dict[str, str]]:
    directory = Path(splits_dir) if splits_dir else SPLITS_DIR
    return read_csv_rows(directory / f"{split_name}_manifest.csv")


def resolve_manifest_filepaths(
    rows: Sequence[Dict[str, object]],
    dataset_root: Path | str | None = None,
) -> List[Dict[str, object]]:
    resolved_root = resolve_dataset_root(dataset_root)
    resolved_rows: List[Dict[str, object]] = []

    for row in rows:
        resolved_row = dict(row)
        relative_path = str(resolved_row.get("relative_path", "")).strip()
        raw_filepath = str(resolved_row.get("filepath", "")).strip()

        if relative_path:
            normalized_relative_path = relative_path.replace("\\", "/")
            filepath = (resolved_root / Path(normalized_relative_path)).resolve()
        elif raw_filepath:
            filepath = Path(raw_filepath).resolve()
        else:
            raise KeyError("Each split row must include either `relative_path` or `filepath`.")

        if not filepath.exists():
            raise FileNotFoundError(f"Referenced image file does not exist: {filepath}")

        resolved_row["filepath"] = str(filepath)
        resolved_rows.append(resolved_row)

    return resolved_rows


def load_part_configs(part_id: int, configs_dir: Path | str | None = None) -> List[Dict[str, object]]:
    directory = Path(configs_dir) if configs_dir else CONFIGS_DIR
    rows = read_csv_rows(directory / f"part_{part_id}_configs.csv")
    parsed_rows: List[Dict[str, object]] = []
    for row in rows:
        parsed_rows.append(
            {
                "config_id": row["config_id"],
                "part_id": int(row["part_id"]),
                "architecture": row["architecture"],
                "dropout": float(row["dropout"]),
                "batch_size": int(row["batch_size"]),
                "dense_units": int(row["dense_units"]),
                "epochs": int(row["epochs"]),
                "learning_rate": float(row["learning_rate"]),
                "image_width": int(row["image_width"]),
                "image_height": int(row["image_height"]),
            }
        )
    return parsed_rows


def compute_inverse_frequency_class_weights(
    train_rows: Sequence[Dict[str, object]],
) -> Dict[int, float]:
    counts = Counter(int(row["class_index"]) for row in train_rows)
    total = sum(counts.values())
    num_classes = len(counts)
    # Square-root of inverse frequency produces softer weights than raw inverse
    # frequency. Raw inverse frequency creates a 4.3x spread between the rarest
    # and most common class (2.33 vs 0.54), which causes gradient spikes that
    # destabilise the small base CNN. Square-root reduces the spread to ~2.1x
    # (1.53 vs 0.73) while still satisfying the requirement to up-weight minority
    # classes. The class_weight argument is still passed to model.fit() as required.
    return {
        class_index: math.sqrt(total / (num_classes * class_count))
        for class_index, class_count in sorted(counts.items())
    }


def compute_multilabel_positive_weights(
    train_rows: Sequence[Dict[str, object]],
    label_columns: Sequence[str],
) -> Dict[str, float]:
    total = len(train_rows)
    weights: Dict[str, float] = {}

    for label_column in label_columns:
        positive_count = sum(int(float(row[label_column])) for row in train_rows)
        negative_count = total - positive_count
        if positive_count == 0 or negative_count == 0:
            weights[label_column] = 1.0
            continue

        # Same spirit as the single-label class weights, but adapted to
        # independent binary targets. The square root keeps rare labels from
        # overwhelming the loss on this small dataset.
        weights[label_column] = math.sqrt(negative_count / positive_count)

    return weights


def ensure_training_dependencies() -> Dict[str, object]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
    except ImportError as exc:
        raise ImportError(
            "Training utilities require tensorflow, pandas, numpy, matplotlib, and scikit-learn."
        ) from exc

    return {
        "plt": plt,
        "np": np,
        "pd": pd,
        "tf": tf,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "matthews_corrcoef": matthews_corrcoef,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
    }


def get_preprocessing_function(architecture: str):
    deps = ensure_training_dependencies()
    tf = deps["tf"]

    if architecture == "base_cnn":
        return lambda image: image / 255.0
    if architecture == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    if architecture == "densenet121":
        return tf.keras.applications.densenet.preprocess_input
    if architecture == "resnet34":
        try:
            from classification_models.tfkeras import Classifiers
        except ImportError as exc:
            raise ImportError(
                "ResNet34 transfer learning requires the `classification-models` package."
            ) from exc

        _, preprocess_input = Classifiers.get("resnet34")
        return preprocess_input

    raise ValueError(f"Unsupported architecture for preprocessing: {architecture}")


def build_data_generators(
    train_df,
    val_df,
    test_df,
    architecture: str,
    batch_size: int,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
):
    deps = ensure_training_dependencies()
    tf = deps["tf"]
    preprocessing_function = get_preprocessing_function(architecture)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=10,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function,
    )

    class_mode = "categorical"
    class_order = [folder for folder in sorted(INCLUDED_CLASS_FOLDERS)]

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="class_folder",
        classes=class_order,
        target_size=image_size,
        color_mode="rgb",
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_gen = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="class_folder",
        classes=class_order,
        target_size=image_size,
        color_mode="rgb",
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="class_folder",
        classes=class_order,
        target_size=image_size,
        color_mode="rgb",
        class_mode=class_mode,
        batch_size=1,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_multilabel_data_generators(
    train_df,
    val_df,
    test_df,
    architecture: str,
    batch_size: int,
    label_columns: Sequence[str],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
):
    deps = ensure_training_dependencies()
    tf = deps["tf"]
    preprocessing_function = get_preprocessing_function(architecture)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=10,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function,
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col=list(label_columns),
        target_size=image_size,
        color_mode="rgb",
        class_mode="raw",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_gen = eval_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col=list(label_columns),
        target_size=image_size,
        color_mode="rgb",
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col=list(label_columns),
        target_size=image_size,
        color_mode="rgb",
        class_mode="raw",
        batch_size=1,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_base_cnn_model(
    num_classes: int,
    dropout: float,
    dense_units: int,
    learning_rate: float,
    output_activation: str = "softmax",
    loss: str | object = "categorical_crossentropy",
    metrics: Sequence[object] | None = None,
):
    deps = ensure_training_dependencies()
    tf = deps["tf"]

    # he_normal initialisation is used instead of BatchNormalization.
    # BN with only ~19 batches/epoch (289 images, batch_size 16) causes its
    # running statistics to never stabilise, producing an exploding val_loss
    # while val_accuracy stays flat. he_normal solves the epoch-1 gradient
    # spike without creating a train/inference discrepancy.
    # Dropout is kept only in the dense head to prevent the original
    # dead-network collapse (loss ≈ 1.3863) that occurred when dropout was
    # stacked after every conv block.
    # Flatten on a 640x640 input after 3x MaxPool(2,2) produces 78x78x32 =
    # 194,688 features. Dense(64) alone then has 12.5M parameters for 289
    # training samples (43,000 params/sample) — the gradient is swamped and
    # training is mathematically impossible. GlobalAveragePooling2D reduces
    # each feature map to one scalar → 32 features total → Dense(64) has
    # 2,112 parameters, which is learnable. The three conv layers (8, 16, 32)
    # are unchanged.
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # GlobalAveragePooling2D replaces Flatten to avoid the 204,800->dense
            # bottleneck that caused the network to collapse (loss stuck at ln(4)=1.3863).
            # After 3x MaxPool(2,2) on 640x640, the spatial map is 80x80x32.
            # GAP compresses this to 32 values, a proportionate input for Dense(16/32/64).
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(dense_units, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation=output_activation),
        ]
    )

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=list(metrics) if metrics is not None else ["accuracy"],
    )
    return model


def build_transfer_model(
    architecture: str,
    num_classes: int,
    dropout: float,
    dense_units: int,
    learning_rate: float,
    output_activation: str = "softmax",
    loss: str | object = "categorical_crossentropy",
    metrics: Sequence[object] | None = None,
):
    deps = ensure_training_dependencies()
    tf = deps["tf"]

    if architecture == "vgg16":
        backbone = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=INPUT_SHAPE,
        )
    elif architecture == "densenet121":
        backbone = tf.keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=INPUT_SHAPE,
        )
    elif architecture == "resnet34":
        try:
            from classification_models.tfkeras import Classifiers
        except ImportError as exc:
            raise ImportError(
                "ResNet34 transfer learning requires the `image-classifiers` package."
            ) from exc

        ResNet34, _ = Classifiers.get("resnet34")
        backbone = ResNet34(
            input_shape=INPUT_SHAPE,
            weights="imagenet",
            include_top=False,
        )
    else:
        raise ValueError(f"Unsupported transfer architecture: {architecture}")

    backbone.trainable = False

    # Build as a functional model so that the `training` flag is forwarded
    # correctly to Dropout layers (backbone stays in inference mode via
    # `training=False`, while the dense-head dropout activates during fit).
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=output_activation)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_transfer")

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=list(metrics) if metrics is not None else ["accuracy"],
    )
    return model


def build_model(config: Dict[str, object], num_classes: int):
    architecture = str(config["architecture"])
    if architecture == "base_cnn":
        return build_base_cnn_model(
            num_classes=num_classes,
            dropout=float(config["dropout"]),
            dense_units=int(config["dense_units"]),
            learning_rate=float(config["learning_rate"]),
        )

    return build_transfer_model(
        architecture=architecture,
        num_classes=num_classes,
        dropout=float(config["dropout"]),
        dense_units=int(config["dense_units"]),
        learning_rate=float(config["learning_rate"]),
    )


def build_multilabel_weighted_binary_crossentropy(
    positive_weights: Dict[str, float],
    label_columns: Sequence[str],
):
    deps = ensure_training_dependencies()
    tf = deps["tf"]
    weight_values = [float(positive_weights[label_column]) for label_column in label_columns]
    weights_tensor = tf.constant(weight_values, dtype=tf.float32)

    def weighted_binary_crossentropy(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        label_weights = 1.0 + y_true * (weights_tensor - 1.0)
        return tf.reduce_mean(bce * label_weights, axis=-1)

    weighted_binary_crossentropy.__name__ = "weighted_binary_crossentropy"
    return weighted_binary_crossentropy


def build_multilabel_model(
    config: Dict[str, object],
    label_columns: Sequence[str],
    positive_weights: Dict[str, float],
):
    deps = ensure_training_dependencies()
    tf = deps["tf"]
    architecture = str(config["architecture"])
    loss = build_multilabel_weighted_binary_crossentropy(positive_weights, label_columns)
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    num_labels = len(label_columns)

    if architecture == "base_cnn":
        return build_base_cnn_model(
            num_classes=num_labels,
            dropout=float(config["dropout"]),
            dense_units=int(config["dense_units"]),
            learning_rate=float(config["learning_rate"]),
            output_activation="sigmoid",
            loss=loss,
            metrics=metrics,
        )

    return build_transfer_model(
        architecture=architecture,
        num_classes=num_labels,
        dropout=float(config["dropout"]),
        dense_units=int(config["dense_units"]),
        learning_rate=float(config["learning_rate"]),
        output_activation="sigmoid",
        loss=loss,
        metrics=metrics,
    )


def _compute_macro_specificity(conf_matrix) -> float:
    import numpy as np

    total = np.sum(conf_matrix)
    specificities: List[float] = []
    for class_index in range(conf_matrix.shape[0]):
        tp = conf_matrix[class_index, class_index]
        fp = np.sum(conf_matrix[:, class_index]) - tp
        fn = np.sum(conf_matrix[class_index, :]) - tp
        tn = total - tp - fp - fn
        denominator = tn + fp
        specificities.append(float(tn / denominator) if denominator else 0.0)
    return float(sum(specificities) / len(specificities))


def _compute_multilabel_macro_specificity(y_true, y_pred) -> float:
    import numpy as np

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    specificities: List[float] = []

    for label_index in range(y_true_array.shape[1]):
        true_label = y_true_array[:, label_index]
        pred_label = y_pred_array[:, label_index]
        true_negative = np.sum((true_label == 0) & (pred_label == 0))
        false_positive = np.sum((true_label == 0) & (pred_label == 1))
        denominator = true_negative + false_positive
        specificities.append(float(true_negative / denominator) if denominator else 0.0)

    return float(sum(specificities) / len(specificities))


def train_single_config(
    config: Dict[str, object],
    part_id: int,
    outputs_dir: Path | str | None = None,
    splits_dir: Path | str | None = None,
    dataset_root: Path | str | None = None,
    seed: int = DEFAULT_SEED,
):
    deps = ensure_training_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    tf = deps["tf"]
    accuracy_score = deps["accuracy_score"]
    confusion_matrix = deps["confusion_matrix"]
    f1_score = deps["f1_score"]
    matthews_corrcoef = deps["matthews_corrcoef"]
    precision_score = deps["precision_score"]
    recall_score = deps["recall_score"]
    roc_auc_score = deps["roc_auc_score"]

    tf.keras.utils.set_random_seed(seed)

    split_directory = Path(splits_dir) if splits_dir else SPLITS_DIR
    train_df = pd.DataFrame(
        resolve_manifest_filepaths(load_split_rows("train", split_directory), dataset_root=dataset_root)
    )
    val_df = pd.DataFrame(
        resolve_manifest_filepaths(load_split_rows("val", split_directory), dataset_root=dataset_root)
    )
    test_df = pd.DataFrame(
        resolve_manifest_filepaths(load_split_rows("test", split_directory), dataset_root=dataset_root)
    )

    class_weights = compute_inverse_frequency_class_weights(train_df.to_dict("records"))
    train_gen, val_gen, test_gen = build_data_generators(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        architecture=str(config["architecture"]),
        batch_size=int(config["batch_size"]),
        image_size=(int(config["image_width"]), int(config["image_height"])),
        seed=seed,
    )

    model = build_model(config, num_classes=len(INCLUDED_CLASS_FOLDERS))
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=int(config["epochs"]),
        class_weight=class_weights,
        verbose=1,
    )

    probabilities = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes
    y_pred = np.argmax(probabilities, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "config_id": str(config["config_id"]),
        "part_id": int(part_id),
        "architecture": str(config["architecture"]),
        "dropout": float(config["dropout"]),
        "batch_size": int(config["batch_size"]),
        "dense_units": int(config["dense_units"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(_compute_macro_specificity(conf_matrix)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc_macro_ovr": float(
            roc_auc_score(
                tf.keras.utils.to_categorical(y_true, num_classes=len(INCLUDED_CLASS_FOLDERS)),
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        ),
    }

    output_root = _ensure_directory(resolve_outputs_dir(outputs_dir))
    run_dir = _ensure_directory(output_root / f"part_{part_id}" / str(config["config_id"]))
    model_artifacts = save_model_artifacts(
        model,
        run_dir,
        metadata={
            "task_type": "single_label_classification",
            "class_names": [INCLUDED_CLASS_FOLDERS[folder] for folder in sorted(INCLUDED_CLASS_FOLDERS)],
        },
    )
    metrics["saved_artifacts"] = model_artifacts

    save_history_artifacts(history, run_dir)
    save_auc_artifacts(
        y_true=y_true,
        probabilities=probabilities,
        run_dir=run_dir,
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "class_weights.json").write_text(json.dumps(class_weights, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(run_dir / "history.csv", index=False)

    metrics_frame = pd.DataFrame([metrics])
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    return metrics


def _safe_roc_auc_score(roc_auc_score, y_true, probabilities, **kwargs):
    try:
        value = float(roc_auc_score(y_true, probabilities, **kwargs))
    except ValueError:
        return None
    return None if math.isnan(value) else value


def train_single_multilabel_config(
    config: Dict[str, object],
    part_id: int,
    outputs_dir: Path | str | None = None,
    splits_dir: Path | str | None = None,
    dataset_root: Path | str | None = None,
    threshold: float = 0.5,
    seed: int = DEFAULT_SEED,
):
    deps = ensure_training_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    tf = deps["tf"]
    accuracy_score = deps["accuracy_score"]
    f1_score = deps["f1_score"]
    matthews_corrcoef = deps["matthews_corrcoef"]
    precision_score = deps["precision_score"]
    recall_score = deps["recall_score"]
    roc_auc_score = deps["roc_auc_score"]

    tf.keras.utils.set_random_seed(seed)

    split_directory = Path(splits_dir) if splits_dir else MULTILABEL_SPLITS_DIR
    raw_train_rows = load_split_rows("train", split_directory)
    raw_val_rows = load_split_rows("val", split_directory)
    raw_test_rows = load_split_rows("test", split_directory)
    label_columns = [column for column in raw_train_rows[0] if _is_multilabel_target_column(column)]

    train_df = pd.DataFrame(resolve_manifest_filepaths(raw_train_rows, dataset_root=dataset_root))
    val_df = pd.DataFrame(resolve_manifest_filepaths(raw_val_rows, dataset_root=dataset_root))
    test_df = pd.DataFrame(resolve_manifest_filepaths(raw_test_rows, dataset_root=dataset_root))

    for frame in (train_df, val_df, test_df):
        for label_column in label_columns:
            frame[label_column] = frame[label_column].astype("float32")

    positive_weights = compute_multilabel_positive_weights(
        train_df.to_dict("records"),
        label_columns=label_columns,
    )
    train_gen, val_gen, test_gen = build_multilabel_data_generators(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        architecture=str(config["architecture"]),
        batch_size=int(config["batch_size"]),
        label_columns=label_columns,
        image_size=(int(config["image_width"]), int(config["image_height"])),
        seed=seed,
    )

    model = build_multilabel_model(
        config=config,
        label_columns=label_columns,
        positive_weights=positive_weights,
    )
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=int(config["epochs"]),
        verbose=1,
    )

    probabilities = model.predict(test_gen, verbose=1)
    y_true = test_df[label_columns].astype("int32").to_numpy()
    y_pred = (probabilities >= threshold).astype("int32")

    auc_macro = _safe_roc_auc_score(roc_auc_score, y_true, probabilities, average="macro")
    auc_micro = _safe_roc_auc_score(roc_auc_score, y_true, probabilities, average="micro")

    metrics = {
        "config_id": str(config["config_id"]),
        "part_id": int(part_id),
        "architecture": str(config["architecture"]),
        "dropout": float(config["dropout"]),
        "batch_size": int(config["batch_size"]),
        "dense_units": int(config["dense_units"]),
        "num_labels": len(label_columns),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "label_accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "specificity_macro": float(_compute_multilabel_macro_specificity(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true.ravel(), y_pred.ravel())),
        "auc_macro_ovr": auc_macro,
        "auc_micro_ovr": auc_micro,
    }

    output_root = _ensure_directory(resolve_outputs_dir(outputs_dir))
    run_dir = _ensure_directory(output_root / f"part_{part_id}" / str(config["config_id"]))
    model_artifacts = save_model_artifacts(
        model,
        run_dir,
        metadata={
            "task_type": "multilabel_classification",
            "label_columns": list(label_columns),
            "threshold": float(threshold),
        },
    )
    metrics["saved_artifacts"] = model_artifacts

    save_history_artifacts(history, run_dir)
    save_multilabel_auc_artifacts(
        y_true=y_true,
        probabilities=probabilities,
        label_columns=label_columns,
        run_dir=run_dir,
    )

    predictions_frame = test_df[["source_image_id", "filename", "relative_path", *label_columns]].copy()
    for label_index, label_column in enumerate(label_columns):
        suffix = label_column.replace("label_", "")
        predictions_frame[f"prob_{suffix}"] = probabilities[:, label_index]
        predictions_frame[f"pred_{suffix}"] = y_pred[:, label_index]

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "positive_class_weights.json").write_text(
        json.dumps(positive_weights, indent=2),
        encoding="utf-8",
    )
    (run_dir / "class_weights.json").write_text(json.dumps(positive_weights, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(run_dir / "history.csv", index=False)

    metrics_frame = pd.DataFrame([metrics])
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame.to_csv(run_dir / "test_predictions.csv", index=False)
    return metrics


def save_history_artifacts(history, run_dir: Path | str):
    deps = ensure_training_dependencies()
    plt = deps["plt"]

    output_dir = _ensure_directory(Path(run_dir))
    figure = plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss Evolution")

    plt.subplot(2, 1, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Evolution")

    figure.tight_layout()
    figure.savefig(output_dir / "loss_accuracy_curves.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_model_artifacts(model, run_dir: Path | str, metadata: Dict[str, object] | None = None) -> Dict[str, str]:
    output_dir = _ensure_directory(Path(run_dir))

    model_path = output_dir / "model.keras"
    weights_path = output_dir / "model.weights.h5"
    metadata_path = output_dir / "model_artifacts.json"

    model.save(model_path)
    model.save_weights(weights_path)

    artifact_metadata = {
        "model_path": str(model_path),
        "weights_path": str(weights_path),
        "metadata_path": str(metadata_path),
        "format": "keras_v3",
    }
    if metadata:
        artifact_metadata.update(metadata)

    metadata_path.write_text(json.dumps(artifact_metadata, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "weights_path": str(weights_path),
        "metadata_path": str(metadata_path),
    }


def save_auc_artifacts(y_true, probabilities, run_dir: Path | str):
    deps = ensure_training_dependencies()
    plt = deps["plt"]
    tf = deps["tf"]
    roc_auc_score = deps["roc_auc_score"]
    roc_curve = deps["roc_curve"]

    output_dir = _ensure_directory(Path(run_dir))
    class_names = [INCLUDED_CLASS_FOLDERS[folder] for folder in sorted(INCLUDED_CLASS_FOLDERS)]
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))

    figure = plt.figure(figsize=(10, 8))
    auc_summary = {}
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, class_index], probabilities[:, class_index])
        auc_value = roc_auc_score(y_true_one_hot[:, class_index], probabilities[:, class_index])
        auc_summary[class_name] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_value:.4f})")

    micro_fpr, micro_tpr, _ = roc_curve(y_true_one_hot.ravel(), probabilities.ravel())
    macro_auc = roc_auc_score(y_true_one_hot, probabilities, multi_class="ovr", average="macro")
    micro_auc = roc_auc_score(y_true_one_hot.ravel(), probabilities.ravel())
    auc_summary["macro_ovr"] = float(macro_auc)
    auc_summary["micro_ovr"] = float(micro_auc)

    plt.plot(micro_fpr, micro_tpr, linestyle="--", label=f"Micro-average (AUC={micro_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "auc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    (output_dir / "auc_summary.json").write_text(json.dumps(auc_summary, indent=2), encoding="utf-8")


def save_multilabel_auc_artifacts(y_true, probabilities, label_columns: Sequence[str], run_dir: Path | str):
    deps = ensure_training_dependencies()
    plt = deps["plt"]
    roc_auc_score = deps["roc_auc_score"]
    roc_curve = deps["roc_curve"]

    output_dir = _ensure_directory(Path(run_dir))
    class_name_by_column = {
        _multilabel_column_name(class_folder): ALL_CLASS_FOLDERS[class_folder]
        for class_folder in ALL_CLASS_FOLDERS
    }

    figure = plt.figure(figsize=(10, 8))
    auc_summary = {}
    for label_index, label_column in enumerate(label_columns):
        true_label = y_true[:, label_index]
        probability = probabilities[:, label_index]
        class_name = class_name_by_column.get(label_column, label_column)

        if len(set(true_label.tolist())) < 2:
            auc_summary[class_name] = None
            continue

        fpr, tpr, _ = roc_curve(true_label, probability)
        auc_value = roc_auc_score(true_label, probability)
        auc_summary[class_name] = float(auc_value)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_value:.4f})")

    micro_auc = _safe_roc_auc_score(roc_auc_score, y_true.ravel(), probabilities.ravel())
    if micro_auc is not None:
        micro_fpr, micro_tpr, _ = roc_curve(y_true.ravel(), probabilities.ravel())
        plt.plot(micro_fpr, micro_tpr, linestyle="--", label=f"Micro-average (AUC={micro_auc:.4f})")

    macro_auc = _safe_roc_auc_score(roc_auc_score, y_true, probabilities, average="macro")
    auc_summary["macro_ovr"] = macro_auc
    auc_summary["micro_ovr"] = micro_auc

    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multilabel ROC Curve")
    if any(value is not None for value in auc_summary.values()):
        plt.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "auc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    (output_dir / "auc_summary.json").write_text(json.dumps(auc_summary, indent=2), encoding="utf-8")


def aggregate_part_results(outputs_dir: Path | str | None = None, part_id: int | None = None):
    deps = ensure_training_dependencies()
    pd = deps["pd"]

    root = resolve_outputs_dir(outputs_dir)
    run_roots: Iterable[Path]
    if part_id is not None:
        run_roots = (root / f"part_{part_id}",)
    else:
        run_roots = sorted(path for path in root.glob("part_*") if path.is_dir())

    records = []
    for part_root in run_roots:
        if not part_root.exists():
            continue
        for metrics_path in part_root.glob("*/metrics.json"):
            records.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["part_id", "config_id"]).reset_index(drop=True)


def load_worker_configs(worker_id: int, configs_dir: Path | str | None = None) -> List[Dict[str, object]]:
    return load_part_configs(part_id=worker_id, configs_dir=configs_dir)


def aggregate_worker_results(outputs_dir: Path | str | None = None, worker_id: int | None = None):
    return aggregate_part_results(outputs_dir=outputs_dir, part_id=worker_id)
