from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dental OPG XRAY Dataset" / "Dental OPG XRAY Dataset" / "Dental OPG (Classification)"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
CONFIGS_DIR = ARTIFACTS_DIR / "configs"
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
    return {
        class_index: total / (num_classes * class_count)
        for class_index, class_count in sorted(counts.items())
    }


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


def build_base_cnn_model(num_classes: int, dropout: float, dense_units: int, learning_rate: float):
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
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(
    architecture: str,
    num_classes: int,
    dropout: float,
    dense_units: int,
    learning_rate: float,
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
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_transfer")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
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
