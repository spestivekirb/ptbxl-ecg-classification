"""PTB-XL metadata preparation, dataset construction, and data loaders."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import ECGDataset
from src.preprocessing import compute_dataset_stats


DIAGNOSTIC_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


@dataclass(frozen=True)
class DataFrameSplits:
    """Metadata rows assigned to the official PTB-XL folds."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class NormalizationStats:
    """Per-lead statistics calculated from training recordings only."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    """Datasets and their shared training-derived normalization statistics."""

    train: ECGDataset
    val: ECGDataset
    test: ECGDataset
    stats: NormalizationStats


@dataclass(frozen=True)
class DataLoaderBundle:
    """Training, validation, and test data loaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _parse_scp_codes(value: object) -> dict[str, float]:
    """Parse one serialized SCP-code dictionary from the PTB-XL metadata."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Expected SCP codes as a string or dict, got {type(value)!r}")

    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("An scp_codes value did not contain a dictionary")
    return parsed


def encode_labels(
    labels: Sequence[str],
    classes: Sequence[str] = DIAGNOSTIC_CLASSES,
) -> list[int]:
    """Encode diagnostic class names as a fixed-order multi-hot vector."""
    label_set = set(labels)
    return [int(class_name in label_set) for class_name in classes]


def load_ptbxl_metadata(
    data_dir: str | Path,
    classes: Sequence[str] = DIAGNOSTIC_CLASSES,
) -> pd.DataFrame:
    """Load PTB-XL metadata and add diagnostic classes and encoded labels.

    Only records with at least one of ``classes`` are returned. SCP-code
    likelihood values are intentionally not thresholded, matching the PTB-XL
    diagnostic-superclass benchmark setup used in the prototype notebook.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "ptbxl_database.csv"
    statements_path = data_dir / "scp_statements.csv"

    if not metadata_path.is_file() or not statements_path.is_file():
        raise FileNotFoundError(
            "Expected ptbxl_database.csv and scp_statements.csv in "
            f"{data_dir}"
        )

    metadata = pd.read_csv(metadata_path)
    statements = pd.read_csv(statements_path, index_col=0)
    required_metadata = {"scp_codes", "strat_fold", "filename_lr"}
    missing_metadata = required_metadata.difference(metadata.columns)
    if missing_metadata:
        raise ValueError(
            "PTB-XL metadata is missing columns: "
            + ", ".join(sorted(missing_metadata))
        )
    if not {"diagnostic", "diagnostic_class"}.issubset(statements.columns):
        raise ValueError(
            "scp_statements.csv must contain diagnostic and diagnostic_class"
        )

    class_order = tuple(classes)
    if len(class_order) == 0 or len(set(class_order)) != len(class_order):
        raise ValueError("classes must contain unique class names")

    diagnostic_map = (
        statements.loc[statements["diagnostic"].eq(1), "diagnostic_class"]
        .dropna()
        .to_dict()
    )
    allowed_classes = set(class_order)

    def diagnostic_classes(codes: dict[str, float]) -> list[str]:
        found = {
            diagnostic_map[code]
            for code in codes
            if code in diagnostic_map and diagnostic_map[code] in allowed_classes
        }
        return [class_name for class_name in class_order if class_name in found]

    prepared = metadata.copy()
    prepared["scp_codes_parsed"] = prepared["scp_codes"].map(_parse_scp_codes)
    prepared["diagnostic_classes"] = prepared["scp_codes_parsed"].map(
        diagnostic_classes
    )
    prepared = prepared.loc[prepared["diagnostic_classes"].map(bool)].copy()
    prepared["labels"] = prepared["diagnostic_classes"].map(
        lambda labels: encode_labels(labels, class_order)
    )
    return prepared


def split_ptbxl_metadata(metadata: pd.DataFrame) -> DataFrameSplits:
    """Split prepared metadata using PTB-XL folds 1-8, 9, and 10."""
    if "strat_fold" not in metadata.columns:
        raise ValueError("metadata must contain a strat_fold column")
    if "labels" not in metadata.columns:
        raise ValueError("metadata must contain encoded labels")

    folds = set(metadata["strat_fold"].dropna().astype(int).unique())
    unexpected_folds = folds.difference(range(1, 11))
    if unexpected_folds:
        raise ValueError(f"Unexpected PTB-XL folds: {sorted(unexpected_folds)}")

    return DataFrameSplits(
        train=metadata.loc[metadata["strat_fold"].between(1, 8)].copy(),
        val=metadata.loc[metadata["strat_fold"].eq(9)].copy(),
        test=metadata.loc[metadata["strat_fold"].eq(10)].copy(),
    )


def prepare_ptbxl_splits(
    data_dir: str | Path,
    classes: Sequence[str] = DIAGNOSTIC_CLASSES,
) -> DataFrameSplits:
    """Load, label, filter, and split PTB-XL metadata."""
    return split_ptbxl_metadata(load_ptbxl_metadata(data_dir, classes))


def compute_pos_weights(
    metadata: pd.DataFrame,
    power: float = 1.0,
) -> np.ndarray:
    """Calculate BCE positive-class weights from encoded training labels.

    ``power=1`` returns the standard negative-to-positive ratio. Values below
    one, such as ``0.5``, soften the weighting when full inverse-frequency
    weighting over-corrects a rare class.
    """
    if "labels" not in metadata.columns or len(metadata) == 0:
        raise ValueError("metadata must contain at least one encoded label row")
    if not 0 <= power <= 1:
        raise ValueError("power must be between 0 and 1")

    labels = np.asarray(metadata["labels"].tolist(), dtype=float)
    if labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("labels must form a non-empty two-dimensional array")
    positives = labels.sum(axis=0)
    if np.any(positives == 0):
        raise ValueError("Every class must have at least one positive example")
    negatives = labels.shape[0] - positives
    return np.power(negatives / positives, power)


def build_datasets(
    splits: DataFrameSplits,
    data_dir: str | Path,
    stats: NormalizationStats | None = None,
) -> DatasetBundle:
    """Build normalized lazy-loading datasets for each data split.

    When ``stats`` is omitted, per-lead statistics are computed exclusively
    from the training split.
    """
    data_dir = Path(data_dir)
    if stats is None:
        filepaths = [data_dir / name for name in splits.train["filename_lr"]]
        mean, std = compute_dataset_stats(filepaths)
        stats = NormalizationStats(mean=mean, std=std)

    return DatasetBundle(
        train=ECGDataset(splits.train, data_dir, stats.mean, stats.std),
        val=ECGDataset(splits.val, data_dir, stats.mean, stats.std),
        test=ECGDataset(splits.test, data_dir, stats.mean, stats.std),
        stats=stats,
    )


def build_dataloaders(
    datasets: DatasetBundle,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoaderBundle:
    """Create consistently configured loaders for all three data splits."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return DataLoaderBundle(
        train=DataLoader(datasets.train, shuffle=True, **common),
        val=DataLoader(datasets.val, shuffle=False, **common),
        test=DataLoader(datasets.test, shuffle=False, **common),
    )
