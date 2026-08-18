"""Inference, threshold selection, and multi-label evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn

from src.data import DIAGNOSTIC_CLASSES


@dataclass(frozen=True)
class PredictionResult:
    """Ground-truth targets and predicted probabilities."""

    labels: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """Predictions and metric tables for a multi-label evaluation."""

    predictions: np.ndarray
    per_class: pd.DataFrame
    overall: pd.DataFrame
    threshold: np.ndarray


def predict_probabilities(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> PredictionResult:
    """Run batched inference and return labels and sigmoid probabilities."""
    model.eval()
    probability_batches = []
    label_batches = []

    with torch.inference_mode():
        for signals, labels in data_loader:
            logits = model(signals.to(device))
            probability_batches.append(torch.sigmoid(logits).cpu())
            label_batches.append(labels.cpu())

    if not probability_batches:
        raise ValueError("data_loader did not yield any examples")

    return PredictionResult(
        labels=torch.cat(label_batches).numpy(),
        probabilities=torch.cat(probability_batches).numpy(),
    )


def _validate_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("labels and probabilities must be two-dimensional")
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have identical shapes")
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("labels and probabilities cannot be empty")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    return labels.astype(int, copy=False), probabilities.astype(float, copy=False)


def _threshold_array(threshold: float | Sequence[float], n_classes: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=float)
    if values.ndim == 0:
        values = np.full(n_classes, values.item())
    if values.shape != (n_classes,):
        raise ValueError("threshold must be a scalar or contain one value per class")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("threshold values must be between 0 and 1")
    return values


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_multilabel(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str] = DIAGNOSTIC_CLASSES,
    threshold: float | Sequence[float] = 0.5,
) -> EvaluationResult:
    """Calculate per-class and aggregate metrics for multi-label predictions."""
    labels, probabilities = _validate_arrays(labels, probabilities)
    names = tuple(class_names)
    if len(names) != labels.shape[1] or len(set(names)) != len(names):
        raise ValueError("class_names must uniquely name every target column")

    thresholds = _threshold_array(threshold, labels.shape[1])
    predictions = (probabilities >= thresholds[None, :]).astype(int)
    metric_rows = []

    for index, class_name in enumerate(names):
        true_values = labels[:, index]
        predicted_values = predictions[:, index]
        tn, fp, fn, tp = confusion_matrix(
            true_values, predicted_values, labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0.0
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        )
        metric_rows.append(
            {
                "Class": class_name,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Precision": precision,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "F1": f1,
                "AUROC": _safe_auroc(true_values, probabilities[:, index]),
            }
        )

    per_class = pd.DataFrame(metric_rows).set_index("Class")
    macro_auroc = float(np.nanmean(per_class["AUROC"]))
    micro_auroc = _safe_auroc(labels.ravel(), probabilities.ravel())
    overall = pd.DataFrame(
        {
            "Score": [
                f1_score(labels, predictions, average="macro", zero_division=0),
                f1_score(labels, predictions, average="micro", zero_division=0),
                macro_auroc,
                micro_auroc,
            ]
        },
        index=["Macro F1", "Micro F1", "Macro AUROC", "Micro AUROC"],
    )
    return EvaluationResult(predictions, per_class, overall, thresholds)


def optimize_f1_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    candidates: Sequence[float] | None = None,
) -> np.ndarray:
    """Select a per-class threshold that maximizes validation F1.

    Ties are resolved in favor of the threshold closest to 0.5, which keeps
    threshold changes conservative when several candidates perform equally.
    """
    labels, probabilities = _validate_arrays(labels, probabilities)
    grid = np.asarray(
        candidates if candidates is not None else np.linspace(0.05, 0.95, 19),
        dtype=float,
    )
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("candidates must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(grid)) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("candidate thresholds must be between 0 and 1")

    selected = np.empty(labels.shape[1], dtype=float)
    for index in range(labels.shape[1]):
        scores = np.asarray(
            [
                f1_score(
                    labels[:, index],
                    probabilities[:, index] >= threshold,
                    zero_division=0,
                )
                for threshold in grid
            ]
        )
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        selected[index] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return selected
