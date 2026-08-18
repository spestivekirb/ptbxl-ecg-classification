"""Waveform loading and normalization utilities."""

from pathlib import Path

import numpy as np
import wfdb


def load_ecg(filepath: str | Path) -> np.ndarray:
    """
    Load a PTB-XL ECG recording.

    Returns:
        ECG signal with shape (12, 1000)
    """

    # This function loads raw ECG waveform data from a recording file.
    signal, _ = wfdb.rdsamp(str(filepath))
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2D ECG signal, received shape {signal.shape}")
    return signal.T


def normalize_signal(
    signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Standardize each ECG lead using training-set statistics.

    Args:
        signal: ECG array with shape (12, 1000).
        mean: Per-lead training mean with shape (12,).
        std: Per-lead training standard deviation with shape (12,).

    Returns:
        Standardized ECG array with shape (12, 1000).
    """
    signal = np.asarray(signal)
    mean = np.asarray(mean)
    std = np.asarray(std)
    if signal.ndim != 2:
        raise ValueError("signal must have shape (leads, samples)")
    if mean.shape != (signal.shape[0],) or std.shape != (signal.shape[0],):
        raise ValueError("mean and std must contain one value per ECG lead")
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("std must contain positive finite values")
    return (signal - mean[:, None]) / std[:, None]


def compute_dataset_stats(filepaths) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-lead mean and standard deviation across ECG recordings.

    Args:
        filepaths: Iterable of PTB-XL record paths.

    Returns:
        mean: Per-lead mean with shape (12,).
        std: Per-lead standard deviation with shape (12,).
    """
    signal_sum = None
    signal_squared_sum = None
    n_samples = 0

    for filepath in filepaths:
        signal = load_ecg(filepath).astype(np.float64, copy=False)

        if signal_sum is None:
            signal_sum = np.zeros(signal.shape[0], dtype=np.float64)
            signal_squared_sum = np.zeros(signal.shape[0], dtype=np.float64)
        elif signal.shape[0] != signal_sum.shape[0]:
            raise ValueError("All ECG records must contain the same number of leads")

        signal_sum += signal.sum(axis=1)
        signal_squared_sum += (signal ** 2).sum(axis=1)

        n_samples += signal.shape[1]

    if n_samples == 0 or signal_sum is None or signal_squared_sum is None:
        raise ValueError("Cannot compute statistics from an empty collection")

    mean = signal_sum / n_samples
    variance = np.maximum(signal_squared_sum / n_samples - mean ** 2, 0.0)
    std = np.sqrt(variance)

    if np.any(std == 0):
        raise ValueError("At least one ECG lead has zero variance")

    return mean, std
