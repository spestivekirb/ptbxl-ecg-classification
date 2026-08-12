# Imports
import numpy as np
import wfdb

def load_ecg(filepath):
    """
    Load a PTB-XL ECG recording.

    Returns:
        ECG signal with shape (12, 1000)
    """

    # This function loads raw ECG waveform data from a recording file.
    signal, metadata = wfdb.rdsamp(filepath)

    signal = signal.T

    return signal


def normalize_signal(signal, mean, std):
    """
    Standardize each ECG lead using training-set statistics.

    Args:
        signal: ECG array with shape (12, 1000).
        mean: Per-lead training mean with shape (12,).
        std: Per-lead training standard deviation with shape (12,).

    Returns:
        Standardized ECG array with shape (12, 1000).
    """
    return (signal - mean[:, None]) / std[:, None]


def compute_dataset_stats(filepaths):
    """
    Compute per-lead mean and standard deviation across ECG recordings.

    Args:
        filepaths: Iterable of PTB-XL record paths.

    Returns:
        mean: Per-lead mean with shape (12,).
        std: Per-lead standard deviation with shape (12,).
    """
    signal_sum = np.zeros(12, dtype=np.float64)
    signal_squared_sum = np.zeros(12, dtype=np.float64)
    n_samples = 0

    for filepath in filepaths:
        signal = load_ecg(str(filepath))

        signal_sum += signal.sum(axis=1)
        signal_squared_sum += (signal ** 2).sum(axis=1)

        n_samples += signal.shape[1]

    mean = signal_sum / n_samples
    variance = signal_squared_sum / n_samples - mean ** 2
    std = np.sqrt(variance)

    return mean, std