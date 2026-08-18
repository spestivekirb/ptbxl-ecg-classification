"""PyTorch dataset for lazily loaded PTB-XL waveforms."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import load_ecg, normalize_signal


class ECGDataset(Dataset):
    def __init__(self, dataframe, data_dir, mean, std):
        required_columns = {"filename_lr", "labels"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "dataframe is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        self.df = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filepath = self.data_dir / row["filename_lr"]
        signal = load_ecg(str(filepath))
        signal = normalize_signal(signal, self.mean, self.std)

        signal = torch.tensor(signal, dtype=torch.float32)
        labels = torch.as_tensor(row["labels"], dtype=torch.float32)

        return signal, labels
