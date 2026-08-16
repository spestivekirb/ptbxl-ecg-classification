import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import load_ecg, normalize_signal


class ECGDataset(Dataset):
    def __init__(self, dataframe, data_dir, mean, std):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filepath = self.data_dir / row["filename_lr"]
        signal = load_ecg(str(filepath))
        signal = normalize_signal(signal, self.mean, self.std)

        signal = torch.tensor(signal, dtype=torch.float32)
        labels = torch.tensor(
            np.asarray(row["labels"], dtype=np.float32)
        )

        return signal, labels