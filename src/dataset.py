"""
PyTorch Dataset classes for fraud detection.

- FraudDataset:      returns (features, label) — used for supervised DNN
- NormalOnlyDataset: returns (features, features) — used for autoencoder training
                     (trained to reconstruct normal transactions only)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """Standard supervised dataset — features and binary labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NormalOnlyDataset(Dataset):
    """
    Autoencoder training dataset.

    Contains ONLY normal (non-fraud) transactions. The target equals the input,
    because the autoencoder is trained to reconstruct normal patterns. At
    inference, the model will produce high reconstruction error on fraud —
    which is how we detect anomalies.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        if labels is not None:
            # Filter to normal transactions only (label == 0)
            normal_mask = labels == 0
            features = features[normal_mask]
            print(f"  NormalOnlyDataset: kept {len(features)} normal samples "
                  f"(dropped {(~normal_mask).sum()} fraud samples)")

        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Input and target are identical — autoencoder reconstructs its input
        x = self.features[idx]
        return x, x
