"""
Supervised DNN classifier for fraud detection.

This is the SECONDARY model — used as an internal comparison point
against the autoencoder-based approach. It uses the fraud labels during
training, which the autoencoder does not.
"""

import torch.nn as nn


class FraudDetectorDNN(nn.Module):
    """
    Deep Neural Network for binary fraud classification.

    Architecture:  Input → [FC → BatchNorm → ReLU → Dropout] × N → FC (logit)

    Notes:
      - Outputs a raw logit (not a sigmoid probability).
        Use BCEWithLogitsLoss for numerical stability.
      - Apply torch.sigmoid() at inference time to get probabilities.
      - Pair with `pos_weight` in the loss to handle class imbalance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        # Output layer — single logit
        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
