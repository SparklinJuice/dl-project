"""
Autoencoder architectures for unsupervised anomaly detection.

The main idea:
  1. Train an autoencoder to reconstruct *only* normal transactions.
  2. At inference, compute the reconstruction error for each new transaction.
  3. High reconstruction error = the transaction doesn't match the normal
     pattern = likely fraud.

Two variants are provided:
  - FraudAutoencoder:    vanilla, symmetric encoder-decoder
  - DenoisingAutoencoder: same architecture but trained with noisy inputs
                          (often more robust for anomaly detection)
"""

import torch
import torch.nn as nn


class FraudAutoencoder(nn.Module):
    """
    Symmetric fully-connected autoencoder.

    Architecture (example with default hidden_dims=[64, 32, 16]):

        Input (D)
          ↓  Linear → BatchNorm → ReLU → Dropout
        Hidden (64)
          ↓  Linear → BatchNorm → ReLU → Dropout
        Hidden (32)
          ↓  Linear → BatchNorm → ReLU
        Bottleneck (16)  ← compressed representation
          ↓  Linear → BatchNorm → ReLU → Dropout
        Hidden (32)
          ↓  Linear → BatchNorm → ReLU → Dropout
        Hidden (64)
          ↓  Linear
        Output (D)       ← reconstruction of input
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]  # last value is the bottleneck size

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = hidden_dims[-1]

        # --- Encoder ---
        encoder_layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            # Don't apply dropout right before the bottleneck
            if i < len(hidden_dims) - 1:
                encoder_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder (mirror of the encoder) ---
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev = self.bottleneck_dim
        for i, h in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev, h))
            # No activation/batchnorm on the final output layer
            if i < len(reversed_dims) - 1:
                decoder_layers.append(nn.BatchNorm1d(h))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
            prev = h
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def encode(self, x):
        """Return the bottleneck representation (useful for visualization)."""
        return self.encoder(x)

    def reconstruction_error(self, x, reduction: str = "per_sample"):
        """
        Compute reconstruction error.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)
        reduction : {'per_sample', 'mean', 'none'}
            - 'per_sample': MSE averaged per sample → shape (batch,)
            - 'mean':       scalar mean over all elements
            - 'none':       element-wise squared error → same shape as x

        Returns
        -------
        Tensor of reconstruction errors — used as the anomaly score.
        """
        reconstruction = self.forward(x)
        sq_err = (x - reconstruction) ** 2

        if reduction == "per_sample":
            return sq_err.mean(dim=1)
        if reduction == "mean":
            return sq_err.mean()
        if reduction == "none":
            return sq_err
        raise ValueError(f"Unknown reduction: {reduction}")


class DenoisingAutoencoder(FraudAutoencoder):
    """
    Denoising variant: adds Gaussian noise to inputs during training, but
    the reconstruction target remains the CLEAN input. This forces the
    model to learn robust features rather than an identity mapping, and
    often produces a more discriminative anomaly score.
    """

    def __init__(self, input_dim, hidden_dims=None, dropout=0.2,
                 noise_std: float = 0.1):
        super().__init__(input_dim, hidden_dims, dropout)
        self.noise_std = noise_std

    def forward(self, x):
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
            z = self.encoder(x_noisy)
        else:
            z = self.encoder(x)
        return self.decoder(z)


def build_autoencoder(
    model_type: str,
    input_dim: int,
    hidden_dims: list = None,
    dropout: float = 0.2,
    noise_std: float = 0.1,
) -> FraudAutoencoder:
    """Factory function — dispatches on model type."""
    if model_type == "vanilla":
        return FraudAutoencoder(input_dim, hidden_dims, dropout)
    if model_type == "denoising":
        return DenoisingAutoencoder(input_dim, hidden_dims, dropout, noise_std)
    raise ValueError(f"Unknown model_type: {model_type}. "
                     f"Use 'vanilla' or 'denoising'.")
