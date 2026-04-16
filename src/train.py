"""
Training loops for both model types.

  - train_autoencoder(): minimises MSE reconstruction loss on NORMAL samples only
  - train_supervised():  minimises BCEWithLogitsLoss with pos_weight for imbalance

Both use Adam + ReduceLROnPlateau + EarlyStopping with best-weight restoration.
"""

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Stop training when validation loss stops improving; restore best weights."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
        self.best_state = None
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep copy weights to CPU so we don't hold extra GPU memory
            self.best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    verbose: bool = True,
):
    """
    Train an autoencoder on normal transactions to reconstruct them.

    Loss: MSE between input and reconstruction.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_samples = 0
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        train_loss /= n_samples

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, target in val_loader:
                x = x.to(device)
                target = target.to(device)
                recon = model(x)
                loss = criterion(recon, target)
                val_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
        val_loss /= n_samples

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)
        early_stop.step(val_loss, model, epoch)

        if verbose and (epoch % 5 == 0 or epoch == 1 or early_stop.should_stop):
            print(f"  Epoch {epoch:3d}/{epochs}  | "
                  f"Train MSE: {train_loss:.6f}  | "
                  f"Val MSE: {val_loss:.6f}  | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if early_stop.should_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch}. "
                      f"Best epoch: {early_stop.best_epoch} "
                      f"(val_loss={early_stop.best_loss:.6f})")
            break

    early_stop.restore(model)
    elapsed = time.time() - start_time
    history["wall_time_sec"] = elapsed
    history["best_epoch"] = early_stop.best_epoch
    history["best_val_loss"] = early_stop.best_loss

    if verbose:
        print(f"  Training completed in {elapsed:.1f}s")

    return history


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    verbose: bool = True,
):
    """
    Train a supervised binary classifier with class-imbalance handling.

    Loss: BCEWithLogitsLoss with pos_weight to upweight the minority (fraud) class.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        train_loss /= n_samples

        model.eval()
        val_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                n_samples += x.size(0)
        val_loss /= n_samples

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)
        early_stop.step(val_loss, model, epoch)

        if verbose and (epoch % 5 == 0 or epoch == 1 or early_stop.should_stop):
            print(f"  Epoch {epoch:3d}/{epochs}  | "
                  f"Train Loss: {train_loss:.5f}  | "
                  f"Val Loss: {val_loss:.5f}  | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if early_stop.should_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch}. "
                      f"Best epoch: {early_stop.best_epoch}")
            break

    early_stop.restore(model)
    elapsed = time.time() - start_time
    history["wall_time_sec"] = elapsed
    history["best_epoch"] = early_stop.best_epoch
    history["best_val_loss"] = early_stop.best_loss

    if verbose:
        print(f"  Training completed in {elapsed:.1f}s")

    return history
