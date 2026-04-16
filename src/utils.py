"""
Shared utilities: seeding, plotting, checkpointing.
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA + MPS) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Determinism flags only apply meaningfully on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # MPS is seeded by torch.manual_seed() above — no separate call needed


def get_device():
    """
    Return the best available device:
      - CUDA (NVIDIA GPU) if available
      - MPS  (Apple Silicon GPU) if available
      - CPU  otherwise
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# CHECKPOINTING
# =============================================================================
def save_checkpoint(model, path: str, metadata: dict = None):
    """
    Save model weights along with all metadata needed to reconstruct it.

    metadata should include: input_dim, hidden_dims, dropout, model_type,
    threshold (for anomaly detection), and any other config.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, device: torch.device = None):
    """Load a checkpoint dict. Caller reconstructs the model from metadata."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


# =============================================================================
# EXPERIMENT LOGGING
# =============================================================================
def log_experiment(results_path: str, row: dict):
    """
    Append a row to the experiments CSV. Creates it (with header) if missing.
    Use this after every training run to build a hyperparameter comparison table.
    """
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# PLOTTING
# =============================================================================
def plot_training_curves(history: dict, save_path: str = None, title: str = None):
    """Plot train vs validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="Train", linewidth=2)
    ax.plot(history["val_loss"], label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(labels, preds, save_path: str = None, title: str = None):
    """Plot a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"])
    ax.set_yticklabels(["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_precision_recall_curve(scores, labels, save_path: str = None, title: str = None):
    """Plot the precision-recall curve (better than ROC for imbalanced data)."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recalls, precisions, linewidth=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_reconstruction_error_distribution(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float = None,
    save_path: str = None,
    title: str = None,
):
    """
    Plot the distribution of reconstruction errors for normal vs fraud.
    This is THE key diagnostic for autoencoder-based anomaly detection.
    If the two distributions don't separate, the model isn't useful.
    """
    normal_errors = errors[labels == 0]
    fraud_errors = errors[labels == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(max(errors.min(), 1e-8)),
                       np.log10(errors.max()), 60)
    ax.hist(normal_errors, bins=bins, alpha=0.6, label=f"Normal (n={len(normal_errors)})",
            color="steelblue", density=True)
    ax.hist(fraud_errors, bins=bins, alpha=0.6, label=f"Fraud (n={len(fraud_errors)})",
            color="crimson", density=True)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--",
                   label=f"Threshold = {threshold:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction Error (MSE per sample)")
    ax.set_ylabel("Density")
    ax.set_title(title or "Reconstruction Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_threshold_sweep(scores, labels, save_path: str = None):
    """
    Plot precision, recall, and F1 as functions of the threshold.
    Useful for justifying the final threshold choice in the report.
    """
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    # Align lengths (precision_recall_curve returns one extra P/R value)
    precisions, recalls, f1 = precisions[:-1], recalls[:-1], f1[:-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", linewidth=2)
    ax.plot(thresholds, f1, label="F1", linewidth=2, linestyle="--")

    best_idx = int(np.argmax(f1))
    ax.axvline(thresholds[best_idx], color="black", linestyle=":",
               label=f"Best F1 at {thresholds[best_idx]:.4f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold Sweep: Precision / Recall / F1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)
