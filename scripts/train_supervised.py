"""
Train the supervised DNN classifier for comparison against the autoencoder.

Two ways to run:
  (A) python scripts/train_supervised.py --csv data/fraud.csv
  (B) python scripts/train_supervised.py --processed     (after preprocess_data.py)

NOTE: The autoencoder split gives us x_train = normal-only, but supervised
training needs BOTH classes. This script reconstructs a fraud-aware training
set by combining (x_train normal + val's fraud samples) for training, and
holds out test as the final evaluation set. This matches the design choice
of keeping the autoencoder's test set completely held out.

Alternative (cleaner): re-preprocess with a stratified supervised split.
For simplicity here, we use the existing processed arrays.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FraudDataset
from src.evaluate import evaluate, find_optimal_threshold, get_predictions
from src.model import FraudDetectorDNN
from src.preprocessing import load_processed, run_preprocessing
from src.train import train_supervised
from src.utils import (
    get_device,
    log_experiment,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--processed", action="store_true")
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64, 32])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--fig_dir", type=str, default="figures")
    p.add_argument("--experiment_log", type=str, default="experiments/results.csv")
    args = p.parse_args()

    if not args.csv and not args.processed:
        p.error("Must specify either --csv or --processed")
    return args


def build_supervised_splits(data):
    """
    Build supervised train/val/test from the autoencoder-style processed arrays.

    Autoencoder processed data:
      x_train = all normal
      x_val   = normal + fraud  (we split this 50/50 → new_train, new_val)
      x_test  = normal + fraud  (held out as-is)

    We merge x_train (all normal) with HALF of x_val's mixed set to form the
    supervised training set, leaving the other half as supervised val.
    The test set is unchanged so autoencoder/DNN evaluations are comparable.
    """
    x_train_normal = data["x_train"]  # shape (N_normal, D)
    x_val_mixed = data["x_val"]
    y_val_mixed = data["y_val"]
    x_test_mixed = data["x_test"]
    y_test_mixed = data["y_test"]

    # Split val 50/50: half joins training, half stays val
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_val_mixed))
    half = len(perm) // 2
    train_extra_idx, new_val_idx = perm[:half], perm[half:]

    x_train_extra = x_val_mixed[train_extra_idx]
    y_train_extra = y_val_mixed[train_extra_idx]
    x_new_val = x_val_mixed[new_val_idx]
    y_new_val = y_val_mixed[new_val_idx]

    # Merge: normal (label 0) + mixed half (with real labels)
    y_train_normal = np.zeros(len(x_train_normal), dtype=np.int64)
    x_train = np.concatenate([x_train_normal, x_train_extra], axis=0)
    y_train = np.concatenate([y_train_normal, y_train_extra], axis=0)

    # Shuffle the merged training set
    perm_train = rng.permutation(len(y_train))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_new_val, "y_val": y_new_val,
        "x_test": x_test_mixed, "y_test": y_test_mixed,
        "input_dim": data["input_dim"],
        "feature_names": data["feature_names"],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # ---- Load data ----
    print("\n" + "=" * 60)
    print("  Step 1/5: Load data")
    print("=" * 60)
    if args.processed:
        raw_data = load_processed(args.processed_dir, args.out_dir)
    else:
        raw_data = run_preprocessing(args.csv, args.processed_dir, args.out_dir)

    # Reassemble into supervised train/val/test
    data = build_supervised_splits(raw_data)
    print(f"\nSupervised splits:")
    print(f"  Train: {data['x_train'].shape}  (fraud rate: {data['y_train'].mean():.4f})")
    print(f"  Val:   {data['x_val'].shape}  (fraud rate: {data['y_val'].mean():.4f})")
    print(f"  Test:  {data['x_test'].shape}  (fraud rate: {data['y_test'].mean():.4f})")

    train_ds = FraudDataset(data["x_train"], data["y_train"])
    val_ds = FraudDataset(data["x_val"], data["y_val"])
    test_ds = FraudDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Compute pos_weight for imbalanced BCE
    n_pos = (data["y_train"] == 1).sum()
    n_neg = (data["y_train"] == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"pos_weight (neg/pos): {pos_weight.item():.2f}")

    # ---- Build model ----
    print("\n" + "=" * 60)
    print("  Step 2/5: Build DNN")
    print("=" * 60)
    model = FraudDetectorDNN(
        input_dim=data["input_dim"],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Trainable parameters: {n_params:,}")

    # ---- Train ----
    print("\n" + "=" * 60)
    print("  Step 3/5: Train")
    print("=" * 60)
    history = train_supervised(
        model, train_loader, val_loader, pos_weight, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience,
    )

    fig_dir = Path(args.fig_dir)
    plot_training_curves(
        history, save_path=fig_dir / "supervised_training_curves.png",
        title="Supervised DNN — Training & Validation Loss",
    )

    # ---- Find optimal threshold on val ----
    print("\n" + "=" * 60)
    print("  Step 4/5: Tune threshold")
    print("=" * 60)
    val_probs, val_labels = get_predictions(model, val_loader, device)
    best_threshold = find_optimal_threshold(val_probs, val_labels)

    # ---- Test evaluation ----
    print("\n" + "=" * 60)
    print("  Step 5/5: Test evaluation")
    print("=" * 60)
    test_probs, test_labels = get_predictions(model, test_loader, device)
    metrics = evaluate(test_probs, test_labels, best_threshold, label="Test")

    test_preds = (test_probs >= best_threshold).astype(int)
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=fig_dir / "supervised_confusion_matrix.png",
        title=f"Supervised DNN — Test Confusion Matrix (thr={best_threshold:.3f})",
    )
    plot_precision_recall_curve(
        test_probs, test_labels,
        save_path=fig_dir / "supervised_pr_curve.png",
        title="Supervised DNN — Test Precision-Recall",
    )

    # ---- Save ----
    # Note: scaler + feature_schema already saved by run_preprocessing()
    out_dir = Path(args.out_dir)
    save_checkpoint(
        model,
        path=out_dir / "supervised_best.pt",
        metadata={
            "model_type": "supervised_dnn",
            "input_dim": data["input_dim"],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "threshold": best_threshold,
            "feature_names": data["feature_names"],
            "best_epoch": history["best_epoch"],
        },
    )

    log_experiment(args.experiment_log, {
        "model": "DNN-supervised",
        "hidden_dims": str(args.hidden_dims),
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "threshold_method": "F1-optimal",
        "threshold": round(best_threshold, 6),
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        "wall_time_sec": round(history["wall_time_sec"], 1),
        "best_epoch": history["best_epoch"],
    })

    print("\nDone.")


if __name__ == "__main__":
    main()
