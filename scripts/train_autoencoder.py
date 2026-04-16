"""
Train the autoencoder (MAIN MODEL).

Two ways to run this:

  (A) Preprocess + train in one command (slower, does preprocessing every time):
        python scripts/train_autoencoder.py --csv data/fraud.csv

  (B) Use pre-processed arrays from scripts/preprocess_data.py (fast):
        python scripts/preprocess_data.py --csv data/fraud.csv   # run once
        python scripts/train_autoencoder.py --processed           # reuses it

Option (B) is the recommended workflow — preprocessing 1.47M rows takes time,
and you only need to do it once.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.autoencoder import build_autoencoder
from src.dataset import FraudDataset, NormalOnlyDataset
from src.evaluate import (
    compute_reconstruction_errors,
    evaluate,
    find_threshold_by_f1,
    find_threshold_by_percentile,
)
from src.preprocessing import load_processed, run_preprocessing
from src.train import train_autoencoder
from src.utils import (
    get_device,
    log_experiment,
    plot_precision_recall_curve,
    plot_reconstruction_error_distribution,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    # Either --csv (run preprocessing) or --processed (load from disk)
    p.add_argument("--csv", type=str, default=None,
                   help="Path to raw CSV (triggers preprocessing)")
    p.add_argument("--processed", action="store_true",
                   help="Load pre-processed arrays from data/processed/")
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--model_type", type=str, default="vanilla",
                   choices=["vanilla", "denoising"])
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32, 16])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--noise_std", type=float, default=0.1,
                   help="Only used for denoising autoencoder")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--threshold_percentile", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--fig_dir", type=str, default="figures")
    p.add_argument("--experiment_log", type=str, default="experiments/results.csv")
    args = p.parse_args()

    if not args.csv and not args.processed:
        p.error("Must specify either --csv (to preprocess) or "
                "--processed (to load pre-processed arrays)")
    return args


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
        print(f"Loading pre-processed arrays from {args.processed_dir}/")
        data = load_processed(args.processed_dir, args.out_dir)
    else:
        data = run_preprocessing(args.csv, args.processed_dir, args.out_dir)

    print(f"Train shape: {data['x_train'].shape}")
    print(f"Val shape:   {data['x_val'].shape}  (fraud rate: {data['y_val'].mean():.2%})")
    print(f"Test shape:  {data['x_test'].shape}  (fraud rate: {data['y_test'].mean():.2%})")

    # ---- Build data loaders ----
    # Training: normal samples only.
    # NOTE: x_train is ALREADY normal-only (from split_normal_fraud). We just
    # wrap it in NormalOnlyDataset without filtering.
    train_ds = NormalOnlyDataset(data["x_train"])

    # For val MSE tracking, we also want normal-only (reconstruction loss
    # measured against the distribution the model was trained on)
    val_normal_mask = data["y_val"] == 0
    val_normal_ds = NormalOnlyDataset(data["x_val"][val_normal_mask])

    # Full val/test sets (normal + fraud) for anomaly-detection evaluation
    val_full_ds = FraudDataset(data["x_val"], data["y_val"])
    test_full_ds = FraudDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_normal_loader = DataLoader(val_normal_ds, batch_size=args.batch_size)
    val_full_loader = DataLoader(val_full_ds, batch_size=args.batch_size)
    test_full_loader = DataLoader(test_full_ds, batch_size=args.batch_size)

    # ---- Build model ----
    print("\n" + "=" * 60)
    print("  Step 2/5: Build autoencoder")
    print("=" * 60)
    model = build_autoencoder(
        model_type=args.model_type,
        input_dim=data["input_dim"],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        noise_std=args.noise_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type} autoencoder")
    print(f"Hidden dims: {args.hidden_dims}  (bottleneck: {args.hidden_dims[-1]})")
    print(f"Trainable parameters: {n_params:,}")

    # ---- Train ----
    print("\n" + "=" * 60)
    print("  Step 3/5: Train")
    print("=" * 60)
    history = train_autoencoder(
        model, train_loader, val_normal_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience,
    )

    fig_dir = Path(args.fig_dir)
    plot_training_curves(
        history,
        save_path=fig_dir / "autoencoder_training_curves.png",
        title="Autoencoder — Training & Validation MSE",
    )

    # ---- Threshold selection on val set ----
    print("\n" + "=" * 60)
    print("  Step 4/5: Pick anomaly threshold")
    print("=" * 60)

    val_errors, val_labels = compute_reconstruction_errors(
        model, val_full_loader, device
    )
    normal_val_errors = val_errors[val_labels == 0]

    print("\nUnsupervised threshold (percentile of normal errors):")
    threshold_pct = find_threshold_by_percentile(
        normal_val_errors, percentile=args.threshold_percentile
    )

    print("\nSupervised threshold (F1-optimal on val set) — for comparison:")
    threshold_f1 = find_threshold_by_f1(val_errors, val_labels)

    # Use the unsupervised threshold as primary (this is an anomaly detector)
    final_threshold = threshold_pct

    plot_reconstruction_error_distribution(
        val_errors, val_labels, threshold=final_threshold,
        save_path=fig_dir / "autoencoder_error_distribution.png",
        title="Validation set — reconstruction error distribution",
    )

    # ---- Final test evaluation ----
    print("\n" + "=" * 60)
    print("  Step 5/5: Test set evaluation")
    print("=" * 60)
    test_errors, test_labels = compute_reconstruction_errors(
        model, test_full_loader, device
    )

    print("\n--- Using unsupervised threshold (percentile) ---")
    metrics_pct = evaluate(test_errors, test_labels, threshold_pct, label="Test")

    print("\n--- Using F1-optimal threshold (val set) ---")
    metrics_f1 = evaluate(test_errors, test_labels, threshold_f1, label="Test")

    plot_precision_recall_curve(
        test_errors, test_labels,
        save_path=fig_dir / "autoencoder_pr_curve.png",
        title="Autoencoder — Test Precision-Recall",
    )

    # ---- Save everything needed for reproducibility ----
    # Note: scaler + feature_schema are already saved by run_preprocessing()
    # (or were saved on the original --csv run). We only save the model here.
    out_dir = Path(args.out_dir)
    save_checkpoint(
        model,
        path=out_dir / "autoencoder_best.pt",
        metadata={
            "model_type": args.model_type,
            "input_dim": data["input_dim"],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "noise_std": args.noise_std,
            "threshold_unsupervised": threshold_pct,
            "threshold_f1_optimal": threshold_f1,
            "threshold_percentile": args.threshold_percentile,
            "feature_names": data["feature_names"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
        },
    )

    # Log to experiments CSV
    log_experiment(args.experiment_log, {
        "model": f"AE-{args.model_type}",
        "hidden_dims": str(args.hidden_dims),
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "threshold_method": "percentile-95",
        "threshold": round(threshold_pct, 6),
        "test_precision": metrics_pct["precision"],
        "test_recall": metrics_pct["recall"],
        "test_f1": metrics_pct["f1"],
        "test_roc_auc": metrics_pct["roc_auc"],
        "test_pr_auc": metrics_pct["pr_auc"],
        "wall_time_sec": round(history["wall_time_sec"], 1),
        "best_epoch": history["best_epoch"],
    })

    print("\nDone. Model and preprocessor saved to:", out_dir)


if __name__ == "__main__":
    main()
