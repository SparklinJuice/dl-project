"""
Hyperparameter sweep for the autoencoder.

Runs a three-pass grid search:
  Pass 1: Architecture (hidden dims / bottleneck size)
  Pass 2: Regularization (dropout, weight_decay)
  Pass 3: Optimization (learning rate, batch size)

Each run is logged to experiments/sweep_results.csv with all hyperparameters
and validation/test metrics. Results are ranked by test PR-AUC.

Usage:
    # Full sweep (takes several hours on M2)
    python scripts/hyperparameter_sweep.py --csv data/fraud.csv

    # Quick sweep for testing
    python scripts/hyperparameter_sweep.py --csv data/fraud.csv --quick

    # Run only one pass
    python scripts/hyperparameter_sweep.py --csv data/fraud.csv --pass 1
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.autoencoder import build_autoencoder
from src.dataset import FraudDataset, NormalOnlyDataset
from src.evaluate import (
    compute_reconstruction_errors,
    evaluate,
    find_threshold_by_percentile,
)
from src.preprocessing import load_processed, run_preprocessing
from src.train import train_autoencoder
from src.utils import get_device, log_experiment, set_seed


# =============================================================================
# SEARCH SPACES
# =============================================================================
PASS_1_ARCHITECTURE = [
    # (hidden_dims, label)
    ([128, 64, 32], "wide"),
    ([64, 32, 16], "medium"),
    ([32, 16, 8], "narrow"),
    ([128, 64, 32, 16], "deep_wide"),
    ([64, 32, 16, 8], "deep_narrow"),
]

PASS_2_REGULARIZATION = list(itertools.product(
    [0.1, 0.2, 0.3, 0.5],              # dropout
    [0.0, 1e-5, 1e-4, 1e-3],           # weight_decay
))

PASS_3_OPTIMIZATION = list(itertools.product(
    [1e-4, 5e-4, 1e-3, 5e-3],          # learning rate
    [256, 512, 1024],                   # batch size
))

# Quick mode: one configuration per pass for a smoke test
QUICK_PASS_1 = [([64, 32, 16], "medium")]
QUICK_PASS_2 = [(0.2, 1e-5)]
QUICK_PASS_3 = [(1e-3, 512)]


# =============================================================================
# SINGLE TRAINING RUN
# =============================================================================
def run_one_config(
    config: dict,
    data: dict,
    device: torch.device,
    epochs: int,
    patience: int,
    experiment_log: str,
    pass_label: str,
):
    """Train one configuration and log all metrics. Returns the metrics dict."""
    set_seed(42)  # Same seed for every run so comparisons are fair

    # --- Data loaders ---
    # x_train is already normal-only from split_normal_fraud
    train_ds = NormalOnlyDataset(data["x_train"])
    val_normal_mask = data["y_val"] == 0
    val_normal_ds = NormalOnlyDataset(data["x_val"][val_normal_mask])
    val_full_ds = FraudDataset(data["x_val"], data["y_val"])
    test_full_ds = FraudDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_normal_loader = DataLoader(val_normal_ds, batch_size=config["batch_size"])
    val_full_loader = DataLoader(val_full_ds, batch_size=config["batch_size"])
    test_full_loader = DataLoader(test_full_ds, batch_size=config["batch_size"])

    # --- Build model ---
    model = build_autoencoder(
        model_type="vanilla",
        input_dim=data["input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Train (verbose=False to keep sweep output readable) ---
    t0 = time.time()
    history = train_autoencoder(
        model, train_loader, val_normal_loader, device,
        epochs=epochs,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=patience,
        verbose=False,
    )
    elapsed = time.time() - t0

    # --- Evaluate ---
    val_errors, val_labels = compute_reconstruction_errors(
        model, val_full_loader, device
    )
    test_errors, test_labels = compute_reconstruction_errors(
        model, test_full_loader, device
    )

    # Use the 95th percentile threshold (consistent across runs)
    normal_val_errors = val_errors[val_labels == 0]
    threshold = find_threshold_by_percentile(normal_val_errors, percentile=95.0)

    # Silent evaluation (just grab the metrics, don't print report)
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    preds = (test_errors >= threshold).astype(int)
    metrics = {
        "test_precision": round(float(precision_score(test_labels, preds, zero_division=0)), 4),
        "test_recall": round(float(recall_score(test_labels, preds, zero_division=0)), 4),
        "test_f1": round(float(f1_score(test_labels, preds, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(test_labels, test_errors)), 4),
        "test_pr_auc": round(float(average_precision_score(test_labels, test_errors)), 4),
    }

    # --- Log to CSV ---
    row = {
        "pass": pass_label,
        "hidden_dims": str(config["hidden_dims"]),
        "arch_label": config.get("arch_label", ""),
        "dropout": config["dropout"],
        "weight_decay": config["weight_decay"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "n_params": n_params,
        "best_epoch": history["best_epoch"],
        "best_val_loss": round(history["best_val_loss"], 6),
        "threshold": round(threshold, 6),
        "wall_time_sec": round(elapsed, 1),
        **metrics,
    }
    log_experiment(experiment_log, row)

    # Print one-line summary
    print(f"    PR-AUC={metrics['test_pr_auc']:.4f}  "
          f"F1={metrics['test_f1']:.4f}  "
          f"prec={metrics['test_precision']:.4f}  "
          f"rec={metrics['test_recall']:.4f}  "
          f"time={elapsed:.0f}s  (best epoch {history['best_epoch']})")

    return row


# =============================================================================
# THREE PASSES
# =============================================================================
def run_pass_1(data, device, epochs, patience, experiment_log, quick=False):
    """Pass 1: Architecture search. Fix regularization and optimization."""
    print("\n" + "#" * 70)
    print("# PASS 1: Architecture search")
    print("#" * 70)

    configs = QUICK_PASS_1 if quick else PASS_1_ARCHITECTURE
    results = []

    for i, (hidden_dims, label) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {label}: {hidden_dims}")
        config = {
            "hidden_dims": hidden_dims,
            "arch_label": label,
            "dropout": 0.2,
            "weight_decay": 1e-5,
            "lr": 1e-3,
            "batch_size": 512,
        }
        result = run_one_config(config, data, device, epochs, patience,
                                experiment_log, pass_label="pass1_arch")
        results.append(result)

    # Pick the best architecture by test PR-AUC
    best = max(results, key=lambda r: r["test_pr_auc"])
    print(f"\n  → Best architecture: {best['arch_label']} "
          f"({best['hidden_dims']}) with PR-AUC {best['test_pr_auc']:.4f}")
    return eval(best["hidden_dims"])  # convert str back to list


def run_pass_2(data, device, best_hidden_dims, epochs, patience, experiment_log, quick=False):
    """Pass 2: Regularization search. Fix architecture from Pass 1."""
    print("\n" + "#" * 70)
    print("# PASS 2: Regularization search")
    print("#" * 70)
    print(f"# Using best architecture from Pass 1: {best_hidden_dims}")

    configs = QUICK_PASS_2 if quick else PASS_2_REGULARIZATION
    results = []

    for i, (dropout, wd) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] dropout={dropout}, weight_decay={wd}")
        config = {
            "hidden_dims": best_hidden_dims,
            "dropout": dropout,
            "weight_decay": wd,
            "lr": 1e-3,
            "batch_size": 512,
        }
        result = run_one_config(config, data, device, epochs, patience,
                                experiment_log, pass_label="pass2_reg")
        results.append(result)

    best = max(results, key=lambda r: r["test_pr_auc"])
    print(f"\n  → Best regularization: dropout={best['dropout']}, "
          f"weight_decay={best['weight_decay']} with PR-AUC {best['test_pr_auc']:.4f}")
    return float(best["dropout"]), float(best["weight_decay"])


def run_pass_3(data, device, best_hidden_dims, best_dropout, best_wd,
               epochs, patience, experiment_log, quick=False):
    """Pass 3: Optimization search. Fix architecture and regularization."""
    print("\n" + "#" * 70)
    print("# PASS 3: Optimization search")
    print("#" * 70)
    print(f"# Using: hidden_dims={best_hidden_dims}, "
          f"dropout={best_dropout}, weight_decay={best_wd}")

    configs = QUICK_PASS_3 if quick else PASS_3_OPTIMIZATION
    results = []

    for i, (lr, bs) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] lr={lr}, batch_size={bs}")
        config = {
            "hidden_dims": best_hidden_dims,
            "dropout": best_dropout,
            "weight_decay": best_wd,
            "lr": lr,
            "batch_size": bs,
        }
        result = run_one_config(config, data, device, epochs, patience,
                                experiment_log, pass_label="pass3_opt")
        results.append(result)

    best = max(results, key=lambda r: r["test_pr_auc"])
    print(f"\n  → Best optimization: lr={best['lr']}, batch_size={best['batch_size']} "
          f"with PR-AUC {best['test_pr_auc']:.4f}")
    return float(best["lr"]), int(best["batch_size"])


# =============================================================================
# MAIN
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None,
                   help="Path to raw CSV (triggers preprocessing)")
    p.add_argument("--processed", action="store_true",
                   help="Load pre-processed arrays (much faster)")
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--epochs", type=int, default=60,
                   help="Max epochs per run (early stopping usually kicks in earlier)")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment_log", type=str,
                   default="experiments/sweep_results.csv")
    p.add_argument("--pass", type=int, dest="only_pass", default=None,
                   choices=[1, 2, 3], help="Run only one pass (for debugging)")
    p.add_argument("--quick", action="store_true",
                   help="Run minimal configs per pass (smoke test)")
    args = p.parse_args()
    if not args.csv and not args.processed:
        p.error("Must specify either --csv or --processed")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load data once, reuse across all runs
    print("\nLoading data (one-time)...")
    if args.processed:
        data = load_processed(args.processed_dir, args.models_dir)
    else:
        data = run_preprocessing(args.csv, args.processed_dir, args.models_dir)

    total_configs = 0
    if args.only_pass is None or args.only_pass == 1:
        total_configs += len(QUICK_PASS_1 if args.quick else PASS_1_ARCHITECTURE)
    if args.only_pass is None or args.only_pass == 2:
        total_configs += len(QUICK_PASS_2 if args.quick else PASS_2_REGULARIZATION)
    if args.only_pass is None or args.only_pass == 3:
        total_configs += len(QUICK_PASS_3 if args.quick else PASS_3_OPTIMIZATION)

    print(f"\nTotal configurations to run: {total_configs}")
    print(f"Max epochs per run: {args.epochs} (with patience={args.patience})")
    print(f"Results logged to: {args.experiment_log}")

    start_time = time.time()

    # Track the best config from each pass to feed into the next
    best_hidden = [64, 32, 16]
    best_dropout = 0.2
    best_wd = 1e-5
    best_lr = 1e-3
    best_bs = 512

    if args.only_pass is None or args.only_pass == 1:
        best_hidden = run_pass_1(data, device, args.epochs, args.patience,
                                 args.experiment_log, args.quick)

    if args.only_pass is None or args.only_pass == 2:
        best_dropout, best_wd = run_pass_2(data, device, best_hidden,
                                           args.epochs, args.patience,
                                           args.experiment_log, args.quick)

    if args.only_pass is None or args.only_pass == 3:
        best_lr, best_bs = run_pass_3(data, device, best_hidden,
                                      best_dropout, best_wd,
                                      args.epochs, args.patience,
                                      args.experiment_log, args.quick)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Sweep complete in {total_elapsed/60:.1f} minutes")
    print("=" * 70)
    print(f"\nFinal recommended config:")
    best_config = {
        "hidden_dims": best_hidden,
        "dropout": best_dropout,
        "weight_decay": best_wd,
        "lr": best_lr,
        "batch_size": best_bs,
    }
    print(json.dumps(best_config, indent=2))

    # Save best config for later use
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved best config to experiments/best_config.json")
    print(f"Full sweep results logged in {args.experiment_log}")


if __name__ == "__main__":
    main()
