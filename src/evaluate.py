"""
Evaluation utilities.

For autoencoders:
  - compute_reconstruction_errors(): anomaly scores per sample
  - find_threshold_by_percentile(): threshold from normal-only val distribution
  - find_threshold_by_f1():         threshold maximising F1 on val set (uses labels)

For supervised DNN:
  - get_predictions(): sigmoid probabilities
  - find_optimal_threshold(): threshold maximising F1 on precision-recall curve

Shared:
  - evaluate(): prints classification report, confusion matrix, all key metrics
"""

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# =============================================================================
# AUTOENCODER — reconstruction-error scoring
# =============================================================================
@torch.no_grad()
def compute_reconstruction_errors(model, data_loader, device):
    """
    Compute per-sample reconstruction error and return labels too.

    Returns
    -------
    errors : np.ndarray of shape (N,)   — anomaly scores
    labels : np.ndarray of shape (N,)   — true labels (if available)
    """
    model.eval()
    all_errors, all_labels = [], []

    for batch in data_loader:
        # Batch format depends on which Dataset is used
        if len(batch) == 2:
            x, y = batch
            # If y has same shape as x, it's a NormalOnlyDataset → no labels
            if y.shape == x.shape:
                labels_batch = None
            else:
                labels_batch = y.numpy().flatten()
        else:
            x = batch
            labels_batch = None

        x = x.to(device)
        errors = model.reconstruction_error(x, reduction="per_sample")
        all_errors.append(errors.cpu().numpy())
        if labels_batch is not None:
            all_labels.append(labels_batch)

    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels) if all_labels else None
    return errors, labels


def find_threshold_by_percentile(errors_on_normal: np.ndarray, percentile: float = 95.0):
    """
    Pick threshold at a percentile of the normal-only error distribution.
    E.g. percentile=95 → anything worse than 95% of normals is flagged.
    This is the TRUE unsupervised approach (no fraud labels needed).
    """
    threshold = np.percentile(errors_on_normal, percentile)
    print(f"Threshold at {percentile:.1f}th percentile of normal errors: "
          f"{threshold:.6f}")
    return threshold


def find_threshold_by_f1(errors: np.ndarray, labels: np.ndarray):
    """
    Supervised threshold selection: pick the threshold on reconstruction error
    that maximises F1. Requires labels on the val set. Useful for comparison.
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, errors)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    print(f"F1-optimal threshold: {best_threshold:.6f}  "
          f"(precision={precisions[best_idx]:.3f}, "
          f"recall={recalls[best_idx]:.3f}, f1={f1[best_idx]:.3f})")
    return float(best_threshold)


# =============================================================================
# SUPERVISED DNN — probability scoring
# =============================================================================
@torch.no_grad()
def get_predictions(model, data_loader, device):
    """Return sigmoid probabilities and labels for a supervised model."""
    model.eval()
    all_probs, all_labels = [], []

    for x, y in data_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        all_labels.append(y.numpy().flatten())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray):
    """Pick threshold that maximises F1 on the precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1))
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    print(f"F1-optimal threshold: {best_threshold:.4f}  "
          f"(precision={precisions[best_idx]:.3f}, "
          f"recall={recalls[best_idx]:.3f}, f1={f1[best_idx]:.3f})")
    return float(best_threshold)


# =============================================================================
# SHARED — evaluation printout
# =============================================================================
def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    label: str = "Test",
):
    """
    Print a full evaluation report.

    Works for any anomaly scoring method:
      - For autoencoder: scores = reconstruction errors
      - For supervised:  scores = sigmoid probabilities
    Higher score = more likely fraud in both cases.
    """
    preds = (scores >= threshold).astype(int)

    print(f"\n{'=' * 60}")
    print(f"  {label} Set Evaluation   (threshold = {threshold:.6f})")
    print(f"{'=' * 60}")
    print(classification_report(labels, preds, target_names=["Normal", "Fraud"],
                                digits=4))

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"                Pred Normal   Pred Fraud")
    print(f"  True Normal   {tn:>11}   {fp:>10}")
    print(f"  True Fraud    {fn:>11}   {tp:>10}")

    metrics = {
        "threshold": float(threshold),
        "accuracy": float((preds == labels).mean()),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    print(f"\n  ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:   {metrics['pr_auc']:.4f}  (better metric for imbalanced data)")
    print(f"  F1:       {metrics['f1']:.4f}")

    return metrics


def get_failure_cases(
    scores: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    threshold: float,
    feature_names: list,
    n_examples: int = 5,
):
    """
    Pull out false-positive and false-negative examples for error analysis.
    Required by the rubric: "show examples of your model malfunctioning".
    """
    import pandas as pd

    preds = (scores >= threshold).astype(int)

    fn_mask = (labels == 1) & (preds == 0)  # Missed fraud
    fp_mask = (labels == 0) & (preds == 1)  # False alarms

    # Handle feature_names if it has fewer columns than features
    if len(feature_names) < features.shape[1]:
        # Pad with generic names for extra features
        padded_names = list(feature_names) + [f"feature_{i}" for i in range(len(feature_names), features.shape[1])]
    else:
        padded_names = feature_names

    results = {}
    if fn_mask.any():
        fn_idx = np.where(fn_mask)[0]
        # Sort by how confident the model was that it was normal
        fn_idx = fn_idx[np.argsort(scores[fn_idx])[:n_examples]]
        fn_df = pd.DataFrame(features[fn_idx], columns=padded_names)
        fn_df["score"] = scores[fn_idx]
        fn_df["true_label"] = labels[fn_idx]
        results["false_negatives"] = fn_df

    if fp_mask.any():
        fp_idx = np.where(fp_mask)[0]
        # Sort by how confident the model was that it was fraud
        fp_idx = fp_idx[np.argsort(-scores[fp_idx])[:n_examples]]
        fp_df = pd.DataFrame(features[fp_idx], columns=padded_names)
        fp_df["score"] = scores[fp_idx]
        fp_df["true_label"] = labels[fp_idx]
        results["false_positives"] = fp_df

    return results


# =============================================================================
# CLI EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    # Import model classes and utilities
    from autoencoder import build_autoencoder
    from model import FraudDetectorDNN
    from dataset import FraudDataset, NormalOnlyDataset
    from utils import load_checkpoint, get_device
    from preprocessing_withIP import load_processed
    
    # 1. Parse the command line arguments
    parser = argparse.ArgumentParser(description="Run evaluation on a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pt file)")
    parser.add_argument("--data_dir", type=str, default="../data/processed_withIP", 
                        help="Directory containing processed data (train.npz, val.npz, test.npz)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model file at {args.model_path}")

    # 2. Setup Device
    device = get_device()
    print(f"Using device: {device}")

    # 3. Load the Model
    print(f"Loading checkpoint...")
    checkpoint = load_checkpoint(args.model_path, device=device)
    metadata = checkpoint["metadata"]
    model_type = metadata.get("model_type", "vanilla")
    input_dim = metadata["input_dim"]
    hidden_dims = metadata.get("hidden_dims", [64, 32, 16])
    dropout = metadata.get("dropout", 0.2)
    noise_std = metadata.get("noise_std", 0.1)
    
    # Determine if it's an autoencoder or supervised model
    if model_type in ["vanilla", "denoising"]:
        # It's an autoencoder
        model = build_autoencoder(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            noise_std=noise_std,
        )
        is_autoencoder = True
    elif model_type == "supervised_dnn":
        # It's a supervised DNN
        model = FraudDetectorDNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        is_autoencoder = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Expected 'vanilla', 'denoising', or 'supervised_dnn'.")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded {model_type} model with input_dim={input_dim}")
    
    # 4. Load the Data
    print(f"Loading data from {args.data_dir}...")
    data = load_processed(processed_dir=args.data_dir, models_dir="../models")
    
    x_test = data["x_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    
    # Create test dataset and dataloader
    test_dataset = FraudDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Run the Evaluation using your functions!
    print(f"Computing predictions on test set...")
    
    if is_autoencoder:
        # For autoencoder: compute reconstruction errors
        errors, labels = compute_reconstruction_errors(model, test_loader, device)
        
        if labels is not None:
            threshold = find_threshold_by_f1(errors, labels)
            metrics = evaluate(errors, labels, threshold=threshold, label="Test")
            
            # Show failure cases if requested
            print("\n" + "="*60)
            print("  Failure Case Analysis")
            print("="*60)
            failure_cases = get_failure_cases(
                errors, labels, x_test, threshold, feature_names, n_examples=5
            )
            if "false_negatives" in failure_cases:
                print("\nFalse Negatives (Missed Fraud):")
                print(failure_cases["false_negatives"])
            if "false_positives" in failure_cases:
                print("\nFalse Positives (False Alarms):")
                print(failure_cases["false_positives"])
        else:
            print("No labels provided in data loader. Cannot compute F1 or run full evaluation.")
    
    else:  # supervised_dnn
        # For supervised DNN: compute sigmoid probabilities
        probs, labels = get_predictions(model, test_loader, device)
        threshold = find_optimal_threshold(probs, labels)
        metrics = evaluate(probs, labels, threshold=threshold, label="Test")
        
        # Show failure cases
        print("\n" + "="*60)
        print("  Failure Case Analysis")
        print("="*60)
        failure_cases = get_failure_cases(
            probs, labels, x_test, threshold, feature_names, n_examples=5
        )
        if "false_negatives" in failure_cases:
            print("\nFalse Negatives (Missed Fraud):")
            print(failure_cases["false_negatives"])
        if "false_positives" in failure_cases:
            print("\nFalse Positives (False Alarms):")
            print(failure_cases["false_positives"])