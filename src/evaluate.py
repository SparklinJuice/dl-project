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

    results = {}
    if fn_mask.any():
        fn_idx = np.where(fn_mask)[0]
        # Sort by how confident the model was that it was normal
        fn_idx = fn_idx[np.argsort(scores[fn_idx])[:n_examples]]
        fn_df = pd.DataFrame(features[fn_idx], columns=feature_names)
        fn_df["score"] = scores[fn_idx]
        fn_df["true_label"] = labels[fn_idx]
        results["false_negatives"] = fn_df

    if fp_mask.any():
        fp_idx = np.where(fp_mask)[0]
        # Sort by how confident the model was that it was fraud
        fp_idx = fp_idx[np.argsort(-scores[fp_idx])[:n_examples]]
        fp_df = pd.DataFrame(features[fp_idx], columns=feature_names)
        fp_df["score"] = scores[fp_idx]
        fp_df["true_label"] = labels[fp_idx]
        results["false_positives"] = fp_df

    return results
