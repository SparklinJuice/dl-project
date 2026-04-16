"""
Baselines for comparison against the autoencoder.

Satisfies the rubric requirement:
  "Give a quick comparison of your model performance against some
   state-of-the-art ones."

For tabular fraud detection, the genuinely SOTA methods are gradient-boosted
trees (XGBoost, LightGBM), NOT deep learning. We include them honestly.

Baselines included:
  - LogisticRegression  (simple linear baseline)
  - RandomForest        (classic tabular baseline)
  - XGBoost             (SOTA for tabular)
  - LightGBM            (SOTA for tabular)
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        print("  [skip] xgboost not installed — run `pip install xgboost`")
        return None


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        print("  [skip] lightgbm not installed — run `pip install lightgbm`")
        return None


def _evaluate_baseline(name, model, X_train, y_train, X_test, y_test):
    """Fit a baseline, evaluate on test set, return metrics dict."""
    print(f"\n  Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Probability of fraud (class 1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # Fall back to decision function if needed
        probs = model.decision_function(X_test)

    preds = (probs >= 0.5).astype(int)

    metrics = {
        "model": name,
        "train_time_sec": round(train_time, 2),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "pr_auc": round(average_precision_score(y_test, probs), 4),
    }

    print(f"    Train time: {train_time:.1f}s  |  "
          f"PR-AUC: {metrics['pr_auc']:.4f}  |  F1: {metrics['f1']:.4f}")
    return metrics


def run_all_baselines(X_train, y_train, X_test, y_test):
    """
    Fit all baselines and return a list of metrics dicts.
    Handles class imbalance where the model supports it.
    """
    results = []

    # Class balance info for imbalance-aware models
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    print("=" * 60)
    print("  Running baseline comparison")
    print("=" * 60)
    print(f"  Train size: {len(y_train)}  (fraud rate: {y_train.mean():.4f})")
    print(f"  Test size:  {len(y_test)}   (fraud rate: {y_test.mean():.4f})")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # --- Logistic Regression ---
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )
    results.append(_evaluate_baseline(
        "LogisticRegression", model, X_train, y_train, X_test, y_test
    ))

    # --- Random Forest ---
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    results.append(_evaluate_baseline(
        "RandomForest", model, X_train, y_train, X_test, y_test
    ))

    # --- XGBoost (SOTA) ---
    XGBClassifier = _try_import_xgboost()
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        results.append(_evaluate_baseline(
            "XGBoost", model, X_train, y_train, X_test, y_test
        ))

    # --- LightGBM (SOTA) ---
    LGBMClassifier = _try_import_lightgbm()
    if LGBMClassifier is not None:
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        results.append(_evaluate_baseline(
            "LightGBM", model, X_train, y_train, X_test, y_test
        ))

    return results


def print_comparison_table(baseline_results: list, our_results: list = None):
    """Pretty-print a comparison table including our models."""
    import pandas as pd

    all_results = list(baseline_results)
    if our_results:
        all_results.extend(our_results)

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    return df
