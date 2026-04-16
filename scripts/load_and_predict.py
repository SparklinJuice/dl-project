"""
Load a saved model and run inference on new data.

Satisfies rubric expectation #5:
  "Save your model to a file, and design a way to load it directly,
   without retraining, for reproducibility."

Usage:
    # Autoencoder
    python scripts/load_and_predict.py \\
        --model models/autoencoder_best.pt \\
        --csv data/new_transactions.csv \\
        --model_type autoencoder

    # Supervised DNN
    python scripts/load_and_predict.py \\
        --model models/supervised_best.pt \\
        --csv data/new_transactions.csv \\
        --model_type supervised

Uses the saved scaler.pkl and feature_schema.pkl in models/ by default.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.autoencoder import build_autoencoder
from src.model import FraudDetectorDNN
from src.preprocessing import transform_new_data
from src.utils import get_device, load_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--scaler", type=str, default="models/scaler.pkl")
    p.add_argument("--schema", type=str, default="models/feature_schema.pkl")
    p.add_argument("--csv", type=str, required=True,
                   help="New data to score (same schema as training)")
    p.add_argument("--model_type", type=str, required=True,
                   choices=["autoencoder", "supervised"])
    p.add_argument("--output", type=str, default="predictions.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override saved threshold")
    return p.parse_args()


def load_autoencoder_model(checkpoint, device):
    meta = checkpoint["metadata"]
    model = build_autoencoder(
        model_type=meta["model_type"],
        input_dim=meta["input_dim"],
        hidden_dims=meta["hidden_dims"],
        dropout=meta["dropout"],
        noise_std=meta.get("noise_std", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def load_supervised_model(checkpoint, device):
    meta = checkpoint["metadata"]
    model = FraudDetectorDNN(
        input_dim=meta["input_dim"],
        hidden_dims=meta["hidden_dims"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # --- Load scaler and schema ---
    print(f"\nLoading scaler from {args.scaler}")
    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)
    print(f"Loading feature schema from {args.schema}")
    with open(args.schema, "rb") as f:
        schema = pickle.load(f)

    # --- Load checkpoint ---
    print(f"Loading checkpoint from {args.model}")
    checkpoint = load_checkpoint(args.model, device=device)
    meta = checkpoint["metadata"]
    print(f"Model: {meta.get('model_type')}, "
          f"hidden_dims={meta.get('hidden_dims')}, "
          f"trained for {meta.get('best_epoch')} epochs")

    # --- Build model ---
    if args.model_type == "autoencoder":
        model = load_autoencoder_model(checkpoint, device)
        threshold = (args.threshold
                     if args.threshold is not None
                     else meta["threshold_unsupervised"])
    else:
        model = load_supervised_model(checkpoint, device)
        threshold = (args.threshold
                     if args.threshold is not None
                     else meta["threshold"])
    print(f"Using threshold: {threshold:.6f}")

    # --- Load and transform new data ---
    print(f"\nLoading new data from {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Raw shape: {df.shape}")
    X = transform_new_data(df, scaler, schema)
    print(f"Transformed shape: {X.shape}")

    # --- Inference ---
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        if args.model_type == "autoencoder":
            scores = model.reconstruction_error(X_tensor).cpu().numpy()
            score_name = "reconstruction_error"
        else:
            logits = model(X_tensor)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            score_name = "fraud_probability"

    predictions = (scores >= threshold).astype(int)

    # --- Save output ---
    result_df = df.copy()
    result_df[score_name] = scores
    result_df["predicted_fraud"] = predictions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")

    # --- Summary ---
    n_flagged = int(predictions.sum())
    print(f"\nSummary:")
    print(f"  Total samples:    {len(predictions)}")
    print(f"  Flagged as fraud: {n_flagged} ({n_flagged/len(predictions)*100:.2f}%)")
    print(f"  Score range:      [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Threshold used:   {threshold:.6f}")

    # --- Ground-truth evaluation if available ---
    if schema["target_col"] in df.columns:
        from sklearn.metrics import (
            average_precision_score,
            classification_report,
            roc_auc_score,
        )
        y_true = df[schema["target_col"]].values
        print("\nGround truth available — evaluation:")
        print(classification_report(y_true, predictions,
                                    target_names=["Normal", "Fraud"]))
        print(f"  ROC-AUC: {roc_auc_score(y_true, scores):.4f}")
        print(f"  PR-AUC:  {average_precision_score(y_true, scores):.4f}")


if __name__ == "__main__":
    main()
