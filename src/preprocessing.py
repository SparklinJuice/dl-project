"""
Preprocessing module for the Fraudulent E-Commerce Transactions dataset.

This module implements the preprocessing logic documented in
notebooks/0_Preprocessing.ipynb, in reusable function form:

  1. Drop non-informative columns: Transaction ID, Customer ID (unique),
     IP Address, Transaction Date, Customer Location (high cardinality)
  2. Engineer 'Address Match' binary feature from Shipping vs Billing Address
  3. Encode 'Transaction Hour' cyclically using sin/cos (hour 23 and 0 should
     be close in feature space)
  4. One-hot encode nominal categoricals: Payment Method, Product Category,
     Device Used  (drop_first=True to avoid multicollinearity)
  5. Split into train (normal only) / val (normal+fraud) / test (normal+fraud)
     using 70/15/15. Normal and fraud are split separately before being
     recombined in val/test so the autoencoder can train on normal only.
  6. Fit StandardScaler on train, transform val and test.
  7. Save processed arrays, scaler, and feature-schema to disk for reuse.

Output files:
  data/processed/train.npz        x_train
  data/processed/val.npz          x_val, y_val
  data/processed/test.npz         x_test, y_test
  models/scaler.pkl               fitted StandardScaler
  models/feature_schema.pkl       column names after one-hot (for inference)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# Columns to drop outright — not useful for modeling
COLS_TO_DROP_INITIAL = [
    "Transaction ID",      # unique per row, pure identifier
    "Customer ID",         # unique per row in this dataset (verified in notebook)
    "IP Address",          # too high-cardinality, string-formatted
    "Transaction Date",    # raw date; hour is captured separately
    "Customer Location",   # very high-cardinality (~99k unique); dropping
]

# Columns dropped AFTER feature engineering has extracted their signal
COLS_TO_DROP_AFTER_ENGINEERING = [
    "Shipping Address",    # captured by 'Address Match'
    "Billing Address",     # captured by 'Address Match'
    "Transaction Hour",    # captured by Hour_Sin / Hour_Cos
]

# Nominal categoricals — one-hot encoded with drop_first=True
ONE_HOT_COLS = ["Payment Method", "Product Category", "Device Used"]

TARGET_COL = "Is Fraudulent"


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps. Input: raw CSV dataframe.
    Output: dataframe with engineered + one-hot features, ready to scale.
    """
    df = df.copy()

    # Drop initial non-features (checking each column exists first)
    drop_initial = [c for c in COLS_TO_DROP_INITIAL if c in df.columns]
    if drop_initial and verbose:
        print(f"Dropping non-feature columns: {drop_initial}")
    df = df.drop(columns=drop_initial, errors="ignore")

    # Address match: 1 if shipping == billing, else 0
    if "Shipping Address" in df.columns and "Billing Address" in df.columns:
        df["Address Match"] = (
            df["Shipping Address"] == df["Billing Address"]
        ).astype(int)
        if verbose:
            match_rate = df["Address Match"].mean()
            print(f"Address Match feature created (match rate: {match_rate:.3f})")

    # Cyclical encoding of Transaction Hour
    if "Transaction Hour" in df.columns:
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Transaction Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Transaction Hour"] / 24)
        if verbose:
            print("Cyclical Hour_Sin / Hour_Cos features created")

    # Drop the original columns we just replaced
    drop_after = [c for c in COLS_TO_DROP_AFTER_ENGINEERING if c in df.columns]
    df = df.drop(columns=drop_after, errors="ignore")

    # One-hot encode nominal categoricals (drop_first=True prevents
    # multicollinearity — one category becomes the implicit reference)
    one_hot_present = [c for c in ONE_HOT_COLS if c in df.columns]
    if one_hot_present:
        df = pd.get_dummies(df, columns=one_hot_present, drop_first=True)
        if verbose:
            print(f"One-hot encoded: {one_hot_present}")

    # Ensure bool columns become int (so np.astype('float32') works cleanly)
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    return df


# =============================================================================
# SPLITTING
# =============================================================================
def split_normal_fraud(df: pd.DataFrame, verbose: bool = True):
    """
    Split into:
      x_train          — normal transactions only (70% of normals)
      val_df, test_df  — normal + fraud combined (15% each)

    Rationale: autoencoder trains on normal only; val/test must include fraud
    to evaluate anomaly detection performance.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. "
                         f"Available: {list(df.columns)}")

    normal = df[df[TARGET_COL] == 0].drop(columns=[TARGET_COL])
    fraud = df[df[TARGET_COL] == 1].drop(columns=[TARGET_COL])

    if verbose:
        print(f"\nTotal normal: {len(normal)}, Total fraud: {len(fraud)}")

    # Normal: 70% train, 15% val, 15% test
    normal_train, normal_temp = train_test_split(
        normal, test_size=0.30, random_state=SEED
    )
    normal_val, normal_test = train_test_split(
        normal_temp, test_size=0.50, random_state=SEED
    )

    # Fraud: 50/50 into val and test (none in train — autoencoder shouldn't see it)
    fraud_val, fraud_test = train_test_split(
        fraud, test_size=0.50, random_state=SEED
    )

    # Combine normal + fraud for val/test, shuffle, record labels
    val_df = (
        pd.concat([normal_val.assign(label=0), fraud_val.assign(label=1)])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([normal_test.assign(label=0), fraud_test.assign(label=1)])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    x_train = normal_train.values.astype("float32")
    x_val = val_df.drop(columns=["label"]).values.astype("float32")
    y_val = val_df["label"].values.astype("int64")
    x_test = test_df.drop(columns=["label"]).values.astype("float32")
    y_test = test_df["label"].values.astype("int64")

    feature_names = list(normal_train.columns)

    if verbose:
        print(f"Train shape: {x_train.shape}  (all normal)")
        print(f"Val shape:   {x_val.shape}  (fraud rate: {y_val.mean():.2%})")
        print(f"Test shape:  {x_test.shape}  (fraud rate: {y_test.mean():.2%})")

    return x_train, x_val, y_val, x_test, y_test, feature_names


# =============================================================================
# SCALING
# =============================================================================
def fit_and_apply_scaler(x_train, x_val, x_test):
    """
    Fit StandardScaler on train only, apply to all three sets.
    This prevents val/test statistics from leaking into training.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype("float32")
    x_val_scaled = scaler.transform(x_val).astype("float32")
    x_test_scaled = scaler.transform(x_test).astype("float32")
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_preprocessing(
    csv_path: str,
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    verbose: bool = True,
):
    """
    Full preprocessing pipeline: load CSV, engineer features, split, scale,
    save everything to disk.

    Returns a dict with all arrays and metadata so the caller can use them
    directly without needing to reload from disk.
    """
    if verbose:
        print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"Loaded shape: {df.shape}")

    df = engineer_features(df, verbose=verbose)
    if verbose:
        print(f"\nShape after feature engineering: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    x_train, x_val, y_val, x_test, y_test, feature_names = split_normal_fraud(
        df, verbose=verbose
    )

    x_train, x_val, x_test, scaler = fit_and_apply_scaler(x_train, x_val, x_test)
    if verbose:
        print(f"\nAfter scaling:")
        print(f"  Train — mean: {x_train.mean():.4f}, std: {x_train.std():.4f}")
        print(f"  Val   — mean: {x_val.mean():.4f}, std: {x_val.std():.4f}")
        print(f"  Test  — mean: {x_test.mean():.4f}, std: {x_test.std():.4f}")

    # Save arrays
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_path / "train.npz", x_train=x_train)
    np.savez_compressed(processed_path / "val.npz", x_val=x_val, y_val=y_val)
    np.savez_compressed(processed_path / "test.npz", x_test=x_test, y_test=y_test)

    # Save scaler + schema
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    with open(models_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    schema = {
        "feature_names": feature_names,
        "input_dim": x_train.shape[1],
        "one_hot_cols": ONE_HOT_COLS,
        "dropped_initial": COLS_TO_DROP_INITIAL,
        "dropped_after_engineering": COLS_TO_DROP_AFTER_ENGINEERING,
        "target_col": TARGET_COL,
    }
    with open(models_path / "feature_schema.pkl", "wb") as f:
        pickle.dump(schema, f)

    if verbose:
        print(f"\nSaved processed arrays to {processed_path}/")
        print(f"Saved scaler + schema to {models_path}/")

    return {
        "x_train": x_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "input_dim": x_train.shape[1],
    }


def load_processed(processed_dir: str = "data/processed",
                   models_dir: str = "models"):
    """
    Load previously-processed arrays from disk. Much faster than re-running
    the full preprocessing pipeline on 1.47M rows.
    """
    processed_path = Path(processed_dir)
    models_path = Path(models_dir)

    train = np.load(processed_path / "train.npz")
    val = np.load(processed_path / "val.npz")
    test = np.load(processed_path / "test.npz")

    with open(models_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_path / "feature_schema.pkl", "rb") as f:
        schema = pickle.load(f)

    return {
        "x_train": train["x_train"],
        "x_val": val["x_val"], "y_val": val["y_val"],
        "x_test": test["x_test"], "y_test": test["y_test"],
        "scaler": scaler,
        "feature_names": schema["feature_names"],
        "input_dim": schema["input_dim"],
        "schema": schema,
    }


def transform_new_data(df: pd.DataFrame, scaler, schema: dict) -> np.ndarray:
    """
    Apply the saved preprocessing pipeline to new data for inference.

    Handles the case where new data may have missing one-hot categories
    (e.g. a category not seen during training) by adding zero columns.
    """
    if schema["target_col"] in df.columns:
        df = df.drop(columns=[schema["target_col"]])

    df = engineer_features(df, verbose=False)

    # Ensure all training columns are present (missing one-hots → zero)
    for col in schema["feature_names"]:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training column order and drop extras
    df = df[schema["feature_names"]]

    x = df.values.astype("float32")
    x = scaler.transform(x).astype("float32")
    return x
