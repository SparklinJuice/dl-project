"""
Preprocessing module for the Fraudulent E-Commerce Transactions dataset.

This module implements the preprocessing logic documented in
notebooks/0_Preprocessing_withIP.ipynb, in reusable function form:

  1. Drop non-informative columns: Customer Location (high cardinality)
  2. Engineer 'Address Match' binary feature from Shipping vs Billing Address
  3. Engineer IP features: IP Velocity (1H rolling) and Unique Customers per IP
  4. Encode 'Transaction Hour' cyclically using sin/cos
  5. One-hot encode nominal categoricals (drop_first=True)
  6. Split into train (normal only) / val (normal+fraud) / test (normal+fraud)
  7. Fit StandardScaler on train, transform val and test.
  8. Save processed arrays, scaler, and feature-schema to disk for reuse.
"""

import pickle
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# Columns to drop outright — not useful for modeling and not needed for feature engineering
COLS_TO_DROP_INITIAL = [
    "Customer Location",   # very high-cardinality (~99k unique); dropping
]

# Columns dropped AFTER feature engineering has extracted their signal
COLS_TO_DROP_AFTER_ENGINEERING = [
    "Transaction ID",      # Needed for IP Velocity rolling count
    "Customer ID",         # Needed for IP Linkage user count
    "IP Address",          # Needed for grouping
    "Transaction Date",    # Needed for 1H rolling window
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

    # Drop initial non-features
    drop_initial = [c for c in COLS_TO_DROP_INITIAL if c in df.columns]
    if drop_initial and verbose:
        print(f"Dropping non-feature columns: {drop_initial}")
    df = df.drop(columns=drop_initial, errors="ignore")

    # 1. Address match: 1 if shipping == billing, else 0
    if "Shipping Address" in df.columns and "Billing Address" in df.columns:
        df["Address Match"] = (
            df["Shipping Address"] == df["Billing Address"]
        ).astype(int)
        if verbose:
            match_rate = df["Address Match"].mean()
            print(f"Address Match feature created (match rate: {match_rate:.3f})")

    # 2. IP Features: Velocity and Linkage
    ip_req_cols = ["Transaction Date", "IP Address", "Transaction ID", "Customer ID"]
    if all(c in df.columns for c in ip_req_cols):
        # Sort by date for rolling windows to work correctly
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
        df = df.sort_values(by="Transaction Date").reset_index(drop=True)

        # IP Velocity: Transactions from this IP in the last 1 hour
        df_time = df.set_index("Transaction Date")
        df["IP_Transaction_Count_1H"] = df_time.groupby("IP Address")["Transaction ID"].rolling("1h").count().values
        
        # Linkage: Total Unique Customers per IP
        ip_user_counts = df.groupby("IP Address")["Customer ID"].nunique().reset_index()
        ip_user_counts.rename(columns={"Customer ID": "Unique_Customers_Per_IP"}, inplace=True)
        df = df.merge(ip_user_counts, on="IP Address", how="left")
        
        # Restore normal index after merging
        df = df.reset_index(drop=True)
        if verbose:
            print("IP features (Velocity & Linkage) created")

    # 4. Cyclical encoding of Transaction Hour
    if "Transaction Hour" in df.columns:
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Transaction Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Transaction Hour"] / 24)
        if verbose:
            print("Cyclical Hour_Sin / Hour_Cos features created")

    # Drop the original columns we just replaced/extracted data from
    drop_after = [c for c in COLS_TO_DROP_AFTER_ENGINEERING if c in df.columns]
    df = df.drop(columns=drop_after, errors="ignore")

    # One-hot encode nominal categoricals
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
      x_train      — normal transactions only (70% of normals)
      val_df, test_df  — normal + fraud combined (15% each)
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. "
                         f"Available: {list(df.columns)}")

    normal = df[df[TARGET_COL] == 0].drop(columns=[TARGET_COL])
    fraud = df[df[TARGET_COL] == 1].drop(columns=[TARGET_COL])

    if verbose:
        print(f"\nTotal normal: {len(normal)}, Total fraud: {len(fraud)}")

    normal_train, normal_temp = train_test_split(
        normal, test_size=0.30, random_state=SEED
    )
    normal_val, normal_test = train_test_split(
        normal_temp, test_size=0.50, random_state=SEED
    )

    fraud_val, fraud_test = train_test_split(
        fraud, test_size=0.50, random_state=SEED
    )

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
    schema_name: str = "feature_schema.pkl", # Added flexibility here
    verbose: bool = True,
):
    """
    Full preprocessing pipeline: load CSV, engineer features, split, scale,
    save everything to disk.
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
    with open(models_path / schema_name, "wb") as f:
        pickle.dump(schema, f)

    if verbose:
        print(f"\nSaved processed arrays to {processed_path}/")
        print(f"Saved scaler to {models_path}/scaler.pkl")
        print(f"Saved schema to {models_path}/{schema_name}")

    return {
        "x_train": x_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "input_dim": x_train.shape[1],
    }


def load_processed(processed_dir: str = "data/processed",
                   models_dir: str = "models",
                   schema_name: str = "feature_schema.pkl"): # Added schema parameter
    """
    Load previously-processed arrays from disk.
    """
    processed_path = Path(processed_dir)
    models_path = Path(models_dir)

    train = np.load(processed_path / "train.npz")
    val = np.load(processed_path / "val.npz")
    test = np.load(processed_path / "test.npz")

    scaler = joblib.load(models_path / "scaler.pkl")
    
    # Now dynamically loads whichever schema name you pass in
    with open(models_path / schema_name, "rb") as f:
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
    """
    if schema["target_col"] in df.columns:
        df = df.drop(columns=[schema["target_col"]])

    df = engineer_features(df, verbose=False)

    for col in schema["feature_names"]:
        if col not in df.columns:
            df[col] = 0

    df = df[schema["feature_names"]]

    x = df.values.astype("float32")
    x = scaler.transform(x).astype("float32")
    return x