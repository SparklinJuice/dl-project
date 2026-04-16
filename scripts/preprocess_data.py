"""
Run preprocessing once and save all processed arrays to disk.

After this runs, all training scripts can load the prepared arrays
instantly via load_processed() instead of re-running preprocessing
on 1.47M rows every time.

Usage:
    python scripts/preprocess_data.py --csv data/Fraudulent_E-Commerce_Transaction_Data.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import run_preprocessing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to raw CSV")
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--models_dir", type=str, default="models")
    args = p.parse_args()

    run_preprocessing(
        csv_path=args.csv,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
    )

    print("\nPreprocessing complete. You can now run training scripts,")
    print("which will load the processed arrays from disk instantly.")


if __name__ == "__main__":
    main()
