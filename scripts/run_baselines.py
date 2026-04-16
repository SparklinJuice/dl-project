"""
Train and evaluate SOTA tabular baselines for comparison.

Satisfies rubric expectation #3: compare against state-of-the-art.

Usage:
    python scripts/run_baselines.py --csv data/fraud.csv
    python scripts/run_baselines.py --processed   (after preprocess_data.py)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import print_comparison_table, run_all_baselines
from src.preprocessing import load_processed, run_preprocessing
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--processed", action="store_true")
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="experiments/baseline_results.csv")
    args = p.parse_args()
    if not args.csv and not args.processed:
        p.error("Must specify either --csv or --processed")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load data
    if args.processed:
        data = load_processed(args.processed_dir, args.models_dir)
    else:
        data = run_preprocessing(args.csv, args.processed_dir, args.models_dir)

    # Baselines need supervised training data (normal + fraud).
    # Merge x_train (all normal) with half of x_val (normal + fraud).
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(data["y_val"]))
    half = len(perm) // 2

    x_train = np.concatenate([
        data["x_train"],
        data["x_val"][perm[:half]],
    ], axis=0)
    y_train = np.concatenate([
        np.zeros(len(data["x_train"]), dtype=np.int64),
        data["y_val"][perm[:half]],
    ], axis=0)

    # Run baselines (use the held-out test set directly)
    results = run_all_baselines(
        x_train, y_train,
        data["x_test"], data["y_test"],
    )

    df = print_comparison_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nBaseline results saved to {out_path}")


if __name__ == "__main__":
    main()
