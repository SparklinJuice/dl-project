"""
Diagnostic: inspect what the preprocessor actually produces.

If reconstruction MSE is in the billions, it almost certainly means one of the
feature columns didn't get scaled. This script tells you which one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_and_preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    args = p.parse_args()

    data = load_and_preprocess(args.csv, verbose=True)

    X = data["X_train"]
    names = data["feature_names"]

    print("\n" + "=" * 70)
    print("  Per-feature statistics AFTER preprocessing")
    print("=" * 70)
    print(f"{'#':>3}  {'feature':<30}  {'min':>12}  {'max':>12}  {'mean':>10}  {'std':>10}")
    print("-" * 90)

    problem_cols = []
    for i, name in enumerate(names):
        col = X[:, i]
        mn, mx, mean, std = col.min(), col.max(), col.mean(), col.std()
        flag = ""
        # Standard-scaled features should have mean~0, std~1
        # Label-encoded categoricals will have integer values, std ~ few
        # Anything with std > 100 is suspicious
        if std > 100 or abs(mean) > 100:
            flag = "  ← NOT SCALED"
            problem_cols.append(name)
        print(f"{i:>3}  {name:<30}  {mn:>12.2f}  {mx:>12.2f}  {mean:>10.2f}  {std:>10.2f}{flag}")

    print("\n" + "=" * 70)
    if problem_cols:
        print(f"  PROBLEM DETECTED: {len(problem_cols)} column(s) not scaled properly")
        print(f"  Columns: {problem_cols}")
        print("\n  This is why your MSE is huge.")
        print("  The preprocessor classified these as categorical (because they're")
        print("  int64 with <30 unique values), so they were label-encoded instead")
        print("  of standard-scaled. But their raw values are still large.")
    else:
        print("  All features look scaled correctly.")
    print("=" * 70)


if __name__ == "__main__":
    main()