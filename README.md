# Deep Learning for Unsupervised Anomaly Detection in Financial Transactions

A PyTorch implementation of autoencoder-based fraud detection on e-commerce transaction data, compared against a supervised DNN and tabular SOTA baselines (XGBoost, LightGBM).

## Project Structure
TO BE UPDATED!!
```
fraud-detection-dl/
├── README.md                   this file
├── requirements.txt            pinned package versions
├── CONTRIBUTIONS.md            group member contributions
├── data/
│   └── download_data.sh        fetches the dataset
├── src/                        library code
│   ├── dataset.py              PyTorch Dataset classes
│   ├── preprocessing.py        load, encode, scale, split
│   ├── autoencoder.py          MAIN MODEL (vanilla + denoising)
│   ├── model.py                supervised DNN for comparison
│   ├── train.py                training loops + early stopping
│   ├── evaluate.py             metrics + threshold tuning
│   ├── baselines.py            SOTA baselines (LR, RF, XGBoost, LightGBM)
│   └── utils.py                seeding, plotting, checkpointing
├── scripts/                      command-line entry points
│   ├── preprocess_data.py        run preprocessing once, save arrays
│   ├── train_autoencoder.py      train main model
│   ├── train_supervised.py       train DNN comparison
│   ├── run_baselines.py          run all SOTA baselines
│   ├── hyperparameter_sweep.py   grid search over architecture/reg/opt
│   └── load_and_predict.py       load saved model, run inference
├── notebooks/                    exploration and analysis
│   └── 0_Preprocessing.ipynb     documented preprocessing logic
├── data/
│   ├── download_data.sh          fetches the dataset
│   └── processed/                saved train/val/test npz files
├── experiments/                  CSV logs of every training run
├── models/                       saved checkpoints, scaler, schema
├── figures/                      plots for the report
└── report/                       final PDF deliverable
```

## Dependencies

All packages and pinned versions are in `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Framework:** PyTorch 2.2.0. No TensorFlow or Keras is used anywhere in this project.

## Getting the Data

    1. Download csv from https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions. 

    2. Place downloaded csv file inside 'data' folder.  

This requires the Kaggle CLI (`pip install kaggle` and valid `~/.kaggle/kaggle.json`). If you already have the CSV, place it in `data/` with any name — every script takes a `--csv` argument.

## Quick Reproduction

The pipeline is now two-stage: preprocess once, then train many times from the saved arrays.

```bash
# 1. Preprocess ONCE (slow: ~1-2 min on M2 for 1.47M rows)
#    Engineers features (Address Match, cyclical Hour), one-hot encodes,
#    splits, scales, and saves processed arrays to disk.
python scripts/preprocess_data.py --csv data/Fraudulent_E-Commerce_Transaction_Data.csv

# 2. Train the main model (autoencoder) — loads processed arrays instantly
python scripts/train_autoencoder.py --processed

# 3. Train the supervised DNN comparison
python scripts/train_supervised.py --processed

# 4. Run the SOTA baselines
python scripts/run_baselines.py --processed

# 5. Load the saved autoencoder and run inference (reproducibility check)
python scripts/load_and_predict.py \
    --model models/autoencoder_best.pt \
    --csv data/Fraudulent_E-Commerce_Transaction_Data.csv \
    --model_type autoencoder
```

Each training script saves its model and metrics automatically. Results are logged to `experiments/results.csv`. The scaler and feature schema needed for inference are saved by the preprocessing step to `models/scaler.pkl` and `models/feature_schema.pkl`.

Every script also accepts `--csv <path>` instead of `--processed` to run preprocessing inline (useful for debugging or a fresh start).

## Hyperparameter Sweep

After preprocessing, run the full hyperparameter sweep:

```bash
# Full sweep (5 + 16 + 12 = 33 runs, several hours on M2)
python scripts/hyperparameter_sweep.py --processed --epochs 60

# Just one pass (for debugging)
python scripts/hyperparameter_sweep.py --processed --pass 1
```

The sweep runs three passes sequentially: architecture → regularization → optimization. Each pass uses the best config from the previous pass. All results are logged to `experiments/sweep_results.csv` and the final best config is saved to `experiments/best_config.json`.

## Preprocessing — What It Does

Implemented in `src/preprocessing.py` (matches `notebooks/0_Preprocessing.ipynb`):

1. **Drops non-informative columns**: `Transaction ID`, `Customer ID` (verified unique), `IP Address`, `Transaction Date`, `Customer Location` (high cardinality).
2. **Engineers `Address Match`**: binary feature (1 if shipping == billing, 0 otherwise) — a classic fraud signal. Original address columns are then dropped.
3. **Cyclically encodes `Transaction Hour`** as `Hour_Sin` and `Hour_Cos` so hour 23 and hour 0 are close in feature space.
4. **One-hot encodes** `Payment Method`, `Product Category`, `Device Used` with `drop_first=True` to avoid multicollinearity.
5. **Splits normal and fraud separately** before recombining. Training set = 70% of normal transactions only (autoencoder never sees fraud during training). Val and test = 15% each, containing both classes.
6. **Fits `StandardScaler` on train only**, applies to val and test.
7. **Saves** `data/processed/{train,val,test}.npz`, `models/scaler.pkl`, `models/feature_schema.pkl`.

## Training from Scratch — Step by Step

The autoencoder is the main model. Here's exactly what happens:

1. **Preprocess** the raw CSV as described above.
2. **Build the autoencoder.** Default architecture: encoder `[input → 64 → 32 → 16]`, decoder `[16 → 32 → 64 → input]`, with BatchNorm and Dropout between layers.
3. **Train** with MSE reconstruction loss on normal transactions only, Adam optimizer, `ReduceLROnPlateau` scheduler, and early stopping (patience 10) restoring the best validation weights.
4. **Pick a threshold.** Primary (unsupervised) method: 95th percentile of normal-sample reconstruction errors on val. Secondary F1-optimal threshold also reported.
5. **Evaluate on test set** using both thresholds. All metrics (precision, recall, F1, ROC-AUC, PR-AUC) are printed and logged.
6. **Save** model weights, metadata, and chosen threshold to `models/autoencoder_best.pt`.

To retrain with different hyperparameters:

```bash
python scripts/train_autoencoder.py \
    --csv data/fraud_data.csv \
    --model_type denoising \
    --hidden_dims 128 64 32 16 \
    --dropout 0.3 \
    --lr 5e-4 \
    --epochs 200 \
    --threshold_percentile 97
```

## Loading a Saved Model (No Retraining)

The `load_and_predict.py` script loads the checkpoint, scaler, and feature schema, reconstructs the model, and runs inference — without touching any training code.

```bash
python scripts/load_and_predict.py \
    --model models/autoencoder_best.pt \
    --csv data/new_transactions.csv \
    --model_type autoencoder \
    --output predictions.csv
```

The `--scaler` and `--schema` arguments default to `models/scaler.pkl` and `models/feature_schema.pkl`, so you normally don't need to specify them.

The output CSV contains the original rows plus `reconstruction_error` and `predicted_fraud` columns.

## Figures in the Report

Every figure is generated by code in this repo. The mapping:

| Figure | Generated by | Output path |
|--------|--------------|-------------|
| Training curves (autoencoder) | `scripts/train_autoencoder.py` | `figures/autoencoder_training_curves.png` |
| Reconstruction error distribution | `scripts/train_autoencoder.py` | `figures/autoencoder_error_distribution.png` |
| PR curve (autoencoder) | `scripts/train_autoencoder.py` | `figures/autoencoder_pr_curve.png` |
| Training curves (DNN) | `scripts/train_supervised.py` | `figures/supervised_training_curves.png` |
| Confusion matrix (DNN) | `scripts/train_supervised.py` | `figures/supervised_confusion_matrix.png` |
| PR curve (DNN) | `scripts/train_supervised.py` | `figures/supervised_pr_curve.png` |
| EDA plots | `notebooks/01_EDA.ipynb` | `figures/fig_eda_*.png` |
| Hyperparameter sweep | `notebooks/03_autoencoder_tuning.ipynb` | `figures/fig_tuning_*.png` |
| Failure case analysis | `notebooks/04_final_evaluation.ipynb` | `figures/fig_failures_*.png` |

Re-running any script regenerates its figures. The `--fig_dir` argument controls where they're written.

## Reproducibility Notes

- All random seeds are set in `src/utils.py::set_seed()` (defaults to 42).
- `torch.backends.cudnn.deterministic = True` is enabled.
- Every training run is logged to `experiments/results.csv` with hyperparameters and metrics.
- Preprocessor (fitted scaler + encoders) is saved alongside the model.

## Contributions

See `CONTRIBUTIONS.md`.
