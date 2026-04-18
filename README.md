# Deep Learning for Anomaly Detection in E-commerce Financial Transactions

A PyTorch implementation of autoencoder-based fraud detection on e-commerce transaction data, compared against a supervised DNN and tabular SOTA baselines (XGBoost, LightGBM).

## Project Structure

```
dl-project/
├── README.md
├── requirements.txt
├── CONTRIBUTIONS.md
├── data/
│   ├── Fraudulent_E-Commerce_Transaction_Data.csv   ← download from Kaggle
│   └── processed/                                   ← auto-created by Notebook 0
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── src/                        library code (imported by all notebooks)
│   ├── dataset.py              PyTorch Dataset classes
│   ├── preprocessing.py        load, encode, scale, split, save/load arrays
│   ├── autoencoder.py          MAIN MODEL (vanilla + denoising variants)
│   ├── model.py                supervised DNN for comparison
│   ├── train.py                training loops + early stopping
│   ├── evaluate.py             metrics + threshold tuning + failure cases
│   ├── baselines.py            SOTA baselines (LR, RF, XGBoost, LightGBM)
│   └── utils.py                seeding, plotting, checkpointing
├── scripts/                    command-line entry points (alternative to notebooks)
│   └── load_and_predict.py
├── notebooks/                  main deliverable — run in order
│   ├── 0_Preprocessing.ipynb   clean data, engineer features, save arrays
│   ├── 1_Autoencoder_Baseline.ipynb   train + evaluate baseline autoencoder
│   ├── 2_Autoencoder_tuned.ipynb      compare 4 variants, save best model
│   ├── 3_Supervised_DNN.ipynb         supervised DNN for comparison
│   └── 4_Final_Comparison.ipynb       full comparison table + plots
├── experiments/
│   ├── results.csv             logged metrics for every training run
│   └── final_comparison.csv    final table used in report
├── models/                     saved checkpoints (auto-created)
│   ├── scaler.pkl
│   ├── feature_schema.pkl
│   ├── autoencoder_baseline.pt
│   ├── autoencoder_best.pt     ← best variant from Notebook 2
│   └── supervised_best.pt
└── figures/                    all plots (auto-created by notebooks)
```

## Dependencies

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Framework:** PyTorch

## Getting the Data

1. Download from https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
2. Place the CSV inside the `data/` folder

## Running the Project — Notebooks (recommended)

Run notebooks in order. Each one depends on the outputs of the previous.

```
notebooks/0_Preprocessing.ipynb        ← Run first. Saves processed arrays to disk.
notebooks/1_Autoencoder_Baseline.ipynb ← Train and evaluate baseline autoencoder.
notebooks/2_Autoencoder_tuned.ipynb    ← Compare 4 variants. Saves best model.
notebooks/3_Supervised_DNN.ipynb       ← Train supervised DNN for comparison.
notebooks/4_Final_Comparison.ipynb     ← Full comparison table + plots.
```

> **Important:** Notebook 0 must be run before any other notebook. It saves
> the processed arrays that all other notebooks load at startup.


## What the Preprocessing Does

Implemented in `src/preprocessing.py` (matches `notebooks/0_Preprocessing.ipynb`):

1. Drops non-informative columns: `Transaction ID`, `Customer ID` (verified unique), `IP Address`, `Transaction Date`, `Customer Location` (high cardinality)
2. Engineers `Address Match`: binary feature (1 if shipping == billing) — a classic fraud signal
3. Cyclically encodes `Transaction Hour` as `Hour_Sin`/`Hour_Cos` so hour 23 and hour 0 are close in feature space
4. One-hot encodes `Payment Method`, `Product Category`, `Device Used` with `drop_first=True`
5. Splits normal and fraud separately: train = 70% normal only, val/test = 15% each with both classes
6. Fits `StandardScaler` on train only, applies to val and test
7. Saves arrays to `data/processed/` and scaler/schema to `models/`

## How the Autoencoder Works

1. **Train** on normal transactions only — learns to reconstruct the normal pattern
2. **At inference**, compute reconstruction error (MSE per sample)
3. **High error** = transaction doesn't match the normal pattern = likely fraud
4. **Threshold**: 95th percentile of normal-sample reconstruction errors on val set (unsupervised — no fraud labels needed)

## Loading a Saved Model (No Retraining)

```bash
python scripts/load_and_predict.py \
    --model models/autoencoder_best.pt \
    --csv data/new_transactions.csv \
    --model_type autoencoder \
    --output predictions.csv
```

Output CSV contains the original columns plus `reconstruction_error` and `predicted_fraud`.

## Figures Generated

| Figure | From | Path |
|--------|--------------|------|
| Training curves (autoencoder) | Notebook 1 | `figures/fig01_baseline_training_curves.png` |
| Reconstruction error distribution | Notebook 1 | `figures/fig02_baseline_error_dist.png` |
| Threshold sweep | Notebook 1 | `figures/fig03_baseline_threshold_sweep.png` |
| Confusion matrix (autoencoder) | Notebook 1 | `figures/fig04_baseline_confusion_matrix.png` |
| PR curve (autoencoder) | Notebook 1 | `figures/fig05_baseline_pr_curve.png` |
| Variant comparison | Notebook 2 | `figures/variant_comparison.png` |
| Training curves (DNN) | Notebook 3 | `figures/fig06_supervised_training_curves.png` |
| Confusion matrix (DNN) | Notebook 3 | `figures/fig07_supervised_confusion_matrix.png` |
| PR curve (DNN) | Notebook 3 | `figures/fig08_supervised_pr_curve.png` |
| All models comparison | Notebook 4 | `figures/fig09_all_models_comparison.png` |

## Reproducibility

- All random seeds set via `src/utils.py::set_seed(42)`
- `torch.backends.cudnn.deterministic = True` enabled
- Every training run logged to `experiments/results.csv`
- Preprocessor (fitted scaler + feature schema) saved alongside the model
- `load_and_predict.py` can reproduce any prediction without retraining

## Contributions

See `CONTRIBUTIONS.md`.
