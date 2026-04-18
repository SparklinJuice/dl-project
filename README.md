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
│   ├── processed/                                   ← auto-created by Notebook 0a
│   └── processed_withIP/                            ← auto-created by Notebook 0b
├── src/                            library code (imported by all notebooks)
│   ├── dataset.py                  PyTorch Dataset classes
│   ├── preprocessing.py            feature engineering, split, scale, save/load
│   ├── preprocessing_withIP.py     same as above but with IP-derived features
│   ├── autoencoder.py              MAIN MODEL (vanilla + denoising variants)
│   ├── model.py                    supervised DNN for comparison
│   ├── train.py                    training loops + early stopping
│   ├── evaluate.py                 metrics, threshold tuning, failure case analysis
│   ├── baselines.py                SOTA baselines (LR, RF, XGBoost, LightGBM)
│   └── utils.py                    seeding, plotting, checkpointing
├── notebooks/                      main deliverable — run in order
│   ├── 0a_Preprocessing.ipynb                  feature engineering, split, save arrays
│   ├── 0b_Preprocessing_withIP.ipynb           same with IP-derived features
│   ├── 1a_Autoencoder_Baseline.ipynb           baseline autoencoder + failure analysis
│   ├── 1b_Autoencoder_Baseline_withIP.ipynb    same with IP features
│   ├── 2a_Autoencoder_tuned.ipynb              compare 4 variants, save best model
│   ├── 2b_Autoencoder_tuned_withIP.ipynb       same with IP features
│   ├── 3a_Supervised_DNN.ipynb                 supervised DNN baseline
│   ├── 3b_DNN_Tuning.ipynb                    DNN hyperparameter tuning (3 passes)
│   └── 4_Final_Comparison.ipynb               all-models comparison table + plots
├── scripts/                        CLI entry points
│   ├── preprocess_data.py          preprocessing
│   ├── train_autoencoder.py        autoencoder training
│   ├── train_supervised.py         DNN training
│   ├── run_baselines.py            SOTA baselines
│   ├── hyperparameter_sweep.py     grid search
│   └── load_and_predict.py         load saved model, run inference
├── models/                         saved checkpoints (auto-created)
│   ├── scaler.pkl                  fitted StandardScaler
│   ├── feature_schema.pkl          column names + metadata
│   ├── autoencoder_best.pt         best autoencoder from Notebook 2a
│   ├── autoencoder_best_withIP.pt  best autoencoder from Notebook 2b
│   ├── supervised_best.pt          supervised DNN from Notebook 3a
│   └── supervised_tuned.pt         tuned DNN from Notebook 3b
├── experiments/                    CSV logs of every training run
├── figures/                        all plots (auto-created by notebooks)
└── report/
    └── project_report.pdf
```

## Dependencies

Python 3.10+ recommended. Tested on Python 3.10.19.
Framework: PyTorch 2.2.0. No TensorFlow or Keras used.

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Contents of requirements.txt:
```
torch==2.2.0
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.2
xgboost==2.0.3
lightgbm==4.3.0
jupyter==1.0.0
```

## Getting the Data

Download from https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions and place the CSV inside data/.

Or use the Kaggle CLI:
```bash
pip install kaggle
cd data && kaggle datasets download -d shriyashjagtap/fraudulent-e-commerce-transactions && unzip *.zip && cd ..
```

## Running the Project

### Notebooks 

Run notebooks in order. Each depends on outputs of the previous.

```
0a_Preprocessing.ipynb           ← Run FIRST. Saves processed arrays to data/processed/
0b_Preprocessing_withIP.ipynb    ← Optional. With IP-derived features
1a_Autoencoder_Baseline.ipynb    ← Train + evaluate baseline autoencoder + failure analysis
1b_Autoencoder_Baseline_withIP.ipynb
2a_Autoencoder_tuned.ipynb       ← Compare 4 architecture variants, save best
2b_Autoencoder_tuned_withIP.ipynb
3a_Supervised_DNN.ipynb          ← Train supervised DNN (comparison model)
3b_DNN_Tuning.ipynb              ← Hyperparameter tuning: architecture, regularization, LR
4_Final_Comparison.ipynb         ← All-models comparison: AE vs DNN vs XGBoost vs LightGBM
```


## Preprocessing

Documented in notebooks/0a_Preprocessing.ipynb, implemented in src/preprocessing.py:

1. Drops non-informative columns: Transaction ID, Customer ID (unique), IP Address, Transaction Date, Customer Location (~99k unique)
2. Engineers Address Match: binary feature (1 if shipping == billing). 90% match rate; mismatches are a fraud signal
3. Cyclically encodes Transaction Hour as Hour_Sin/Hour_Cos
4. One-hot encodes Payment Method, Product Category, Device Used with drop_first=True
5. Splits normal/fraud separately: train = 70% normals only, val/test = 15% each with both classes
6. Fits StandardScaler on train only, applies to val and test
7. Saves arrays to data/processed/ and scaler/schema to models/

## How the Autoencoder Works

1. Train on normal transactions only — learns to reconstruct the normal pattern
2. At inference, compute reconstruction error (MSE per sample)
3. High error = transaction deviates from normal = likely fraud
4. Threshold: 95th percentile of normal reconstruction errors on validation set

## Key Results

| Model | Type | PR-AUC | F1 | ROC-AUC |
|-------|------|--------|-----|---------|
| LightGBM | Supervised (ML) | 0.5725 | 0.4879 | — |
| Supervised DNN | Supervised (DL) | 0.5708 | 0.5284 | 0.8132 |
| XGBoost | Supervised (ML) | 0.5671 | 0.5057 | — |
| Logistic Regression | Supervised (ML) | 0.4941 | 0.4116 | — |
| AE without IP [128,64,32] | Unsupervised | 0.3911 | 0.3242 | 0.6939 |
| AE with IP [14,10,6] | Unsupervised | 0.3889 | 0.3298 | — |
| Random baseline | — | 0.1496 | — | 0.5000 |

## Loading a Saved Model (No Retraining)

```bash
python scripts/load_and_predict.py \
    --model models/autoencoder_best.pt \
    --csv data/Fraudulent_E-Commerce_Transaction_Data.csv \
    --model_type autoencoder \
    --output predictions.csv
```

Or programmatically:
```python
from src.autoencoder import build_autoencoder
from src.utils import load_checkpoint, get_device

device = get_device()
checkpoint = load_checkpoint('models/autoencoder_best.pt', device)
meta = checkpoint['metadata']

model = build_autoencoder(
    model_type=meta['model_type'],
    input_dim=meta['input_dim'],
    hidden_dims=meta['hidden_dims'],
    dropout=meta['dropout'],
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

threshold = meta['threshold_unsupervised']
```

## Figures

| Figure | Source | Path |
|--------|--------|------|
| Training curves (AE) | Notebook 1a | figures/fig01_baseline_training_curves.png |
| Error distribution | Notebook 1a | figures/fig02_baseline_error_distribution.png |
| Threshold sweep | Notebook 1a | figures/fig03_baseline_threshold_sweep.png |
| Confusion matrix (AE) | Notebook 1a | figures/fig04_baseline_confusion_matrix.png |
| PR curve (AE) | Notebook 1a | figures/fig05_baseline_pr_curve.png |
| Variant comparison | Notebook 2a | figures/variant_comparison.png |
| Training curves (DNN) | Notebook 3a | figures/fig06_supervised_training_curves.png |
| Confusion matrix (DNN) | Notebook 3a | figures/fig07_supervised_confusion_matrix.png |
| PR curve (DNN) | Notebook 3a | figures/fig08_supervised_pr_curve.png |
| All models comparison | Notebook 4 | figures/fig09_all_models_comparison.png |
| DNN tuning | Notebook 3b | figures/fig10_dnn_tuning.png |

## Reproducibility

- Seeds set to 42 via src/utils.py::set_seed(42)
- torch.backends.cudnn.deterministic = True
- Every run logged to experiments/results.csv
- Scaler + feature schema saved alongside model
- MPS (Apple Silicon), CUDA, and CPU all auto-detected

## Contributions

See CONTRIBUTIONS.md.