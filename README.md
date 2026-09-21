# BLACKLEDGER — Financial Fraud Detection & Investigation Engine

> "If the transaction looks innocent, examine it twice."

A professional desktop application for financial fraud analytics, built entirely
in Python (Tkinter, pandas, scikit-learn, matplotlib, seaborn, SQLite, joblib).

## Architecture

The application follows a **Model–View–Controller (MVC)** pattern:

- **Model** — `core/` modules (data loading, cleaning, feature engineering,
  imbalance handling, model training, evaluation, anomaly detection, risk
  scoring, exporting) and `database/` (SQLite persistence).
- **View** — `views/` (Tkinter Notebook with 10 tabs, dark Wednesday Addams
  theme).
- **Controller** — `controllers/app_controller.py` orchestrates the pipeline
  and mediates between Model and View.

### Pipeline
DATA IMPORT → VALIDATION → CLEANING → EDA → FEATURE ENGINEERING
→ IMBALANCE ANALYSIS → MODEL TRAINING → MODEL EVALUATION
→ ANOMALY DETECTION → RISK SCORING → INVESTIGATION DASHBOARD


## Dataset

The application accepts **any CSV** with a binary fraud label. It has been
designed with the following Kaggle datasets in mind:

- **Credit Card Fraud Detection** (ULB) — `Time`, `V1`–`V28` (PCA), `Amount`,
  `Class`[reference:0].
- **Online Payment Fraud (PaySim)** — `step`, `type`, `amount`,
  `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`,
  `newbalanceDest`, `isFraud`[reference:1].
- **Synthetic sample** — generated in-app for demonstration.

## Methodology

### Class Imbalance

The default strategy is **SMOTE** on the training set only (never on the test
set). Alternatives include SMOTETomek and `class_weight='balanced'`. The
imbalance ratio and severity are reported during validation[reference:2].

### Models

| Model | Key Parameters |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=2000` |
| Decision Tree | `class_weight='balanced'`, `max_depth=8` |
| Random Forest | `n_estimators=200`, `class_weight='balanced'` |
| Gradient Boosting | `n_estimators=150`, `learning_rate=0.1` |

### Evaluation

Accuracy is **not** reported as a primary metric. For highly imbalanced fraud
data, ROC-AUC can mask operational differences — **PR-AUC**, **F1**, and
**recall** are the primary indicators[reference:3].

### Anomaly Detection

- **Isolation Forest** (unsupervised) — flags statistically unusual
  transactions[reference:4].
- **Statistical z-score** — flags transactions with |z| > 3 on amount.
- **Combined flag** — union of both methods.

> **Critical distinction:** *"Potential anomaly"* ≠ *"Confirmed fraud"*. Anomaly
> flags prioritise human investigation; they do **not** prove criminal activity.

### Feature Engineering

All engineered features are documented in-app with their rationale. Key
features include `amount_deviation` (z-score vs. customer history),
`tx_velocity_hourly` (transactions per hour), `time_since_prev`, and
`amount_to_balance`[reference:5].

## Installation

```bash
pip install -r requirements.txt
python main.py