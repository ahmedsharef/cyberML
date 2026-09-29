# 🛡️ ML-Based Cyberattack Detection and Classification System

> **Final-year Major Project — VTU 7th Semester CSE**
> A production-ready, demo-ready end-to-end machine learning system for detecting and classifying network cyberattacks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Adding the Datasets](#adding-the-datasets)
7. [Running the Training Pipeline](#running-the-training-pipeline)
8. [Launching the Dashboard](#launching-the-dashboard)
9. [Running Tests](#running-tests)
10. [Label Mapping / Config](#label-mapping--config)
11. [Models Trained](#models-trained)
12. [Screenshots](#screenshots)
13. [Viva Q&A Guide](#viva-qa-guide)

---

## Project Overview

This system detects whether network traffic is **normal or malicious**, and if malicious, classifies it into one of **7 attack categories**:

| Category | Examples |
|---|---|
| **DDoS** | SYN flood, UDP flood, Smurf, Neptune |
| **Malware** | Rootkit, buffer overflow, backdoor, worm |
| **Phishing** | Credential harvesting, spear phishing |
| **Ransomware** | File-encrypting ransomware variants |
| **SQL Injection** | Classic SQLi, blind SQLi |
| **MITM** | ARP spoofing, HTTP tunnelling |
| **Other/Unknown** | Reconnaissance, probe, generic |

### Two-Stage Pipeline

```
Traffic Record → [Stage 1: Binary Detector] → Normal? → Done
                                              ↓ Attack?
                            [Stage 2: Multi-class Classifier]
                                              ↓
                         {ddos | malware | phishing | ransomware | ...}
```

---

## Architecture

```
Raw Data (NSL-KDD / UNSW-NB15)
    │
    ▼
Preprocessing Pipeline (src/preprocessing.py)
    • Missing value imputation
    • Categorical encoding (LabelEncoder)
    • Feature normalisation (StandardScaler)
    │
    ▼
Feature Engineering (src/feature_engineering.py)
    • Low-variance removal
    • High-correlation removal
    • RandomForest importance ranking → top-30 features
    │
    ▼
SMOTE Oversampling (imbalanced-learn)
    │
    ▼
Model Training × 5 (src/train_models.py)
    • Random Forest  • Decision Tree  • SVM  • XGBoost  • MLP
    │
    ▼
Evaluation (src/evaluate.py)
    • Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
    • Auto-select best model → models/best_model.txt
    │
    ▼
Two-Stage Predict Pipeline (src/predict.py)
    │
    ├──▶ SQLite DB (src/db.py)         — traffic log, alerts, notifications
    ├──▶ Alerting Module (src/alerting.py) — severity scoring + SMTP email
    └──▶ Streamlit Dashboard (dashboard/)  — 8 pages, real-time feed, KPIs
```

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.10+ |
| ML / Data | scikit-learn 1.4, XGBoost 2.0, pandas, numpy |
| Class balance | imbalanced-learn (SMOTE) |
| Dashboard | Streamlit 1.35 |
| Visualisation | Plotly (interactive), matplotlib + seaborn (static plots) |
| Database | SQLite via SQLAlchemy 2.0 |
| Model persistence | joblib |
| Alerting | smtplib + python-dotenv |
| Testing | pytest |

---

## Project Structure

```
cyberattack-detection/
├── README.md
├── requirements.txt
├── .env.example
├── Makefile
├── config/
│   └── label_mapping.yaml      ← edit attack→category mappings here
├── data/
│   ├── raw/                    ← drop NSL-KDD / UNSW-NB15 files here
│   ├── processed/              ← auto-generated after training
│   ├── cyberattack.db          ← SQLite database (auto-created)
│   └── download_instructions.md
├── src/
│   ├── config.py               ← all config loaded from .env + yaml
│   ├── preprocessing.py        ← data loading, encoding, splitting
│   ├── feature_engineering.py  ← variance/correlation/importance selection
│   ├── train_models.py         ← train all 5 models end-to-end
│   ├── evaluate.py             ← metrics, plots, comparison report
│   ├── predict.py              ← two-stage inference pipeline
│   ├── db.py                   ← SQLite schema + CRUD
│   └── alerting.py             ← severity scoring + email alerts
├── models/                     ← saved .joblib model files
│   └── best_model.txt          ← name of the auto-selected best model
├── reports/
│   ├── model_comparison_report.md
│   └── figures/                ← confusion matrices, ROC curves, bar charts
├── dashboard/
│   ├── app.py                  ← streamlit entry point
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Live_Monitoring.py
│       ├── 3_Attack_Distribution.py
│       ├── 4_Traffic_Over_Time.py
│       ├── 5_Alerts_Reports.py
│       ├── 6_Model_Performance.py
│       ├── 7_Alert_Response.py
│       └── 8_Settings.py
└── tests/
    ├── test_preprocessing.py
    └── test_predict.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) a virtual environment

### Steps

```bash
# 1. Clone / navigate to the project folder
cd cyberattack-detection

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment file
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac

# 5. Edit .env to set DATASET and optional SMTP credentials
```

---

## Adding the Datasets

See **`data/download_instructions.md`** for full URLs and steps.

**Quick summary:**

### NSL-KDD

1. Download from https://www.unb.ca/cic/datasets/nsl.html
2. Place these files in `data/raw/`:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`

### UNSW-NB15

1. Download from https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Place these files in `data/raw/`:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

### Set the active dataset in `.env`

```ini
DATASET=nsl_kdd    # or: DATASET=unsw_nb15
```

**Verify:**
```bash
python -c "from src.preprocessing import load_raw_data; df = load_raw_data(); print(df.shape)"
```

---

## Running the Training Pipeline

This single command does everything: load data → preprocess → feature select → SMOTE → train 5 models → evaluate → save best model → generate report + plots.

```bash
python -m src.train_models
```

**Expected outputs:**
- `data/processed/` — preprocessed arrays and encoder metadata
- `models/` — 10 `.joblib` files (binary + multi for each model) + `best_model.txt`
- `reports/model_comparison_report.md` — full metrics table
- `reports/figures/` — confusion matrices, ROC curves, comparison bar charts

Typical runtime: **5–20 minutes** depending on dataset size and hardware.

---

## Launching the Dashboard

```bash
streamlit run dashboard/app.py
```

The app opens at **http://localhost:8501**

### Dashboard Pages

| Page | What it shows |
|---|---|
| 📊 Overview | KPI cards: total traffic, attacks, accuracy |
| 📡 Live Monitoring | Simulated real-time feed replaying test-set rows |
| 🥧 Attack Distribution | Donut + bar chart of detected attack types |
| 📈 Traffic Over Time | Line chart of traffic volume over configurable window |
| 🚨 Alerts & Reports | Filterable alert table with acknowledge/resolve actions |
| 🤖 Model Performance | Accuracy gauge + confusion matrices + comparison charts |
| ⚡ Alert & Response | Open alert banner + simulated threat injection for demos |
| ⚙️ Settings | Change active model, thresholds, feed speed |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests use synthetic data — **no dataset files required**.

---

## Label Mapping / Config

`config/label_mapping.yaml` is the single source of truth for how raw dataset attack labels map to the 7 unified categories. Edit this file freely without touching any Python code.

Example:
```yaml
nsl_kdd:
  normal: normal
  neptune: ddos
  smurf: ddos
  rootkit: malware
  guess_passwd: phishing
  sqlattack: sql_injection
  httptunnel: mitm
  # ... etc.
```

---

## Models Trained

All 5 models are trained identically on the same preprocessed + SMOTE-balanced data:

| Model | Key hyperparameters |
|---|---|
| Random Forest | 200 trees, balanced class weights |
| Decision Tree | max_depth=15, balanced class weights |
| SVM | LinearSVC + CalibratedClassifierCV (for probabilities) |
| XGBoost | 200 estimators, lr=0.1, max_depth=6 |
| MLP (Neural Net) | Hidden layers: 128→64→32, early stopping |

The best model (by weighted F1) is auto-selected and saved as the production model.

---

## Screenshots

> Paste your screenshots here for your report.

### Dashboard – Overview
<!-- ![Overview](screenshots/overview.png) -->

### Dashboard – Live Monitoring
<!-- ![Live Monitoring](screenshots/live_monitoring.png) -->

### Dashboard – Attack Distribution
<!-- ![Attack Distribution](screenshots/attack_distribution.png) -->

### Dashboard – Model Performance
<!-- ![Model Performance](screenshots/model_performance.png) -->

### Training Output
<!-- ![Training](screenshots/training_output.png) -->

---

## Viva Q&A Guide

**Q: Why did you choose NSL-KDD and UNSW-NB15?**
A: NSL-KDD is the most widely cited network intrusion benchmark (fixes KDD'99 duplicates). UNSW-NB15 is more modern (2015) and covers contemporary attack types like backdoors, shellcode, and worms. Using both lets us validate generalisability.

**Q: Why two stages instead of a single multi-class classifier?**
A: In practice, the classes are severely imbalanced — normal traffic is the majority. Stage 1 handles the binary decision quickly and cleanly; Stage 2 only runs for flagged traffic, which improves precision on rare attack types and makes the system more explainable.

**Q: How do you handle class imbalance?**
A: SMOTE (Synthetic Minority Oversampling Technique) oversamples the minority attack classes in feature space. Combined with `class_weight="balanced"` in tree models, this prevents the classifier from defaulting to "normal" for everything.

**Q: What is feature selection doing?**
A: Three steps: (1) remove near-zero variance features — they carry no signal; (2) remove one of each highly-correlated pair (|r| > 0.95) to reduce redundancy; (3) rank remaining features by RandomForest importance and keep top-30. This speeds up training and reduces overfitting.

**Q: How are models evaluated fairly?**
A: All models use the same stratified 80/20 split with a fixed random seed (42) for reproducibility. Evaluation is on the held-out test set only — no data leakage from the training set.

**Q: What does the dashboard pull from?**
A: Every number on the dashboard comes from either (a) the SQLite database populated by real classifications or (b) the `model_comparison_report.md` written during training. There are no hardcoded metrics anywhere.
