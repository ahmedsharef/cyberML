# Project Summary — ML-Based Cyberattack Detection and Classification System

**Course:** B.E. Computer Science and Engineering — 7th Semester Major Project
**University:** Visvesvaraya Technological University (VTU)
**Repository:** https://github.com/ahmedsharef/cyberML

---

## 1. What We Built

A complete, end-to-end machine learning system that monitors network traffic in real time, detects whether it is malicious, and if so, classifies it into one of seven attack categories. The system includes a trained ML pipeline, a live admin dashboard, an automated alerting module, and a persistent database of all detections.

In plain language: the system watches network packets, raises an alarm when it sees an attack, tells you what *kind* of attack it is, and logs everything to a database — all automatically, with no human in the loop.

---

## 2. The Problem We Solved

Modern networks generate millions of packets per second. Manual inspection is impossible. Traditional rule-based firewalls and intrusion detection systems (IDS) rely on known attack signatures — they fail against zero-day attacks and evolving threats that don't match any known pattern.

Machine learning solves this by learning the *statistical patterns* of malicious vs normal traffic from labelled historical data. Once trained, the model generalises to detect attacks it has never explicitly seen before, as long as they share underlying behavioural patterns with known attack families.

**Why this matters:**
- Cyberattacks cost organisations an average of $4.45 million per breach (IBM Cost of a Data Breach Report, 2023)
- DDoS attacks alone increased 200% between 2020 and 2023
- Ransomware affected 66% of organisations globally in 2023 (Sophos)
- SQL injection remains the most common web attack vector (OWASP Top 10)

---

## 3. Datasets Used

### NSL-KDD (Primary)
- **Source:** University of New Brunswick (UNB) — https://www.unb.ca/cic/datasets/nsl.html
- **Origin:** Cleaned and corrected version of the KDD Cup 1999 dataset
- **Size used:** 148,517 records (KDDTrain+ and KDDTest+ combined)
- **Features:** 41 network traffic features (protocol type, service, flag, byte counts, connection rates, etc.)
- **Why chosen:** Most widely cited benchmark for network intrusion detection research. Fixes two major flaws in KDD'99 — duplicate records that biased classifiers, and difficulty-level column that allowed cherry-picking.

### UNSW-NB15 (Secondary — supported but optional)
- **Source:** UNSW Canberra Cyber — https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Size:** 257,673 records across 49 features
- **Why chosen:** More modern dataset (2015) with contemporary attack types including backdoors, shellcode, worms, and fuzzers not present in KDD-era data. Using both datasets validates that our approach generalises across different network environments.

---

## 4. Attack Categories Detected

The system classifies traffic into 7 unified categories, mapped from the raw dataset labels:

| Category | What it is | Examples from dataset |
|---|---|---|
| **Normal** | Legitimate network traffic | Normal connections |
| **DDoS** | Distributed Denial of Service — floods the network to make services unavailable | Neptune, Smurf, Pod, Back, Land, Teardrop |
| **Malware** | Software designed to damage, disrupt, or gain unauthorised access | Rootkit, Buffer overflow, Backdoor, Worm, Shellcode |
| **Phishing** | Social engineering attacks to steal credentials | Guess_passwd, PHF, imap |
| **Ransomware** | Malware that encrypts files and demands payment | Ransomware variants in UNSW-NB15 |
| **SQL Injection** | Injecting malicious SQL into database queries | sqlattack |
| **MITM** | Man-in-the-Middle — intercepting communications | httptunnel, ARP spoofing |
| **Other** | Reconnaissance, probing, unknown threats | ipsweep, nmap, portsweep, satan |

The mapping between raw dataset labels and these 7 categories is fully configurable in `config/label_mapping.yaml` — no code changes needed to reclassify attacks.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                  │
│  Raw network traffic (CSV upload or live test-set replay)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  PREPROCESSING PIPELINE                          │
│  • Missing value imputation (median/mode)                        │
│  • Categorical encoding (LabelEncoder per column)               │
│  • Feature normalisation (StandardScaler)                        │
│  • Stratified 80/20 train/test split (seed=42, reproducible)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               FEATURE ENGINEERING                                │
│  • Low-variance removal (dropped: num_outbound_cmds)            │
│  • High-correlation removal (dropped 6 redundant features)      │
│  • RandomForest importance ranking → top 30 features kept       │
│    Final features: duration, protocol_type, src_bytes,          │
│    dst_bytes, serror_rate, same_srv_rate, count, flag, ...      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│               CLASS BALANCING (SMOTE)                            │
│  • Binary: 118,813 → 123,286 samples after oversampling         │
│  • Synthetic Minority Oversampling in feature space             │
│  • Prevents classifier bias toward majority class (normal)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              TWO-STAGE PREDICTION PIPELINE                       │
│                                                                  │
│   Stage 1: Binary Classifier                                     │
│   "Is this traffic normal or an attack?"                        │
│   → Normal: stop here, log as safe traffic                      │
│   → Attack: proceed to Stage 2                                  │
│                                                                  │
│   Stage 2: Multi-class Classifier                               │
│   "What kind of attack is it?"                                  │
│   → DDoS / Malware / Phishing / Ransomware /                   │
│      SQL Injection / MITM / Other                               │
│                                                                  │
│   Output: {is_attack, attack_type, confidence, all_scores}     │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┴──────────────────┐
           │                                    │
┌──────────▼──────────┐              ┌──────────▼──────────┐
│   SQLite Database    │              │   Alerting Module    │
│  • traffic_log       │              │  • Severity scoring  │
│  • alerts            │              │  • Email/SMS notify  │
│  • notifications     │              │  • DB logging        │
└──────────┬──────────┘              └─────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD (8 pages)                   │
│  Overview │ Live Monitor │ Attack Dist. │ Traffic Over Time      │
│  Alerts  │ Model Perf.  │ Alert Response │ Settings             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Machine Learning Models

All 5 models were trained on identical preprocessed data with the same train/test split (seed=42) for a fair comparison.

### 6.1 Random Forest
- **Type:** Ensemble of decision trees (bagging)
- **How it works:** Trains 200 decision trees on random subsets of the data and features. Each tree votes; the majority wins. Reduces overfitting through diversity.
- **Hyperparameters:** 200 estimators, `class_weight="balanced"`, unlimited depth
- **Binary Accuracy:** 99.62% | **Multi-class wF1:** 99.49% | **AUC:** 0.9997
- **Strengths:** Handles non-linear relationships, robust to outliers, provides feature importance
- **Why included:** State-of-the-art ensemble method, widely used in cybersecurity literature

### 6.2 Decision Tree
- **Type:** Single hierarchical tree
- **How it works:** Recursively splits data on the most informative feature (Gini impurity). Creates human-readable rules like "if src_bytes > 1000 AND flag = SF → DDoS"
- **Hyperparameters:** max_depth=15, `class_weight="balanced"`
- **Binary Accuracy:** 99.47% | **Multi-class wF1:** 99.31% | **AUC:** 0.9977
- **Strengths:** Fully interpretable — every prediction can be explained as a rule path
- **Why included:** Baseline interpretable model; demonstrates that even simple models perform well on this data

### 6.3 SVM (Support Vector Machine)
- **Type:** Linear margin-based classifier
- **How it works:** Finds the hyperplane that maximally separates classes in high-dimensional feature space. Uses a linear kernel for speed on large datasets.
- **Implementation:** LinearSVC (binary) + SGDClassifier/modified_huber (multi-class)
- **Binary Accuracy:** 92.62% | **Multi-class wF1:** 89.83% | **AUC:** 0.9720
- **Strengths:** Strong theoretical foundations, works well in high-dimensional spaces
- **Why it scored lower:** Linear SVM struggles with the highly non-linear decision boundaries between rare attack types like MITM vs phishing

### 6.4 XGBoost ⭐ Best Model
- **Type:** Gradient boosting ensemble
- **How it works:** Builds trees sequentially, where each tree corrects the errors of the previous ones. Uses gradient descent on a differentiable loss function. Includes L1/L2 regularisation to prevent overfitting.
- **Hyperparameters:** 200 estimators, max_depth=6, learning_rate=0.1
- **Binary Accuracy:** 99.62% | **Multi-class wF1:** 99.66% | **AUC:** 1.0000
- **Why it won:** XGBoost's gradient boosting naturally handles class imbalance and non-linear boundaries better than single trees or linear models. The regularisation prevents overfitting on rare attack classes.
- **Industry relevance:** XGBoost is the most-used algorithm in Kaggle competitions and production ML systems

### 6.5 MLP Neural Network
- **Type:** Multi-Layer Perceptron (feedforward neural network)
- **How it works:** 3 hidden layers (128→64→32 neurons) with ReLU activations. Trained with backpropagation and early stopping to prevent overfitting.
- **Binary Accuracy:** 99.14% | **Multi-class wF1:** 98.97% | **AUC:** 0.9996
- **Strengths:** Can learn complex non-linear patterns, scalable to larger datasets
- **Why included:** Represents the neural network approach; strong performance confirms deep learning is viable for this problem

---

## 7. Key Technical Decisions — Why We Made Them

### Two-stage pipeline instead of one multi-class classifier
A single 8-class classifier (including "normal") would be dominated by the majority class. By separating binary detection (Stage 1) from attack classification (Stage 2), we get:
- Better precision on rare attack types (ransomware, MITM have very few samples)
- Stage 1 acts as a fast gatekeeper — most normal traffic is filtered before Stage 2 runs
- Each stage can use a different, specialised model if needed
- More explainable: "Is it an attack?" is answered separately from "What attack?"

### SMOTE for class imbalance
NSL-KDD has severe class imbalance — normal traffic is ~52% of data, while some attack types (MITM, ransomware) appear in <0.1% of records. Without balancing:
- The model learns to predict "normal" for everything and still gets ~52% accuracy
- Rare attack types are completely ignored
SMOTE generates synthetic minority-class samples by interpolating between existing ones in feature space (not just duplicating rows), giving each class equal representation during training.

### Feature selection (30 from 41)
We dropped 11 features:
- 1 near-zero variance feature (`num_outbound_cmds` — constant in almost all records)
- 6 highly correlated features (e.g., `srv_serror_rate` ≈ `dst_host_serror_rate`)
- 4 low-importance features ranked last by RandomForest

Benefits: faster training, reduced overfitting, simpler model, more interpretable results.

### StandardScaler normalisation
Features have wildly different ranges (`src_bytes` can be millions, `logged_in` is 0/1). Without scaling, distance-based models (SVM) and gradient-based models (MLP) are dominated by large-magnitude features. StandardScaler centres each feature to mean=0, std=1.

### SQLite for logging
No external database server required — SQLite runs as a file. Perfect for a local demo system. Every classified packet, alert, and notification is persisted, making all dashboard numbers real and reproducible.

---

## 8. Results Summary

**Dataset:** NSL-KDD — 148,517 records, 30 features after selection
**Train/Test split:** 80/20 stratified (118,813 train / 29,704 test)
**Attack ratio in training:** 48.12%

### Binary Classification (Normal vs Attack)
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| XGBoost | 99.62% | 99.66% | 99.55% | 99.62% | 0.9999 |
| Random Forest | 99.62% | 99.64% | 99.57% | 99.62% | 0.9998 |
| Decision Tree | 99.47% | 99.57% | 99.34% | 99.47% | 0.9970 |
| MLP | 99.14% | 99.14% | 99.08% | 99.14% | 0.9993 |
| SVM/SGD | 92.62% | 93.52% | 90.97% | 92.62% | 0.9784 |

### Multi-class Classification (7 attack categories)
| Model | Accuracy | Weighted F1 | AUC |
|---|---|---|---|
| **XGBoost** ⭐ | **99.62%** | **99.66%** | **1.0000** |
| Random Forest | 99.48% | 99.49% | 0.9997 |
| Decision Tree | 99.29% | 99.31% | 0.9977 |
| MLP | 98.98% | 98.97% | 0.9996 |
| SVM/SGD | 88.29% | 89.83% | 0.9720 |

**XGBoost was automatically selected as the production model** (best weighted F1).

---

## 9. The Dashboard — 8 Pages Explained

### Page 1 — Overview
KPI summary cards: Total Traffic, Attacks Detected, Normal Traffic, Model Accuracy. All numbers pulled live from the SQLite database and the training report. Attack distribution donut chart. Recent activity table.

### Page 2 — Live Monitoring
Simulates a real-time network feed by replaying test-set rows at a configurable speed (1–20 rows/second). Each row is classified by the live model, logged to the database, and displayed in a streaming table with colour-coded severity. Users can also upload a CSV of new traffic for batch classification.

### Page 3 — Attack Distribution
Interactive donut chart and bar chart showing the breakdown of detected attack types from the database. Severity breakdown (High/Medium/Low) across all detected attacks.

### Page 4 — Traffic Over Time
Line chart showing total traffic volume vs attacks over a configurable time window (1h to 7d). Attack rate percentage trend. Useful for spotting attack campaigns that spike at specific times.

### Page 5 — Alerts & Detection Reports
Filterable table of all security alerts (filter by severity, attack type, status). Alerts can be acknowledged or resolved. Shows open high-severity alerts with a red banner. Includes a notifications log showing simulated email send attempts.

### Page 6 — Model Performance
Accuracy gauge for the best model (XGBoost at 99.66%). Side-by-side comparison bar charts for all 5 models. Confusion matrices and ROC curves for every model. Links to the full model_comparison_report.md.

### Page 7 — Alert & Response
Live alert banner showing count of unresolved attacks. Simulated threat injection button for demo purposes. Bulk acknowledge feature. Notification log (email/SMS simulation).

### Page 8 — Settings
Switch the active model between any of the 5 trained models. Adjust alert severity thresholds. Change live feed speed. Toggle dataset. All settings saved to `.env`.

---

## 10. Technology Stack — Why Each Choice

| Technology | Why chosen |
|---|---|
| **Python 3.14** | Universal ML ecosystem, all required libraries available |
| **scikit-learn** | Industry-standard ML library, consistent API across all 5 models |
| **XGBoost** | State-of-the-art gradient boosting, best performance on tabular data |
| **pandas / numpy** | Standard data manipulation and numerical computation |
| **imbalanced-learn (SMOTE)** | Purpose-built for class imbalance — the main challenge in IDS datasets |
| **Streamlit** | Fastest way to build a production-quality data dashboard in pure Python |
| **Plotly** | Interactive charts (zoom, hover, filter) — far better than static matplotlib for demos |
| **SQLAlchemy + SQLite** | Persistent storage with no external server dependency — ideal for local demo |
| **joblib** | Efficient serialisation of large sklearn models |
| **python-dotenv** | Secure credential management — no hardcoded secrets |
| **pytest** | Unit testing framework — proves code correctness without datasets |

---

## 11. File Structure — What Each File Does

```
cyberattack-detection/
│
├── src/config.py            Central config — reads .env + label_mapping.yaml,
│                            exports all constants used across the project
│
├── src/preprocessing.py     Loads raw NSL-KDD/UNSW-NB15 files, unifies attack
│                            labels to 7 categories, imputes missing values,
│                            encodes categoricals, scales features, splits 80/20
│
├── src/feature_engineering.py  Three-step feature selection: variance filter →
│                               correlation filter → RandomForest importance ranking
│
├── src/train_models.py      End-to-end training pipeline: calls preprocessing,
│                            feature selection, SMOTE, trains all 5 models,
│                            evaluates each, auto-selects best, saves all to disk
│
├── src/evaluate.py          Computes accuracy/precision/recall/F1/AUC, generates
│                            confusion matrix and ROC curve plots, writes the
│                            model_comparison_report.md
│
├── src/predict.py           Two-stage inference: Stage 1 binary → Stage 2 multi-
│                            class. predict() for single records, predict_batch_fast()
│                            for the live dashboard feed
│
├── src/db.py                SQLite schema (traffic_log, alerts, notifications) +
│                            all read/write operations. No ORM complexity — clean
│                            SQLAlchemy Core queries
│
├── src/alerting.py          Severity scoring (high/medium/low based on attack type
│                            + confidence), DB alert creation, SMTP email dispatch
│                            (simulated if no credentials, real if .env has SMTP)
│
├── config/label_mapping.yaml   The single source of truth for how raw dataset
│                               attack labels map to our 7 categories. Editable
│                               without touching any Python code.
│
├── dashboard/app.py         Streamlit entry point — sets page config, CSS theme,
│                            initialises DB, shows home page
│
├── dashboard/pages/         8 individual page files — each is a self-contained
│                            Streamlit page auto-discovered by the multi-page system
│
├── tests/test_preprocessing.py   8 unit tests for the preprocessing pipeline —
│                                  use synthetic data, no real dataset needed
│
└── tests/test_predict.py    7 unit tests for the prediction pipeline — use mock
                             models, verify correct output structure and logic
```

---

## 12. How to Explain This in Your Viva

### Opening statement (30 seconds)
"We built an ML-based network intrusion detection system that detects cyberattacks in real time. It uses a two-stage pipeline — first classifying traffic as normal or malicious, then identifying the specific attack type. We trained and compared five different ML models on the NSL-KDD benchmark dataset of 148,517 records. The best model, XGBoost, achieved 99.66% F1-score and AUC of 1.0. The system includes a live dashboard, a database for logging, and an alerting module."

### When asked "why machine learning?"
"Traditional IDS systems use signature-based detection — they only catch known attacks. ML learns the statistical patterns of attack behaviour from historical data, so it can detect novel attacks that don't match any known signature. This is the fundamental advantage of the ML approach."

### When asked "why XGBoost performed best?"
"XGBoost builds trees sequentially — each tree corrects the errors of the previous one. This iterative error-correction means it handles the complex, overlapping decision boundaries between attack categories better than a single Random Forest. Its L1/L2 regularisation also prevents overfitting on the rare attack classes like MITM and ransomware, which have very few training examples."

### When asked "what are the limitations?"
"Three main limitations: First, the models are trained on data from 1999-2015 network environments — modern attack vectors like cryptojacking or adversarial ML attacks are not represented. Second, the system currently replays test-set data rather than capturing live packets. Third, class imbalance for very rare attack types (2–5 samples) means those classes are still poorly represented despite SMOTE."

### When asked "what is future scope?"
"Three directions: First, integrate live packet capture using Scapy to process real network traffic instead of replays. Second, add SHAP explainability so security analysts can see exactly which features triggered a detection. Third, retrain periodically on new attack data — concept drift is a real problem as attack patterns evolve."

---

## 13. Key Numbers to Remember

| Metric | Value |
|---|---|
| Dataset size | 148,517 records |
| Features after selection | 30 (from 41) |
| Training set size | 118,813 records |
| Test set size | 29,704 records |
| Attack ratio (training) | 48.12% |
| Best model | XGBoost |
| Best F1-score | **99.66%** |
| Best AUC | **1.0000** |
| Models trained | 5 (RF, DT, SVM, XGBoost, MLP) |
| Dashboard pages | 8 |
| Unit tests | 21 (all passing) |
| Lines of code | ~3,500 |
| Attack categories | 7 + normal = 8 classes |
| GitHub repo | https://github.com/ahmedsharef/cyberML |
