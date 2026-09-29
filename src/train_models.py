"""
train_models.py
───────────────
Trains all 5 models on the preprocessed data, evaluates them, saves each
trained model, and writes a model_comparison_report.md.

Models trained
──────────────
    1. Random Forest
    2. Decision Tree
    3. SVM (LinearSVC — faster for large datasets)
    4. XGBoost
    5. MLP Neural Network (scikit-learn MLPClassifier)

For each model, trains BOTH:
    - A binary classifier (normal vs attack)   → saved as <name>_binary.joblib
    - A multi-class classifier                  → saved as <name>_multi.joblib

Auto-selects the best multi-class model by weighted F1-score and writes:
    models/best_model.txt   (name of best model)

Usage
─────
    python -m src.train_models
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import MODELS_DIR
from src.evaluate import evaluate_model, save_comparison_report
from src.feature_engineering import (
    save_feature_selection,
    select_features,
)
from src.preprocessing import load_raw_data, preprocess, save_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Return a dict of {name: (binary_clf, multi_clf)} untrained estimators."""
    return {
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=None,
                random_state=42, n_jobs=-1, class_weight="balanced",
            ),
            RandomForestClassifier(
                n_estimators=200, max_depth=None,
                random_state=42, n_jobs=-1, class_weight="balanced",
            ),
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                max_depth=15, random_state=42, class_weight="balanced",
            ),
            DecisionTreeClassifier(
                max_depth=15, random_state=42, class_weight="balanced",
            ),
        ),
        "svm": (
            # Binary: LinearSVC + calibration (binary has only 2 classes, cv=2 is fine)
            CalibratedClassifierCV(
                LinearSVC(max_iter=5000, random_state=42, class_weight="balanced"),
                cv=2,
            ),
            # Multi-class: SGDClassifier with modified_huber loss
            # (linear SVM equivalent, supports predict_proba natively, no CV needed)
            SGDClassifier(
                loss="modified_huber", max_iter=1000, random_state=42,
                class_weight="balanced", n_jobs=-1,
            ),
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=42, n_jobs=-1,
            ),
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=200,
                random_state=42, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=10,
            ),
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=200,
                random_state=42, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=10,
            ),
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SMOTE resampling (only on train set)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_smote(
    X: np.ndarray, y: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to the training set to handle class imbalance."""
    # Only apply if minority class has enough samples (SMOTE needs k_neighbors)
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if min_count < 6:
        logger.warning(
            "Skipping SMOTE for %s — minority class has only %d samples",
            label, min_count,
        )
        return X, y

    k = min(5, min_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "SMOTE applied (%s): %d → %d samples", label, len(X), len(X_res)
    )
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train_all() -> None:
    """End-to-end: load → preprocess → feature select → train → evaluate → report."""

    # 1. Load and preprocess ──────────────────────────────────────────────────
    logger.info("=== Step 1: Loading raw data ===")
    df = load_raw_data()

    logger.info("=== Step 2: Preprocessing ===")
    data = preprocess(df)
    save_processed(data)

    X_train      = data["X_train"]
    X_test       = data["X_test"]
    y_bin_train  = data["y_bin_train"]
    y_bin_test   = data["y_bin_test"]
    y_multi_train = data["y_multi_train"]
    y_multi_test  = data["y_multi_test"]
    feature_names = data["feature_names"]
    class_names   = data["class_names"]

    # 2. Feature selection ────────────────────────────────────────────────────
    logger.info("=== Step 3: Feature engineering/selection ===")
    X_train_sel, X_test_sel, selected_features = select_features(
        X_train, X_test, feature_names, y_train=y_multi_train,
        variance_threshold=0.01, correlation_threshold=0.95,
        use_importance=True, top_k=30,
    )
    save_feature_selection(selected_features)

    # Store selected feature names back into the data dict so predict.py
    # can apply the same column filter at inference time
    data["selected_features"] = selected_features
    data["xgb_label_remap"]   = {}   # will be filled after label analysis
    save_processed(data)   # re-save with selected_features included

    # 3. SMOTE on selected features ───────────────────────────────────────────
    logger.info("=== Step 4: Applying SMOTE for class balance ===")
    X_bin_res, y_bin_res     = _apply_smote(X_train_sel, y_bin_train,  "binary")
    X_multi_res, y_multi_res = _apply_smote(X_train_sel, y_multi_train, "multi-class")

    # 4. Train and evaluate all models ────────────────────────────────────────
    logger.info("=== Step 5: Training all models ===")
    model_defs = _build_models()
    results    = {}

    # XGBoost requires consecutive integer labels starting from 0.
    # Re-encode y_multi to compact form and keep a reverse map.
    unique_labels   = np.unique(np.concatenate([y_multi_train, y_multi_test]))
    label_remap     = {old: new for new, old in enumerate(unique_labels)}
    label_unmap     = {new: old for old, new in label_remap.items()}
    y_multi_res_xgb = np.array([label_remap[l] for l in y_multi_res])
    y_multi_test_xgb= np.array([label_remap[l] for l in y_multi_test])
    # Compact class names aligned to XGBoost labels
    class_names_xgb = [class_names[label_unmap[i]] for i in range(len(unique_labels))]

    # Persist the remap so predict.py can decode XGBoost predictions
    data["xgb_label_remap"]  = {int(k): int(v) for k, v in label_remap.items()}
    data["xgb_label_unmap"]  = {int(k): int(v) for k, v in label_unmap.items()}
    data["class_names_xgb"]  = class_names_xgb
    save_processed(data)

    best_model_name = None
    best_f1         = -1.0

    for name, (bin_clf, multi_clf) in model_defs.items():
        logger.info("── Training %s ──", name)

        bin_path   = MODELS_DIR / f"{name}_binary.joblib"
        multi_path = MODELS_DIR / f"{name}_multi.joblib"

        # ── Binary model ───────────────────────────────────────────────────
        if bin_path.exists():
            logger.info("  Binary already saved — loading from disk")
            bin_clf = joblib.load(bin_path)
            bin_train_time = 0.0
        else:
            t0 = time.time()
            bin_clf.fit(X_bin_res, y_bin_res)
            bin_train_time = time.time() - t0
            logger.info("  Binary trained in %.1fs", bin_train_time)
            joblib.dump(bin_clf, bin_path)

        bin_metrics = evaluate_model(
            bin_clf, X_test_sel, y_bin_test,
            class_names=["normal", "attack"],
            model_name=f"{name}_binary",
            is_binary=True,
        )

        # ── Multi-class model ──────────────────────────────────────────────
        # XGBoost requires consecutive labels; use remapped version for it
        is_xgb   = name == "xgboost"
        y_m_fit  = y_multi_res_xgb  if is_xgb else y_multi_res
        y_m_test = y_multi_test_xgb if is_xgb else y_multi_test
        cn_multi = class_names_xgb  if is_xgb else class_names

        if multi_path.exists():
            logger.info("  Multi-class already saved — loading from disk")
            multi_clf = joblib.load(multi_path)
            multi_train_time = 0.0
        else:
            t0 = time.time()
            multi_clf.fit(X_multi_res, y_m_fit)
            multi_train_time = time.time() - t0
            logger.info("  Multi-class trained in %.1fs", multi_train_time)
            joblib.dump(multi_clf, multi_path)

        multi_metrics = evaluate_model(
            multi_clf, X_test_sel, y_m_test,
            class_names=cn_multi,
            model_name=f"{name}_multi",
            is_binary=False,
        )

        results[name] = {
            "binary": {**bin_metrics,  "train_time": bin_train_time},
            "multi":  {**multi_metrics, "train_time": multi_train_time},
        }

        # Track best by weighted F1 on multi-class
        wf1 = multi_metrics["weighted_f1"]
        if wf1 > best_f1:
            best_f1         = wf1
            best_model_name = name

        logger.info("  Saved %s binary + multi models", name)

    # 5. Record best model ────────────────────────────────────────────────────
    best_path = MODELS_DIR / "best_model.txt"
    best_path.write_text(best_model_name)
    logger.info(
        "=== Best model: %s (weighted F1 = %.4f) ===", best_model_name, best_f1
    )

    # 6. Save comparison report ───────────────────────────────────────────────
    logger.info("=== Step 6: Generating comparison report ===")
    save_comparison_report(results, class_names)

    logger.info("=== Training complete. ===")


if __name__ == "__main__":
    train_all()
