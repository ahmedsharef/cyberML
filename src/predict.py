"""
predict.py
──────────
Two-stage inference pipeline.

    Stage 1 — Binary detector  : Is this traffic normal or an attack?
    Stage 2 — Multi-class clf  : If attack, which category?

Public API
──────────
    load_pipeline()
        → PipelineBundle  (loads models + preprocessing meta from disk)

    predict(record: dict, bundle: PipelineBundle)
        → {"is_attack": bool, "attack_type": str, "confidence": float,
           "binary_confidence": float, "all_scores": dict}

    predict_batch(df: pd.DataFrame, bundle: PipelineBundle)
        → pd.DataFrame  (one result row per input row)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.preprocessing import load_processed, transform_single_record

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bundle — holds everything needed at inference time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineBundle:
    binary_clf:          Any         # Stage-1 model
    multi_clf:           Any         # Stage-2 model
    preprocessing_meta:  dict        # from load_processed()
    best_model_name:     str
    class_names:         list[str]   # multiclass label names


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_best_model_name() -> str:
    """Read the best model name from models/best_model.txt."""
    path = MODELS_DIR / "best_model.txt"
    if not path.exists():
        raise FileNotFoundError(
            "models/best_model.txt not found. "
            "Run `python -m src.train_models` first."
        )
    return path.read_text().strip()


def load_pipeline(model_name: Optional[str] = None) -> PipelineBundle:
    """
    Load the two-stage pipeline from disk.

    Parameters
    ----------
    model_name : Which model to load (e.g. "random_forest"). Defaults to the
                 best model recorded in models/best_model.txt.
    """
    if model_name is None:
        model_name = get_best_model_name()

    binary_path = MODELS_DIR / f"{model_name}_binary.joblib"
    multi_path  = MODELS_DIR / f"{model_name}_multi.joblib"

    if not binary_path.exists() or not multi_path.exists():
        raise FileNotFoundError(
            f"Model files for '{model_name}' not found in {MODELS_DIR}. "
            "Run `python -m src.train_models` first."
        )

    binary_clf = joblib.load(binary_path)
    multi_clf  = joblib.load(multi_path)
    meta       = load_processed()

    logger.info("Pipeline loaded: %s", model_name)

    return PipelineBundle(
        binary_clf=binary_clf,
        multi_clf=multi_clf,
        preprocessing_meta=meta,
        best_model_name=model_name,
        class_names=meta["class_names"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_confidence(clf, X: np.ndarray, label_idx: int) -> float:
    """Return the probability for the predicted class if available."""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        return float(probs[label_idx])
    return 1.0  # SVM without proba — return 1.0 as a fallback


def _get_all_scores(clf, X: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Return a dict of {class_name: probability} if the model supports it."""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        # predict_proba classes_ may not match full class_names if some classes
        # were absent in training — map carefully
        clf_classes = list(clf.classes_) if hasattr(clf, "classes_") else list(range(len(probs)))
        scores = {}
        for idx, cls_idx in enumerate(clf_classes):
            if isinstance(cls_idx, (int, np.integer)) and int(cls_idx) < len(class_names):
                scores[class_names[int(cls_idx)]] = float(probs[idx])
            else:
                scores[str(cls_idx)] = float(probs[idx])
        return scores
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Single-record prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(record: dict, bundle: PipelineBundle) -> dict:
    """
    Two-stage prediction for a single traffic record.

    Parameters
    ----------
    record : dict of {feature_name: value}
    bundle : loaded PipelineBundle

    Returns
    -------
    {
        "is_attack"          : bool,
        "attack_type"        : str,   # "normal" or one of the 7 categories
        "confidence"         : float, # confidence of the final predicted class
        "binary_confidence"  : float, # Stage-1 attack probability
        "all_scores"         : dict,  # {class_name: prob} for multi-class stage
    }
    """
    meta = bundle.preprocessing_meta

    # Apply preprocessing transformations
    X = transform_single_record(record, meta)

    # Apply feature selection (keep only selected feature indices)
    selected_names  = meta.get("selected_features")
    feature_names   = meta["feature_names"]

    if selected_names is not None:
        indices = [feature_names.index(f) for f in selected_names if f in feature_names]
        X = X[:, indices]

    # ── Stage 1: Binary ───────────────────────────────────────────────────────
    bin_pred = int(bundle.binary_clf.predict(X)[0])
    binary_confidence = _get_confidence(bundle.binary_clf, X, bin_pred)

    if bin_pred == 0:
        # Normal traffic
        return {
            "is_attack":         False,
            "attack_type":       "normal",
            "confidence":        float(binary_confidence),
            "binary_confidence": float(binary_confidence),
            "all_scores":        {"normal": float(binary_confidence)},
        }

    # ── Stage 2: Multi-class ──────────────────────────────────────────────────
    multi_pred = int(bundle.multi_clf.predict(X)[0])
    all_scores = _get_all_scores(bundle.multi_clf, X, bundle.class_names)

    # Map encoded label index back to class name
    attack_type = bundle.class_names[multi_pred] if multi_pred < len(bundle.class_names) else "other"
    multi_confidence = _get_confidence(bundle.multi_clf, X, multi_pred)

    return {
        "is_attack":         True,
        "attack_type":       attack_type,
        "confidence":        float(multi_confidence),
        "binary_confidence": float(binary_confidence),
        "all_scores":        all_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, bundle: PipelineBundle) -> pd.DataFrame:
    """
    Run predict() on every row of a DataFrame.

    Parameters
    ----------
    df     : DataFrame where each row is a traffic record
    bundle : loaded PipelineBundle

    Returns
    -------
    Original DataFrame with three extra columns:
        is_attack, attack_type, confidence
    """
    results = []
    for _, row in df.iterrows():
        res = predict(row.to_dict(), bundle)
        results.append({
            "is_attack":   res["is_attack"],
            "attack_type": res["attack_type"],
            "confidence":  res["confidence"],
        })

    out_df = df.copy()
    out_df["is_attack"]   = [r["is_attack"]   for r in results]
    out_df["attack_type"] = [r["attack_type"] for r in results]
    out_df["confidence"]  = [r["confidence"]  for r in results]
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised batch prediction (faster — bypasses per-row dict conversion)
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch_fast(
    X: np.ndarray,
    bundle: PipelineBundle,
) -> pd.DataFrame:
    """
    Batch prediction directly on a pre-scaled numpy array (e.g. from the test set).
    Used by the live monitoring dashboard for speed.

    Returns DataFrame with columns: is_attack, attack_type, confidence.
    """
    # Stage 1
    bin_preds = bundle.binary_clf.predict(X)

    if hasattr(bundle.binary_clf, "predict_proba"):
        bin_probs = bundle.binary_clf.predict_proba(X)
        bin_conf  = bin_probs[np.arange(len(bin_preds)), bin_preds]
    else:
        bin_conf = np.ones(len(bin_preds))

    # Stage 2 — only for rows flagged as attacks
    attack_mask  = bin_preds == 1
    attack_types = np.where(attack_mask, "other", "normal").tolist()
    confidences  = bin_conf.tolist()

    if attack_mask.any():
        X_attacks = X[attack_mask]
        multi_preds = bundle.multi_clf.predict(X_attacks)

        if hasattr(bundle.multi_clf, "predict_proba"):
            multi_probs = bundle.multi_clf.predict_proba(X_attacks)
            multi_conf  = multi_probs[np.arange(len(multi_preds)), multi_preds]
        else:
            multi_conf = np.ones(len(multi_preds))

        attack_indices = np.where(attack_mask)[0]
        for i, (pred, conf) in enumerate(zip(multi_preds, multi_conf)):
            orig_idx = attack_indices[i]
            cls_name = bundle.class_names[pred] if pred < len(bundle.class_names) else "other"
            attack_types[orig_idx] = cls_name
            confidences[orig_idx]  = float(conf)

    return pd.DataFrame({
        "is_attack":   attack_mask.tolist(),
        "attack_type": attack_types,
        "confidence":  confidences,
    })
