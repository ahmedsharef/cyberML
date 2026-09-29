"""
feature_engineering.py
───────────────────────
Feature selection and dimensionality reduction on top of the preprocessed data.

Steps
─────
1. Remove near-zero variance features (threshold configurable)
2. Remove highly-correlated features (threshold configurable)
3. Optionally run feature importances from a fast RandomForest and keep top-k

Public API
──────────
    select_features(X_train, X_test, feature_names, ...) → (X_train_sel, X_test_sel, selected_names)
    save_feature_selection(selected_names)
    load_feature_selection() → list[str]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

from src.config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

_SELECTION_PATH = DATA_PROCESSED_DIR / "selected_features.pkl"


def _remove_low_variance(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop features with variance below `threshold` (computed on train set)."""
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel  = selector.transform(X_test)
    mask = selector.get_support()
    selected = [f for f, m in zip(feature_names, mask) if m]
    dropped  = [f for f, m in zip(feature_names, mask) if not m]
    logger.info(
        "Low-variance removal: kept %d / %d features (dropped: %s)",
        len(selected), len(feature_names), dropped or "none",
    )
    return X_train_sel, X_test_sel, selected


def _remove_high_correlation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Drop one of each pair of features with |Pearson correlation| > threshold.
    Computed only on the training set to avoid data leakage.
    """
    df = pd.DataFrame(X_train, columns=feature_names)
    corr_matrix = df.corr(method="pearson").abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info(
        "High-correlation removal (threshold=%.2f): dropping %d features: %s",
        threshold, len(to_drop), to_drop or "none",
    )

    keep_mask = [f not in to_drop for f in feature_names]
    selected  = [f for f in feature_names if f not in to_drop]
    indices   = [i for i, m in enumerate(keep_mask) if m]

    return X_train[:, indices], X_test[:, indices], selected


def _importance_selection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    y_train: np.ndarray,
    top_k: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fit a fast, shallow RandomForest on training data to rank features,
    then keep the top-k most important ones.
    """
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    # Sort by descending importance
    sorted_idx = np.argsort(importances)[::-1]

    k = min(top_k, len(feature_names))
    top_indices = sorted(sorted_idx[:k])   # sort back to original order

    selected = [feature_names[i] for i in top_indices]
    logger.info(
        "Importance-based selection: keeping top %d features: %s",
        k, selected,
    )

    return X_train[:, top_indices], X_test[:, top_indices], selected


def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    y_train: Optional[np.ndarray] = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    use_importance: bool = True,
    top_k: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Full feature selection pipeline.

    Parameters
    ----------
    X_train / X_test       : Scaled feature matrices
    feature_names          : Column names matching X_train columns
    y_train                : Training labels (needed for importance selection)
    variance_threshold     : Drop features with variance < this value
    correlation_threshold  : Drop one of each pair with |r| > this value
    use_importance         : Whether to run RandomForest importance filtering
    top_k                  : How many features to keep after importance ranking

    Returns
    -------
    (X_train_selected, X_test_selected, selected_feature_names)
    """
    logger.info(
        "Feature selection starting with %d features", len(feature_names)
    )

    # Step 1: Variance filter
    X_train, X_test, feature_names = _remove_low_variance(
        X_train, X_test, feature_names, threshold=variance_threshold
    )

    # Step 2: Correlation filter
    X_train, X_test, feature_names = _remove_high_correlation(
        X_train, X_test, feature_names, threshold=correlation_threshold
    )

    # Step 3: Importance-based top-k selection
    if use_importance and y_train is not None and len(feature_names) > top_k:
        X_train, X_test, feature_names = _importance_selection(
            X_train, X_test, feature_names, y_train, top_k=top_k
        )

    logger.info(
        "Feature selection complete: %d features selected", len(feature_names)
    )
    return X_train, X_test, feature_names


def save_feature_selection(selected_names: list[str]) -> None:
    """Persist the list of selected feature names."""
    with open(_SELECTION_PATH, "wb") as f:
        pickle.dump(selected_names, f)
    logger.info("Feature selection saved → %s", _SELECTION_PATH)


def load_feature_selection() -> list[str]:
    """Load the previously saved feature selection."""
    if not _SELECTION_PATH.exists():
        raise FileNotFoundError(
            "No feature selection found. Run training pipeline first."
        )
    with open(_SELECTION_PATH, "rb") as f:
        return pickle.load(f)
