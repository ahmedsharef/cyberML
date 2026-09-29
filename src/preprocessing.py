"""
preprocessing.py
────────────────
Handles raw data loading, label unification, missing-value imputation,
categorical encoding, and train/test splitting for both NSL-KDD and UNSW-NB15.

Public API
──────────
    load_raw_data()            → pd.DataFrame  (raw, unified columns)
    preprocess(df)             → (X, y_binary, y_multi, encoders)
    save_processed(...)        → None
    load_processed()           → (X_train, X_test, y_bin_train, y_bin_test,
                                   y_multi_train, y_multi_test, meta)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import (
    DATASET,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    LABEL_MAPPING,
    TARGET_CATEGORIES,
    NSL_KDD_COLUMNS,
    NSL_KDD_CATEGORICAL,
    NSL_KDD_DROP,
    UNSW_NB15_CATEGORICAL,
    UNSW_NB15_DROP,
    UNSW_NB15_LABEL_COL,
    UNSW_NB15_BINARY_COL,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_nsl_kdd() -> pd.DataFrame:
    """Load NSL-KDD train + (optionally) test files from data/raw/."""
    train_path = DATA_RAW_DIR / "KDDTrain+.txt"
    test_path  = DATA_RAW_DIR / "KDDTest+.txt"

    dfs = []
    for p in (train_path, test_path):
        if p.exists():
            logger.info("Loading NSL-KDD from %s", p)
            df = pd.read_csv(p, header=None, names=NSL_KDD_COLUMNS)
            dfs.append(df)
        else:
            logger.warning("NSL-KDD file not found: %s (skipping)", p)

    if not dfs:
        raise FileNotFoundError(
            "No NSL-KDD files found in data/raw/. "
            "See data/download_instructions.md for instructions."
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Drop the 'difficulty' column — not a real feature
    combined.drop(columns=[c for c in NSL_KDD_DROP if c in combined.columns], inplace=True)

    # Rename label column to unified name
    combined.rename(columns={"label": "raw_label"}, inplace=True)

    # Map to unified target categories
    combined["attack_category"] = (
        combined["raw_label"]
        .str.strip()
        .str.lower()
        .map(LABEL_MAPPING)
        .fillna("other")
    )

    # Binary label: 0 = normal, 1 = attack
    combined["is_attack"] = (combined["attack_category"] != "normal").astype(int)

    logger.info("NSL-KDD loaded: %d rows, %d columns", *combined.shape)
    return combined


def _load_unsw_nb15() -> pd.DataFrame:
    """Load UNSW-NB15 train + (optionally) test CSV files from data/raw/."""
    train_path = DATA_RAW_DIR / "UNSW_NB15_training-set.csv"
    test_path  = DATA_RAW_DIR / "UNSW_NB15_testing-set.csv"

    dfs = []
    for p in (train_path, test_path):
        if p.exists():
            logger.info("Loading UNSW-NB15 from %s", p)
            df = pd.read_csv(p)
            dfs.append(df)
        else:
            logger.warning("UNSW-NB15 file not found: %s (skipping)", p)

    if not dfs:
        raise FileNotFoundError(
            "No UNSW-NB15 files found in data/raw/. "
            "See data/download_instructions.md for instructions."
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Normalise column names
    combined.columns = [c.strip().lower() for c in combined.columns]

    # Drop identifier columns
    combined.drop(
        columns=[c for c in UNSW_NB15_DROP if c in combined.columns],
        inplace=True,
    )

    # Rename attack_cat → raw_label
    if UNSW_NB15_LABEL_COL.lower() in combined.columns:
        combined.rename(columns={UNSW_NB15_LABEL_COL.lower(): "raw_label"}, inplace=True)
    elif "attack_cat" in combined.columns:
        combined.rename(columns={"attack_cat": "raw_label"}, inplace=True)

    # Fill NaN in raw_label with "Normal"
    combined["raw_label"] = combined["raw_label"].fillna("Normal").str.strip()

    # Map to unified target categories
    combined["attack_category"] = (
        combined["raw_label"]
        .map(LABEL_MAPPING)
        .fillna("other")
    )

    # Binary label (use existing column or derive)
    if UNSW_NB15_BINARY_COL in combined.columns:
        combined["is_attack"] = combined[UNSW_NB15_BINARY_COL].astype(int)
    else:
        combined["is_attack"] = (combined["attack_category"] != "normal").astype(int)

    logger.info("UNSW-NB15 loaded: %d rows, %d columns", *combined.shape)
    return combined


def load_raw_data() -> pd.DataFrame:
    """
    Load and unify the configured dataset.

    Returns
    -------
    pd.DataFrame with columns:
        - All feature columns from the source dataset
        - raw_label         : original attack label string
        - attack_category   : unified category (from label_mapping.yaml)
        - is_attack         : 0 / 1 binary flag
    """
    if DATASET == "nsl_kdd":
        return _load_nsl_kdd()
    elif DATASET == "unsw_nb15":
        return _load_unsw_nb15()
    else:
        raise ValueError(f"Unknown DATASET='{DATASET}'")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the categorical feature columns for the active dataset."""
    if DATASET == "nsl_kdd":
        return [c for c in NSL_KDD_CATEGORICAL if c in df.columns]
    else:
        return [c for c in UNSW_NB15_CATEGORICAL if c in df.columns]


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Full preprocessing pipeline.

    Steps
    ─────
    1. Drop metadata columns (raw_label)
    2. Impute missing values
    3. Encode categorical features (LabelEncoder per column)
    4. Scale numeric features (StandardScaler)
    5. Stratified train/test split (by attack_category)

    Parameters
    ----------
    df          : Raw DataFrame from load_raw_data()
    test_size   : Fraction for test split
    random_state: Reproducibility seed

    Returns
    -------
    dict with keys:
        X_train, X_test           : np.ndarray — feature matrices
        y_bin_train, y_bin_test   : np.ndarray — binary labels (0/1)
        y_multi_train, y_multi_test: np.ndarray — encoded multi-class labels
        feature_names             : list[str]
        label_encoders            : dict[col -> LabelEncoder]  (categorical cols)
        multiclass_encoder        : LabelEncoder for attack_category
        scaler                    : StandardScaler
        class_names               : list[str]  (ordered target categories)
    """
    logger.info("Starting preprocessing (%d rows)", len(df))

    # ── Separate targets ──────────────────────────────────────────────────────
    y_binary = df["is_attack"].values.astype(int)
    y_labels = df["attack_category"].values  # string labels

    # ── Build feature matrix ──────────────────────────────────────────────────
    drop_cols = {"raw_label", "attack_category", "is_attack"}
    # Also drop UNSW-NB15 binary label col if present to avoid leakage
    if DATASET == "unsw_nb15" and UNSW_NB15_BINARY_COL in df.columns:
        drop_cols.add(UNSW_NB15_BINARY_COL)

    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # ── Impute missing values ─────────────────────────────────────────────────
    cat_cols = _get_categorical_columns(X_df)
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    # Numeric: fill with median
    num_imputer = SimpleImputer(strategy="median")
    X_df[num_cols] = num_imputer.fit_transform(X_df[num_cols])

    # Categorical: fill with most frequent
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_df[cat_cols] = cat_imputer.fit_transform(X_df[cat_cols])

    # ── Encode categorical columns ────────────────────────────────────────────
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        label_encoders[col] = le
        logger.debug("Encoded '%s': %d unique values", col, len(le.classes_))

    # Ensure everything is numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ── Encode multi-class labels ─────────────────────────────────────────────
    multiclass_encoder = LabelEncoder()
    # Fit on all target categories to ensure consistent encoding even if some
    # are absent in the current split
    multiclass_encoder.fit(TARGET_CATEGORIES)
    # Map any unseen labels to 'other'
    y_labels_mapped = np.array([
        lbl if lbl in multiclass_encoder.classes_ else "other"
        for lbl in y_labels
    ])
    y_multi = multiclass_encoder.transform(y_labels_mapped)

    # ── Train/test split ──────────────────────────────────────────────────────
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    X_train, X_test, \
    y_bin_train, y_bin_test, \
    y_multi_train, y_multi_test = train_test_split(
        X, y_binary, y_multi,
        test_size=test_size,
        random_state=random_state,
        stratify=y_multi,  # stratify by multiclass for balanced splits
    )

    logger.info(
        "Split → train: %d, test: %d | attack ratio train: %.2f%%",
        len(X_train), len(X_test),
        100 * y_bin_train.mean(),
    )

    # ── Scale numeric features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return {
        "X_train":           X_train,
        "X_test":            X_test,
        "y_bin_train":       y_bin_train,
        "y_bin_test":        y_bin_test,
        "y_multi_train":     y_multi_train,
        "y_multi_test":      y_multi_test,
        "feature_names":     feature_names,
        "label_encoders":    label_encoders,
        "multiclass_encoder": multiclass_encoder,
        "scaler":            scaler,
        "class_names":       list(multiclass_encoder.classes_),
        "num_imputer":       num_imputer,
        "cat_imputer":       cat_imputer if cat_cols else None,
        "cat_cols":          cat_cols,
        "num_cols":          num_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(data: dict[str, Any]) -> None:
    """Persist processed arrays and encoders to data/processed/."""
    out = DATA_PROCESSED_DIR

    # Save numpy arrays
    for key in ("X_train", "X_test", "y_bin_train", "y_bin_test",
                "y_multi_train", "y_multi_test"):
        np.save(out / f"{key}.npy", data[key])

    # Save metadata / encoders
    meta = {k: v for k, v in data.items()
            if k not in ("X_train", "X_test",
                         "y_bin_train", "y_bin_test",
                         "y_multi_train", "y_multi_test")}
    with open(out / "preprocessing_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    logger.info("Processed data saved to %s", out)


def load_processed() -> dict[str, Any]:
    """Load the processed arrays and encoders from data/processed/."""
    out = DATA_PROCESSED_DIR
    meta_path = out / "preprocessing_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `python -m src.train_models` first "
            "(which calls preprocessing automatically)."
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    for key in ("X_train", "X_test", "y_bin_train", "y_bin_test",
                "y_multi_train", "y_multi_test"):
        meta[key] = np.load(out / f"{key}.npy")

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. Single-record transform (used by predict.py at inference time)
# ─────────────────────────────────────────────────────────────────────────────

def transform_single_record(record: dict, meta: dict[str, Any]) -> np.ndarray:
    """
    Apply the fitted preprocessing pipeline to a single traffic record dict.

    Parameters
    ----------
    record : dict mapping feature_name → raw value
    meta   : preprocessing metadata dict (from load_processed)

    Returns
    -------
    np.ndarray of shape (1, n_features) — ready for model.predict()
    """
    feature_names = meta["feature_names"]
    cat_cols      = meta["cat_cols"]
    label_encoders = meta["label_encoders"]
    scaler         = meta["scaler"]
    num_imputer    = meta["num_imputer"]

    # Build a single-row DataFrame with the expected columns
    row = {col: record.get(col, np.nan) for col in feature_names}
    df_row = pd.DataFrame([row])

    # Impute numeric
    num_cols = [c for c in feature_names if c not in cat_cols]
    df_row[num_cols] = num_imputer.transform(df_row[num_cols])

    # Encode categorical
    for col in cat_cols:
        le = label_encoders[col]
        val = str(df_row[col].iloc[0])
        if val in le.classes_:
            df_row[col] = le.transform([val])
        else:
            # Unseen category → encode as 0 (most common fallback)
            df_row[col] = 0

    # Coerce and scale
    X = df_row[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)
    return X_scaled
