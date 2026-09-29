"""
tests/test_preprocessing.py
────────────────────────────
Unit tests for the preprocessing pipeline.
Uses synthetic data so no real dataset files are required.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force NSL-KDD mode for tests (no .env needed)
os.environ.setdefault("DATASET", "nsl_kdd")

from src.preprocessing import preprocess, transform_single_record
from src.config import NSL_KDD_COLUMNS, TARGET_CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_nslkdd(n: int = 500) -> pd.DataFrame:
    """Build a minimal synthetic NSL-KDD-shaped DataFrame."""
    rng = np.random.default_rng(42)

    feature_cols = [c for c in NSL_KDD_COLUMNS if c not in ("label", "difficulty")]
    numeric_cols = [c for c in feature_cols if c not in ("protocol_type", "service", "flag")]
    cat_cols     = ["protocol_type", "service", "flag"]

    data = {}
    for col in numeric_cols:
        data[col] = rng.uniform(0, 100, n)
    data["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    data["service"]       = rng.choice(["http", "ftp", "smtp", "ssh", "other"], n)
    data["flag"]          = rng.choice(["SF", "S0", "REJ", "RSTO"], n)

    # Labels: 60% normal, 40% attacks
    labels = rng.choice(
        ["normal", "neptune", "smurf", "back", "guess_passwd", "rootkit"],
        n,
        p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
    )
    data["raw_label"] = labels

    # Map to attack_category
    from src.config import LABEL_MAPPING
    data["attack_category"] = pd.Series(labels).str.lower().map(LABEL_MAPPING).fillna("other")
    data["is_attack"] = (pd.Series(data["attack_category"]) != "normal").astype(int)

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_output_keys(self):
        """preprocess() must return all required keys."""
        df = _make_synthetic_nslkdd()
        result = preprocess(df, test_size=0.2, random_state=0)
        required = {
            "X_train","X_test","y_bin_train","y_bin_test",
            "y_multi_train","y_multi_test","feature_names",
            "label_encoders","multiclass_encoder","scaler","class_names",
        }
        assert required.issubset(result.keys())

    def test_train_test_split_ratio(self):
        """Train/test split should be approximately 80/20."""
        df = _make_synthetic_nslkdd(500)
        result = preprocess(df, test_size=0.2, random_state=0)
        n_total = len(result["X_train"]) + len(result["X_test"])
        test_ratio = len(result["X_test"]) / n_total
        assert abs(test_ratio - 0.2) < 0.05, f"Test ratio off: {test_ratio:.3f}"

    def test_feature_matrix_shape(self):
        """X_train and X_test must have the same number of columns."""
        df = _make_synthetic_nslkdd(300)
        result = preprocess(df)
        assert result["X_train"].shape[1] == result["X_test"].shape[1]

    def test_no_nan_in_output(self):
        """Processed feature matrices must contain no NaN values."""
        df = _make_synthetic_nslkdd(200)
        # Introduce some NaNs
        df.iloc[10:15, 2] = np.nan
        result = preprocess(df)
        assert not np.isnan(result["X_train"]).any(), "NaN found in X_train"
        assert not np.isnan(result["X_test"]).any(),  "NaN found in X_test"

    def test_binary_labels_are_0_1(self):
        """Binary labels must only contain 0 and 1."""
        df = _make_synthetic_nslkdd(200)
        result = preprocess(df)
        for split in ("y_bin_train", "y_bin_test"):
            uniq = set(result[split].tolist())
            assert uniq.issubset({0, 1}), f"Unexpected values in {split}: {uniq}"

    def test_multiclass_labels_within_range(self):
        """Multi-class encoded labels must be within [0, n_classes)."""
        df = _make_synthetic_nslkdd(300)
        result = preprocess(df)
        n_classes = len(result["class_names"])
        for split in ("y_multi_train", "y_multi_test"):
            arr = result[split]
            assert arr.min() >= 0, f"Negative label in {split}"
            assert arr.max() < n_classes, f"Label >= n_classes in {split}: {arr.max()} >= {n_classes}"

    def test_class_names_subset_of_targets(self):
        """class_names must be a subset of the defined TARGET_CATEGORIES."""
        df = _make_synthetic_nslkdd(200)
        result = preprocess(df)
        assert set(result["class_names"]).issubset(set(TARGET_CATEGORIES))

    def test_scaled_range(self):
        """After StandardScaler, X_train should have ~zero mean per feature."""
        df = _make_synthetic_nslkdd(500)
        result = preprocess(df)
        col_means = result["X_train"].mean(axis=0)
        assert np.allclose(col_means, 0, atol=0.5), "Features not centred after scaling"

    def test_reproducibility(self):
        """Same seed must give identical results."""
        df = _make_synthetic_nslkdd(300)
        r1 = preprocess(df, random_state=7)
        r2 = preprocess(df, random_state=7)
        np.testing.assert_array_equal(r1["X_train"], r2["X_train"])
        np.testing.assert_array_equal(r1["y_bin_train"], r2["y_bin_train"])


class TestTransformSingleRecord:

    def test_output_shape(self):
        """transform_single_record must return (1, n_features)."""
        df = _make_synthetic_nslkdd(200)
        meta = preprocess(df)
        record = {col: 0.0 for col in meta["feature_names"]}
        X = transform_single_record(record, meta)
        assert X.shape == (1, len(meta["feature_names"]))

    def test_handles_missing_fields(self):
        """Missing fields in the record dict must be filled (no error)."""
        df = _make_synthetic_nslkdd(200)
        meta = preprocess(df)
        partial_record = {"duration": 10.0}   # most fields missing
        X = transform_single_record(partial_record, meta)
        assert X.shape[0] == 1
        assert not np.isnan(X).any()

    def test_handles_unseen_categorical(self):
        """Unseen categorical values must not crash the transform."""
        df = _make_synthetic_nslkdd(200)
        meta = preprocess(df)
        record = {col: 0.0 for col in meta["feature_names"]}
        record["protocol_type"] = "UNSEEN_PROTOCOL_XYZ"
        X = transform_single_record(record, meta)  # should not raise
        assert X.shape[0] == 1
