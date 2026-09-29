"""
tests/test_predict.py
──────────────────────
Unit tests for the predict module.
Uses a lightweight mock pipeline to avoid needing real trained models.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATASET", "nsl_kdd")

from src.predict import (
    PipelineBundle,
    predict,
    predict_batch_fast,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock bundle factory
# ─────────────────────────────────────────────────────────────────────────────

def _mock_bundle(
    binary_pred: int = 1,
    binary_proba: list | None = None,
    multi_pred: int = 1,
    multi_proba: list | None = None,
    class_names: list | None = None,
) -> PipelineBundle:
    """Build a fake PipelineBundle for testing without real models."""

    cn = class_names or ["normal","ddos","malware","phishing","ransomware","sql_injection","mitm","other"]
    n_classes = len(cn)

    # Binary classifier mock
    bin_clf = MagicMock()
    bin_clf.predict.return_value = np.array([binary_pred])
    if binary_proba is None:
        bp = [0.1, 0.9] if binary_pred == 1 else [0.9, 0.1]
    else:
        bp = binary_proba
    bin_clf.predict_proba.return_value = np.array([bp])
    bin_clf.classes_ = [0, 1]

    # Multi-class classifier mock
    multi_clf = MagicMock()
    multi_clf.predict.return_value = np.array([multi_pred])
    if multi_proba is None:
        mp = [0.05] * n_classes
        mp[multi_pred] = 0.6 + 0.05 * multi_pred
    else:
        mp = multi_proba
    multi_clf.predict_proba.return_value = np.array([mp])
    multi_clf.classes_ = list(range(n_classes))

    # Preprocessing meta mock (minimal)
    meta = {
        "feature_names":      ["f1","f2","f3"],
        "cat_cols":           [],
        "label_encoders":     {},
        "scaler":             MagicMock(**{"transform.side_effect": lambda x: x}),
        "num_imputer":        MagicMock(**{"transform.side_effect": lambda x: x}),
        "class_names":        cn,
        "selected_features":  None,
    }

    return PipelineBundle(
        binary_clf=bin_clf,
        multi_clf=multi_clf,
        preprocessing_meta=meta,
        best_model_name="mock_model",
        class_names=cn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests for predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:

    def test_normal_traffic_returns_not_attack(self):
        """When binary classifier returns 0 (normal), is_attack must be False."""
        bundle = _mock_bundle(binary_pred=0)
        record = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

        with patch("src.predict.transform_single_record",
                   return_value=np.array([[1.0, 2.0, 3.0]])):
            result = predict(record, bundle)

        assert result["is_attack"] is False
        assert result["attack_type"] == "normal"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_attack_returns_is_attack_true(self):
        """When binary classifier returns 1 (attack), is_attack must be True."""
        bundle = _mock_bundle(binary_pred=1, multi_pred=1)
        record = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

        with patch("src.predict.transform_single_record",
                   return_value=np.array([[1.0, 2.0, 3.0]])):
            result = predict(record, bundle)

        assert result["is_attack"] is True
        assert result["attack_type"] != "normal"

    def test_attack_type_matches_class_names(self):
        """The returned attack_type must be one of the defined class names."""
        cn = ["normal","ddos","malware","phishing"]
        bundle = _mock_bundle(binary_pred=1, multi_pred=2, class_names=cn)
        record = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

        with patch("src.predict.transform_single_record",
                   return_value=np.array([[1.0, 2.0, 3.0]])):
            result = predict(record, bundle)

        assert result["attack_type"] in cn

    def test_confidence_in_unit_interval(self):
        """Confidence must always be between 0 and 1."""
        for pred in [0, 1]:
            bundle = _mock_bundle(binary_pred=pred)
            record = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

            with patch("src.predict.transform_single_record",
                       return_value=np.array([[1.0, 2.0, 3.0]])):
                result = predict(record, bundle)

            assert 0.0 <= result["confidence"] <= 1.0, (
                f"confidence={result['confidence']} out of range for pred={pred}"
            )

    def test_result_has_required_keys(self):
        """predict() output must contain all required keys."""
        bundle = _mock_bundle(binary_pred=1)
        record = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

        with patch("src.predict.transform_single_record",
                   return_value=np.array([[1.0, 2.0, 3.0]])):
            result = predict(record, bundle)

        for key in ("is_attack","attack_type","confidence","binary_confidence","all_scores"):
            assert key in result, f"Missing key: {key}"

    def test_binary_confidence_present_for_normal(self):
        """binary_confidence must be returned even for normal traffic."""
        bundle = _mock_bundle(binary_pred=0)
        record = {"f1": 1.0, "f2": 2.0, "f3": 3.0}

        with patch("src.predict.transform_single_record",
                   return_value=np.array([[1.0, 2.0, 3.0]])):
            result = predict(record, bundle)

        assert "binary_confidence" in result
        assert 0.0 <= result["binary_confidence"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests for predict_batch_fast()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBatchFast:

    def test_output_length_matches_input(self):
        """Output DataFrame must have the same number of rows as input."""
        n = 20
        X = np.random.randn(n, 3).astype(np.float32)
        bundle = _mock_bundle(binary_pred=1)

        # Override predict to return per-row results
        bundle.binary_clf.predict.return_value = np.array([1] * n)
        bundle.binary_clf.predict_proba.return_value = np.tile([0.1, 0.9], (n, 1))
        bundle.multi_clf.predict.return_value = np.array([2] * n)
        bundle.multi_clf.predict_proba.return_value = np.tile(
            [0.05, 0.05, 0.8, 0.05, 0.05], (n, 1)
        )

        result = predict_batch_fast(X, bundle)
        assert len(result) == n

    def test_output_columns(self):
        """Output must contain is_attack, attack_type, confidence columns."""
        n = 5
        X = np.random.randn(n, 3).astype(np.float32)
        bundle = _mock_bundle(binary_pred=0)
        bundle.binary_clf.predict.return_value = np.zeros(n, dtype=int)
        bundle.binary_clf.predict_proba.return_value = np.tile([0.95, 0.05], (n, 1))

        result = predict_batch_fast(X, bundle)
        assert set(result.columns) >= {"is_attack","attack_type","confidence"}

    def test_all_normal_when_binary_returns_zero(self):
        """If binary model flags everything as normal, no attacks should appear."""
        n = 10
        X = np.random.randn(n, 3).astype(np.float32)
        bundle = _mock_bundle(binary_pred=0)
        bundle.binary_clf.predict.return_value = np.zeros(n, dtype=int)
        bundle.binary_clf.predict_proba.return_value = np.tile([0.9, 0.1], (n, 1))

        result = predict_batch_fast(X, bundle)
        assert (result["is_attack"] == False).all()
        assert (result["attack_type"] == "normal").all()
