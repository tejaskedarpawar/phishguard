"""
tests/test_prediction.py
------------------------
Tests for URL validation, feature extraction, full-record inference, URL heuristic
inference, demo test fixtures, and missing-artifact resilience.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import FEATURE_COLUMNS
from src.feature_extraction import extract_features, validate_url
from src.prediction import (
    load_all_models,
    load_demo_samples,
    load_model,
    predict_feature_record,
    predict_url,
)
from src.train_models import build_pipelines


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Build and fit a fast Decision Tree pipeline on small synthetic data."""
    rng = np.random.default_rng(42)
    data = {col: rng.choice([-1, 0, 1], size=60) for col in FEATURE_COLUMNS}
    X = pd.DataFrame(data)
    y = pd.Series(rng.choice([-1, 1], size=60))
    pipe = build_pipelines()["Decision Tree"]
    pipe.fit(X, y)
    return pipe


class TestUrlValidation:
    def test_valid_http_and_https(self) -> None:
        assert validate_url("https://secure.bank.com")[0] is True
        assert validate_url("http://example.org/test?id=1")[0] is True

    def test_valid_bare_domain(self) -> None:
        assert validate_url("google.com")[0] is True
        assert validate_url("sub.domain.co.uk/path")[0] is True

    def test_reject_empty_and_whitespace(self) -> None:
        assert validate_url("")[0] is False
        assert validate_url("   ")[0] is False
        assert validate_url("https://in valid.com")[0] is False

    def test_reject_excessively_long_url(self) -> None:
        long_url = "https://example.com/" + "a" * 2050
        assert validate_url(long_url)[0] is False


class TestUrlFeatureExtraction:
    def test_column_names_and_order_match_training(self) -> None:
        """Extracted dataframe columns must exactly match FEATURE_COLUMNS in name and order."""
        df, _ = extract_features("https://github.com")
        assert list(df.columns) == FEATURE_COLUMNS
        assert df.shape == (1, 30)

    def test_feature_values_in_valid_range(self) -> None:
        df, _ = extract_features("https://paypal-update-account.com/login?u=1")
        for col in FEATURE_COLUMNS:
            val = df[col].iloc[0]
            assert val in {-1, 0, 1, 2}, f"Column {col} has unexpected value {val}"

    def test_ip_address_heuristic(self) -> None:
        df_ip, _ = extract_features("http://192.168.0.1/admin")
        assert df_ip["having_IP_Address"].iloc[0] == -1

        df_dom, _ = extract_features("https://example.com/admin")
        assert df_dom["having_IP_Address"].iloc[0] == 1

    def test_at_symbol_heuristic(self) -> None:
        df_at, _ = extract_features("http://user:pass@evil.com")
        assert df_at["having_At_Symbol"].iloc[0] == -1

    def test_shortener_heuristic(self) -> None:
        df_short, _ = extract_features("https://bit.ly/sample-link")
        assert df_short["Shortining_Service"].iloc[0] == -1

    def test_prefix_suffix_dash_heuristic(self) -> None:
        df_dash, _ = extract_features("https://secure-login-bank.com")
        assert df_dash["Prefix_Suffix"].iloc[0] == -1


class TestInferenceWorkflows:
    def test_predict_full_feature_record(self, fitted_pipeline) -> None:
        """Full 30-feature record prediction must return valid prediction code and label."""
        record = {col: 1 for col in FEATURE_COLUMNS}
        res = predict_feature_record(record, fitted_pipeline, model_name="TestDT")
        assert res["status"] == "success"
        assert res["prediction_code"] in {-1, 1}
        assert res["label"] in ["Potentially Phishing", "Likely Legitimate"]
        assert res["features_evaluated"] == 30

    def test_predict_full_record_rejects_missing_features(self, fitted_pipeline) -> None:
        """Missing features in record must raise ValueError."""
        incomplete_record = {"having_IP_Address": 1, "URL_Length": -1}
        with pytest.raises(ValueError):
            predict_feature_record(incomplete_record, fitted_pipeline)

    def test_predict_url_valid(self, fitted_pipeline) -> None:
        """Valid URL prediction must return structured dictionary with disclosures."""
        res = predict_url("https://wikipedia.org", fitted_pipeline, model_name="TestDT")
        assert res["status"] == "success"
        assert res["input_type"] == "url_lexical_heuristic"
        assert res["features_evaluated"] == 10
        assert res["features_approximated"] == 20
        assert "disclaimer" in res
        assert res["label"] in ["Potentially Phishing", "Likely Legitimate"]

    def test_predict_url_invalid(self, fitted_pipeline) -> None:
        """Invalid URL must return error status without crashing."""
        res = predict_url("", fitted_pipeline, model_name="TestDT")
        assert res["status"] == "error"
        assert "error" in res


class TestDemoFixturesAndArtifacts:
    def test_demo_samples_loading(self) -> None:
        """Demo samples fixture must load successfully and contain legitimate and phishing entries."""
        demo_path = PROJECT_ROOT / "data" / "demo_samples.json"
        if demo_path.exists():
            fixtures = load_demo_samples(demo_path)
            assert len(fixtures.get("legitimate_samples", [])) > 0
            assert len(fixtures.get("phishing_samples", [])) > 0
            assert len(fixtures.get("tested_urls", [])) > 0

    def test_missing_model_handling(self) -> None:
        """Loading a missing model must return None safely without raising an unhandled exception."""
        model = load_model("Nonexistent Model", Path("/invalid/path"))
        assert model is None
