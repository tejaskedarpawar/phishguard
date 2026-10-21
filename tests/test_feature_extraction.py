"""
tests/test_feature_extraction.py
---------------------------------
Tests for the URL feature extractor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import FEATURE_COLUMNS
from src.feature_extraction import extract_features, validate_url


class TestValidateUrl:
    def test_valid_https(self) -> None:
        valid, msg = validate_url("https://example.com")
        assert valid is True
        assert msg == ""

    def test_valid_bare_domain(self) -> None:
        valid, msg = validate_url("google.com")
        assert valid is True

    def test_empty_url(self) -> None:
        valid, msg = validate_url("")
        assert valid is False

    def test_whitespace_in_url(self) -> None:
        valid, msg = validate_url("https://ex ample.com")
        assert valid is False

    def test_very_long_url(self) -> None:
        valid, msg = validate_url("http://x.com/" + "a" * 2048)
        assert valid is False


class TestExtractFeatures:
    def test_output_columns_match_training(self) -> None:
        df, _ = extract_features("https://example.com")
        assert list(df.columns) == FEATURE_COLUMNS

    def test_output_shape_is_one_row(self) -> None:
        df, _ = extract_features("https://example.com")
        assert df.shape == (1, 30)

    def test_values_in_valid_range(self) -> None:
        df, _ = extract_features("https://example.com/path?q=1")
        for col in FEATURE_COLUMNS:
            assert df[col].iloc[0] in {-1, 0, 1}, f"Column {col} out of range"

    def test_ip_address_detected(self) -> None:
        df, _ = extract_features("http://192.168.1.1/login")
        assert df["having_IP_Address"].iloc[0] == -1

    def test_normal_domain_not_ip(self) -> None:
        df, _ = extract_features("https://google.com")
        assert df["having_IP_Address"].iloc[0] == 1

    def test_at_symbol_detected(self) -> None:
        df, _ = extract_features("http://user@evil.com/login")
        assert df["having_At_Symbol"].iloc[0] == -1

    def test_shortener_detected(self) -> None:
        df, _ = extract_features("https://bit.ly/abc123")
        assert df["Shortining_Service"].iloc[0] == -1

    def test_dash_in_domain_detected(self) -> None:
        df, _ = extract_features("https://secure-login-paypal.com")
        assert df["Prefix_Suffix"].iloc[0] == -1

    def test_https_scheme(self) -> None:
        df, _ = extract_features("https://legitimate.com")
        assert df["SSLfinal_State"].iloc[0] == 1

    def test_http_scheme(self) -> None:
        df, _ = extract_features("http://example.com")
        assert df["SSLfinal_State"].iloc[0] == -1

    def test_feature_details_returned(self) -> None:
        _, details = extract_features("https://google.com")
        assert isinstance(details, dict)
        assert len(details) > 0
