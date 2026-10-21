"""
tests/test_feature_consistency.py
---------------------------------
Automated tests verifying feature extraction consistency, feature ordering,
vector dimensions, and determinism between training and inference.
"""

from __future__ import annotations

import math
import pytest
from src.url_features import (
    FEATURE_NAMES,
    extract_features,
    extract_features_dict,
)


class TestFeatureConsistency:
    """Ensure strict alignment of features between training and inference."""

    def test_canonical_feature_names_count(self) -> None:
        """Verify the exact canonical 16 features exist."""
        assert len(FEATURE_NAMES) == 16
        expected = [
            "url_length",
            "hostname_length",
            "path_length",
            "count_dots",
            "count_hyphens",
            "count_at",
            "count_slash",
            "count_digits",
            "digit_ratio",
            "is_https",
            "count_subdomains",
            "is_ip",
            "has_shortener",
            "has_hyphen_in_domain",
            "count_sensitive_keywords",
            "path_depth",
        ]
        assert FEATURE_NAMES == expected

    def test_dictionary_and_list_alignment(self) -> None:
        """extract_features list must match extract_features_dict in exact order."""
        test_url = "https://sub.example.com/path/to/page?user=1#section"
        f_dict = extract_features_dict(test_url)
        f_list = extract_features(test_url)

        assert len(f_dict) == 16
        assert len(f_list) == 16

        for i, name in enumerate(FEATURE_NAMES):
            assert name in f_dict
            assert f_list[i] == f_dict[name], f"Mismatch for feature {name} at index {i}"

    def test_feature_determinism(self) -> None:
        """Same URL must yield identical features across repeated invocations."""
        url = "https://github.com/torvalds/linux"
        f1 = extract_features(url)
        f2 = extract_features(url)
        assert f1 == f2

    def test_no_null_or_infinite_values(self) -> None:
        """Extracted features must never contain NaN, None, or infinite values."""
        sample_urls = [
            "https://github.com",
            "http://192.168.0.1/admin",
            "https://bit.ly/login-test",
            "https://a.b.c.d.e.f.example.org/path/sub/sub2?q=1&v=2",
            "https://user:pass@evil-domain-phish.xyz:8080/auth/login.html",
        ]
        for url in sample_urls:
            feats = extract_features(url)
            for val in feats:
                assert val is not None
                assert isinstance(val, (int, float))
                assert not math.isnan(val)
                assert not math.isinf(val)

    def test_ip_address_detection(self) -> None:
        """Feature is_ip must be 1 for raw IP addresses and 0 for domain names."""
        ip_feats = extract_features_dict("http://192.168.1.1/login")
        assert ip_feats["is_ip"] == 1

        dom_feats = extract_features_dict("https://github.com")
        assert dom_feats["is_ip"] == 0

    def test_shortener_detection(self) -> None:
        """Feature has_shortener must detect known URL shortening domains."""
        short_feats = extract_features_dict("https://bit.ly/secure-auth")
        assert short_feats["has_shortener"] == 1

        reg_feats = extract_features_dict("https://github.com")
        assert reg_feats["has_shortener"] == 0

    def test_sensitive_keywords_counter(self) -> None:
        """Detect sensitive security/credential tokens in URL segments."""
        phish_url = "http://fake-site.com/banking/secure/verify/login.php"
        f = extract_features_dict(phish_url)
        assert f["count_sensitive_keywords"] >= 3

        clean_url = "https://github.com/torvalds/linux"
        f_clean = extract_features_dict(clean_url)
        assert f_clean["count_sensitive_keywords"] == 0
