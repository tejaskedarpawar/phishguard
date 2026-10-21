"""
tests/test_url_normalization.py
--------------------------------
Automated unit tests for URL validation and canonical normalization.
"""

from __future__ import annotations

import pytest
from src.url_features import normalize_url


class TestUrlNormalization:
    """Validate behavior of normalize_url under diverse input conditions."""

    def test_empty_and_whitespace_rejection(self) -> None:
        """Reject empty strings, None, and whitespace-only inputs."""
        is_valid, _, err = normalize_url("")
        assert not is_valid
        assert "empty" in err.lower()

        is_valid, _, err = normalize_url("   ")
        assert not is_valid
        assert "blank" in err.lower()

        is_valid, _, err = normalize_url(None)  # type: ignore[arg-type]
        assert not is_valid

    def test_whitespace_inside_url_rejected(self) -> None:
        """Reject URLs containing internal whitespace or newline characters."""
        is_valid, _, err = normalize_url("https://git hub.com")
        assert not is_valid
        assert "whitespace" in err.lower()

        is_valid, _, err = normalize_url("https://github.com\n/malicious")
        assert not is_valid

    def test_excessive_length_rejection(self) -> None:
        """Reject inputs exceeding 2048 characters."""
        oversized = "https://example.com/" + "a" * 2100
        is_valid, _, err = normalize_url(oversized)
        assert not is_valid
        assert "maximum permitted length" in err.lower()

    def test_scheme_prepending(self) -> None:
        """URLs without scheme should default to https://."""
        is_valid, norm, err = normalize_url("github.com")
        assert is_valid
        assert norm == "https://github.com"
        assert err == ""

        is_valid, norm, _ = normalize_url("www.github.com/features")
        assert is_valid
        assert norm == "https://www.github.com/features"

    def test_scheme_and_hostname_lowercasing(self) -> None:
        """Scheme and hostname must be lowercased while preserving path case."""
        is_valid, norm, _ = normalize_url("HTTPS://GITHUB.COM/Torvalds/Linux")
        assert is_valid
        assert norm == "https://github.com/Torvalds/Linux"

    def test_port_normalization(self) -> None:
        """Standard ports (80, 443) should be stripped; non-standard ports kept."""
        is_valid, norm, _ = normalize_url("https://github.com:443/test")
        assert is_valid
        assert norm == "https://github.com/test"

        is_valid, norm, _ = normalize_url("http://example.com:8080/api")
        assert is_valid
        assert norm == "http://example.com:8080/api"

    def test_query_and_fragment_preservation(self) -> None:
        """Query strings and fragments must be preserved."""
        is_valid, norm, _ = normalize_url("https://example.com/search?q=machine+learning#top")
        assert is_valid
        assert norm == "https://example.com/search?q=machine+learning#top"

    def test_ip_address_hostnames(self) -> None:
        """Direct IPv4 hostnames must normalize correctly."""
        is_valid, norm, _ = normalize_url("http://192.168.1.1:8000/login.php")
        assert is_valid
        assert norm == "http://192.168.1.1:8000/login.php"

    def test_defanged_threat_intel_urls(self) -> None:
        """Refang hxxp, hxxps, [.], and [:] indicators."""
        is_valid, norm, _ = normalize_url("hxxps://malicious-domain[.]xyz/verify")
        assert is_valid
        assert norm == "https://malicious-domain.xyz/verify"

        is_valid, norm, _ = normalize_url("hxxp[:]//192.168.1.1/login")
        assert is_valid
        assert norm == "http://192.168.1.1/login"

    def test_quoted_and_bracketed_urls(self) -> None:
        """Clean quotes, angle brackets, and trailing sentence punctuation."""
        is_valid, norm, _ = normalize_url('"https://github.com"')
        assert is_valid
        assert norm == "https://github.com"

        is_valid, norm, _ = normalize_url("<https://www.github.com/features.>")
        assert is_valid
        assert norm == "https://www.github.com/features"

    def test_subpath_trailing_slash_normalization(self) -> None:
        """Trailing slashes on sub-paths must be normalized consistently."""
        is_valid, norm, _ = normalize_url("https://www.python.org/downloads/")
        assert is_valid
        assert norm == "https://www.python.org/downloads"

    def test_unsupported_schemes_rejection(self) -> None:
        """Reject non-web schemes like javascript: or data:."""
        is_valid, _, err = normalize_url("javascript:alert(1)")
        assert not is_valid
        assert "unsupported protocol" in err.lower()

    def test_internationalized_domain_names(self) -> None:
        """Unicode IDN domains must convert to punycode cleanly."""
        is_valid, norm, _ = normalize_url("https://яндекс.рф/search")
        assert is_valid
        assert "xn--" in norm

