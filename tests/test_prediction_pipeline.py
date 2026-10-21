"""
tests/test_prediction_pipeline.py
---------------------------------
Automated end-to-end unit and regression tests for the PhishGuard prediction pipeline.
Specifically validates model loading, input validation, legitimate URL classification,
phishing detection, and the GitHub false-positive regression.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from src.prediction import predict_url

MODEL_PATH = Path("models/best_model.joblib")


@pytest.fixture(scope="module")
def ensure_model_exists() -> Path:
    """Ensure trained model artifact exists for tests."""
    assert MODEL_PATH.exists(), f"Model artifact {MODEL_PATH} must exist before running prediction tests."
    return MODEL_PATH


class TestPredictionPipeline:
    """End-to-end prediction tests on live model pipeline."""

    def test_missing_model_error_handling(self, tmp_path: Path) -> None:
        """Pipeline must gracefully report when model file does not exist."""
        non_existent = tmp_path / "missing_model.joblib"
        res = predict_url("https://github.com", non_existent)
        assert res["status"] == "error"
        assert "not found" in res["error_message"].lower()

    def test_invalid_url_error_handling(self, ensure_model_exists: Path) -> None:
        """Pipeline must cleanly reject malformed or blank inputs."""
        res = predict_url("", ensure_model_exists)
        assert res["status"] == "error"
        assert "url cannot be empty" in res["error_message"].lower()

        res_ws = predict_url("https://git hub.com", ensure_model_exists)
        assert res_ws["status"] == "error"
        assert "whitespace" in res_ws["error_message"].lower()

    def test_github_regression_not_flagged_as_phishing(self, ensure_model_exists: Path) -> None:
        """REGRESSION TEST: Ensure github.com variations are classified as Legitimate."""
        github_urls = [
            "https://github.com",
            "https://www.github.com",
            "https://github.com/features",
            "https://docs.github.com",
        ]
        for url in github_urls:
            res = predict_url(url, ensure_model_exists)
            assert res["status"] == "success", f"Prediction failed for {url}: {res.get('error_message')}"
            assert res["prediction"] == 0, (
                f"REGRESSION FAILURE: {url} was incorrectly flagged as phishing! "
                f"Probabilities: {res.get('phishing_probability')}"
            )
            assert res["predicted_label"] == "Legitimate"
            assert "LOW RISK" in res["risk_status"]
            # Confidence should be greater than 50%
            assert res["confidence"] > 0.50

    def test_legitimate_domains_classification(self, ensure_model_exists: Path) -> None:
        """Ensure standard top web services are classified as Legitimate."""
        legit_samples = [
            "https://www.google.com",
            "https://stackoverflow.com/questions",
            "https://www.apple.com/shop",
            "https://news.ycombinator.com",
            "https://www.wikipedia.org/wiki/Phishing",
        ]
        for url in legit_samples:
            res = predict_url(url, ensure_model_exists)
            assert res["status"] == "success"
            assert res["prediction"] == 0, f"Expected Legitimate for {url}, got {res['predicted_label']}"

    def test_phishing_attacks_detection(self, ensure_model_exists: Path) -> None:
        """Ensure real-world phishing URL patterns are detected as Phishing."""
        phishing_samples = [
            "http://192.168.1.100/paypal-update/login.php",
            "http://secure-login-account-verification-alert.com/signin?user=test",
            "http://paypal.com.verify-security-check.xyz/login?account=123",
            "http://apple-id-verify-account-locked.org/login",
            "http://bit.ly/secure-banking-update-notice",
            "http://chase-bank-online-verification.net/signin.htm",
            "http://netflix-billing-issue-update.com/account/login",
        ]
        for url in phishing_samples:
            res = predict_url(url, ensure_model_exists)
            assert res["status"] == "success"
            assert res["prediction"] == 1, f"Expected Phishing for {url}, got {res['predicted_label']}"
            assert res["predicted_label"] == "Phishing"
            assert "HIGH RISK" in res["risk_status"]
            assert res["phishing_probability"] > 0.50

    def test_prediction_output_schema(self, ensure_model_exists: Path) -> None:
        """Verify the full dictionary response schema returned by predict_url."""
        res = predict_url("https://github.com", ensure_model_exists)
        expected_keys = {
            "status",
            "url",
            "normalized_url",
            "hostname",
            "prediction",
            "predicted_label",
            "confidence",
            "phishing_probability",
            "legitimate_probability",
            "risk_status",
            "model_name",
            "feature_highlights",
            "disclaimer",
        }
        assert expected_keys.issubset(set(res.keys()))
