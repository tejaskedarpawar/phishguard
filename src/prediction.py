"""
src/prediction.py
-----------------
Single-URL inference engine for PhishGuard.

Takes a raw URL, normalizes it, extracts the exact 16 lexical features,
runs the fitted Scikit-Learn pipeline, and returns an honest, calibrated
prediction with a safety disclaimer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import joblib
import pandas as pd

from src.url_features import FEATURE_NAMES, extract_features, normalize_url

logger = logging.getLogger(__name__)


def predict_url(url: str, model_path: Path) -> Dict[str, Any]:
    """Classify a single URL as Legitimate (0) or Phishing (1).

    Parameters
    ----------
    url:
        Raw user-provided URL string.
    model_path:
        Path to fitted model pipeline joblib file (e.g. models/best_model.joblib).

    Returns
    -------
    dict
        Structured prediction output containing normalized URL, predicted label,
        calibrated probability, feature breakdown, and disclaimers.
    """
    # 1. Validation and Normalization
    is_valid, norm_url, err_msg = normalize_url(url)
    if not is_valid:
        return {
            "status": "error",
            "raw_url": url,
            "error_message": err_msg,
            "predicted_label": "Invalid Input",
            "prediction": -1,
        }

    # 2. Check model existence
    if not model_path.exists():
        return {
            "status": "error",
            "raw_url": url,
            "normalized_url": norm_url,
            "error_message": f"Model artifact not found at: {model_path}. Please run 'python train.py' first.",
            "predicted_label": "Model Not Found",
            "prediction": -1,
        }

    # 3. Load pipeline
    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        return {
            "status": "error",
            "raw_url": url,
            "normalized_url": norm_url,
            "error_message": f"Failed to load model pipeline: {exc}",
            "predicted_label": "Load Error",
            "prediction": -1,
        }

    # 4. Extract exact features (same schema and order as training)
    feat_vector = extract_features(norm_url)
    feat_df = pd.DataFrame([feat_vector], columns=FEATURE_NAMES)

    # 5. Execute prediction
    raw_pred = int(pipeline.predict(feat_df)[0])
    is_phishing = (raw_pred == 1)

    # 6. Extract probability estimates if supported
    confidence = None
    prob_phish = None
    prob_legit = None

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(feat_df)[0]
            classes = list(pipeline.classes_)
            if 0 in classes and 1 in classes:
                prob_legit = round(float(proba[classes.index(0)]), 4)
                prob_phish = round(float(proba[classes.index(1)]), 4)
                confidence = prob_phish if is_phishing else prob_legit
        except Exception as exc:
            logger.debug("predict_proba failed: %s", exc)

    parsed = urlparse(norm_url)
    hostname = parsed.hostname or ""

    # Feature highlights for terminal display
    feature_highlights = {
        "URL Length": feat_vector[0],
        "Hostname Length": feat_vector[1],
        "Dot Count": feat_vector[3],
        "Hyphen Count": feat_vector[4],
        "Subdomains": feat_vector[10],
        "IP Address Used": "Yes" if feat_vector[11] == 1 else "No",
        "URL Shortener": "Yes" if feat_vector[12] == 1 else "No",
        "Sensitive Keywords": feat_vector[14],
    }

    # Extract model name
    estimator_step = pipeline.named_steps.get("clf", pipeline)
    model_name = type(estimator_step).__name__

    return {
        "status": "success",
        "url": url,
        "raw_url": url,
        "normalized_url": norm_url,
        "hostname": hostname,
        "prediction": raw_pred,
        "predicted_label": "Phishing" if is_phishing else "Legitimate",
        "risk_status": "HIGH RISK (Potentially Phishing)" if is_phishing else "LOW RISK (Likely Legitimate)",
        "confidence": confidence,
        "phishing_probability": prob_phish,
        "legitimate_probability": prob_legit,
        "model_name": model_name,
        "feature_highlights": feature_highlights,
        "disclaimer": (
            "Note: This is an ML assessment based on 16 lexical and structural URL attributes. "
            "It is not an absolute guarantee of website safety or malice."
        ),
    }
