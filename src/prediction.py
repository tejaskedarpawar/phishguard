"""
src/prediction.py
-----------------
Prediction and inference engine for PhishGuard.

Handles:
1. Model loading and pipeline caching.
2. Direct 30-feature record evaluation (ideal for verified benchmarks and testing).
3. URL lexical heuristic analysis (extracts 10 computable URL features with full disclosure).
4. Validation and error handling for inputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.data_cleaning import FEATURE_COLUMNS, LABEL_LEGITIMATE, LABEL_PHISHING, VALID_VALUES
from src.feature_extraction import COMPUTABLE_FEATURES, APPROXIMATED_FEATURES, extract_features, validate_url

logger = logging.getLogger(__name__)

# Canonical model mapping
MODEL_FILENAMES = {
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def load_model(model_name: str, models_dir: Path) -> Optional[Any]:
    """Load a single model pipeline by name."""
    filename = MODEL_FILENAMES.get(model_name)
    if not filename:
        slug = model_name.replace(" ", "_").lower() + ".joblib"
        filename = slug

    model_path = models_dir / filename
    if not model_path.exists():
        logger.warning("Model file not found: %s", model_path)
        return None

    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as exc:
        logger.error("Failed to load model from '%s': %s", model_path, exc)
        return None


def load_all_models(models_dir: Path) -> Dict[str, Any]:
    """Load all four trained model pipelines."""
    models = {}
    for name, filename in MODEL_FILENAMES.items():
        path = models_dir / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", name, exc)
    return models


def predict_feature_record(
    record: Union[Dict[str, int], pd.DataFrame],
    pipeline: Any,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """Perform exact inference on a full 30-feature record.

    Parameters
    ----------
    record:
        Dictionary or DataFrame containing all 30 expected feature keys.
    pipeline:
        Fitted Scikit-Learn Pipeline.
    model_name:
        Display name of the model.

    Returns
    -------
    dict with prediction label, confidence, and validation status.
    """
    if isinstance(record, dict):
        missing = [col for col in FEATURE_COLUMNS if col not in record]
        if missing:
            raise ValueError(f"Record is missing required features: {missing[:5]}...")
        df = pd.DataFrame([[record[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    elif isinstance(record, pd.DataFrame):
        missing = [col for col in FEATURE_COLUMNS if col not in record.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required features: {missing[:5]}...")
        df = record[FEATURE_COLUMNS].copy()
    else:
        raise TypeError("Record must be a dict or a pandas DataFrame.")

    # Validate values are in valid integer code space
    for col in FEATURE_COLUMNS:
        val = df[col].iloc[0]
        if val not in VALID_VALUES:
            logger.warning("Feature '%s' value %s is outside expected %s", col, val, VALID_VALUES)

    # Perform prediction
    raw_pred = int(pipeline.predict(df)[0])
    is_phishing = (raw_pred == LABEL_PHISHING)

    # Calculate confidence/probability
    confidence: Optional[float] = None
    prob_phishing: Optional[float] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(df)[0]
            classes = list(pipeline.classes_)
            if LABEL_PHISHING in classes:
                phish_idx = classes.index(LABEL_PHISHING)
                prob_phishing = float(proba[phish_idx])
            pred_idx = classes.index(raw_pred)
            confidence = float(proba[pred_idx])
        except Exception as exc:
            logger.debug("Failed to extract probability: %s", exc)

    return {
        "status": "success",
        "input_type": "full_feature_record",
        "prediction_code": raw_pred,
        "is_phishing": is_phishing,
        "label": "Potentially Phishing" if is_phishing else "Likely Legitimate",
        "confidence": confidence,
        "prob_phishing": prob_phishing,
        "model_name": model_name,
        "features_evaluated": 30,
        "disclaimer": "Full 30-feature assessment. Evaluated directly by the trained model pipeline.",
    }


def predict_url(
    url: str,
    pipeline: Any,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """Perform heuristic URL inference by extracting 10 address-bar features.

    Parameters
    ----------
    url:
        User-submitted URL.
    pipeline:
        Fitted Scikit-Learn Pipeline.
    model_name:
        Display name of the model.

    Returns
    -------
    dict with prediction label, confidence, extracted features, and disclosures.
    """
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        return {
            "status": "error",
            "error": error_msg,
            "label": "Invalid Input",
            "is_phishing": False,
        }

    feature_df, feature_details = extract_features(url)

    raw_pred = int(pipeline.predict(feature_df)[0])
    is_phishing = (raw_pred == LABEL_PHISHING)

    confidence: Optional[float] = None
    prob_phishing: Optional[float] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(feature_df)[0]
            classes = list(pipeline.classes_)
            if LABEL_PHISHING in classes:
                phish_idx = classes.index(LABEL_PHISHING)
                prob_phishing = float(proba[phish_idx])
            pred_idx = classes.index(raw_pred)
            confidence = float(proba[pred_idx])
        except Exception as exc:
            logger.debug("Failed to extract probability: %s", exc)

    return {
        "status": "success",
        "input_type": "url_lexical_heuristic",
        "url": url,
        "prediction_code": raw_pred,
        "is_phishing": is_phishing,
        "label": "Potentially Phishing" if is_phishing else "Likely Legitimate",
        "confidence": confidence,
        "prob_phishing": prob_phishing,
        "model_name": model_name,
        "feature_details": feature_details,
        "features_evaluated": 10,
        "features_approximated": 20,
        "computable_features": list(COMPUTABLE_FEATURES.keys()),
        "approximated_features": list(APPROXIMATED_FEATURES.keys()),
        "disclaimer": (
            "URL Heuristic Assessment: 10 address-bar lexical features were extracted directly from the URL. "
            "20 web-content and DNS features are held at baseline because live scanning is disabled for security and offline reproducibility. "
            "This result is an ML estimate, NOT a guarantee of website safety or malice."
        ),
    }


def load_demo_samples(samples_path: Path) -> Dict[str, Any]:
    """Load verified demonstration samples."""
    if not samples_path.exists():
        return {"legitimate_samples": [], "phishing_samples": [], "tested_urls": []}
    try:
        with open(samples_path, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to load demo samples from '%s': %s", samples_path, exc)
        return {"legitimate_samples": [], "phishing_samples": [], "tested_urls": []}
