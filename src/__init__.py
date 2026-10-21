"""
src/__init__.py
---------------
PhishGuard core package.
"""

from src.url_features import FEATURE_NAMES, extract_features, normalize_url
from src.data_loader import (
    LABEL_LEGITIMATE,
    LABEL_NAMES,
    LABEL_PHISHING,
    get_dataset_summary,
    load_dataset,
)
from src.preprocessing import extract_dataset_features, split_data
from src.model_pipeline import build_model_pipelines, run_cross_validation, train_and_select_best
from src.prediction import predict_url

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "normalize_url",
    "load_dataset",
    "get_dataset_summary",
    "LABEL_LEGITIMATE",
    "LABEL_PHISHING",
    "LABEL_NAMES",
    "extract_dataset_features",
    "split_data",
    "build_model_pipelines",
    "run_cross_validation",
    "train_and_select_best",
    "predict_url",
]
