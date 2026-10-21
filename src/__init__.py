"""
src/__init__.py
---------------
PhishGuard core machine learning package.
"""

from src.data_cleaning import clean_dataset, load_dataset, split_data, FEATURE_COLUMNS, TARGET_COLUMN
from src.feature_extraction import extract_features, validate_url
from src.train_models import build_pipelines, perform_cross_validation, train_and_evaluate_all
from src.evaluation import evaluate_model, select_best_model
from src.prediction import predict_url, predict_feature_record, load_model, load_all_models

__all__ = [
    "clean_dataset",
    "load_dataset",
    "split_data",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "extract_features",
    "validate_url",
    "build_pipelines",
    "perform_cross_validation",
    "train_and_evaluate_all",
    "evaluate_model",
    "select_best_model",
    "predict_url",
    "predict_feature_record",
    "load_model",
    "load_all_models",
]
