"""
src/train_models.py
-------------------
Defines model pipelines, hyperparameter configurations, cross-validation,
and training routines for the four PhishGuard classifiers:
  1. Decision Tree (Entropy / Information Gain)
  2. K-Nearest Neighbors (Euclidean distance with StandardScaler)
  3. Random Forest (Ensemble with 150 estimators)
  4. Support Vector Machine (RBF kernel with StandardScaler and probability calibration)

All models adhere to scikit-learn Pipeline architecture to prevent data leakage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_cleaning import FEATURE_COLUMNS, LABEL_PHISHING
from src.evaluation import evaluate_model, select_best_model

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Canonical model names
MODEL_NAMES = [
    "Decision Tree",
    "K-Nearest Neighbors",
    "Random Forest",
    "Support Vector Machine",
]


def build_pipelines(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Construct Scikit-Learn Pipelines for all four classifiers.

    Design & Justification:
    -----------------------
    1. Decision Tree:
       - Uses criterion="entropy" (Information Gain principle, analogous to ID3/C4.5
         implemented within scikit-learn's optimized CART framework).
       - Regularized with max_depth=10, min_samples_split=10, min_samples_leaf=5
         to prevent overfitting on categorical integer features.

    2. K-Nearest Neighbors:
       - Requires feature scaling because integer distances can otherwise distort
         neighborhood metrics.
       - StandardScaler is placed INSIDE the pipeline so it is fitted strictly on
         the training fold/split.
       - n_neighbors=7, distance-weighted Euclidean metric.

    3. Random Forest:
       - 150 trees, max_depth=15, min_samples_split=10, min_samples_leaf=4.
       - Robust ensemble reducing individual tree variance.

    4. Support Vector Machine:
       - RBF kernel with C=1.0, gamma='scale'.
       - StandardScaler placed inside pipeline to standardize distance space.
       - Wrapped in CalibratedClassifierCV(ensemble=False) to provide reliable
         probability estimates for ROC-AUC without using deprecated SVC(probability=True).
    """
    dt = Pipeline([
        ("clf", DecisionTreeClassifier(
            criterion="entropy",  # Information Gain (ID3 / C4.5 style)
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
        ))
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ))
    ])

    rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=random_state,
            ),
            ensemble=False,
        ))
    ])

    return {
        "Decision Tree": dt,
        "K-Nearest Neighbors": knn,
        "Random Forest": rf,
        "Support Vector Machine": svm,
    }


def perform_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Perform Stratified K-Fold cross-validation on training data only.

    Ensures that model comparison and hyperparameter choices are validated
    without leaking the independent test set.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_summary: Dict[str, Dict[str, float]] = {}

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    # Binarize target for standard positive label calculation in sklearn scorers
    # (Label -1 is Phishing; we binarize where 1 = Phishing, 0 = Legitimate)
    y_binary = (y_train == LABEL_PHISHING).astype(int)

    for name, pipeline in pipelines.items():
        logger.info("Running %d-fold cross-validation for '%s' ...", n_splits, name)
        scores = cross_validate(
            pipeline,
            X_train,
            y_binary,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_summary[name] = {
            "cv_accuracy_mean": float(np.round(np.mean(scores["test_accuracy"]), 4)),
            "cv_accuracy_std": float(np.round(np.std(scores["test_accuracy"]), 4)),
            "cv_f1_mean": float(np.round(np.mean(scores["test_f1"]), 4)),
            "cv_f1_std": float(np.round(np.std(scores["test_f1"]), 4)),
            "cv_recall_mean": float(np.round(np.mean(scores["test_recall"]), 4)),
            "cv_precision_mean": float(np.round(np.mean(scores["test_precision"]), 4)),
            "cv_roc_auc_mean": float(np.round(np.mean(scores["test_roc_auc"]), 4)),
        }
        logger.info(
            "  %s CV F1: %.4f (±%.4f) | CV Acc: %.4f",
            name,
            cv_summary[name]["cv_f1_mean"],
            cv_summary[name]["cv_f1_std"],
            cv_summary[name]["cv_accuracy_mean"],
        )

    return cv_summary


def train_and_evaluate_all(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train all pipelines, evaluate on test set, and optionally save artifacts.

    Returns
    -------
    results: List of evaluation metric dictionaries for each model.
    best_model: The selected best model based on F1-score with ROC-AUC tiebreaker.
    """
    results: List[Dict[str, Any]] = []

    for name, pipeline in pipelines.items():
        res = evaluate_model(
            name=name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(res)

        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            slug = name.replace(" ", "_").lower()
            artifact_path = models_dir / f"{slug}.joblib"
            joblib.dump(pipeline, artifact_path)
            logger.info("Saved model artifact: %s", artifact_path)

    best_model = select_best_model(results)
    return results, best_model
