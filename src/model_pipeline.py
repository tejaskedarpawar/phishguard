"""
src/model_pipeline.py
---------------------
Model pipeline construction, cross-validation, training, and artifact persistence.
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

MODEL_NAMES = [
    "Random Forest",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Support Vector Machine",
]


def build_model_pipelines(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Build Scikit-Learn pipelines for all four comparative algorithms."""
    rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    dt = Pipeline([
        ("clf", DecisionTreeClassifier(
            criterion="entropy",
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
        "Random Forest": rf,
        "Decision Tree": dt,
        "K-Nearest Neighbors": knn,
        "Support Vector Machine": svm,
    }


def run_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Execute stratified 5-fold cross-validation on training data only."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = {}

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, pipe in pipelines.items():
        logger.info("Running %d-fold cross-validation for '%s' ...", n_splits, name)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "cv_accuracy_mean": float(np.round(np.mean(scores["test_accuracy"]), 4)),
            "cv_accuracy_std": float(np.round(np.std(scores["test_accuracy"]), 4)),
            "cv_f1_mean": float(np.round(np.mean(scores["test_f1"]), 4)),
            "cv_f1_std": float(np.round(np.std(scores["test_f1"]), 4)),
            "cv_recall_mean": float(np.round(np.mean(scores["test_recall"]), 4)),
            "cv_precision_mean": float(np.round(np.mean(scores["test_precision"]), 4)),
            "cv_roc_auc_mean": float(np.round(np.mean(scores["test_roc_auc"]), 4)),
        }

    return cv_results


def evaluate_single_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train pipeline and calculate full test metrics."""
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    y_pred = pipeline.predict(X_test)

    # Probabilities for positive class (1 = Phishing)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            classes = list(pipeline.classes_)
            pos_idx = classes.index(1)
            y_prob = proba[:, pos_idx]
        except Exception:
            pass

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None

    # Confusion matrix: row 0=Legit (0), row 1=Phish (1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    fpr = round(fp / max(fp + tn, 1), 4)
    fnr = round(fn / max(fn + tp, 1), 4)

    return {
        "name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "training_time_s": round(training_time, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "fpr": fpr,
        "fnr": fnr,
        "pipeline": pipeline,
    }


def train_and_select_best(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train all pipelines, save artifacts, and select the highest F1 model."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name, pipe in pipelines.items():
        res = evaluate_single_pipeline(name, pipe, X_train, y_train, X_test, y_test)
        results.append(res)

        slug = name.replace(" ", "_").lower()
        model_file = models_dir / f"{slug}.joblib"
        joblib.dump(res["pipeline"], model_file)
        logger.info("Saved pipeline artifact: %s", model_file)

    # Best model chosen by highest F1 score on phishing with ROC-AUC tiebreaker
    best = max(results, key=lambda r: (r["f1"], r["roc_auc"] or 0.0))

    # Save as canonical best_model.joblib
    best_file = models_dir / "best_model.joblib"
    joblib.dump(best["pipeline"], best_file)
    logger.info("Saved selected best model: %s (F1=%.4f)", best["name"], best["f1"])

    return results, best
