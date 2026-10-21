"""
train.py
--------
End-to-end training script for PhishGuard.

Usage
-----
    python train.py

Reads:   data/phishing.csv
Writes:  models/*.joblib, results/*.csv, results/*.png, results/*.json

All four classifiers are evaluated on the same held-out test set.
The best model is selected by F1-score (phishing class), with ROC-AUC as
a tiebreaker.  Model artifacts are saved so the Streamlit app can load
them without retraining.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Resolve project root so the script works from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "phishing.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_dataset, load_dataset, split_data, FEATURE_COLUMNS
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrices,
    plot_f1_comparison,
    plot_feature_importance,
    plot_roc_curves,
    save_best_model_info,
    save_comparison_csv,
    select_best_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
#
# Decision Tree and Random Forest work directly on the integer feature values.
# KNN and SVM benefit from feature scaling.
# Pipelines ensure that the scaler is fitted on training data only.


def build_pipelines() -> dict[str, Pipeline]:
    dt = Pipeline([
        ("clf", DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=RANDOM_STATE,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("PhishGuard — Training Pipeline")
    logger.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    logger.info("Loading dataset from '%s' …", DATA_PATH)
    try:
        df = load_dataset(DATA_PATH)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # 2. Clean dataset
    X, y, cleaning_summary = clean_dataset(df, results_dir=RESULTS_DIR)
    logger.info(
        "Clean dataset: %d rows × %d features", len(X), X.shape[1]
    )
    logger.info(
        "Class distribution — Phishing: %d | Legitimate: %d",
        cleaning_summary["class_distribution"]["Phishing (−1)"],
        cleaning_summary["class_distribution"]["Legitimate (1)"],
    )

    # 3. Train / test split
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_STATE)
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # 4. Save dataset info for the Streamlit app
    dataset_info = {
        **cleaning_summary,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(RESULTS_DIR / "dataset_info.json", "w") as fh:
        json.dump(dataset_info, fh, indent=2)

    # 5. Train & evaluate all models
    pipelines = build_pipelines()
    results = []

    for model_name, pipeline in pipelines.items():
        result = evaluate_model(
            name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(result)

        # Save the fitted pipeline
        artifact_path = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.joblib"
        joblib.dump(pipeline, artifact_path)
        logger.info("Saved pipeline: %s", artifact_path)

    # 6. Select best model
    best = select_best_model(results)
    save_best_model_info(
        best,
        criterion="Highest F1-score on phishing class (−1); ROC-AUC as tiebreaker",
        results_dir=RESULTS_DIR,
    )

    # 7. Generate plots & reports
    logger.info("Generating plots …")
    save_comparison_csv(results, RESULTS_DIR)
    plot_confusion_matrices(results, y_test, RESULTS_DIR)
    plot_roc_curves(results, y_test, RESULTS_DIR)
    plot_f1_comparison(results, RESULTS_DIR)

    # Feature importance for tree-based models
    for res in results:
        model_name = res["name"]
        pipeline = joblib.load(
            MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.joblib"
        )
        plot_feature_importance(pipeline, list(FEATURE_COLUMNS), model_name, RESULTS_DIR)

    # 8. Summary
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Best model : %s", best["name"])
    logger.info("  Accuracy : %.4f", best["accuracy"])
    logger.info("  F1-Score : %.4f", best["f1"])
    logger.info("  ROC-AUC  : %s", f"{best['roc_auc']:.4f}" if best["roc_auc"] else "N/A")
    logger.info("Artifacts  : %s", MODELS_DIR)
    logger.info("Results    : %s", RESULTS_DIR)
    logger.info("=" * 60)
    logger.info("Run  streamlit run app.py  to launch the dashboard.")


if __name__ == "__main__":
    main()
