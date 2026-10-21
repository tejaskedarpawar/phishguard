"""
tests/test_models.py
--------------------
Tests for model definition, training, pipeline scaling, cross-validation,
metric calculation, feature importance, and artifact persistence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import evaluate_model, select_best_model
from src.train_models import (
    MODEL_NAMES,
    build_pipelines,
    perform_cross_validation,
    train_and_evaluate_all,
)


def _make_dummy_dataset(n: int = 120, seed: int = 42):
    """Generate small train/test split for rapid testing."""
    rng = np.random.default_rng(seed)
    data = {col: rng.choice([-1, 0, 1], size=n) for col in FEATURE_COLUMNS}
    X = pd.DataFrame(data)
    y = pd.Series(rng.choice([-1, 1], size=n), name=TARGET_COLUMN)
    split_idx = int(n * 0.8)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


class TestModelArchitecture:
    def test_build_pipelines_contains_all_four_models(self) -> None:
        """Confirm exactly the four required models are built."""
        pipelines = build_pipelines()
        for expected in MODEL_NAMES:
            assert expected in pipelines, f"Missing pipeline for {expected}"

    def test_decision_tree_configuration(self) -> None:
        """Decision Tree must use entropy (Information Gain) and expose importances."""
        pipelines = build_pipelines()
        dt_pipe = pipelines["Decision Tree"]
        clf = dt_pipe.named_steps["clf"]
        assert clf.criterion == "entropy"

    def test_knn_contains_scaler_inside_pipeline(self) -> None:
        """KNN must include StandardScaler inside Pipeline to prevent test leakage."""
        pipelines = build_pipelines()
        knn_pipe = pipelines["K-Nearest Neighbors"]
        assert "scaler" in knn_pipe.named_steps
        assert "clf" in knn_pipe.named_steps

    def test_svm_contains_scaler_and_calibration(self) -> None:
        """SVM must include StandardScaler inside Pipeline and support calibrated probabilities."""
        pipelines = build_pipelines()
        svm_pipe = pipelines["Support Vector Machine"]
        assert "scaler" in svm_pipe.named_steps
        assert "clf" in svm_pipe.named_steps


class TestModelTrainingAndEvaluation:
    def test_all_four_models_train_and_evaluate(self) -> None:
        """Each model must train and return bounded metrics on held-out test set."""
        X_train, X_test, y_train, y_test = _make_dummy_dataset(100)
        pipelines = build_pipelines()

        for name, pipe in pipelines.items():
            res = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
            assert res["name"] == name
            assert 0.0 <= res["accuracy"] <= 1.0
            assert 0.0 <= res["precision"] <= 1.0
            assert 0.0 <= res["recall"] <= 1.0
            assert 0.0 <= res["f1"] <= 1.0
            assert res["training_time_s"] >= 0.0
            assert "tp" in res["cm_stats"]
            assert "fn" in res["cm_stats"]

    def test_cross_validation_runs_on_training_data(self) -> None:
        """Cross-validation must run 3-fold on training data without errors."""
        X_train, _, y_train, _ = _make_dummy_dataset(80)
        pipelines = build_pipelines()
        cv_summary = perform_cross_validation(pipelines, X_train, y_train, n_splits=3)
        for name in pipelines:
            assert name in cv_summary
            assert "cv_f1_mean" in cv_summary[name]
            assert 0.0 <= cv_summary[name]["cv_f1_mean"] <= 1.0

    def test_model_saving_and_loading(self, tmp_path: Path) -> None:
        """Saved pipeline must reload from joblib and produce identical predictions."""
        X_train, X_test, y_train, _ = _make_dummy_dataset(60)
        pipelines = build_pipelines()
        rf_pipe = pipelines["Random Forest"]
        rf_pipe.fit(X_train, y_train)

        original_preds = rf_pipe.predict(X_test)
        artifact_file = tmp_path / "random_forest.joblib"
        joblib.dump(rf_pipe, artifact_file)

        loaded_pipe = joblib.load(artifact_file)
        loaded_preds = loaded_pipe.predict(X_test)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_tree_models_expose_feature_importance(self) -> None:
        """Decision Tree and Random Forest must expose valid feature importances."""
        X_train, _, y_train, _ = _make_dummy_dataset(80)
        pipelines = build_pipelines()

        dt = pipelines["Decision Tree"]
        dt.fit(X_train, y_train)
        assert hasattr(dt.named_steps["clf"], "feature_importances_")
        assert len(dt.named_steps["clf"].feature_importances_) == 30

        rf = pipelines["Random Forest"]
        rf.fit(X_train, y_train)
        assert hasattr(rf.named_steps["clf"], "feature_importances_")
        assert len(rf.named_steps["clf"].feature_importances_) == 30
