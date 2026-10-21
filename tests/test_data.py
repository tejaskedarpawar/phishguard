"""
tests/test_data.py
------------------
Tests for dataset loading, schema validation, data cleaning, target-label mapping,
and split behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (
    FEATURE_COLUMNS,
    LABEL_LEGITIMATE,
    LABEL_PHISHING,
    TARGET_COLUMN,
    clean_dataset,
    load_dataset,
    split_data,
)


def _make_synthetic_df(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Create a minimal synthetic dataframe with exact schema."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.choice([-1, 0, 1], size=n)
    data[TARGET_COLUMN] = rng.choice([-1, 1], size=n)
    return pd.DataFrame(data)


class TestDatasetSchema:
    def test_feature_columns_count(self) -> None:
        """Verify exactly 30 feature columns are defined."""
        assert len(FEATURE_COLUMNS) == 30

    def test_target_label_conventions(self) -> None:
        """Verify explicit target label mapping: -1 is Phishing, 1 is Legitimate."""
        assert LABEL_PHISHING == -1
        assert LABEL_LEGITIMATE == 1

    def test_load_dataset_nonexistent_raises_error(self) -> None:
        """Attempting to load a missing file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/path/phishing.csv"))

    def test_load_dataset_missing_columns_raises_error(self, tmp_path: Path) -> None:
        """A CSV missing required columns must raise ValueError with descriptive text."""
        bad_csv = tmp_path / "bad.csv"
        bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        bad_df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError) as exc:
            load_dataset(bad_csv)
        assert "Required columns not found" in str(exc.value)

    def test_real_dataset_loads_if_present(self) -> None:
        """If data/phishing.csv exists on disk, it must load and pass schema validation."""
        real_csv = PROJECT_ROOT / "data" / "phishing.csv"
        if real_csv.exists():
            df = load_dataset(real_csv)
            assert len(df) > 0
            assert set(FEATURE_COLUMNS).issubset(df.columns)
            assert TARGET_COLUMN in df.columns


class TestDataCleaning:
    def test_clean_dataset_separates_features_and_target(self) -> None:
        """X must contain 30 features, y must be the target series, Target not in X."""
        df = _make_synthetic_df(100)
        X, y, summary = clean_dataset(df)
        assert X.shape[1] == 30
        assert len(y) == len(X)
        assert TARGET_COLUMN not in X.columns
        assert list(X.columns) == FEATURE_COLUMNS

    def test_clean_dataset_removes_exact_duplicates(self) -> None:
        """Duplicate rows must be detected, counted, and dropped."""
        df = _make_synthetic_df(50)
        # Duplicate first 5 rows
        dups = df.iloc[:5].copy()
        df_with_dups = pd.concat([df, dups], ignore_index=True)
        _, _, summary = clean_dataset(df_with_dups)
        assert summary["duplicates_removed"] >= 5
        assert summary["cleaned_shape"][0] == 50

    def test_clean_dataset_reports_class_distribution(self) -> None:
        """Summary must accurately count both classes."""
        df = _make_synthetic_df(120)
        _, _, summary = clean_dataset(df)
        dist = summary["class_distribution"]
        assert "Phishing (−1)" in dist
        assert "Legitimate (1)" in dist
        assert dist["Phishing (−1)"] + dist["Legitimate (1)"] == summary["cleaned_shape"][0]


class TestDataSplitting:
    def test_stratified_split_ratios(self) -> None:
        """Split must preserve 80/20 proportion and balance."""
        df = _make_synthetic_df(400)
        X, y, _ = clean_dataset(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
        assert len(X_train) == 320
        assert len(X_test) == 80

        # Class ratio must match closely
        train_phish_pct = (y_train == -1).mean()
        test_phish_pct = (y_test == -1).mean()
        assert abs(train_phish_pct - test_phish_pct) < 0.05

    def test_no_data_leakage_in_splits(self) -> None:
        """Indices of training and testing sets must be completely disjoint."""
        df = _make_synthetic_df(200)
        X, y, _ = clean_dataset(df)
        X_train, X_test, _, _ = split_data(X, y, test_size=0.20, random_state=42)
        overlap = set(X_train.index).intersection(set(X_test.index))
        assert len(overlap) == 0
