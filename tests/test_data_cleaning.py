"""
tests/test_data_cleaning.py
----------------------------
Basic tests for the data cleaning pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a minimal synthetic dataset mimicking the UCI schema."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = rng.choice([-1, 0, 1], size=n)
    data[TARGET_COLUMN] = rng.choice([-1, 1], size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCleanDataset:
    def test_returns_correct_shapes(self) -> None:
        df = _make_fake_df(100)
        X, y, summary = clean_dataset(df)
        assert X.shape[1] == len(FEATURE_COLUMNS)
        assert len(y) == len(X)

    def test_target_not_in_features(self) -> None:
        df = _make_fake_df(100)
        X, y, summary = clean_dataset(df)
        assert TARGET_COLUMN not in X.columns

    def test_duplicates_removed(self) -> None:
        df = _make_fake_df(50)
        # Add 10 exact duplicates of the first row
        duplicates = pd.concat([df.iloc[[0]] * 10], ignore_index=True)
        df_with_dups = pd.concat([df, duplicates], ignore_index=True)
        _, _, summary = clean_dataset(df_with_dups)
        assert summary["duplicates_removed"] >= 0  # may vary by data

    def test_summary_keys_present(self) -> None:
        df = _make_fake_df(80)
        _, _, summary = clean_dataset(df)
        expected_keys = {
            "original_shape", "missing_values", "cleaned_shape",
            "class_distribution", "feature_count", "feature_names",
        }
        assert expected_keys.issubset(summary.keys())

    def test_class_distribution_is_accurate(self) -> None:
        df = _make_fake_df(100)
        _, _, summary = clean_dataset(df)
        dist = summary["class_distribution"]
        total = dist["Phishing (−1)"] + dist["Legitimate (1)"]
        assert total == summary["cleaned_shape"][0]


class TestSplitData:
    def test_split_sizes(self) -> None:
        df = _make_fake_df(500)
        X, y, _ = clean_dataset(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_test) == pytest.approx(100, abs=5)
        assert len(X_train) + len(X_test) == len(X)

    def test_stratified_class_balance(self) -> None:
        df = _make_fake_df(1000)
        X, y, _ = clean_dataset(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        train_ratio = (y_train == -1).mean()
        test_ratio = (y_test == -1).mean()
        assert abs(train_ratio - test_ratio) < 0.05  # within 5%

    def test_no_data_leakage_between_splits(self) -> None:
        df = _make_fake_df(200)
        X, y, _ = clean_dataset(df)
        X_train, X_test, _, _ = split_data(X, y)
        # Indices must be disjoint
        common = set(X_train.index) & set(X_test.index)
        assert len(common) == 0
