"""
src/preprocessing.py
--------------------
Feature extraction orchestration and stratified splitting for PhishGuard.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.url_features import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42


def extract_dataset_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert raw URL strings in dataframe to the 16 lexical features.

    Parameters
    ----------
    df:
        DataFrame containing 'url' and 'label' columns.

    Returns
    -------
    X: pd.DataFrame
        16-column numerical feature matrix.
    y: pd.Series
        Binary target label vector.
    """
    logger.info("Extracting lexical URL features for %d URLs ...", len(df))
    feature_matrix = [extract_features(u) for u in df["url"]]
    X = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    y = df["label"].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train/test split.

    Ensures balanced class representations across partitions and guarantees
    that indices between train and test are strictly disjoint.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Stratified Split: Train = %d instances, Test = %d instances (test_size=%.2f)",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test
