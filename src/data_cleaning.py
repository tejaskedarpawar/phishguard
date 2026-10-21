"""
src/data_cleaning.py
--------------------
Loads, validates, and cleans the UCI Phishing Websites dataset.

Dataset schema
--------------
  30 integer feature columns (values in {-1, 0, 1})
  Target column: 'Result'  →  -1 = Phishing, 1 = Legitimate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",  # NB: source dataset has this typo — match it exactly
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]

TARGET_COLUMN = "Result"

# Label conventions in the raw dataset
LABEL_PHISHING = -1
LABEL_LEGITIMATE = 1

# Valid values each feature may take
VALID_VALUES = {-1, 0, 1, 2}  # dataset uses these integer codes

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load CSV dataset and perform a basic schema check.

    Parameters
    ----------
    data_path:
        Absolute path to ``phishing.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as-loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the CSV is not present at *data_path*.
    ValueError
        If required columns are missing from the file.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.  "
            "See data/README.md for download instructions."
        )

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset: %d rows × %d columns", len(df), df.shape[1])

    # Strip accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Handle an unnamed index column sometimes present in Kaggle mirrors
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        logger.info("Dropped unnamed index column.")

    _validate_schema(df)
    return df


def clean_dataset(
    df: pd.DataFrame,
    results_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Clean the raw dataframe and return features, target, and a summary.

    Steps
    -----
    1. Report shape and column names.
    2. Check missing values.
    3. Identify unexpected integer values.
    4. Remove duplicate rows (documented).
    5. Examine class balance.
    6. Separate features from target.

    Parameters
    ----------
    df:
        Raw dataframe returned by :func:`load_dataset`.
    results_dir:
        If provided, writes ``cleaning_summary.json`` here.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (30 columns).
    y : pd.Series
        Target labels (−1 / 1).
    summary : dict
        Human-readable cleaning report.
    """
    summary: dict = {}

    # --- 1. Shape & columns ---------------------------------------------------
    summary["original_shape"] = list(df.shape)
    summary["columns"] = list(df.columns)
    logger.info("Shape: %s", df.shape)

    # --- 2. Missing values ----------------------------------------------------
    missing = df.isnull().sum()
    summary["missing_values"] = missing[missing > 0].to_dict()
    if missing.sum() == 0:
        logger.info("No missing values found.")
    else:
        logger.warning("Missing values detected:\n%s", missing[missing > 0])

    # --- 3. Unexpected values -------------------------------------------------
    unexpected: dict = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        unique_vals = set(df[col].unique())
        extra = unique_vals - VALID_VALUES
        if extra:
            unexpected[col] = sorted(extra)

    summary["unexpected_values"] = unexpected
    if unexpected:
        logger.warning("Unexpected feature values: %s", unexpected)
    else:
        logger.info("All feature values are within expected range {-1, 0, 1}.")

    # --- 4. Data types --------------------------------------------------------
    type_issues = {
        col: str(df[col].dtype)
        for col in FEATURE_COLUMNS
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col])
    }
    summary["dtype_issues"] = type_issues

    # Cast to int if possible
    for col in type_issues:
        try:
            df[col] = df[col].astype(int)
        except (ValueError, TypeError):
            logger.warning("Could not cast column '%s' to int.", col)

    # --- 5. Duplicates --------------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    n_dropped = n_before - n_after
    summary["duplicates_removed"] = n_dropped
    if n_dropped > 0:
        logger.info("Removed %d duplicate rows.", n_dropped)
    else:
        logger.info("No duplicate rows found.")

    summary["cleaned_shape"] = list(df.shape)

    # --- 6. Class balance -----------------------------------------------------
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    summary["class_distribution"] = {
        "Phishing (−1)": int(target_counts.get(LABEL_PHISHING, 0)),
        "Legitimate (1)": int(target_counts.get(LABEL_LEGITIMATE, 0)),
    }
    total = len(df)
    summary["class_balance_pct"] = {
        "Phishing (−1)": round(target_counts.get(LABEL_PHISHING, 0) / total * 100, 2),
        "Legitimate (1)": round(target_counts.get(LABEL_LEGITIMATE, 0) / total * 100, 2),
    }
    logger.info("Class distribution: %s", summary["class_distribution"])

    # --- 7. Separate features from target ------------------------------------
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()

    summary["feature_count"] = len(available_features)
    summary["feature_names"] = available_features

    # --- 8. Persist summary ---------------------------------------------------
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "cleaning_summary.json"
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Cleaning summary written to '%s'.", out_path)

    return X, y, summary


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X, y:
        Feature matrix and target vector.
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split → train: %d rows, test: %d rows (stratified).",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if critical columns are missing."""
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        # Try case-insensitive match for common Kaggle variations
        lower_map = {c.lower(): c for c in df.columns}
        still_missing = []
        rename_map: dict[str, str] = {}
        for col in missing_cols:
            if col.lower() in lower_map:
                rename_map[lower_map[col.lower()]] = col
            else:
                still_missing.append(col)
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.info("Auto-renamed columns: %s", rename_map)
        if still_missing:
            raise ValueError(
                f"Required columns not found in dataset: {still_missing}\n"
                "Ensure you are using the correct UCI Phishing Websites CSV."
            )
