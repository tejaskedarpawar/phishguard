"""
src/data_loader.py
------------------
Loads, validates, and cleans raw URL datasets for PhishGuard.

Labels:
  0 = Legitimate (Negative Class)
  1 = Phishing (Positive Class)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Standard binary label definitions
LABEL_LEGITIMATE = 0
LABEL_PHISHING = 1

LABEL_NAMES: Dict[int, str] = {
    LABEL_LEGITIMATE: "Legitimate",
    LABEL_PHISHING: "Phishing",
}

EXPECTED_COLUMNS = ["url", "label"]


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load and validate the raw URL dataset.

    Parameters
    ----------
    data_path:
        Path to the CSV dataset (e.g., data/urls.csv).

    Returns
    -------
    pd.DataFrame
        Deduplicated, schema-validated dataframe.
    """
    path_obj = Path(data_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path_obj}")

    df = pd.read_csv(path_obj)
    df.columns = df.columns.str.strip().str.lower()

    # Validate required columns
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at '{data_path}' is missing required columns: {missing}. "
            f"Expected columns: {EXPECTED_COLUMNS}."
        )

    # Drop nulls in url or label
    initial_len = len(df)
    df = df.dropna(subset=["url", "label"]).copy()
    dropped_nulls = initial_len - len(df)
    if dropped_nulls > 0:
        logger.info("Dropped %d rows with null values.", dropped_nulls)

    # Cast label to integer
    df["label"] = df["label"].astype(int)

    # Validate label space
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({LABEL_LEGITIMATE, LABEL_PHISHING}):
        raise ValueError(
            f"Invalid target labels detected: {unique_labels}. "
            f"Expected binary labels {{0: Legitimate, 1: Phishing}}."
        )

    # Deduplicate on url string
    n_before = len(df)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    n_dups = n_before - len(df)
    if n_dups > 0:
        logger.info("Removed %d duplicate URL entries.", n_dups)

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate high-level metadata and class balance summary."""
    total = len(df)
    legit_count = int((df["label"] == LABEL_LEGITIMATE).sum())
    phish_count = int((df["label"] == LABEL_PHISHING).sum())

    return {
        "total_instances": total,
        "legitimate_instances": legit_count,
        "phishing_instances": phish_count,
        "legitimate_percentage": round((legit_count / max(total, 1)) * 100, 2),
        "phishing_percentage": round((phish_count / max(total, 1)) * 100, 2),
    }
