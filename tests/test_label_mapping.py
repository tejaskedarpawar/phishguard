"""
tests/test_label_mapping.py
---------------------------
Automated unit tests validating label definitions, binary encoding,
and confusion matrix alignment for phishing detection.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.data_loader import (
    LABEL_LEGITIMATE,
    LABEL_NAMES,
    LABEL_PHISHING,
    get_dataset_summary,
    load_dataset,
)


class TestLabelMapping:
    """Validate canonical binary class labels and mapping."""

    def test_canonical_class_constants(self) -> None:
        """Verify binary encoding constants: 0 = Legitimate, 1 = Phishing."""
        assert LABEL_LEGITIMATE == 0
        assert LABEL_PHISHING == 1

        assert LABEL_NAMES[LABEL_LEGITIMATE] == "Legitimate"
        assert LABEL_NAMES[LABEL_PHISHING] == "Phishing"

    def test_dataset_contains_only_valid_labels(self) -> None:
        """Verify that data/urls.csv only contains labels in {0, 1}."""
        df = load_dataset("data/urls.csv")
        unique_labels = set(df["label"].unique())
        assert unique_labels == {0, 1}

    def test_dataset_summary_labels(self) -> None:
        """Verify summary accurately reflects legitimate and phishing counts."""
        mock_df = pd.DataFrame({
            "url": ["https://legit1.com", "https://legit2.com", "http://phish1.com"],
            "label": [0, 0, 1],
        })
        summary = get_dataset_summary(mock_df)
        assert summary["total_instances"] == 3
        assert summary["legitimate_instances"] == 2
        assert summary["phishing_instances"] == 1
        assert summary["legitimate_percentage"] == 66.67
        assert summary["phishing_percentage"] == 33.33

    def test_positive_class_is_phishing(self) -> None:
        """In security evaluation, the positive class must be Phishing (1)."""
        # When precision, recall, and F1 are calculated, label=1 must represent the detection of malicious URLs
        positive_class = LABEL_PHISHING
        assert positive_class == 1
        assert LABEL_NAMES[positive_class] == "Phishing"
