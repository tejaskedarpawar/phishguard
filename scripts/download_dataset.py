"""
scripts/download_dataset.py
----------------------------
Attempts to download and convert the UCI Phishing Websites Dataset.

Usage:
    python scripts/download_dataset.py

Requires network access.  If the download fails, follow the manual
instructions in data/README.md.
"""

from __future__ import annotations

import io
import logging
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "phishing.csv"


def try_download_from_uci() -> bool:
    """Try to fetch the ARFF zip directly from UCI."""
    import urllib.request

    UCI_URL = (
        "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
    )
    zip_path = DATA_DIR / "phishing_websites.zip"

    logger.info("Downloading from UCI: %s", UCI_URL)
    try:
        urllib.request.urlretrieve(UCI_URL, zip_path)
    except Exception as exc:
        logger.warning("UCI download failed: %s", exc)
        return False

    # Extract ARFF
    try:
        with zipfile.ZipFile(zip_path) as zf:
            arff_name = next(
                (n for n in zf.namelist() if n.endswith(".arff")), None
            )
            if not arff_name:
                logger.warning("No .arff file found in zip.")
                return False
            arff_bytes = zf.read(arff_name)
    except Exception as exc:
        logger.warning("Zip extraction failed: %s", exc)
        return False
    finally:
        zip_path.unlink(missing_ok=True)

    # Parse ARFF to CSV
    try:
        _arff_to_csv(arff_bytes.decode("utf-8", errors="replace"))
        return True
    except Exception as exc:
        logger.warning("ARFF conversion failed: %s", exc)
        return False


def _arff_to_csv(arff_text: str) -> None:
    """Convert ARFF text to CSV and write to OUTPUT_CSV."""
    import pandas as pd

    lines = arff_text.splitlines()
    attr_names = []
    data_start = False
    data_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        upper = stripped.upper()
        if upper.startswith("@ATTRIBUTE"):
            parts = stripped.split()
            name = parts[1].strip("'\"")
            attr_names.append(name)
        elif upper.startswith("@DATA"):
            data_start = True
        elif data_start:
            data_lines.append(stripped)

    if not attr_names or not data_lines:
        raise ValueError("Could not parse ARFF structure.")

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=attr_names)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved CSV: %s  (%d rows × %d cols)", OUTPUT_CSV, len(df), df.shape[1])


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists():
        logger.info("Dataset already exists at '%s'.  Delete it to re-download.", OUTPUT_CSV)
        return

    success = try_download_from_uci()

    if success:
        logger.info("Dataset ready at '%s'.", OUTPUT_CSV)
    else:
        logger.error(
            "\nAutomatic download failed.\n"
            "Please download the dataset manually:\n"
            "  • Kaggle: https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector\n"
            "  • UCI:    https://archive.ics.uci.edu/dataset/327/phishing+websites\n"
            "Place the CSV file at:  %s",
            OUTPUT_CSV,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
