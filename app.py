"""
app.py
------
PhishGuard — Streamlit web application.

Run with:
    streamlit run app.py

Loads pre-trained model artifacts from models/ and results/ directories.
Does NOT retrain models on startup — run train.py first.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Minimalist cybersecurity dashboard
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}

/* Background */
.stApp {
    background-color: #f8f9fb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e4ea;
    padding-top: 1rem;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 0.2rem;
}

/* Sidebar brand */
.brand-wordmark {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1e3a5f;
    letter-spacing: -0.5px;
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 2px solid #e0e4ea;
    margin-bottom: 1rem;
}

.brand-shield {
    color: #2563eb;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e0e4ea;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* Result banners */
.result-phishing {
    background: #fff5f5;
    border: 1.5px solid #fc8181;
    border-left: 4px solid #e53e3e;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
}

.result-legitimate {
    background: #f0fff4;
    border: 1.5px solid #68d391;
    border-left: 4px solid #38a169;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
}

.result-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.result-phishing .result-title { color: #c53030; }
.result-legitimate .result-title { color: #276749; }

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a2e;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 0.3rem;
    margin: 1.5rem 0 1rem 0;
    display: inline-block;
}

/* Warning / info boxes */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #1e40af;
    margin: 0.75rem 0;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #92400e;
    margin: 0.75rem 0;
}

/* Table */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Input */
.stTextInput input {
    border: 1.5px solid #d1d5db;
    border-radius: 6px;
    font-size: 0.9rem;
    padding: 0.5rem 0.75rem;
}
.stTextInput input:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
}

/* Disclaimer */
.disclaimer {
    font-size: 0.78rem;
    color: #9ca3af;
    font-style: italic;
    margin-top: 0.5rem;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

MODEL_FILENAMES = {
    "Decision Tree":          "decision_tree.joblib",
    "K-Nearest Neighbors":    "k-nearest_neighbors.joblib",
    "Random Forest":          "random_forest.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """Load all saved model pipelines (cached across reruns)."""
    loaded = {}
    for name, fname in MODEL_FILENAMES.items():
        path = MODELS_DIR / fname
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", fname, exc)
    return loaded


@st.cache_data
def load_comparison_csv() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_json(filename: str) -> Optional[dict]:
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return None


def models_trained() -> bool:
    return any((MODELS_DIR / f).exists() for f in MODEL_FILENAMES.values())


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand-wordmark">'
            '<span class="brand-shield">🛡</span> PhishGuard'
            '</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "URL Analyzer",
                "Model Performance",
                "Dataset Insights",
                "About Project",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<div class="disclaimer">ML-based assessment tool.<br>'
            'Not a security guarantee.</div>',
            unsafe_allow_html=True,
        )
    return page


# ---------------------------------------------------------------------------
# Page 1 — Dashboard
# ---------------------------------------------------------------------------

def page_dashboard(models: Dict[str, Any]) -> None:
    st.title("PhishGuard")
    st.markdown(
        "**ML-Based Phishing Website Detection** — "
        "Analyze URLs and compare four machine learning classifiers "
        "trained on the UCI Phishing Websites dataset."
    )

    if not models_trained():
        st.error(
            "⚠ No trained models found.  "
            "Run `python train.py` first, then refresh this page."
        )
        return

    # Quick URL analysis at the top
    st.markdown('<div class="section-header">Quick URL Analysis</div>', unsafe_allow_html=True)
    with st.form("dashboard_url_form"):
        url_input = st.text_input(
            "Enter a URL to analyze",
            placeholder="https://example.com",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze URL")

    if submitted and url_input.strip():
        _run_prediction(url_input.strip(), models, compact=True)

    # Stats row
    comparison_df = load_comparison_csv()
    dataset_info = load_json("dataset_info.json")
    best_info = load_json("best_model.json")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = dataset_info.get("cleaned_shape", [0])[0] if dataset_info else "—"
        _metric_card("Dataset Size", f"{total:,}" if isinstance(total, int) else total, "instances")

    with col2:
        _metric_card("ML Models", str(len(models)), "trained classifiers")

    with col3:
        best_name = best_info["best_model"] if best_info else "—"
        _metric_card("Best Model", best_name, "by F1-score")

    with col4:
        best_f1 = f"{best_info['f1']:.4f}" if best_info else "—"
        _metric_card("Best F1-Score", best_f1, "phishing class")

    # F1 chart
    if comparison_df is not None:
        st.markdown('<div class="section-header">Model F1-Score Comparison</div>', unsafe_allow_html=True)
        _render_f1_bar(comparison_df)

    # Performance table
    if comparison_df is not None:
        st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
        display_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_f1_bar(comparison_df: pd.DataFrame) -> None:
    colors = {
        "Decision Tree": "#2196F3",
        "K-Nearest Neighbors": "#FF9800",
        "Random Forest": "#4CAF50",
        "Support Vector Machine": "#9C27B0",
    }
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = comparison_df["Model"].tolist()
    f1s = comparison_df["F1-Score"].tolist()
    bar_colors = [colors.get(n, "#607D8B") for n in names]
    bars = ax.bar(names, f1s, color=bar_colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1-Score")
    ax.set_facecolor("#f8f9fb")
    fig.patch.set_facecolor("#f8f9fb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2 — URL Analyzer
# ---------------------------------------------------------------------------

def page_url_analyzer(models: Dict[str, Any]) -> None:
    st.title("URL Analyzer")
    st.markdown(
        "Enter a URL to extract its features and receive a phishing classification. "
        "Analysis is performed locally — no data is sent to external servers."
    )

    if not models:
        st.error(
            "No trained models available.  "
            "Run `python train.py` to train the models first."
        )
        return

    col_left, col_right = st.columns([3, 1])
    with col_left:
        url_input = st.text_input(
            "URL to analyze",
            placeholder="https://example.com or example.com",
            help="Enter a full URL or bare domain name.",
        )
    with col_right:
        model_choice = st.selectbox(
            "Model",
            options=list(models.keys()),
        )

    analyze_btn = st.button("Analyze URL", key="analyzer_btn")

    st.markdown(
        '<div class="info-box">'
        '<strong>Feature scope:</strong> This tool computes URL-derivable features directly '
        '(IP address check, URL length, shortener detection, etc.). '
        'Features requiring live network access (SSL validity, DNS records, web traffic) '
        'are approximated conservatively. Results reflect URL-level heuristics only.'
        '</div>',
        unsafe_allow_html=True,
    )

    if analyze_btn:
        _run_prediction(url_input.strip(), models, compact=False, model_choice=model_choice)

    st.markdown(
        '<div class="disclaimer">'
        'This is an experimental ML-based assessment tool for educational purposes. '
        'It does not guarantee website safety or maliciousness. '
        'Always use reputable security tools for production decisions.'
        '</div>',
        unsafe_allow_html=True,
    )


def _run_prediction(
    url: str,
    models: Dict[str, Any],
    compact: bool = False,
    model_choice: Optional[str] = None,
) -> None:
    """Core prediction logic shared by Dashboard and URL Analyzer."""
    from src.feature_extraction import extract_features, validate_url

    is_valid, error_msg = validate_url(url)
    if not is_valid:
        st.error(f"Invalid URL: {error_msg}")
        return

    if not models:
        st.warning("No models available. Run `python train.py` first.")
        return

    chosen = model_choice or list(models.keys())[0]
    pipeline = models.get(chosen)
    if pipeline is None:
        st.error(f"Model '{chosen}' is not loaded.")
        return

    feature_df, feature_details = extract_features(url)

    try:
        prediction = pipeline.predict(feature_df)[0]
        label = "Potentially Phishing" if prediction == -1 else "Likely Legitimate"
        css_class = "result-phishing" if prediction == -1 else "result-legitimate"
        icon = "⚠" if prediction == -1 else "✓"

        confidence_str = ""
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(feature_df)[0]
                classes = list(pipeline.classes_)
                pos_idx = classes.index(prediction)
                confidence = proba[pos_idx]
                confidence_str = f"  ·  Confidence: {confidence:.1%}"
            except Exception:
                pass

        st.markdown(
            f'<div class="{css_class}">'
            f'<div class="result-title">{icon} {label}{confidence_str}</div>'
            f'<div style="font-size:0.85rem;margin-top:0.35rem;">'
            f'<strong>URL:</strong> {url}<br>'
            f'<strong>Model:</strong> {chosen}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not compact:
            st.markdown(
                '<div class="section-header">Extracted URL Features</div>',
                unsafe_allow_html=True,
            )
            feat_col1, feat_col2 = st.columns(2)
            items = list(feature_details.items())
            half = len(items) // 2
            with feat_col1:
                for k, v in items[:half]:
                    st.markdown(f"**{k}:** {v}")
            with feat_col2:
                for k, v in items[half:]:
                    st.markdown(f"**{k}:** {v}")

            st.markdown(
                '<div class="warning-box">'
                '<strong>Interpretation:</strong> '
                + (
                    "Several URL characteristics associated with phishing were detected. "
                    "Exercise caution before visiting this site. This is an ML-based heuristic "
                    "— it may produce false positives."
                    if prediction == -1 else
                    "No strong URL-level phishing indicators were detected. "
                    "However, this assessment is based on URL features only and does not "
                    "guarantee the site is safe."
                )
                + '</div>',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        logger.exception("Prediction error")


# ---------------------------------------------------------------------------
# Page 3 — Model Performance
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    st.title("Model Performance")
    st.markdown(
        "All metrics are evaluated on the same held-out test set (20% of the data). "
        "**Positive class = −1 (Phishing)** for precision, recall, and F1-score."
    )

    comparison_df = load_comparison_csv()
    best_info = load_json("best_model.json")

    if comparison_df is None:
        st.error("No results found. Run `python train.py` first.")
        return

    # Best model callout
    if best_info:
        st.markdown(
            f'<div class="result-legitimate">'
            f'<div class="result-title">✓ Best Model: {best_info["best_model"]}</div>'
            f'<div style="font-size:0.85rem;margin-top:0.35rem;">'
            f'Selected by F1-score ({best_info["f1"]:.4f}) with ROC-AUC as tiebreaker. '
            f'F1 balances precision and recall — important in phishing detection where '
            f'missing a phishing site (false negative) is more costly than a false alarm.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Comparison table
    st.markdown('<div class="section-header">Metrics Comparison</div>', unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # F1 chart from saved PNG or re-rendered
    st.markdown('<div class="section-header">F1-Score Comparison</div>', unsafe_allow_html=True)
    f1_img = RESULTS_DIR / "f1_comparison.png"
    if f1_img.exists():
        st.image(str(f1_img), use_column_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_img = RESULTS_DIR / "confusion_matrices.png"
    if cm_img.exists():
        st.image(str(cm_img), use_column_width=True)
    else:
        st.info("Confusion matrix plot not found. Re-run `python train.py`.")

    # ROC curves
    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    roc_img = RESULTS_DIR / "roc_curves.png"
    if roc_img.exists():
        st.image(str(roc_img), use_column_width=True)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    fi_models = ["Decision_Tree", "Random_Forest"]
    shown = False
    for m in fi_models:
        fi_img = RESULTS_DIR / f"feature_importance_{m.lower()}.png"
        if fi_img.exists():
            st.markdown(f"**{m.replace('_', ' ')}**")
            st.image(str(fi_img), use_column_width=True)
            shown = True
    if not shown:
        st.info("Feature importance plots not found. Re-run `python train.py`.")

    # Explanation
    st.markdown('<div class="section-header">Trade-off Discussion</div>', unsafe_allow_html=True)
    st.markdown("""
In phishing detection, the cost of a **false negative** (missing a phishing site) is
generally higher than a **false positive** (flagging a legitimate site).

- **Recall** measures how well a model catches phishing sites (avoids false negatives).
- **Precision** measures how accurately a positive prediction is (avoids false alarms).
- **F1-score** is their harmonic mean and was used as the primary selection criterion.
- **ROC-AUC** measures overall discrimination ability regardless of threshold.

A model with high recall but lower precision will raise more false alarms but miss
fewer phishing sites — often the safer trade-off in security applications.
""")


# ---------------------------------------------------------------------------
# Page 4 — Dataset Insights
# ---------------------------------------------------------------------------

def page_dataset_insights() -> None:
    st.title("Dataset Insights")
    st.markdown(
        "Summary of the UCI Phishing Websites Dataset as loaded and cleaned for training."
    )

    dataset_info = load_json("dataset_info.json")
    cleaning = load_json("cleaning_summary.json")

    if dataset_info is None:
        st.error("Dataset info not found. Run `python train.py` first.")
        return

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        shape = dataset_info.get("cleaned_shape", [0, 0])
        _metric_card("Instances", f"{shape[0]:,}", "after deduplication")
    with col2:
        _metric_card("Features", str(dataset_info.get("feature_count", 30)), "input columns")
    with col3:
        dist = dataset_info.get("class_distribution", {})
        phishing_n = dist.get("Phishing (−1)", 0)
        _metric_card("Phishing", f"{phishing_n:,}", "class = −1")
    with col4:
        legit_n = dist.get("Legitimate (1)", 0)
        _metric_card("Legitimate", f"{legit_n:,}", "class = 1")

    # Class balance chart
    st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
    dist_pct = dataset_info.get("class_balance_pct", {})
    if dist_pct:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["Phishing (−1)", "Legitimate (1)"]
        values = [dist_pct.get("Phishing (−1)", 0), dist_pct.get("Legitimate (1)", 0)]
        bars = ax.bar(labels, values, color=["#e53e3e", "#38a169"], width=0.45, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # Missing values / duplicates
    st.markdown('<div class="section-header">Data Quality</div>', unsafe_allow_html=True)
    missing = dataset_info.get("missing_values", {})
    dups = dataset_info.get("duplicates_removed", 0)
    dtype_issues = dataset_info.get("dtype_issues", {})
    unexpected = dataset_info.get("unexpected_values", {})

    qcol1, qcol2 = st.columns(2)
    with qcol1:
        st.markdown(f"**Missing values:** {'None' if not missing else str(missing)}")
        st.markdown(f"**Duplicate rows removed:** {dups}")
    with qcol2:
        st.markdown(f"**Data type issues:** {'None' if not dtype_issues else str(dtype_issues)}")
        st.markdown(f"**Unexpected feature values:** {'None' if not unexpected else str(unexpected)}")

    # Feature list
    st.markdown('<div class="section-header">Feature Columns (30)</div>', unsafe_allow_html=True)
    feat_names = dataset_info.get("feature_names", [])
    if feat_names:
        half = len(feat_names) // 2
        fc1, fc2 = st.columns(2)
        with fc1:
            for f in feat_names[:half]:
                st.markdown(f"- `{f}`")
        with fc2:
            for f in feat_names[half:]:
                st.markdown(f"- `{f}`")

    # Preprocessing summary
    st.markdown('<div class="section-header">Preprocessing Summary</div>', unsafe_allow_html=True)
    orig_shape = dataset_info.get("original_shape", [0, 0])
    clean_shape = dataset_info.get("cleaned_shape", [0, 0])
    st.markdown(f"""
| Step | Detail |
|------|--------|
| Original shape | {orig_shape[0]:,} rows × {orig_shape[1]} columns |
| Unnamed index column | Removed if present |
| Missing values | {len(missing)} columns affected — {sum(missing.values()) if missing else 0} total |
| Duplicate rows removed | {dups} |
| Cleaned shape | {clean_shape[0]:,} rows × {clean_shape[1]} columns |
| Feature / target separation | Target `Result` excluded from X |
| Train / test split | 80% / 20% stratified |
| Scaling | Applied for KNN and SVM pipelines only (fit on train set) |
| Data leakage | Prevented — scaler fit on X_train only |
""")

    st.markdown('<div class="section-header">Dataset Attribution</div>', unsafe_allow_html=True)
    st.markdown("""
- **Name**: Phishing Websites Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Donated by**: Rami Mohammad, Fadi Thabtah, Lee McCluskey (2015)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Instances**: 11,055  |  **Features**: 30  |  **Classes**: 2
""")


# ---------------------------------------------------------------------------
# Page 5 — About Project
# ---------------------------------------------------------------------------

def page_about() -> None:
    st.title("About PhishGuard")

    st.markdown("""
### Problem Statement
Phishing websites impersonate legitimate organisations to steal credentials, financial data,
and personal information.  Traditional blacklist-based defences cannot keep up with the speed
at which new phishing domains are created.  Machine learning can identify phishing patterns
from URL and page structure features without relying on a known-bad list.

### Project Objectives
1. Build and compare four ML classifiers on the UCI Phishing Websites dataset.
2. Demonstrate a complete pipeline: data cleaning → training → evaluation → deployment.
3. Provide a minimalist, responsive web interface for live URL analysis.
4. Produce reproducible, documented results suitable for academic presentation.

### Dataset
| Attribute | Detail |
|-----------|--------|
| Name | Phishing Websites Dataset |
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/327/phishing+websites |
| Authors | Rami Mohammad, Fadi Thabtah, Lee McCluskey |
| Year | 2015 |
| License | CC BY 4.0 |
| Instances | 11,055 |
| Features | 30 (pre-extracted integer codes) |
| Target | `Result` (−1 = Phishing, 1 = Legitimate) |

### ML Algorithms Used
| Model | Notes |
|-------|-------|
| Decision Tree | Interpretable baseline; feature importance available |
| K-Nearest Neighbors | Instance-based learner; uses scaled features |
| Random Forest | Ensemble of trees; typically highest accuracy |
| Support Vector Machine (RBF) | Kernel-based; uses scaled features; probability output enabled |

### Technologies
- **Python 3** — Core language
- **pandas / NumPy** — Data processing
- **scikit-learn** — ML pipelines and evaluation
- **Matplotlib** — Visualisation
- **Streamlit** — Web interface
- **Joblib** — Model persistence

### Limitations
1. **URL-only inference** in the live tool — many dataset features require visiting the target
   website (DNS, SSL, page content, traffic rank).  The live analyzer approximates these features
   conservatively; results are URL-heuristic based only.
2. **Static dataset** — The training data is from 2015.  Phishing techniques evolve; the model
   may not recognise novel attack patterns.
3. **Binary classification** — The model does not distinguish phishing subtypes (credential
   harvesting, malware distribution, etc.).
4. **No live website inspection** — The tool never visits submitted URLs, which limits
   feature completeness but ensures user safety.
5. **Threshold fixed at default 0.5** — A security deployment might lower the decision
   threshold to increase recall at the cost of precision.

### Educational Prototype Notice
This project is developed for an academic Machine Learning laboratory evaluation.
It is an educational prototype and should not be used as the sole means of determining
website safety in any production or operational context.
""")

    st.markdown("---")
    st.markdown(
        "Built with Python · scikit-learn · Streamlit &nbsp;|&nbsp; "
        "Dataset © UCI ML Repository (CC BY 4.0)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    page = render_sidebar()
    models = load_all_models()

    if page == "Dashboard":
        page_dashboard(models)
    elif page == "URL Analyzer":
        page_url_analyzer(models)
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Dataset Insights":
        page_dataset_insights()
    elif page == "About Project":
        page_about()


if __name__ == "__main__":
    main()
