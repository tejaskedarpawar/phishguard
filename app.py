"""
app.py
------
PhishGuard — Presentation-Ready Streamlit Web Application.

Run with:
    streamlit run app.py

Features:
- Dashboard: Real dataset statistics, best model highlight, compact comparison chart.
- URL Analyzer & Record Inspector:
    1. URL Lexical Heuristic Analyzer (extracts 10 address-bar features with honest disclosure).
    2. Verified Test Benchmark (tests held-out samples with known ground truth for viva demo).
- Model Performance: Full comparison table (Acc, Prec, Rec, F1, ROC-AUC), confusion matrices,
  ROC curves, feature importances, and trade-off analysis.
- Dataset Insights: Schema details, class distribution, data quality, deduplication summary.
- About Project: Academic problem statement, methodology, algorithm details, limitations.
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
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
DEMO_SAMPLES_PATH = DATA_DIR / "demo_samples.json"

# ---------------------------------------------------------------------------
# Streamlit Page Config (Must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PhishGuard — Phishing Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Clean, Professional, Minimalist Cybersecurity Palette
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1a1a2e;
}

.stApp {
    background-color: #f8f9fb;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

.sidebar-brand {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.sidebar-badge {
    background-color: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    display: inline-block;
    margin-bottom: 1rem;
}

/* Metric Cards */
.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.metric-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.35rem;
}

.metric-number {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.metric-caption {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Result Banners */
.result-card-phishing {
    background-color: #fef2f2;
    border: 1.5px solid #f87171;
    border-left: 5px solid #dc2626;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin: 1rem 0;
}

.result-card-legitimate {
    background-color: #f0fdf4;
    border: 1.5px solid #86efac;
    border-left: 5px solid #16a34a;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    margin: 1rem 0;
}

.result-card-title {
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.result-card-phishing .result-card-title { color: #991b1b; }
.result-card-legitimate .result-card-title { color: #166534; }

/* Notice & Warning Callouts */
.callout-info {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.86rem;
    color: #1e293b;
    margin: 0.8rem 0;
}

.callout-warning {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #d97706;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.86rem;
    color: #92400e;
    margin: 0.8rem 0;
}

/* Section Header Bar */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 0.35rem;
    margin: 1.5rem 0 1rem 0;
    display: inline-block;
}

/* Button overrides */
.stButton > button {
    background-color: #2563eb;
    color: #ffffff;
    font-weight: 600;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1.25rem;
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: #ffffff;
}

/* Footer & Disclaimers */
.app-disclaimer {
    font-size: 0.78rem;
    color: #64748b;
    font-style: italic;
    margin-top: 1.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data and Model Loaders (Cached for high performance)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_trained_models() -> Dict[str, Any]:
    """Load all four fitted scikit-learn model pipelines."""
    from src.prediction import load_all_models
    return load_all_models(MODELS_DIR)


@st.cache_data
def get_comparison_csv() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def get_json_artifact(filename: str) -> Optional[dict]:
    path = RESULTS_DIR / filename
    if path.exists():
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


@st.cache_data
def get_demo_samples() -> Dict[str, Any]:
    from src.prediction import load_demo_samples
    return load_demo_samples(DEMO_SAMPLES_PATH)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

def render_sidebar(models: Dict[str, Any]) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">🛡️ PhishGuard</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-badge">ML Cybersecurity Lab · Evaluation</div>',
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Navigation",
            options=[
                "Dashboard",
                "URL Analyzer & Inspector",
                "Model Performance",
                "Dataset Insights",
                "About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            f"**Pipeline Status:** {'🟢 4 Models Ready' if len(models) == 4 else '🟡 Incomplete'}"
        )
        st.markdown(
            '<div class="app-disclaimer">'
            'Academic Machine Learning Project.<br>'
            'Heuristic and pipeline assessment only.<br>'
            'Not an absolute safety guarantee.'
            '</div>',
            unsafe_allow_html=True,
        )

    return selected_page


# ---------------------------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard(models: Dict[str, Any]) -> None:
    st.title("PhishGuard Dashboard")
    st.markdown(
        "**ML-Based Phishing Website Detection System** — "
        "A rigorous, reproducible comparative study of four machine learning algorithms "
        "trained on the UCI Phishing Websites dataset."
    )

    if not models:
        st.error(
            "⚠️ No trained model artifacts detected in `models/`.  \n"
            "Please run `python train.py` in your terminal to train and evaluate all models."
        )
        return

    # Metric summary row
    dataset_info = get_json_artifact("dataset_info.json")
    best_info = get_json_artifact("best_model.json")
    comparison_df = get_comparison_csv()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_instances = dataset_info.get("cleaned_shape", [0])[0] if dataset_info else 1716
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Cleaned Instances</div>'
            f'<div class="metric-number">{n_instances:,}</div>'
            f'<div class="metric-caption">After removing 740 duplicates</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Evaluated Classifiers</div>'
            f'<div class="metric-number">{len(models)}</div>'
            f'<div class="metric-caption">DT, KNN, RF, SVM</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        best_name = best_info.get("best_model", "Random Forest") if best_info else "Random Forest"
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Selected Model</div>'
            f'<div class="metric-number" style="font-size:1.25rem;padding-top:0.2rem;">{best_name}</div>'
            f'<div class="metric-caption">Criterion: Phishing F1-Score</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col4:
        best_f1 = f"{best_info['f1']:.4f}" if best_info else "0.9479"
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Highest Test F1</div>'
            f'<div class="metric-number">{best_f1}</div>'
            f'<div class="metric-caption">Phishing class (-1)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Quick interactive test form
    st.markdown('<div class="section-heading">Quick URL Analysis</div>', unsafe_allow_html=True)
    with st.form("quick_url_form"):
        col_input, col_model = st.columns([3, 1])
        with col_input:
            quick_url = st.text_input(
                "Enter URL for rapid heuristic check",
                value="https://secure-login-bank-verification.com/login",
                placeholder="https://example.com",
                label_visibility="collapsed",
            )
        with col_model:
            selected_model_name = st.selectbox("Classifier", options=list(models.keys()), index=2)
        submit_quick = st.form_submit_button("Analyze Heuristics")

    if submit_quick and quick_url.strip():
        from src.prediction import predict_url
        res = predict_url(quick_url.strip(), models[selected_model_name], selected_model_name)
        _render_prediction_result(res)

    # Model comparison chart & table
    if comparison_df is not None:
        st.markdown('<div class="section-heading">Model Performance Comparison</div>', unsafe_allow_html=True)
        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            _render_f1_bar_chart(comparison_df)
        with col_table:
            st.dataframe(
                comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]],
                use_container_width=True,
                hide_index=True,
            )

    # Faculty Demo Quick Guide
    st.markdown('<div class="section-heading">Laboratory Demonstration Outline</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Step 1:** Review dataset hygiene on **Dataset Insights** (2,456 raw rows cleaned to 1,716 non-duplicate rows).
    - **Step 2:** Inspect test metrics and confusion matrices on **Model Performance**.
    - **Step 3:** Use **URL Analyzer & Inspector** to test verified legitimate and phishing benchmark samples from the held-out test set.
    - **Step 4:** Test live URL heuristics and observe the 10 address-bar feature extractions.
    """)


# ---------------------------------------------------------------------------
# Page 2: URL Analyzer & Inspector
# ---------------------------------------------------------------------------

def page_url_analyzer(models: Dict[str, Any]) -> None:
    st.title("URL Analyzer & Feature Inspector")
    st.markdown(
        "Evaluate website addresses using machine learning classifiers. "
        "Choose between **URL Heuristic Extraction** (safe lexical analysis of input URLs) "
        "and **Verified Test Set Benchmarking** (testing exact 30-feature held-out records)."
    )

    if not models:
        st.error("No models loaded. Please run `python train.py` first.")
        return

    tab1, tab2 = st.tabs(["🌐 Live URL Heuristic Analysis", "🧪 Verified Test Record Benchmarks"])

    # ---------------- TAB 1: URL Heuristic Analysis ----------------
    with tab1:
        st.markdown(
            '<div class="callout-info">'
            '<strong>Architectural Honesty Note:</strong> The UCI dataset models were trained on 30 features, '
            'of which 10 are lexical address-bar attributes extractable from the URL string itself. '
            'The remaining 20 features require fetching page content, executing JavaScript, and querying external DNS/WHOIS databases. '
            'To guarantee safe, zero-risk, offline-compatible operation, live web requests are disabled and non-lexical features '
            'are held at baseline. This provides an authentic lexical heuristic estimate.'
            '</div>',
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([3, 1])
        with col_left:
            url_input = st.text_input(
                "Website URL to analyze:",
                value="",
                placeholder="e.g. https://paypal-account-security-update.com/signin",
                help="Enter any full URL or bare domain name.",
            )
        with col_right:
            model_name = st.selectbox("Select Model:", options=list(models.keys()), index=2, key="tab1_model")

        col_presets, _ = st.columns([3, 1])
        with col_presets:
            st.caption("Quick Test Presets:")
            p_cols = st.columns(4)
            if p_cols[0].button("Google (Legitimate)"):
                url_input = "https://www.google.com"
            if p_cols[1].button("Wikipedia (Legitimate)"):
                url_input = "https://www.wikipedia.org"
            if p_cols[2].button("IP Address URL (Suspicious)"):
                url_input = "http://192.168.1.100/paypal-update/login.php"
            if p_cols[3].button("Brand Spoofing (Suspicious)"):
                url_input = "https://secure-login-account-verification-alert.com/signin"

        analyze_button = st.button("Run URL Analysis", key="btn_run_url")

        if (analyze_button or url_input) and url_input.strip():
            from src.prediction import predict_url
            res = predict_url(url_input.strip(), models[model_name], model_name)
            _render_prediction_result(res)

            if res["status"] == "success":
                st.markdown('<div class="section-heading">Extracted Address-Bar Lexical Features</div>', unsafe_allow_html=True)
                f_items = list(res["feature_details"].items())
                fc1, fc2 = st.columns(2)
                half = len(f_items) // 2
                with fc1:
                    for k, v in f_items[:half]:
                        st.markdown(f"- **{k}:** `{v}`")
                with fc2:
                    for k, v in f_items[half:]:
                        st.markdown(f"- **{k}:** `{v}`")

    # ---------------- TAB 2: Verified Test Record Benchmarks ----------------
    with tab2:
        st.markdown(
            '<div class="callout-info">'
            '<strong>Faculty Lab Demonstration Mode:</strong> '
            'This tab tests complete, verified 30-feature records from the held-out test set (unseen during training). '
            'Because all 30 features are present, this produces the mathematically exact classification '
            'reported in the experimental results.'
            '</div>',
            unsafe_allow_html=True,
        )

        demo_fixtures = get_demo_samples()
        legit_samples = demo_fixtures.get("legitimate_samples", [])
        phish_samples = demo_fixtures.get("phishing_samples", [])

        if not legit_samples and not phish_samples:
            st.warning("Demo fixtures file not found. Re-run `python train.py` to generate `data/demo_samples.json`.")
        else:
            sample_options = {}
            for s in legit_samples:
                sample_options[f"🟢 {s['description']}"] = s
            for s in phish_samples:
                sample_options[f"🔴 {s['description']}"] = s

            col_sel, col_m = st.columns([3, 1])
            with col_sel:
                selected_sample_key = st.selectbox(
                    "Choose a verified held-out test instance:",
                    options=list(sample_options.keys()),
                )
            with col_m:
                eval_model_name = st.selectbox("Classifier to evaluate:", options=list(models.keys()), key="tab2_model")

            chosen_sample = sample_options[selected_sample_key]

            if st.button("Evaluate Test Sample Record", key="btn_eval_sample"):
                from src.prediction import predict_feature_record
                pred_result = predict_feature_record(
                    chosen_sample["features"],
                    models[eval_model_name],
                    eval_model_name,
                )

                # Display match with ground truth
                gt_label = chosen_sample["ground_truth_label"]
                gt_code = chosen_sample["ground_truth"]
                pred_label = pred_result["label"]
                pred_code = pred_result["prediction_code"]
                match = (pred_code == gt_code)

                banner_cls = "result-card-legitimate" if not pred_result["is_phishing"] else "result-card-phishing"
                icon = "✓" if match else "⚠️"

                conf_text = f" · Confidence: {pred_result['confidence']:.1%}" if pred_result["confidence"] else ""
                st.markdown(
                    f'<div class="{banner_cls}">'
                    f'<div class="result-card-title">{icon} {pred_label}{conf_text}</div>'
                    f'<div style="font-size:0.9rem;margin-top:0.35rem;">'
                    f'<strong>Ground Truth:</strong> {gt_label} (Label = {gt_code})<br>'
                    f'<strong>Model Decision:</strong> {pred_label} (Prediction = {pred_code})<br>'
                    f'<strong>Outcome:</strong> {"Correct Classification (True Positive / True Negative)" if match else "Classification Error"}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="section-heading">30-Feature Vector Values for Sample</div>', unsafe_allow_html=True)
                sample_df = pd.DataFrame([chosen_sample["features"]])
                st.dataframe(sample_df, use_container_width=True, hide_index=True)


def _render_prediction_result(res: Dict[str, Any]) -> None:
    """Render prediction result card."""
    if res.get("status") == "error":
        st.error(f"Input Error: {res.get('error')}")
        return

    is_phishing = res["is_phishing"]
    label = res["label"]
    card_cls = "result-card-phishing" if is_phishing else "result-card-legitimate"
    icon = "⚠️" if is_phishing else "✓"
    conf_str = f" · Confidence: {res['confidence']:.1%}" if res.get("confidence") else ""

    st.markdown(
        f'<div class="{card_cls}">'
        f'<div class="result-card-title">{icon} {label}{conf_str}</div>'
        f'<div style="font-size:0.88rem;margin-top:0.4rem;">'
        f'<strong>Evaluated URL / Target:</strong> <code>{res.get("url", "Record")}</code><br>'
        f'<strong>Evaluated By:</strong> {res["model_name"]}<br>'
        f'<span style="font-size:0.8rem;color:#475569;">{res["disclaimer"]}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 3: Model Performance
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    st.title("Model Evaluation and Comparison")
    st.markdown(
        "All performance metrics were generated strictly on the independent held-out test set "
        "(20% stratified split). **Phishing (-1)** is explicitly treated as the positive class of interest."
    )

    comparison_df = get_comparison_csv()
    best_info = get_json_artifact("best_model.json")
    exp_summary = get_json_artifact("experiment_summary.json")

    if comparison_df is None:
        st.error("No experimental results found. Please execute `python train.py` first.")
        return

    # Best model banner
    if best_info:
        st.markdown(
            f'<div class="result-card-legitimate">'
            f'<div class="result-card-title">✓ Best Performing Classifier: {best_info["best_model"]}</div>'
            f'<div style="font-size:0.88rem;margin-top:0.35rem;">'
            f'<strong>Selection Criterion:</strong> Highest F1-Score on Phishing Class (-1) with ROC-AUC as tiebreaker.<br>'
            f'<strong>Test Results:</strong> F1-Score = <b>{best_info["f1"]:.4f}</b> · '
            f'Recall = <b>{best_info["recall"]:.4f}</b> · '
            f'Accuracy = <b>{best_info["accuracy"]:.4f}</b> · '
            f'ROC-AUC = <b>{best_info["roc_auc"]:.4f}</b>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Metrics comparison table
    st.markdown('<div class="section-heading">Independent Test Set Metrics</div>', unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Cross-validation table if available
    if exp_summary and "models" in exp_summary:
        cv_rows = []
        for m_name, m_data in exp_summary["models"].items():
            cv_info = m_data.get("cross_validation_train")
            if cv_info:
                cv_rows.append({
                    "Model": m_name,
                    "5-Fold CV F1 (Mean ± Std)": f"{cv_info['cv_f1_mean']:.4f} ± {cv_info['cv_f1_std']:.4f}",
                    "5-Fold CV Accuracy": f"{cv_info['cv_accuracy_mean']:.4f} ± {cv_info['cv_accuracy_std']:.4f}",
                    "5-Fold CV Recall": f"{cv_info['cv_recall_mean']:.4f}",
                    "5-Fold CV Precision": f"{cv_info['cv_precision_mean']:.4f}",
                    "5-Fold CV ROC-AUC": f"{cv_info['cv_roc_auc_mean']:.4f}",
                })
        if cv_rows:
            st.markdown('<div class="section-heading">5-Fold Cross-Validation on Training Data (X_train)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-heading">Confusion Matrices (Held-out Test Set)</div>', unsafe_allow_html=True)
        cm_path = RESULTS_DIR / "confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), use_column_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with col2:
        st.markdown('<div class="section-heading">ROC Curves (Discriminative Ability)</div>', unsafe_allow_html=True)
        roc_path = RESULTS_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), use_column_width=True)
        else:
            st.info("ROC curves image not found.")

    # Feature Importance Chart
    st.markdown('<div class="section-heading">Tree-Based Feature Importances (Information Gain / Gini Reduction)</div>', unsafe_allow_html=True)
    fi_path = RESULTS_DIR / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), use_column_width=True)

    # Metric trade-off discussion
    st.markdown('<div class="section-heading">Academic Discussion: Precision vs. Recall Trade-Off</div>', unsafe_allow_html=True)
    st.markdown("""
    In cybersecurity phishing detection:
    - **False Negative (FN):** A phishing website is classified as legitimate. The user visits the fraudulent site and suffers credential or financial theft. This is the **highest risk outcome**.
    - **False Positive (FP):** A legitimate website is flagged as phishing. The user experiences minor inconvenience.
    - **Why F1-Score & Recall Matter:** Models cannot be selected purely on Accuracy. In our test set, **Random Forest achieved a Recall of 0.9611**, successfully intercepting 173 out of 180 phishing instances, yielding the highest F1-score (**0.9479**).
    """)


# ---------------------------------------------------------------------------
# Page 4: Dataset Insights
# ---------------------------------------------------------------------------

def page_dataset_insights() -> None:
    st.title("Dataset Insights and Data Hygiene")
    st.markdown(
        "Inspection of the UCI Phishing Websites Dataset schema, duplicate cleansing, "
        "and distribution statistics."
    )

    dataset_info = get_json_artifact("dataset_info.json")
    cleaning_summary = get_json_artifact("cleaning_summary.json")

    if not dataset_info:
        st.error("Dataset info not found. Please run `python train.py` first.")
        return

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Raw Instances</div>'
            f'<div class="metric-number">{dataset_info.get("original_shape", [0])[0]:,}</div>'
            f'<div class="metric-caption">UCI direct archive</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Duplicates Dropped</div>'
            f'<div class="metric-number">{dataset_info.get("duplicates_removed", 0):,}</div>'
            f'<div class="metric-caption">Prevents train/test leakage</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Cleaned Instances</div>'
            f'<div class="metric-number">{dataset_info.get("cleaned_shape", [0])[0]:,}</div>'
            f'<div class="metric-caption">Unique feature vectors</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-title">Feature Dimensions</div>'
            f'<div class="metric-number">{dataset_info.get("feature_count", 30)}</div>'
            f'<div class="metric-caption">All integer-coded {-1, 0, 1}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Class balance and splitting
    st.markdown('<div class="section-heading">Target Class Balance (Deduplicated Dataset)</div>', unsafe_allow_html=True)
    dist = dataset_info.get("class_distribution", {})
    pct = dataset_info.get("class_balance_pct", {})

    col_chart, col_details = st.columns([1, 1])
    with col_chart:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        labels = ["Phishing (-1)", "Legitimate (1)"]
        counts = [dist.get("Phishing (−1)", 900), dist.get("Legitimate (1)", 816)]
        bars = ax.bar(labels, counts, color=["#dc2626", "#16a34a"], width=0.45, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:,}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.set_ylabel("Instance Count", fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_details:
        st.markdown(f"""
        | Attribute | Value |
        |---|---|
        | **Phishing Class (-1)** | {dist.get('Phishing (−1)', 900):,} instances ({pct.get('Phishing (−1)', 52.45):.2f}%) |
        | **Legitimate Class (1)** | {dist.get('Legitimate (1)', 816):,} instances ({pct.get('Legitimate (1)', 47.55):.2f}%) |
        | **Class Balance** | Naturally balanced (≈ 52% to 48%) |
        | **Missing Values** | 0 nulls detected across all columns |
        | **Training Split (80%)** | {dataset_info.get('train_size', 1372):,} instances |
        | **Test Split (20%)** | {dataset_info.get('test_size', 344):,} instances |
        """)

    # 30 Features Documentation
    st.markdown('<div class="section-heading">Catalog of 30 Website Features</div>', unsafe_allow_html=True)
    feature_names = dataset_info.get("feature_names", [])
    if feature_names:
        half = len(feature_names) // 2
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("##### Address Bar & Lexical Features (1–15)")
            for i, f in enumerate(feature_names[:half], 1):
                st.markdown(f"{i}. `{f}`")
        with fc2:
            st.markdown("##### HTML, JavaScript & Domain Features (16–30)")
            for i, f in enumerate(feature_names[half:], half + 1):
                st.markdown(f"{i}. `{f}`")


# ---------------------------------------------------------------------------
# Page 5: About Project
# ---------------------------------------------------------------------------

def page_about() -> None:
    st.title("About PhishGuard")
    st.markdown("""
    ### Project Overview
    **PhishGuard** is an academically defensible machine learning laboratory project evaluating
    phishing website detection. It demonstrates the complete end-to-end data science lifecycle:
    data hygiene, duplicate resolution, stratified splitting, cross-validation, metric analysis,
    and interactive deployment.

    ### Machine Learning Algorithms Implemented
    1. **Decision Tree Classifier:**
       - **Splitting Criterion:** `entropy` (Information Gain principle, analogous to ID3 / C4.5).
       - **Architecture:** Scikit-learn's optimized CART tree regularized with `max_depth=10`, `min_samples_split=10`.
       - **Feature Importance:** Direct Gini/Entropy reduction per feature.
    2. **K-Nearest Neighbors (KNN):**
       - **Distance Metric:** Euclidean distance with `weights='distance'` and `k=7`.
       - **Pipeline Scaling:** `StandardScaler` fitted strictly on training data inside the pipeline to eliminate test leakage.
    3. **Random Forest Classifier:**
       - **Ensemble Architecture:** 150 bootstrapped decision trees with `max_depth=15`.
       - **Regularization:** `min_samples_leaf=4` to avoid individual tree overfitting.
    4. **Support Vector Machine (SVM):**
       - **Kernel:** Radial Basis Function (RBF) with $C=1.0$ and $\\gamma=\\text{'scale'}$.
       - **Pipeline Scaling:** `StandardScaler` inside pipeline.
       - **Probability Calibration:** `CalibratedClassifierCV` for reliable ROC-AUC probability estimation.

    ### Dataset Attribution
    - **Dataset:** Phishing Websites Dataset
    - **Repository:** UCI Machine Learning Repository
    - **Authors:** Rami Mohammad, Fadi Thabtah, Lee McCluskey (2015)
    - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

    ### Limitations and Ethical Cybersecurity Practices
    - **Lexical vs. Full-Page Detection:** The live URL analyzer derives 10 address-bar lexical features. Full 30-feature assessment requires active web scraping, DNS queries, and external traffic API lookups, which are safely bypassed in this offline tool.
    - **Static Dataset:** Phishing threats evolve constantly; offline historical models must be periodically retrained on newly emerging phishing campaigns.
    - **Safety Guarantee:** Machine learning provides probabilistic risk scoring, not an absolute guarantee.
    """)


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def _render_f1_bar_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = df["Model"].tolist()
    f1s = df["F1-Score"].tolist()
    colors = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6"]

    bars = ax.bar(names, f1s, color=colors[:len(names)], width=0.45, edgecolor="white")
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-Score (Phishing)", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=15, ha="right", fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Routing
# ---------------------------------------------------------------------------

def main() -> None:
    models = get_trained_models()
    selected_page = render_sidebar(models)

    if selected_page == "Dashboard":
        page_dashboard(models)
    elif selected_page == "URL Analyzer & Inspector":
        page_url_analyzer(models)
    elif selected_page == "Model Performance":
        page_model_performance()
    elif selected_page == "Dataset Insights":
        page_dataset_insights()
    elif selected_page == "About Project":
        page_about()


if __name__ == "__main__":
    main()
