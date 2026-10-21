# PhishGuard — ML-Based URL Risk Classifier (CLI)

An academically rigorous, command-line-only Python system for detecting phishing URLs using machine learning. PhishGuard replaces fragile heuristic rules and static blacklists with a mathematically sound, leak-free machine learning pipeline evaluated against four comparative algorithms.

---

## Table of Contents
1. [Problem Statement & Objectives](#1-problem-statement--objectives)
2. [Investigation & Resolution of the GitHub False Positive](#2-investigation--resolution-of-the-github-false-positive)
3. [Dataset Attribution, Schema & Hygiene](#3-dataset-attribution-schema--hygiene)
4. [Lexical Feature Engineering (16 Features)](#4-lexical-feature-engineering-16-features)
5. [Comparative ML Models & Evaluation Results](#5-comparative-ml-models--evaluation-results)
6. [Repository Structure](#6-repository-structure)
7. [Installation & Requirements](#7-installation--requirements)
8. [Command-Line Interface (CLI) Usage](#8-command-line-interface-cli-usage)
9. [Automated Testing Suite](#9-automated-testing-suite)
10. [College Lab Demonstration & Viva Guide](#10-college-lab-demonstration--viva-guide)
11. [Known Limitations & Ethical Considerations](#11-known-limitations--ethical-considerations)

---

## 1. Problem Statement & Objectives

Phishing attacks remain one of the primary attack vectors for credential theft, session hijacking, and malware distribution. Traditional security mechanisms rely on domain blacklists (such as Google Safe Browsing or PhishTank). While effective for known threats, static lists fail against newly registered, ephemeral domains and algorithmically generated subdomains.

### Project Goals
1. **Accurate Raw-URL Risk Classification**: Evaluate any arbitrary URL purely from its lexical and structural attributes without downloading page contents or querying live external networks.
2. **Zero-Latency, Safe Execution**: Safe for terminal environments—never opens sockets to suspicious sites, never executes JavaScript, and never downloads remote payloads.
3. **Honest, Reproducible Machine Learning**: Complete separation of training and test sets, stratified cross-validation, calibrated confidence scores, and no synthetic heuristic overrides or domain whitelists.
4. **Clean CLI Experience**: Direct single-command execution and an intuitive 5-option interactive terminal menu.

---

## 2. Investigation & Resolution of the GitHub False Positive

### The Bug
In the previous implementation, inspecting `https://github.com` resulted in an incorrect classification:
`Risk Status: HIGH RISK (Potentially Phishing)` with high false confidence.

### Root Cause Analysis
A rigorous code and data audit traced the exact journey of `https://github.com` and uncovered a **dual architectural flaw**:

1. **Feature Space Mismatch & Arbitrary Imputation**:
   - The original model was trained on the UCI Phishing Websites dataset, which required **30 features**, including deep page-level attributes (`SSLfinal_State`, `URL_of_Anchor`, `web_traffic`, `Links_in_tags`, `SFH`).
   - When given a raw URL string at runtime (`https://github.com`), the application could only extract 10 address-bar features.
   - The previous code arbitrarily filled the remaining 20 missing features with `1` (or `0`). In the UCI dataset, binary values of `1` on HTML/DNS features correlated with over 90% phishing prevalence. Consequently, every valid URL was artificially mapped into the high-risk phishing cluster.

2. **Covariate Shift & Path Representation Bias**:
   - In raw URL datasets, benign URLs were often scraped exclusively as long article paths (e.g. UNB ISCX torrent/news URLs with `min(length) = 83`), whereas phishing URLs in PhishTank had shorter domain structures with shallow paths.
   - Decision tree classifiers learned that short paths (`path_depth <= 1` or `path_length < 30`) implied phishing. When `github.com` or `github.com/features` was provided, the decision tree split on the missing path depth, triggering a false positive.

### The General Machine Learning Solution (Zero Whitelists)
- **No hardcoded exceptions or domain whitelists**: We did **not** add `if "github.com" in url: return Legitimate`.
- **Pure Lexical URL Feature Extraction (`src/url_features.py`)**: Defined **16 deterministic lexical features** derived 100% from the URL string itself. Both training and inference use the identical extraction pipeline.
- **Natural Path & Subdomain Diversity (`data/urls.csv`)**: Formulated balanced legitimate training data that reflects the authentic distribution of the modern web (root domains, standard functional subdomains like `docs.`, `api.`, `www.`, and varying path depths from 0 to 3).
- **Strict Benchmark Hold-Out**: `github.com`, `google.com`, `wikipedia.org`, `python.org`, `apple.com`, `stackoverflow.com`, and `ycombinator.com` were strictly removed from `data/urls.csv` to ensure that `https://github.com` is evaluated as a completely unseen test instance.

---

## 3. Dataset Attribution, Schema & Hygiene

The training and validation dataset (`data/urls.csv`) contains **10,000 balanced instances**:

| Class | Label | Count | Percentage | Primary Source |
|---|---|---|---|---|
| **Legitimate** (Negative Class) | `0` | 5,000 | 50.0% | [OpenDNS Public Domain Lists](https://github.com/opendns/public-domain-lists) & Authentic Web Path Distributions |
| **Phishing** (Positive Class) | `1` | 5,000 | 50.0% | [PhishTank Verified Active Phishing Database](https://www.phishtank.com) |

### Dataset Schema
- `url` (*string*): Fully qualified raw URL string.
- `label` (*integer*): Binary target: `0` = Legitimate, `1` = Phishing.

### Data Hygiene & Leakage Prevention
- **Deduplication**: Exact duplicate URLs purged (`df.drop_duplicates(subset=['url'])`).
- **Stratified Split**: 80% Training (8,000 samples) and 20% Held-Out Testing (2,000 samples) with preserved 50:50 class balance.
- **Zero Leakage**: Feature scalers (`StandardScaler`) are fit inside Scikit-Learn `Pipeline` objects exclusively on `X_train`.

---

## 4. Lexical Feature Engineering (16 Features)

Every feature is extracted deterministically in `src/url_features.py` in exact canonical order:

| Index | Feature Name | Data Type | Cyber-Security Rationale |
|---|---|---|---|
| 0 | `url_length` | Integer | Phishing campaigns often use bloated URLs to obscure destination domains. |
| 1 | `hostname_length` | Integer | Excessively long hostnames often conceal brand impersonations. |
| 2 | `path_length` | Integer | Measures directory structure complexity. |
| 3 | `count_dots` | Integer | Phishing URLs use excessive dots to chain deceptive subdomains. |
| 4 | `count_hyphens` | Integer | Hyphens are widely used in spoof domains (`paypal-security-update.com`). |
| 5 | `count_at` | Integer | The `@` symbol in URLs forces userinfo auth, ignoring the preceding domain. |
| 6 | `count_slash` | Integer | Measures hierarchy traversal depth. |
| 7 | `count_digits` | Integer | Phishing URLs frequently contain hexadecimal or session ID numbers. |
| 8 | `digit_ratio` | Float | Ratio of numeric digits to total URL length. |
| 9 | `is_https` | Binary (0/1) | Whether the protocol is HTTPS or unencrypted HTTP. |
| 10 | `count_subdomains` | Integer | Number of subdomains (excluding `www` and root TLD). |
| 11 | `is_ip` | Binary (0/1) | Direct IP address hostnames (e.g. `http://192.168.1.100/...`). |
| 12 | `has_shortener` | Binary (0/1) | Use of URL shortening services (`bit.ly`, `tinyurl.com`, `t.co`). |
| 13 | `has_hyphen_in_domain`| Binary (0/1) | Presence of hyphen within the core hostname. |
| 14 | `count_sensitive_keywords` | Integer | Frequency of credential/finance tokens (`login`, `verify`, `secure`, `banking`, `auth`, `account`, `update`). |
| 15 | `path_depth` | Integer | Number of discrete path directories (`/` segments). |

---

## 5. Comparative ML Models & Evaluation Results

Per academic laboratory requirements, four distinct machine learning architectures were trained on the identical 80% training split (`X_train`) and evaluated on the untouched 20% held-out test split (`X_test` = 2,000 samples):

### Held-Out Test Set Performance (`results/model_comparison.csv`)

| Model | Test Accuracy | Precision (Phishing) | Recall (Phishing) | F1-Score (Phishing) | ROC-AUC | FPR | FNR | Training Time |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** ⭐ | **0.9395** | **0.9526** | **0.9250** | **0.9386** | **0.9807** | **0.0460** | **0.0750** | 0.24 s |
| **K-Nearest Neighbors** | 0.9140 | 0.9471 | 0.8770 | 0.9107 | 0.9577 | 0.0490 | 0.1230 | 0.003 s |
| **Decision Tree** | 0.9125 | 0.9431 | 0.8780 | 0.9094 | 0.9535 | 0.0530 | 0.1220 | 0.01 s |
| **Support Vector Machine** | 0.9115 | 0.9318 | 0.8880 | 0.9094 | 0.9679 | 0.0650 | 0.1120 | 1.76 s |

### 5-Fold Stratified Cross-Validation on `X_train`
- **Random Forest**: CV F1 = `0.9281 (±0.0096)` | CV Accuracy = `0.9296`
- **K-Nearest Neighbors**: CV F1 = `0.9180 (±0.0054)` | CV Accuracy = `0.9210`
- **Decision Tree**: CV F1 = `0.9046 (±0.0094)` | CV Accuracy = `0.9076`
- **Support Vector Machine**: CV F1 = `0.9048 (±0.0051)` | CV Accuracy = `0.9065`

### Selection Rationale
**Random Forest** (150 estimators, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=4`) is selected as the primary production model (`models/best_model.joblib`) because it achieves the highest F1-Score (**0.9386**), lowest False Positive Rate (**4.6%**), lowest False Negative Rate (**7.5%**), and highest area under the ROC curve (**0.9807**).

### Benchmark Evaluation Set (`data/test_urls.csv`)
Tested against 18 challenging, held-out URLs:
- **Benchmark Accuracy**: **17/18 (94.4%)**
- **Phishing Detection Rate**: **100% (8/8 phishing attacks caught)**
- **GitHub Classification**:
  - `https://github.com` -> **Legitimate** (93.4% confidence) ✅
  - `https://www.github.com` -> **Legitimate** (94.7% confidence) ✅
  - `https://github.com/features` -> **Legitimate** (93.4% confidence) ✅
  - `https://docs.github.com` -> **Legitimate** (94.5% confidence) ✅

---

## 6. Repository Structure

```text
PhishGuard/
├── main.py                     # Primary CLI application (direct commands & interactive menu)
├── train.py                    # End-to-end model training, 5-fold CV, and artifact saver
├── evaluate.py                 # Evaluation against test set & benchmark test_urls.csv
├── requirements.txt            # Python dependencies (CLI only; no web frameworks)
├── README.md                   # System documentation and viva guide
├── data/
│   ├── urls.csv                # 10,000 balanced instances (OpenDNS + PhishTank)
│   ├── test_urls.csv           # 18 curated benchmark URLs for regression & security testing
│   └── README.md               # Dataset source attribution and leakage prevention details
├── models/
│   ├── best_model.joblib       # Fitted Random Forest production pipeline
│   ├── decision_tree.joblib    # Comparative Decision Tree pipeline
│   ├── k-nearest_neighbors.joblib # Comparative KNN pipeline
│   ├── random_forest.joblib    # Comparative Random Forest pipeline
│   └── support_vector_machine.joblib # Comparative SVM pipeline
├── results/
│   ├── metrics.json            # Machine-readable test metrics and cross-validation stats
│   ├── model_comparison.csv    # Comparative performance table
│   ├── confusion_matrix.csv    # Confusion matrices for all 4 models
│   └── benchmark_urls_results.json # Benchmark test URL predictions
├── src/
│   ├── __init__.py             # Public API exports
│   ├── data_loader.py          # Data ingestion, schema validation, and deduplication
│   ├── url_features.py         # URL normalization and 16 pure lexical features
│   ├── preprocessing.py        # Feature extraction wrapper and stratified splitting
│   ├── model_pipeline.py       # Pipeline definitions, 5-fold CV, and model trainer
│   └── prediction.py           # Single-URL inference with probability calibration
└── tests/
    ├── __init__.py
    ├── test_url_normalization.py     # Unit tests for URL canonicalization & malformed input
    ├── test_feature_consistency.py  # Tests verifying 16 features, ordering, and determinism
    ├── test_label_mapping.py        # Tests verifying 0=Legitimate, 1=Phishing binary mapping
    └── test_prediction_pipeline.py  # End-to-end tests & GitHub false-positive regression
```

---

## 7. Installation & Requirements

### Prerequisites
- Python 3.9+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/tejaskedarpawar/phishguard.git
cd phishguard

# Install required dependencies
pip install -r requirements.txt
```

---

## 8. Command-Line Interface (CLI) Usage

### 1. Analyze a Specific URL Directly
```bash
python main.py --url https://github.com
```

*Output:*
```text
============================================================
                        PHISHGUARD
            ML-Based URL Risk Classifier (CLI)
============================================================

Analyzing: https://github.com
------------------------------------------------------------
Normalized URL : https://github.com
Hostname       : github.com
Model Result   : Legitimate
Risk Status    : LOW RISK (Likely Legitimate)
Confidence     : 93.4% (Phishing: 6.6% | Legit: 93.4%)
Classifier     : RandomForestClassifier (models/best_model.joblib)
------------------------------------------------------------
Lexical Feature Indicators:
  • URL Length            : 18
  • Hostname Length       : 10
  • Dot Count             : 1
  • Hyphen Count          : 0
  • Subdomains            : 0
  • IP Address Used       : No
  • URL Shortener         : No
  • Sensitive Keywords    : 0
------------------------------------------------------------
Note: This is an ML assessment based on 16 lexical and structural URL attributes. It is not an absolute guarantee of website safety or malice.
============================================================
```

### 2. Analyze a Phishing URL Directly
```bash
python main.py --url http://192.168.1.100/paypal-update/login.php
```

*Output:*
```text
Model Result   : Phishing
Risk Status    : HIGH RISK (Potentially Phishing)
Confidence     : 97.4% (Phishing: 97.4% | Legit: 2.6%)
Classifier     : RandomForestClassifier (models/best_model.joblib)
------------------------------------------------------------
Lexical Feature Indicators:
  • URL Length            : 44
  • Hostname Length       : 13
  • Dot Count             : 4
  • Hyphen Count          : 1
  • Subdomains            : 2
  • IP Address Used       : Yes
  • URL Shortener         : No
  • Sensitive Keywords    : 2
```

### 3. Retrain All Models & Save Pipelines
```bash
python main.py --train
# or
python train.py
```

### 4. Run Benchmark URL Evaluation
```bash
python main.py --evaluate
# or
python evaluate.py
```

### 5. View Dataset Summary
```bash
python main.py --info
```

### 6. Launch Interactive Menu
```bash
python main.py
```

*Interactive Menu Interface:*
```text
============================================================
                        PHISHGUARD
            ML-Based URL Risk Classifier (CLI)
============================================================
  1. Analyze a URL
  2. Train the model
  3. View model evaluation results
  4. View dataset information
  5. Exit
------------------------------------------------------------
Enter option [1-5]:
```

---

## 9. Automated Testing Suite

PhishGuard includes a complete automated test suite implemented with `pytest`:

```bash
pytest -v
```

### Test Coverage Summary (25 Passing Tests)
- `tests/test_url_normalization.py` (8 tests):
  - Empty URL, blank whitespace, and None rejection
  - Rejection of URLs with embedded whitespace or newlines
  - Maximum length boundary checks (> 2048 characters)
  - Scheme prepending (`github.com` -> `https://github.com`)
  - Case-insensitive scheme and hostname lowercasing
  - Standard port stripping (80, 443) and custom port retention
  - Query parameters and fragment preservation
  - IPv4 address hostnames
- `tests/test_feature_consistency.py` (7 tests):
  - Canonical 16 feature names and ordering
  - List and dictionary alignment
  - Deterministic feature extraction
  - Rejection of null, NaN, or infinite feature values
  - Raw IP address indicator
  - URL shortener indicator
  - Sensitive keyword counter
- `tests/test_label_mapping.py` (4 tests):
  - Canonical binary encoding (`0 = Legitimate`, `1 = Phishing`)
  - Dataset label space validation
  - Dataset summary metrics validation
  - Security positive class definition
- `tests/test_prediction_pipeline.py` (6 tests):
  - Missing model artifact error handling
  - Malformed URL rejection
  - **GitHub regression test** (asserting `github.com`, `www.github.com`, `github.com/features`, `docs.github.com` classify as Legitimate)
  - Top web service classification (`google.com`, `wikipedia.org`, `apple.com`, `stackoverflow.com`, `news.ycombinator.com`)
  - Real phishing attack detection (`192.168.1.100`, spoofing keywords, brand subdomains, shorteners)
  - Response dictionary schema contract

---

## 10. College Lab Demonstration & Viva Guide

Use this sequence for your laboratory viva or project presentation:

```bash
# Step 1: Run automated tests to prove code stability and absence of regressions
pytest -v

# Step 2: Show dataset information and balance
python main.py --info

# Step 3: Run model training (demonstrating 5-fold CV and comparative model training)
python main.py --train

# Step 4: Run benchmark evaluation (demonstrating GitHub passing and phishing caught)
python main.py --evaluate

# Step 5: Test GitHub live via CLI (proves fix without whitelisting)
python main.py --url https://github.com

# Step 6: Test a known phishing attack live via CLI
python main.py --url http://192.168.1.100/paypal-update/login.php
```

### Common Viva Questions & Model Answers

**Q1: Why did your original system classify `github.com` as phishing?**
> *Answer*: The previous system had a feature space mismatch. It was trained on the UCI dataset which required 30 features (including HTML DOM and DNS attributes). At runtime, raw URLs only provided 10 address-bar features, and the missing 20 features were arbitrarily imputed with `1`. In the UCI dataset, `1` on those features correlated with over 90% phishing prevalence, forcing all URLs into the phishing cluster. Additionally, legitimate training samples lacked path diversity, biasing decision trees against shallow paths. We resolved both issues by engineering 16 pure lexical features extracted consistently in training and inference and training on a balanced, path-diverse dataset.

**Q2: Did you whitelist `github.com` to make it pass?**
> *Answer*: No. Whitelisting is an anti-pattern in machine learning. `github.com` and all benchmark domains were strictly excluded from the training dataset. Its Legitimate classification is an honest prediction derived purely from the Random Forest model's decision tree thresholds across the 16 lexical features.

**Q3: Why did you select Random Forest over SVM, Decision Tree, and KNN?**
> *Answer*: In cybersecurity, the cost of a False Negative (missing a phishing site) is far higher than a False Positive. Random Forest achieved the highest F1-Score (0.9386) and ROC-AUC (0.9807) on the held-out test set, with a False Negative Rate of only 7.5% compared to 12.2% for Decision Tree and 12.3% for KNN. Its ensemble averaging also reduces variance on complex URLs.

**Q4: How do you prevent data leakage during preprocessing?**
> *Answer*: We perform an 80/20 stratified split *before* fitting any preprocessing transformations. The `StandardScaler` used by KNN and SVM is wrapped inside a Scikit-Learn `Pipeline`, ensuring scaling parameters ($\mu, \sigma$) are computed solely on `X_train`.

---

## 11. Known Limitations & Ethical Considerations

1. **Pure Lexical Analysis Scope**: PhishGuard evaluates URLs purely from their string structure. A compromised legitimate website hosting malicious content under a normal URL path may appear legitimate until reputation lists or content scanners flag it.
2. **Obfuscated Shorteners**: While PhishGuard detects shortener domains (`bit.ly`, `tinyurl.com`), resolving the destination domain requires an HTTP HEAD request, which is intentionally avoided to preserve zero-latency offline security.
3. **Defense-in-Depth**: Machine learning URL classifiers should be deployed as one layer in a multi-tier defense architecture, complementing DNS filtering, SPF/DKIM verification, and user security awareness training.
