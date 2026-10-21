# PhishGuard: ML-Based Phishing Website Detection System
## Academic Laboratory Report Outline (10 Pages)

---

### Title Page
- **Project Title:** PhishGuard: An ML-Based Phishing Website Detection System
- **Course:** Machine Learning Laboratory
- **Author:** [Student Name]
- **Institution:** [College / University Department of Computer Science & Engineering]
- **Date:** Academic Session 2026

---

### Abstract (≈ 150 words)
Phishing attacks remain one of the most pervasive cyber threats, circumventing traditional static blacklists through fast-flux domains and ephemeral hosting. This report presents **PhishGuard**, an empirical comparative machine learning study evaluating four supervised classifiers—Decision Tree (Information Gain / Entropy), K-Nearest Neighbors (Euclidean distance with feature scaling), Random Forest (150 bootstrapped estimators), and Support Vector Machine (RBF kernel with probability calibration)—on the benchmark UCI Phishing Websites dataset. After deduplicating 740 redundant instances from the raw 2,456-sample corpus, models were trained using 5-fold stratified cross-validation on an 80% training partition and evaluated on a held-out 20% test set (344 samples). Random Forest achieved the superior performance with a test F1-score of **0.9479**, recall of **0.9611**, and ROC-AUC of **0.9891**, catching 173 of 180 phishing instances. An interactive, offline-safe Streamlit web application with lexical URL extraction and test benchmark inspection was developed to operationalize model evaluation for faculty demonstration.

---

### Table of Contents & Page Budget Allocation
- **Section 1: Introduction** (0.75 Page)
- **Section 2: Literature Review** (1.5 Pages)
- **Section 3: Methodology & System Architecture** (1.25 Pages)
- **Section 4: Data Hygiene, Preprocessing & Feature Engineering** (1.5 Pages)
- **Section 5: Experimental Results and Metric Analysis** (2.0 Pages)
- **Section 6: Traditional Baselines vs. State-of-the-Art Approaches** (1.0 Page)
- **Section 7: User Interface Design and Deployment Architecture** (1.0 Page)
- **Section 8: Conclusion, Ethical Considerations & Future Work** (0.5 Page)
- **Section 9: Academic References & Dataset Attribution** (0.5 Page)

---

### 1. Introduction
#### 1.1 Background & Threat Landscape
- Growth of targeted credential harvesting, social engineering, and financial fraud.
- Inherent latency and coverage limitations of reactive domain blacklists (e.g., DNSBL, Google Safe Browsing lag).
- The role of automated machine learning heuristics for zero-hour detection.

#### 1.2 Problem Statement
- Formally define the binary classification objective: mapping a 30-dimensional feature vector $\mathbf{x} \in \{-1, 0, 1\}^{30}$ to target class $y \in \{-1, 1\}$, where $-1$ denotes a malicious phishing website and $1$ denotes a legitimate website.

#### 1.3 Key Objectives
- Complete end-to-end ML pipeline adhering to strict data hygiene (no test leakage).
- Comparative analysis of four distinct algorithmic paradigms (tree-based, instance-based, ensemble, kernel-based).
- Rigorous metric selection emphasizing phishing class recall and F1-score over raw accuracy.
- Presentation-ready interactive dashboard with transparent lexical inspection.

---

### 2. Literature Review
#### 2.1 Taxonomy of Phishing Detection Techniques
- **List-based mechanisms:** Blacklisting / Whitelisting (high precision, low recall on zero-day domains).
- **Heuristic & Lexical analysis:** Character frequencies, domain length, entropy, keyword spoofing.
- **Content & Visual similarity:** DOM structure inspection, visual screenshot hashing (computationally intensive, requires active rendering).
- **Machine Learning Classifiers:** Supervised learning over standardized feature matrices.

#### 2.2 Literature Review Matrix

| Authors & Year | Methodology / Model | Dataset Evaluated | Key Findings | Documented Limitations | Relevance to PhishGuard |
|---|---|---|---|---|---|
| Mohammad, Thabtah, & McCluskey (2015) | Self-Structuring Neural Networks & Rule Induction | UCI Phishing Websites (11,055 instances) | Demonstrated 30-feature extraction framework combining address bar, abnormal, HTML, and domain features | Neural network lacked interpretability; high feature extraction latency | Original donors of the benchmark dataset used in this project |
| Rao & Ali (2020) | Random Forest & Gradient Boosted Trees | PhishTank + Alexa Top 1M | Ensemble trees achieved >97% accuracy on pre-extracted features | Heavy reliance on third-party Alexa traffic metrics (now deprecated) | Justifies selection of Random Forest as primary ensemble baseline |
| Sahingoz et al. (2019) | NLP & Word Vectors on raw URLs with Decision Trees & SVM | 73,575 URLs | Character n-grams and word vectors detect brand impersonation in URL text | Susceptible to adversarial evasion via character substitution | Informs our lexical URL feature extraction design |
| *Verified Literature Status:* | *Pending peer-reviewed library verification* | — | — | — | *No fabricated citations used* |

---

### 3. Methodology & System Architecture
#### 3.1 Pipeline Architecture
- Strict Scikit-Learn `Pipeline` integration.
- `StandardScaler` embedded within KNN and SVM pipelines so scaling parameters $\mu, \sigma$ are fitted strictly on training folds, eliminating test set contamination.
- Cross-validation: 5-fold stratified cross-validation on `X_train` ($N=1,372$) for model validation prior to single-pass evaluation on the held-out test partition ($N=344$).

#### 3.2 Formal Description of Implemented Algorithms
1. **Decision Tree Classifier:**
   - Evaluated using Information Gain / Entropy criterion:
     $$H(S) = -\sum_{i} p_i \log_2 p_i, \quad IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$
   - Implementation note: Executed via scikit-learn's optimized CART algorithm configured with `criterion="entropy"`, regularized by `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5`.
2. **K-Nearest Neighbors (KNN):**
   - Distance metric: Euclidean distance over standardized features, weighted inversely by distance ($w_i = 1/d_i$). $k=7$.
3. **Random Forest Classifier:**
   - Ensemble of $B=150$ randomized decision trees with bootstrap aggregation (bagging) and random feature subspace selection ($\sqrt{p}$ features per split).
4. **Support Vector Machine (SVM):**
   - RBF kernel: $K(x, x') = \exp(-\gamma ||x - x'||^2)$.
   - Probability calibration via `CalibratedClassifierCV` to generate reliable continuous probability scores for ROC-AUC analysis.

---

### 4. Data Hygiene, Preprocessing & Feature Engineering
#### 4.1 Dataset Characteristics & Schema
- Raw data: 2,456 instances, 30 features, 1 target column (`Result`).
- Data hygiene finding: Detected and purged **740 duplicate records**, yielding 1,716 unique feature vectors (900 Phishing [52.45%], 816 Legitimate [47.55%]).
- Failure to drop duplicates would artificially inflate test performance due to identical instances co-occurring across splits.

#### 4.2 Feature Categorization (30 Features)
- **Group 1: Address Bar Features (1–12):** `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`, `having_Sub_Domain`, `SSLfinal_State`, `Domain_registeration_length`, `Favicon`, `port`, `HTTPS_token`.
- **Group 2: Abnormal Features (13–18):** `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email`, `Abnormal_URL`.
- **Group 3: HTML & JavaScript Features (19–23):** `Redirect`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe`.
- **Group 4: Domain-Based Features (24–30):** `age_of_domain`, `DNSRecord`, `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`, `Statistical_report`.

---

### 5. Experimental Results and Metric Analysis
#### 5.1 Test Set Performance Summary ($N=344$, Positive Class: Phishing $-1$)

| Metric | Random Forest | Decision Tree | Support Vector Machine | K-Nearest Neighbors |
|---|---|---|---|---|
| **Accuracy** | **0.9448** | 0.9419 | 0.9331 | 0.9128 |
| **Precision (Phishing)** | 0.9351 | **0.9396** | 0.9337 | 0.9213 |
| **Recall (Phishing)** | **0.9611** | 0.9500 | 0.9389 | 0.9111 |
| **F1-Score (Phishing)** | **0.9479** | 0.9448 | 0.9363 | 0.9162 |
| **ROC-AUC** | **0.9891** | 0.9720 | 0.9868 | 0.9766 |
| **Training Time (s)** | 0.1825 s | **0.0042 s** | 0.0915 s | 0.0048 s |

#### 5.2 Confusion Matrix Breakdown
- **Random Forest:** True Positives (Phishing Caught) = **173**, False Negatives (Phishing Missed) = **7**, False Positives = **12**, True Negatives = **152**.
- **Decision Tree:** TP = 171, FN = 9, FP = 11, TN = 153.
- **Support Vector Machine:** TP = 169, FN = 11, FP = 12, TN = 152.
- **K-Nearest Neighbors:** TP = 164, FN = 16, FP = 14, TN = 150.

#### 5.3 5-Fold Stratified Cross-Validation on Training Data
- Demonstrates consistent generalization without overfitting. Random Forest achieved mean CV F1 of $0.9563 \pm 0.0105$, corroborating its superior rank on the independent test set.

---

### 6. Traditional Baselines vs. State-of-the-Art Approaches
- **Clarification:** Traditional supervised classifiers (Decision Tree, KNN, Random Forest, SVM) represent established, interpretable tabular baselines.
- **Genuine SOTA Approaches in Research Literature:**
  - Deep Learning on raw URL character sequences (1D-CNNs, BiLSTM with attention).
  - Pretrained Transformer models (BERT/RoBERTa fine-tuned on URL text or HTML token streams).
  - Graph Neural Networks (GNNs) analyzing IP/Domain co-occurrence graphs.
- **Comparative Trade-off:** While deep learning approaches achieve higher resilience against lexical perturbations, our tabular baseline pipeline provides near-instantaneous training ($<0.2$ s), zero GPU dependency, and high interpretability for college laboratory environments.

---

### 7. UI Development & Deployment Architecture
- Built with **Streamlit** following minimalist cybersecurity UI/UX principles (light `#f8f9fb` canvas, dark typography, restrained blue `#2563eb`, red `#dc2626` alerts).
- Implements two evaluation workflows:
  1. **Live URL Heuristic Extraction:** Computes 10 address-bar lexical features directly from URL strings, transparently explaining why 20 content/DNS features are held neutral offline.
  2. **Verified Test Record Benchmark:** Evaluates complete 30-feature records from the held-out test set for faculty viva examination.

---

### 8. Conclusion & Limitations
- Summary of findings confirming Random Forest as the optimal classifier (0.9479 F1, 0.9611 Recall).
- Disclosures regarding lexical approximation limitations, lack of active DNS lookup, and temporal dataset drift.

---

### 9. References & Attribution
- Mohammad, R. M., Thabtah, F., & McCluskey, L. (2015). *Phishing Websites Dataset*. UCI Machine Learning Repository. DOI: 10.24432/C51W2X.
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
