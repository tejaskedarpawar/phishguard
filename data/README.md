# PhishGuard Dataset Documentation

This directory contains the training and evaluation datasets used by PhishGuard for raw-URL phishing detection.

---

## 1. Primary Training Dataset: `urls.csv`

- **Filename**: `data/urls.csv`
- **Total Records**: 10,000 balanced instances (5,000 Legitimate, 5,000 Phishing)
- **Format**: CSV (comma-separated values)
- **Columns**:
  - `url` (string): The complete raw URL string.
  - `label` (integer): Binary target indicator:
    - `0`: **Legitimate** (Negative Class)
    - `1`: **Phishing** (Positive Class)

### Data Sources & Attribution

1. **Legitimate URLs (Label = 0, 5,000 instances)**:
   - **OpenDNS Public Domain Lists (Top Domains)**: Real-world high-reputation domains representing standard web traffic. Source: [OpenDNS Public Domain Lists](https://github.com/opendns/public-domain-lists).
   - **Realistic Path and Subdomain Distribution**: To avoid covariate shift (where legitimate training data previously consisted only of root domains or exclusively long torrent paths), URLs feature authentic web structure distributions:
     - 25% root domains (`/`, empty)
     - 45% single-segment paths (`/about`, `/contact`, `/features`, `/pricing`, `/docs`, `/blog`, etc.)
     - 20% two-segment paths (`/wiki/Article`, `/docs/api`, `/products/category`, `/help/faq`, etc.)
     - 10% multi-segment paths (`/docs/v2/getting-started`, `/blog/2024/update`, etc.)
     - 25% `www.` prefixes, 15% standard functional subdomains (`docs.`, `api.`, `blog.`, `support.`, `news.`, `shop.`, `m.`, `en.`)
     - 92% HTTPS / 8% HTTP protocol mix

2. **Phishing URLs (Label = 1, 5,000 instances)**:
   - **PhishTank Verified Active Phishing Database**: Verified community-submitted and expert-confirmed phishing campaigns targeting financial institutions, web services, and user credentials. Source: [PhishTank](https://www.phishtank.com).

### Data Hygiene & Leakage Prevention
- **Deduplication**: Exact duplicate URLs have been purged.
- **Strict Benchmark Hold-Out**: Crucially, test evaluation domains (including `github.com`, `google.com`, `wikipedia.org`, `python.org`, `stackoverflow.com`, `apple.com`, `microsoft.com`, `ycombinator.com`) were strictly excluded from the training dataset. This ensures that evaluation on `https://github.com` and related benchmark URLs tests true model generalization with zero training leakage.

---

## 2. Separate Evaluation Benchmark: `test_urls.csv`

- **Filename**: `data/test_urls.csv`
- **Total Records**: 18 curated benchmark URLs
- **Columns**: `url`, `label`, `category`, `source`
- **Purpose**: Used by `python evaluate.py` to test model behavior on:
  - Legitimate root domains (`https://github.com`, `https://news.ycombinator.com`)
  - Legitimate subdomains (`https://www.github.com`, `https://docs.github.com`)
  - Legitimate deep paths (`https://github.com/features`, `https://www.wikipedia.org/wiki/Phishing`)
  - Phishing IP addresses (`http://192.168.1.100/paypal-update/login.php`)
  - Phishing keyword traps (`http://secure-login-account-verification-alert.com/signin?user=test`)
  - Phishing subdomain brand impersonation (`http://paypal.com.verify-security-check.xyz/login?account=123`)
  - Phishing userinfo/at-symbol spoofing (`http://user@evil-bank-credential-harvest.net/auth`)
  - Phishing URL shortener obfuscation (`http://bit.ly/secure-banking-update-notice`)
