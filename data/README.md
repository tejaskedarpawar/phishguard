# Dataset Directory

Place the UCI Phishing Websites dataset CSV file here as `phishing.csv`.

## Source

- **Name**: Phishing Websites Dataset
- **Repository**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/327/phishing+websites
- **Donated by**: Rami Mohammad, Fadi Thabtah, Lee McCluskey (2015)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Download Instructions

### Option A — UCI Repository (ARFF format, then convert)
1. Visit https://archive.ics.uci.edu/dataset/327/phishing+websites
2. Download `phishing+websites.zip`
3. Extract the `.arff` file
4. Convert to CSV using the provided `scripts/arff_to_csv.py` helper, or use the Kaggle mirror below.

### Option B — Kaggle CSV mirror (recommended for quick setup)
1. Visit https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector
2. Download `phishing.csv`
3. Place it in this `data/` folder as `phishing.csv`

## Expected Schema

| Column | Type | Values |
|--------|------|--------|
| having_IP_Address | int | -1, 1 |
| URL_Length | int | -1, 0, 1 |
| Shortining_Service | int | -1, 1 |
| having_At_Symbol | int | -1, 1 |
| double_slash_redirecting | int | -1, 1 |
| Prefix_Suffix | int | -1, 1 |
| having_Sub_Domain | int | -1, 0, 1 |
| SSLfinal_State | int | -1, 0, 1 |
| Domain_registeration_length | int | -1, 1 |
| Favicon | int | -1, 1 |
| port | int | -1, 1 |
| HTTPS_token | int | -1, 1 |
| Request_URL | int | -1, 1 |
| URL_of_Anchor | int | -1, 0, 1 |
| Links_in_tags | int | -1, 0, 1 |
| SFH | int | -1, 0, 1 |
| Submitting_to_email | int | -1, 1 |
| Abnormal_URL | int | -1, 1 |
| Redirect | int | 0, 1 |
| on_mouseover | int | -1, 1 |
| RightClick | int | -1, 1 |
| popUpWindow | int | -1, 1 |
| Iframe | int | -1, 1 |
| age_of_domain | int | -1, 1 |
| DNSRecord | int | -1, 1 |
| web_traffic | int | -1, 0, 1 |
| Page_Rank | int | -1, 1 |
| Google_Index | int | -1, 1 |
| Links_pointing_to_page | int | -1, 0, 1 |
| Statistical_report | int | -1, 1 |
| **Result** | int | **-1 (Phishing), 1 (Legitimate)** |

## Notes

- All 30 features are pre-extracted categorical integer codes: -1 (phishing indicator), 0 (suspicious), 1 (legitimate indicator).
- The target column `Result` uses -1 for Phishing and 1 for Legitimate.
- Do NOT include the target column in the feature matrix when training or predicting.
- The dataset contains 11,055 instances total.
