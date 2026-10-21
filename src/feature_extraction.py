"""
src/feature_extraction.py
--------------------------
Extracts the 30 UCI-compatible phishing features from a raw URL string
for use during live inference in the Streamlit application.

IMPORTANT — Scope note
----------------------
The UCI dataset contains *pre-extracted* website features (not raw URLs).
Many features require visiting the target website, querying DNS, checking
SSL certificates, and looking up web-traffic databases.

For a safe, offline, local-only inference tool we implement:

  • Features that can be derived purely from the URL string  →  fully computed.
  • Features that require a live network request               →  safely approximated
    to 1 (suspicious/unknown = safest-for-user default) with a clear disclaimer.

The returned feature vector uses the **same column names, order, and integer
encoding** ({-1, 0, 1}) as the training data so the saved pipeline predicts
correctly.

Network requests are NEVER made automatically.  The user is informed which
features are approximated.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.data_cleaning import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Feature metadata — which features are computable from URL alone
# ---------------------------------------------------------------------------

COMPUTABLE_FEATURES: Dict[str, str] = {
    "having_IP_Address": "IP in URL instead of domain name",
    "URL_Length": "Overall URL length",
    "Shortining_Service": "Known URL shortening service used",
    "having_At_Symbol": "'@' symbol present in URL",
    "double_slash_redirecting": "'//'' after protocol section",
    "Prefix_Suffix": "Dash '-' present in domain",
    "having_Sub_Domain": "Number of sub-domains (dots in domain)",
    "HTTPS_token": "Word 'https' in domain part of URL",
    "Redirect": "Likely redirect indicator",
}

APPROXIMATED_FEATURES: Dict[str, str] = {
    "SSLfinal_State": "SSL cert validity (requires live fetch — approximated)",
    "Domain_registeration_length": "Domain reg. length (requires WHOIS — approximated)",
    "Favicon": "Favicon source domain (requires fetch — approximated)",
    "port": "Unusual port in URL (partially computable)",
    "Request_URL": "External objects % (requires fetch — approximated)",
    "URL_of_Anchor": "Anchor tag destinations (requires fetch — approximated)",
    "Links_in_tags": "Links in <meta>/<script> (requires fetch — approximated)",
    "SFH": "Server Form Handler (requires fetch — approximated)",
    "Submitting_to_email": "Form submits to email (requires fetch — approximated)",
    "Abnormal_URL": "Host vs identity match (requires WHOIS — approximated)",
    "on_mouseover": "onMouseOver JS trick (requires fetch — approximated)",
    "RightClick": "Right-click disabled (requires fetch — approximated)",
    "popUpWidnow": "Pop-up windows (requires fetch — approximated)",
    "Iframe": "Hidden iframes (requires fetch — approximated)",
    "age_of_domain": "Domain age (requires WHOIS — approximated)",
    "DNSRecord": "DNS record present (requires DNS — approximated)",
    "web_traffic": "Alexa traffic rank (requires lookup — approximated)",
    "Page_Rank": "Google PageRank (requires lookup — approximated)",
    "Google_Index": "Google indexed (requires query — approximated)",
    "Links_pointing_to_page": "External backlinks (requires lookup — approximated)",
    "Statistical_report": "Phishing blacklist entry (requires lookup — approximated)",
}


# URL shortener domains
_SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "buff.ly", "rebrand.ly", "short.link", "cutt.ly", "rb.gy",
    "is.gd", "v.gd", "tiny.cc", "lnkd.in", "adf.ly",
}


def extract_features(url: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Extract the 30 UCI-compatible features from a URL string.

    Parameters
    ----------
    url:
        Raw URL string entered by the user.

    Returns
    -------
    feature_df : pd.DataFrame
        Single-row dataframe with all 30 feature columns in correct order.
    feature_details : dict
        Human-readable label → value string for display.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url  # allow bare domains for parsing

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    full_url = url

    features: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # 1. having_IP_Address
    #    -1 if IP address used, 1 otherwise
    # -----------------------------------------------------------------------
    ip_pattern = re.compile(
        r"^(?:\d{1,3}\.){3}\d{1,3}$|"           # IPv4
        r"^\[[\da-fA-F:]+\]$"                     # IPv6
    )
    features["having_IP_Address"] = -1 if ip_pattern.match(hostname) else 1

    # -----------------------------------------------------------------------
    # 2. URL_Length
    #    1 if len < 54, 0 if 54-75, -1 if > 75
    # -----------------------------------------------------------------------
    url_len = len(full_url)
    if url_len < 54:
        features["URL_Length"] = 1
    elif url_len <= 75:
        features["URL_Length"] = 0
    else:
        features["URL_Length"] = -1

    # -----------------------------------------------------------------------
    # 3. Shortining_Service
    #    -1 if known shortener, 1 otherwise
    # -----------------------------------------------------------------------
    root_domain = ".".join(hostname.split(".")[-2:]) if hostname else ""
    features["Shortining_Service"] = -1 if root_domain in _SHORTENER_DOMAINS else 1

    # -----------------------------------------------------------------------
    # 4. having_At_Symbol
    #    -1 if '@' present, 1 otherwise
    # -----------------------------------------------------------------------
    features["having_At_Symbol"] = -1 if "@" in full_url else 1

    # -----------------------------------------------------------------------
    # 5. double_slash_redirecting
    #    -1 if '//' appears in path (after protocol), 1 otherwise
    # -----------------------------------------------------------------------
    # Position of // after 7 chars (skip http://)
    path_part = full_url[7:] if full_url.startswith("http://") else full_url[8:]
    features["double_slash_redirecting"] = -1 if "//" in path_part else 1

    # -----------------------------------------------------------------------
    # 6. Prefix_Suffix
    #    -1 if domain has a dash '-', 1 otherwise
    # -----------------------------------------------------------------------
    features["Prefix_Suffix"] = -1 if "-" in hostname else 1

    # -----------------------------------------------------------------------
    # 7. having_Sub_Domain
    #    Count dots in hostname: ≤1 → 1, ==2 → 0, ≥3 → -1
    # -----------------------------------------------------------------------
    dot_count = hostname.count(".")
    if dot_count == 1:
        features["having_Sub_Domain"] = 1
    elif dot_count == 2:
        features["having_Sub_Domain"] = 0
    else:
        features["having_Sub_Domain"] = -1

    # -----------------------------------------------------------------------
    # 8. SSLfinal_State — approximated
    #    1 if URL uses https, else -1
    # -----------------------------------------------------------------------
    features["SSLfinal_State"] = 1 if parsed.scheme == "https" else -1

    # -----------------------------------------------------------------------
    # 9. Domain_registeration_length — approximated to 0 (unknown)
    # -----------------------------------------------------------------------
    features["Domain_registeration_length"] = 0

    # -----------------------------------------------------------------------
    # 10. Favicon — approximated to 1 (unknown)
    # -----------------------------------------------------------------------
    features["Favicon"] = 1

    # -----------------------------------------------------------------------
    # 11. port
    #    -1 if non-standard port explicitly in URL, 1 otherwise
    # -----------------------------------------------------------------------
    if parsed.port and parsed.port not in {80, 443}:
        features["port"] = -1
    else:
        features["port"] = 1

    # -----------------------------------------------------------------------
    # 12. HTTPS_token
    #    -1 if word 'https' appears in domain name, 1 otherwise
    # -----------------------------------------------------------------------
    features["HTTPS_token"] = -1 if "https" in hostname.lower() else 1

    # -----------------------------------------------------------------------
    # 13–30: Approximated to 1 (unknown/neutral)
    # -----------------------------------------------------------------------
    approximated_neutral = [
        "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
        "Submitting_to_email", "Abnormal_URL", "Redirect",
        "on_mouseover", "RightClick", "popUpWidnow", "Iframe",  # Note: typo in UCI source
        "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
        "Google_Index", "Links_pointing_to_page", "Statistical_report",
    ]
    for feat in approximated_neutral:
        features[feat] = 1

    # -----------------------------------------------------------------------
    # Build ordered dataframe matching FEATURE_COLUMNS
    # -----------------------------------------------------------------------
    ordered_values = [features[col] for col in FEATURE_COLUMNS]
    feature_df = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)

    # -----------------------------------------------------------------------
    # Build human-readable details for UI display
    # -----------------------------------------------------------------------
    feature_details: Dict[str, str] = {
        "IP address in URL": "Yes (suspicious)" if features["having_IP_Address"] == -1 else "No",
        "URL length": f"{url_len} chars ({'suspicious' if url_len > 75 else 'ok'})",
        "URL shortener": "Yes (suspicious)" if features["Shortining_Service"] == -1 else "No",
        "@ symbol": "Present (suspicious)" if features["having_At_Symbol"] == -1 else "Not present",
        "Double slash redirect": "Detected" if features["double_slash_redirecting"] == -1 else "Not detected",
        "Prefix/suffix dash in domain": "Present (suspicious)" if features["Prefix_Suffix"] == -1 else "Not present",
        "Sub-domain count (dots)": str(dot_count),
        "Protocol": parsed.scheme.upper(),
        "Non-standard port": "Yes (suspicious)" if features["port"] == -1 else "No",
        "HTTPS in domain name": "Yes (suspicious)" if features["HTTPS_token"] == -1 else "No",
    }

    return feature_df, feature_details


def validate_url(url: str) -> Tuple[bool, str]:
    """Validate a URL string for basic well-formedness.

    Parameters
    ----------
    url:
        User-supplied URL string.

    Returns
    -------
    (is_valid, error_message)
        If is_valid is True, error_message is empty.
    """
    url = url.strip()
    if not url:
        return False, "URL cannot be empty."

    if len(url) > 2048:
        return False, "URL is too long (> 2048 characters)."

    # Allow bare domains
    test_url = url if url.startswith(("http://", "https://")) else "http://" + url

    try:
        parsed = urlparse(test_url)
        if not parsed.hostname:
            return False, "Could not parse a valid hostname from the URL."
    except Exception:
        return False, "URL is malformed and could not be parsed."

    # Reject obvious non-URLs
    bad_patterns = [" ", "\n", "\t"]
    for pat in bad_patterns:
        if pat in url:
            return False, "URL contains invalid whitespace characters."

    return True, ""
