"""
src/url_features.py
-------------------
URL normalization and pure lexical feature extraction.

All features are mathematically and deterministically derived directly from
the URL string. No network calls, DNS queries, or live page scraping are
performed, ensuring zero-latency, private, and offline-compatible execution.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# Feature names in canonical order
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_slash",
    "count_digits",
    "digit_ratio",
    "is_https",
    "count_subdomains",
    "is_ip",
    "has_shortener",
    "has_hyphen_in_domain",
    "count_sensitive_keywords",
    "path_depth",
]

# Known URL shortener services
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy", "v.gd", "tiny.cc", "lnkd.in", "rebrand.ly",
}

# Sensitive authentication and financial keywords frequently targeted by phishing campaigns
SENSITIVE_KEYWORDS = {
    "login", "signin", "verify", "verification", "secure", "account",
    "banking", "update", "confirm", "password", "credential", "auth",
    "wallet", "security", "recovery", "suspend", "ebayisapi", "webscr",
}


def normalize_url(raw_url: str) -> Tuple[bool, str, str]:
    """Validate and normalize any user-supplied URL.

    Handles:
    - Raw domains without scheme (github.com -> https://github.com)
    - Surrounding quotes, angle brackets, parentheses, or markdown syntax
    - Defanged threat intel feeds (hxxp://, hxxps://, [.], [:])
    - Trailing sentence punctuation (periods, commas, semicolons)
    - Internationalized Domain Names (IDN / punycode)
    - Custom ports, userinfo, queries, and fragments
    - Path normalization (removes redundant trailing slashes on sub-paths)
    - Rejects unsupported schemes (javascript:, data:, etc.) and internal whitespace

    Returns
    -------
    (is_valid, normalized_url, error_message)
    """
    if not raw_url or not isinstance(raw_url, str):
        return False, "", "URL cannot be empty."

    url = raw_url.strip()
    if not url:
        return False, "", "URL cannot be blank or whitespace only."

    # 1. Strip surrounding quotes, angle brackets, braces, and parentheses
    url = url.strip("\"'`<>(){}[]")

    # 2. Extract URL if wrapped in markdown link syntax [text](url)
    md_match = re.match(r"^\[.*?\]\((https?://.*?)\)$", url, flags=re.IGNORECASE)
    if md_match:
        url = md_match.group(1).strip()

    # 3. Refang defanged threat intelligence URLs (e.g. hxxp://, [.], [:])
    url = url.replace("[.]", ".").replace("[:]", ":")
    if re.match(r"^hxxps?://", url, flags=re.IGNORECASE):
        url = re.sub(r"^hxxps://", "https://", url, flags=re.IGNORECASE)
        url = re.sub(r"^hxxp://", "http://", url, flags=re.IGNORECASE)

    # 4. Strip trailing punctuation commonly picked up from sentences (e.g. "https://github.com.")
    url = re.sub(r"[\.,;:!?]+$", "", url)

    # 5. Length boundary check
    if len(url) > 2048:
        return False, "", "URL exceeds maximum permitted length (2048 characters)."

    # 6. Reject unsupported non-web protocols
    for unsupp in ("javascript:", "data:", "file:", "vbscript:", "mailto:", "tel:", "sms:"):
        if url.lower().startswith(unsupp):
            return False, "", f"Unsupported protocol '{unsupp}'. PhishGuard analyzes web URLs (HTTP/HTTPS)."

    # 7. Reject invalid internal whitespace
    if any(ws in url for ws in (" ", "\t", "\n", "\r")):
        return False, "", "URL contains invalid whitespace characters."

    # 8. Ensure scheme is present (defaults to https://)
    lower_check = url.lower()
    if not (lower_check.startswith("http://") or lower_check.startswith("https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False, "", "Could not parse a valid hostname from the URL."

        # Handle IDN / Unicode domains safely via Punycode
        try:
            hostname_ascii = hostname.encode("idna").decode("ascii").lower()
        except Exception:
            hostname_ascii = hostname.lower()

        # Reconstruct userinfo if present in netloc (preserves @ symbol for phishing indicators)
        userinfo = ""
        if "@" in parsed.netloc:
            user_part = parsed.netloc.split("@")[0]
            userinfo = f"{user_part}@"

        # Canonicalize path: strip redundant trailing slashes on sub-paths (/downloads/ -> /downloads)
        clean_path = parsed.path or ""
        if clean_path and clean_path != "/" and clean_path.endswith("/"):
            clean_path = clean_path.rstrip("/")

        # Reconstruct canonical normalized URL
        normalized = f"{parsed.scheme.lower()}://{userinfo}{hostname_ascii}"
        if parsed.port and parsed.port not in (80, 443):
            normalized += f":{parsed.port}"
        if clean_path:
            normalized += clean_path
        if parsed.query:
            normalized += f"?{parsed.query}"
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"

        return True, normalized, ""
    except Exception as exc:
        return False, "", f"Malformed URL syntax: {exc}"


def extract_features_dict(url: str) -> Dict[str, Any]:
    """Extract 16 pure lexical features as a named dictionary."""
    is_valid, norm_url, _ = normalize_url(url)
    if not is_valid:
        norm_url = url.strip()

    if not norm_url.startswith(("http://", "https://")):
        parsed_url = "https://" + norm_url
    else:
        parsed_url = norm_url

    parsed = urlparse(parsed_url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    url_length = len(norm_url)
    hostname_length = len(hostname)
    path_length = len(path)

    count_dots = norm_url.count(".")
    count_hyphens = norm_url.count("-")
    count_at = norm_url.count("@")
    count_slash = path.count("/")
    count_digits = sum(c.isdigit() for c in norm_url)
    digit_ratio = round(count_digits / max(url_length, 1), 4)

    is_https = 1 if norm_url.lower().startswith("https://") else 0

    # Count subdomains (dots in hostname minus 1 for base domain)
    count_subdomains = max(0, hostname.count(".") - 1)

    # Check for IPv4 or IPv6 address in hostname
    is_ip = 1 if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", hostname) or ":" in hostname else 0

    # Check for known URL shortening service
    has_shortener = 1 if any(short in hostname for short in SHORTENER_DOMAINS) else 0

    # Check for hyphen in domain name (common in deceptive brand impersonation)
    has_hyphen_in_domain = 1 if "-" in hostname else 0

    # Tokenize URL and count phishing sensitive keywords
    tokens = re.split(r"[^a-zA-Z0-9]", (hostname + "/" + path + "?" + query).lower())
    count_sensitive_keywords = sum(1 for t in tokens if t in SENSITIVE_KEYWORDS)

    # Path directory depth
    path_depth = len([segment for segment in path.split("/") if segment])

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "count_dots": count_dots,
        "count_hyphens": count_hyphens,
        "count_at": count_at,
        "count_slash": count_slash,
        "count_digits": count_digits,
        "digit_ratio": digit_ratio,
        "is_https": is_https,
        "count_subdomains": count_subdomains,
        "is_ip": is_ip,
        "has_shortener": has_shortener,
        "has_hyphen_in_domain": has_hyphen_in_domain,
        "count_sensitive_keywords": count_sensitive_keywords,
        "path_depth": path_depth,
    }


def extract_features(url: str) -> List[Any]:
    """Extract 16 pure lexical features in exact canonical order."""
    f_dict = extract_features_dict(url)
    return [f_dict[name] for name in FEATURE_NAMES]
