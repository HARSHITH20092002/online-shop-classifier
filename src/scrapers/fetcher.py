import json
import urllib3
import requests
from urllib.parse import urlparse
from curl_cffi import requests as cffi_requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FULL_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9,fr-FR,fr;q=0.8,de;q=0.7,es;q=0.6",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def fetch_wayback_archive(url):
    """
    Tier 3 Fallback: Queries Internet Archive Wayback Machine API
    to retrieve the most recent active snapshot of an unreachable domain.
    """
    try:
        api_url = f"https://archive.org/wayback/available?url={url}"
        res = requests.get(api_url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            closest = data.get("archived_snapshots", {}).get("closest", {})
            if closest.get("available") and closest.get("url"):
                snapshot_url = closest.get("url")
                snap_res = requests.get(snapshot_url, headers=FULL_HEADERS, timeout=8)
                if snap_res.status_code == 200 and len(snap_res.text) > 100:
                    return {
                        "url": url,
                        "raw_text": snap_res.text,
                        "status_code": 200,
                        "data_source": "Wayback Machine Archive",
                        "is_reachable": True
                    }
    except Exception:
        pass
    return None

def fetch_website_content(url, historical_row=None):
    """
    Multi-Tier Data Ingestion Engine:
    - Tier 1: Direct TLS Browser Fingerprint Impersonation (Chrome 120 / Safari 15.5)
    - Tier 2: Configuration Path Extraction (/robots.txt)
    - Tier 3: Internet Archive Wayback Machine API
    - Tier 4: Discovery Search Log Metadata (Snippet + Query Title)
    """
    # Tier 1: Live Web Request via TLS Fingerprint Impersonation
    for impersonate_target in ["chrome120", "safari15_5"]:
        try:
            resp = cffi_requests.get(
                url, 
                impersonate=impersonate_target, 
                headers=FULL_HEADERS, 
                timeout=6, 
                verify=False,
                allow_redirects=True
            )
            if resp.status_code < 400 and len(resp.text) > 150:
                return {
                    "url": url,
                    "raw_text": resp.text,
                    "status_code": resp.status_code,
                    "data_source": "Live DOM",
                    "is_reachable": True
                }
        except Exception:
            continue

    # Tier 2: Robots.txt Fallback
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        resp = cffi_requests.get(robots_url, impersonate="chrome120", headers=FULL_HEADERS, timeout=5, verify=False)
        if resp.status_code == 200 and len(resp.text) > 30:
            return {
                "url": url,
                "raw_text": resp.text,
                "status_code": 200,
                "data_source": "Live Robots.txt",
                "is_reachable": True
            }
    except Exception:
        pass

    # Tier 3: Internet Archive Snapshot (Wayback Machine)
    archive_result = fetch_wayback_archive(url)
    if archive_result:
        return archive_result

    # Tier 4: Dataset Historical Metadata (Search Title + Snippet)
    if historical_row is not None:
        title = str(historical_row.get("title", "")) if pd_not_na(historical_row.get("title")) else ""
        snippet = str(historical_row.get("snippet", "")) if pd_not_na(historical_row.get("snippet")) else ""
        item_name = str(historical_row.get("item_name", "")) if pd_not_na(historical_row.get("item_name")) else ""
        combined_text = f"{title} {snippet} {item_name}".strip()

        if len(combined_text) > 5:
            return {
                "url": url,
                "raw_text": combined_text,
                "status_code": 404,
                "data_source": "Historical Search Snippet",
                "is_reachable": False
            }

    # Complete Unreachability
    return {
        "url": url,
        "raw_text": "",
        "status_code": 0,
        "data_source": "Unreachable (No History)",
        "is_reachable": False
    }

def pd_not_na(val):
    return val is not None and str(val).lower() not in ["nan", "none", ""]