"""Web search tool. Uses Serper.dev for Google results, then fetches top pages.

Results are cached per query (in-memory) so a critic revision never spends
a second API call on the same symptom.
"""
import re
import requests
from .. import config

_cache: dict = {}

_MOCK_RESULTS = [
    {"title": "P0171 on a Corolla - fixed, it was a vacuum leak",
     "link": "mock://forum-thread-1",
     "snippet": "LTFT was +18 at idle, dropped to +5 cruising. Smoke test found a cracked intake boot."},
    {"title": "Cleaning the MAF fixed my lean code",
     "link": "mock://forum-thread-2",
     "snippet": "MAF was reading low at idle. CRC MAF cleaner brought trims back to normal."},
]


def search(query: str, num: int = 5) -> list[dict]:
    """Return a list of {title, link, snippet} for a query."""
    if query in _cache:
        return _cache[query]
    if config.MOCK_MODE:
        _cache[query] = _MOCK_RESULTS
        return _MOCK_RESULTS
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num},
        timeout=15,
    )
    if not resp.ok:
        key = config.SERPER_API_KEY
        fingerprint = f"len={len(key)} tail=...{key[-4:]}" if key else "EMPTY"
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} from Serper: {resp.text[:300]} "
            f"(key used: {fingerprint})", response=resp)
    organic = resp.json().get("organic", [])
    results = [{"title": r.get("title", ""), "link": r.get("link", ""),
                "snippet": r.get("snippet", "")} for r in organic[:num]]
    if len(_cache) > 256:  # long-running server: don't grow forever
        _cache.clear()
    _cache[query] = results
    return results


def fetch_page(url: str, max_chars: int = 4000) -> str:
    """Fetch a page and return roughly readable text, truncated."""
    if config.MOCK_MODE or url.startswith("mock://"):
        return ("Thread summary: multiple owners report P0171 with high long-term fuel "
                "trims at idle that improve under load; the most common confirmed fixes "
                "were vacuum/intake leaks, followed by MAF cleaning.")
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", resp.text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except requests.RequestException:
        return ""
