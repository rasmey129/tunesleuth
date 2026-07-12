"""NHTSA recall lookup. Free public API, no key required.

Recalls are the highest-value hit for an ordinary owner: "this matches an
open recall — the dealer fixes it free." Cached per vehicle; degrades to
an empty list on any failure instead of breaking the diagnosis.
"""
import logging
import requests

from .. import config

log = logging.getLogger("tunesleuth")

_cache: dict = {}

_MOCK_RECALLS = [
    {"component": "FUEL SYSTEM, GASOLINE (mock)",
     "summary": "Mock recall entry for offline demo: low-pressure fuel pump "
                "impeller may deform and cause the engine to stall while "
                "driving.",
     "campaign": "20V-682", "date": "2020-11-04"},
]


def lookup(year: str, make: str, model: str, limit: int = 5) -> list[dict]:
    """Open recalls for a vehicle: [{component, summary, campaign, date}]."""
    key = (str(year).strip(), make.strip().lower(), model.strip().lower())
    if not all(key):
        return []
    if key in _cache:
        return _cache[key]
    if config.MOCK_MODE:
        _cache[key] = _MOCK_RECALLS
        return _MOCK_RECALLS
    try:
        resp = requests.get(
            "https://api.nhtsa.gov/recalls/recallsByVehicle",
            params={"make": make.strip(), "model": model.strip(),
                    "modelYear": str(year).strip()},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as exc:  # recall lookup is a bonus, never a blocker
        log.warning("NHTSA recall lookup failed for %s: %s", key, exc)
        return []
    recalls = [{
        "component": r.get("Component", ""),
        "summary": (r.get("Summary", "") or "")[:400],
        "campaign": r.get("NHTSACampaignNumber", ""),
        "date": r.get("ReportReceivedDate", ""),
    } for r in results[:limit]]
    if len(_cache) > 128:
        _cache.clear()
    _cache[key] = recalls
    return recalls
