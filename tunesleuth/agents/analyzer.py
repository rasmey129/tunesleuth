"""Agent 2: Analyzer.

Finds the anomaly in the parsed data using deterministic thresholds, then
uses the LLM to turn the anomaly into good search queries. If the LLM is
unavailable the heuristic queries are used as-is.
"""
import json
from .. import config, llm

SYSTEM = ("You write Google search queries for automotive diagnosis. Given an "
          "anomaly summary, produce 2-3 short queries an experienced mechanic "
          "would type. Return JSON: {\"queries\": [\"...\"]}.")


def _detect_anomalies(parsed: dict) -> list[str]:
    anomalies = []
    stats = parsed.get("stats", {})

    if parsed.get("obd_code"):
        anomalies.append(f"OBD-II trouble code {parsed['obd_code']}")

    ltft = stats.get("ltft")
    if ltft and ltft["mean"] > 10:
        anomalies.append(f"long-term fuel trim elevated (mean {ltft['mean']}%), engine running lean")
    if ltft and ltft["mean"] < -10:
        anomalies.append(f"long-term fuel trim negative (mean {ltft['mean']}%), engine running rich")

    afr_high = stats.get("afr_above_4000rpm")
    if afr_high and afr_high["mean"] > 13.5:
        anomalies.append(f"AFR lean under load (mean {afr_high['mean']} above 4000 RPM)")

    knock = stats.get("knock_events", 0)
    if knock > 0:
        anomalies.append(f"{knock} knock feedback events detected")

    return anomalies


def analyze(parsed: dict) -> dict:
    """Return {anomalies, knock_detected, queries}."""
    anomalies = _detect_anomalies(parsed)
    knock_detected = parsed.get("stats", {}).get("knock_events", 0) > 0

    if not anomalies:
        return {"anomalies": [], "knock_detected": False, "queries": [],
                "healthy": True}

    # Heuristic fallback queries built from the anomaly text
    fallback = [a + " causes" for a in anomalies[:2]]

    payload = llm.complete_json(
        SYSTEM,
        "Anomalies found in the datalog:\n" + "\n".join(f"- {a}" for a in anomalies)
        + "\nStats: " + json.dumps(parsed.get("stats", {})),
        tag="queries",
    )
    queries = payload.get("queries") or fallback
    return {"anomalies": anomalies, "knock_detected": knock_detected,
            "queries": queries[: config.MAX_SEARCHES_PER_RUN], "healthy": False}
