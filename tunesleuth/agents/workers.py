"""Agents 3-5: Web Researcher, Synthesizer, Critic."""
import json
from .. import config, llm
from ..tools import serp


# ---------------------------------------------------------------- Researcher
def research(queries: list[str]) -> list[dict]:
    """Search each query, fetch top pages, return source-tagged evidence."""
    evidence = []
    pages_fetched = 0
    for q in queries[: config.MAX_SEARCHES_PER_RUN]:
        try:
            results = serp.search(q)
        except Exception as exc:  # tool failure: degrade, don't die
            evidence.append({"query": q, "title": "search failed",
                             "link": "", "snippet": str(exc), "page_text": ""})
            continue
        for r in results[:2]:
            page_text = ""
            if pages_fetched < config.MAX_PAGES_FETCHED:
                page_text = serp.fetch_page(r["link"])
                pages_fetched += 1
            evidence.append({"query": q, **r, "page_text": page_text})
    return evidence


# ---------------------------------------------------------------- Synthesizer
SYNTH_SYSTEM = (
    "You are an automotive diagnostic assistant. Combine the datalog statistics "
    "with the web evidence into a ranked list of probable causes. Every cause "
    "must cite either a specific log statistic or one of the provided source "
    "links. Do not invent causes the evidence does not support. Return JSON: "
    '{"diagnoses": [{"cause", "confidence" (high/medium/low), "evidence", "source"}]}'
)


def synthesize(parsed: dict, analysis: dict, evidence: list[dict],
               critic_notes: str = "") -> list[dict]:
    user = (
        "Datalog stats: " + json.dumps(parsed.get("stats", {})) + "\n"
        "OBD code: " + str(parsed.get("obd_code")) + "\n"
        "Anomalies: " + json.dumps(analysis.get("anomalies", [])) + "\n"
        "Web evidence: " + json.dumps(
            [{k: e[k] for k in ("title", "link", "snippet")} for e in evidence]) + "\n"
    )
    if critic_notes:
        user += f"\nA reviewer rejected the previous draft: {critic_notes}\nRevise accordingly."
    payload = llm.complete_json(SYNTH_SYSTEM, user, tag="synthesis")
    return payload.get("diagnoses", [])


# ---------------------------------------------------------------- Critic
CRITIC_SYSTEM = (
    "You are a skeptical reviewer of automotive diagnoses. For each candidate "
    "diagnosis, check whether its evidence actually traces to the provided log "
    "stats or a provided source link. Reject anything unsupported or unsafe. "
    'Return JSON: {"verdict": "accept"|"revise", "kept": [indices], '
    '"dropped": [indices], "notes": "...", "overall_confidence": "high"|"medium"|"low"}'
)


def critique(parsed: dict, diagnoses: list[dict], evidence: list[dict]) -> dict:
    user = (
        "Log stats: " + json.dumps(parsed.get("stats", {})) + "\n"
        "Available source links: " + json.dumps([e["link"] for e in evidence]) + "\n"
        "Candidate diagnoses: " + json.dumps(diagnoses)
    )
    verdict = llm.complete_json(CRITIC_SYSTEM, user, tag="critique")
    # Defensive defaults if the LLM returns malformed JSON
    verdict.setdefault("verdict", "accept")
    verdict.setdefault("kept", list(range(len(diagnoses))))
    verdict.setdefault("dropped", [])
    verdict.setdefault("notes", "")
    verdict.setdefault("overall_confidence", "low")
    return verdict
