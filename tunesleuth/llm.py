"""Thin wrapper around the LLM API, with a mock fallback for offline demos.

Supports Anthropic and OpenAI. Set LLM_PROVIDER=openai or anthropic in .env;
if unset, whichever API key is present decides.
"""
import json
from . import config

_MOCK_RESPONSES = {
    "queries": json.dumps({
        "queries": ["P0171 lean rough idle common causes", "long term fuel trim high vacuum leak"]
    }),
    "synthesis": json.dumps({
        "diagnoses": [
            {"cause": "Vacuum leak after the MAF sensor", "confidence": "high",
             "evidence": "LTFT elevated at idle but normalizes under load, classic unmetered-air signature.",
             "source": "mock://forum-thread-1"},
            {"cause": "Dirty or failing MAF sensor", "confidence": "medium",
             "evidence": "MAF g/s slightly under expected for displacement at idle.",
             "source": "mock://forum-thread-2"}
        ]
    }),
    "critique": json.dumps({
        "verdict": "accept",
        "kept": [0, 1],
        "dropped": [],
        "notes": "Both claims trace to log statistics and a cited source.",
        "overall_confidence": "medium"
    }),
    "chat_queries": json.dumps({
        "queries": ["2017 Toyota 86 intake manifold leak lean fuel trims",
                    "FA20 high LTFT vacuum leak owners"]
    }),
    "followup": ("Plausible, yes. An intake/manifold leak admits unmetered air, "
                 "which the ECU compensates for with positive fuel trims — and "
                 "this log's elevated LTFT is consistent with that mechanism. "
                 "Owner reports agree: across the threads found, the most "
                 "common confirmed fix for high trims like these was a "
                 "vacuum/intake leak, with MAF cleaning a distant second "
                 "(mock://forum-thread-1, mock://forum-thread-2). The log "
                 "can't confirm the leak's location; a smoke test of the "
                 "intake tract is the usual way to find it. (This is a mock "
                 "response — set an API key for live follow-up chat.)"),
}


def _complete_anthropic(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.MODEL,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _complete_openai(system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def complete(system: str, user: str, tag: str = "") -> str:
    """One LLM call. `tag` selects the canned answer in mock mode."""
    if config.MOCK_MODE:
        return _MOCK_RESPONSES.get(tag, "{}")
    if config.PROVIDER == "openai":
        return _complete_openai(system, user)
    return _complete_anthropic(system, user)


def complete_json(system: str, user: str, tag: str = "") -> dict:
    """LLM call that expects a JSON object back; strips code fences if present."""
    raw = complete(system + "\nRespond with a single JSON object and nothing else.", user, tag)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
