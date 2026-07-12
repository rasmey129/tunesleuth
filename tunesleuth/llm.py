"""Thin wrapper around the LLM API, with a mock fallback for offline demos.

Supports Anthropic and OpenAI. Set LLM_PROVIDER=openai or anthropic in .env;
if unset, whichever API key is present decides.
"""
import json
import logging

from . import config

log = logging.getLogger("tunesleuth")


class LLMUnavailableError(RuntimeError):
    """The LLM provider could not be reached or refused the request.

    The message is written for end users; the pipeline surfaces it verbatim.
    """

_MOCK_RESPONSES = {
    "queries": json.dumps({
        "queries": ["P0171 lean rough idle common causes", "long term fuel trim high vacuum leak"]
    }),
    "synthesis": json.dumps({
        "diagnoses": [
            {"cause": "Vacuum leak after the MAF sensor", "confidence": "high",
             "evidence": "LTFT elevated at idle but normalizes under load, classic unmetered-air signature.",
             "source": "mock://forum-thread-1",
             "check": "With the engine idling, listen for hissing and spray "
                      "soapy water on intake joints — bubbles or an RPM "
                      "change mark the leak.",
             "cost": "$0-40 DIY (hose or clamp); $80-150 for a shop smoke test",
             "difficulty": "diy-easy"},
            {"cause": "Dirty or failing MAF sensor", "confidence": "medium",
             "evidence": "MAF g/s slightly under expected for displacement at idle.",
             "source": "mock://forum-thread-2",
             "check": "Pull the MAF sensor and inspect the wire element for "
                      "dirt or oil film.",
             "cost": "$10-15 DIY (MAF cleaner spray); $150-300 if replacement is needed",
             "difficulty": "diy-easy"}
        ],
        "action_plan": [
            "Check the intake tract for cracked hoses or loose clamps between the MAF sensor and the engine (free, 15 minutes).",
            "Clean the MAF sensor with MAF-specific cleaner spray ($10-15).",
            "Clear the code and drive a few days; if the trims climb back up, have a shop smoke-test the intake ($80-150).",
            "If it returns after that, ask the shop to check fuel pressure before replacing parts."
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
    try:
        resp = client.messages.create(
            model=config.MODEL,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailableError(
            "The AI service rejected the configured API key. If you run this "
            "app, check ANTHROPIC_API_KEY.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailableError(
            "The AI service is rate-limited right now. Wait a minute and try "
            "again.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailableError(
            f"The AI service returned an error ({exc.status_code}). Try again "
            "in a moment.") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailableError(
            "Could not reach the AI service. Check your connection and try "
            "again.") from exc
    return resp.content[0].text


def _complete_openai(system: str, user: str) -> str:
    import openai
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except openai.AuthenticationError as exc:
        raise LLMUnavailableError(
            "The AI service rejected the configured API key. If you run this "
            "app, check OPENAI_API_KEY.") from exc
    except openai.RateLimitError as exc:
        raise LLMUnavailableError(
            "The AI service is rate-limited right now. Wait a minute and try "
            "again.") from exc
    except openai.APIStatusError as exc:
        raise LLMUnavailableError(
            f"The AI service returned an error ({exc.status_code}). Try again "
            "in a moment.") from exc
    except openai.APIConnectionError as exc:
        raise LLMUnavailableError(
            "Could not reach the AI service. Check your connection and try "
            "again.") from exc
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
        log.warning("LLM returned non-JSON for tag=%r: %.200s", tag, raw)
        return {}
