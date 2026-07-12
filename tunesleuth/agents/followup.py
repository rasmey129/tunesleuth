"""Follow-up chat agent.

Answers user questions after a diagnosis with two-tier grounding: facts
about this car's log come only from the diagnosis context (stats,
anomalies, kept causes, sources); reasoning about what commonly causes
the diagnosed condition is allowed but labeled as hypothesis. When the
question is one other owners may have reported ("could an intake leak
cause this?"), it runs a small web search (capped per question) and
identifies the consensus across sources. Any knock warning stays pinned
in its context.
"""
import json
from .. import config, llm
from . import workers

SYSTEM = (
    "You are the follow-up assistant for a vehicle diagnosis that has already "
    "been completed and verified. Rules:\n"
    "- Facts about THIS car's log and diagnosis come ONLY from the provided "
    "context (stats, anomalies, ranked causes, sources, warnings). Never "
    "invent readings or claim the log shows something it does not.\n"
    "- You MAY reason about plausible causes and mechanisms for the diagnosed "
    "conditions — e.g. whether an intake leak could explain a lean condition — "
    "including failure points common to the specific vehicle when one is "
    "given. Present this as general knowledge, not as a log finding: say "
    "whether the log evidence is consistent with it, and suggest a concrete "
    "way to confirm (smoke test, targeted re-log, inspection).\n"
    "- When web evidence is provided, identify the consensus: what do most "
    "reports agree caused or fixed this, on this platform if the sources are "
    "model-specific? Note real disagreement instead of papering over it, "
    "cite the links you lean on, and keep community reports clearly separate "
    "from what this car's log shows.\n"
    "- Questions unrelated to this diagnosis (other cars, new symptoms not in "
    "the log, decisions the diagnosis doesn't settle) are out of scope: say "
    "so plainly and suggest a new datalog or a shop.\n"
    "- If a safety warning or severity assessment is present, never downplay "
    "it, and repeat it when the user asks about continuing to drive.\n"
    "- Be concise and plain-spoken. Explain jargon (LTFT, AFR, knock) simply "
    "when asked."
)

QUERY_SYSTEM = (
    "Decide whether a web search would help answer a follow-up question about "
    "a completed vehicle diagnosis. Search when the question asks about a "
    "potential cause, fix, or whether others have experienced the issue — the "
    "goal is finding owner reports to establish a consensus. Do not search "
    "for definitions, restatements of the diagnosis, or out-of-scope topics. "
    "Work the vehicle and the diagnosed condition into each query. Return "
    'JSON: {"queries": ["..."]} with at most 2 short queries, or an empty '
    "list if no search is needed."
)


def build_context(result: dict) -> str:
    """Distill a pipeline result into the context block the chat runs on."""
    parsed = result.get("trace", {}).get("parsed", {})
    return json.dumps({
        "vehicle": result.get("vehicle"),
        "symptoms": result.get("symptoms"),
        "stats": parsed.get("stats", {}),
        "obd_codes": parsed.get("obd_codes", []),
        "anomalies": result.get("anomalies", []),
        "diagnoses": result.get("diagnoses", []),
        "action_plan": result.get("action_plan", []),
        "confidence": result.get("confidence"),
        "severity": result.get("severity"),
        "recalls": result.get("recalls", []),
        "safety_warning": result.get("safety_warning"),
        "sensor_warnings": result.get("sensor_warnings", []),
        "warmup_note": result.get("warmup_note"),
    }, indent=1)


def _research(context: str, question: str) -> list[dict]:
    """Search for owner reports relevant to the question, if it warrants one."""
    payload = llm.complete_json(
        QUERY_SYSTEM,
        f"Diagnosis context:\n{context}\n\nFollow-up question: {question}",
        tag="chat_queries",
    )
    queries = payload.get("queries") or []
    if not queries:
        return []
    return workers.research(queries[: config.MAX_CHAT_SEARCHES])


def answer(context: str, history: list[dict], question: str) -> str:
    """One chat turn. `history` is a list of {role, content} dicts."""
    evidence = _research(context, question)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    user = (f"Diagnosis context:\n{context}\n\n"
            f"Conversation so far:\n{convo}\n\n")
    if evidence:
        user += "Web evidence for this question:\n" + json.dumps(
            [{k: e.get(k, "") for k in ("title", "link", "snippet")} for e in evidence]) + "\n\n"
    user += f"user: {question}"
    return llm.complete(SYSTEM, user, tag="followup")
