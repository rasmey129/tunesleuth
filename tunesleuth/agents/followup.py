"""Follow-up chat agent.

Answers user questions after a diagnosis, grounded strictly in that
diagnosis: the parsed stats, anomalies, kept causes, and sources. It does
not run new searches or new pipelines, and it says so when a question goes
beyond its evidence. Any knock warning stays pinned in its context.
"""
import json
from .. import llm

SYSTEM = (
    "You are the follow-up assistant for a vehicle diagnosis that has already "
    "been completed and verified. Answer the user's questions using ONLY the "
    "diagnosis context provided. Rules:\n"
    "- Ground every claim in the provided stats, anomalies, causes, or sources.\n"
    "- If the question goes beyond this evidence (new symptoms, other cars, "
    "purchase decisions the diagnosis doesn't settle), say plainly that it is "
    "outside what this diagnosis can answer and suggest a new datalog or a shop.\n"
    "- If a safety warning is present, never downplay it, and repeat it when "
    "the user asks about continuing to drive.\n"
    "- Be concise and plain-spoken. Explain jargon (LTFT, AFR, knock) simply "
    "when asked."
)


def build_context(result: dict) -> str:
    """Distill a pipeline result into the context block the chat runs on."""
    parsed = result.get("trace", {}).get("parsed", {})
    return json.dumps({
        "stats": parsed.get("stats", {}),
        "obd_code": parsed.get("obd_code"),
        "anomalies": result.get("anomalies", []),
        "diagnoses": result.get("diagnoses", []),
        "confidence": result.get("confidence"),
        "safety_warning": result.get("safety_warning"),
    }, indent=1)


def answer(context: str, history: list[dict], question: str) -> str:
    """One chat turn. `history` is a list of {role, content} dicts."""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    user = (f"Diagnosis context:\n{context}\n\n"
            f"Conversation so far:\n{convo}\n\n"
            f"user: {question}")
    return llm.complete(SYSTEM, user, tag="followup")
