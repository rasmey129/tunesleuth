"""Orchestration: the fixed pipeline plus the generate-critique loop.

Parser -> Analyzer -> [Recalls] -> Researcher -> Synthesizer -> Critic
The Critic may send the result back to the Synthesizer once. Two hard
safety rules live in code, not prompts: any detected knock escalates to a
conservative warning, and every finding carries a deterministic severity
tier ("can I keep driving?").
"""
import logging

from . import config, llm, severity
from .agents import parser, analyzer
from .agents import workers
from .tools import recalls as recalls_tool

log = logging.getLogger("tunesleuth")

KNOCK_WARNING = (
    "Knock events were detected in this log. Regardless of the ranked causes "
    "below, treat this as potentially engine-damaging: avoid high load and "
    "high RPM, confirm fuel quality/octane, and have the tune or engine "
    "inspected before continuing to drive hard."
)


def run(csv_text: str | None = None, obd_code: str | None = None,
        vehicle: str | None = None, year: str | None = None,
        make: str | None = None, model: str | None = None,
        symptoms: str | None = None, progress=None) -> dict:
    """Run the full diagnosis. `progress` is an optional callback(step, detail).

    Never raises: any failure comes back as {"ok": False, "message": ...}
    with a message written for the end user.

    `obd_code` accepts one code or several ("P0171, P0300"). `vehicle` is a
    free-text description used in prompts; `year`/`make`/`model` (all three
    required together) additionally enable the NHTSA recall lookup.
    `symptoms` is optional owner-reported context ("rough idle when cold");
    it enriches research and synthesis but never creates findings by itself.
    """
    trace = {"steps": []}
    try:
        return _run(csv_text, obd_code, vehicle, year, make, model,
                    symptoms, progress, trace)
    except llm.LLMUnavailableError as exc:
        log.warning("diagnosis aborted: %s", exc)
        return {"ok": False, "message": str(exc), "trace": trace}
    except Exception:
        log.exception("diagnosis failed unexpectedly")
        return {"ok": False,
                "message": ("Something went wrong during the diagnosis. Try "
                            "again; if it keeps happening, the log may be in "
                            "a format we don't handle yet."),
                "trace": trace}


def _run(csv_text, obd_code, vehicle, year, make, model,
         symptoms, progress, trace) -> dict:
    vehicle = vehicle.strip() if vehicle else None
    symptoms = symptoms.strip() if symptoms else None

    def report(step, detail=""):
        if progress:
            progress(step, detail)

    # 1. Parse
    report("parser", "reading input")
    parsed = parser.parse(csv_text=csv_text, obd_code=obd_code)
    preview = parsed.pop("preview", None)  # chart data; keep the trace readable
    trace["parsed"] = parsed
    trace["steps"].append("parser")
    if not parsed["usable"]:
        return {"ok": False, "message": parsed["reason"], "trace": trace}

    # 2. Analyze
    report("analyzer", "detecting anomalies")
    analysis = analyzer.analyze(parsed, vehicle=vehicle, symptoms=symptoms)
    trace["analysis"] = analysis
    trace["steps"].append("analyzer")

    # 2b. Recalls (free NHTSA lookup; useful even when the log is healthy)
    recalls = []
    if year and make and model:
        report("recalls", f"checking NHTSA recalls for {year} {make} {model}")
        recalls = recalls_tool.lookup(year, make, model)
        trace["recalls"] = recalls
        trace["steps"].append("recalls")

    base = {
        "sensor_warnings": parsed.get("sensor_warnings", []),
        "notes": parsed.get("notes", []),
        "warmup_note": analysis.get("warmup_note"),
        "severity": severity.describe(analysis.get("severity")),
        "recalls": recalls,
        "symptoms": symptoms,
        "preview": preview,
        "trace": trace,
        "vehicle": vehicle,
    }

    if analysis.get("healthy"):
        message = "No anomalies detected."
        if analysis.get("warmup_note"):
            message += " (Fueling checks were skipped: see the warmup note.)"
        else:
            message += " Trims, AFR, and knock all look normal."
        if symptoms:
            message += (" You reported symptoms, but this data doesn't show "
                        "them — try capturing a log while the symptom is "
                        "actually happening.")
        return {"ok": True, "healthy": True, "message": message, **base}

    # 3. Research
    report("researcher", f"searching: {analysis['queries']}")
    evidence = workers.research(analysis["queries"])
    trace["evidence"] = evidence
    trace["steps"].append("researcher")

    # 4/5. Synthesize with a bounded critique loop
    critic_notes = ""
    diagnoses, action_plan, verdict = [], [], {}
    for attempt in range(config.MAX_CRITIC_REVISIONS + 1):
        report("synthesizer", f"attempt {attempt + 1}")
        synthesis = workers.synthesize(parsed, analysis, evidence, critic_notes,
                                       vehicle=vehicle, symptoms=symptoms,
                                       recalls=recalls)
        diagnoses = synthesis["diagnoses"]
        action_plan = synthesis["action_plan"]
        report("critic", "verifying claims")
        verdict = workers.critique(parsed, diagnoses, evidence)
        trace["steps"].append(f"synthesizer/critic pass {attempt + 1}")
        if verdict["verdict"] == "accept":
            break
        critic_notes = verdict["notes"]

    kept = [diagnoses[i] for i in verdict["kept"] if i < len(diagnoses)]
    result = {
        "ok": True,
        "healthy": False,
        "anomalies": analysis["anomalies"],
        "diagnoses": kept,
        "action_plan": action_plan,
        "confidence": verdict["overall_confidence"],
        "critic_notes": verdict["notes"],
        "safety_warning": None,
        **base,
    }

    # Hard safety rule: knock always escalates, no exceptions.
    if analysis.get("knock_detected"):
        result["safety_warning"] = KNOCK_WARNING
        result["confidence"] = "low" if not kept else result["confidence"]

    return result
