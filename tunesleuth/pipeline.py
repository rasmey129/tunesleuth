"""Orchestration: the fixed pipeline plus the generate-critique loop.

Parser -> Analyzer -> Researcher -> Synthesizer -> Critic
The Critic may send the result back to the Synthesizer once. A hard safety
rule escalates any detected knock to a conservative warning no matter what
the synthesis says.
"""
from . import config
from .agents import parser, analyzer
from .agents import workers

KNOCK_WARNING = (
    "Knock events were detected in this log. Regardless of the ranked causes "
    "below, treat this as potentially engine-damaging: avoid high load and "
    "high RPM, confirm fuel quality/octane, and have the tune or engine "
    "inspected before continuing to drive hard."
)


def run(csv_text: str | None = None, obd_code: str | None = None,
        progress=None) -> dict:
    """Run the full diagnosis. `progress` is an optional callback(step, detail)."""
    def report(step, detail=""):
        if progress:
            progress(step, detail)

    trace = {"steps": []}

    # 1. Parse
    report("parser", "reading input")
    parsed = parser.parse(csv_text=csv_text, obd_code=obd_code)
    trace["parsed"] = parsed
    trace["steps"].append("parser")
    if not parsed["usable"]:
        return {"ok": False, "message": parsed["reason"], "trace": trace}

    # 2. Analyze
    report("analyzer", "detecting anomalies")
    analysis = analyzer.analyze(parsed)
    trace["analysis"] = analysis
    trace["steps"].append("analyzer")
    if analysis.get("healthy"):
        return {"ok": True, "healthy": True,
                "message": "No anomalies detected. Trims, AFR, and knock all look normal.",
                "trace": trace}

    # 3. Research
    report("researcher", f"searching: {analysis['queries']}")
    evidence = workers.research(analysis["queries"])
    trace["evidence"] = evidence
    trace["steps"].append("researcher")

    # 4/5. Synthesize with a bounded critique loop
    critic_notes = ""
    diagnoses, verdict = [], {}
    for attempt in range(config.MAX_CRITIC_REVISIONS + 1):
        report("synthesizer", f"attempt {attempt + 1}")
        diagnoses = workers.synthesize(parsed, analysis, evidence, critic_notes)
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
        "confidence": verdict["overall_confidence"],
        "critic_notes": verdict["notes"],
        "safety_warning": None,
        "trace": trace,
    }

    # Hard safety rule: knock always escalates, no exceptions.
    if analysis.get("knock_detected"):
        result["safety_warning"] = KNOCK_WARNING
        result["confidence"] = "low" if not kept else result["confidence"]

    return result
