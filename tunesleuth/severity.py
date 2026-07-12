"""Deterministic severity triage: answers "can I keep driving?".

Enforced in code, not in a prompt — same philosophy as the knock rule.
The analyzer tags every finding with a level as it detects it (so warmup
suppression naturally suppresses the severity too); the pipeline reports
the worst level found.

Levels, ascending urgency: low < soon < caution < stop.
"""

LEVEL_ORDER = {"low": 0, "soon": 1, "caution": 2, "stop": 3}

DISPLAY = {
    "stop": ("Stop driving",
             "Address this before driving further. Continuing to drive risks "
             "engine or catalytic converter damage — if the check-engine "
             "light is flashing, pull over when safe."),
    "caution": ("Drive gently, fix promptly",
                "Avoid high RPM and hard acceleration until this is resolved. "
                "The condition is most dangerous under load."),
    "soon": ("Fix soon",
             "Safe to drive normally for now, but have this looked at in the "
             "next few weeks — small problems here tend to grow."),
    "low": ("Low urgency",
            "Fix at your convenience. This mostly affects emissions or fuel "
            "economy, not the engine's health."),
}

# Exact-code overrides, checked before the structural rules below.
_CODE_SEVERITY = {
    "P0217": "stop",     # engine over temperature
    "P0170": "caution", "P0171": "caution", "P0172": "caution",
    "P0174": "caution", "P0175": "caution",
    "P0325": "caution", "P0327": "caution",  # knock protection compromised
    "P0420": "low", "P0430": "low",          # catalyst efficiency
    "P0442": "low", "P0455": "low", "P0456": "low",  # EVAP, often gas cap
    "P0128": "low",      # thermostat below regulating temperature
}


def code_severity(code: str) -> str:
    """Severity tier for a single OBD-II code."""
    code = code.strip().upper()
    if code in _CODE_SEVERITY:
        return _CODE_SEVERITY[code]
    if code.startswith("P03") and code[1:].isdigit():
        n = int(code[1:])
        if 300 <= n <= 316:
            return "stop"      # misfires can destroy the catalytic converter
        if 325 <= n <= 334:
            return "caution"   # knock sensor circuit: knock protection is off
    return "soon"


def worst(levels) -> str | None:
    """The most urgent level in an iterable, or None if empty."""
    levels = [lv for lv in levels if lv in LEVEL_ORDER]
    if not levels:
        return None
    return max(levels, key=LEVEL_ORDER.__getitem__)


def describe(level: str | None) -> dict | None:
    """UI-ready {level, label, advice} for a level, or None."""
    if level not in DISPLAY:
        return None
    label, advice = DISPLAY[level]
    return {"level": level, "label": label, "advice": advice}
