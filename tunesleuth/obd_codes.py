"""Local OBD-II trouble code decoding.

Static code structure (what P/B/C/U + the digits mean) is free and doesn't
need a web search. Model-specific meaning and root causes still come from
live research in the Researcher/Synthesizer. This module only answers
"what does this code generically mean," never "why did my car throw it."

Two layers:
- COMMON_CODES: a lookup table of well-known codes with a plain description.
- SAE structural fallback: decodes any well-formed code (first letter +
  4 digits) into system/generic-vs-manufacturer/subsystem, per the SAE J2012
  numbering convention, so an unlisted code still gets a meaningful label.
"""

COMMON_CODES = {
    "P0100": "Mass or volume air flow circuit malfunction",
    "P0101": "Mass or volume air flow circuit range/performance problem",
    "P0102": "Mass or volume air flow circuit low input",
    "P0113": "Intake air temperature circuit high input",
    "P0128": "Coolant thermostat below regulating temperature",
    "P0130": "O2 sensor circuit malfunction (bank 1, sensor 1)",
    "P0133": "O2 sensor circuit slow response (bank 1, sensor 1)",
    "P0155": "O2 sensor heater circuit malfunction (bank 2, sensor 1)",
    "P0170": "Fuel trim malfunction (bank 1)",
    "P0171": "System too lean (bank 1)",
    "P0172": "System too rich (bank 1)",
    "P0174": "System too lean (bank 2)",
    "P0175": "System too rich (bank 2)",
    "P0201": "Injector circuit malfunction, cylinder 1",
    "P0217": "Engine over temperature condition",
    "P0230": "Fuel pump primary circuit malfunction",
    "P0300": "Random/multiple cylinder misfire detected",
    "P0301": "Cylinder 1 misfire detected",
    "P0302": "Cylinder 2 misfire detected",
    "P0303": "Cylinder 3 misfire detected",
    "P0304": "Cylinder 4 misfire detected",
    "P0325": "Knock sensor 1 circuit malfunction (bank 1)",
    "P0327": "Knock sensor 1 circuit low input (bank 1)",
    "P0335": "Crankshaft position sensor circuit malfunction",
    "P0340": "Camshaft position sensor circuit malfunction",
    "P0401": "Exhaust gas recirculation flow insufficient",
    "P0420": "Catalyst system efficiency below threshold (bank 1)",
    "P0430": "Catalyst system efficiency below threshold (bank 2)",
    "P0442": "Evaporative emission system leak detected (small leak)",
    "P0455": "Evaporative emission system leak detected (large leak)",
    "P0500": "Vehicle speed sensor malfunction",
    "P0505": "Idle control system malfunction",
    "P0507": "Idle control system RPM higher than expected",
    "P0562": "System voltage low",
    "P0606": "ECM/PCM processor malfunction",
    "P0700": "Transmission control system malfunction",
}

_SYSTEM_BY_LETTER = {
    "P": "Powertrain (engine/transmission)",
    "B": "Body",
    "C": "Chassis",
    "U": "Network/communication",
}

# SAE J2012: second character of a P-code splits generic (0) vs
# manufacturer-specific (1); third digit groups the subsystem.
_P_SUBSYSTEM_BY_DIGIT = {
    "0": "Fuel and air metering / auxiliary emission controls",
    "1": "Fuel and air metering",
    "2": "Fuel and air metering (injector circuit)",
    "3": "Ignition system or misfire",
    "4": "Auxiliary emission controls",
    "5": "Vehicle speed, idle control, and auxiliary inputs",
    "6": "Computer and output circuits",
    "7": "Transmission",
    "8": "Transmission",
}


def decode(code: str) -> str:
    """Return a plain-language description for a well-formed OBD-II code.

    Falls back to the SAE structural meaning if the exact code isn't in the
    lookup table. Caller is expected to have already validated the code's
    format (see `parser.OBD_CODE_RE`).
    """
    code = code.strip().upper()
    if code in COMMON_CODES:
        return COMMON_CODES[code]

    letter, digits = code[0], code[1:]
    system = _SYSTEM_BY_LETTER.get(letter, "Unknown system")
    origin = "generic (SAE-defined)" if digits[0] == "0" else "manufacturer-specific"

    if letter == "P" and digits[1] in _P_SUBSYSTEM_BY_DIGIT:
        subsystem = _P_SUBSYSTEM_BY_DIGIT[digits[1]]
        return f"{system}, {origin} code — {subsystem}"
    return f"{system}, {origin} code"
