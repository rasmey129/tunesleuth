"""Agent 1: Input Parser.

Deterministic (no LLM). Reads a datalog CSV and/or an OBD-II code, validates
the columns, computes summary statistics, and reports unusable data instead
of guessing.

Handles plain CSVs and OpenFlash Tablet exports, which prepend a preamble
("Procede Data Log" / "OpenFlash Data File 1" / channel count) before the
real header row.
"""
import io
import re
import pandas as pd

from .. import obd_codes

# Channel names we recognize, mapped from the many aliases logging tools use.
# Order within a list matters: earlier aliases are preferred.
CHANNEL_ALIASES = {
    "rpm": ["rpm", "engine speed"],
    "afr": ["wideband afr", "air fuel ratio", "afr", "a/f ratio"],
    "afr_target": ["command afr", "commanded afr", "afr target", "target afr", "eq target"],
    "stft": ["short term fuel trim", "stft"],
    "ltft": ["long term fuel trim", "ltft"],
    "maf": ["maf g/s", "mass airflow", "mass air flow", "maf"],
    "load": ["engine load", "calculated load", "load"],
    "timing": ["ignition adv", "ignition timing", "spark advance", "ign timing", "timing"],
    "knock": ["flkc", "fbkc", "knock retard", "knock feedback", "knock count", "knock"],
    "iat": ["intake air temp", "intake temp", "manifold air temp", "iat"],
    "throttle": ["throttle", "tps", "accelerator"],
    "oil_temp": ["oil temp"],
}

# Columns that must never satisfy these channels (voltage is not a flow rate)
EXCLUDE = {"maf": ["volt"], "afr": ["sensor v", "volt"]}

OBD_CODE_RE = re.compile(r"^[PBCU]\d{4}$", re.IGNORECASE)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _find_header_row(lines: list[str]) -> int:
    """Locate the real header row, skipping tool preambles (OpenFlash etc.).

    The header is the first line whose comma-separated fields are mostly
    non-numeric names and which contains at least 3 fields.
    """
    for i, line in enumerate(lines[:10]):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 3:
            continue
        non_numeric = sum(1 for f in fields if f and not _is_number(f))
        if non_numeric >= len(fields) * 0.7:
            return i
    return 0


def _match_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    lowered = {c.lower().strip(): c for c in df.columns}
    for channel, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:  # alias priority order
            for col_lower, col in lowered.items():
                if alias in col_lower and channel not in mapping:
                    if any(bad in col_lower for bad in EXCLUDE.get(channel, [])):
                        continue
                    mapping[channel] = col
            if channel in mapping:
                break
    return mapping


def parse(csv_text: str | None = None, obd_code: str | None = None) -> dict:
    """Parse the user's input. Returns a dict the Analyzer can consume.

    Never raises on bad data; sets `usable` to False with a reason instead.
    """
    result = {"usable": False, "reason": "", "obd_code": None, "obd_meaning": None,
              "channels": {}, "stats": {}, "rows": 0, "format": "csv"}

    if obd_code:
        code = obd_code.strip().upper()
        if OBD_CODE_RE.match(code):
            result["obd_code"] = code
            result["obd_meaning"] = obd_codes.decode(code)
        else:
            result["reason"] = f"'{obd_code}' does not look like an OBD-II code (e.g. P0171)."
            if not csv_text:
                return result

    if not csv_text:
        result["usable"] = result["obd_code"] is not None
        if not result["usable"] and not result["reason"]:
            result["reason"] = "No datalog or trouble code provided."
        return result

    lines = csv_text.splitlines()
    header_idx = _find_header_row(lines)
    if header_idx > 0:
        result["format"] = "openflash" if any(
            "openflash" in ln.lower() or "procede" in ln.lower()
            for ln in lines[:header_idx]) else "preamble-csv"

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    except Exception as exc:
        result["reason"] = f"Could not read CSV: {exc}"
        return result

    if df.empty or len(df.columns) < 2:
        result["reason"] = "CSV is empty or has too few columns to analyze."
        return result

    mapping = _match_columns(df)
    if not mapping:
        result["reason"] = ("No recognizable channels found. Expected columns like "
                            "RPM, AFR, STFT, LTFT, MAF, timing, or knock/FLKC.")
        return result

    stats = {}
    for channel, col in mapping.items():
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[channel] = {
            "mean": round(float(series.mean()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }

    if not stats:
        result["reason"] = "Recognized columns contained no numeric data."
        return result

    # Derived stats the Analyzer relies on
    if "afr" in mapping and "rpm" in mapping:
        afr = pd.to_numeric(df[mapping["afr"]], errors="coerce")
        rpm = pd.to_numeric(df[mapping["rpm"]], errors="coerce")
        high = afr[rpm > 4000].dropna()
        if not high.empty:
            stats["afr_above_4000rpm"] = {"mean": round(float(high.mean()), 2),
                                          "max": round(float(high.max()), 2)}
        # Divergence from commanded AFR under load, if the target is logged
        if "afr_target" in mapping:
            target = pd.to_numeric(df[mapping["afr_target"]], errors="coerce")
            diff = (afr - target)[rpm > 4000].dropna()
            if not diff.empty:
                stats["afr_minus_target_above_4000rpm"] = {
                    "mean": round(float(diff.mean()), 2),
                    "max": round(float(diff.max()), 2)}

    if "knock" in mapping:
        knock = pd.to_numeric(df[mapping["knock"]], errors="coerce").dropna()
        # FLKC-style channels report negative timing correction on knock;
        # count any nonzero excursion as an event.
        stats["knock_events"] = int((knock.abs() > 0).sum())
        if stats["knock_events"]:
            stats["knock_worst"] = round(float(knock.abs().max()), 2)

    result.update({"usable": True, "channels": mapping, "stats": stats,
                   "rows": len(df)})
    return result
