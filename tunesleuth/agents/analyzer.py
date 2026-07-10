"""Agent 2: Analyzer.

Finds the anomaly in the parsed data using deterministic thresholds, then
uses the LLM to turn the anomaly into good search queries. If the LLM is
unavailable the heuristic queries are used as-is.
"""
import json
from .. import config, llm

SYSTEM = ("You write Google search queries for automotive diagnosis. Given an "
          "anomaly summary, produce 2-3 short queries an experienced mechanic "
          "would type. If a vehicle is given, work its year/make/model/engine "
          "into the queries so results are model-specific. Return JSON: "
          '{"queries": ["..."]}.')


# Below this oil/coolant temperature (F) the engine never reached operating
# temperature, so trims and AFR include open-loop warmup enrichment and must
# not be read as lean/rich faults. Warmed-up logs from the same car run
# 183-212F; coldstart.csv peaks at 158F.
WARMUP_TEMP_F = 170

# Detection thresholds, calibrated against data/ and data/real/ so no
# healthy or known-cause log false-positives. The worst benign readings
# observed: STFT mean 4.0, AFR-target divergence 1.48, oil max 212F,
# IAT max 131F. Tuned WOT targets run ~11-12 AFR, so rich-under-load
# sits below that.
LTFT_LIMIT = 10          # % long-term trim beyond which fueling is off
STFT_LIMIT = 8           # % sustained short-term correction
TOTAL_TRIM_LIMIT = 15    # % combined STFT+LTFT
AFR_LEAN_LOAD = 13.5     # AFR above 4000 RPM leaner than this
AFR_RICH_LOAD = 10.5     # AFR above 4000 RPM richer than this
AFR_TARGET_LEAN = 1.5    # AFR points leaner than commanded under load
AFR_TARGET_RICH = -2.0   # AFR points richer than commanded under load
OIL_OVERHEAT_F = 240
COOLANT_OVERHEAT_F = 230
IAT_HEAT_SOAK_F = 150


def _in_warmup(stats: dict) -> bool:
    temp = stats.get("oil_temp") or stats.get("coolant")
    return bool(temp and temp["max"] < WARMUP_TEMP_F)


def _detect_anomalies(parsed: dict, suppress_fueling: bool = False) -> list[str]:
    anomalies = []
    stats = parsed.get("stats", {})

    if parsed.get("obd_code"):
        meaning = parsed.get("obd_meaning")
        if meaning:
            anomalies.append(f"OBD-II trouble code {parsed['obd_code']} ({meaning})")
        else:
            anomalies.append(f"OBD-II trouble code {parsed['obd_code']}")

    if not suppress_fueling:
        ltft = stats.get("ltft")
        stft = stats.get("stft")
        ltft_flagged = stft_flagged = False

        if ltft and ltft["mean"] > LTFT_LIMIT:
            ltft_flagged = True
            anomalies.append(f"long-term fuel trim elevated (mean {ltft['mean']}%), engine running lean")
        if ltft and ltft["mean"] < -LTFT_LIMIT:
            ltft_flagged = True
            anomalies.append(f"long-term fuel trim negative (mean {ltft['mean']}%), engine running rich")

        if stft and abs(stft["mean"]) > STFT_LIMIT:
            stft_flagged = True
            direction = "lean" if stft["mean"] > 0 else "rich"
            anomalies.append(
                f"short-term fuel trim sustained at {stft['mean']}% — the ECU is "
                f"actively correcting {direction} right now (recent or intermittent issue)")

        # Moderate STFT and LTFT can add up to a real problem without
        # either tripping its own limit.
        if stft and ltft and not (ltft_flagged or stft_flagged):
            total = round(stft["mean"] + ltft["mean"], 2)
            if abs(total) > TOTAL_TRIM_LIMIT:
                direction = "lean" if total > 0 else "rich"
                anomalies.append(
                    f"combined fuel trims at {total}% (STFT {stft['mean']}% + "
                    f"LTFT {ltft['mean']}%), engine running {direction}")

        afr_high = stats.get("afr_above_4000rpm")
        if afr_high and afr_high["mean"] > AFR_LEAN_LOAD:
            anomalies.append(f"AFR lean under load (mean {afr_high['mean']} above 4000 RPM)")
        if afr_high and afr_high["mean"] < AFR_RICH_LOAD:
            anomalies.append(
                f"AFR rich under load (mean {afr_high['mean']} above 4000 RPM) — "
                "over-fueling wastes power and washes cylinder walls")

        afr_vs_target = stats.get("afr_minus_target_above_4000rpm")
        if afr_vs_target and afr_vs_target["mean"] > AFR_TARGET_LEAN:
            anomalies.append(
                f"AFR runs {afr_vs_target['mean']} points leaner than commanded above "
                "4000 RPM — fuel delivery cannot keep up with the target (pump, "
                "injectors, or fuel pressure)")
        if afr_vs_target and afr_vs_target["mean"] < AFR_TARGET_RICH:
            anomalies.append(
                f"AFR runs {abs(afr_vs_target['mean'])} points richer than commanded "
                "above 4000 RPM — over-delivering fuel (injector, regulator, or "
                "sensor scaling)")

    oil = stats.get("oil_temp")
    if oil and oil["max"] > OIL_OVERHEAT_F:
        anomalies.append(f"oil temperature reached {oil['max']}F — overheating")
    coolant = stats.get("coolant")
    if coolant and coolant["max"] > COOLANT_OVERHEAT_F:
        anomalies.append(f"coolant temperature reached {coolant['max']}F — overheating")
    iat = stats.get("iat")
    if iat and iat["max"] > IAT_HEAT_SOAK_F:
        anomalies.append(
            f"intake air temperature reached {iat['max']}F — heat soak; expect "
            "timing pull and reduced power until intake temps drop")

    knock = stats.get("knock_events", 0)
    if knock > 0:
        anomalies.append(f"{knock} knock feedback events detected")

    return anomalies


def analyze(parsed: dict, vehicle: str | None = None) -> dict:
    """Return {anomalies, knock_detected, queries, warmup_note}."""
    stats = parsed.get("stats", {})
    warmup = _in_warmup(stats)
    anomalies = _detect_anomalies(parsed, suppress_fueling=warmup)
    knock_detected = stats.get("knock_events", 0) > 0

    warmup_note = None
    if warmup:
        temp = stats.get("oil_temp") or stats.get("coolant")
        warmup_note = (
            f"The engine never reached operating temperature in this log "
            f"(oil/coolant peaked at {temp['max']}F). Fuel trims and AFR during "
            "warmup include open-loop enrichment, so lean/rich checks were "
            "skipped. Re-log once warm to evaluate fueling.")

    if not anomalies:
        return {"anomalies": [], "knock_detected": False, "queries": [],
                "healthy": True, "warmup_note": warmup_note}

    # Heuristic fallback queries built from the anomaly text
    vehicle_suffix = f" {vehicle}" if vehicle else ""
    fallback = [a + " causes" + vehicle_suffix for a in anomalies[:2]]

    user = ("Anomalies found in the datalog:\n" + "\n".join(f"- {a}" for a in anomalies)
            + "\nStats: " + json.dumps(parsed.get("stats", {})))
    if vehicle:
        user += f"\nVehicle: {vehicle}"

    payload = llm.complete_json(SYSTEM, user, tag="queries")
    queries = payload.get("queries") or fallback
    return {"anomalies": anomalies, "knock_detected": knock_detected,
            "queries": queries[: config.MAX_SEARCHES_PER_RUN], "healthy": False,
            "warmup_note": warmup_note}
