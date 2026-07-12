"""TuneSleuth front-end. Run with: streamlit run app.py"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tunesleuth import pipeline, config, llm
from tunesleuth.agents import followup

st.set_page_config(
    page_title="TuneSleuth", page_icon="🔧", layout="centered",
    menu_items={"About": (
        "TuneSleuth diagnoses vehicle datalogs and OBD-II trouble codes with "
        "a multi-agent pipeline: deterministic log analysis, web research, "
        "and a verification step that checks every claimed cause against "
        "evidence.")})

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {
    "Lean condition": "sample_lean.csv",
    "Knock event": "sample_knock.csv",
    "Rich condition": "sample_rich.csv",
    "Overheating": "sample_overheat.csv",
    "Healthy engine": "sample_healthy.csv",
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Chart colors: categorical slot-1 blue for every single-series line panel
# (identity is carried by each panel's title, not by hue), status-critical
# red reserved for knock event marks. Values from the validated reference
# palette; never reuse the red for an ordinary series.
SERIES_BLUE = "#2a78d6"
CRITICAL_RED = "#d03b3b"
CHART_WIDTH = 600

CONFIDENCE_BADGE = {"high": "green", "medium": "orange", "low": "gray"}


def log_chart(preview: dict) -> alt.VConcatChart | None:
    """Small multiples of the headline channels on a shared time axis.

    One panel per channel (single series each, titled by its y-axis — no
    legend needed); knock events are full-height red rules in their own
    strip. Scales differ wildly (RPM vs AFR), so panels, never dual axes.
    """
    channels = preview["channels"]
    df = pd.DataFrame(channels)
    x_title = "time (s)" if preview["time_is_seconds"] else "sample"

    specs = [("rpm", "RPM"), ("afr", "AFR"), ("ltft", "LTFT %")]
    present = [(c, t) for c, t in specs if c in df.columns and df[c].notna().any()]

    knock_events = None
    if "knock" in df.columns:
        k = df[df["knock"].abs() > 0]
        if not k.empty:
            knock_events = k

    if not present and knock_events is None:
        return None

    x_inner = alt.X("time:Q", axis=alt.Axis(labels=False, ticks=False,
                                            domain=False, title=None))
    x_bottom = alt.X("time:Q", title=x_title)

    panels = []
    for i, (col, title) in enumerate(present):
        bottom = (i == len(present) - 1) and knock_events is None
        base = alt.Chart(df).encode(x=x_bottom if bottom else x_inner)
        line = base.mark_line(strokeWidth=2, color=SERIES_BLUE).encode(
            y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)))
        # invisible points give the line a hover target bigger than the mark
        hover = base.mark_point(size=100, opacity=0).encode(
            y=alt.Y(f"{col}:Q"),
            tooltip=[alt.Tooltip("time:Q", title=x_title),
                     alt.Tooltip(f"{col}:Q", title=title)])
        panels.append((line + hover).properties(height=110, width=CHART_WIDTH))

    if knock_events is not None:
        rules = alt.Chart(knock_events).mark_rule(
            color=CRITICAL_RED, strokeWidth=2).encode(
            x=x_bottom,
            tooltip=[alt.Tooltip("time:Q", title=x_title),
                     alt.Tooltip("knock:Q", title="knock feedback")])
        panels.append(rules.properties(
            height=48, width=CHART_WIDTH,
            title=alt.TitleParams(f"knock events ({len(knock_events)})",
                                  fontSize=12, anchor="start")))

    return alt.vconcat(*panels, spacing=6).resolve_scale(x="shared")


st.title("TuneSleuth")
st.caption("Multi-agent diagnosis for vehicle datalogs and trouble codes")

if config.MOCK_MODE:
    st.info("Running in mock mode (no API keys found). Set ANTHROPIC_API_KEY "
            "and SERPER_API_KEY in a .env file for live diagnosis.")

with st.form("diagnose"):
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader(
            "Datalog CSV", type=["csv"],
            help="Any CSV with channels like RPM, AFR, fuel trims, or knock. "
                 "OpenFlash Tablet and consumer OBD app exports (Torque, OBD "
                 "Fusion, Car Scanner) work, and so does a freeze-frame "
                 "snapshot.")
        sample = st.selectbox("...or try a sample log",
                              ["(none)"] + list(SAMPLES),
                              help="Bundled example logs so you can try "
                                   "TuneSleuth without your own datalog.")
    with col2:
        obd_code = st.text_input(
            "OBD-II code(s)", placeholder="P0171, P0300", max_chars=60,
            help="One or several, separated by commas or spaces — scans "
                 "usually return more than one, and the combination matters.")
        symptoms = st.text_input(
            "Symptoms (optional)", max_chars=200,
            placeholder="rough idle when cold, smells like gas")

    y_col, mk_col, md_col, mods_col = st.columns([1, 1, 1, 2])
    year = y_col.text_input("Year", placeholder="2017", max_chars=4)
    make = mk_col.text_input("Make", placeholder="Toyota", max_chars=20)
    model = md_col.text_input("Model", placeholder="86", max_chars=30)
    mods = mods_col.text_input("Engine / mods (optional)", max_chars=80,
                               placeholder="FA20, OpenFlash Stage 2 tune")
    st.caption("Year + make + model also enables a free NHTSA recall check.")
    submitted = st.form_submit_button("Diagnose", type="primary")

if submitted:
    if not uploaded and not obd_code and sample == "(none)":
        st.warning("Give me a datalog, a sample log, or a trouble code first.")
        st.stop()
    if uploaded and uploaded.size > MAX_UPLOAD_BYTES:
        st.error("That file is over 20 MB. Trim the log (most tools can "
                 "export a time range) and try again.")
        st.stop()

    if uploaded:
        csv_text = uploaded.getvalue().decode("utf-8", errors="replace")
    elif sample != "(none)":
        csv_text = (DATA_DIR / SAMPLES[sample]).read_text()
    else:
        csv_text = None

    vehicle = " ".join(p.strip() for p in (year, make, model) if p.strip())
    if mods.strip():
        vehicle = f"{vehicle} ({mods.strip()})" if vehicle else mods.strip()

    with st.status("Agents working...", expanded=True) as status:
        def progress(step, detail):
            st.write(f"**{step}** — {detail}")

        result = pipeline.run(csv_text=csv_text, obd_code=obd_code or None,
                              vehicle=vehicle or None,
                              year=year or None, make=make or None,
                              model=model or None,
                              symptoms=symptoms or None, progress=progress)
        status.update(label="Diagnosis complete" if result["ok"] else "Input not usable",
                      state="complete" if result["ok"] else "error",
                      expanded=False)

    st.session_state["result"] = result
    st.session_state["chat_history"] = []

result = st.session_state.get("result")
if result is not None:
    if not result["ok"]:
        st.error(result["message"])
    else:
        sev = result.get("severity")
        if sev:
            banner = f"**{sev['label']}.** {sev['advice']}"
            if sev["level"] == "stop":
                st.error("🛑 " + banner)
            elif sev["level"] == "caution":
                st.warning("⚠️ " + banner)
            else:
                st.info(("🔧 " if sev["level"] == "soon" else "ℹ️ ") + banner)

        if result.get("safety_warning"):
            st.error("⚠️ " + result["safety_warning"])
        for w in result.get("sensor_warnings", []):
            st.warning(w)
        if result.get("warmup_note"):
            st.info(result["warmup_note"])
        for n in result.get("notes", []):
            st.caption("ℹ️ " + n)

        parsed = result["trace"]["parsed"]
        stats = parsed.get("stats", {})

        for c in parsed.get("obd_codes", []):
            st.markdown(f"**{c['code']}** — {c['meaning']}")

        if result.get("recalls"):
            with st.expander(f"🚨 Open recalls reported for this vehicle "
                             f"({len(result['recalls'])}) — dealers fix "
                             "these free"):
                for r in result["recalls"]:
                    st.markdown(f"**{r['component']}** — {r['summary']}")
                    st.caption(f"NHTSA campaign {r['campaign']} · {r['date']}")
                st.caption("Source: NHTSA. A recall matching your symptom "
                           "means a free dealer repair — call with your VIN.")

        if stats:
            metrics = [("Log rows", f"{parsed.get('rows', 0):,}")]
            if "ltft" in stats:
                metrics.append(("LTFT mean", f"{stats['ltft']['mean']}%"))
            if "afr" in stats:
                metrics.append(("AFR mean", f"{stats['afr']['mean']}"))
            metrics.append(("Knock events", f"{stats.get('knock_events', 0)}"))
            with st.container(border=True):
                for col, (label, value) in zip(st.columns(len(metrics)), metrics):
                    col.metric(label, value)

        if result.get("preview") and parsed.get("rows", 0) >= 10:
            chart = log_chart(result["preview"])
            if chart is not None:
                st.altair_chart(chart)

        if result.get("healthy"):
            st.success(result["message"])
        else:
            st.subheader("What the log shows")
            for a in result["anomalies"]:
                st.write("- " + a)

            st.subheader("Probable causes")
            if not result["diagnoses"]:
                st.info("The agents couldn't build well-supported causes from "
                        "web evidence this time — the findings above still "
                        "stand. Try Diagnose again, or add your vehicle "
                        "details for more specific research.")
            else:
                st.caption(f"Overall confidence: {result['confidence']}")
            for i, d in enumerate(result["diagnoses"], 1):
                conf = str(d.get("confidence", "low")).lower()
                badge = CONFIDENCE_BADGE.get(conf, "gray")
                with st.expander(f"{i}\\. {d.get('cause', '?')} "
                                 f":{badge}-badge[{conf} confidence]"):
                    st.write(d.get("evidence", ""))
                    if d.get("check"):
                        st.write(f"**How to check:** {d['check']}")
                    if d.get("cost"):
                        st.write(f"**Typical cost:** {d['cost']}")
                    if d.get("difficulty"):
                        labels = {"diy-easy": "easy DIY",
                                  "diy-moderate": "moderate DIY",
                                  "shop": "shop job"}
                        st.write(f"**Difficulty:** {labels.get(d['difficulty'], d['difficulty'])}")
                    if d.get("source"):
                        st.write(f"Source: {d['source']}")

            if result.get("action_plan"):
                st.subheader("What to do next")
                st.caption("Cheapest way to narrow it down first.")
                for i, step in enumerate(result["action_plan"], 1):
                    st.write(f"{i}. {step}")

            if result.get("critic_notes"):
                st.caption(f"Critic notes: {result['critic_notes']}")

        with st.expander("Full agent trace"):
            st.json(result["trace"])

        st.divider()
        st.subheader("Ask about this diagnosis")
        st.caption("Grounded in the diagnosis above. Ask about potential causes — "
                   "it will look up whether other owners have seen the same issue "
                   "and report the consensus, without inventing log readings.")

        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("e.g. Is it safe to keep driving?", max_chars=500)
        if question:
            history = st.session_state.setdefault("chat_history", [])
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            context = followup.build_context(result)
            with st.spinner("Checking owner reports..."):
                try:
                    answer = followup.answer(context, history[:-1], question)
                except llm.LLMUnavailableError as exc:
                    answer = str(exc)
                except Exception:
                    answer = ("Sorry — I hit an error answering that. The "
                              "diagnosis above is unaffected; try asking again "
                              "in a moment.")

            history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

st.divider()
st.caption("TuneSleuth suggests likely causes from your log and public owner "
           "reports — it is not professional mechanical advice. Verify before "
           "wrenching, and see a shop for anything safety-critical. Logs are "
           "processed in memory for your session and never stored.")
