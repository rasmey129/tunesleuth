"""TuneSleuth front-end. Run with: streamlit run app.py"""
import altair as alt
import pandas as pd
import streamlit as st

from tunesleuth import pipeline, config
from tunesleuth.agents import followup

st.set_page_config(page_title="TuneSleuth", page_icon="🔧", layout="centered")

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
        uploaded = st.file_uploader("Datalog CSV", type=["csv"])
    with col2:
        obd_code = st.text_input("OBD-II code", placeholder="P0171",
                                 help="Either input works; give both if you have both.")
        vehicle = st.text_input("Vehicle (optional)",
                                placeholder="2017 Toyota 86, FA20, OFT Stage 2")
    submitted = st.form_submit_button("Diagnose", type="primary")

if submitted:
    if not uploaded and not obd_code:
        st.warning("Give me a datalog or a trouble code first.")
        st.stop()

    csv_text = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else None
    with st.status("Agents working...", expanded=True) as status:
        def progress(step, detail):
            st.write(f"**{step}** — {detail}")

        result = pipeline.run(csv_text=csv_text, obd_code=obd_code or None,
                              vehicle=vehicle or None, progress=progress)
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
        if result.get("safety_warning"):
            st.error("⚠️ " + result["safety_warning"])
        for w in result.get("sensor_warnings", []):
            st.warning(w)
        if result.get("warmup_note"):
            st.info(result["warmup_note"])

        parsed = result["trace"]["parsed"]
        stats = parsed.get("stats", {})

        if parsed.get("obd_code") and parsed.get("obd_meaning"):
            st.markdown(f"**{parsed['obd_code']}** — {parsed['obd_meaning']}")

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

        if result.get("preview"):
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
            st.caption(f"Overall confidence: {result['confidence']}")
            for i, d in enumerate(result["diagnoses"], 1):
                conf = str(d.get("confidence", "low")).lower()
                badge = CONFIDENCE_BADGE.get(conf, "gray")
                with st.expander(f"{i}\\. {d.get('cause', '?')} "
                                 f":{badge}-badge[{conf} confidence]"):
                    st.write(d.get("evidence", ""))
                    if d.get("source"):
                        st.write(f"Source: {d['source']}")
            if result.get("critic_notes"):
                st.caption(f"Critic notes: {result['critic_notes']}")

        with st.expander("Full agent trace"):
            st.json(result["trace"])

        st.divider()
        st.subheader("Ask about this diagnosis")
        st.caption("Grounded strictly in the diagnosis above — it won't run new "
                   "searches or speculate beyond this evidence.")

        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("e.g. Is it safe to keep driving?")
        if question:
            history = st.session_state.setdefault("chat_history", [])
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            context = followup.build_context(result)
            with st.spinner("Thinking..."):
                answer = followup.answer(context, history[:-1], question)

            history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
