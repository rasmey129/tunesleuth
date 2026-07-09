"""TuneSleuth front-end. Run with: streamlit run app.py"""
import streamlit as st
from tunesleuth import pipeline, config

st.set_page_config(page_title="TuneSleuth", page_icon=None, layout="centered")
st.title("TuneSleuth")
st.caption("Multi-agent diagnosis for vehicle datalogs and trouble codes")

if config.MOCK_MODE:
    st.info("Running in mock mode (no API keys found). Set ANTHROPIC_API_KEY "
            "and SERPER_API_KEY in a .env file for live diagnosis.")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload a datalog CSV", type=["csv"])
with col2:
    obd_code = st.text_input("Or enter an OBD-II code", placeholder="P0171")

if st.button("Diagnose", type="primary"):
    if not uploaded and not obd_code:
        st.warning("Give me a datalog or a trouble code first.")
        st.stop()

    csv_text = uploaded.getvalue().decode("utf-8", errors="replace") if uploaded else None
    steps_box = st.container()
    step_log = []

    def progress(step, detail):
        step_log.append((step, detail))
        with steps_box:
            st.write(f"**{step}** — {detail}")

    with st.spinner("Agents working..."):
        result = pipeline.run(csv_text=csv_text, obd_code=obd_code or None,
                              progress=progress)

    st.divider()
    if not result["ok"]:
        st.error(result["message"])
        st.stop()

    if result.get("safety_warning"):
        st.error(result["safety_warning"])

    if result.get("healthy"):
        st.success(result["message"])
    else:
        st.subheader("What the log shows")
        for a in result["anomalies"]:
            st.write("- " + a)

        st.subheader(f"Probable causes (overall confidence: {result['confidence']})")
        for i, d in enumerate(result["diagnoses"], 1):
            with st.expander(f"{i}. {d.get('cause', '?')} ({d.get('confidence', '?')})"):
                st.write(d.get("evidence", ""))
                if d.get("source"):
                    st.write(f"Source: {d['source']}")
        if result.get("critic_notes"):
            st.caption(f"Critic notes: {result['critic_notes']}")

    with st.expander("Full agent trace"):
        st.json(result["trace"])
