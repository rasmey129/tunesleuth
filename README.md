# TuneSleuth

A multi-agent diagnostic assistant for vehicle datalogs and OBD-II trouble
codes. Built for CIS 8045 (Agentic AI), Summer 2026.

Five agents run in a fixed pipeline with one verification loop:

    Input Parser -> Analyzer -> Web Researcher -> Synthesizer -> Critic

The Parser reads a datalog CSV or a trouble code and computes statistics
deterministically (no LLM). The Analyzer flags anomalies and writes search
queries. The Researcher hits a SERP API (Serper.dev) and fetches top pages.
The Synthesizer combines log evidence with web sources into ranked causes,
each with a citation. The Critic verifies every claim traces to the log or
a source, and can send the result back for one revision. A hard rule
escalates any detected knock to a conservative safety warning no matter
what the synthesis says.

## Setup

    pip install -r requirements.txt
    cp .env.example .env    # add your key (Anthropic or OpenAI, plus Serper)

Either an Anthropic or an OpenAI key works; set LLM_PROVIDER in .env to
choose explicitly. Without keys the system runs in mock mode with canned LLM and search
responses, so the pipeline and UI can be demoed offline.

## Run the app

    streamlit run app.py

Upload one of the sample logs in `data/` (lean, knock, healthy) or type a
code like P0171.

## Run the evaluation

    python eval/evaluate.py

Runs the labeled test cases in `eval/test_cases.json` and prints task
success plus a check that the knock safety rule always held.

## Layout

    app.py                  Streamlit front-end
    tunesleuth/config.py    keys, model, limits, mock mode
    tunesleuth/llm.py       LLM wrapper (Anthropic) with mock fallback
    tunesleuth/pipeline.py  orchestration, critic loop, knock rule
    tunesleuth/agents/      parser, analyzer, researcher/synthesizer/critic
    tunesleuth/tools/       SERP search + page fetch, cached
    data/                   synthetic sample logs; data/real/ has OpenFlash logs
    eval/                   evaluation script and test cases
