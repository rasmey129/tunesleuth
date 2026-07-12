# TuneSleuth

Upload a datalog or type a trouble code, get a ranked list of probable
causes with evidence. TuneSleuth reads engine datalogs (any CSV with
channels like RPM, AFR, fuel trims, knock; OpenFlash Tablet exports work
as-is) and OBD-II codes, detects what's wrong, researches what has fixed
the same problem for other owners, and verifies every claimed cause
against the log or a cited source. A follow-up chat answers questions
about the diagnosis and can look up whether other owners have seen the
same issue on your car.

Built for CIS 8045 (Agentic AI), Summer 2026. Not professional mechanical
advice: verify before wrenching. Logs are processed in memory and never
stored.

## How it works

Five agents run in a fixed pipeline with one verification loop:

    Input Parser -> Analyzer -> Web Researcher -> Synthesizer -> Critic

The Parser reads the CSV or code and computes statistics deterministically
(no LLM), flags dead sensors (a wideband pegged at one value), and decodes
OBD codes locally. The Analyzer applies calibrated thresholds for lean and
rich running, sustained trims, AFR-vs-commanded divergence, overheating,
heat soak, and knock — and suppresses fueling checks during engine warmup
so a cold start isn't misread as a lean fault. The Researcher searches the
web (Serper.dev) and fetches top pages. The Synthesizer combines log
evidence with sources into ranked causes, each with a citation. The Critic
verifies every claim traces to the log or a source and can send the result
back for one revision. A hard rule in code escalates any detected knock to
a safety warning no matter what the synthesis says.

## Run it locally

    pip install -r requirements.txt
    cp .env.example .env    # add your keys, or skip for mock mode

Either an Anthropic or an OpenAI key works; set LLM_PROVIDER in .env to
choose explicitly. SERPER_API_KEY (serper.dev, free tier) enables live web
research. Without keys the system runs in mock mode with canned LLM and
search responses, so the pipeline and UI can be demoed offline.

    streamlit run app.py

No datalog handy? Pick one of the built-in sample logs (lean, knock, rich,
overheating, healthy) from the dropdown, or type a code like P0171.

## Run the evaluation

    python eval/evaluate.py

Runs the labeled test cases in `eval/test_cases.json` — synthetic logs,
real OpenFlash logs, OBD codes, and rejection cases — and prints task
success plus safety checks (knock warning, dead-sensor flag, warmup
suppression). Exits nonzero on failure, so it works in CI.

## Layout

    app.py                   Streamlit front-end
    tunesleuth/config.py     keys (env or Streamlit secrets), model, limits
    tunesleuth/llm.py        LLM wrapper (Anthropic/OpenAI) with mock fallback
    tunesleuth/obd_codes.py  local OBD-II code decoding
    tunesleuth/pipeline.py   orchestration, critic loop, knock rule
    tunesleuth/agents/       parser, analyzer, workers, follow-up chat
    tunesleuth/tools/        SERP search + page fetch, cached
    data/                    synthetic sample logs; data/real/ has OpenFlash logs
    eval/                    evaluation script and test cases
