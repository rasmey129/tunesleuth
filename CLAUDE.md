# CLAUDE.md — TuneSleuth

Context file for Claude Code. Read this before making changes.

## What this project is

TuneSleuth is a multi-agent diagnostic assistant for vehicle datalogs and
OBD-II trouble codes. It is the individual term project for CIS 8045
(Agentic AI), Georgia State, Summer 2026 — worth 55% of the course grade.
The owner is Ras (github.com/rasmey129). The project was scoped from an
original aftermarket-tuning-only idea up to general vehicle issue diagnosis;
the proposal reflects that broader framing.

The domain is real: Ras owns a 2017 Toyota 86 (FA20, OpenFlash Stage 2 tune,
UEL headers) and has personally diagnosed a lean condition on it. The real
logs in `data/real/` are from that car.

## Architecture (do not change without good reason)

Five agents in a fixed pipeline with one bounded critique loop:

    Input Parser -> Analyzer -> Web Researcher -> Synthesizer -> Critic

Locked design decisions and the reasoning behind them:

- The Parser (`tunesleuth/agents/parser.py`) is deterministic pandas, no
  LLM. Parsing must never hallucinate. It handles plain CSVs, OpenFlash
  Tablet exports (which have a 3-line "Procede Data Log" preamble before the
  header — `_find_header_row` skips it), consumer OBD app exports (Torque
  etc. — covered by substring aliases), and single-row freeze-frame
  snapshots (flagged via parsed["format"]/notes; stats still work). It
  accepts multiple OBD codes at once (real scans return several; the
  combination is diagnostic) — parsed["obd_codes"] is the list,
  obd_code/obd_meaning stay as first-code compat keys. Channel aliases map
  real-world names: FLKC is the knock channel, "Command AFR" is the target,
  "MAF Volt." must never satisfy the MAF channel (voltage is not flow),
  "Engine Speed" must never satisfy the vehicle-speed channel.
- The Analyzer (`agents/analyzer.py`) detects anomalies with deterministic
  thresholds and uses the LLM only to write search queries, with a
  heuristic fallback if the LLM call fails. Checks: LTFT beyond +-10%,
  sustained STFT beyond +-8%, combined trims beyond +-15%, AFR lean
  (>13.5) or rich (<10.5) above 4000 RPM, AFR diverging from commanded
  under load (>1.5 lean / <-2.0 rich), oil >240F / coolant >230F
  overheating, IAT >150F heat soak, nonzero knock. Thresholds were
  calibrated against every log in data/ so none false-positive (worst
  benign readings: STFT 4.0, AFR-target divergence 1.48, oil 212F,
  IAT 131F). An idle-speed check was tried and rejected: throttle<5%
  in driving logs captures closed-throttle decel, not idle. OBD codes are decoded locally by
  `tunesleuth/obd_codes.py`: a table of ~30 common codes plus an SAE
  structural fallback so any valid code gets a meaningful description.
  Deliberate split: static code structure is decoded locally and free; the
  model-specific meaning comes from live web research.
- The Researcher (`agents/workers.py`) calls Serper.dev, caches per query,
  caps at 3 searches and 3 page fetches per run (free-tier protection),
  and degrades gracefully on tool failure instead of dying.
- The Critic can send the synthesis back for exactly one revision
  (MAX_CRITIC_REVISIONS = 1), then accepts or returns low-confidence.
- Hard safety rules enforced in code, not prompts. (1) `pipeline.py`: any
  detected knock escalates to a conservative warning regardless of what
  the synthesis says. Never weaken this. (2) `tunesleuth/severity.py`:
  every finding carries a deterministic urgency tier (low < soon <
  caution < stop) answering "can I keep driving?"; the analyzer tags
  findings as it detects them (so warmup suppression suppresses severity
  too — coldstart must stay severity None), the worst tier wins, and
  misfire codes (P0300-P0316) and overheating are "stop", EVAP/catalyst
  codes are "low". The app shows it as the top banner.
- The Synthesizer returns, alongside ranked causes (each with check /
  cost / difficulty fields), an `action_plan`: 3-5 plain-language steps
  ordered cheapest-to-verify first. The Critic judges only cause+evidence,
  not the practical-guidance fields.
- Recall lookup (`tools/recalls.py`): free NHTSA API by year/make/model,
  no key, cached, degrades to [] on any failure. Runs when the user gives
  year+make+model (also enabled on healthy logs), feeds the synthesizer
  context and a UI expander. Verified live 7/10 (2017 Toyota 86 returns
  2 campaigns).
- Owner-reported symptoms are context only: they enrich analyzer queries
  and synthesis but never create findings (no symptom-only diagnosis —
  that would bypass the verified pipeline). A healthy log + symptoms gets
  a "re-log while the symptom is happening" note.
- Anomaly strings are written for non-mechanics: "<technical finding> —
  <plain-language meaning>" (the eval keywords live in the technical
  half; keep them when rewording).
- A vehicle string (year/make/model/engine/mods, free text) threads through
  analyzer queries, synthesizer context, and follow-up chat so diagnoses are
  model-specific. Optional; everything works without it.
- The follow-up chat (`agents/followup.py`) uses two-tier grounding
  (Ras loosened the original strict-grounding rule on 7/10, twice):
  facts about this car's log come ONLY from the diagnosis context and
  must never be invented, but the bot may reason about plausible causes
  for the diagnosed condition — including vehicle-specific common
  failure points (e.g. "could an intake leak cause this lean AFR on a
  2017 86?"). For cause/fix/has-anyone-seen-this questions it runs its
  own small search (a query-writing LLM step, then workers.research,
  capped at MAX_CHAT_SEARCHES = 2 per question) and reports the
  consensus across owner reports, citing links and noting disagreement.
  Definitional or restating questions skip the search. It never re-runs
  the pipeline and never downplays a safety warning.

## LLM provider

Both Anthropic and OpenAI are supported, isolated in `tunesleuth/llm.py`.
`LLM_PROVIDER` in .env picks; if unset, whichever key exists wins
(anthropic on a tie). Agents only call `llm.complete_json()` /
`llm.complete()` and must stay provider-agnostic.

Mock mode: with no keys (or TUNESLEUTH_MOCK=1), canned LLM and search
responses are used so the pipeline and UI demo offline. Mock responses
live in `llm.py` (`_MOCK_RESPONSES`) and `tools/serp.py` (`_MOCK_RESULTS`).

IMPORTANT: the live (keyed) LLM and SERP paths have been written and
syntax-checked but never executed against real APIs. First task in any
live session: run one real diagnosis, read `result["trace"]`, and expect
to tune the Synthesizer/Critic prompts against real model output.

## Environment and conventions

- Ras is on Windows PowerShell. No Unix commands in instructions
  (grep -> Select-String, etc.). Repo code itself is cross-platform.
- Secrets: .env locally (.env.example is the template), Streamlit Cloud
  Secrets (TOML) when deployed. `config.py` checks env vars first, then
  st.secrets (wrapped so the eval harness works without streamlit).
  Never commit .env; .gitignore excludes it and .streamlit/secrets.toml.
- Production hardening (added 7/10): pipeline.run never raises (LLM
  failures surface as friendly ok=False messages via
  llm.LLMUnavailableError; everything else is caught and logged);
  complete_json logs malformed LLM output; app has sample-log picker,
  upload size cap (20MB in-app, 25MB via .streamlit/config.toml), input
  length caps, chat error guard, and a disclaimer/privacy footer; serp
  cache is capped.
- Deployed on Streamlit Community Cloud from github.com/rasmey129/tunesleuth,
  main file `app.py`, branch `main`.
- `data/` holds synthetic logs with known causes; `data/real/` holds nine
  real OpenFlash FA20 logs. run2.csv and run3.csv are byte-identical
  duplicates. e85.bin is a tune ROM, not a log — ignore it.

## Known findings in the real logs (useful for evaluation writeups)

- 02.csv / 022.csv: LTFT +25-28%, seriously lean. Their wideband AFR
  channel reads a flat 18.0 throughout — pegged or disconnected sensor.
  HANDLED: the parser's flat-sensor check (`_is_flat`) detects it, falls
  back to the live "AFR" column, and emits a sensor warning shown in the
  UI. Good failure-case-found-and-fixed material for Milestone II.
- datalog1.csv: elevated LTFT plus 21 knock events.
- datalog3.csv: LTFT ~21%, 180 knock events — heaviest knock log, always
  triggers the safety warning.
- coldstart.csv: high trims that are open-loop warmup enrichment — was a
  textbook false positive. HANDLED: the analyzer's warmup guard
  (WARMUP_TEMP_F = 170; warmed logs run 183-212F oil, coldstart peaks at
  158F) suppresses lean/rich flags when oil/coolant never reaches
  operating temperature and reports a warmup note instead. Knock
  detection is never suppressed.

## Testing

    python eval/evaluate.py

21 labeled cases in `eval/test_cases.json`, currently 21/21 in mock mode,
grouped (synthetic / real-log / obd-code / rejection) with per-group
stats printed: real-log parsing, lean/rich detection, overheating and
heat soak, knock warnings on real logs, healthy log, OBD codes (P0171,
P0300, P0420, unlisted P0733, multi-code P0171+P0300, low-urgency
P0455), garbage input rejection, pegged-sensor warnings (02/022), and
the coldstart warmup suppression. The harness verifies safety/sanity
checks per case (knock rule, sensor warning, warmup note, and
must_severity tier assertions) — keep those green. It exits nonzero on
any failure, so it works in CI. sample_rich.csv and sample_overheat.csv
are synthetic (numpy, seed 42) covering detectors no real log exercises.

## Course deadlines (Summer 2026)

- 7/4  Milestone I: workflow + architecture diagram + 2-3 page design doc
- 7/11 Milestone II: evaluation plan, pilot results, risk matrix,
       failure-case analysis (2-3 pages + scripts)
- 7/18 Milestone III: front-end (exists: Streamlit) + 1-2 page README/desc
       with screenshots
- 7/25 Final: 10-min video, 8-10 page report, code repo, peer reviews

## Near-term work, in rough priority

1. Live keyed run; tune Synthesizer/Critic prompts on real output.
2. Architecture diagram (Milestone I deliverable).
3. Label the real logs with ground truth (Ras knows what fixed the car);
   eval now has 16 cases with group stats — more labeled cases welcome.
4. Screenshots of the UI for Milestone III.

Done: warmup/coldstart guard (Analyzer), pegged-sensor check (Parser),
follow-up chat wired into the UI, local OBD decoding (obd_codes.py),
vehicle string threading, .gitignore, severity triage (severity.py),
multi-code input, NHTSA recall lookup, action plans with cost/DIY info,
symptoms input, consumer OBD app formats + freeze frames, plain-language
findings, production hardening.

## Writing style for any documents

Ras prefers plain, direct prose that doesn't read AI-generated: no
bolded bullet lead-ins, minimal em-dashes, varied sentence length, no
decorative formatting or colors in Word docs. Match the tone of the
existing proposal files.
