"""Configuration for TuneSleuth.

Secrets resolve from the environment / .env first, then from Streamlit
secrets (secrets.toml on Streamlit Cloud). The st.secrets fallback is
wrapped so everything still works where streamlit isn't installed or no
secrets file exists (e.g. the eval harness).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _secret(name: str) -> str:
    val = os.getenv(name, "")
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _secret("OPENAI_API_KEY")
SERPER_API_KEY = _secret("SERPER_API_KEY")

# Which LLM provider to use: "anthropic" or "openai".
# If unset, whichever API key is present decides (anthropic wins a tie).
PROVIDER = _secret("LLM_PROVIDER").lower()
if PROVIDER not in ("anthropic", "openai"):
    PROVIDER = "anthropic" if ANTHROPIC_API_KEY else ("openai" if OPENAI_API_KEY else "")

# Models per provider
MODEL = os.getenv("TUNESLEUTH_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("TUNESLEUTH_OPENAI_MODEL", "gpt-4o-mini")

# If no API keys are present, the system runs in mock mode so it can be
# demoed offline. Mock mode uses canned LLM/search responses.
MOCK_MODE = (os.getenv("TUNESLEUTH_MOCK", "").lower() in ("1", "true")
             or not (ANTHROPIC_API_KEY or OPENAI_API_KEY))

# Hard limits to stay inside free API tiers
MAX_SEARCHES_PER_RUN = 3
MAX_PAGES_FETCHED = 3
MAX_CRITIC_REVISIONS = 1
MAX_CHAT_SEARCHES = 2    # per follow-up question
