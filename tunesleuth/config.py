"""Configuration for TuneSleuth. Reads API keys from environment / .env."""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Which LLM provider to use: "anthropic" or "openai".
# If unset, whichever API key is present decides (anthropic wins a tie).
PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
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
