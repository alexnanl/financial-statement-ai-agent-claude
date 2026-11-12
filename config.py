"""Configuration for the Financial Statement Analysis Agent.

LLM provider: Claude (Anthropic) when ANTHROPIC_API_KEY is set, otherwise
OpenAI (OPENAI_API_KEY). Keys come ONLY from environment variables or
Streamlit secrets, never from the UI.
"""
import os
from dataclasses import dataclass


def _secret(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets."""
    val = os.getenv(name, "")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(name, "")
        except Exception:
            pass
    return val or default


@dataclass
class Config:
    """Central configuration."""
    # --- Claude (preferred) ---
    # Main model: Analyst (reads the charts). Cheap model: Planner, Peer Selector, Critic.
    CLAUDE_MODEL: str = _secret("CLAUDE_MODEL", "claude-opus-5-5")
    CLAUDE_MODEL_CHEAP: str = _secret("CLAUDE_MODEL_CHEAP", "claude-haiku-5-5")
    CLAUDE_EFFORT: str = _secret("CLAUDE_EFFORT", "medium")          # low | medium | high
    CLAUDE_EFFORT_CHEAP: str = _secret("CLAUDE_EFFORT_CHEAP", "low")
    CLAUDE_MAX_TOKENS: int = 16000

    # --- OpenAI (fallback when no Claude key is set) ---
    OPENAI_MODEL: str = "gpt-4o"            # Analyst uses this (needs vision)
    OPENAI_MODEL_CHEAP: str = "gpt-4o-mini" # Planner, Critic, Peer Selector
    OPENAI_TEMPERATURE: float = 0.2
    OPENAI_MAX_TOKENS: int = 2500

    # Data settings
    DEFAULT_LOOKBACK_YEARS: int = 4

    # Quality thresholds
    MIN_DATA_COMPLETENESS: float = 0.6

    # Critic settings
    MAX_CRITIC_ROUNDS: int = 2

    # Peer analysis
    PEER_COUNT: int = 4  # how many peers to auto-select

    # Chart settings
    CHART_DPI: int = 130
    CHART_WIDTH: float = 8.0   # inches
    CHART_HEIGHT: float = 4.5


CONFIG = Config()


def get_anthropic_key() -> str:
    return _secret("ANTHROPIC_API_KEY")


def get_openai_key() -> str:
    return _secret("OPENAI_API_KEY")


def get_provider() -> str:
    """'claude', 'openai', or '' when no key is configured."""
    if get_anthropic_key():
        return "claude"
    if get_openai_key():
        return "openai"
    return ""


def provider_label() -> str:
    p = get_provider()
    if p == "claude":
        return f"Claude ({CONFIG.CLAUDE_MODEL} + {CONFIG.CLAUDE_MODEL_CHEAP})"
    if p == "openai":
        return f"OpenAI ({CONFIG.OPENAI_MODEL} + {CONFIG.OPENAI_MODEL_CHEAP})"
    return "none"
