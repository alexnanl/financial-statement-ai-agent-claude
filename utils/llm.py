"""Single point of contact for LLM calls. Supports text and vision.

Provider: Claude (Anthropic) when ANTHROPIC_API_KEY is set, otherwise OpenAI.
The two public functions keep their signatures, so every agent works unchanged:
  chat(system, user, json_mode=False, temperature=None, cheap=False) -> str
  chat_with_images(system, user_text, image_paths, temperature=None) -> str
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from config import CONFIG, get_anthropic_key, get_openai_key, get_provider

_clients: dict = {}

_JSON_RULE = ("\n\nRespond with ONLY one valid JSON object. No prose before or after it, "
              "no markdown code fences.")


# ---------------------------------------------------------------- Claude
def _claude():
    if "claude" not in _clients:
        import anthropic
        _clients["claude"] = anthropic.Anthropic(api_key=get_anthropic_key(), max_retries=3, timeout=300)
    return _clients["claude"]


def _extract_json(text: str) -> str:
    """Return the first top-level JSON object in `text` (tolerates code fences)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if (start != -1 and end > start) else t


def _claude_call(model: str, effort: str, system: str, content) -> str:
    kwargs = dict(model=model, max_tokens=CONFIG.CLAUDE_MAX_TOKENS, system=system,
                  messages=[{"role": "user", "content": content}],
                  output_config={"effort": effort})
    client = _claude()
    if model == CONFIG.CLAUDE_MODEL:
        # Server-side fallback: if the main model declines a request, the API
        # re-runs it on a suitable model inside the same call.
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                           fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request.")
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------- OpenAI
def _openai():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=get_openai_key())
    return _clients["openai"]


def _openai_chat(system, user_content, json_mode, temperature, model):
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_content}],
        "temperature": temperature if temperature is not None else CONFIG.OPENAI_TEMPERATURE,
        "max_tokens": CONFIG.OPENAI_MAX_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _openai().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _require_provider() -> str:
    p = get_provider()
    if not p:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY (Claude) or OPENAI_API_KEY as an "
            "environment variable or in Streamlit secrets. The UI does not accept API keys "
            "for security reasons.")
    return p


# ---------------------------------------------------------------- public API
def chat(system: str, user: str, json_mode: bool = False,
         temperature: float | None = None, cheap: bool = False) -> str:
    """Standard text call. `cheap=True` uses the fast, low-cost model."""
    if _require_provider() == "claude":
        model = CONFIG.CLAUDE_MODEL_CHEAP if cheap else CONFIG.CLAUDE_MODEL
        effort = CONFIG.CLAUDE_EFFORT_CHEAP if cheap else CONFIG.CLAUDE_EFFORT
        out = _claude_call(model, effort, system + (_JSON_RULE if json_mode else ""), user)
        return _extract_json(out) if json_mode else out
    model = CONFIG.OPENAI_MODEL_CHEAP if cheap else CONFIG.OPENAI_MODEL
    return _openai_chat(system, user, json_mode, temperature, model)


def chat_with_images(system: str, user_text: str, image_paths: list[str],
                     temperature: float | None = None) -> str:
    """Vision call - send PNG charts to the LLM so it can describe them."""
    images = []
    for path in image_paths:
        p = Path(path)
        if p.exists():
            images.append(base64.standard_b64encode(p.read_bytes()).decode("utf-8"))

    if _require_provider() == "claude":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}}
                   for b in images]
        content.append({"type": "text", "text": user_text})
        return _claude_call(CONFIG.CLAUDE_MODEL, CONFIG.CLAUDE_EFFORT, system, content)

    user_content = [{"type": "text", "text": user_text}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}", "detail": "high"}}
        for b in images]
    return _openai_chat(system, user_content, False, temperature, CONFIG.OPENAI_MODEL)
