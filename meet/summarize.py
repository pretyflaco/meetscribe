"""Meeting summary generation using LLMs.

Supports multiple backends:
  - claudemax:  Claude Sonnet 4.6 via claude-max-api-proxy on localhost:3456
                ($0 extra — uses existing Claude Max subscription).
  - openrouter: OpenRouter API (OpenAI-compatible, requires OPENROUTER_API_KEY).
  - ollama:     Local Ollama server (free, lowest quality, last resort).

Fallback chain: claudemax -> openrouter -> ollama.
When the configured primary backend is unavailable, the system automatically
tries the next backend in the fallback order.

Configuration precedence (highest to lowest):
  1. Explicit keyword arguments / CLI flags (--summary-backend, --summary-model)
  2. Environment variables (MEETSCRIBE_SUMMARY_BACKEND, MEETSCRIBE_SUMMARY_MODEL)
  3. Hardcoded defaults (ollama / qwen3.5:9b)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import re
import requests

# ─── Constants ──────────────────────────────────────────────────────────────

# Ollama defaults
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 600  # 10 minutes max

# OpenRouter defaults
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Claude Max API Proxy defaults
DEFAULT_CLAUDEMAX_MODEL = "claude-sonnet-4-6"
CLAUDEMAX_BASE_URL = "http://localhost:3457/v1"
CLAUDEMAX_HEALTH_URL = "http://localhost:3457/health"

# OpenAI-compatible generic endpoint defaults
DEFAULT_OPENAI_COMPAT_MODEL = "gpt-4o-mini"

# Supported backends
BACKENDS = ("ollama", "openrouter", "claudemax", "openai")

# Fallback order: try claudemax first, then openrouter, then ollama
# (openai is not in fallback — it's opt-in only via explicit config)
FALLBACK_ORDER = ("claudemax", "openrouter", "ollama")

# Backward-compatible aliases (referenced by translate command, etc.)
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

from meet.languages import SECTION_HEADERS as _SECTION_HEADERS, LANG_NAMES as _LANGUAGE_NAMES  # noqa: E402

# ─── Prompt loading ────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str | None:
    """Load a prompt template from the prompts directory. Returns None if missing."""
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _build_system_prompt(language: str | None = None) -> str:
    """Build the system prompt with section headers in the target language."""
    lang = language or "en"
    h = _SECTION_HEADERS.get(lang, _SECTION_HEADERS["en"])

    lang_instruction = ""
    if lang != "en":
        lang_name = _LANGUAGE_NAMES.get(lang, lang)
        lang_instruction = (
            f"\n- CRITICAL: Write the ENTIRE summary in {lang_name}, "
            f"including ALL section headers. Do NOT use any English text."
        )

    template = _load_prompt("summarize_system.md")
    if template is not None:
        return template.format(
            overview=h["overview"],
            topics=h["topics"],
            actions=h["actions"],
            decisions=h["decisions"],
            questions=h["questions"],
            none_stated=h["none_stated"],
            lang_instruction=lang_instruction,
        )

    # Inline fallback if prompt file is missing
    return f"""\
You are a professional meeting assistant. Analyze the meeting transcript \
and produce a structured summary.

## {h['overview']}
2-3 sentences covering: what the meeting was about, who was involved, and the main themes.

## {h['topics']}
* **Topic name:** 1-2 sentence description with key technical details.

## {h['actions']}
* Action item — **Owner**
(If none, write "{h['none_stated']}".)

## {h['decisions']}
* Concrete decision stated as a fact.
(If none, write "{h['none_stated']}".)

## {h['questions']}
* Unresolved question or follow-up item.
(If none, write "{h['none_stated']}".)

Rules:
- Use speaker labels exactly as they appear — do not rename or invent names
- Do not hallucinate — every item must be traceable to the transcript
- Be concise but information-dense
- Preserve technical specificity: name exact tools, APIs, frameworks mentioned
- Keep the summary professional and objective{lang_instruction}"""


def _load_user_prompt_template() -> str:
    """Load the user prompt template."""
    template = _load_prompt("summarize_user.md")
    if template is not None:
        return template
    return "Please summarize the following meeting transcript:\n\n---\n{transcript}\n---"


def _load_user_prompt_template_lang() -> str:
    """Load the language-specific user prompt template."""
    template = _load_prompt("summarize_user_lang.md")
    if template is not None:
        return template
    return (
        "The following meeting transcript is in {language}. "
        "Please summarize it in {language}.\n\n---\n{transcript}\n---"
    )


USER_PROMPT_TEMPLATE = _load_user_prompt_template()

USER_PROMPT_TEMPLATE_LANG = _load_user_prompt_template_lang()


# ─── Data classes ───────────────────────────────────────────────────────────

def _resolve_backend() -> str:
    """Resolve the default backend from env var or hardcoded default."""
    return os.environ.get("MEETSCRIBE_SUMMARY_BACKEND", "ollama").lower()


def _resolve_model(backend: str) -> str:
    """Resolve the default model for a backend from env var or hardcoded default."""
    env_model = os.environ.get("MEETSCRIBE_SUMMARY_MODEL")
    if env_model:
        return env_model
    if backend == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    if backend == "claudemax":
        return DEFAULT_CLAUDEMAX_MODEL
    if backend == "openai":
        return DEFAULT_OPENAI_COMPAT_MODEL
    return DEFAULT_OLLAMA_MODEL


@dataclass
class SummaryConfig:
    """Configuration for meeting summary generation.

    Supports multiple backends. The ``backend`` and ``model`` fields
    respect environment variables when left at their sentinel values:

        MEETSCRIBE_SUMMARY_BACKEND  -> backend  (default: "ollama")
        MEETSCRIBE_SUMMARY_MODEL    -> model    (default: per-backend)
        OPENROUTER_API_KEY          -> required for openrouter backend
    """

    backend: str | None = None   # None = resolve from env/default
    model: str | None = None     # None = resolve from env/default per backend
    ollama_url: str = OLLAMA_BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = 0.3
    num_ctx: int = 8192  # Ollama-specific context window

    def __post_init__(self):
        # Resolve backend: explicit arg > env var > "ollama"
        if self.backend is None:
            self.backend = _resolve_backend()
        self.backend = self.backend.lower()

        if self.backend not in BACKENDS:
            raise ValueError(
                f"Unknown summary backend '{self.backend}'. "
                f"Supported: {', '.join(BACKENDS)}"
            )

        # Resolve model: explicit arg > env var > per-backend default
        if self.model is None:
            self.model = _resolve_model(self.backend)


@dataclass
class MeetingSummary:
    """Result of a meeting summary generation."""

    markdown: str
    model: str
    elapsed_seconds: float
    backend: str = ""

    def save(self, output_dir: str | Path, basename: str) -> Path:
        """Save the summary as a .summary.md file and a .summary.meta.json sidecar.

        The sidecar records which backend/model produced the summary so that
        it can be determined post-hoc.

        Returns the path to the saved .summary.md file.
        """
        import datetime

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / f"{basename}.summary.md"
        md_path.write_text(self.markdown, encoding="utf-8")

        meta = {
            "backend": self.backend,
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        meta_path = output_dir / f"{basename}.summary.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        return md_path


# ─── Ollama availability check ─────────────────────────────────────────────

def is_ollama_available(url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama is running and reachable."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_models(url: str = OLLAMA_BASE_URL) -> list[str]:
    """List available Ollama models."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─── Backend availability checks ───────────────────────────────────────────

def is_claudemax_available() -> bool:
    """Check if the claude-max-api-proxy is running and healthy."""
    try:
        resp = requests.get(CLAUDEMAX_HEALTH_URL, timeout=3)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def is_backend_available(config: SummaryConfig | None = None) -> bool:
    """Check if the configured summary backend is reachable.

    For claudemax: checks the local proxy health endpoint.
    For openrouter: checks that OPENROUTER_API_KEY is set.
    For ollama: checks the local server.
    """
    if config is None:
        config = SummaryConfig()

    if config.backend == "claudemax":
        return is_claudemax_available()
    elif config.backend == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    elif config.backend == "openai":
        return bool(os.environ.get("MEETSCRIBE_OPENAI_BASE_URL"))
    else:
        return is_ollama_available(config.ollama_url)


def _backend_not_available_message(config: SummaryConfig) -> str:
    """Return a user-friendly message when the backend is unavailable."""
    if config.backend == "claudemax":
        return (
            "Claude Max API Proxy is not running at localhost:3456. "
            "Start it with: systemctl --user start claude-max-proxy"
        )
    if config.backend == "openrouter":
        return (
            "OPENROUTER_API_KEY is not set. "
            "Export it or use --summary-backend ollama."
        )
    if config.backend == "openai":
        return (
            "MEETSCRIBE_OPENAI_BASE_URL is not set. "
            "Export it with the base URL of your OpenAI-compatible API."
        )
    return (
        f"Ollama is not running at {config.ollama_url}. "
        "Start it with: ollama serve"
    )


# ─── Ollama backend ───────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters for English text.

    This is a conservative heuristic.  Real tokenizers vary by model,
    but 4 chars/token is a safe lower bound (i.e. overestimates tokens)
    which is what we want when sizing the context window.
    """
    return len(text) // 4


def _dynamic_num_ctx(
    system_prompt: str,
    user_prompt: str,
    floor: int = 8192,
    ceiling: int = 65536,
    output_reserve: int = 4096,
) -> int:
    """Calculate a context window size that fits the full prompt.

    Returns a value between *floor* and *ceiling* (inclusive).  The
    calculation adds an *output_reserve* buffer so the model has room
    to generate the summary without truncating its own output.
    """
    prompt_tokens = _estimate_tokens(system_prompt + user_prompt)
    needed = prompt_tokens + output_reserve
    # Round up to nearest 1024 for tidiness
    needed = ((needed + 1023) // 1024) * 1024
    return max(floor, min(needed, ceiling))


def _summarize_ollama(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a summarization request to local Ollama."""
    import time

    if not is_ollama_available(config.ollama_url):
        raise ConnectionError(
            f"Ollama is not running at {config.ollama_url}. "
            "Start it with: ollama serve"
        )

    # Dynamically size the context window based on actual prompt length
    # so long transcripts are not silently truncated.
    num_ctx = _dynamic_num_ctx(
        system_prompt, user_prompt,
        floor=config.num_ctx,  # never go below the configured minimum
    )

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # Disable thinking/reasoning for speed
        "options": {
            "temperature": config.temperature,
            "num_ctx": num_ctx,
        },
    }

    url = f"{config.ollama_url}/api/chat"
    t0 = time.time()

    try:
        resp = requests.post(url, json=payload, timeout=config.timeout)
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError(
            f"Ollama timed out after {config.timeout}s. "
            f"The model '{config.model}' may be too large or slow. "
            "Try a smaller model with --summary-model."
        )
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")

    elapsed = time.time() - t0
    data = resp.json()
    content = data.get("message", {}).get("content", "")

    if not content.strip():
        raise RuntimeError(
            f"Ollama returned an empty response. Model '{config.model}' may "
            "not be available. Check with: ollama list"
        )

    return MeetingSummary(
        markdown=content.strip(),
        model=config.model,
        elapsed_seconds=elapsed,
        backend="ollama",
    )


# ─── OpenRouter backend ───────────────────────────────────────────────────

def _summarize_openrouter(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a summarization request to OpenRouter (OpenAI-compatible API)."""
    import time

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Export it or use --summary-backend ollama."
        )

    # Lazy import — only needed when openrouter is actually used
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            timeout=config.timeout,
        )
    except Exception as e:
        raise RuntimeError(f"OpenRouter API error: {e}")

    elapsed = time.time() - t0
    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise RuntimeError(
            f"OpenRouter returned an empty response for model '{config.model}'."
        )

    # Use a clean display name for the model (strip org prefix for display)
    display_model = config.model.split("/")[-1] if "/" in config.model else config.model

    return MeetingSummary(
        markdown=content,
        model=display_model,
        elapsed_seconds=elapsed,
        backend="openrouter",
    )


# ─── Claude Max API Proxy backend ─────────────────────────────────────────

def _summarize_claudemax(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a summarization request to Claude Max API Proxy (OpenAI-compatible)."""
    import time

    if not is_claudemax_available():
        raise ConnectionError(
            "Claude Max API Proxy is not running at localhost:3456. "
            "Start it with: systemctl --user start claude-max-proxy"
        )

    # Lazy import — only needed when claudemax is actually used
    from openai import OpenAI

    client = OpenAI(
        base_url=CLAUDEMAX_BASE_URL,
        api_key="not-needed",  # proxy doesn't require an API key
    )

    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            timeout=config.timeout,
        )
    except Exception as e:
        raise RuntimeError(f"Claude Max API Proxy error: {e}")

    elapsed = time.time() - t0
    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise RuntimeError(
            f"Claude Max API Proxy returned an empty response for model '{config.model}'."
        )

    return MeetingSummary(
        markdown=content,
        model=config.model,
        elapsed_seconds=elapsed,
        backend="claudemax",
    )


# ─── Generic OpenAI-compatible backend ────────────────────────────────────

def _summarize_openai(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a summarization request to any OpenAI-compatible API endpoint.

    Configured via environment variables:
        MEETSCRIBE_OPENAI_BASE_URL  — required (e.g. http://localhost:8000/v1)
        MEETSCRIBE_OPENAI_API_KEY   — optional (defaults to "not-needed")
    """
    import time

    base_url = os.environ.get("MEETSCRIBE_OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "MEETSCRIBE_OPENAI_BASE_URL environment variable is not set. "
            "Set it to the base URL of your OpenAI-compatible API."
        )

    api_key = os.environ.get("MEETSCRIBE_OPENAI_API_KEY", "not-needed")

    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    t0 = time.time()

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            timeout=config.timeout,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI-compatible API error ({base_url}): {e}")

    elapsed = time.time() - t0
    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise RuntimeError(
            f"OpenAI-compatible API returned an empty response for model '{config.model}'."
        )

    return MeetingSummary(
        markdown=content,
        model=config.model,
        elapsed_seconds=elapsed,
        backend="openai",
    )


# ─── Response validation ──────────────────────────────────────────────────

# Patterns that indicate the "summary" is actually an error response from
# an upstream API, not real meeting content.  These are checked as a
# defense-in-depth measure so that even if a backend proxy returns error
# text as a 200/valid completion, we catch it and trigger the fallback.
_ERROR_PATTERNS = re.compile(
    r'"type"\s*:\s*"error"'           # JSON error envelope
    r"|authentication_error"          # Anthropic auth failure
    r"|Invalid\s+(authentication\s+)?credentials"
    r"|Failed\s+to\s+authenticate"
    r"|rate_limit_error"
    r"|overloaded_error",
    re.IGNORECASE,
)


def _validate_summary_content(content: str, backend: str) -> None:
    """Raise RuntimeError if *content* looks like an error message, not a summary.

    This prevents upstream API errors (e.g. expired OAuth tokens returning
    401 error JSON) from being silently saved as the meeting summary.
    """
    # Short responses that match known error patterns are almost certainly
    # not real summaries (real summaries are typically 500+ chars).
    if len(content) < 400 and _ERROR_PATTERNS.search(content):
        raise RuntimeError(
            f"{backend} returned an error instead of a summary: "
            f"{content[:200]}"
        )


# ─── Core summarization (dispatcher with fallback chain) ──────────────────

def _dispatch(
    backend: str,
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Dispatch to a specific backend's summarization function.

    Creates a temporary config with the correct backend and model if
    falling back from the originally configured backend.
    """
    if backend != config.backend:
        # Build a new config for the fallback backend with its own default model
        fallback_config = SummaryConfig(
            backend=backend,
            model=_resolve_model(backend),
            ollama_url=config.ollama_url,
            timeout=config.timeout,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
        )
    else:
        fallback_config = config

    if backend == "claudemax":
        result = _summarize_claudemax(system_prompt, user_prompt, fallback_config)
    elif backend == "openrouter":
        result = _summarize_openrouter(system_prompt, user_prompt, fallback_config)
    elif backend == "openai":
        result = _summarize_openai(system_prompt, user_prompt, fallback_config)
    else:
        result = _summarize_ollama(system_prompt, user_prompt, fallback_config)

    # Defense-in-depth: catch error text masquerading as a valid summary
    _validate_summary_content(result.markdown, backend)
    return result


def summarize(
    transcript_text: str,
    config: SummaryConfig | None = None,
    language: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> MeetingSummary:
    """Generate a structured meeting summary from transcript text.

    Dispatches to the appropriate backend based on ``config.backend``.
    If the configured backend is unavailable, automatically tries the
    next backend in the fallback order: claudemax -> openrouter -> ollama.

    Args:
        transcript_text: The plain-text transcript (as produced by
            Transcript.to_text()).
        config: Summary configuration. Uses defaults if not provided.
        language: Language code of the transcript (e.g. "de", "fa").
            When provided (and not "en") the LLM is instructed to
            write the summary in that language.
        progress_callback: Optional callable(str) for status messages
            (e.g. reporting fallback attempts to the GUI/CLI).

    Returns:
        MeetingSummary with the Markdown summary, model used, and timing.

    Raises:
        ConnectionError: If no backend is reachable.
        RuntimeError: If all backends fail to generate a response.
    """
    if config is None:
        config = SummaryConfig()

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # Build prompts with language-aware section headers.
    system_prompt = _build_system_prompt(language)

    if language and language != "en":
        lang_name = _LANGUAGE_NAMES.get(language, language)
        user_prompt = USER_PROMPT_TEMPLATE_LANG.format(
            language=lang_name, transcript=transcript_text,
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript_text)

    # Build the list of backends to try: configured first, then fallback order
    backends_to_try = [config.backend]
    for fb in FALLBACK_ORDER:
        if fb not in backends_to_try:
            backends_to_try.append(fb)

    last_error = None
    for backend in backends_to_try:
        # Check availability before attempting
        avail_config = SummaryConfig(backend=backend)
        if not is_backend_available(avail_config):
            if backend == config.backend:
                _log(f"{backend} is unavailable: {_backend_not_available_message(avail_config)}")
            else:
                _log(f"Fallback {backend} also unavailable, skipping...")
            continue

        # If this is a fallback, log it
        if backend != config.backend:
            _log(f"Falling back to {backend} ({_resolve_model(backend)})...")

        try:
            result = _dispatch(backend, system_prompt, user_prompt, config)
            if backend != config.backend:
                _log(f"Summary generated via fallback backend {backend}")
            return result
        except Exception as exc:
            last_error = exc
            _log(f"{backend} failed: {exc}")
            continue

    # All backends failed
    raise RuntimeError(
        f"All summary backends failed. Last error: {last_error}"
    )
