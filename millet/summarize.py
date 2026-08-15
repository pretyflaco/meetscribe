"""Meeting summary generation using LLMs.

Supports multiple backends:
  - claudemax:  Claude Sonnet via claude-max-api-proxy on localhost:3457
                ($0 extra — uses existing Claude Max subscription).
  - tinfoil:    Hardware-attested TEE inference (requires TINFOIL_API_KEY).
  - openrouter: OpenRouter API (OpenAI-compatible, requires OPENROUTER_API_KEY).
  - ollama:     Local Ollama server (free, fully local).
  - openai:     Any OpenAI-compatible endpoint (opt-in; never in fallback).

Fallback chain: claudemax -> tinfoil -> openrouter -> ollama (see
FALLBACK_ORDER).  When the configured primary backend is unavailable, the
system automatically tries the next backend in the fallback order.  The
MILLET_SUMMARY_MODEL override applies to the user's chosen backend only; each
fallback backend uses its own hardcoded default model.

Configuration precedence (highest to lowest):
  1. Explicit keyword arguments / CLI flags (--summary-backend, --summary-model)
  2. Environment variables (MILLET_SUMMARY_BACKEND, MILLET_SUMMARY_MODEL)
  3. Hardcoded defaults (ollama / qwen3.5:9b)
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from millet.frontmatter import FrontmatterContext

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

# Tinfoil TEE defaults (hardware-enforced prompt privacy)
DEFAULT_TINFOIL_MODEL = "glm-5-2"
TINFOIL_API_KEY_ENV = "TINFOIL_API_KEY"
_TINFOIL_KEY_FILE = Path.home() / "models" / "tinfoil" / "tinfoil.txt"


def _resolve_tinfoil_api_key() -> str | None:
    """Resolve Tinfoil API key from env var or standard key file."""
    key = os.environ.get(TINFOIL_API_KEY_ENV)
    if key:
        return key.strip()
    if _TINFOIL_KEY_FILE.exists():
        try:
            return _TINFOIL_KEY_FILE.read_text().strip()
        except OSError:
            pass
    return None

# Supported backends
BACKENDS = ("ollama", "openrouter", "claudemax", "openai", "tinfoil")

# Fallback order: try claudemax first, then tinfoil, openrouter, then ollama
# (openai is not in fallback — it's opt-in only via explicit config)
FALLBACK_ORDER = ("claudemax", "tinfoil", "openrouter", "ollama")

# ─── Summarization presets ──────────────────────────────────────────────────
# Friendly names that map to backend+model pairs for the GUI/CLI dropdown.

SUMMARY_PRESETS = {
    "high-quality": {"backend": "claudemax", "model": "claude-sonnet-4-6"},
    "confidential": {"backend": "tinfoil",  "model": "glm-5-2"},
    "alternative":  {"backend": "openrouter", "model": "moonshotai/kimi-k2.6"},
}
DEFAULT_PRESET = "high-quality"

# Backward-compatible aliases (referenced by translate command, etc.)
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

from millet.languages import LANG_NAMES as _LANGUAGE_NAMES
from millet.languages import SECTION_HEADERS as _SECTION_HEADERS

# ─── Prompt loading ────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str | None:
    """Load a prompt template from the prompts directory. Returns None if missing."""
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _lang_instruction(language: str | None) -> str:
    """Return the appended CRITICAL language instruction line for a non-English language.

    Returns an empty string for English / unknown so prompts stay clean.
    """
    if not language or language == "en":
        return ""
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"\n- CRITICAL: Write the ENTIRE summary in {lang_name}, "
        f"including ALL section headers. Do NOT use any English text."
    )


def _build_system_prompt(language: str | None = None) -> str:
    """Build the system prompt with section headers in the target language."""
    lang = language or "en"
    h = _SECTION_HEADERS.get(lang, _SECTION_HEADERS["en"])

    lang_instruction = _lang_instruction(lang)

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

After the Markdown sections, append exactly ONE fenced JSON block with the same content as structured data:

```json
{{
  "participants": ["Alice", "Bob"],
  "topics": ["Topic name"],
  "action_items": [
    {{"assignee": "Alice", "task": "Send doc", "due": null, "status": "open"}}
  ],
  "decisions": [
    {{"text": "Use X over Y", "topic": null}}
  ]
}}
```

Rules:
- Use speaker labels exactly as they appear — do not rename or invent names
- Do not hallucinate — every item must be traceable to the transcript
- Be concise but information-dense
- Preserve technical specificity: name exact tools, APIs, frameworks mentioned
- Keep the summary professional and objective
- The JSON block: every field is REQUIRED. Use [] for empty lists, null for unknown assignee/due/topic. action_items.status must be one of "open", "closed", "blocked" — default to "open". The JSON content must be in English even when the body is in another language.{lang_instruction}"""


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


# ─── Two-pass (extract + format) prompts ──────────────────────────────────

def _extract_lang_instruction(language: str | None) -> str:
    """Lang instruction appended to Pass 1 (extraction) system prompt."""
    if not language or language == "en":
        return ""
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return f"\n- CRITICAL: Output the extracted lists in {lang_name}."


def _format_lang_instruction(language: str | None) -> str:
    """Lang instruction appended to Pass 2 (formatting) system prompt."""
    if not language or language == "en":
        return ""
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"\n- CRITICAL: Output everything in {lang_name}, including section headers. "
        "Do NOT use any English text."
    )


def _build_extract_system_prompt(language: str | None = None) -> str:
    """Build the Pass 1 (extraction) system prompt for the two-pass Ollama flow."""
    lang_instruction = _extract_lang_instruction(language)
    template = _load_prompt("summarize_extract_system.md")
    if template is not None:
        return template.format(lang_instruction=lang_instruction)
    # Inline fallback
    return (
        "You are a meeting transcript analyzer. Extract topics, actions, "
        "decisions, and questions from the transcript as plain numbered "
        f"lists.{lang_instruction}"
    )


def _build_format_system_prompt(language: str | None = None) -> str:
    """Build the Pass 2 (formatting) system prompt for the two-pass Ollama flow."""
    lang = language or "en"
    h = _SECTION_HEADERS.get(lang, _SECTION_HEADERS["en"])
    lang_instruction = _format_lang_instruction(lang)
    template = _load_prompt("summarize_format_system.md")
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
    # Inline fallback
    return (
        f"Format the extracted meeting data into Markdown with sections: "
        f"## {h['overview']}, ## {h['topics']}, ## {h['actions']}, "
        f"## {h['decisions']}, ## {h['questions']}.\n\n"
        "After the Markdown sections, append exactly ONE fenced ```json block "
        'with keys "participants", "topics", "action_items", "decisions". '
        "Every field is REQUIRED — use [] for empty lists, null for unknown "
        'assignee/due/topic. action_items.status must be one of "open", '
        '"closed", or "blocked". JSON must be in English even when the body '
        f"is in another language.{lang_instruction}"
    )


def _load_extract_user_template() -> str:
    template = _load_prompt("summarize_extract_user.md")
    if template is not None:
        return template
    return (
        "Extract all topics, actions, decisions, and questions from this "
        "transcript:\n\n---\n{transcript}\n---"
    )


def _load_format_user_template() -> str:
    template = _load_prompt("summarize_format_user.md")
    if template is not None:
        return template
    return (
        "Organize the following extracted meeting data into the required "
        "format:\n\n---\n{extracted}\n---"
    )


USER_PROMPT_TEMPLATE = _load_user_prompt_template()

USER_PROMPT_TEMPLATE_LANG = _load_user_prompt_template_lang()


# ─── Data classes ───────────────────────────────────────────────────────────

def _resolve_backend() -> str:
    """Resolve the default backend from env var or hardcoded default."""
    from .paths import getenv_renamed
    return getenv_renamed(
        "MILLET_SUMMARY_BACKEND", "MEETSCRIBE_SUMMARY_BACKEND",
        default="ollama",
    ).lower()


def _default_model_for_backend(backend: str) -> str:
    """Hardcoded default model for a backend, ignoring any env override."""
    if backend == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    if backend == "claudemax":
        return DEFAULT_CLAUDEMAX_MODEL
    if backend == "openai":
        return DEFAULT_OPENAI_COMPAT_MODEL
    if backend == "tinfoil":
        return DEFAULT_TINFOIL_MODEL
    return DEFAULT_OLLAMA_MODEL


def _resolve_model(backend: str) -> str:
    """Resolve the default model for a backend from env var or hardcoded default.

    The ``MILLET_SUMMARY_MODEL`` env override applies to the user's *chosen*
    backend only.  For fallback backends (see :func:`_default_model_for_backend`)
    the env model must be ignored — a model name valid for e.g. Ollama would
    otherwise be forced onto OpenRouter/claudemax and fail the whole chain.
    """
    from .paths import getenv_renamed
    env_model = getenv_renamed("MILLET_SUMMARY_MODEL", "MEETSCRIBE_SUMMARY_MODEL")
    if env_model:
        return env_model
    return _default_model_for_backend(backend)


def _resolve_ollama_singlepass() -> bool:
    """Resolve the default for the ollama single-pass opt-out from the env var."""
    from .paths import getenv_renamed
    raw = (getenv_renamed(
        "MILLET_OLLAMA_SINGLEPASS", "MEETSCRIBE_OLLAMA_SINGLEPASS", default="",
    ) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


@dataclass
class SummaryConfig:
    """Configuration for meeting summary generation.

    Supports multiple backends. The ``backend`` and ``model`` fields
    respect environment variables when left at their sentinel values:

        MILLET_SUMMARY_BACKEND      -> backend  (default: "ollama")
        MILLET_SUMMARY_MODEL        -> model    (default: per-backend)
        OPENROUTER_API_KEY          -> required for openrouter backend
    """

    backend: str | None = None   # None = resolve from env/default
    model: str | None = None     # None = resolve from env/default per backend
    preset: str | None = None    # None = no preset; "high-quality"|"confidential"|"alternative"
    ollama_url: str = OLLAMA_BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = 0.3
    num_ctx: int = 8192  # Ollama-specific context window
    ollama_singlepass: bool | None = None  # None = resolve from env (default: two-pass)

    def __post_init__(self):
        # Resolve preset: explicit arg > env var > None
        if self.preset is None:
            from .paths import getenv_renamed
            self.preset = getenv_renamed(
                "MILLET_SUMMARY_PRESET", "MEETSCRIBE_SUMMARY_PRESET",
            )
        if self.preset:
            self.preset = self.preset.lower().strip()
            if self.preset in SUMMARY_PRESETS:
                p = SUMMARY_PRESETS[self.preset]
                # Preset sets backend/model only if not explicitly provided
                if self.backend is None:
                    self.backend = p["backend"]
                if self.model is None:
                    self.model = p["model"]

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

        # Resolve ollama two-pass opt-out: explicit arg > env > False (two-pass on)
        if self.ollama_singlepass is None:
            self.ollama_singlepass = _resolve_ollama_singlepass()


@dataclass
class MeetingSummary:
    """Result of a meeting summary generation.

    ``markdown`` always holds the human-readable Markdown body suitable for
    PDF rendering — never the trailing JSON data block.  When the LLM
    emitted a structured data block (the contract since schema_version 1),
    the parsed dict is stashed in ``data`` and the body is stripped before
    storage.  See ``meet.frontmatter`` for the schema.
    """

    markdown: str
    model: str
    elapsed_seconds: float
    backend: str = ""
    # Optional fields populated by the two-pass Ollama flow
    pass1_seconds: float | None = None
    pass2_seconds: float | None = None
    pass1_chars: int | None = None
    extraction: str | None = None  # Raw Pass 1 output (kept in-memory only)
    # Structured data parsed out of the LLM completion (schema_version 1).
    # ``None`` means the model didn't emit a JSON block or it failed to
    # parse; ``data_error`` records why so the indexer can flag it.
    data: dict[str, Any] | None = None
    data_error: str | None = None

    def save(
        self,
        output_dir: str | Path,
        basename: str,
        *,
        frontmatter_context: FrontmatterContext | None = None,
        lang_suffix: str | None = None,
    ) -> Path:
        """Save the summary as a ``.summary.md`` file plus sidecars.

        When ``frontmatter_context`` is provided, the saved Markdown is
        prefixed with a YAML frontmatter block built from the LLM's
        structured data + the session-level context.  A
        ``.frontmatter.json`` sidecar is also written so consumers that
        don't want to parse YAML can read the same data verbatim.

        When ``frontmatter_context`` is ``None`` we fall back to the
        legacy behavior (raw Markdown body, no frontmatter) for callers
        that haven't been migrated yet.

        The ``.summary.meta.json`` sidecar continues to record which
        backend/model produced the summary, plus per-pass timings for
        the two-pass Ollama flow.

        ``lang_suffix`` (e.g. ``"de"``) writes an ADDITIONAL, language-tagged
        summary — ``<basename>.summary.de.md`` (with matching
        ``.summary.de.meta.json`` and ``<basename>.de.frontmatter.json``
        sidecars) — without clobbering the primary auto-detected
        ``<basename>.summary.md``.  When ``None`` the primary filename is used.

        Returns the path to the saved ``.summary[.<lang>].md`` file.
        """
        import datetime

        # Local import to avoid a circular import at module load time.
        from millet.frontmatter import (
            build_frontmatter,
            render_frontmatter_block,
            write_frontmatter_sidecar,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # A language-tagged additional summary uses a suffixed name so it
        # coexists with the primary <basename>.summary.md.
        suffix = f".{lang_suffix}" if lang_suffix else ""
        md_path = output_dir / f"{basename}.summary{suffix}.md"
        # Sidecar basename carries the suffix too (frontmatter writer appends
        # ".frontmatter.json"), so additional-language sidecars don't clobber
        # the primary's.
        sidecar_basename = f"{basename}{suffix}"

        if frontmatter_context is not None:
            fm = build_frontmatter(
                self.data,
                frontmatter_context,
                extraction_error=self.data_error,
            )
            md_path.write_text(
                render_frontmatter_block(fm) + self.markdown,
                encoding="utf-8",
            )
            write_frontmatter_sidecar(output_dir, sidecar_basename, fm)
        else:
            md_path.write_text(self.markdown, encoding="utf-8")

        meta: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if self.pass1_seconds is not None:
            meta["mode"] = "two_pass"
            meta["pass1_seconds"] = round(self.pass1_seconds, 2)
            meta["pass2_seconds"] = round(self.pass2_seconds or 0.0, 2)
            if self.pass1_chars is not None:
                meta["pass1_chars"] = self.pass1_chars
        if self.data_error:
            meta["data_error"] = self.data_error
        elif self.data is not None:
            meta["data_extracted"] = True
        meta_path = output_dir / f"{basename}.summary{suffix}.meta.json"
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
    elif config.backend == "tinfoil":
        return bool(_resolve_tinfoil_api_key())
    elif config.backend == "openai":
        from .paths import getenv_renamed
        return bool(getenv_renamed(
            "MILLET_OPENAI_BASE_URL", "MEETSCRIBE_OPENAI_BASE_URL",
        ))
    else:
        return is_ollama_available(config.ollama_url)


def _backend_not_available_message(config: SummaryConfig) -> str:
    """Return a user-friendly message when the backend is unavailable."""
    if config.backend == "claudemax":
        return (
            "Claude Max API Proxy is not running at localhost:3457. "
            "Start it with: systemctl --user start claude-max-proxy"
        )
    if config.backend == "openrouter":
        return (
            "OPENROUTER_API_KEY is not set. "
            "Export it or use --summary-backend ollama."
        )
    if config.backend == "tinfoil":
        return (
            f"TINFOIL_API_KEY is not set and key file {_TINFOIL_KEY_FILE} "
            "not found. Get an API key at https://tinfoil.sh"
        )
    if config.backend == "openai":
        return (
            "MILLET_OPENAI_BASE_URL is not set. "
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


def _call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
    *,
    num_ctx: int | None = None,
    output_reserve: int = 4096,
    timeout: int | None = None,
    temperature: float | None = None,
) -> tuple[str, float]:
    """Single Ollama /api/chat call. Returns (content, elapsed_seconds).

    Raises ConnectionError if Ollama is unreachable, RuntimeError on API
    error / empty response.  Used by both the single-pass and two-pass flows.

    ``output_reserve`` is the token budget reserved for the model's own
    output when ``num_ctx`` is auto-sized.  The two-pass extraction (Pass 1)
    overrides the 4096 default: thinking-heavy models like qwen3.8:27b emit
    long exhaustive lists and, with only 4096 reserved, hit the context
    window mid-extraction (``done_reason: length``) and silently truncate.
    """
    import time

    if not is_ollama_available(config.ollama_url):
        raise ConnectionError(
            f"Ollama is not running at {config.ollama_url}. "
            "Start it with: ollama serve"
        )

    if num_ctx is None:
        num_ctx = _dynamic_num_ctx(
            system_prompt, user_prompt, floor=config.num_ctx,
            output_reserve=output_reserve,
        )
    if timeout is None:
        timeout = config.timeout
    if temperature is None:
        temperature = config.temperature

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # Disable thinking/reasoning for speed
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    url = f"{config.ollama_url}/api/chat"
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.Timeout as e:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s. "
            f"The model '{config.model}' may be too large or slow. "
            "Try a smaller model with --summary-model."
        ) from e
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}") from e
    elapsed = time.time() - t0
    data = resp.json()
    content = (data.get("message", {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(
            f"Ollama returned an empty response. Model '{config.model}' may "
            "not be available. Check with: ollama list"
        )
    return content, elapsed


def _summarize_ollama(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a single-pass summarization request to local Ollama."""
    content, elapsed = _call_ollama_chat(system_prompt, user_prompt, config)
    return MeetingSummary(
        markdown=content,
        model=config.model,
        elapsed_seconds=elapsed,
        backend="ollama",
    )


def _summarize_ollama_twopass(
    transcript_text: str,
    config: SummaryConfig,
    language: str | None = None,
) -> MeetingSummary:
    """Two-pass Ollama summarization: extract (Pass 1) then format (Pass 2).

    Pass 1 uses a wide context window sized to the transcript and a long
    timeout to extract topics/actions/decisions/questions as plain numbered
    lists.  Pass 2 takes the much smaller extracted data and formats it into
    the canonical Markdown structure with a fixed 8K context window and a
    shorter timeout.

    This dramatically improves format compliance and reduces hallucinations
    on local 20B-class models like gpt-oss:20b and qwen3.6:27b, at the cost
    of one additional LLM call (typically ~30-90s extra).
    """
    # ── Pass 1: extraction ────────────────────────────────────────────────
    extract_sys = _build_extract_system_prompt(language)
    extract_user_tmpl = _load_extract_user_template()
    extract_user = extract_user_tmpl.format(transcript=transcript_text)
    extracted, t1 = _call_ollama_chat(
        extract_sys, extract_user, config,
        # Pass 1 needs the full transcript to fit AND room for a long
        # exhaustive extraction. Reserve 16K output tokens (not the 4K
        # default) so thinking-heavy models don't truncate mid-list.
        num_ctx=None,
        output_reserve=16384,
        timeout=config.timeout,
        temperature=config.temperature,
    )

    # ── Pass 2: formatting ────────────────────────────────────────────────
    format_sys = _build_format_system_prompt(language)
    format_user_tmpl = _load_format_user_template()
    format_user = format_user_tmpl.format(extracted=extracted)
    # Pass 2 input is small (the extracted lists). Cap context at 8K and use a
    # shorter timeout so we fail fast if something goes wrong.
    pass2_timeout = min(config.timeout, 240)
    formatted, t2 = _call_ollama_chat(
        format_sys, format_user, config,
        num_ctx=8192,
        timeout=pass2_timeout,
        temperature=config.temperature,
    )

    return MeetingSummary(
        markdown=formatted,
        model=config.model,
        elapsed_seconds=t1 + t2,
        backend="ollama",
        pass1_seconds=t1,
        pass2_seconds=t2,
        pass1_chars=len(extracted),
        extraction=extracted,
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

    # Kimi K2.6 burns all output tokens on hidden reasoning if not disabled,
    # producing empty visible content.  Disable reasoning for known models.
    extra_kwargs: dict = {}
    if "kimi" in config.model.lower():
        extra_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            timeout=config.timeout,
            **extra_kwargs,
        )
    except Exception as e:
        raise RuntimeError(f"OpenRouter API error: {e}") from e

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


# ─── Tinfoil TEE backend ──────────────────────────────────────────────────

# Number of attempts + base backoff for the Tinfoil path.  The SDK does
# a network fetch at client construction (GET https://atc.tinfoil.sh/routers)
# AND for the inference call; on hosts with flaky DNS a single transient
# lookup failure used to hard-fail the whole summarization.  Retry both.
_TINFOIL_MAX_ATTEMPTS = 3
_TINFOIL_BACKOFF_BASE = 2.0  # seconds: ~2s, 4s, 8s


def _is_transient_network_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient connectivity/DNS blip worth
    retrying (as opposed to a genuine auth/model/config error that won't
    improve on retry).

    The Tinfoil SDK surfaces a DNS failure during router discovery as
    ``ValueError("Failed to fetch router addresses: <urlopen error ...>")``;
    other backends raise socket/urllib/httpx connection errors.
    """
    import socket
    import urllib.error

    transient_types: tuple[type[BaseException], ...] = (
        socket.gaierror,
        socket.timeout,
        ConnectionError,
        TimeoutError,
        urllib.error.URLError,
    )
    try:
        import httpx
        transient_types += (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
        )
    except Exception:
        pass

    if isinstance(exc, transient_types):
        return True
    # Walk the cause/context chain (the SDK wraps URLError in ValueError).
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, transient_types):
            return True
        cur = cur.__cause__ or cur.__context__
    # Last resort: match the SDK's router-discovery message + common DNS text.
    msg = str(exc).lower()
    markers = (
        "failed to fetch router addresses",
        "name or service not known",
        "temporary failure in name resolution",
        "no address associated with hostname",
        "connection reset",
        "connection refused",
        "timed out",
    )
    return any(m in msg for m in markers)


def _summarize_tinfoil(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a summarization request to Tinfoil TEE (hardware-private inference).

    Prompts are encrypted into the secure enclave and processed inside
    NVIDIA H100 Confidential Computing.  The provider cannot see the data.

    Resilience (v0.9.2): both the client construction (which fetches the
    enclave router list over the network) and the completion call are
    retried on transient connectivity/DNS errors with exponential
    backoff.  Genuine auth/model errors fail fast (no retry).
    """
    import time

    api_key = _resolve_tinfoil_api_key()
    if not api_key:
        raise RuntimeError(
            f"{TINFOIL_API_KEY_ENV} environment variable is not set and "
            f"key file {_TINFOIL_KEY_FILE} not found. "
            "Get an API key at https://tinfoil.sh"
        )

    from tinfoil import TinfoilAI

    t0 = time.time()
    response = None

    for attempt in range(1, _TINFOIL_MAX_ATTEMPTS + 1):
        try:
            # Client init does a network fetch (router discovery) — keep it
            # inside the retry so a DNS blip here doesn't hard-fail.
            client = TinfoilAI(api_key=api_key)
            # timeout: the only backend call that previously omitted it —
            # a stalled TLS connection to the enclave hung the whole
            # pipeline indefinitely, worst for the `confidential` preset
            # which (by design) has no fallback. The Tinfoil SDK is
            # OpenAI-compatible and honors per-request timeouts.
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                timeout=config.timeout,
            )
            break
        except Exception as e:
            if attempt < _TINFOIL_MAX_ATTEMPTS and _is_transient_network_error(e):
                wait = _TINFOIL_BACKOFF_BASE ** attempt
                import logging
                logging.getLogger("millet.summarize").warning(
                    "Tinfoil attempt %d/%d hit a transient network/DNS error; "
                    "retrying in %.0fs: %s",
                    attempt, _TINFOIL_MAX_ATTEMPTS, wait, e,
                )
                time.sleep(wait)
                continue
            # Either out of attempts, or a non-transient (auth/model) error.
            if _is_transient_network_error(e):
                raise RuntimeError(
                    "Tinfoil TEE unreachable after "
                    f"{_TINFOIL_MAX_ATTEMPTS} attempts (transient network/DNS "
                    f"reaching atc.tinfoil.sh): {e}"
                ) from e
            raise RuntimeError(f"Tinfoil TEE API error: {e}") from e

    elapsed = time.time() - t0
    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise RuntimeError(
            f"Tinfoil returned an empty response for model '{config.model}'."
        )

    return MeetingSummary(
        markdown=content,
        model=f"{config.model} (TEE)",
        elapsed_seconds=elapsed,
        backend="tinfoil",
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
            "Claude Max API Proxy is not running at localhost:3457. "
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
        raise RuntimeError(f"Claude Max API Proxy error: {e}") from e

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
        MILLET_OPENAI_BASE_URL  — required (e.g. http://localhost:8000/v1)
        MILLET_OPENAI_API_KEY   — optional (defaults to "not-needed")
    """
    import time

    from .paths import getenv_renamed
    base_url = getenv_renamed("MILLET_OPENAI_BASE_URL", "MEETSCRIBE_OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "MILLET_OPENAI_BASE_URL environment variable is not set. "
            "Set it to the base URL of your OpenAI-compatible API."
        )

    api_key = getenv_renamed(
        "MILLET_OPENAI_API_KEY", "MEETSCRIBE_OPENAI_API_KEY", default="not-needed",
    )

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
        raise RuntimeError(f"OpenAI-compatible API error ({base_url}): {e}") from e

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
    *,
    transcript_text: str | None = None,
    language: str | None = None,
) -> MeetingSummary:
    """Dispatch to a specific backend's summarization function.

    Creates a temporary config with the correct backend and model if
    falling back from the originally configured backend.

    For the ollama backend, uses the two-pass (extract+format) flow by
    default unless ``config.ollama_singlepass`` is True.  Two-pass requires
    ``transcript_text`` and ``language`` to be passed through.
    """
    if backend != config.backend:
        # Build a new config for the fallback backend with its own default
        # model.  Use the hardcoded default (not _resolve_model) so a user's
        # MILLET_SUMMARY_MODEL set for the primary backend does not leak into
        # a different fallback backend and fail the entire chain.
        fallback_config = SummaryConfig(
            backend=backend,
            model=_default_model_for_backend(backend),
            ollama_url=config.ollama_url,
            timeout=config.timeout,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
            ollama_singlepass=config.ollama_singlepass,
        )
    else:
        fallback_config = config

    if backend == "claudemax":
        result = _summarize_claudemax(system_prompt, user_prompt, fallback_config)
    elif backend == "openrouter":
        result = _summarize_openrouter(system_prompt, user_prompt, fallback_config)
    elif backend == "tinfoil":
        result = _summarize_tinfoil(system_prompt, user_prompt, fallback_config)
    elif backend == "openai":
        result = _summarize_openai(system_prompt, user_prompt, fallback_config)
    else:
        # Ollama: prefer the two-pass flow unless explicitly opted out
        if not fallback_config.ollama_singlepass and transcript_text is not None:
            result = _summarize_ollama_twopass(
                transcript_text, fallback_config, language=language,
            )
        else:
            result = _summarize_ollama(system_prompt, user_prompt, fallback_config)

    # Split the trailing JSON data block off the markdown body so that
    # PDF rendering keeps using the body and frontmatter writers can
    # consume the parsed data.  Done once here so every backend benefits.
    from millet.frontmatter import split_body_and_data

    body, data, data_error = split_body_and_data(result.markdown)
    result.markdown = body
    result.data = data
    # Only record an error if extraction failed AND the prompt should
    # have produced data; "no JSON block found" on a model that ignored
    # the contract is still useful diagnostic info, so keep it.
    if data is None:
        result.data_error = data_error

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

    # When a preset was explicitly selected (e.g. "confidential"), the user
    # chose a specific privacy/quality level.  Silently falling back to a
    # different backend would violate that expectation — especially for the
    # "confidential" preset where falling back to a trust-based provider
    # defeats the purpose.  Fail loudly instead.
    if config.preset and config.preset in SUMMARY_PRESETS:
        avail_config = SummaryConfig(
            backend=config.backend, ollama_url=config.ollama_url,
        )
        if not is_backend_available(avail_config):
            msg = _backend_not_available_message(avail_config)
            preset_label = config.preset
            raise RuntimeError(
                f"Summarization preset '{preset_label}' requires the "
                f"'{config.backend}' backend, but it is unavailable: {msg}\n"
                f"Set the required environment variable and try again."
            )

    # Build the list of backends to try: configured first, then fallback order
    backends_to_try = [config.backend]
    for fb in FALLBACK_ORDER:
        if fb not in backends_to_try:
            backends_to_try.append(fb)

    last_error = None
    for backend in backends_to_try:
        # Check availability before attempting.  Carry the caller's ollama_url
        # so a custom Ollama server isn't reported unavailable just because the
        # probe defaulted to localhost.
        avail_config = SummaryConfig(backend=backend, ollama_url=config.ollama_url)
        if not is_backend_available(avail_config):
            if backend == config.backend:
                _log(f"{backend} is unavailable: {_backend_not_available_message(avail_config)}")
            else:
                _log(f"Fallback {backend} also unavailable, skipping...")
            continue

        # If this is a fallback, log it (with the model actually used)
        if backend != config.backend:
            _log(f"Falling back to {backend} ({_default_model_for_backend(backend)})...")

        # Inform the user when the local two-pass flow is about to run, since
        # it takes noticeably longer than a single LLM call.
        if backend == "ollama" and not config.ollama_singlepass:
            _log("Running Ollama two-pass summarization (extract + format)...")

        try:
            result = _dispatch(
                backend, system_prompt, user_prompt, config,
                transcript_text=transcript_text, language=language,
            )
            if backend != config.backend:
                _log(f"Summary generated via fallback backend {backend}")
            return result
        except Exception as exc:
            last_error = exc
            _log(f"{backend} failed: {exc}")
            # When a preset was explicitly selected, do NOT silently fall
            # back to a different backend — the user chose a specific
            # privacy/quality level.  Re-raise so the failure is visible.
            if config.preset and config.preset in SUMMARY_PRESETS and backend == config.backend:
                raise
            continue

    # All backends failed
    raise RuntimeError(
        f"All summary backends failed. Last error: {last_error}"
    )
