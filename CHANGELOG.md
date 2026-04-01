# Changelog

## v0.3.0 — 2026-04-01

### New features

- **Multi-backend summarization** — supports four backends with automatic
  fallback: `claudemax` (Claude Max API Proxy), `openrouter` (OpenRouter API),
  `openai` (any OpenAI-compatible endpoint), and `ollama` (local). If the
  configured backend is unavailable, meetscribe automatically tries the next
  one. Use `--summary-backend` and `--summary-model` flags, or set
  `MEETSCRIBE_SUMMARY_BACKEND` / `MEETSCRIBE_SUMMARY_MODEL` env vars.

- **Generic OpenAI-compatible backend** — use any OpenAI-compatible API for
  summarization (Lemonade, LiteLLM, vLLM, LocalAI, self-hosted endpoints).
  Set `MEETSCRIBE_OPENAI_BASE_URL` and optionally `MEETSCRIBE_OPENAI_API_KEY`.

- **Voiceprint speaker recognition** — automatically identifies speakers across
  meetings using voice embeddings. After labeling a meeting, speaker profiles
  are stored in `~/.config/meet/speaker_profiles.json`. Future meetings match
  voices against the database using cosine similarity. Use `meet enroll` to
  build profiles from past sessions, or let the GUI update profiles
  automatically after each labeling.

- **Meeting sync** — push meeting artifacts (transcript, summary, PDF, SRT) to
  any configured Git repository on a schedule. Configure your repo URL and
  meeting schedule in `~/.config/meet/sync_config.json`. Use `meet sync` to
  push manually or let the GUI auto-sync after recording. Run
  `meet sync --init-config` to generate an example config.

- **Improved summarization prompts** — prompt templates extracted to standalone
  markdown files (`meet/prompts/summarize_system.md`, etc.) for easy iteration
  without touching Python code. Prompt rewritten for better results with
  local/open-source models: more information-dense, preserves technical
  specificity, captures implied action items, provides format guidance.

### Improvements

- Dynamic context window sizing for ollama — automatically sizes `num_ctx` to
  fit long transcripts (up to 64K tokens) instead of truncating.
- Response validation catches upstream API errors (expired tokens, rate limits)
  that would otherwise be silently saved as the meeting summary.
- Thinking mode explicitly disabled for ollama models (`think: false`) to avoid
  wasting tokens on hidden reasoning with models like GLM-4.7-flash and Qwen 3.5.
- GUI auto-sync guarded by `is_sync_configured()` — silently skips if no repo
  is configured.

### Testing

- All 81 existing tests pass with the new prompt loading system.

---

## v0.2.0 — 2026-03-14

### New features

- **Multilingual support** — Whisper large-v3-turbo supports 99 languages.
  meetscribe now passes language hints through the full pipeline: transcription,
  wav2vec2 alignment, Ollama summary (prompted in the source language), and PDF.
  Use `--language auto` (default) or specify a code: `en`, `de`, `tr`, `fr`,
  `es`, `fa`.

- **Farsi / RTL support** — Farsi transcripts render correctly in PDF using
  Noto Naskh Arabic with arabic-reshaper + python-bidi for right-to-left layout.
  Install optional deps with `pip install "meetscribe-offline[rtl]"`.

- **`meet label` CLI command** — assign real names to speakers after the fact.
  For each speaker: shows a summary table, plays a short audio clip from the
  correct stereo channel (via ffplay), prompts for a name. Regenerates all
  outputs (txt, srt, json, summary.md, pdf) with the new names. Options:
  `--no-audio`, `--no-summary`.

- **GUI speaker labeling dialog** — when 2+ speakers are detected, a dialog
  appears before results are saved. Shows each speaker's channel and a sample
  line. Labels are applied before writing any output files.

### Improvements

- PDF now uses DejaVu Sans for full Unicode coverage (replaces previous
  Latin-only font). Handles Cyrillic, Greek, Turkish special characters, etc.
- Ollama summary prompts are now language-aware: when a non-English language is
  detected, the prompt instructs the LLM to write the summary in that language.
- `post_process()` function centralises all output generation (txt, srt, json,
  pdf) so that `meet label` and the GUI dialog share the same code path.
- Shared utilities extracted to `meet/audio.py`, `meet/languages.py`,
  `meet/utils.py`, `meet/label.py` for cleaner architecture.

### Testing

- 81-test suite added covering `label`, `pdf`, `summarize`, `transcribe`, and
  `utils` modules.

### Package

- PyPI package renamed to `meetscribe-offline` to distinguish from an unrelated
  squatted project. Install with `pip install meetscribe-offline`.

---

## v0.1.0 — 2026-03-01

Initial release.

- Dual-channel audio capture (mic left, system audio right) via
  PipeWire/PulseAudio + ffmpeg
- WhisperX transcription (faster-whisper + wav2vec2 alignment)
- pyannote-audio speaker diarization with YOU/REMOTE channel mapping
- Ollama AI meeting summaries (qwen3.5:9b default)
- PDF output (summary + full transcript)
- Output formats: `.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`
- GTK3 GUI widget (always-on-top, record/stop, live timer, open results)
- CLI: `meet run`, `meet record`, `meet transcribe`, `meet gui`, `meet devices`,
  `meet check`
