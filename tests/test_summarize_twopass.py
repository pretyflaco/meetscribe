"""Tests for the two-pass (extract + format) Ollama summarization flow."""

from __future__ import annotations

import json

import pytest

from millet import summarize as sm
from millet.summarize import (
    DEFAULT_CLAUDEMAX_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    MeetingSummary,
    SummaryConfig,
    _build_extract_system_prompt,
    _build_format_system_prompt,
    _default_model_for_backend,
    _dispatch,
    _dynamic_num_ctx,
    _resolve_model,
    _resolve_ollama_singlepass,
)

# ─── Dynamic context sizing ────────────────────────────────────────────────

class TestDynamicNumCtx:
    def test_output_reserve_widens_context(self):
        # A larger output reserve must grow the computed context so a long
        # extraction has room; Pass 1 relies on this to avoid truncation.
        sys_p = "system " * 200
        user = "word " * 4000  # a sizeable transcript
        small = _dynamic_num_ctx(sys_p, user, output_reserve=4096)
        large = _dynamic_num_ctx(sys_p, user, output_reserve=16384)
        assert large > small
        # The delta reflects the extra reserve (rounded to 1024).
        assert large - small == pytest.approx(16384 - 4096, abs=1024)

    def test_respects_ceiling(self):
        huge = "x " * 40000
        assert _dynamic_num_ctx(huge, huge, output_reserve=16384) <= 65536

# ─── Fallback model resolution ─────────────────────────────────────────────

class TestFallbackModelResolution:
    def test_default_model_ignores_env_override(self, monkeypatch):
        # A user's MILLET_SUMMARY_MODEL (set for their chosen backend) must
        # not leak into a different fallback backend and fail the chain.
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "some-ollama-only:9b")
        assert _default_model_for_backend("openrouter") == DEFAULT_OPENROUTER_MODEL
        assert _default_model_for_backend("claudemax") == DEFAULT_CLAUDEMAX_MODEL

    def test_resolve_model_honors_env_for_chosen_backend(self, monkeypatch):
        monkeypatch.setenv("MILLET_SUMMARY_MODEL", "my-model:latest")
        assert _resolve_model("ollama") == "my-model:latest"

# ─── Env-var resolution ────────────────────────────────────────────────────

class TestResolveOllamaSinglepass:
    @pytest.mark.parametrize("val", ["1", "true", "True", "YES", "on", "On"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("MEETSCRIBE_OLLAMA_SINGLEPASS", val)
        assert _resolve_ollama_singlepass() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "anything"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("MEETSCRIBE_OLLAMA_SINGLEPASS", val)
        assert _resolve_ollama_singlepass() is False

    def test_unset_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("MEETSCRIBE_OLLAMA_SINGLEPASS", raising=False)
        assert _resolve_ollama_singlepass() is False


class TestSummaryConfigOllamaSinglepass:
    def test_default_resolves_from_env_unset(self, monkeypatch):
        monkeypatch.delenv("MEETSCRIBE_OLLAMA_SINGLEPASS", raising=False)
        cfg = SummaryConfig(backend="ollama")
        assert cfg.ollama_singlepass is False

    def test_default_resolves_from_env_set(self, monkeypatch):
        monkeypatch.setenv("MEETSCRIBE_OLLAMA_SINGLEPASS", "1")
        cfg = SummaryConfig(backend="ollama")
        assert cfg.ollama_singlepass is True

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MEETSCRIBE_OLLAMA_SINGLEPASS", "1")
        cfg = SummaryConfig(backend="ollama", ollama_singlepass=False)
        assert cfg.ollama_singlepass is False


# ─── Two-pass system prompts ───────────────────────────────────────────────

class TestExtractSystemPrompt:
    def test_english_no_lang_instruction(self):
        prompt = _build_extract_system_prompt("en")
        assert "CRITICAL: Output the extracted lists in" not in prompt

    def test_german_lang_instruction(self):
        prompt = _build_extract_system_prompt("de")
        assert "German" in prompt

    def test_none_treated_as_english(self):
        prompt = _build_extract_system_prompt(None)
        assert "CRITICAL: Output the extracted lists in" not in prompt


class TestFormatSystemPrompt:
    def test_english_headers(self):
        prompt = _build_format_system_prompt("en")
        # Should reference English headers
        assert "Meeting Overview" in prompt
        assert "Key Topics Discussed" in prompt
        assert "Action Items" in prompt
        assert "Decisions Made" in prompt
        assert "Open Questions" in prompt
        assert "CRITICAL: Output everything in" not in prompt

    def test_german_headers_and_lang_instruction(self):
        prompt = _build_format_system_prompt("de")
        h = sm._SECTION_HEADERS["de"]
        assert h["overview"] in prompt
        assert h["topics"] in prompt
        assert "German" in prompt


# ─── Two-pass call flow (mocked) ───────────────────────────────────────────

class TestSummarizeOllamaTwopass:
    def test_calls_pass1_then_pass2(self, monkeypatch):
        calls: list[dict] = []

        def fake_call(system_prompt, user_prompt, config, *, num_ctx=None,
                      output_reserve=4096, timeout=None, temperature=None):
            calls.append({
                "system": system_prompt,
                "user": user_prompt,
                "num_ctx": num_ctx,
                "output_reserve": output_reserve,
                "timeout": timeout,
            })
            if len(calls) == 1:
                return ("1. Topic A\n2. Action B", 12.5)
            return ("## Meeting Overview\nA test meeting.\n", 3.5)

        monkeypatch.setattr(sm, "_call_ollama_chat", fake_call)
        cfg = SummaryConfig(backend="ollama", ollama_singlepass=False)

        result = sm._summarize_ollama_twopass(
            "Some long transcript text here.", cfg, language="en",
        )

        assert len(calls) == 2
        # Pass 1: dynamic num_ctx (None passed in), full timeout, and a wide
        # output reserve so thinking-heavy models don't truncate mid-extraction
        # (qwen3.8:27b hit done_reason=length with the 4096 default).
        assert calls[0]["num_ctx"] is None
        assert calls[0]["output_reserve"] == 16384
        assert calls[0]["timeout"] == cfg.timeout
        # Pass 2: capped at 8192 ctx, capped at 240s, default reserve
        assert calls[1]["num_ctx"] == 8192
        assert calls[1]["output_reserve"] == 4096
        assert calls[1]["timeout"] == min(cfg.timeout, 240)
        # Pass 2 user prompt should contain the Pass 1 extraction
        assert "1. Topic A" in calls[1]["user"]

        # Result populated correctly
        assert result.markdown.startswith("## Meeting Overview")
        assert result.backend == "ollama"
        assert result.pass1_seconds == 12.5
        assert result.pass2_seconds == 3.5
        assert result.elapsed_seconds == pytest.approx(16.0)
        assert result.pass1_chars == len("1. Topic A\n2. Action B")
        assert result.extraction == "1. Topic A\n2. Action B"


# ─── MeetingSummary.save sidecar ────────────────────────────────────────────

class TestMeetingSummarySaveTwoPassSidecar:
    def test_two_pass_sidecar_fields(self, tmp_path):
        summary = MeetingSummary(
            markdown="## Meeting Overview\nHi.",
            model="gpt-oss:20b",
            elapsed_seconds=63.2,
            backend="ollama",
            pass1_seconds=58.1,
            pass2_seconds=5.1,
            pass1_chars=1234,
            extraction="1. Topic\n2. Action",
        )
        summary.save(tmp_path, "meeting-test")

        sidecar = tmp_path / "meeting-test.summary.meta.json"
        meta = json.loads(sidecar.read_text())
        assert meta["mode"] == "two_pass"
        assert meta["backend"] == "ollama"
        assert meta["model"] == "gpt-oss:20b"
        assert meta["pass1_seconds"] == 58.1
        assert meta["pass2_seconds"] == 5.1
        assert meta["pass1_chars"] == 1234
        # extraction is in-memory only — never persisted
        assert "extraction" not in meta

    def test_single_pass_sidecar_omits_two_pass_fields(self, tmp_path):
        summary = MeetingSummary(
            markdown="## Meeting Overview\nHi.",
            model="gpt-oss:20b",
            elapsed_seconds=42.0,
            backend="ollama",
        )
        summary.save(tmp_path, "meeting-test")

        meta = json.loads(
            (tmp_path / "meeting-test.summary.meta.json").read_text()
        )
        assert "mode" not in meta
        assert "pass1_seconds" not in meta
        assert "pass2_seconds" not in meta


# ─── Dispatcher routing ────────────────────────────────────────────────────

class TestDispatchOllamaRouting:
    def test_default_routes_to_twopass(self, monkeypatch):
        seen: list[str] = []

        def fake_twopass(transcript_text, config, language=None):
            seen.append("twopass")
            return MeetingSummary(
                markdown="## Meeting Overview\nx" * 100,
                model=config.model, elapsed_seconds=1.0, backend="ollama",
            )

        def fake_singlepass(system_prompt, user_prompt, config):
            seen.append("singlepass")
            return MeetingSummary(
                markdown="## Meeting Overview\nx" * 100,
                model=config.model, elapsed_seconds=1.0, backend="ollama",
            )

        monkeypatch.setattr(sm, "_summarize_ollama_twopass", fake_twopass)
        monkeypatch.setattr(sm, "_summarize_ollama", fake_singlepass)

        cfg = SummaryConfig(backend="ollama", ollama_singlepass=False)
        _dispatch(
            "ollama", "sys", "usr", cfg,
            transcript_text="some transcript", language="en",
        )
        assert seen == ["twopass"]

    def test_singlepass_when_opted_out(self, monkeypatch):
        seen: list[str] = []

        def fake_twopass(transcript_text, config, language=None):
            seen.append("twopass")
            return MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="ollama",
            )

        def fake_singlepass(system_prompt, user_prompt, config):
            seen.append("singlepass")
            return MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="ollama",
            )

        monkeypatch.setattr(sm, "_summarize_ollama_twopass", fake_twopass)
        monkeypatch.setattr(sm, "_summarize_ollama", fake_singlepass)

        cfg = SummaryConfig(backend="ollama", ollama_singlepass=True)
        _dispatch(
            "ollama", "sys", "usr", cfg,
            transcript_text="some transcript", language="en",
        )
        assert seen == ["singlepass"]

    def test_singlepass_when_no_transcript_text(self, monkeypatch):
        """If transcript_text is not plumbed through, fall back to singlepass."""
        seen: list[str] = []

        def fake_twopass(*a, **kw):
            seen.append("twopass")
            return MeetingSummary(
                markdown="x" * 500, model="m", elapsed_seconds=1.0,
                backend="ollama",
            )

        def fake_singlepass(system_prompt, user_prompt, config):
            seen.append("singlepass")
            return MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="ollama",
            )

        monkeypatch.setattr(sm, "_summarize_ollama_twopass", fake_twopass)
        monkeypatch.setattr(sm, "_summarize_ollama", fake_singlepass)

        cfg = SummaryConfig(backend="ollama", ollama_singlepass=False)
        _dispatch("ollama", "sys", "usr", cfg, transcript_text=None)
        assert seen == ["singlepass"]
