# Research: ASR / diarization / alignment tech refresh (July 2026)

Status: **research** (2026-07-30). Survey of speech-AI developments since millet's
first release (March 2026) that could improve its core pipeline, with a scoped,
license-aware upgrade plan. **No streaming/real-time work is in scope** (explicit
decision). Turkish + Farsi support is a **hard constraint**. Whisper
large-v3-turbo stays the **default** ASR backend.

This is a planning document. It records the landscape, the constraints, and a
phased plan. It does not change any code.

---

## 1. Where millet is today

Current core pipeline (`millet/transcribe.py`):

1. Load dual-channel WAV → mono for transcription.
2. Transcribe with WhisperX / faster-whisper — default `whisper-large-v3-turbo`
   (CTranslate2). MLX path on Apple Silicon; opt-in Parakeet ONNX path.
3. Align with **wav2vec2** for word-level timestamps
   (`ALIGNMENT_MODELS`, `transcribe.py:212`).
4. Diarize with **pyannote `speaker-diarization-community-1`** via WhisperX's
   `DiarizationPipeline` (`transcribe.py:1876`, `2109`).
5. Merge diarization with transcription; voiceprint naming; outputs.

Relevant observations:

- millet is **already current** on two big pieces: it uses pyannote
  community-1 (the 4.0-era OSS model) and Whisper large-v3-turbo.
- The **Parakeet backend exists but is pinned to the English-only v2**
  (`nemo-parakeet-tdt-0.6b-v2`, `parakeet.py:42`).
- The **alignment stage is the oldest component**: 2020-era wav2vec2 models,
  with heavyweight per-language downloads (tr/fa ≈ 1.2 GB each,
  `transcribe.py:230`).

---

## 2. Constraints (decided)

- **Offline / local-first.** Every recommended model must run locally with no
  mandatory API dependency.
- **License-permissive.** Prefer MIT / Apache 2.0. Treat CC-BY-4.0 (attribution
  required) as second-class.
- **Keep Turkish + Farsi.** Rules out EN-only / EU-only models as *defaults*.
- **No streaming / real-time.** Batch pipeline only.
- **Whisper large-v3-turbo stays the default.** Newer models are opt-in
  backends, never a forced replacement.

---

## 3. The 2026 landscape

The "Whisper monoculture" ended in 2026. Whisper large-v3 has been overtaken on
the [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
by roughly ten open models — but leaderboard WER (English short-form) is a poor
proxy for millet's job (multilingual, long-form, noisy meeting audio).

### 3.1 ASR / transcription

| Model | License | Languages | Local | Notes |
|---|---|---|---|---|
| Whisper large-v3-turbo (**current default**) | **MIT** | 99+ | ✅ | 809M, ~6× faster than large-v3, mature runtime (whisper.cpp / faster-whisper / WhisperX). Still the best license-clean multilingual default. |
| **Qwen3-ASR 0.6B / 1.7B** | **Apache 2.0** | 52 (incl. **tr, fa**) | ✅ | Qwen3-Omni-based. Robust in low-SNR/noisy audio. Ships inference toolkit + separate forced-alignment model for timestamps (11 langs). Community reports it competitive with / beating Whisper on many inputs. Ecosystem younger than Whisper. |
| Parakeet-TDT-0.6b-v3 | CC-BY-4.0 | 25 EU (**no tr/fa**) | ✅ | 600M FastConformer-TDT, ~6.34% avg WER, ~49× Whisper throughput, native word/segment timestamps, ONNX/CoreML/ANE. Attribution burden; drops Turkish/Farsi. |
| Canary-Qwen-2.5B | CC-BY-4.0 | English | ✅ | SALM: transcribe **and** summarize in one model (~5.63% WER). EN-only, 2.5B, CC-BY. Niche. |
| IBM Granite Speech 4.1 / 3.3 | Apache 2.0 | EN-focused | ✅ | Top of leaderboard by WER, but English-focused. |
| Voxtral / Kyutai STT | Apache 2.0 / CC-BY | few | ✅ | Streaming-oriented — **out of scope**. |

**Takeaway:** with a "license-permissive + keep tr/fa" filter, the standout
*new* backend is **Qwen3-ASR (Apache 2.0, 52 langs)** — not Parakeet v3. Parakeet
v3 is a genuine speed win but is CC-BY and EU-only, so it belongs as a
speed-focused opt-in, not a default.

### 3.2 Diarization

| Model | License | Notes |
|---|---|---|
| pyannote `speaker-diarization-community-1` (**current**) | CC-BY-4.0 | pyannote.audio 4.0 OSS flagship. ~50% less speaker confusion vs 3.1; VBx clustering (not AHC); new `exclusive_speaker_diarization` output for cleaner STT reconciliation. ~11–13% DER range on academic benchmarks. |
| DiariZen | (OSS) | Competitive open alternative (~13.3% DER in the 2026 benchmark paper, arXiv:2509.26177), where pyannote precision-2 (commercial) leads at 11.2%. Candidate only if millet's audio exposes pyannote weaknesses. |
| NVIDIA Sortformer (offline/streaming) | CC-BY-4.0 (NeMo) | End-to-end; streaming variant is real-time. 4-speaker cap, English-optimized, heavy NeMo dep. Streaming → **out of scope**; offline variant not compelling enough to displace pyannote for millet. |

**Takeaway:** millet is already on the best OSS diarizer. The open question is
whether it is *fully exploiting* pyannote 4.0 (exclusive mode, VBx) through the
WhisperX wrapper, or leaving those gains on the table.

### 3.3 Alignment (word timestamps)

millet's wav2vec2 alignment (`ALIGNMENT_MODELS`) is the oldest component.
Newer ASR models (Parakeet v3, Qwen3-ASR) emit **native word-level timestamps**,
which can remove the separate wav2vec2 stage for those backends — fewer
downloads, less code, simpler pipeline.

---

## 4. Plan (phased, license-aware, no streaming)

### Phase 0 — Verification & benchmark harness (prerequisite, gates everything)

1. **Confirm pyannote 4.0 feature usage.** Determine whether WhisperX's
   `DiarizationPipeline` actually surfaces community-1's `exclusive_speaker_diarization`
   output and VBx clustering, or whether a direct `pyannote.audio` 4.0 call is
   needed to unlock the reduced-confusion benefits. (The dual-diarize
   consolidation heuristics — `_consolidate_dual_diarize_speakers` — and the
   backchannel guards from the cluster-bleed spike may be partly compensating
   for reconciliation weaknesses that 4.0 exclusive mode fixes upstream.)
2. **Build a small labeled benchmark set** with multilingual meeting clips
   (**including tr + fa**), reference transcripts, and RTTM. Compute **WER** and
   **DER**. No model swap ships without a measured win on this harness.

### Phase 1 — ASR upgrades

**1a. Bump the existing Parakeet backend v2 → v3** (low effort — `parakeet.py:42`)
- Set `DEFAULT_PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"`.
- Gains 25 EU languages + auto-language-detection for free, same ~600M size,
  native word timestamps.
- Keep opt-in (`--asr-backend parakeet`). **Document the CC-BY-4.0 attribution
  obligation.** Does not affect tr/fa (Parakeet has neither).

**1b. Add a Qwen3-ASR backend (Apache 2.0, multilingual) — strategic addition**
- New module `millet/qwen_asr.py` mirroring `parakeet.py`'s isolation pattern:
  same WhisperX-shaped result dict (`segments`/`language`/`text`), VAD chunking
  for long recordings, lazy `millet download qwen-asr` fetch, opt-in via
  `--asr-backend`.
- Covers **tr/fa permissively**, where Parakeet v3 cannot — a cleaner
  multilingual story than the current wav2vec2-per-language approach.
- Uses its native timestamps → feeds Phase 2.

**1c. Keep Whisper large-v3-turbo as the default** (MIT, 99 langs, mature).
Newer models remain opt-in backends.

### Phase 2 — Simplify the alignment stage

- For ASR backends that emit native word-level timestamps (Parakeet v3,
  Qwen3-ASR), **bypass the wav2vec2 alignment step**, gated on backend
  capability.
- Keep wav2vec2 alignment for the Whisper path where it is still required.
- Benefit: fewer heavyweight downloads (esp. tr/fa ≈ 1.2 GB each), simpler
  pipeline, less code in the alignment registry.
- Validate native-timestamp quality per-language against the Phase 0 harness
  before dropping alignment for that language.

### Phase 3 — Diarization enhancements

1. **Adopt pyannote 4.0 exclusive mode** if Phase 0 shows WhisperX isn't already
   using it — cleaner ASR↔diarization reconciliation and ~50% less speaker
   confusion vs 3.1. Some existing dual-diarize consolidation heuristics may then
   be simplifiable.
2. **Evaluate DiariZen** only if the benchmark exposes pyannote weaknesses on
   millet's real audio. Optional.

### Out of scope (this cycle)

- **Streaming / real-time transcription and diarization.** (Sortformer streaming,
  Voxtral/Kyutai realtime, `millet live`.) Explicitly deferred.
- **Summarization LLM refresh.** Minor/incremental; not part of this ASR-focused
  effort.

---

## 5. Sequencing & risk

| Phase | Effort | Value | Risk |
|---|---|---|---|
| 0 Benchmark harness | Low–Med | Critical (gates all) | Low |
| 1a Parakeet v2→v3 | **Low** | Med | Low (opt-in) |
| 3 pyannote 4.0 exclusive mode | Low–Med | Med–High | Low |
| 1b Qwen3-ASR backend | Med | **High** | Med |
| 2 Alignment simplification | Med | Med | Med (timestamp quality) |

**Recommended order:** 0 → 1a → 3 → 1b → 2.

---

## 6. Decisions locked

1. **No streaming.** Batch pipeline only.
2. **Keep Turkish + Farsi.** → Qwen3-ASR (Apache 2.0) is the multilingual bet;
   Parakeet v3 stays a speed-focused, EU-only, opt-in backend.
3. **Whisper large-v3-turbo stays the default.** No auto-selection heuristic;
   new models are opt-in.

## 7. Sources (retrieved 2026-07-30)

- Hugging Face Open ASR Leaderboard — https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
- Qwen3-ASR-0.6B — https://huggingface.co/Qwen (Apache 2.0, 52 languages)
- Parakeet-TDT-0.6b-v3 — https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Canary-1B-v2 & Parakeet-TDT-0.6B-v3 report — https://arxiv.org/abs/2509.14128
- pyannote community-1 — https://huggingface.co/pyannote/speaker-diarization-community-1
- pyannote 4.0 / community-1 announcement — https://www.pyannote.ai/blog/community-1
- Benchmarking Diarization Models (2026) — https://arxiv.org/html/2509.26177v1
- ASR license comparison (2026) — https://www.marktechpost.com/2026/07/23/best-open-speech-recognition-asr-models-in-2026-wer-languages-latency-and-license-compared/
