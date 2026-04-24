# meetscribe

Fully local meeting transcription with speaker diarization, AI-generated
summaries, and professional PDF output.

Records dual-channel audio (your mic + system audio) from **any** meeting
app and produces diarized transcripts using WhisperX + pyannote-audio.
Works fully offline with local models, or optionally use cloud APIs
(OpenRouter, Claude Max) for higher-quality summaries.

## Works with any meeting app

Because meetscribe captures system audio at the OS level, it works with
every voice/video call application:

- **Zoom**
- **Google Meet**
- **Microsoft Teams**
- **Slack** (huddles and calls)
- **Discord**
- **Signal** (voice and video calls)
- **Telegram** (voice and video calls)
- **WhatsApp** (desktop voice and video calls)
- **Keet** (P2P calls)
- **Jitsi Meet**
- **Webex**
- **Skype**
- **FaceTime** (via browser)
- **GoTo Meeting**
- **RingCentral**
- **Amazon Chime**
- **BlueJeans**

Any app that plays audio through your system speakers will work --
including browser-based meetings and standalone desktop clients.

## Features

- **Dual-channel audio capture** -- records your mic (left channel) and remote
  participants (right channel) simultaneously via PipeWire/PulseAudio + ffmpeg
- **WhisperX transcription** -- fast batched inference with
  `openai/whisper-large-v3-turbo`, word-level timestamps via wav2vec2 alignment
- **Multilingual** -- auto-detects language or manually set it; supports
  English, German, Turkish, French, Spanish, Farsi, and 90+ other languages
- **Speaker diarization** -- pyannote-audio identifies who said what, with
  automatic YOU/REMOTE labeling from the dual-channel signal
- **AI meeting summaries** -- local LLMs via Ollama, or cloud APIs via
  OpenRouter / Claude Max, with automatic fallback between backends
- **Voiceprint speaker recognition** -- automatically identifies speakers
  across meetings using voice embedding profiles
- **Meeting sync** -- push transcripts and summaries to any Git repository
  on a configurable schedule
- **Professional PDF output** -- summary + full transcript in a clean,
  page-numbered PDF with full Unicode support (DejaVu Sans) and RTL for Farsi
- **Multiple output formats** -- `.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`
- **GTK3 GUI widget** -- small always-on-top window with record/stop, timer,
  and one-click access to results
- **CLI** -- `meet record`, `meet transcribe`, `meet run`, `meet gui`,
  `meet label`, `meet enroll`, `meet sync`, `meet devices`, `meet check`
- **Per-session folders** -- each recording gets its own organized directory
- **Offline-first** -- after initial model download, core features work without
  internet; cloud backends are optional upgrades

## Quick start

```bash
# Install from PyPI
pip install meetscribe-offline

# Set your HuggingFace token (required for speaker diarization)
export HF_TOKEN=hf_your_token_here

# Record a meeting, then auto-transcribe + summarize when you stop
meet run
# Press Ctrl+C when the meeting ends
```

## Requirements

- **Linux** with PipeWire or PulseAudio
- **NVIDIA GPU** with CUDA (8GB+ VRAM recommended; CPU mode available but slower)
- **Python 3.10+**
- **ffmpeg**
- **HuggingFace token** (free) for the diarization model
- **Ollama** (optional) for AI meeting summaries

See [REQUIREMENTS.md](REQUIREMENTS.md) for full hardware/software details.

## Installation

### 1. System dependencies

```bash
# Ubuntu / Pop!_OS / Debian
sudo apt install ffmpeg pulseaudio-utils

# Fedora
sudo dnf install ffmpeg pulseaudio-utils
```

### 2. Install meetscribe

```bash
# From PyPI (recommended)
pip install meetscribe-offline

# From source
git clone https://github.com/pretyflaco/meetscribe
cd meetscribe
pip install -e .
```

This creates the `meet` command in your PATH.

### 3. HuggingFace token (for speaker diarization)

1. Create a free account at https://huggingface.co
2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1
3. Create a read token at https://huggingface.co/settings/tokens
4. Set it:

```bash
export HF_TOKEN=hf_your_token_here
# Add to ~/.bashrc for persistence:
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
```

### 4. Ollama (optional, for AI summaries)

Install from https://ollama.com, then pull the default summary model:

```bash
ollama pull qwen3.5:9b
```

### 5. Verify setup

```bash
meet check
```

## Usage

### Check audio devices

```bash
meet devices
```

### Record a meeting

Start recording before or during your meeting:

```bash
meet record
```

Press Ctrl+C when the meeting ends. A 10-second drain buffer ensures all audio
is captured. Recordings are saved to `~/meet-recordings/`.

Options:
- `-o /path` -- save recordings elsewhere
- `--virtual-sink` -- create isolated virtual sink (avoids capturing notification sounds)
- `--mic <source>` -- specify mic source (use `meet devices` to find names)
- `--monitor <source>` -- specify monitor source

### Transcribe a recording

```bash
meet transcribe ~/meet-recordings/meeting-20260312-140000/meeting-20260312-140000.wav
```

Options:
- `-m large-v3-turbo` -- Whisper model (default: `large-v3-turbo`; also: `base`, `medium`, `large-v2`)
- `-l auto` -- language code or `auto` to auto-detect (default: `auto`; e.g. `en`, `de`, `tr`, `fa`)
- `--device cuda` -- `cuda` or `cpu` (default: `cuda`)
- `--compute-type float16` -- `float16` or `int8` for lower VRAM (default: `float16`)
- `-b 16` -- batch size, reduce if running low on VRAM (default: `16`)
- `--min-speakers 2` / `--max-speakers 6` -- hint for number of speakers
- `--no-diarize` -- skip speaker diarization
- `--no-summarize` -- skip AI summary generation
- `--summary-backend openrouter` -- summary backend (`ollama`, `openrouter`, `claudemax`, `openai`)
- `--summary-model <model>` -- model for summary (default: per-backend)
- `--skip-alignment` -- skip word-level alignment (useful if alignment model is unavailable)
- `--mixdown mono|dual` -- stereo mixdown mode (default: `mono`). Use `dual` for
  headphone setups where mic and system audio don't bleed into each other (see below)

#### Dual-channel mode for headphone users

If you use headphones, your mic captures only your voice while the system
channel captures only the remote participants. In this setup the default mono
mixdown creates a ~20× energy imbalance that causes WhisperX to suppress the
quieter voice.

Use `--mixdown dual` to transcribe each channel independently:

```bash
meet transcribe --mixdown dual ~/meet-recordings/meeting-20260312-140000/
```

This skips diarization entirely (channel identity = speaker identity) and
labels segments as YOU (mic) or REMOTE (system). Default `--mixdown mono`
behavior is unchanged -- use it when your speakers play into the room and
both voices appear on both channels.

### Record + transcribe in one shot

```bash
meet run
```

Records until Ctrl+C, then automatically transcribes, generates a summary,
and produces a PDF. Takes all options from both `record` and `transcribe`
(including `--mixdown dual`).

### Launch the GUI widget

```bash
meet gui
```

A small always-on-top window with:
- Record / Stop button
- Live timer and file size
- Status indicator (Recording, Flushing, Transcribing, Summarizing, Done)
- "Open PDF" and "Open Folder" buttons after completion

When 2 or more speakers are detected, a **speaker labeling dialog** appears
before the results are saved. Each speaker is shown with their channel and a
sample line of text. If voice profiles exist, confident matches are shown
automatically. Enter a real name or leave blank to keep the auto-assigned
label (YOU, REMOTE_1, etc.).

If meeting sync is configured and the recording matches a scheduled meeting,
a **sync confirmation prompt** appears with Push / Skip buttons.

![meetscribe GUI](screenshot.png)

### Label speakers after the fact

```bash
meet label ~/meet-recordings/meeting-20260313-214133
```

For each speaker in the recording, `meet label`:
1. Shows a table of all speakers (label, channel, segment count, sample text)
2. Plays a short audio clip from that speaker's channel (requires `ffplay`)
3. Prompts you to enter a real name (press Enter to keep the existing label)
4. Regenerates all outputs (`.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`) with the new names

With `--auto`, voice profiles are used to automatically identify known speakers.
Confident matches are applied without prompting; only unrecognized speakers get
the interactive prompt:

```bash
meet label --auto ~/meet-recordings/meeting-20260313-214133
```

Options:
- `--auto` -- auto-label using voice profiles (see [Voiceprint speaker recognition](#voiceprint-speaker-recognition))
- `--no-audio` -- skip audio playback, just show text samples
- `--no-summary` -- use find-and-replace instead of re-running Ollama
- `--summary-backend` / `--summary-model` -- override summary backend and model for regeneration

## Output

Each recording gets its own session directory:

```
~/meet-recordings/meeting-20260312-140000/
    meeting-20260312-140000.wav            # Stereo audio (16kHz)
    meeting-20260312-140000.session.json   # Recording metadata
    meeting-20260312-140000.ffmpeg.log     # ffmpeg capture log
    meeting-20260312-140000.txt            # Plain text transcript
    meeting-20260312-140000.srt            # Subtitle format
    meeting-20260312-140000.json           # Full detail (word-level timestamps)
    meeting-20260312-140000.summary.md     # AI meeting summary (Markdown)
    meeting-20260312-140000.pdf            # Professional PDF (summary + transcript)
```

Example `.txt` output:

```
[00:00:12 --> 00:00:18] YOU: So the main issue we're seeing is with the API rate limiting.
[00:00:19 --> 00:00:25] REMOTE_1: Right, I think we should implement exponential backoff.
[00:00:26 --> 00:00:31] YOU: Agreed. Can you also look at caching the responses?
```

## AI summary

meetscribe generates a structured meeting summary with:
- Overview
- Key topics discussed
- Action items (with owners when mentioned)
- Decisions made
- Open questions / follow-ups

### Supported models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `qwen3.5:9b` | 6.6 GB | ~18-35s | **Default** -- best balance of quality and speed |
| `gemma3:12b` | 8.1 GB | ~15s | Fastest |
| `qwen3:14b` | 9.3 GB | ~39s | Good quality |
| `glm-4.7-flash` | 19 GB | ~37s | Must use thinking-off mode (handled automatically) |

Change the model:

```bash
meet run --summary-model gemma3:12b
```

Disable summaries:

```bash
meet run --no-summarize
```

### Summary backends

meetscribe supports three backends with automatic fallback:

| Backend | Setup | Cost | Quality |
|---------|-------|------|---------|
| `ollama` (default) | `ollama serve` + `ollama pull qwen3.5:9b` | Free | Good |
| `openrouter` | Set `OPENROUTER_API_KEY` | Pay-per-use | Excellent |
| `claudemax` | Run claude-max-api-proxy on localhost:3457 | Claude Max subscription | Excellent |
| `openai` | Set `MEETSCRIBE_OPENAI_BASE_URL` | Varies | Varies |

The `openai` backend works with any OpenAI-compatible API — Lemonade, LiteLLM,
vLLM, text-generation-webui, LocalAI, or any self-hosted endpoint.

```bash
# Use OpenRouter
meet run --summary-backend openrouter --summary-model anthropic/claude-sonnet-4.6

# Use any OpenAI-compatible endpoint
export MEETSCRIBE_SUMMARY_BACKEND=openai
export MEETSCRIBE_OPENAI_BASE_URL=http://localhost:8000/v1
export MEETSCRIBE_SUMMARY_MODEL=your-model-name
# Optional: export MEETSCRIBE_OPENAI_API_KEY=your-key

# Or set via environment variables
export MEETSCRIBE_SUMMARY_BACKEND=openrouter
export MEETSCRIBE_SUMMARY_MODEL=anthropic/claude-sonnet-4.6
```

If the configured backend is unavailable, meetscribe automatically tries the
next one in the fallback chain: claudemax → openrouter → ollama.

### Two-pass local summarization

When the **ollama** backend is selected (the default), meetscribe runs two
LLM calls instead of one:

1. **Pass 1 (extraction)** — pulls topics, actions, decisions, and open
   questions out of the transcript as plain numbered lists, using a
   context window sized to the full transcript.
2. **Pass 2 (formatting)** — takes the much smaller extracted data and
   organizes it into the canonical Markdown structure with a fixed 8K
   context window.

This dramatically improves format compliance and reduces hallucinations
on 20B-class local models (`gpt-oss:20b`, `qwen3.6:27b`) compared to a
single-pass call, at the cost of one additional LLM call (~30–90s extra).
Cloud backends (claudemax, openrouter, openai) remain single-pass — they
already produce well-structured output in one shot.

To opt out and use the previous single-pass behavior:

```bash
meet run --ollama-singlepass
# Or via environment:
export MEETSCRIBE_OLLAMA_SINGLEPASS=1
```

The `.summary.meta.json` sidecar records per-pass timings
(`pass1_seconds`, `pass2_seconds`, `pass1_chars`) when two-pass was used.

See [docs/local-model-evaluation.md](docs/local-model-evaluation.md) for
the full evaluation that motivated this design, including known failure
modes of local 20B-class models.

### Customizing the prompt

The summarization prompt lives in `meet/prompts/summarize_system.md`. Edit it
to change the summary format, add domain-specific instructions, or tune for
your preferred model. No Python changes needed.

## Voiceprint speaker recognition

meetscribe can automatically identify speakers across meetings using voice
embeddings. After you label speakers in one meeting, their voice profiles are
stored and matched against future recordings.

```bash
# Build profiles from already-labeled sessions
meet enroll ~/meet-recordings/meeting-20260330-*

# Auto-label speakers in future meetings using voice profiles
meet label --auto ~/meet-recordings/meeting-20260401-093000
```

Profiles are stored in `~/.config/meet/speaker_profiles.json` and improve
with each labeled session (running average of embeddings).

## Meeting sync

Push meeting artifacts to a Git repository on a configurable schedule.

```bash
# Create an example config
meet sync --init-config
# Edit ~/.config/meet/sync_config.json with your repo URL and schedule

# Push a session manually
meet sync ~/meet-recordings/meeting-20260331-110038_STANDUP

# View configured schedule
meet sync --list-schedule
```

When the GUI detects a matching scheduled meeting, it prompts for confirmation
before syncing. Sessions that don't match the schedule are skipped. The CLI
uses `--force` to sync unmatched sessions.

You can also configure a `team_members` list and `min_team_members` threshold
in `sync_config.json` to require that a minimum number of recognized speakers
are present before offering to sync.

## Multilingual support

meetscribe auto-detects the spoken language by default (Whisper large-v3-turbo
supports 99 languages). You can also set it explicitly:

```bash
meet run --language de       # German
meet run --language tr       # Turkish
meet run --language fr       # French
meet run --language es       # Spanish
meet run --language fa       # Farsi (Persian)
meet run --language auto     # Auto-detect (default)
```

### How it works

- **Transcription**: The same Whisper model handles all languages -- no extra
  download or VRAM cost. When set to `auto`, the detected language is used for
  alignment and all downstream steps.
- **Speaker diarization**: Completely language-agnostic (based on voice
  characteristics, not speech content).
- **AI summary**: When a non-English language is detected, the summary prompt
  instructs the LLM to write the summary in the same language as the transcript.
- **PDF output**: Uses DejaVu Sans for full Unicode coverage (Latin, Cyrillic,
  Greek, Turkish special characters, etc.). Farsi uses Noto Naskh Arabic with
  RTL text reshaping.

### Tested languages

| Language | Code | Alignment model | PDF font | Notes |
|----------|------|----------------|----------|-------|
| English  | `en` | wav2vec2 (torchaudio) | DejaVu Sans | |
| German   | `de` | VoxPopuli (torchaudio) | DejaVu Sans | |
| French   | `fr` | VoxPopuli (torchaudio) | DejaVu Sans | |
| Spanish  | `es` | VoxPopuli (torchaudio) | DejaVu Sans | |
| Turkish  | `tr` | wav2vec2 (HuggingFace) | DejaVu Sans | ~1.2 GB alignment model download |
| Farsi    | `fa` | wav2vec2 (HuggingFace) | Noto Naskh Arabic | ~1.2 GB alignment model download, RTL |

### Farsi RTL requirements

Farsi uses right-to-left text. For proper PDF rendering, install the optional
RTL dependencies:

```bash
pip install arabic-reshaper python-bidi
# Or with the optional extra:
pip install "meetscribe-offline[rtl]"
```

Without these libraries, Farsi text will appear in the PDF but glyphs may not
be joined correctly and reading order may be wrong.

## Virtual sink mode

By default, `meet record` captures all system audio (including notification
sounds, music, etc.). For cleaner recordings, use `--virtual-sink`:

```bash
meet record --virtual-sink
```

This creates an isolated audio sink. Route your meeting app's audio to it:

1. Open `pavucontrol` (PulseAudio Volume Control)
2. Go to the "Playback" tab
3. Find your browser or meeting app
4. Change its output to "Meet-Capture"

You'll still hear the meeting through your normal speakers via automatic loopback.

## VRAM usage

With an NVIDIA GPU (12 GB VRAM):

| Model | Transcription | + Diarization | Recommended batch_size |
|-------|--------------|---------------|----------------------|
| large-v3-turbo | ~4 GB | ~7 GB total | 16 |
| medium | ~3 GB | ~6 GB total | 16 |
| base | ~1 GB | ~4 GB total | 16 |

If you hit OOM errors:
1. Reduce `--batch-size` to 4 or 8
2. Use `--compute-type int8`
3. Use a smaller model (`--model medium` or `--model base`)
4. Use `--device cpu` as a last resort

## How it works

```
[Meeting App] --> [PipeWire/PulseAudio] --> [ffmpeg dual-channel capture] --> meeting.wav
                                                                                  |
                  [WhisperX: faster-whisper + wav2vec2 alignment + pyannote diarization]
                                                                                  |
                                      [Ollama LLM summary]     [Diarized transcript]
                                              |                         |
                                        .summary.md          .txt / .srt / .json
                                              |                         |
                                              +--------> .pdf <---------+
```

**Capture**: Records your mic (left channel) and system audio (right channel)
simultaneously into a single stereo WAV file at 16 kHz.

**Transcribe**: Runs the WhisperX pipeline -- batched Whisper transcription,
wav2vec2 forced alignment for word-level timestamps, and pyannote speaker
diarization. Dual-channel energy analysis maps speakers to YOU or REMOTE.

**Summarize**: Sends the transcript to a local Ollama model that extracts
a structured summary.

**PDF**: Combines the summary and full transcript into a professional
page-numbered PDF document.

## CUDA NVRTC note

The pyannote diarization model requires CUDA NVRTC for JIT compilation. If your
CUDA driver version doesn't match the installed libnvrtc-builtins version,
meetscribe automatically creates a compatibility symlink. This happens
transparently on first use.

If you still see NVRTC errors:

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/cuda:$LD_LIBRARY_PATH
```

## Limitations

- Overlapping speech is not handled well (Whisper limitation)
- Speaker labels default to role-based (YOU, REMOTE_1, REMOTE_2) — use `meet label` or the GUI dialog to assign real names
- Diarization accuracy varies with audio quality and number of speakers
- Linux only (PulseAudio/PipeWire dependency)
- Local 20B-class summary models (e.g. `gpt-oss:20b`) can hallucinate on
  transcripts dominated by very short low-information utterances ("yes",
  "okay") and may exceed the default 600s timeout on very large
  (>100 KB) non-English transcripts. For these cases configure a cloud
  backend (claudemax / openrouter) — the fallback chain takes over
  automatically. See [docs/local-model-evaluation.md](docs/local-model-evaluation.md).

## FAQ

**Is there a GUI?** Yes — run `meet gui` for a small always-on-top GTK3
widget with Record/Stop, live timer, status indicator, and one-click
access to the resulting PDF and session folder. See
[Launch the GUI widget](#launch-the-gui-widget) for details.

**Does it work on Windows / macOS?** No. meetscribe currently requires
Linux with PipeWire or PulseAudio for system-audio capture. See
[Limitations](#limitations).

**Can I use it without a GPU?** Yes, with `--device cpu`, but
transcription will be 5–20× slower depending on the Whisper model.
See [VRAM usage](#vram-usage).

## Contributing

```bash
git clone https://github.com/pretyflaco/meetscribe
cd meetscribe
pip install -e .[dev]
/usr/bin/python3 -m pytest tests/
```

Pull requests welcome. Please run the test suite before submitting.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[GPL-3.0](LICENSE)
