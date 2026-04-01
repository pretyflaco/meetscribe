"""Speaker labeling module for meetscribe.

Provides functions to:
  - List speakers in a transcribed session
  - Extract short audio clips per speaker for voice identification
  - Play audio clips via ffplay
  - Apply user-assigned speaker names and regenerate all outputs
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from meet.audio import read_stereo_channels, compute_speaker_channel_energy
from meet.transcribe import Segment, Speaker, Transcript


# ─── Data ───────────────────────────────────────────────────────────────────

@dataclass
class SpeakerInfo:
    """Information about a speaker for the labeling UI."""

    id: str
    """Current speaker label (e.g. 'YOU', 'REMOTE', 'REMOTE_1')."""

    channel: str
    """Which audio channel this speaker is dominant on: 'mic' or 'system'."""

    sample_text: str
    """A short text excerpt from this speaker's longest segment."""

    sample_start: float
    """Start time (seconds) of the representative segment."""

    sample_end: float
    """End time (seconds) of the representative segment."""

    segment_count: int
    """Number of segments attributed to this speaker."""


# ─── Load transcript from session directory ─────────────────────────────────

def _find_session_files(session_dir: Path) -> dict[str, Path]:
    """Locate key files in a session directory. Returns dict of type -> path."""
    files: dict[str, Path] = {}

    # Find JSON transcript (exclude translation, session, and summary meta files)
    for p in sorted(session_dir.glob("*.json")):
        if ".session." in p.name:
            files["session"] = p
        elif ".translation." not in p.name and ".summary." not in p.name:
            files["json"] = p

    # Find audio file (prefer WAV if still present, fall back to OGG)
    wavs = sorted(session_dir.glob("*.wav"))
    oggs = sorted(session_dir.glob("*.ogg"))
    if wavs:
        files["wav"] = wavs[0]
    elif oggs:
        files["wav"] = oggs[0]  # key stays "wav" for backward compat

    # Find summary
    summaries = sorted(session_dir.glob("*.summary.md"))
    if summaries:
        files["summary"] = summaries[0]

    # Find PDF
    pdfs = sorted(session_dir.glob("*.pdf"))
    if pdfs:
        files["pdf"] = pdfs[0]

    return files



find_session_files = _find_session_files  # public accessor

def _load_transcript(json_path: Path) -> Transcript:
    """Reconstruct a Transcript object from a saved JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    segments = []
    for s in data["segments"]:
        segments.append(Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
            words=s.get("words"),
        ))

    speakers = []
    for sp in data.get("speakers", []):
        speakers.append(Speaker(id=sp["id"], label=sp.get("label")))

    return Transcript(
        segments=segments,
        speakers=speakers,
        language=data.get("language", "en"),
        audio_file=data.get("audio_file", ""),
        duration=data.get("duration"),
    )


# ─── Speaker analysis ──────────────────────────────────────────────────────

def _detect_speaker_channels(
    wav_path: Path,
    segments: list[Segment],
    speakers: list[Speaker],
) -> dict[str, str]:
    """Determine which audio channel each speaker is dominant on.

    Returns a dict mapping speaker ID -> 'mic' or 'system'.
    Uses the same RMS energy analysis as the channel labeling logic.
    """
    stereo = read_stereo_channels(wav_path)
    if stereo is None:
        # Mono, unsupported, or unreadable — default all to 'mic'
        return {sp.id: "mic" for sp in speakers}

    mic_ratio = compute_speaker_channel_energy(
        stereo.mic, stereo.system, segments, stereo.sample_rate
    )

    channel_map: dict[str, str] = {}
    for sp in speakers:
        ratio = mic_ratio.get(sp.id, 0.5)
        channel_map[sp.id] = "mic" if ratio > 0.5 else "system"
    return channel_map


def get_speakers(session_dir: str | Path) -> list[SpeakerInfo]:
    """Get information about all speakers in a session.

    Args:
        session_dir: Path to the session directory.

    Returns:
        List of SpeakerInfo objects, one per speaker.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    session_dir = Path(session_dir)
    files = _find_session_files(session_dir)

    if "json" not in files:
        raise FileNotFoundError(f"No transcript JSON found in {session_dir}")

    transcript = _load_transcript(files["json"])

    if not transcript.speakers:
        return []

    # Detect which channel each speaker is on
    wav_path = files.get("wav")
    if wav_path and wav_path.exists():
        channel_map = _detect_speaker_channels(
            wav_path, transcript.segments, transcript.speakers,
        )
    else:
        channel_map = {sp.id: "mic" for sp in transcript.speakers}

    # Find best sample segment per speaker (longest, capped at ~15s for display)
    speaker_segments: dict[str, list[Segment]] = {}
    for seg in transcript.segments:
        if seg.speaker:
            speaker_segments.setdefault(seg.speaker, []).append(seg)

    result: list[SpeakerInfo] = []
    for sp in transcript.speakers:
        segs = speaker_segments.get(sp.id, [])
        if not segs:
            continue

        # Pick the longest segment for the sample
        best = max(segs, key=lambda s: s.end - s.start)
        sample_text = best.text.strip()[:120]
        if len(best.text.strip()) > 120:
            sample_text += "..."

        result.append(SpeakerInfo(
            id=sp.id,
            channel=channel_map.get(sp.id, "mic"),
            sample_text=sample_text,
            sample_start=best.start,
            sample_end=best.end,
            segment_count=len(segs),
        ))

    return result


# ─── Audio clip extraction ──────────────────────────────────────────────────

def extract_speaker_clip(
    wav_path: str | Path,
    speaker_info: SpeakerInfo,
    max_duration: float = 8.0,
) -> Path:
    """Extract a short audio clip for a speaker from the session audio.

    Reads the appropriate channel (mic for YOU-like speakers, system for
    REMOTE-like speakers) and writes a temporary mono WAV file.

    Args:
        wav_path: Path to the stereo session audio file (WAV or OGG).
        speaker_info: SpeakerInfo with channel hint and sample timestamps.
        max_duration: Maximum clip duration in seconds.

    Returns:
        Path to a temporary mono WAV file containing the clip.
        Caller is responsible for cleanup.
    """
    wav_path = Path(wav_path)

    stereo = read_stereo_channels(wav_path)
    if stereo is not None:
        # Stereo: pick the speaker's dominant channel
        ch_data = stereo.mic if speaker_info.channel == "mic" else stereo.system
        file_sr = stereo.sample_rate
        sampwidth = stereo.sampwidth
        # Convert back to integer dtype for writing
        if sampwidth == 2:
            channel_data = ch_data.astype(np.int16)
        else:
            channel_data = ch_data.astype(np.int32)
    else:
        # Mono fallback: decode via ffmpeg (works for WAV, OGG, etc.)
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-i", str(wav_path),
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError(f"Cannot decode audio: {wav_path}")
        sampwidth = 2
        # Get sample rate via ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=sample_rate",
             "-of", "csv=p=0", str(wav_path)],
            capture_output=True, text=True,
        )
        file_sr = int(probe.stdout.strip()) if probe.returncode == 0 else 16000
        channel_data = np.frombuffer(result.stdout, dtype=np.int16)

    # Extract the time range
    start_frame = max(0, int(speaker_info.sample_start * file_sr))
    duration = min(speaker_info.sample_end - speaker_info.sample_start, max_duration)
    end_frame = min(int((speaker_info.sample_start + duration) * file_sr), len(channel_data))

    clip = channel_data[start_frame:end_frame]

    if len(clip) == 0:
        raise RuntimeError(f"Empty audio clip for speaker {speaker_info.id}")

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(sampwidth)
        out.setframerate(file_sr)
        out.writeframes(clip.tobytes())

    return Path(tmp.name)


def play_clip(clip_path: str | Path) -> subprocess.Popen:
    """Play an audio clip using ffplay.

    Args:
        clip_path: Path to a WAV file to play.

    Returns:
        The subprocess.Popen object. Call .wait() to block until playback
        finishes, or .kill() to stop early.
    """
    return subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(clip_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ─── Apply labels ──────────────────────────────────────────────────────────

def apply_labels(
    session_dir: str | Path,
    label_map: dict[str, str],
    regenerate_summary: bool = True,
    summary_backend: str | None = None,
    summary_model: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Path]:
    """Apply user-assigned speaker names to a session's outputs.

    Relabels the transcript and regenerates all output files (txt, srt, json,
    summary, pdf).

    Args:
        session_dir: Path to the session directory.
        label_map: Mapping of current speaker ID to new name.
            E.g. {"YOU": "Kasita", "REMOTE_1": "Ahmad"}.
            Only speakers present in the map are relabeled.
        regenerate_summary: If True, re-run the LLM to generate a new summary
            with updated speaker names. If False, do a best-effort
            find-and-replace on the existing summary.
        summary_backend: Backend override ("ollama" or "openrouter"); None uses default.
        summary_model: Model name override; None uses the per-backend default.
        progress_callback: Optional callable(str) for status messages.

    Returns:
        Dict of format name -> file path for all updated files.

    Raises:
        FileNotFoundError: If required session files are missing.
    """
    session_dir = Path(session_dir)
    files = _find_session_files(session_dir)

    if "json" not in files:
        raise FileNotFoundError(f"No transcript JSON found in {session_dir}")

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    transcript = _load_transcript(files["json"])
    basename = files["json"].stem  # e.g. meeting-20260313-231509

    # ── Relabel transcript in-memory ──
    _log("Relabeling transcript...")
    transcript = relabel_transcript_in_memory(transcript, label_map)

    # ── Save updated transcript files ──
    _log("Saving updated transcript files...")
    result_files = transcript.save(session_dir, basename=basename)

    # ── Summary ──
    summary_result = None

    if regenerate_summary:
        _log("Regenerating summary with updated speaker names...")
        try:
            from meet.summarize import (
                summarize as do_summarize, SummaryConfig,
                is_backend_available, _backend_not_available_message,
            )
            from meet.transcribe import ensure_gpu_available

            cfg_kwargs: dict = {}
            if summary_backend:
                cfg_kwargs["backend"] = summary_backend
            if summary_model:
                cfg_kwargs["model"] = summary_model
            summary_config = SummaryConfig(**cfg_kwargs)

            if is_backend_available(summary_config):
                # Only free GPU for Ollama backend
                if summary_config.backend == "ollama":
                    ensure_gpu_available()
                summary_result = do_summarize(
                    transcript.to_text(), summary_config,
                    language=transcript.language,
                )
                path = summary_result.save(session_dir, basename)
                result_files["summary"] = path
                _log(f"Summary regenerated in {summary_result.elapsed_seconds:.1f}s")
            else:
                _log(_backend_not_available_message(summary_config))
                # Fall back to find-and-replace
                regenerate_summary = False
        except Exception as exc:
            _log(f"Summary regeneration failed: {exc}")
            regenerate_summary = False

    if not regenerate_summary and files.get("summary") and files["summary"].exists():
        # Best-effort find-and-replace on existing summary
        _log("Updating speaker names in existing summary...")
        summary_text = files["summary"].read_text(encoding="utf-8")
        for old_label, new_label in label_map.items():
            summary_text = summary_text.replace(old_label, new_label)
        files["summary"].write_text(summary_text, encoding="utf-8")
        result_files["summary"] = files["summary"]

        # Load summary for PDF embedding
        from meet.summarize import MeetingSummary
        summary_result = MeetingSummary(
            markdown=summary_text, model="(relabeled)", elapsed_seconds=0,
        )

    # ── PDF ──
    _log("Regenerating PDF...")
    try:
        from meet.pdf import generate_pdf

        pdf_path = session_dir / f"{basename}.pdf"
        generate_pdf(
            transcript, pdf_path,
            summary=summary_result,
            language=transcript.language,
        )
        result_files["pdf"] = pdf_path
    except Exception as exc:
        _log(f"PDF regeneration failed: {exc}")

    # ── Update session.json with label mapping ──
    if files.get("session") and files["session"].exists():
        _log("Updating session metadata...")
        try:
            meta = json.loads(files["session"].read_text(encoding="utf-8"))
            meta["speaker_labels"] = label_map
            files["session"].write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        except Exception:
            pass

    _log("Done!")
    return result_files


def relabel_transcript_in_memory(
    transcript: Transcript,
    label_map: dict[str, str],
) -> Transcript:
    """Apply label_map to a Transcript object and return a new one.

    This is used by the GUI to relabel in-memory before the first save,
    avoiding the need to regenerate outputs.
    """
    if not label_map:
        return transcript

    new_segments = []
    for seg in transcript.segments:
        new_speaker = label_map.get(seg.speaker, seg.speaker) if seg.speaker else seg.speaker
        new_segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=new_speaker,
            words=seg.words,
        ))

    new_speakers = []
    for sp in transcript.speakers:
        new_id = label_map.get(sp.id, sp.id)
        new_speakers.append(Speaker(id=new_id, label=new_id))

    return Transcript(
        segments=new_segments,
        speakers=new_speakers,
        language=transcript.language,
        audio_file=transcript.audio_file,
        duration=transcript.duration,
    )
