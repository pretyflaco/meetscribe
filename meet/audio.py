"""Audio utilities for meetscribe.

Low-level helpers for reading stereo audio files, computing per-speaker
channel energy, and compressing recordings.

Extracted from label.py and transcribe.py to eliminate duplication.
All I/O uses ffmpeg/ffprobe (via subprocess) so that any audio format
supported by ffmpeg (WAV, OGG/Opus, FLAC, …) can be read transparently.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

log = logging.getLogger(__name__)


class StereoChannels(NamedTuple):
    """Parsed stereo audio data returned by :func:`read_stereo_channels`."""

    mic: np.ndarray    # Left channel (your microphone), float32
    system: np.ndarray # Right channel (system/remote audio), float32
    sample_rate: int   # Frames per second
    sampwidth: int     # Bytes per sample (always 2 — decoded to int16)


def read_stereo_channels(audio_path: Path) -> StereoChannels | None:
    """Read a stereo audio file and return separate mic and system channels.

    Uses ffmpeg to decode to raw PCM, so any format ffmpeg supports
    (WAV, OGG/Opus, FLAC, …) works transparently.

    Returns None (instead of raising) if the file is mono, cannot be
    opened, or decoding fails.  Callers should fall back to a safe
    default in that case.

    The returned arrays are float32 copies — safe to modify.
    """
    # Probe channel count first.
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "stream=channels,sample_rate",
        "-of", "json",
        str(audio_path),
    ]
    try:
        probe = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe.returncode != 0:
            return None
        info = json.loads(probe.stdout)
        stream = info.get("streams", [{}])[0]
        n_channels = int(stream.get("channels", 0))
        sample_rate = int(stream.get("sample_rate", 0))
    except Exception:
        return None

    if n_channels != 2 or sample_rate == 0:
        return None

    # Decode full file to raw s16le PCM via ffmpeg.
    decode_cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", str(audio_path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "2",
        "-",   # write to stdout
    ]
    try:
        result = subprocess.run(decode_cmd, capture_output=True)
        if result.returncode != 0:
            return None
        raw = result.stdout
    except Exception:
        return None

    if len(raw) == 0:
        return None

    samples = np.frombuffer(raw, dtype=np.int16)
    if len(samples) % 2 != 0:
        samples = samples[:-1]
    samples = samples.reshape(-1, 2).astype(np.float32)

    return StereoChannels(
        mic=samples[:, 0],
        system=samples[:, 1],
        sample_rate=sample_rate,
        sampwidth=2,
    )


# ─── Audio compression ─────────────────────────────────────────────────────

def _get_audio_duration(path: Path) -> float | None:
    """Return duration in seconds via ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def compress_audio(
    wav_path: Path,
    *,
    keep_wav: bool = False,
    bitrate: str = "48k",
) -> Path:
    """Compress a WAV file to OGG/Opus and optionally delete the original.

    Args:
        wav_path: Path to the stereo WAV recording.
        keep_wav: If True, keep the WAV file after compression.
        bitrate: Opus bitrate (default 48k — transparent for speech).

    Returns:
        Path to the compressed .ogg file.

    Raises:
        RuntimeError: If ffmpeg fails or duration validation fails.
        FileNotFoundError: If the WAV file does not exist.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    ogg_path = wav_path.with_suffix(".ogg")

    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", str(wav_path),
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-vn",
        str(ogg_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up partial output.
        ogg_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Audio compression failed (ffmpeg exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )

    # Validate: durations must match within 1 second.
    wav_dur = _get_audio_duration(wav_path)
    ogg_dur = _get_audio_duration(ogg_path)
    if wav_dur is not None and ogg_dur is not None:
        if abs(wav_dur - ogg_dur) > 1.0:
            ogg_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Duration mismatch after compression: WAV={wav_dur:.1f}s "
                f"vs OGG={ogg_dur:.1f}s (diff > 1s)"
            )

    # Gather sizes for logging before potentially deleting the WAV.
    wav_size = wav_path.stat().st_size
    ogg_size = ogg_path.stat().st_size
    ratio = wav_size / ogg_size if ogg_size > 0 else 0

    if not keep_wav:
        wav_path.unlink()
        log.info("Deleted %s after compression", wav_path.name)

    log.info(
        "Compressed %s -> %s (%.1f MB -> %.1f MB, %.0fx)",
        wav_path.name, ogg_path.name,
        wav_size / 1_048_576, ogg_size / 1_048_576, ratio,
    )

    return ogg_path


def compute_speaker_channel_energy(
    mic_ch: np.ndarray,
    sys_ch: np.ndarray,
    segments: list,          # list[Segment] — avoid circular import
    sample_rate: int,
) -> dict[str, float]:
    """Compute the mic-channel energy ratio for each speaker.

    For each speaker, accumulates RMS energy on the mic channel and on
    the system channel across all their segments, then returns a dict
    mapping ``speaker_id -> mic_ratio`` where::

        mic_ratio = avg_mic_rms / (avg_mic_rms + avg_sys_rms)

    A ratio > 0.5 means the speaker is dominant on the mic (i.e. YOU).
    Speakers with no audio frames get a ratio of 0.5 (unknown).

    Args:
        mic_ch:      Float32 array of left-channel (mic) samples.
        sys_ch:      Float32 array of right-channel (system) samples.
        segments:    List of Segment objects with .start, .end, .speaker.
        sample_rate: Frames per second (used to convert timestamps to indices).

    Returns:
        Dict mapping speaker ID to mic-ratio float in [0.0, 1.0].
    """
    n = len(mic_ch)
    mic_energy: dict[str, float] = {}
    sys_energy: dict[str, float] = {}
    total_frames: dict[str, int] = {}

    for seg in segments:
        if not seg.speaker:
            continue
        start = max(0, min(int(seg.start * sample_rate), n))
        end = max(0, min(int(seg.end * sample_rate), n))
        if end <= start:
            continue

        mic_slice = mic_ch[start:end]
        sys_slice = sys_ch[start:end]
        count = end - start

        mic_rms = float(np.sqrt(np.mean(mic_slice ** 2)))
        sys_rms = float(np.sqrt(np.mean(sys_slice ** 2)))

        spk = seg.speaker
        mic_energy[spk] = mic_energy.get(spk, 0.0) + mic_rms * count
        sys_energy[spk] = sys_energy.get(spk, 0.0) + sys_rms * count
        total_frames[spk] = total_frames.get(spk, 0) + count

    mic_ratio: dict[str, float] = {}
    for spk, frames in total_frames.items():
        if frames == 0:
            mic_ratio[spk] = 0.5
            continue
        avg_mic = mic_energy.get(spk, 0.0) / frames
        avg_sys = sys_energy.get(spk, 0.0) / frames
        denom = avg_mic + avg_sys
        mic_ratio[spk] = avg_mic / denom if denom > 0 else 0.5

    return mic_ratio
