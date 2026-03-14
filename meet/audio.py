"""Audio utilities for meetscribe.

Low-level helpers for reading stereo WAV files and computing per-speaker
channel energy.  Extracted from label.py and transcribe.py to eliminate
~150 lines of duplication.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import NamedTuple

import numpy as np


class StereoChannels(NamedTuple):
    """Parsed stereo WAV data returned by :func:`read_stereo_channels`."""

    mic: np.ndarray    # Left channel (your microphone), float32
    system: np.ndarray # Right channel (system/remote audio), float32
    sample_rate: int   # Frames per second
    sampwidth: int     # Bytes per sample (2 = int16, 4 = int32)


def read_stereo_channels(wav_path: Path) -> StereoChannels | None:
    """Read a stereo WAV file and return separate mic and system channels.

    Returns None (instead of raising) if the file is mono, has an
    unsupported sample width, or cannot be opened.  Callers should
    fall back to a safe default in that case.

    The returned arrays are float32 copies — safe to modify.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            if n_channels != 2:
                return None
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return None

    if sampwidth == 2:
        dtype: type = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        return None

    samples = np.frombuffer(raw, dtype=dtype)
    if len(samples) % 2 != 0:
        samples = samples[:-1]
    samples = samples.reshape(-1, 2).astype(np.float32)

    return StereoChannels(
        mic=samples[:, 0],
        system=samples[:, 1],
        sample_rate=sample_rate,
        sampwidth=sampwidth,
    )


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
