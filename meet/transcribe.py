"""Transcription module using WhisperX with speaker diarization.

Pipeline:
1. Load audio (dual-channel WAV -> mono for transcription)
2. Transcribe with faster-whisper (batched, GPU-accelerated)
3. Align with wav2vec2 for word-level timestamps
4. Diarize with pyannote-audio for speaker labels
5. Merge diarization with transcription
6. Output formatted transcript
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Fix for CUDA NVRTC JIT compilation: pyannote's wespeaker embedding model
# triggers torch.fft.rfft -> CUDA JIT -> NVRTC, which needs libnvrtc-builtins.so
# matching the CUDA version.  Setting LD_LIBRARY_PATH at runtime is too late
# (libnvrtc.so is already loaded), so we preload the library with ctypes.CDLL
# to make it available in the process address space before NVRTC needs it.


def _preload_nvrtc_builtins():
    """Preload libnvrtc-builtins.so so NVRTC JIT compilation can find it.

    When NVRTC's libnvrtc.so calls dlopen("libnvrtc-builtins.so.X.Y"), the
    dynamic linker checks already-loaded libraries first.  By loading the
    correct version early via ctypes.CDLL, we ensure it's found regardless
    of LD_LIBRARY_PATH at process startup time.
    """
    import ctypes
    import importlib.util
    import sys

    # Determine the CUDA major.minor that PyTorch / the driver expects.
    # We look for libnvrtc-builtins.so.<major>.<minor> in nvidia pip packages.
    search_dirs: list[Path] = []

    # 1. nvidia.cu<major> packages (e.g. nvidia-cuda-nvrtc 13.x installs here)
    site_dirs = [Path(p) for p in sys.path if "site-packages" in p]
    for sp in site_dirs:
        nvidia_dir = sp / "nvidia"
        if nvidia_dir.is_dir():
            for child in sorted(nvidia_dir.iterdir(), reverse=True):
                if child.name.startswith("cu") and child.name[2:].isdigit():
                    lib_dir = child / "lib"
                    if lib_dir.is_dir():
                        search_dirs.append(lib_dir)

    # 2. nvidia.cuda_nvrtc package (older layout)
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent / "lib"
            if pkg_dir.is_dir():
                search_dirs.append(pkg_dir)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    # 3. Common system paths
    search_dirs.extend(
        [
            Path("/usr/local/cuda/lib64"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]
    )

    # Find the highest-version libnvrtc-builtins.so.*
    candidates: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            found = sorted(d.glob("libnvrtc-builtins.so.*"))
            candidates.extend(c for c in found if "alt" not in c.name)

    if not candidates:
        return

    def _version_key(p: Path) -> tuple[int, ...]:
        """Extract numeric version tuple from libnvrtc-builtins.so.X.Y"""
        suffix = p.name.removeprefix("libnvrtc-builtins.so.")
        try:
            return tuple(int(x) for x in suffix.split("."))
        except ValueError:
            return (0,)

    # Pick the highest version by numeric comparison (13.0 > 12.8 > 12.4)
    best = max(candidates, key=_version_key)
    try:
        ctypes.CDLL(str(best))
    except OSError:
        pass  # Nothing more we can do

    # Also update LD_LIBRARY_PATH for any child processes
    lib_dir = str(best.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir


_preload_nvrtc_builtins()


# Local model aliases: map short names to local CTranslate2 model directories.
# These are populated by offline conversion from HuggingFace models.
_LOCAL_MODEL_ALIASES: dict[str, Path] = {}

_ct2_cache = Path.home() / ".cache"
for _candidate in [
    ("large-v3-turbo", _ct2_cache / "faster-whisper-large-v3-turbo-ct2"),
]:
    if _candidate[1].exists() and (_candidate[1] / "model.bin").exists():
        _LOCAL_MODEL_ALIASES[_candidate[0]] = _candidate[1]


def resolve_model(name: str) -> str:
    """Resolve a model name, checking local aliases first."""
    if name in _LOCAL_MODEL_ALIASES:
        return str(_LOCAL_MODEL_ALIASES[name])
    return name


# ── Alignment model registry ───────────────────────────────────────────────
# Maps language codes to (model_name, model_type) where model_type is
# "torchaudio" or "huggingface".  Only languages we actively support are
# listed here; WhisperX supports ~39 but we only manage downloads for the
# ones the user cares about.

ALIGNMENT_MODELS: dict[str, tuple[str, str]] = {
    "en": ("WAV2VEC2_ASR_BASE_960H", "torchaudio"),
    "de": ("VOXPOPULI_ASR_BASE_10K_DE", "torchaudio"),
    "fr": ("VOXPOPULI_ASR_BASE_10K_FR", "torchaudio"),
    "es": ("VOXPOPULI_ASR_BASE_10K_ES", "torchaudio"),
    "tr": ("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "huggingface"),
    "fa": ("jonatasgrosman/wav2vec2-large-xlsr-53-persian", "huggingface"),
}

# torchaudio pipeline names → local .pt filenames in ~/.cache/torch/hub/checkpoints/
_TORCHAUDIO_FILENAMES: dict[str, str] = {
    "WAV2VEC2_ASR_BASE_960H": "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
    "VOXPOPULI_ASR_BASE_10K_DE": "wav2vec2_voxpopuli_base_10k_asr_de.pt",
    "VOXPOPULI_ASR_BASE_10K_FR": "wav2vec2_voxpopuli_base_10k_asr_fr.pt",
    "VOXPOPULI_ASR_BASE_10K_ES": "wav2vec2_voxpopuli_base_10k_asr_es.pt",
}

# Approximate download sizes for user display
_MODEL_SIZES: dict[str, str] = {
    "en": "~360 MB",
    "de": "~360 MB",
    "fr": "~360 MB",
    "es": "~360 MB",
    "tr": "~1.2 GB",
    "fa": "~1.2 GB",
}

from meet.languages import LANG_NAMES as _LANG_NAMES  # noqa: E402

MODEL_SIZES = _MODEL_SIZES  # public accessor


class AlignmentModelMissing(Exception):
    """Raised when an alignment model is not cached locally.

    Carries enough context for CLI/GUI to display actionable guidance.
    """

    def __init__(self, lang: str):
        model_name, model_type = ALIGNMENT_MODELS.get(lang, ("unknown", "unknown"))
        self.lang = lang
        self.lang_name = _LANG_NAMES.get(lang, lang)
        self.model_name = model_name
        self.model_type = model_type
        self.estimated_size = _MODEL_SIZES.get(lang, "unknown size")
        super().__init__(
            f"Alignment model for {self.lang_name} ({lang}) is not downloaded.\n"
            f"  Model: {model_name} ({model_type}, {self.estimated_size})\n"
            f"  Run:   meet download {lang}"
        )


def check_alignment_model_cached(lang: str) -> bool:
    """Check if the alignment model for *lang* is cached locally.

    Does NOT download anything — purely a filesystem check.

    Returns True if cached, False if missing.
    """
    if lang not in ALIGNMENT_MODELS:
        # Language not in our registry — WhisperX may still support it
        # (it has 39 languages).  We can't check, so assume it's fine
        # and let WhisperX handle the error at runtime.
        return True

    model_name, model_type = ALIGNMENT_MODELS[lang]

    if model_type == "torchaudio":
        filename = _TORCHAUDIO_FILENAMES.get(model_name)
        if not filename:
            return True  # Unknown filename — can't check
        cache_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename
        return cache_path.exists()

    elif model_type == "huggingface":
        # HuggingFace models are stored as models--<org>--<model>/
        # with a snapshots/ subdirectory containing the actual weights.
        safe_name = model_name.replace("/", "--")
        model_dir = (
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe_name}"
        )
        if not model_dir.exists():
            return False
        # Check that there's at least one snapshot (complete download)
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return False
        # At least one non-empty snapshot directory
        return any(snapshots_dir.iterdir())

    return True


def download_alignment_model(
    lang: str,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Download the alignment model for *lang*.

    Args:
        lang: Language code (e.g. "de", "tr").
        progress_callback: Optional callable(status_message) for progress updates.

    Raises:
        ValueError: If the language is not in our registry.
        RuntimeError: If the download fails.
    """
    if lang not in ALIGNMENT_MODELS:
        supported = ", ".join(sorted(ALIGNMENT_MODELS.keys()))
        raise ValueError(
            f"No alignment model registered for '{lang}'. Supported: {supported}"
        )

    model_name, model_type = ALIGNMENT_MODELS[lang]
    lang_name = _LANG_NAMES.get(lang, lang)
    size = _MODEL_SIZES.get(lang, "unknown size")

    def _status(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    _status(f"Downloading alignment model for {lang_name} ({model_name}, {size})...")

    if model_type == "torchaudio":
        import torchaudio

        # Load the pipeline — this triggers the download.
        bundle = getattr(torchaudio.pipelines, model_name)
        _status(f"Loading torchaudio bundle {model_name}...")
        bundle.get_model()
        _status(f"Alignment model for {lang_name} downloaded successfully.")

    elif model_type == "huggingface":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        _status(f"Downloading HuggingFace model {model_name}...")
        Wav2Vec2Processor.from_pretrained(model_name)
        Wav2Vec2ForCTC.from_pretrained(model_name)
        _status(f"Alignment model for {lang_name} downloaded successfully.")


def get_supported_alignment_languages() -> dict[str, dict[str, str]]:
    """Return info about all supported alignment languages.

    Returns dict of lang_code -> {name, model, type, size, cached}.
    """
    result = {}
    for lang, (model_name, model_type) in ALIGNMENT_MODELS.items():
        result[lang] = {
            "name": _LANG_NAMES.get(lang, lang),
            "model": model_name,
            "type": model_type,
            "size": _MODEL_SIZES.get(lang, "unknown"),
            "cached": check_alignment_model_cached(lang),
        }
    return result


# ── GPU resource management ────────────────────────────────────────────────


def ensure_gpu_available(
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Ensure GPU VRAM is available for transcription.

    Checks if Ollama has models loaded and unloads them to free VRAM.
    This is transparent — we tell the user what we're doing.
    """
    import json as _json

    def _status(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return  # Ollama not running — nothing to do

        data = _json.loads(result.stdout)
        models = data.get("models", [])
        if not models:
            return  # No models loaded

        for model_info in models:
            model_name = model_info.get("name", "unknown")
            _status(
                f"Unloading Ollama model ({model_name}) to free GPU memory for transcription..."
            )
            subprocess.run(
                [
                    "curl",
                    "-s",
                    "http://localhost:11434/api/generate",
                    "-d",
                    _json.dumps({"model": model_name, "keep_alive": 0}),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

        # Brief pause to let VRAM actually free up
        import time

        time.sleep(2)
        _status("GPU memory freed.")

    except (subprocess.TimeoutExpired, _json.JSONDecodeError, FileNotFoundError):
        pass  # Ollama not installed or not responding — fine


@dataclass
class TranscriptionConfig:
    """Configuration for the transcription pipeline."""

    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    language: str = "auto"
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Whether to use the dual-channel layout to improve diarization:
    # If True, the mic channel is labeled as SPEAKER_YOU and the system
    # channel helps confirm remote speaker segments.
    use_dual_channel: bool = True
    # VAD tuning: lower values = more sensitive to quiet/trailing speech.
    # Defaults are tuned below upstream (onset=0.5, offset=0.363) to avoid
    # cutting off the last few seconds of a recording.
    vad_onset: float = 0.40
    vad_offset: float = 0.25
    # Seconds of silence to pad at the end of audio before transcription.
    # Gives the VAD room to properly close the final speech segment.
    audio_pad_seconds: float = 3.0
    # If True, skip the alignment step entirely (no word-level timestamps,
    # fewer segments).  Used when alignment model is missing and the user
    # chose not to download it.
    skip_alignment: bool = False
    # Mixdown mode for stereo recordings:
    #   "mic" = extract left (mic) channel only (default, uses diarization)
    #   "avg" = transcribe each channel separately, label as YOU/REMOTE
    mixdown: str = "mic"

    def __post_init__(self):
        if self.mixdown not in ("mic", "avg"):
            raise ValueError(
                f"Invalid mixdown mode '{self.mixdown}': must be 'mic' or 'avg'"
            )
        # Resolve model aliases (e.g. "large-v3-turbo" -> local CTranslate2 path)
        self.model = resolve_model(self.model)

        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")
        if self.hf_token is None:
            # Try reading from huggingface-cli cache
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                self.hf_token = token_path.read_text().strip()


@dataclass
class Speaker:
    """A speaker in the transcript."""

    id: str
    label: str | None = None  # User-assigned name


@dataclass
class Segment:
    """A single segment of the transcript."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict] | None = None


@dataclass
class Transcript:
    """Complete transcript with metadata."""

    segments: list[Segment]
    speakers: list[Speaker]
    language: str
    audio_file: str
    duration: float | None = None

    def to_text(self) -> str:
        """Plain text output with speaker labels."""
        lines = []
        for seg in self.segments:
            speaker = seg.speaker or "UNKNOWN"
            start = _fmt_time(seg.start)
            end = _fmt_time(seg.end)
            lines.append(f"[{start} --> {end}] {speaker}: {seg.text.strip()}")
        return "\n".join(lines)

    def to_srt(self) -> str:
        """SRT subtitle format with speaker labels."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            speaker = seg.speaker or "UNKNOWN"
            start = _fmt_srt_time(seg.start)
            end = _fmt_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(f"[{speaker}] {seg.text.strip()}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """JSON output with full detail."""
        data = {
            "audio_file": self.audio_file,
            "language": self.language,
            "duration": self.duration,
            "speakers": [{"id": s.id, "label": s.label} for s in self.speakers],
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "speaker": seg.speaker,
                    "words": seg.words,
                }
                for seg in self.segments
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(
        self, output_dir: str | Path, basename: str | None = None
    ) -> dict[str, Path]:
        """Save transcript in all formats. Returns dict of format -> filepath."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if basename is None:
            basename = Path(self.audio_file).stem

        files = {}
        for fmt, ext, content in [
            ("text", ".txt", self.to_text()),
            ("srt", ".srt", self.to_srt()),
            ("json", ".json", self.to_json()),
        ]:
            path = output_dir / f"{basename}{ext}"
            path.write_text(content, encoding="utf-8")
            files[fmt] = path

        return files


from meet.utils import fmt_time as _fmt_time, fmt_srt_time as _fmt_srt_time  # noqa: E402


def _extract_mono(audio_file: Path, channel: int = 0) -> Path:
    """Extract a single channel from a stereo WAV file.

    Args:
        audio_file: Path to stereo WAV file.
        channel: 0 for left (mic), 1 for right (system).

    Returns:
        Path to temporary mono WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    # Use ffmpeg to extract a single channel
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_file),
        "-filter_complex",
        f"[0:a]pan=mono|c0=c{channel}[out]",
        "-map",
        "[out]",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract channel {channel}: {result.stderr}")
    return Path(tmp.name)


def _mixdown_to_mono(audio_file: Path) -> Path:
    """Mix both channels to mono with RMS normalization for transcription.

    Normalizes each channel to equal RMS before averaging so that both
    your mic and the remote participants contribute equally to the mono
    signal.  This prevents the common failure mode where a quiet mic
    channel causes Whisper to miss all remote speech (which lives only
    on the system-audio channel).

    Falls back to a simple ffmpeg average if numpy is unavailable.
    """
    import struct
    import wave

    try:
        import numpy as np
    except ImportError:
        # Fallback: simple ffmpeg average of both channels.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_file),
            "-filter_complex",
            "[0:a]pan=mono|c0=0.5*c0+0.5*c1[out]",
            "-map",
            "[out]",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Mono mixdown failed: {result.stderr}")
        return Path(tmp.name)

    # ── Read stereo WAV (only works for .wav files) ──
    try:
        with wave.open(str(audio_file), "r") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        # Non-WAV format (e.g. OGG/Opus) — fall through to ffmpeg.
        n_channels = 0
        sampwidth = 0
        framerate = 0
        raw = b""

    if n_channels != 2 or sampwidth != 2:
        # Not the expected stereo 16-bit — fall through to ffmpeg.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_file),
            "-filter_complex",
            "[0:a]pan=mono|c0=0.5*c0+0.5*c1[out]",
            "-map",
            "[out]",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Mono mixdown failed: {result.stderr}")
        return Path(tmp.name)

    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    left = data[:, 0].astype(np.float32)  # mic (YOU)
    right = data[:, 1].astype(np.float32)  # system (REMOTE)

    # ── RMS of each channel (ignore near-silence) ──
    silence_thr = 50.0  # ~-50 dBFS for 16-bit
    left_active = left[np.abs(left) > silence_thr]
    right_active = right[np.abs(right) > silence_thr]

    left_rms = np.sqrt(np.mean(left_active**2)) if len(left_active) > 0 else 0.0
    right_rms = np.sqrt(np.mean(right_active**2)) if len(right_active) > 0 else 0.0

    # ── Normalize to equal RMS, then average ──
    if left_rms > 0 and right_rms > 0:
        # Scale the quieter channel up to match the louder one.
        target_rms = max(left_rms, right_rms)
        left_scaled = left * (target_rms / left_rms)
        right_scaled = right * (target_rms / right_rms)
        mono = (left_scaled + right_scaled) * 0.5
    elif right_rms > 0:
        # Mic is dead — use system channel only.
        mono = right
    elif left_rms > 0:
        # System channel is dead — use mic only.
        mono = left
    else:
        # Both silent — just average.
        mono = (left + right) * 0.5

    # Clip to int16 range and convert.
    mono = np.clip(mono, -32768, 32767).astype(np.int16)

    # ── Write mono WAV ──
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "w") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(framerate)
        wf_out.writeframes(mono.tobytes())

    return Path(tmp.name)


def get_audio_duration(audio_file: Path) -> float:
    """Get duration of an audio file in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(audio_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _transcribe_dual_channel(
    audio_file: Path, config: TranscriptionConfig
) -> Transcript:
    """Transcribe each stereo channel separately and merge results.

    Used when mixdown="avg" — instead of mixing both channels into mono
    (which causes WhisperX to suppress the quieter voice), this transcribes
    the mic and system channels independently and merges by timestamp.

    Skips diarization entirely: channel identity = speaker identity.
    """
    import torch
    import whisperx

    duration = get_audio_duration(audio_file)

    mic_path = _extract_mono(audio_file, channel=0)
    sys_path = _extract_mono(audio_file, channel=1)

    try:
        # ── Load model once ──
        vad_options = {
            "vad_onset": config.vad_onset,
            "vad_offset": config.vad_offset,
        }
        whisper_lang = None if config.language == "auto" else config.language

        print(
            f"  Loading model: {config.model} ({config.compute_type}) on {config.device}"
        )
        model = whisperx.load_model(
            config.model,
            config.device,
            compute_type=config.compute_type,
            language=whisper_lang,
            vad_options=vad_options,
        )

        # ── Transcribe mic channel ──
        print(f"  Transcribing mic channel (left)...")
        mic_audio = whisperx.load_audio(str(mic_path))
        if config.audio_pad_seconds > 0:
            import numpy as np

            pad_samples = int(config.audio_pad_seconds * 16000)
            mic_audio = np.concatenate(
                [mic_audio, np.zeros(pad_samples, dtype=mic_audio.dtype)]
            )
        mic_result = model.transcribe(mic_audio, batch_size=config.batch_size)

        # Resolve language from first transcription result
        detected_language = mic_result.get("language", whisper_lang or "en")
        if config.language == "auto":
            print(f"  Detected language: {detected_language}")

        # ── Transcribe system channel ──
        print(f"  Transcribing system channel (right)...")
        sys_audio = whisperx.load_audio(str(sys_path))
        if config.audio_pad_seconds > 0:
            import numpy as np

            pad_samples = int(config.audio_pad_seconds * 16000)
            sys_audio = np.concatenate(
                [sys_audio, np.zeros(pad_samples, dtype=sys_audio.dtype)]
            )
        sys_result = model.transcribe(sys_audio, batch_size=config.batch_size)

        # Free transcription model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # ── Align both channels ──
        if config.skip_alignment:
            print(f"  Skipping alignment (--skip-alignment)")
        elif detected_language in ALIGNMENT_MODELS and not check_alignment_model_cached(
            detected_language
        ):
            gc.collect()
            torch.cuda.empty_cache()
            raise AlignmentModelMissing(detected_language)
        else:
            print(f"  Aligning word timestamps ({detected_language})...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=config.device,
                )
                mic_result = whisperx.align(
                    mic_result["segments"],
                    model_a,
                    metadata,
                    mic_audio,
                    config.device,
                    return_char_alignments=False,
                )
                sys_result = whisperx.align(
                    sys_result["segments"],
                    model_a,
                    metadata,
                    sys_audio,
                    config.device,
                    return_char_alignments=False,
                )
                del model_a
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as align_exc:
                if detected_language in ALIGNMENT_MODELS:
                    raise AlignmentModelMissing(detected_language) from align_exc
                print(
                    f"  Warning: alignment failed ({align_exc}), continuing without word-level timestamps"
                )

        # ── Merge segments ──
        max_t = duration if duration and duration > 0 else float("inf")
        segments: list[Segment] = []

        for seg in mic_result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue
            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker="YOU",
                    words=seg.get("words"),
                )
            )

        for seg in sys_result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue
            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker="REMOTE",
                    words=seg.get("words"),
                )
            )

        segments.sort(key=lambda s: s.start)

        speakers = [
            Speaker(id="YOU", label="YOU"),
            Speaker(id="REMOTE", label="REMOTE"),
        ]

        return Transcript(
            segments=segments,
            speakers=speakers,
            language=detected_language,
            audio_file=str(audio_file),
            duration=duration,
        )

    finally:
        for p in (mic_path, sys_path):
            try:
                p.unlink()
            except OSError:
                pass


def transcribe(
    audio_file: str | Path, config: TranscriptionConfig | None = None
) -> Transcript:
    """Run the full transcription + diarization pipeline.

    Args:
        audio_file: Path to the audio file (WAV preferred, any ffmpeg-supported format works).
        config: Transcription configuration. Uses defaults if not provided.

    Returns:
        Transcript object with diarized segments.
    """
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    if config is None:
        config = TranscriptionConfig()

    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = get_audio_duration(audio_path)

    # If dual-channel, mixdown to mono for the main transcription pipeline
    # but keep the stereo file for channel-aware diarization hints
    is_stereo = _is_stereo(audio_path)

    if is_stereo and config.use_dual_channel and config.mixdown == "avg":
        print(f"  Dual-channel detected: transcribing channels separately")
        return _transcribe_dual_channel(audio_path, config)

    if is_stereo and config.use_dual_channel:
        mono_path = _extract_mono(audio_path, channel=0)
        print(f"  Dual-channel detected: extracting mic channel for transcription")
    else:
        mono_path = audio_path

    try:
        # ── Step 1: Transcribe with faster-whisper ──
        print(
            f"  Loading model: {config.model} ({config.compute_type}) on {config.device}"
        )

        vad_options = {
            "vad_onset": config.vad_onset,
            "vad_offset": config.vad_offset,
        }

        # "auto" means let WhisperX detect the language from the audio.
        whisper_lang = None if config.language == "auto" else config.language

        model = whisperx.load_model(
            config.model,
            config.device,
            compute_type=config.compute_type,
            language=whisper_lang,
            vad_options=vad_options,
        )

        print(
            f"  Transcribing (VAD onset={config.vad_onset}, offset={config.vad_offset})..."
        )
        audio = whisperx.load_audio(str(mono_path))

        # Pad audio with silence at the end so the VAD properly closes the
        # final speech segment instead of cutting it off abruptly.
        if config.audio_pad_seconds > 0:
            import numpy as np

            pad_samples = int(config.audio_pad_seconds * 16000)
            audio = np.concatenate([audio, np.zeros(pad_samples, dtype=audio.dtype)])

        result = model.transcribe(audio, batch_size=config.batch_size)

        # Resolve the actual language (important when auto-detecting).
        detected_language = result.get("language", whisper_lang or "en")
        if config.language == "auto":
            print(f"  Detected language: {detected_language}")

        # Free transcription model memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 2: Align for word-level timestamps ──
        if config.skip_alignment:
            print(f"  Skipping alignment (--skip-alignment)")
        elif detected_language in ALIGNMENT_MODELS and not check_alignment_model_cached(
            detected_language
        ):
            # Model is in our registry but not downloaded — raise so
            # the caller (CLI/GUI) can show an actionable error.
            # Free VRAM first so the error handler can download if needed.
            gc.collect()
            torch.cuda.empty_cache()
            raise AlignmentModelMissing(detected_language)
        else:
            print(f"  Aligning word timestamps ({detected_language})...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=config.device,
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    config.device,
                    return_char_alignments=False,
                )

                del model_a
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as align_exc:
                # For languages NOT in our registry (WhisperX supports ~39),
                # we can't pre-check the cache.  If the download fails at
                # runtime, fall back gracefully since there's no actionable
                # fix we can offer.
                if detected_language in ALIGNMENT_MODELS:
                    # This shouldn't happen (we checked cache above), but
                    # if it does, re-raise as AlignmentModelMissing.
                    raise AlignmentModelMissing(detected_language) from align_exc
                print(
                    f"  Warning: alignment failed ({align_exc}), continuing without word-level timestamps"
                )

        # ── Step 3: Speaker diarization ──
        if config.hf_token:
            print(f"  Running speaker diarization...")
            diarize_model = DiarizationPipeline(
                token=config.hf_token,
                device=config.device,
            )

            diarize_kwargs: dict[str, Any] = {}
            if config.min_speakers is not None:
                diarize_kwargs["min_speakers"] = config.min_speakers
            if config.max_speakers is not None:
                diarize_kwargs["max_speakers"] = config.max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"  Skipping diarization (no HF_TOKEN provided)")

        # ── Step 4: Build Transcript object ──
        # Clamp segment timestamps to actual audio duration (we may have
        # padded silence at the end to help the VAD).
        max_t = duration if duration and duration > 0 else float("inf")

        speaker_ids = set()
        segments = []
        for seg in result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue  # skip segments that fall entirely in the padding
            speaker = seg.get("speaker")
            if speaker:
                speaker_ids.add(speaker)
            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker=speaker,
                    words=seg.get("words"),
                )
            )

        speakers = [Speaker(id=sid) for sid in sorted(speaker_ids)]

        # ── Step 5: Dual-channel speaker labeling ──
        if is_stereo and config.use_dual_channel:
            if len(speakers) >= 2:
                # Pyannote found multiple speakers — map them to YOU/REMOTE
                # using channel energy ratios.
                print(f"  Labeling speakers from dual-channel audio...")
                segments, speakers = _label_speakers_from_channels(
                    audio_path,
                    segments,
                    speakers,
                )
            elif segments:
                # Pyannote found 0-1 speakers — fall back to per-segment
                # channel energy to split into YOU vs REMOTE.
                print(
                    f"  Diarization found {len(speakers)} speaker(s) in stereo"
                    f" audio — splitting by channel energy..."
                )
                segments, speakers = _split_by_channel(audio_path, segments)

        return Transcript(
            segments=segments,
            speakers=speakers,
            language=detected_language,
            audio_file=str(audio_path),
            duration=duration,
        )

    finally:
        # Clean up temp files
        if is_stereo and config.use_dual_channel and mono_path != audio_path:
            try:
                mono_path.unlink()
            except OSError:
                pass


def _label_speakers_from_channels(
    stereo_file: Path,
    segments: list[Segment],
    speakers: list[Speaker],
    sample_rate: int = 16000,
) -> tuple[list[Segment], list[Speaker]]:
    """Use dual-channel stereo info to label speakers as YOU or REMOTE.

    Left channel = mic (your voice), right channel = system (remote participants).
    For each diarized speaker, compute RMS energy on each channel during their
    segments. The speaker with highest mic-channel energy ratio is labeled YOU;
    others are labeled REMOTE (or REMOTE_1, REMOTE_2, etc. if multiple).

    Args:
        stereo_file: Path to the original stereo WAV file.
        segments: Diarized segments with SPEAKER_XX labels.
        speakers: Speaker objects from diarization.
        sample_rate: Sample rate of the audio (default 16000).

    Returns:
        Updated (segments, speakers) with relabeled speaker IDs.
    """
    from meet.audio import read_stereo_channels, compute_speaker_channel_energy

    if not speakers:
        return segments, speakers

    stereo = read_stereo_channels(stereo_file)
    if stereo is None:
        print(f"  Channel labeling: skipping, not stereo or unreadable")
        return segments, speakers

    speaker_mic_ratio = compute_speaker_channel_energy(
        stereo.mic, stereo.system, segments, stereo.sample_rate
    )

    if not speaker_mic_ratio:
        return segments, speakers

    # Log the ratios for debugging
    print(f"  Channel analysis:")
    for spk, ratio in sorted(speaker_mic_ratio.items()):
        label = "mic-dominant" if ratio > 0.5 else "system-dominant"
        print(f"    {spk}: mic_ratio={ratio:.3f} ({label})")

    # The speaker with the highest mic ratio is YOU — but only if they
    # are actually mic-dominant.  When no speaker exceeds the threshold
    # (e.g. only system audio was captured), label everyone as REMOTE.
    you_speaker = max(speaker_mic_ratio, key=lambda s: speaker_mic_ratio[s])

    label_map: dict[str, str] = {}
    if speaker_mic_ratio[you_speaker] > 0.5:
        # At least one speaker is mic-dominant
        label_map[you_speaker] = "YOU"
        remote_speakers = [s for s in sorted(speaker_mic_ratio) if s != you_speaker]
        if len(remote_speakers) == 1:
            label_map[remote_speakers[0]] = "REMOTE"
        else:
            for i, spk in enumerate(remote_speakers):
                label_map[spk] = f"REMOTE_{i + 1}"
    else:
        # No speaker is mic-dominant — label all as REMOTE
        all_speakers = sorted(speaker_mic_ratio)
        if len(all_speakers) == 1:
            label_map[all_speakers[0]] = "REMOTE"
        else:
            for i, spk in enumerate(all_speakers):
                label_map[spk] = f"REMOTE_{i + 1}"

    print(f"  Speaker labels: {label_map}")

    # Relabel segments
    new_segments = []
    for seg in segments:
        new_speaker = (
            label_map.get(seg.speaker, seg.speaker) if seg.speaker else seg.speaker
        )
        new_segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=new_speaker,
                words=seg.words,
            )
        )

    # Relabel speakers
    new_speakers = []
    for spk in speakers:
        new_label = label_map.get(spk.id, spk.id)
        new_speakers.append(Speaker(id=new_label, label=new_label))

    return new_segments, new_speakers


def _split_by_channel(
    stereo_file: Path,
    segments: list[Segment],
) -> tuple[list[Segment], list[Speaker]]:
    """Split a single-speaker diarization into YOU/REMOTE using channel energy.

    This is a fallback for when pyannote diarization detects only one speaker
    in a stereo recording (e.g. due to short duration or GPU-dependent
    floating-point differences in speaker embeddings).

    For each segment, the mic (left) and system (right) channel RMS energy
    is compared.  Segments where the mic channel dominates are labeled YOU;
    segments where the system channel dominates are labeled REMOTE.

    Args:
        stereo_file: Path to the original stereo audio file (WAV or OGG).
        segments:    List of Segment objects (all assigned to a single speaker).

    Returns:
        Updated (segments, speakers) with YOU/REMOTE labels.
        If the stereo file can't be read or all segments land on one speaker,
        the original segments are returned unchanged with a single speaker.
    """
    from meet.audio import read_stereo_channels

    stereo = read_stereo_channels(stereo_file)
    if stereo is None:
        print("  Channel split: skipping, stereo data unreadable")
        speaker_ids = {s.speaker for s in segments if s.speaker}
        return segments, [Speaker(id=sid) for sid in sorted(speaker_ids)]

    sr = stereo.sample_rate
    n = len(stereo.mic)

    import numpy as np

    for seg in segments:
        start = max(0, min(int(seg.start * sr), n))
        end = max(0, min(int(seg.end * sr), n))
        if end <= start:
            continue

        mic_rms = float(np.sqrt(np.mean(stereo.mic[start:end] ** 2)))
        sys_rms = float(np.sqrt(np.mean(stereo.system[start:end] ** 2)))

        # Assign based on dominant channel
        if mic_rms + sys_rms < 1e-8:
            # Near-silence — keep existing label
            continue
        seg.speaker = "YOU" if mic_rms >= sys_rms else "REMOTE"

        # Update word-level labels too
        if seg.words:
            for word in seg.words:
                w_start_t = word.get("start")
                w_end_t = word.get("end")
                if w_start_t is None:
                    word["speaker"] = seg.speaker
                    continue

                ws = max(0, min(int(w_start_t * sr), n))
                we = max(0, min(int((w_end_t or w_start_t) * sr), n))
                if we <= ws:
                    word["speaker"] = seg.speaker
                    continue

                w_mic = float(np.sqrt(np.mean(stereo.mic[ws:we] ** 2)))
                w_sys = float(np.sqrt(np.mean(stereo.system[ws:we] ** 2)))
                if w_mic + w_sys < 1e-8:
                    word["speaker"] = seg.speaker
                else:
                    word["speaker"] = "YOU" if w_mic >= w_sys else "REMOTE"

    # Build speaker list from what was actually assigned
    speaker_ids = {s.speaker for s in segments if s.speaker}
    speakers = [Speaker(id=sid) for sid in sorted(speaker_ids)]

    # Log the split results
    from collections import Counter

    counts = Counter(s.speaker for s in segments if s.speaker)
    print(f"  Channel split result: {dict(counts)}")

    return segments, speakers


def _is_stereo(audio_file: Path) -> bool:
    """Check if an audio file has 2 channels."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "stream=channels",
        "-of",
        "csv=p=0",
        str(audio_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    try:
        return int(result.stdout.strip()) == 2
    except ValueError:
        return False


def post_process(
    transcript: "Transcript",
    output_dir: Path,
    basename: str,
    *,
    summarize: bool = True,
    summary_backend: str | None = None,
    summary_model: str | None = None,
    progress_callback=None,
) -> dict:
    """Run summarization and PDF generation after transcription.

    This is the shared post-processing step used by both CLI and GUI.
    Summary generation is best-effort — failures are caught and logged via
    ``progress_callback`` but do not abort PDF generation.

    Args:
        transcript:       The completed Transcript object.
        output_dir:       Directory to write output files into.
        basename:         Stem for output filenames (e.g. "meeting-20260313-231509").
        summarize:        Whether to attempt AI summarization.
        summary_backend:  Backend override ("ollama" or "openrouter"); None uses default.
        summary_model:    Model name override; None uses the per-backend default.
        progress_callback: Optional callable(str) for status/error messages.

    Returns:
        Dict with keys "summary" (Path or None) and "pdf" (Path or None).
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    result: dict = {"summary": None, "pdf": None}
    summary_result = None

    if summarize:
        try:
            from meet.summarize import summarize as do_summarize, SummaryConfig

            cfg_kwargs: dict = {}
            if summary_backend:
                cfg_kwargs["backend"] = summary_backend
            if summary_model:
                cfg_kwargs["model"] = summary_model
            summary_config = SummaryConfig(**cfg_kwargs)

            summary_result = do_summarize(
                transcript.to_text(),
                summary_config,
                language=transcript.language,
                progress_callback=progress_callback,
            )
            path = summary_result.save(output_dir, basename)
            result["summary"] = path
            _log(f"Summary generated in {summary_result.elapsed_seconds:.1f}s")
        except Exception as exc:
            _log(f"Summary failed: {exc}")

    try:
        from meet.pdf import generate_pdf

        pdf_path = output_dir / f"{basename}.pdf"
        generate_pdf(
            transcript,
            pdf_path,
            summary=summary_result,
            language=getattr(transcript, "language", "en"),
        )
        result["pdf"] = pdf_path
    except Exception as exc:
        _log(f"PDF generation failed: {exc}")

    # ── Compress WAV to OGG/Opus ──
    wav_path = output_dir / f"{basename}.wav"
    if wav_path.exists():
        try:
            from meet.audio import compress_audio

            _log("Compressing audio to OGG/Opus...")
            ogg_path = compress_audio(wav_path)
            result["audio"] = ogg_path
            _log(f"Audio compressed: {ogg_path.name}")
        except Exception as exc:
            _log(f"Audio compression failed (WAV kept): {exc}")

    return result
