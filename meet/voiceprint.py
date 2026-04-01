"""Speaker voiceprint database for cross-session speaker recognition.

Extracts speaker embeddings from labeled sessions using pyannote's bundled
WeSpeakerResNet34 model, stores averaged profiles per person, and identifies
speakers in new meetings by cosine similarity.

Profile database lives at ~/.config/meet/speaker_profiles.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

PROFILES_PATH = Path.home() / ".config" / "meet" / "speaker_profiles.json"
MATCH_THRESHOLD = 0.65  # cosine similarity — below this, don't auto-label
MIN_SEGMENT_DURATION = 1.5   # seconds — skip very short segments for embedding
MAX_SEGMENTS_PER_SPEAKER = 10  # how many segments to average per speaker


# ─── Embedding model loading ──────────────────────────────────────────────────

_inference = None  # lazy singleton


def _get_inference():
    """Load and return the pyannote WeSpeaker embedding Inference object.

    Uses the embedding model bundled inside the cached
    pyannote/speaker-diarization-community-1 model, which is always present
    if diarization has been run at least once.
    """
    global _inference
    if _inference is not None:
        return _inference

    from pyannote.audio import Inference, Model

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = sorted(hub.glob("models--pyannote--speaker-diarization*"))
    if not candidates:
        raise RuntimeError(
            "pyannote diarization model not found in HuggingFace cache. "
            "Run meet on a recording first to download it."
        )

    # Find the embedding model inside the first matching snapshot
    emb_path = None
    for model_dir in candidates:
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        for snap in sorted(snapshots_dir.iterdir()):
            candidate = snap / "embedding" / "pytorch_model.bin"
            if candidate.exists():
                emb_path = candidate
                break
        if emb_path:
            break

    if emb_path is None:
        raise RuntimeError(
            "Could not find embedding/pytorch_model.bin inside pyannote model cache."
        )

    log.debug("Loading embedding model from %s", emb_path)
    model = Model.from_pretrained(emb_path)
    _inference = Inference(model, window="whole")
    return _inference


# ─── Profile storage ──────────────────────────────────────────────────────────

class SpeakerProfile(NamedTuple):
    name: str
    embedding: np.ndarray  # 256-dim, L2-normalized
    n_sessions: int


def load_profiles() -> dict[str, SpeakerProfile]:
    """Load speaker profiles from disk. Returns empty dict if not found."""
    if not PROFILES_PATH.exists():
        return {}

    try:
        data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load speaker profiles: %s", exc)
        return {}

    profiles: dict[str, SpeakerProfile] = {}
    for name, info in data.items():
        emb = np.array(info["embedding"], dtype=np.float32)
        profiles[name] = SpeakerProfile(
            name=name,
            embedding=emb,
            n_sessions=info.get("n_sessions", 1),
        )
    return profiles


def save_profiles(profiles: dict[str, SpeakerProfile]) -> None:
    """Save speaker profiles to disk."""
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        name: {
            "embedding": p.embedding.tolist(),
            "n_sessions": p.n_sessions,
        }
        for name, p in profiles.items()
    }
    PROFILES_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _merge_embedding(
    existing: SpeakerProfile,
    new_emb: np.ndarray,
) -> SpeakerProfile:
    """Weighted running average of speaker embeddings."""
    n = existing.n_sessions
    merged = (existing.embedding * n + new_emb) / (n + 1)
    merged = _l2_norm(merged)
    return SpeakerProfile(
        name=existing.name,
        embedding=merged,
        n_sessions=n + 1,
    )


def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ─── Embedding extraction ─────────────────────────────────────────────────────

def _extract_channel_audio(audio_path: Path, channel: str) -> tuple[np.ndarray, int] | None:
    """Extract a single audio channel as float32 array.

    Args:
        audio_path: Path to stereo audio file.
        channel: 'mic' (left) or 'system' (right).

    Returns:
        (samples_float32, sample_rate) or None on failure.
    """
    import subprocess

    ch_idx = 0 if channel == "mic" else 1
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", str(audio_path),
        "-filter_complex", f"[0:a]pan=mono|c0=c{ch_idx}[out]",
        "-map", "[out]",
        "-ar", "16000",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        samples /= 32768.0  # normalize to [-1, 1] for pyannote
        return samples, 16000
    except Exception as exc:
        log.debug("Channel audio extraction failed: %s", exc)
        return None


def _embed_segments(
    samples: np.ndarray,
    sample_rate: int,
    segments: list,  # list of (start, end) float tuples
    inference,
) -> np.ndarray | None:
    """Extract and average embeddings from a list of time segments.

    Args:
        samples: float32 mono audio array (normalized to [-1, 1]).
        sample_rate: typically 16000.
        segments: list of (start_sec, end_sec) tuples.
        inference: pyannote Inference object (window='whole').

    Returns:
        Averaged, L2-normalized 256-dim embedding, or None if extraction fails.
    """
    import torch

    embeddings = []
    for start, end in segments:
        start_frame = int(start * sample_rate)
        end_frame = min(int(end * sample_rate), len(samples))
        clip = samples[start_frame:end_frame]

        if len(clip) < int(sample_rate * MIN_SEGMENT_DURATION):
            continue  # too short

        try:
            # Pass the clip directly as a torch tensor — pyannote Inference
            # accepts {'waveform': Tensor(C, T), 'sample_rate': int}
            waveform = torch.from_numpy(clip).unsqueeze(0)  # (1, T)
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            emb = inference(audio_dict)
            # emb is an np.ndarray of shape (dim,) when window='whole'
            if emb is not None:
                vec = np.array(emb).flatten().astype(np.float32)
                if len(vec) > 0:
                    embeddings.append(vec)
        except Exception as exc:
            log.debug("Embedding extraction failed for segment %.1f-%.1f: %s", start, end, exc)
            continue

    if not embeddings:
        return None

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    return _l2_norm(avg)


def extract_speaker_embeddings(
    audio_path: Path,
    transcript_segments: list,       # list of Segment objects
    speaker_labels: dict[str, str],  # {REMOTE_N: "Name"} or {speaker_id: name}
    channel_map: dict[str, str],     # {speaker_id: 'mic' | 'system'}
) -> dict[str, np.ndarray]:
    """Extract embeddings for all labeled speakers in a session.

    Args:
        audio_path: Path to the session audio file (OGG or WAV).
        transcript_segments: Segment objects with .speaker, .start, .end.
        speaker_labels: Map from speaker_id to human name.
        channel_map: Map from speaker_id to dominant channel ('mic' or 'system').

    Returns:
        Dict mapping human name to 256-dim embedding array.
        Speakers with insufficient audio are omitted.
    """
    inference = _get_inference()

    # Group segments by speaker
    segs_by_speaker: dict[str, list[tuple[float, float]]] = {}
    for seg in transcript_segments:
        if not seg.speaker or seg.speaker not in speaker_labels:
            continue
        duration = seg.end - seg.start
        if duration < MIN_SEGMENT_DURATION:
            continue
        segs_by_speaker.setdefault(seg.speaker, []).append((seg.start, seg.end))

    # For each speaker, pick the longest segments up to MAX_SEGMENTS_PER_SPEAKER
    result: dict[str, np.ndarray] = {}
    for speaker_id, name in speaker_labels.items():
        segs = segs_by_speaker.get(speaker_id, [])
        if not segs:
            log.debug("No suitable segments for speaker %s (%s)", speaker_id, name)
            continue

        # Sort by duration descending, take top N
        segs.sort(key=lambda s: s[1] - s[0], reverse=True)
        selected = segs[:MAX_SEGMENTS_PER_SPEAKER]

        channel = channel_map.get(speaker_id, "system")
        channel_data = _extract_channel_audio(audio_path, channel)
        if channel_data is None:
            log.warning("Could not extract %s channel for %s", channel, name)
            continue

        samples, sr = channel_data
        emb = _embed_segments(samples, sr, selected, inference)
        if emb is not None:
            result[name] = emb
            log.debug(
                "Extracted embedding for %s (%s) from %d segments",
                name, speaker_id, len(selected),
            )
        else:
            log.warning("Could not extract embedding for %s (%s)", name, speaker_id)

    return result


# ─── Session enrollment ───────────────────────────────────────────────────────

def enroll_session(
    session_dir: Path,
    progress_callback=None,
) -> dict[str, bool]:
    """Enroll all labeled speakers from a session into the profile database.

    Args:
        session_dir: Path to a labeled session directory (must have
                     session.json with speaker_labels and a transcript JSON).
        progress_callback: Optional callable(str) for status messages.

    Returns:
        Dict mapping speaker name to True (enrolled) or False (failed).
    """
    from meet.label import _find_session_files, _load_transcript
    from meet.audio import read_stereo_channels
    from meet.transcribe import Segment

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            log.info(msg)

    session_dir = Path(session_dir)
    files = _find_session_files(session_dir)

    # Load speaker labels from session.json
    session_json = files.get("session")
    if not session_json or not session_json.exists():
        raise FileNotFoundError(f"No session.json found in {session_dir}")

    meta = json.loads(session_json.read_text(encoding="utf-8"))
    speaker_labels = meta.get("speaker_labels", {})
    if not speaker_labels:
        raise ValueError(f"No speaker_labels in {session_json} — label this session first")

    # Load transcript
    transcript_json = files.get("json")
    if not transcript_json or not transcript_json.exists():
        raise FileNotFoundError(f"No transcript JSON found in {session_dir}")

    transcript = _load_transcript(transcript_json)

    # Build a reverse map: relabeled_name -> original speaker_id
    # The transcript has already been relabeled, so segments use real names.
    # speaker_labels maps REMOTE_N -> Name, so we need to find segments by name.
    # Build a channel map: name -> channel based on whether it was YOU or REMOTE
    channel_map: dict[str, str] = {}
    for speaker_id, name in speaker_labels.items():
        if speaker_id == "YOU":
            channel_map[name] = "mic"
        else:
            channel_map[name] = "system"

    # Transcript segments already have real names as speaker IDs (post-relabel)
    # Build a pseudo speaker_labels where key == value (name -> name)
    name_to_name = {name: name for name in speaker_labels.values()}

    # Find audio file
    audio_path = files.get("wav")
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError(f"No audio file found in {session_dir}")

    _log(f"  Extracting embeddings from {audio_path.name}...")

    embeddings = extract_speaker_embeddings(
        audio_path,
        transcript.segments,
        name_to_name,
        channel_map,
    )

    if not embeddings:
        _log("  No embeddings extracted — check audio file and transcript.")
        return {}

    # Merge into profiles
    profiles = load_profiles()
    status: dict[str, bool] = {}

    for name, emb in embeddings.items():
        if name in profiles:
            profiles[name] = _merge_embedding(profiles[name], emb)
            _log(f"  Updated profile: {name} (now {profiles[name].n_sessions} sessions)")
        else:
            profiles[name] = SpeakerProfile(
                name=name,
                embedding=emb,
                n_sessions=1,
            )
            _log(f"  New profile: {name}")
        status[name] = True

    save_profiles(profiles)
    return status


# ─── Speaker identification ───────────────────────────────────────────────────

class SpeakerMatch(NamedTuple):
    name: str
    confidence: float  # cosine similarity in [0, 1]


def identify_speakers(
    audio_path: Path,
    transcript_segments: list,
    speakers: list,              # list of Speaker objects (have .id attribute)
    channel_map: dict[str, str], # speaker_id -> 'mic' | 'system'
) -> dict[str, SpeakerMatch]:
    """Identify speakers in a new meeting against the profile database.

    Uses cosine similarity with greedy 1:1 matching (best match first).
    Only returns matches above MATCH_THRESHOLD.

    Args:
        audio_path: Path to the session audio (OGG or WAV).
        transcript_segments: Segment objects from the new transcript.
        speakers: Speaker objects — their .id fields are used.
        channel_map: Map from speaker_id to dominant channel.

    Returns:
        Dict mapping speaker_id to (matched_name, confidence).
        Speakers without a confident match are omitted.
    """
    profiles = load_profiles()
    if not profiles:
        return {}

    # Build speaker_labels dict: all speaker IDs map to themselves
    # (we pass them through extract_speaker_embeddings)
    all_ids = {sp.id: sp.id for sp in speakers}

    try:
        inference = _get_inference()
    except Exception as exc:
        log.warning("Could not load embedding model: %s", exc)
        return {}

    # Extract embeddings for each speaker in the new transcript
    new_embeddings: dict[str, np.ndarray] = {}
    for speaker_id in all_ids:
        segs = [
            (seg.start, seg.end)
            for seg in transcript_segments
            if seg.speaker == speaker_id and (seg.end - seg.start) >= MIN_SEGMENT_DURATION
        ]
        if not segs:
            continue
        segs.sort(key=lambda s: s[1] - s[0], reverse=True)
        selected = segs[:MAX_SEGMENTS_PER_SPEAKER]

        channel = channel_map.get(speaker_id, "system")
        channel_data = _extract_channel_audio(audio_path, channel)
        if channel_data is None:
            continue

        samples, sr = channel_data
        emb = _embed_segments(samples, sr, selected, inference)
        if emb is not None:
            new_embeddings[speaker_id] = emb

    if not new_embeddings:
        return {}

    # Compute cosine similarity matrix (new speakers × profiles)
    profile_names = list(profiles.keys())
    profile_matrix = np.stack([profiles[n].embedding for n in profile_names])  # (P, 256)

    speaker_ids = list(new_embeddings.keys())
    new_matrix = np.stack([new_embeddings[sid] for sid in speaker_ids])  # (S, 256)

    # Cosine similarity: since both are L2-normalized, dot product = cosine similarity
    sim_matrix = new_matrix @ profile_matrix.T  # (S, P)

    # Greedy 1:1 matching: assign best available match above threshold
    matches: dict[str, SpeakerMatch] = {}
    used_profiles: set[int] = set()

    # Iterate over (speaker_idx, profile_idx) pairs sorted by similarity desc
    indices = np.argsort(-sim_matrix, axis=None)  # flat indices, sorted desc
    for flat_idx in indices:
        s_idx, p_idx = np.unravel_index(flat_idx, sim_matrix.shape)
        score = float(sim_matrix[s_idx, p_idx])
        if score < MATCH_THRESHOLD:
            break

        speaker_id = speaker_ids[s_idx]
        profile_name = profile_names[p_idx]

        if speaker_id in matches or p_idx in used_profiles:
            continue

        matches[speaker_id] = SpeakerMatch(name=profile_name, confidence=score)
        used_profiles.add(p_idx)

    return matches


def update_profiles_from_confirmed_labels(
    audio_path: Path,
    transcript_segments: list,
    confirmed_label_map: dict[str, str],  # speaker_id -> confirmed name
    channel_map: dict[str, str],
) -> None:
    """Update profiles with confirmed labels from a just-completed meeting.

    Called automatically after the GUI's label dialog is accepted, so that
    the database improves over time without explicit `meet enroll` runs.

    Args:
        audio_path: Session audio file.
        transcript_segments: Segments from the transcript.
        confirmed_label_map: Map from speaker_id to confirmed human name.
        channel_map: Map from speaker_id to 'mic' | 'system'.
    """
    if not confirmed_label_map:
        return

    try:
        inference = _get_inference()
    except Exception as exc:
        log.warning("Could not load embedding model for profile update: %s", exc)
        return

    profiles = load_profiles()
    updated = []

    for speaker_id, name in confirmed_label_map.items():
        segs = [
            (seg.start, seg.end)
            for seg in transcript_segments
            if seg.speaker == speaker_id and (seg.end - seg.start) >= MIN_SEGMENT_DURATION
        ]
        if not segs:
            continue

        segs.sort(key=lambda s: s[1] - s[0], reverse=True)
        selected = segs[:MAX_SEGMENTS_PER_SPEAKER]

        channel = channel_map.get(speaker_id, "system")
        channel_data = _extract_channel_audio(audio_path, channel)
        if channel_data is None:
            continue

        samples, sr = channel_data
        emb = _embed_segments(samples, sr, selected, inference)
        if emb is None:
            continue

        if name in profiles:
            profiles[name] = _merge_embedding(profiles[name], emb)
        else:
            profiles[name] = SpeakerProfile(name=name, embedding=emb, n_sessions=1)
        updated.append(name)

    if updated:
        save_profiles(profiles)
        log.info("Updated voice profiles for: %s", ", ".join(updated))
