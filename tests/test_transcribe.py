"""Tests for meet.transcribe — Transcript dataclass methods and speaker labeling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unittest.mock import patch

from meet.transcribe import Segment, Speaker, Transcript, TranscriptionConfig, _fmt_time, _fmt_srt_time
from meet.transcribe import transcribe as do_transcribe


# ─── Transcript.to_text() ──────────────────────────────────────────────────

class TestToText:
    def test_basic_format(self, transcript):
        text = transcript.to_text()
        lines = text.strip().split("\n")
        assert len(lines) == 6

    def test_speaker_labels_present(self, transcript):
        text = transcript.to_text()
        assert "YOU:" in text
        assert "REMOTE_1:" in text
        assert "REMOTE_2:" in text

    def test_timestamp_format(self, transcript):
        text = transcript.to_text()
        # First line: [00:00:00 --> 00:00:05]
        assert "[00:00:00 --> 00:00:05]" in text

    def test_missing_speaker(self):
        t = Transcript(
            segments=[Segment(start=0, end=1, text="hello", speaker=None)],
            speakers=[], language="en", audio_file="test.wav",
        )
        assert "UNKNOWN:" in t.to_text()


# ─── Transcript.to_srt() ───────────────────────────────────────────────────

class TestToSrt:
    def test_srt_numbering(self, transcript):
        srt = transcript.to_srt()
        # SRT entries are numbered 1..6
        assert srt.startswith("1\n")
        assert "\n6\n" in srt

    def test_srt_timestamp_format(self, transcript):
        srt = transcript.to_srt()
        # First timestamp
        assert "00:00:00,000 --> 00:00:05,500" in srt

    def test_srt_speaker_brackets(self, transcript):
        srt = transcript.to_srt()
        assert "[YOU]" in srt
        assert "[REMOTE_1]" in srt


# ─── Transcript.to_json() ──────────────────────────────────────────────────

class TestToJson:
    def test_round_trip(self, transcript):
        """JSON output should parse back to the same data."""
        data = json.loads(transcript.to_json())
        assert data["language"] == "en"
        assert data["duration"] == 42.0
        assert len(data["segments"]) == 6
        assert len(data["speakers"]) == 3

    def test_segment_fields(self, transcript):
        data = json.loads(transcript.to_json())
        seg = data["segments"][0]
        assert seg["start"] == 0.0
        assert seg["end"] == 5.5
        assert seg["speaker"] == "YOU"
        assert "Hello everyone" in seg["text"]

    def test_speaker_fields(self, transcript):
        data = json.loads(transcript.to_json())
        sp = data["speakers"][0]
        assert sp["id"] == "YOU"
        assert sp["label"] == "YOU"


# ─── Transcript.save() ─────────────────────────────────────────────────────

class TestSave:
    def test_creates_all_files(self, transcript, tmp_path):
        files = transcript.save(tmp_path, basename="test")
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.srt").exists()
        assert (tmp_path / "test.json").exists()
        assert "text" in files
        assert "srt" in files
        assert "json" in files

    def test_json_content_matches(self, transcript, tmp_path):
        transcript.save(tmp_path, basename="test")
        data = json.loads((tmp_path / "test.json").read_text())
        assert len(data["segments"]) == 6

    def test_creates_output_dir(self, transcript, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        transcript.save(subdir, basename="test")
        assert (subdir / "test.txt").exists()


# ─── _label_speakers_from_channels() ───────────────────────────────────────

class TestLabelSpeakersFromChannels:
    def test_assigns_you_to_mic_dominant(self, transcript, stereo_wav_with_speakers):
        """The speaker with highest mic energy should be labeled YOU."""
        from meet.transcribe import _label_speakers_from_channels

        # Use raw SPEAKER_XX labels for the input
        raw_segments = [
            Segment(start=s.start, end=s.end, text=s.text, speaker=f"SPEAKER_{i:02d}")
            for i, s in enumerate(transcript.segments)
        ]
        # Map: segments 0,3 are YOU (SPEAKER_00, SPEAKER_03)
        #       segments 1,4 are REMOTE_1 (SPEAKER_01, SPEAKER_04)
        #       segments 2,5 are REMOTE_2 (SPEAKER_02, SPEAKER_05)
        # But diarization would group by speaker, not by segment index.
        # Let's use consistent speaker IDs:
        raw_segments[0].speaker = "SPEAKER_00"  # YOU
        raw_segments[3].speaker = "SPEAKER_00"  # YOU
        raw_segments[1].speaker = "SPEAKER_01"  # REMOTE_1
        raw_segments[4].speaker = "SPEAKER_01"  # REMOTE_1
        raw_segments[2].speaker = "SPEAKER_02"  # REMOTE_2
        raw_segments[5].speaker = "SPEAKER_02"  # REMOTE_2

        raw_speakers = [
            Speaker(id="SPEAKER_00"), Speaker(id="SPEAKER_01"), Speaker(id="SPEAKER_02"),
        ]

        new_segs, new_spks = _label_speakers_from_channels(
            stereo_wav_with_speakers, raw_segments, raw_speakers,
        )

        # Find which raw speaker became YOU
        you_segs = [s for s in new_segs if s.speaker == "YOU"]
        assert len(you_segs) == 2
        # The YOU segments should correspond to our mic-loud segments (0.0-5.5 and 20.3-28.0)
        you_starts = sorted(s.start for s in you_segs)
        assert you_starts[0] == 0.0
        assert you_starts[1] == 20.3

    def test_remote_speakers_labeled(self, transcript, stereo_wav_with_speakers):
        """Non-YOU speakers should get REMOTE labels."""
        from meet.transcribe import _label_speakers_from_channels

        raw_segments = []
        speaker_map = {0: "SPEAKER_00", 1: "SPEAKER_01", 2: "SPEAKER_02",
                       3: "SPEAKER_00", 4: "SPEAKER_01", 5: "SPEAKER_02"}
        for i, s in enumerate(transcript.segments):
            raw_segments.append(
                Segment(start=s.start, end=s.end, text=s.text, speaker=speaker_map[i])
            )

        raw_speakers = [
            Speaker(id="SPEAKER_00"), Speaker(id="SPEAKER_01"), Speaker(id="SPEAKER_02"),
        ]

        new_segs, new_spks = _label_speakers_from_channels(
            stereo_wav_with_speakers, raw_segments, raw_speakers,
        )

        remote_labels = {s.speaker for s in new_segs if s.speaker != "YOU"}
        # With 2 remote speakers, labels should be REMOTE_1 and REMOTE_2
        assert "REMOTE_1" in remote_labels or "REMOTE_2" in remote_labels

    def test_empty_speakers(self, stereo_wav_with_speakers):
        """Empty speaker list should return inputs unchanged."""
        from meet.transcribe import _label_speakers_from_channels

        segs, spks = _label_speakers_from_channels(
            stereo_wav_with_speakers, [], [],
        )
        assert segs == []
        assert spks == []

    def test_no_mic_dominant_speaker_all_remote(self, tmp_path):
        """When no speaker has mic_ratio > 0.5, all should be labeled REMOTE."""
        import numpy as np
        import wave as wave_mod
        from meet.transcribe import _label_speakers_from_channels

        # Create a stereo WAV where system channel is always louder
        sr = 16000
        duration = 10.0
        n_frames = int(duration * sr)
        t = np.linspace(0, duration, n_frames, dtype=np.float32)

        # Mic: very quiet, System: loud
        mic = (500 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        system = (20000 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)
        stereo = np.column_stack((mic, system)).flatten()

        wav_path = tmp_path / "system-only.wav"
        with wave_mod.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        segments = [
            Segment(start=0.0, end=5.0, text="Hello", speaker="SPEAKER_00"),
            Segment(start=5.0, end=10.0, text="World", speaker="SPEAKER_01"),
        ]
        speakers = [Speaker(id="SPEAKER_00"), Speaker(id="SPEAKER_01")]

        new_segs, new_spks = _label_speakers_from_channels(
            wav_path, segments, speakers,
        )

        # No speaker should be labeled YOU
        labels = {s.speaker for s in new_segs}
        assert "YOU" not in labels
        # All should be REMOTE variants
        assert all("REMOTE" in label for label in labels)


# ─── TranscriptionConfig validation ──────────────────────────────────────

class TestTranscriptionConfig:
    def test_default_mixdown_is_mono(self):
        config = TranscriptionConfig()
        assert config.mixdown == "mono"

    def test_valid_mixdown_dual(self):
        config = TranscriptionConfig(mixdown="dual")
        assert config.mixdown == "dual"

    def test_invalid_mixdown_raises(self):
        with pytest.raises(ValueError, match="Invalid mixdown mode"):
            TranscriptionConfig(mixdown="stereo")


# ─── Dual-channel dispatch ────────────────────────────────────────────────

class TestDualChannelDispatch:
    def test_dual_mixdown_dispatches_to_dual_channel(self, stereo_wav):
        """Stereo audio with mixdown='dual' should call _transcribe_dual_channel."""
        dummy = Transcript(
            segments=[], speakers=[], language="en",
            audio_file=str(stereo_wav), duration=5.0,
        )
        with patch("meet.transcribe._transcribe_dual_channel", return_value=dummy) as mock_dual:
            config = TranscriptionConfig(mixdown="dual")
            result = do_transcribe(str(stereo_wav), config)
            mock_dual.assert_called_once()
            assert result is dummy

    def test_mono_mixdown_does_not_dispatch_to_dual_channel(self, stereo_wav):
        """Stereo audio with mixdown='mono' should NOT call _transcribe_dual_channel."""
        with patch("meet.transcribe._transcribe_dual_channel") as mock_dual:
            config = TranscriptionConfig(mixdown="mono")
            try:
                do_transcribe(str(stereo_wav), config)
            except Exception:
                pass  # Full pipeline needs GPU/models — we only care about dispatch
            mock_dual.assert_not_called()
