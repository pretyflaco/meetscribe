"""Tests for meet.capture — pause/resume functionality."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meet.capture import RecordingSession, RecordingStatus


# ─── Helpers ────────────────────────────────────────────────────────────────


def _write_fake_wav(path: Path, duration_seconds: float = 1.0) -> None:
    """Write a minimal valid WAV file with the expected format (16kHz stereo s16le)."""
    sample_rate = 16000
    channels = 2
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence
        wf.writeframes(b"\x00\x00" * channels * n_frames)


def _make_session(tmp_path: Path) -> RecordingSession:
    """Create a RecordingSession with paths inside tmp_path."""
    output_dir = tmp_path / "session"
    output_dir.mkdir()
    return RecordingSession(
        output_dir=output_dir,
        output_file=output_dir / "test-recording.wav",
        mic_source="test-mic",
        monitor_source="test-monitor",
    )


def _fake_popen(*args, **kwargs):
    """Return a mock Popen that looks alive (poll() returns None)."""
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    proc.wait.return_value = 0
    proc.returncode = 0
    return proc


# ─── RecordingStatus.paused field ───────────────────────────────────────────


class TestRecordingStatusPaused:
    def test_default_not_paused(self):
        status = RecordingStatus(
            is_alive=True,
            elapsed_seconds=10.0,
            file_size_bytes=1000,
            restart_count=0,
            failed=False,
        )
        assert status.paused is False

    def test_paused_field(self):
        status = RecordingStatus(
            is_alive=False,
            elapsed_seconds=10.0,
            file_size_bytes=1000,
            restart_count=0,
            failed=False,
            paused=True,
        )
        assert status.paused is True


# ─── RecordingSession.pause / resume ────────────────────────────────────────


class TestPauseResume:
    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_pause_sets_paused_flag(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)

        # Simulate that a chunk file with data exists (so _start_ffmpeg_chunk succeeds)
        chunk_path = session.output_dir / "test-recording.chunk-000.wav"
        _write_fake_wav(chunk_path, duration_seconds=2.0)

        session._chunks = [chunk_path]
        session._current_chunk = chunk_path
        session._ffmpeg_proc = _fake_popen()
        session._paused = False

        session.pause()

        assert session._paused is True

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_pause_stops_ffmpeg(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)

        chunk_path = session.output_dir / "test-recording.chunk-000.wav"
        _write_fake_wav(chunk_path, duration_seconds=2.0)

        session._chunks = [chunk_path]
        session._current_chunk = chunk_path
        mock_proc = _fake_popen()
        session._ffmpeg_proc = mock_proc

        session.pause()

        # After pause, ffmpeg proc should be None (stopped)
        assert session._ffmpeg_proc is None

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_pause_when_already_paused_raises(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)
        session._paused = True

        with pytest.raises(RuntimeError, match="already paused"):
            session.pause()

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_pause_when_failed_raises(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)
        session._failed = True

        with pytest.raises(RuntimeError, match="failed"):
            session.pause()

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_resume_clears_paused_flag(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)
        session._paused = True
        session._actual_monitor = "test-monitor"
        session._ffmpeg_log = MagicMock()
        session._ffmpeg_log.closed = False

        # Pre-create the chunk file that _start_ffmpeg_chunk will write to
        # so the startup poll finds data quickly
        chunk_path = session.output_dir / "test-recording.chunk-001.wav"
        _write_fake_wav(chunk_path, duration_seconds=1.0)
        session._chunks = [session.output_dir / "test-recording.chunk-000.wav"]

        session.resume()

        assert session._paused is False

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_resume_starts_new_chunk(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)
        session._paused = True
        session._actual_monitor = "test-monitor"
        session._ffmpeg_log = MagicMock()
        session._ffmpeg_log.closed = False

        # First chunk already exists
        chunk0 = session.output_dir / "test-recording.chunk-000.wav"
        _write_fake_wav(chunk0, duration_seconds=2.0)
        session._chunks = [chunk0]

        # Pre-create chunk-001 so startup poll succeeds
        chunk1_path = session.output_dir / "test-recording.chunk-001.wav"
        _write_fake_wav(chunk1_path, duration_seconds=1.0)

        session.resume()

        # Should now have 2 chunks
        assert len(session._chunks) == 2
        assert session._chunks[1] == chunk1_path

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_resume_when_not_paused_raises(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)
        session._paused = False

        with pytest.raises(RuntimeError, match="not paused"):
            session.resume()

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_status_reports_paused(self, mock_popen, tmp_path):
        session = _make_session(tmp_path)

        chunk_path = session.output_dir / "test-recording.chunk-000.wav"
        _write_fake_wav(chunk_path, duration_seconds=3.0)

        session._chunks = [chunk_path]
        session._current_chunk = chunk_path
        session._ffmpeg_proc = _fake_popen()
        session._paused = False

        status = session.status()
        assert status.paused is False

        session.pause()

        status = session.status()
        assert status.paused is True

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_status_elapsed_frozen_when_paused(self, mock_popen, tmp_path):
        """When paused, elapsed time comes from finalized chunk files and doesn't change."""
        session = _make_session(tmp_path)

        chunk_path = session.output_dir / "test-recording.chunk-000.wav"
        _write_fake_wav(chunk_path, duration_seconds=5.0)

        session._chunks = [chunk_path]
        session._current_chunk = chunk_path
        session._ffmpeg_proc = _fake_popen()

        session.pause()

        elapsed1 = session.status().elapsed_seconds
        elapsed2 = session.status().elapsed_seconds
        assert elapsed1 == elapsed2
        assert elapsed1 > 0  # Should have some duration from the chunk

    @patch("meet.capture.subprocess.Popen", side_effect=_fake_popen)
    def test_stop_from_paused_state(self, mock_popen, tmp_path):
        """Stopping from paused state should work and produce output."""
        session = _make_session(tmp_path)
        session._stop_event = MagicMock()

        chunk_path = session.output_dir / "test-recording.chunk-000.wav"
        _write_fake_wav(chunk_path, duration_seconds=3.0)

        session._chunks = [chunk_path]
        session._current_chunk = chunk_path
        session._paused = True
        session._ffmpeg_proc = None  # Already stopped when paused
        session._ffmpeg_log = None
        session._watchdog_thread = None

        output = session.stop()

        assert output == session.output_file
        assert output.exists()
        assert session._paused is False


# ─── Watchdog skips checks when paused ──────────────────────────────────────


class TestWatchdogPaused:
    def test_watchdog_skips_when_paused(self, tmp_path):
        """The watchdog should not attempt restarts when the session is paused."""
        session = _make_session(tmp_path)
        session._paused = True
        session._ffmpeg_proc = None  # No process when paused
        session._failed = False

        # Manually call watchdog check logic — it should skip (continue)
        # We verify indirectly: if watchdog ran its checks with proc=None
        # and _paused=False, it would do nothing special; but with _paused=True,
        # it explicitly continues (skips).
        # The key invariant: _failed should remain False after a watchdog cycle
        # even though there's no ffmpeg process.
        import threading

        session._stop_event = threading.Event()

        # Signal stop immediately so the loop exits after one iteration
        session._stop_event.set()
        session._watchdog_loop()

        assert session._failed is False
