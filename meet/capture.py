"""Audio capture module for recording meeting audio.

Captures dual-channel audio: microphone (your voice) on one channel,
system audio (remote participants) on the other. Uses PipeWire/PulseAudio
monitor sources via ffmpeg.

Reliability features:
- ffmpeg stderr goes to a log file (prevents pipe buffer deadlock)
- Watchdog thread monitors ffmpeg health and file growth
- Auto-restart on ffmpeg failure with chunk-based recording
- Chunk stitching on stop via ffmpeg concat
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO


# ─── Constants ───────────────────────────────────────────────────────────────

_WATCHDOG_INTERVAL = 3.0  # seconds between health checks
_STARTUP_POLL_INTERVAL = 0.1  # seconds between startup file-size checks
_STARTUP_TIMEOUT = 10.0  # max seconds to wait for ffmpeg to produce data
_MAX_RESTART_ATTEMPTS = 5  # max consecutive restart attempts
_STALL_TIMEOUT = 15.0  # seconds of no file growth before declaring stall
DRAIN_SECONDS = 10  # seconds to keep recording after user requests stop,
# allowing ffmpeg's ~0.9x realtime pipeline to flush

# WAV format constants (must match ffmpeg output settings)
_WAV_HEADER_BYTES = 44  # standard WAV header size
_SAMPLE_RATE = 16000  # Hz
_CHANNELS = 2  # stereo (left=mic, right=system)
_BYTES_PER_SAMPLE = 2  # pcm_s16le = 16-bit = 2 bytes
_BYTES_PER_SECOND = _SAMPLE_RATE * _CHANNELS * _BYTES_PER_SAMPLE  # 64000


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class AudioDevice:
    """Represents a PulseAudio/PipeWire audio device."""

    index: int
    name: str
    driver: str
    sample_spec: str
    state: str

    @property
    def is_monitor(self) -> bool:
        return self.name.endswith(".monitor")


@dataclass
class RecordingStatus:
    """Snapshot of current recording state."""

    is_alive: bool
    elapsed_seconds: float
    file_size_bytes: int
    restart_count: int
    failed: bool
    fail_reason: str | None = None
    paused: bool = False


@dataclass
class RecordingSession:
    """Manages a single recording session with auto-restart on failure.

    Recording is chunk-based: each ffmpeg invocation writes to a separate
    chunk file. On stop(), chunks are concatenated into the final output.
    """

    output_dir: Path
    output_file: Path
    mic_source: str
    monitor_source: str
    use_virtual_sink: bool = False

    # ── Internal state (not part of repr) ──
    _ffmpeg_proc: subprocess.Popen | None = field(default=None, repr=False)
    _ffmpeg_log: IO | None = field(default=None, repr=False)
    _virtual_sink_module: int | None = field(default=None, repr=False)
    _loopback_module: int | None = field(default=None, repr=False)
    _metadata: dict = field(default_factory=dict, repr=False)

    # Chunk tracking
    _chunks: list[Path] = field(default_factory=list, repr=False)
    _current_chunk: Path | None = field(default=None, repr=False)

    # Pause state
    _paused: bool = field(default=False, repr=False)

    # Watchdog state
    _watchdog_thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _restart_count: int = field(default=0, repr=False)
    _failed: bool = field(default=False, repr=False)
    _fail_reason: str | None = field(default=None, repr=False)
    _last_file_size: int = field(default=0, repr=False)
    _last_growth_time: float = field(default=0.0, repr=False)
    _actual_monitor: str = field(default="", repr=False)

    def start(self) -> None:
        """Start recording with watchdog monitoring."""
        self._actual_monitor = self.monitor_source

        if self.use_virtual_sink:
            self._actual_monitor = self._setup_virtual_sink()

        self._metadata = {
            "started_at": datetime.now().isoformat(),
            "mic_source": self.mic_source,
            "monitor_source": self._actual_monitor,
            "virtual_sink": self.use_virtual_sink,
            "output_file": str(self.output_file),
        }

        self._stop_event.clear()
        self._failed = False
        self._fail_reason = None
        self._restart_count = 0
        self._paused = False
        self._chunks = []

        # Start first chunk — this polls until ffmpeg is actually
        # writing audio data before we continue.
        self._start_ffmpeg_chunk()

        # Record when we started (for metadata); elapsed time is derived
        # from file size, not from this timestamp.
        self._start_time = time.monotonic()
        self._last_growth_time = self._start_time

        # Launch watchdog thread
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="meet-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def stop(self) -> Path:
        """Stop recording, stitch chunks, and return the output file path.

        Works from both recording and paused states.

        Returns:
            Path to the final output WAV file.
        """
        # Signal watchdog to stop
        self._stop_event.set()

        # Stop current ffmpeg process (no-op if already paused/stopped)
        if not self._paused:
            self._stop_ffmpeg()
        self._paused = False

        # Close ffmpeg log
        if self._ffmpeg_log:
            try:
                self._ffmpeg_log.close()
            except OSError:
                pass
            self._ffmpeg_log = None

        # Wait for watchdog to exit
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=5)

        if self.use_virtual_sink:
            self._teardown_virtual_sink()

        # Stitch chunks into final output
        valid_chunks = [c for c in self._chunks if c.exists() and c.stat().st_size > 0]

        if len(valid_chunks) == 0:
            # No audio captured at all
            pass
        elif len(valid_chunks) == 1:
            # Single chunk — just rename
            valid_chunks[0].rename(self.output_file)
        else:
            # Multiple chunks — concatenate with ffmpeg
            self._concat_chunks(valid_chunks)

        # Clean up any remaining chunk files
        for chunk in self._chunks:
            if chunk.exists() and chunk != self.output_file:
                try:
                    chunk.unlink()
                except OSError:
                    pass

        # Write session metadata
        self._metadata["stopped_at"] = datetime.now().isoformat()
        self._metadata["restart_count"] = self._restart_count
        self._metadata["chunk_count"] = len(valid_chunks)
        self._metadata["failed"] = self._failed
        if self._fail_reason:
            self._metadata["fail_reason"] = self._fail_reason
        self._metadata["file_exists"] = self.output_file.exists()
        if self.output_file.exists():
            self._metadata["file_size_bytes"] = self.output_file.stat().st_size

        meta_file = self.output_file.with_suffix(".session.json")
        meta_file.write_text(json.dumps(self._metadata, indent=2))

        return self.output_file

    def pause(self) -> None:
        """Pause recording by stopping the current ffmpeg chunk.

        The current chunk is finalized so no audio is lost.  Call
        :meth:`resume` to start a new chunk and continue recording.

        Raises:
            RuntimeError: If not currently recording or already paused.
        """
        with self._lock:
            if self._paused:
                raise RuntimeError("Recording is already paused")
            if self._failed:
                raise RuntimeError("Recording has failed; cannot pause")

        # Stop the current ffmpeg process (finalizes the chunk WAV)
        self._stop_ffmpeg()
        self._paused = True

    def resume(self) -> None:
        """Resume recording after a pause by starting a new chunk.

        Raises:
            RuntimeError: If not currently paused.
        """
        with self._lock:
            if not self._paused:
                raise RuntimeError("Recording is not paused")

        self._paused = False
        self._start_ffmpeg_chunk()
        self._last_growth_time = time.monotonic()

    def status(self) -> RecordingStatus:
        """Get current recording status (thread-safe).

        Elapsed time is derived from the actual WAV file size on disk,
        so it always matches the real audio duration exactly.
        """
        with self._lock:
            # Total audio bytes across all chunks (each has its own WAV header)
            total_audio_bytes = 0
            for chunk in self._chunks:
                try:
                    if chunk.exists():
                        sz = chunk.stat().st_size
                        if sz > _WAV_HEADER_BYTES:
                            total_audio_bytes += sz - _WAV_HEADER_BYTES
                except OSError:
                    pass

            total_size = total_audio_bytes + _WAV_HEADER_BYTES * len(self._chunks)
            elapsed = (
                total_audio_bytes / _BYTES_PER_SECOND if _BYTES_PER_SECOND else 0.0
            )

            is_alive = (
                self._ffmpeg_proc is not None
                and self._ffmpeg_proc.poll() is None
                and not self._failed
            )

            return RecordingStatus(
                is_alive=is_alive,
                elapsed_seconds=elapsed,
                file_size_bytes=total_size,
                restart_count=self._restart_count,
                failed=self._failed,
                fail_reason=self._fail_reason,
                paused=self._paused,
            )

    # ── ffmpeg process management ────────────────────────────────────────

    def _build_ffmpeg_cmd(self, output_path: Path) -> list[str]:
        """Build the ffmpeg command for dual-channel recording.

        Uses -use_wallclock_as_timestamps 1 on both inputs so that amerge
        syncs them by real wall-clock time instead of waiting for both sources
        to start producing samples. Without this, the PulseAudio monitor
        source typically starts ~3-4 seconds after the mic, and amerge blocks
        until then — silently losing those seconds of mic audio.

        Uses -flush_packets 1 so data is flushed to disk promptly, improving
        both the watchdog file-growth check and SIGINT graceful shutdown.
        """
        return [
            "ffmpeg",
            "-y",
            # Mic input (wall-clock timestamps to avoid amerge blocking)
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "pulse",
            "-channel_layout",
            "mono",
            "-i",
            self.mic_source,
            # System audio monitor input (wall-clock timestamps)
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "pulse",
            "-channel_layout",
            "mono",
            "-i",
            self._actual_monitor,
            # Merge into 2-channel stereo (left=mic, right=system)
            "-filter_complex",
            "[0:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[mic];"
            "[1:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[sys];"
            "[mic][sys]amerge=inputs=2[out]",
            "-map",
            "[out]",
            "-ac",
            "2",
            "-ar",
            "16000",
            # Output as WAV — flush packets for reliable watchdog + clean shutdown
            "-flush_packets",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

    def _start_ffmpeg_chunk(self) -> None:
        """Start ffmpeg writing to a new chunk file."""
        chunk_idx = len(self._chunks)
        stem = self.output_file.stem
        if chunk_idx == 0:
            # First chunk — use output name directly (will rename/concat on stop)
            chunk_path = self.output_dir / f"{stem}.chunk-{chunk_idx:03d}.wav"
        else:
            chunk_path = self.output_dir / f"{stem}.chunk-{chunk_idx:03d}.wav"

        self._current_chunk = chunk_path
        self._chunks.append(chunk_path)

        # Open log file for ffmpeg stderr (append mode — shared across restarts)
        log_path = self.output_file.with_suffix(".ffmpeg.log")
        if self._ffmpeg_log is None or self._ffmpeg_log.closed:
            self._ffmpeg_log = open(log_path, "a")

        self._ffmpeg_log.write(
            f"\n--- Chunk {chunk_idx} started at {datetime.now().isoformat()} ---\n"
        )
        self._ffmpeg_log.flush()

        cmd = self._build_ffmpeg_cmd(chunk_path)

        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._ffmpeg_log,
        )

        # Wait for ffmpeg to start producing audio data.
        # Instead of a fixed sleep, poll the output file until it has
        # actual audio data (size > WAV header ~44 bytes).  This way
        # our elapsed timer starts from when recording truly begins.
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            # Check if ffmpeg died during startup
            if self._ffmpeg_proc.poll() is not None:
                raise RuntimeError(
                    f"ffmpeg failed to start (exit code {self._ffmpeg_proc.returncode}). "
                    f"Check log: {log_path}"
                )
            # Check if output file has audio data (WAV header is ~44 bytes)
            try:
                if chunk_path.exists() and chunk_path.stat().st_size > 1024:
                    break
            except OSError:
                pass
            time.sleep(_STARTUP_POLL_INTERVAL)
        else:
            # Timed out — check if ffmpeg is still alive at least
            if self._ffmpeg_proc.poll() is not None:
                raise RuntimeError(
                    f"ffmpeg failed to start (exit code {self._ffmpeg_proc.returncode}). "
                    f"Check log: {log_path}"
                )
            # ffmpeg is alive but no data yet — continue anyway,
            # the watchdog will handle stalls

        # Reset growth tracking for new chunk
        self._last_file_size = 0
        self._last_growth_time = time.monotonic()

    def _stop_ffmpeg(self) -> None:
        """Gracefully stop the current ffmpeg process.

        Sends 'q' to ffmpeg's stdin for a clean shutdown that flushes all
        buffered audio and writes a proper WAV trailer.  Falls back to
        SIGINT → SIGTERM → SIGKILL if ffmpeg doesn't exit in time.
        """
        proc = self._ffmpeg_proc
        if proc is None:
            return

        if proc.poll() is not None:
            # Already exited
            self._ffmpeg_proc = None
            return

        # Step 1: Send 'q' command via stdin for graceful flush.
        # This tells ffmpeg to finish processing buffered data, write the
        # WAV trailer, and exit — no audio samples lost.
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write(b"q")
                proc.stdin.flush()
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        try:
            proc.wait(timeout=5)
            self._ffmpeg_proc = None
            return
        except subprocess.TimeoutExpired:
            pass

        # Step 2: Fall back to SIGINT
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
            self._ffmpeg_proc = None
            return
        except subprocess.TimeoutExpired:
            pass

        # Step 3: SIGTERM
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        self._ffmpeg_proc = None

    def _attempt_restart(self, reason: str) -> bool:
        """Try to restart ffmpeg to a new chunk. Returns True if successful."""
        if self._restart_count >= _MAX_RESTART_ATTEMPTS:
            self._failed = True
            self._fail_reason = f"Max restart attempts ({_MAX_RESTART_ATTEMPTS}) exceeded. Last: {reason}"
            return False

        # Stop current (possibly dead) process
        self._stop_ffmpeg()

        self._restart_count += 1

        if self._ffmpeg_log and not self._ffmpeg_log.closed:
            self._ffmpeg_log.write(
                f"\n--- RESTART #{self._restart_count} at {datetime.now().isoformat()} "
                f"reason: {reason} ---\n"
            )
            self._ffmpeg_log.flush()

        try:
            self._start_ffmpeg_chunk()
            return True
        except Exception as e:
            self._failed = True
            self._fail_reason = f"Restart failed: {e}"
            return False

    # ── Watchdog ─────────────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Background thread that monitors ffmpeg health."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_WATCHDOG_INTERVAL)
            if self._stop_event.is_set():
                break

            with self._lock:
                if self._failed:
                    break

                # Skip health checks while paused (no ffmpeg process running)
                if self._paused:
                    continue

                proc = self._ffmpeg_proc
                if proc is None:
                    continue

                # Check 1: Is ffmpeg still running?
                exit_code = proc.poll()
                if exit_code is not None:
                    reason = f"ffmpeg exited with code {exit_code}"
                    self._attempt_restart(reason)
                    continue

                # Check 2: Is the file still growing?
                chunk = self._current_chunk
                if chunk and chunk.exists():
                    try:
                        current_size = chunk.stat().st_size
                    except OSError:
                        continue

                    if current_size > self._last_file_size:
                        self._last_file_size = current_size
                        self._last_growth_time = time.monotonic()
                    else:
                        stall_duration = time.monotonic() - self._last_growth_time
                        if stall_duration > _STALL_TIMEOUT:
                            reason = f"Output file stalled for {stall_duration:.0f}s"
                            self._attempt_restart(reason)

    # ── Chunk stitching ──────────────────────────────────────────────────

    def _concat_chunks(self, chunks: list[Path]) -> None:
        """Concatenate multiple WAV chunks into the final output file."""
        # Build ffmpeg concat demuxer input file
        concat_list = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            dir=self.output_dir,
        )
        try:
            for chunk in chunks:
                # ffmpeg concat requires single-quoted paths with escaped quotes
                safe_path = str(chunk).replace("'", "'\\''")
                concat_list.write(f"file '{safe_path}'\n")
            concat_list.close()

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list.name,
                "-c",
                "copy",
                str(self.output_file),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Fallback: just use the largest chunk
                largest = max(chunks, key=lambda c: c.stat().st_size)
                largest.rename(self.output_file)
        finally:
            try:
                Path(concat_list.name).unlink()
            except OSError:
                pass

    # ── Virtual sink management ──────────────────────────────────────────

    def _setup_virtual_sink(self) -> str:
        """Create a virtual null sink for isolated meeting audio capture."""
        sink_name = "meet_capture"

        result = subprocess.run(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={sink_name}",
                f"sink_properties=device.description=Meet-Capture",
                "rate=16000",
                "channels=1",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create virtual sink: {result.stderr}")
        self._virtual_sink_module = int(result.stdout.strip())

        result = subprocess.run(
            [
                "pactl",
                "load-module",
                "module-loopback",
                f"source={sink_name}.monitor",
                "sink=@DEFAULT_SINK@",
                "latency_msec=1",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            self._loopback_module = int(result.stdout.strip())

        return f"{sink_name}.monitor"

    def _teardown_virtual_sink(self) -> None:
        """Remove virtual sink and loopback modules."""
        if self._loopback_module is not None:
            subprocess.run(
                ["pactl", "unload-module", str(self._loopback_module)],
                capture_output=True,
            )
            self._loopback_module = None

        if self._virtual_sink_module is not None:
            subprocess.run(
                ["pactl", "unload-module", str(self._virtual_sink_module)],
                capture_output=True,
            )
            self._virtual_sink_module = None


# ─── Module-level helpers ────────────────────────────────────────────────────


def list_sources() -> list[AudioDevice]:
    """List all PulseAudio/PipeWire audio sources."""
    result = subprocess.run(
        ["pactl", "list", "short", "sources"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list sources: {result.stderr}")

    devices = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            devices.append(
                AudioDevice(
                    index=int(parts[0]),
                    name=parts[1],
                    driver=parts[2],
                    sample_spec=parts[3],
                    state=parts[4],
                )
            )
    return devices


def get_default_sink() -> str:
    """Get the name of the default audio output sink."""
    result = subprocess.run(
        ["pactl", "get-default-sink"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get default sink: {result.stderr}")
    return result.stdout.strip()


def get_default_source() -> str:
    """Get the name of the default audio input source (mic)."""
    result = subprocess.run(
        ["pactl", "get-default-source"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get default source: {result.stderr}")
    return result.stdout.strip()


def get_monitor_source() -> str:
    """Get the monitor source for the default sink (captures system audio)."""
    return f"{get_default_sink()}.monitor"


def create_session(
    output_dir: str | Path | None = None,
    filename: str | None = None,
    mic: str | None = None,
    monitor: str | None = None,
    virtual_sink: bool = False,
) -> RecordingSession:
    """Create a new recording session.

    Args:
        output_dir: Directory to save recordings. Defaults to ~/meet-recordings.
        filename: Output filename. Defaults to timestamped name.
        mic: Mic source name. Defaults to system default.
        monitor: Monitor source name. Defaults to default sink monitor.
        virtual_sink: Whether to create an isolated virtual sink.

    Returns:
        A RecordingSession ready to start.
    """
    if output_dir is None:
        output_dir = Path.home() / "meet-recordings"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # Auto-generated name: create a per-session subdirectory so all
        # session artefacts (wav, logs, transcripts) live together.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_name = f"meeting-{timestamp}"
        session_dir = output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{session_name}.wav"
        output_dir = session_dir
    # When filename is explicitly provided via -f, keep flat layout.

    mic_source = mic or get_default_source()
    monitor_source = monitor or get_monitor_source()

    return RecordingSession(
        output_dir=output_dir,
        output_file=output_dir / filename,
        mic_source=mic_source,
        monitor_source=monitor_source,
        use_virtual_sink=virtual_sink,
    )


def check_prerequisites() -> list[str]:
    """Check that required system tools are available. Returns list of issues."""
    issues = []

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        issues.append("ffmpeg is not installed. Install with: sudo apt install ffmpeg")

    try:
        subprocess.run(["pactl", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        issues.append("pactl is not available. Install PulseAudio or PipeWire.")

    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            issues.append("PulseAudio/PipeWire server is not running.")
        elif not result.stdout.strip():
            issues.append("No audio sources detected.")
    except Exception as e:
        issues.append(f"Cannot communicate with audio server: {e}")

    return issues
