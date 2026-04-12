"""CLI entrypoint for the meet tool.

Commands:
    meet record          - Record meeting audio (Ctrl+C to stop)
    meet transcribe FILE - Transcribe a recorded audio file
    meet run             - Record then transcribe when stopped
    meet gui             - Launch GUI widget for recording
    meet devices         - List available audio devices
    meet check           - Check system prerequisites
    meet download        - Download alignment models
    meet translate       - Translate a session's transcript
    meet label           - Assign real names to speakers in a session
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click

from meet.capture import DRAIN_SECONDS
from meet.utils import fmt_elapsed, fmt_size


def _drain_countdown(session, seconds: int = DRAIN_SECONDS) -> None:
    """Keep recording for *seconds* more to let ffmpeg's delayed pipeline flush.

    During the countdown:
    - Additional Ctrl+C signals are ignored (SIGINT → SIG_IGN)
    - A single status line updates in-place each second showing remaining time,
      elapsed recording time, and file size
    After the countdown, default SIGINT handling is restored.
    """
    # Ignore further Ctrl+C during the drain window
    prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        for remaining in range(seconds, 0, -1):
            status = session.status()
            elapsed = fmt_elapsed(status.elapsed_seconds)
            size = fmt_size(status.file_size_bytes)
            click.echo(
                f"\r\033[K\033[1;33m⏳ Flushing audio buffer... {remaining}s\033[0m"
                f"  {elapsed}  {size}",
                nl=False,
            )
            time.sleep(1)
        # Final line
        status = session.status()
        elapsed = fmt_elapsed(status.elapsed_seconds)
        size = fmt_size(status.file_size_bytes)
        click.echo(f"\r\033[K\033[1;32m✔ Buffer flushed\033[0m  {elapsed}  {size}")
    finally:
        # Restore previous SIGINT handler
        signal.signal(signal.SIGINT, prev_handler)


def _generate_summary(
    transcript, out_dir, basename, summary_model, files, summary_backend=None
):
    """Generate an AI meeting summary. Returns MeetingSummary or None.

    Supports multiple backends (claudemax, openrouter, ollama) via SummaryConfig.
    The fallback chain is handled inside summarize() — callers should not
    gate on is_backend_available().
    """
    from meet.summarize import summarize as do_summarize, SummaryConfig

    config_kwargs = {}
    if summary_backend:
        config_kwargs["backend"] = summary_backend
    if summary_model:
        config_kwargs["model"] = summary_model
    summary_config = SummaryConfig(**config_kwargs)

    def _cli_progress(msg: str) -> None:
        click.echo(f"  {msg}")

    click.echo(
        f"Generating meeting summary ({summary_config.model} via {summary_config.backend})..."
    )
    try:
        result = do_summarize(
            transcript.to_text(),
            summary_config,
            language=transcript.language,
            progress_callback=_cli_progress,
        )
        path = result.save(out_dir, basename)
        files["summary"] = path
        click.echo(f"  Summary generated in {result.elapsed_seconds:.1f}s")
        return result
    except Exception as exc:
        click.echo(f"  Summary failed: {exc}", err=True)
        return None


def _generate_pdf(transcript, out_dir, basename, summary_result, files):
    """Generate a PDF transcript with optional summary."""
    from meet.pdf import generate_pdf

    pdf_path = out_dir / f"{basename}.pdf"
    try:
        generate_pdf(
            transcript,
            pdf_path,
            summary=summary_result,
            language=getattr(transcript, "language", "en"),
        )
        files["pdf"] = pdf_path
    except Exception as exc:
        click.echo(f"  PDF generation failed: {exc}", err=True)


def _recording_loop(session) -> None:
    """Run the live recording status display loop.

    Shows an updating single-line status indicator. Replaces signal.pause()
    with an active monitoring loop that displays:
        REC  00:07:23  14.2 MB  Ctrl+C to stop

    Immediately alerts if recording fails or restarts.
    """
    last_restart_count = 0
    warned_failed = False

    try:
        while True:
            status = session.status()

            elapsed = fmt_elapsed(status.elapsed_seconds)
            size = fmt_size(status.file_size_bytes)

            if status.failed and not warned_failed:
                # Recording failed and could not restart
                reason = status.fail_reason or "unknown error"
                click.echo(
                    f"\r\033[K\033[1;31m✖ RECORDING FAILED\033[0m  {elapsed}  {size}  — {reason}"
                )
                click.echo(f"  Press Ctrl+C to transcribe what was captured.")
                warned_failed = True
            elif status.restart_count > last_restart_count:
                # ffmpeg was restarted — show brief warning
                last_restart_count = status.restart_count
                click.echo(
                    f"\r\033[K\033[1;33m⚠ Recording restarted\033[0m (attempt {status.restart_count})  {elapsed}  {size}"
                )
            elif not warned_failed:
                # Normal status line — overwrite in place
                if status.is_alive:
                    line = f"\r\033[K\033[1;32m● REC\033[0m  {elapsed}  {size}  Ctrl+C to stop"
                else:
                    line = f"\r\033[K\033[1;33m● REC (starting...)\033[0m  {elapsed}  {size}"
                click.echo(line, nl=False)

            time.sleep(1)
    except KeyboardInterrupt:
        # Clear the status line before returning
        click.echo(f"\r\033[K", nl=False)
        raise


@click.group()
@click.version_option(version="0.3.3")
def main():
    """Local meeting transcription with speaker diarization."""
    pass


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory to save recordings (default: ~/meet-recordings)",
)
@click.option(
    "--filename",
    "-f",
    type=str,
    default=None,
    help="Output filename (default: meeting-YYYYMMDD-HHMMSS.wav)",
)
@click.option(
    "--mic", type=str, default=None, help="Mic source name (default: system default)"
)
@click.option(
    "--monitor",
    type=str,
    default=None,
    help="Monitor source name (default: default sink monitor)",
)
@click.option(
    "--virtual-sink",
    is_flag=True,
    default=False,
    help="Use a virtual sink for isolated capture",
)
def record(output_dir, filename, mic, monitor, virtual_sink):
    """Record meeting audio. Press Ctrl+C to stop."""
    from meet.capture import create_session, check_prerequisites

    issues = check_prerequisites()
    if issues:
        click.echo("Prerequisites check failed:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)

    session = create_session(
        output_dir=output_dir,
        filename=filename,
        mic=mic,
        monitor=monitor,
        virtual_sink=virtual_sink,
    )

    click.echo(f"Recording to: {session.output_file}")
    click.echo(f"  Mic source:     {session.mic_source}")
    click.echo(f"  Monitor source: {session.monitor_source}")
    click.echo(f"  Virtual sink:   {session.use_virtual_sink}")
    if virtual_sink:
        click.echo(
            f"  NOTE: Route your meeting app's audio to 'Meet-Capture' in pavucontrol"
        )
    click.echo()

    session.start()

    try:
        _recording_loop(session)
    except KeyboardInterrupt:
        _drain_countdown(session)
        click.echo("Stopping recording...")
        output = session.stop()
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            click.echo(f"Saved: {output} ({size_mb:.1f} MB)")
            click.echo(f"Transcribe with: meet transcribe {output}")
            status = session.status()
            if status.restart_count > 0:
                click.echo(
                    f"  Note: recording restarted {status.restart_count} time(s) — check .ffmpeg.log if audio seems off"
                )
        else:
            click.echo("Warning: output file was not created", err=True)
        sys.exit(0)


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    type=str,
    default="large-v3-turbo",
    help="Whisper model (default: large-v3-turbo). Also: base, medium, large-v2, or a local path.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to run on (default: cuda)",
)
@click.option(
    "--compute-type",
    type=str,
    default="float16",
    help="Compute type: float16, int8 (default: float16)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=16,
    help="Batch size for transcription (default: 16)",
)
@click.option(
    "--language",
    "-l",
    type=str,
    default="auto",
    help="Language code or 'auto' to detect (default: auto). Examples: en, de, fr, es, tr, fa",
)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace token for diarization (or set HF_TOKEN env var)",
)
@click.option(
    "--min-speakers", type=int, default=None, help="Minimum number of speakers"
)
@click.option(
    "--max-speakers", type=int, default=None, help="Maximum number of speakers"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for transcripts (default: same as audio file)",
)
@click.option(
    "--no-diarize", is_flag=True, default=False, help="Skip speaker diarization"
)
@click.option(
    "--summarize/--no-summarize",
    default=True,
    help="Generate AI meeting summary (default: on)",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["ollama", "openrouter", "claudemax", "openai"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: ollama, or MEETSCRIBE_SUMMARY_BACKEND env var)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MEETSCRIBE_SUMMARY_MODEL env var)",
)
@click.option(
    "--skip-alignment",
    is_flag=True,
    default=False,
    help="Skip word-level alignment (useful if alignment model is unavailable)",
)
@click.option(
    "--mixdown",
    type=click.Choice(["mono", "dual"]),
    default="mono",
    help="Stereo mixdown mode: mono=mic channel only, dual=transcribe both channels separately (default: mono)",
)
def transcribe(
    audio_file,
    model,
    device,
    compute_type,
    batch_size,
    language,
    hf_token,
    min_speakers,
    max_speakers,
    output_dir,
    no_diarize,
    summarize,
    summary_backend,
    summary_model,
    skip_alignment,
    mixdown,
):
    """Transcribe a recorded audio file with speaker diarization."""
    from meet.transcribe import (
        TranscriptionConfig,
        transcribe as do_transcribe,
        AlignmentModelMissing,
        ensure_gpu_available,
    )

    audio_path = Path(audio_file)

    # If user passed a session directory, find the audio file inside it.
    if audio_path.is_dir():
        wavs = sorted(audio_path.glob("*.wav"))
        oggs = sorted(audio_path.glob("*.ogg"))
        audio_files = wavs or oggs
        if not audio_files:
            click.echo(
                f"Error: no audio file (.wav/.ogg) found in {audio_path}", err=True
            )
            raise SystemExit(1)
        audio_path = audio_files[0]
        click.echo(f"  Resolved to: {audio_path}")

    config = TranscriptionConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token if not no_diarize else None,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        skip_alignment=skip_alignment,
        mixdown=mixdown,
    )

    if not no_diarize and not config.hf_token and mixdown != "dual":
        click.echo("Warning: No HF_TOKEN found. Diarization will be skipped.", err=True)
        click.echo("  Set HF_TOKEN env var or pass --hf-token", err=True)
        click.echo("  Get a token at: https://huggingface.co/settings/tokens", err=True)
        click.echo(
            "  Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-community-1",
            err=True,
        )
        click.echo()

    click.echo(f"Transcribing: {audio_path}")
    click.echo(f"  Model:    {config.model} ({config.compute_type})")
    click.echo(f"  Device:   {config.device}")
    click.echo(f"  Language: {config.language}")
    click.echo(f"  Diarize:  {bool(config.hf_token)}")
    click.echo()

    # Free GPU memory from Ollama before transcription
    ensure_gpu_available()

    try:
        transcript = do_transcribe(audio_path, config)
    except AlignmentModelMissing as exc:
        click.echo()
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        click.echo(err=True)
        click.echo(f"  To download it, run:", err=True)
        click.echo(f"    meet download {exc.lang}", err=True)
        click.echo(err=True)
        click.echo(
            f"  Or skip alignment (fewer segments, no word-level timestamps):", err=True
        )
        click.echo(
            f"    meet transcribe {audio_file} --language {exc.lang} --skip-alignment",
            err=True,
        )
        raise SystemExit(1)

    # Determine output directory
    if output_dir is None:
        out_dir = audio_path.parent
    else:
        out_dir = Path(output_dir)

    files = transcript.save(out_dir, basename=audio_path.stem)

    # ── Summary + PDF ──
    summary_result = None
    if summarize:
        summary_result = _generate_summary(
            transcript,
            out_dir,
            audio_path.stem,
            summary_model,
            files,
            summary_backend=summary_backend,
        )

    _generate_pdf(transcript, out_dir, audio_path.stem, summary_result, files)

    click.echo()
    click.echo(f"Transcription complete!")
    click.echo(f"  Duration: {transcript.duration:.0f}s" if transcript.duration else "")
    click.echo(f"  Speakers: {len(transcript.speakers)}")
    click.echo(f"  Segments: {len(transcript.segments)}")
    click.echo()
    click.echo("Output files:")
    for fmt, path in files.items():
        click.echo(f"  {fmt}: {path}")

    click.echo()
    click.echo("--- Transcript Preview ---")
    click.echo()
    # Show first 20 lines
    lines = transcript.to_text().split("\n")
    for line in lines[:20]:
        click.echo(line)
    if len(lines) > 20:
        click.echo(f"  ... ({len(lines) - 20} more lines, see {files['text']})")


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory for recordings and transcripts",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="large-v3-turbo",
    help="Whisper model (default: large-v3-turbo)",
)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option("--compute-type", type=str, default="float16")
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--language", "-l", type=str, default="auto")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--virtual-sink", is_flag=True, default=False)
@click.option(
    "--summarize/--no-summarize",
    default=True,
    help="Generate AI meeting summary (default: on)",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["ollama", "openrouter", "claudemax", "openai"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: ollama, or MEETSCRIBE_SUMMARY_BACKEND env var)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MEETSCRIBE_SUMMARY_MODEL env var)",
)
@click.option(
    "--skip-alignment",
    is_flag=True,
    default=False,
    help="Skip word-level alignment (useful if alignment model is unavailable)",
)
@click.option(
    "--mixdown",
    type=click.Choice(["mono", "dual"]),
    default="mono",
    help="Stereo mixdown mode: mono=mic channel only, dual=transcribe both channels separately (default: mono)",
)
def run(
    output_dir,
    model,
    device,
    compute_type,
    batch_size,
    language,
    hf_token,
    min_speakers,
    max_speakers,
    virtual_sink,
    summarize,
    summary_backend,
    summary_model,
    skip_alignment,
    mixdown,
):
    """Record a meeting, then transcribe when stopped with Ctrl+C."""
    from meet.capture import create_session, check_prerequisites
    from meet.transcribe import (
        TranscriptionConfig,
        transcribe as do_transcribe,
        AlignmentModelMissing,
        ensure_gpu_available,
    )

    issues = check_prerequisites()
    if issues:
        click.echo("Prerequisites check failed:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)

    session = create_session(
        output_dir=output_dir,
        virtual_sink=virtual_sink,
    )

    config = TranscriptionConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        skip_alignment=skip_alignment,
        mixdown=mixdown,
    )

    if not config.hf_token and mixdown != "dual":
        click.echo("Warning: No HF_TOKEN found. Diarization will be skipped.", err=True)
        click.echo("  Set HF_TOKEN env var or pass --hf-token", err=True)
        click.echo()

    click.echo(f"Recording to: {session.output_file}")
    click.echo(f"  Mic:     {session.mic_source}")
    click.echo(f"  Monitor: {session.monitor_source}")
    click.echo(f"  Diarize: {bool(config.hf_token)}")
    click.echo()

    session.start()

    try:
        _recording_loop(session)
    except KeyboardInterrupt:
        _drain_countdown(session)
        click.echo("Stopping recording...")
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            click.echo("Error: No audio was recorded.", err=True)
            sys.exit(1)

        size_mb = output.stat().st_size / (1024 * 1024)
        rec_status = session.status()
        click.echo(f"Saved recording: {output} ({size_mb:.1f} MB)")
        if rec_status.restart_count > 0:
            click.echo(
                f"  Note: recording restarted {rec_status.restart_count} time(s)"
            )
        click.echo()
        click.echo("Starting transcription...")
        click.echo()

        # Free GPU memory from Ollama before transcription
        ensure_gpu_available()

        try:
            transcript = do_transcribe(output, config)
        except AlignmentModelMissing as exc:
            click.echo()
            click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
            click.echo(err=True)
            click.echo(f"  To download it, run:", err=True)
            click.echo(f"    meet download {exc.lang}", err=True)
            click.echo(err=True)
            click.echo(f"  Or re-run with --skip-alignment:", err=True)
            click.echo(
                f"    meet transcribe {output} --language {exc.lang} --skip-alignment",
                err=True,
            )
            click.echo(err=True)
            click.echo(f"  Your recording is saved at: {output}", err=True)
            sys.exit(1)
        files = transcript.save(output.parent, basename=output.stem)

        # ── Summary + PDF ──
        summary_result = None
        if summarize:
            summary_result = _generate_summary(
                transcript,
                output.parent,
                output.stem,
                summary_model,
                files,
                summary_backend=summary_backend,
            )

        _generate_pdf(transcript, output.parent, output.stem, summary_result, files)

        click.echo()
        click.echo(f"Done!")
        click.echo(
            f"  Duration: {transcript.duration:.0f}s" if transcript.duration else ""
        )
        click.echo(f"  Speakers: {len(transcript.speakers)}")
        click.echo(f"  Segments: {len(transcript.segments)}")
        click.echo()
        click.echo("Output files:")
        for fmt, path in files.items():
            click.echo(f"  {fmt}: {path}")

        click.echo()
        click.echo("--- Transcript ---")
        click.echo()
        click.echo(transcript.to_text())
        sys.exit(0)


@main.command()
def devices():
    """List available audio devices."""
    from meet.capture import list_sources, get_default_sink, get_default_source

    try:
        default_source = get_default_source()
        default_sink = get_default_sink()
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Default mic (source):  {default_source}")
    click.echo(f"Default output (sink): {default_sink}")
    click.echo(f"Monitor source:        {default_sink}.monitor")
    click.echo()

    sources = list_sources()

    click.echo("All sources:")
    click.echo(f"  {'IDX':<5} {'STATE':<12} {'NAME'}")
    click.echo(f"  {'---':<5} {'-----':<12} {'----'}")
    for src in sources:
        marker = ""
        if src.name == default_source:
            marker = " <-- default mic"
        elif src.is_monitor and src.name == f"{default_sink}.monitor":
            marker = " <-- default monitor"
        click.echo(f"  {src.index:<5} {src.state:<12} {src.name}{marker}")


@main.command()
def check():
    """Check system prerequisites."""
    from meet.capture import check_prerequisites

    click.echo("Checking prerequisites...")
    click.echo()

    issues = check_prerequisites()
    if issues:
        click.echo("Issues found:")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)
    else:
        click.echo("  ffmpeg:           OK")
        click.echo("  PulseAudio/PipeWire: OK")

    # Check Python packages
    click.echo()
    try:
        import whisperx

        click.echo(f"  whisperx:         OK")
    except ImportError:
        click.echo(f"  whisperx:         NOT INSTALLED (pip install whisperx)")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"  CUDA:             OK ({gpu_name})")
        else:
            click.echo(f"  CUDA:             Not available (will use CPU)")
    except ImportError:
        click.echo(f"  torch:            NOT INSTALLED")

    # Check HF token
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    if hf_token:
        click.echo(f"  HF_TOKEN:         OK")
    else:
        click.echo(f"  HF_TOKEN:         NOT SET (diarization won't work)")
        click.echo(f"                    Set with: export HF_TOKEN=hf_...")
        click.echo(f"                    Or run: huggingface-cli login")

    click.echo()
    if not issues:
        click.echo("All prerequisites met!")


@main.command()
@click.argument("languages", nargs=-1)
@click.option(
    "--all",
    "download_all",
    is_flag=True,
    default=False,
    help="Download alignment models for all supported languages",
)
def download(languages, download_all):
    """Download alignment models for specified languages.

    \b
    Examples:
        meet download de tr fa    # download German, Turkish, Farsi
        meet download --all       # download all supported models
    """
    from meet.transcribe import (
        get_supported_alignment_languages,
        download_alignment_model,
        check_alignment_model_cached,
    )

    info = get_supported_alignment_languages()

    if download_all:
        languages = tuple(info.keys())
    elif not languages:
        # No arguments — show status of all models
        click.echo("Alignment model status:")
        click.echo()
        click.echo(f"  {'Lang':<6} {'Name':<10} {'Model':<50} {'Size':<10} {'Status'}")
        click.echo(f"  {'----':<6} {'----':<10} {'-----':<50} {'----':<10} {'------'}")
        for lang, details in info.items():
            status = (
                click.style("cached", fg="green")
                if details["cached"]
                else click.style("missing", fg="red")
            )
            click.echo(
                f"  {lang:<6} {details['name']:<10} {details['model']:<50} {details['size']:<10} {status}"
            )
        click.echo()
        click.echo("To download: meet download <lang> [<lang> ...]")
        click.echo("To download all: meet download --all")
        return

    # Validate languages
    invalid = [l for l in languages if l not in info]
    if invalid:
        supported = ", ".join(sorted(info.keys()))
        click.echo(f"Error: unsupported language(s): {', '.join(invalid)}", err=True)
        click.echo(f"  Supported: {supported}", err=True)
        raise SystemExit(1)

    # Download each model
    for lang in languages:
        details = info[lang]
        if details["cached"]:
            click.echo(f"  {details['name']} ({lang}): already cached, skipping.")
            continue
        try:
            download_alignment_model(
                lang, progress_callback=lambda msg: click.echo(f"  {msg}")
            )
        except Exception as exc:
            click.echo(
                f"  Error downloading {details['name']} ({lang}): {exc}", err=True
            )


@main.command()
@click.argument("session_dir", type=click.Path(exists=True))
@click.option(
    "--to",
    "target_lang",
    type=str,
    default="en",
    help="Target language for translation (default: en)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Ollama model to use (default: qwen3.5:9b)",
)
def translate(session_dir, target_lang, summary_model):
    """Translate a session's transcript to another language.

    \b
    SESSION_DIR is the path to a meet recording session directory.

    The translated transcript is saved as <basename>.translation.<lang>.txt
    in the same session directory.

    \b
    Examples:
        meet translate ~/meet-recordings/meeting-20260313-231509
        meet translate ~/meet-recordings/meeting-20260313-231509 --to de
    """
    import requests as req

    session_path = Path(session_dir)
    if not session_path.is_dir():
        click.echo(f"Error: {session_path} is not a directory", err=True)
        raise SystemExit(1)

    # Find the .txt transcript file
    txt_files = sorted(session_path.glob("*.txt"))
    # Exclude any existing translation files
    txt_files = [f for f in txt_files if ".translation." not in f.name]
    if not txt_files:
        click.echo(f"Error: no .txt transcript found in {session_path}", err=True)
        raise SystemExit(1)

    txt_file = txt_files[0]
    transcript_text = txt_file.read_text(encoding="utf-8").strip()
    if not transcript_text:
        click.echo(f"Error: transcript file is empty: {txt_file}", err=True)
        raise SystemExit(1)

    basename = txt_file.stem

    from meet.summarize import OLLAMA_BASE_URL, DEFAULT_MODEL, is_ollama_available

    ollama_url = OLLAMA_BASE_URL
    model_name = summary_model or DEFAULT_MODEL

    if not is_ollama_available(ollama_url):
        click.echo("Error: Ollama is not running. Start with: ollama serve", err=True)
        raise SystemExit(1)

    from meet.languages import LANG_NAMES

    target_name = LANG_NAMES.get(target_lang, target_lang)

    click.echo(f"Translating: {txt_file}")
    click.echo(f"  Target language: {target_name}")
    click.echo(f"  Model: {model_name}")
    click.echo()

    # Free GPU memory from Ollama models that might be loaded
    from meet.transcribe import ensure_gpu_available

    ensure_gpu_available()

    import time as _time

    t0 = _time.time()

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. Translate the following "
                    f"meeting transcript to {target_name}. "
                    f"Preserve the exact formatting: keep the timestamp markers "
                    f"like [HH:MM:SS --> HH:MM:SS] and speaker labels (YOU, REMOTE, etc.) "
                    f"unchanged. Only translate the spoken text. "
                    f"Be accurate and natural — do not add or remove information."
                ),
            },
            {
                "role": "user",
                "content": f"Translate this transcript to {target_name}:\n\n{transcript_text}",
            },
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    }

    try:
        resp = req.post(f"{ollama_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
    except req.Timeout:
        click.echo("Error: Ollama timed out. Try a smaller model.", err=True)
        raise SystemExit(1)
    except req.HTTPError as e:
        click.echo(f"Error: Ollama API error: {e}", err=True)
        raise SystemExit(1)

    elapsed = _time.time() - t0
    data = resp.json()
    translated = data.get("message", {}).get("content", "").strip()

    if not translated:
        click.echo("Error: Ollama returned an empty translation.", err=True)
        raise SystemExit(1)

    # Save translation
    out_path = session_path / f"{basename}.translation.{target_lang}.txt"
    out_path.write_text(translated, encoding="utf-8")

    click.echo(f"Translation complete in {elapsed:.1f}s")
    click.echo(f"  Saved to: {out_path}")
    click.echo()
    click.echo("--- Translation ---")
    click.echo()
    click.echo(translated)


@main.command()
@click.argument("session_dir", type=click.Path(exists=True))
@click.option(
    "--no-audio",
    is_flag=True,
    default=False,
    help="Skip audio playback (just show text samples)",
)
@click.option(
    "--no-summary",
    is_flag=True,
    default=False,
    help="Skip summary regeneration (use find-and-replace on existing summary)",
)
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Auto-label using voice profiles. Confident matches are applied "
    "without prompting; unrecognized speakers are prompted interactively.",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["ollama", "openrouter", "claudemax", "openai"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: ollama, or MEETSCRIBE_SUMMARY_BACKEND env var)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MEETSCRIBE_SUMMARY_MODEL env var)",
)
def label(session_dir, no_audio, no_summary, auto, summary_backend, summary_model):
    """Assign real names to speakers in a transcribed session.

    \b
    SESSION_DIR is the path to a meet recording session directory.

    For each speaker detected in the transcript, plays a short audio clip
    (from the appropriate channel) and prompts you to enter a name.
    Press Enter to keep the current label unchanged.

    With --auto, speaker voice profiles are used to automatically identify
    known speakers. Confident matches are applied without prompting.
    Unrecognized speakers are still prompted interactively.

    After labeling, all outputs (txt, srt, json, summary, pdf) are
    regenerated with the new speaker names.

    \b
    Examples:
        meet label ~/meet-recordings/meeting-20260313-214133
        meet label ~/meet-recordings/meeting-20260313-214133 --no-audio
        meet label ~/meet-recordings/meeting-20260313-214133 --auto
        meet label ~/meet-recordings/meeting-20260313-214133 --auto --no-summary
    """
    from meet.label import (
        get_speakers,
        extract_speaker_clip,
        play_clip,
        apply_labels,
        _find_session_files,
        _load_transcript,
        _detect_speaker_channels,
    )

    session_path = Path(session_dir)
    files = _find_session_files(session_path)

    if "json" not in files:
        click.echo(f"Error: no transcript JSON found in {session_path}", err=True)
        click.echo("  Run 'meet transcribe' on this session first.", err=True)
        raise SystemExit(1)

    # Get speaker info
    try:
        speakers = get_speakers(session_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)

    if not speakers:
        click.echo("No speakers found in this session.")
        return

    if len(speakers) == 1:
        click.echo(f"Only one speaker found: {speakers[0].id}")
        click.echo("You can still assign a name if you like.")
        click.echo()

    click.echo(f"Session: {session_path.name}")
    click.echo(f"Speakers found: {len(speakers)}")
    click.echo()

    # ── Voice profile auto-identification ──
    auto_matches: dict = {}
    wav_path = files.get("wav")
    channel_map: dict[str, str] = {}

    if auto and wav_path and wav_path.exists():
        click.echo("Running voice identification against speaker profiles...")
        transcript = _load_transcript(files["json"])
        channel_map = _detect_speaker_channels(
            wav_path,
            transcript.segments,
            transcript.speakers,
        )

        try:
            from meet.voiceprint import identify_speakers, load_profiles

            profiles = load_profiles()
            if not profiles:
                click.echo("  No speaker profiles found. Run 'meet enroll' first.")
                click.echo("  Falling back to interactive labeling.")
                click.echo()
            else:
                click.echo(f"  {len(profiles)} voice profiles loaded.")
                auto_matches = identify_speakers(
                    wav_path,
                    transcript.segments,
                    transcript.speakers,
                    channel_map,
                )
                if auto_matches:
                    click.echo(
                        f"  Identified {len(auto_matches)}/{len(speakers)} speakers:"
                    )
                    for spk_id, match in sorted(auto_matches.items()):
                        click.echo(
                            f"    {spk_id} -> {click.style(match.name, fg='green')}"
                            f"  (confidence: {match.confidence:.2f})"
                        )
                else:
                    click.echo("  No confident matches found.")
                click.echo()
        except Exception as exc:
            click.echo(f"  Voice identification failed: {exc}", err=True)
            click.echo("  Falling back to interactive labeling.")
            click.echo()

    # Show summary table
    click.echo(
        f"  {'#':<4} {'Label':<14} {'Channel':<10} {'Segments':<10} {'Auto-ID':<20} {'Sample Text'}"
    )
    click.echo(f"  {'─' * 4} {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 20} {'─' * 40}")
    for i, sp in enumerate(speakers, 1):
        auto_name = ""
        if sp.id in auto_matches:
            m = auto_matches[sp.id]
            auto_name = f"{m.name} ({m.confidence:.0%})"
        click.echo(
            f"  {i:<4} {sp.id:<14} {sp.channel:<10} {sp.segment_count:<10} {auto_name:<20} {sp.sample_text[:40]}"
        )
    click.echo()

    # ── Build label map ──
    can_play = not no_audio and wav_path and wav_path.exists()

    if not can_play and not no_audio:
        click.echo("  (No WAV file found — skipping audio playback)")
        click.echo()

    label_map: dict[str, str] = {}
    temp_clips: list[Path] = []

    # Separate speakers into auto-matched and unrecognized
    auto_labeled = {sp.id for sp in speakers if sp.id in auto_matches}
    unrecognized = [sp for sp in speakers if sp.id not in auto_matches]

    # Apply auto-matched labels directly
    if auto and auto_matches:
        click.echo("Auto-applying confident voice matches:")
        for sp in speakers:
            if sp.id in auto_matches:
                match = auto_matches[sp.id]
                label_map[sp.id] = match.name
                click.echo(
                    f"  {sp.id} -> {click.style(match.name, fg='green')}  ({match.confidence:.0%})"
                )
        click.echo()

    # Interactive labeling for unrecognized speakers (or all speakers if not --auto)
    speakers_to_prompt = unrecognized if auto else speakers

    if speakers_to_prompt:
        if auto and unrecognized:
            click.echo(
                f"{len(unrecognized)} unrecognized speaker(s) — prompting interactively:"
            )
            click.echo()

        try:
            for i, sp in enumerate(speakers_to_prompt, 1):
                click.echo(
                    f"Speaker {i}/{len(speakers_to_prompt)}: {click.style(sp.id, bold=True)}"
                )
                click.echo(f"  Channel: {sp.channel}  |  Segments: {sp.segment_count}")
                click.echo(f'  Sample:  "{sp.sample_text}"')

                # Play audio clip
                if can_play:
                    try:
                        clip_path = extract_speaker_clip(wav_path, sp)
                        temp_clips.append(clip_path)
                        click.echo(f"  Playing audio clip... ", nl=False)
                        proc = play_clip(clip_path)
                        proc.wait()
                        click.echo("done")
                    except Exception as exc:
                        click.echo(f"  (Audio playback failed: {exc})")

                # Prompt for name
                new_name = click.prompt(
                    f"  Enter name for {sp.id} (Enter to keep)",
                    default="",
                    show_default=False,
                ).strip()

                if new_name and new_name != sp.id:
                    label_map[sp.id] = new_name
                    click.echo(f"  {sp.id} -> {click.style(new_name, fg='green')}")
                else:
                    click.echo(f"  Keeping: {sp.id}")
                click.echo()

        finally:
            # Clean up temp clips
            for clip in temp_clips:
                try:
                    clip.unlink(missing_ok=True)
                except Exception:
                    pass

    if not label_map:
        click.echo("No labels changed. Nothing to do.")
        return

    click.echo("Applying labels:")
    for old, new in sorted(label_map.items()):
        click.echo(f"  {old} -> {new}")
    click.echo()

    # Apply labels and regenerate outputs
    regenerate_summary = not no_summary

    result_files = apply_labels(
        session_path,
        label_map,
        regenerate_summary=regenerate_summary,
        summary_backend=summary_backend,
        summary_model=summary_model,
        progress_callback=lambda msg: click.echo(f"  {msg}"),
    )

    click.echo()
    click.echo("Updated files:")
    for fmt, path in result_files.items():
        click.echo(f"  {fmt}: {path}")

    # ── Update voice profiles with confirmed labels ──
    if auto and label_map:
        click.echo()
        click.echo("Updating voice profiles with confirmed labels...")
        try:
            from meet.voiceprint import update_profiles_from_confirmed_labels

            transcript = _load_transcript(files["json"])
            # Rebuild channel_map if not already done
            if not channel_map:
                channel_map = _detect_speaker_channels(
                    wav_path,
                    transcript.segments,
                    transcript.speakers,
                )
            update_profiles_from_confirmed_labels(
                wav_path,
                transcript.segments,
                label_map,
                channel_map,
            )
            click.echo("  Voice profiles updated.")
        except Exception as exc:
            click.echo(f"  Profile update failed: {exc}", err=True)


@main.command()
@click.argument("session_dirs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--list",
    "list_profiles",
    is_flag=True,
    default=False,
    help="List enrolled speaker profiles and exit",
)
def enroll(session_dirs, list_profiles):
    """Enroll speaker voice profiles from labeled session directories.

    Extracts voice embeddings from sessions that already have speaker labels
    (set via 'meet label') and stores them in ~/.config/meet/speaker_profiles.json.
    Future meetings will automatically recognize these speakers.

    \b
    Examples:
        meet enroll ~/meet-recordings/meeting-20260330-170216_WeeklySync
        meet enroll ~/meet-recordings/meeting-20260330-*
        meet enroll --list
    """
    from meet.voiceprint import enroll_session, load_profiles, PROFILES_PATH

    if list_profiles:
        profiles = load_profiles()
        if not profiles:
            click.echo("No speaker profiles enrolled yet.")
            click.echo(f"  Run: meet enroll <session_dir>")
            return
        click.echo(f"Enrolled speaker profiles ({PROFILES_PATH}):")
        click.echo()
        click.echo(f"  {'Name':<20} {'Sessions'}")
        click.echo(f"  {'----':<20} {'--------'}")
        for name, profile in sorted(profiles.items()):
            click.echo(f"  {name:<20} {profile.n_sessions}")
        return

    if not session_dirs:
        click.echo(
            "Error: provide at least one session directory, or use --list", err=True
        )
        raise SystemExit(1)

    total_enrolled = 0
    total_updated = 0

    for session_dir in session_dirs:
        session_path = Path(session_dir)
        click.echo(f"Enrolling: {session_path.name}")

        try:
            status = enroll_session(
                session_path,
                progress_callback=lambda msg: click.echo(msg),
            )
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"  Skipped: {exc}", err=True)
            continue
        except Exception as exc:
            click.echo(f"  Error: {exc}", err=True)
            continue

        enrolled = sum(1 for ok in status.values() if ok)
        total_enrolled += enrolled
        click.echo(f"  Done: {enrolled} speaker(s) enrolled/updated")
        click.echo()

    # Final summary
    profiles = load_profiles()
    click.echo(f"Profile database now contains {len(profiles)} speaker(s):")
    for name, p in sorted(profiles.items()):
        click.echo(f"  {name} ({p.n_sessions} session(s))")


@main.command()
@click.argument("session_dirs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Push even if the meeting doesn't match a scheduled meeting",
)
@click.option(
    "--meeting-type",
    type=str,
    default=None,
    help="Override meeting type folder (e.g. 'weekly-sync', 'dev-standup')",
)
@click.option(
    "--list-schedule",
    is_flag=True,
    default=False,
    help="Show the current sync schedule and exit",
)
@click.option(
    "--init-config",
    is_flag=True,
    default=False,
    help="Create an example sync config and exit",
)
def sync(session_dirs, force, meeting_type, list_schedule, init_config):
    """Sync meeting artifacts to a configured Git repository.

    Detects whether each session matches a configured meeting schedule and
    pushes the transcript, summary, PDF, SRT, and JSON to the team repo.
    Audio files and internal metadata are excluded.

    \b
    Setup:
        meet sync --init-config          # create example config
        # edit ~/.config/meet/sync_config.json with your repo URL and schedule

    \b
    Examples:
        meet sync ~/meet-recordings/meeting-20260330-170216_WeeklySync
        meet sync --force --meeting-type weekly-sync ~/meet-recordings/meeting-20260330-*
        meet sync --list-schedule
    """
    from meet.sync import (
        detect_meeting_type,
        sync_session,
        load_sync_config,
        save_sync_config,
        is_sync_configured,
        MeetingMatch,
        ensure_repo_cloned,
        SYNC_CONFIG_PATH,
        EXAMPLE_CONFIG,
    )

    if init_config:
        if SYNC_CONFIG_PATH.exists():
            click.echo(f"Config already exists: {SYNC_CONFIG_PATH}")
            click.echo("Edit it manually or delete it to regenerate.")
        else:
            save_sync_config(EXAMPLE_CONFIG)
            click.echo(f"Example config created: {SYNC_CONFIG_PATH}")
            click.echo("Edit it with your repo URL and meeting schedule.")
        return

    if list_schedule:
        config = load_sync_config()
        repo_url = config.get("repo_url", "")
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        if repo_url:
            click.echo(f"Sync repo: {repo_url}")
        else:
            click.echo("Sync repo: (not configured)")
        click.echo()

        meetings = config.get("meetings", [])
        if not meetings:
            click.echo("No meetings configured.")
            click.echo(f"Edit {SYNC_CONFIG_PATH} to add your schedule.")
        else:
            click.echo("Meeting schedule:")
            click.echo()
            for m in meetings:
                days = ", ".join(day_names[d] for d in m.get("days", []))
                click.echo(f"  {m['name']}")
                click.echo(f"    folder:  meetings/{m['folder']}/")
                click.echo(f"    days:    {days}")
                click.echo(
                    f"    time:    {m['hour_utc']:02d}:00 UTC ±{m.get('window_minutes', 60)} min"
                )
                click.echo()
        return

    if not session_dirs:
        click.echo("Error: provide at least one session directory", err=True)
        raise SystemExit(1)

    if not is_sync_configured():
        click.echo(
            "Error: sync not configured. Run 'meet sync --init-config' to get started.",
            err=True,
        )
        raise SystemExit(1)

    for session_dir in session_dirs:
        session_path = Path(session_dir)
        click.echo(f"Syncing: {session_path.name}")

        if meeting_type:
            match = MeetingMatch(name=meeting_type, folder=meeting_type)
        else:
            match = detect_meeting_type(session_path)

        if match is None and not force:
            click.echo(
                f"  Skipped: not a scheduled meeting "
                f"(use --force to push anyway, --meeting-type to specify type)",
                err=True,
            )
            continue

        if match is None and force:
            click.echo("  Warning: no schedule match, using 'other' folder", err=True)
            match = MeetingMatch(name="Meeting", folder="other")

        try:
            files = sync_session(
                session_path,
                match,
                progress_callback=lambda msg: click.echo(msg),
            )
            click.echo(f"  Done: {len(files)} file(s) pushed as {match.folder}/")
        except Exception as exc:
            click.echo(f"  Error: {exc}", err=True)
        click.echo()


@main.command()
@click.argument("session_dirs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--older-than",
    type=int,
    default=None,
    help="Only compress sessions older than N days",
)
@click.option(
    "--keep-wav",
    is_flag=True,
    default=False,
    help="Keep original WAV files after compression",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be compressed without doing it",
)
def archive(session_dirs, older_than, keep_wav, dry_run):
    """Compress session WAV files to OGG/Opus to save disk space.

    If no SESSION_DIRS are given, scans ~/meet-recordings/ for all sessions
    that still have uncompressed WAV files.

    \b
    Examples:
        meet archive
        meet archive --dry-run
        meet archive --older-than 7
        meet archive ~/meet-recordings/meeting-20260325-150203_SOVEREIGNJORDAN
        meet archive --keep-wav ~/meet-recordings/meeting-*
    """
    import datetime
    from meet.audio import compress_audio

    recordings_dir = Path.home() / "meet-recordings"

    # Collect target directories.
    if session_dirs:
        dirs = [Path(d) for d in session_dirs]
    elif recordings_dir.is_dir():
        dirs = sorted([d for d in recordings_dir.iterdir() if d.is_dir()])
    else:
        click.echo(f"No session directories found in {recordings_dir}")
        return

    # Find WAV files to compress.
    targets: list[Path] = []
    for d in dirs:
        for wav in sorted(d.glob("*.wav")):
            # Skip chunk files (intermediate recording artifacts).
            if ".chunk-" in wav.name:
                continue
            # Already has a compressed version?
            if wav.with_suffix(".ogg").exists():
                continue
            # Age filter.
            if older_than is not None:
                mtime = datetime.datetime.fromtimestamp(wav.stat().st_mtime)
                age_days = (datetime.datetime.now() - mtime).days
                if age_days < older_than:
                    continue
            targets.append(wav)

    if not targets:
        click.echo("No uncompressed WAV files to archive.")
        return

    # Compute totals.
    total_wav_size = sum(w.stat().st_size for w in targets)
    click.echo(
        f"Found {len(targets)} WAV file(s) totaling {total_wav_size / 1_048_576:.1f} MB"
    )

    if dry_run:
        click.echo()
        for wav in targets:
            size_mb = wav.stat().st_size / 1_048_576
            click.echo(f"  [DRY RUN] {wav.parent.name}/{wav.name} ({size_mb:.1f} MB)")
        # Estimate: Opus at 48 kbps ~= 0.36 MB/min; WAV at 16kHz stereo = 3.84 MB/min
        estimated_ratio = 10.5
        estimated_ogg = total_wav_size / estimated_ratio
        click.echo(
            f"\n  Estimated compressed size: ~{estimated_ogg / 1_048_576:.0f} MB "
            f"(~{estimated_ratio:.0f}x reduction)"
        )
        return

    # Compress each file.
    compressed_count = 0
    saved_bytes = 0
    for wav in targets:
        label = f"{wav.parent.name}/{wav.name}"
        wav_size = wav.stat().st_size
        click.echo(
            f"  Compressing {label} ({wav_size / 1_048_576:.1f} MB)...", nl=False
        )
        try:
            ogg_path = compress_audio(wav, keep_wav=keep_wav)
            ogg_size = ogg_path.stat().st_size
            ratio = wav_size / ogg_size if ogg_size > 0 else 0
            saved = wav_size - ogg_size
            saved_bytes += saved
            compressed_count += 1
            click.echo(f" -> {ogg_size / 1_048_576:.1f} MB ({ratio:.0f}x)")
        except Exception as exc:
            click.echo(f" FAILED: {exc}", err=True)

    click.echo(
        f"\nDone: {compressed_count}/{len(targets)} files compressed, "
        f"{saved_bytes / 1_048_576:.0f} MB saved"
    )


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory for recordings and transcripts",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="large-v3-turbo",
    help="Whisper model (default: large-v3-turbo)",
)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option("--compute-type", type=str, default="float16")
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--language", "-l", type=str, default="auto")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--virtual-sink", is_flag=True, default=False)
@click.option(
    "--mic", type=str, default=None, help="Mic source name (default: system default)"
)
@click.option(
    "--monitor",
    type=str,
    default=None,
    help="Monitor source name (default: default sink monitor)",
)
@click.option(
    "--summarize/--no-summarize",
    default=True,
    help="Generate AI meeting summary (default: on)",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["ollama", "openrouter", "claudemax", "openai"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: ollama, or MEETSCRIBE_SUMMARY_BACKEND env var)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MEETSCRIBE_SUMMARY_MODEL env var)",
)
def gui(
    output_dir,
    model,
    device,
    compute_type,
    batch_size,
    language,
    hf_token,
    min_speakers,
    max_speakers,
    virtual_sink,
    mic,
    monitor,
    summarize,
    summary_backend,
    summary_model,
):
    """Launch the GUI recording widget."""
    from meet.gui import launch

    launch(
        output_dir=output_dir,
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        virtual_sink=virtual_sink,
        mic=mic,
        monitor=monitor,
        summarize=summarize,
        summary_backend=summary_backend,
        summary_model=summary_model,
    )


if __name__ == "__main__":
    main()
