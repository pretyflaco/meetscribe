"""GTK3 GUI widget for meet — a small always-on-top recording control.

Window layout (~300x180px):
    ┌──────────────────────────────┐
    │  Meet Recorder               │
    │                              │
    │     00:00:00    0 KB         │
    │     Ready                    │
    │                              │
    │     [ ● Record ]             │
    │   Open Transcript  Open Folder│
    └──────────────────────────────┘

States:
    idle        → "Ready", green Record button
    recording   → "Recording...", red Stop button, timer ticking
    draining    → "Flushing buffer... Xs", buttons disabled
    transcribing → "Transcribing...", buttons disabled
    done        → "Done — transcript saved", green Record button

The recording session runs in a background thread. UI updates are
dispatched via GLib.timeout_add (every 500ms poll).
"""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, Gdk, Pango  # noqa: E402

from meet.capture import DRAIN_SECONDS
from meet.utils import fmt_elapsed, fmt_size


# ─── CSS ────────────────────────────────────────────────────────────────────

_CSS = b"""
.record-btn {
    background: #2ecc71;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.record-btn:hover {
    background: #27ae60;
}
.stop-btn {
    background: #e74c3c;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.stop-btn:hover {
    background: #c0392b;
}
.disabled-btn {
    background: #95a5a6;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.timer-label {
    font-size: 28px;
    font-weight: bold;
    font-family: monospace;
}
.size-label {
    font-size: 14px;
    color: #7f8c8d;
    font-family: monospace;
}
.status-label {
    font-size: 13px;
    color: #7f8c8d;
}
.status-recording {
    font-size: 13px;
    color: #e74c3c;
    font-weight: bold;
}
.status-draining {
    font-size: 13px;
    color: #f39c12;
    font-weight: bold;
}
.status-transcribing {
    font-size: 13px;
    color: #3498db;
    font-weight: bold;
}
.status-preparing {
    font-size: 13px;
    color: #f39c12;
    font-weight: bold;
}
.status-downloading {
    font-size: 13px;
    color: #e67e22;
    font-weight: bold;
}
.status-awaiting {
    font-size: 13px;
    color: #e67e22;
    font-weight: bold;
}
.status-summarizing {
    font-size: 13px;
    color: #9b59b6;
    font-weight: bold;
}
.status-labeling {
    font-size: 13px;
    color: #e67e22;
    font-weight: bold;
}
.status-done {
    font-size: 13px;
    color: #2ecc71;
    font-weight: bold;
}
.status-error {
    font-size: 13px;
    color: #e74c3c;
    font-weight: bold;
}
.action-btn {
    background: transparent;
    color: #3498db;
    font-size: 12px;
    border: none;
    padding: 2px 8px;
    text-decoration: underline;
}
.action-btn:hover {
    color: #2980b9;
}
progressbar trough {
    min-height: 6px;
    background: #ecf0f1;
    border-radius: 3px;
}
progressbar progress {
    min-height: 6px;
    background: #e67e22;
    border-radius: 3px;
}
"""


# ─── State enum ─────────────────────────────────────────────────────────────

class _State:
    IDLE = "idle"
    RECORDING = "recording"
    DRAINING = "draining"
    PREPARING_GPU = "preparing_gpu"
    TRANSCRIBING = "transcribing"
    AWAITING_ALIGNMENT = "awaiting_alignment"
    DOWNLOADING_MODEL = "downloading_model"
    LABELING_SPEAKERS = "labeling_speakers"
    SUMMARIZING = "summarizing"
    DONE = "done"
    ERROR = "error"


# ─── Main Window ────────────────────────────────────────────────────────────

class MeetRecorderWindow(Gtk.Window):

    def __init__(self, capture_kwargs: dict, transcribe_kwargs: dict,
                 summarize: bool = True, summary_backend: str | None = None,
                 summary_model: str | None = None):
        super().__init__(title="Meet Recorder")

        self._capture_kwargs = capture_kwargs
        self._transcribe_kwargs = transcribe_kwargs
        self._summarize = summarize
        self._summary_backend = summary_backend
        self._summary_model = summary_model
        self._session = None
        self._state = _State.IDLE
        self._worker_thread = None
        self._drain_remaining = 0
        self._last_output: Path | None = None
        self._last_pdf: Path | None = None
        self._error_msg: str | None = None

        # Threading synchronization for alignment model prompt
        self._alignment_event = threading.Event()
        self._alignment_choice: str | None = None  # "download" or "skip"
        self._alignment_lang: str | None = None

        # Threading synchronization for speaker labeling
        self._label_event = threading.Event()
        self._label_result: dict[str, str] | None = None  # label_map or None (skip)
        self._label_speakers: list = []  # SpeakerInfo list, set by worker
        self._label_entries: list = []   # Gtk.Entry widgets
        self._label_temp_clips: list[Path] = []  # temp WAV files for cleanup
        self._label_auto_matches: dict = {}  # speaker_id -> SpeakerMatch, set by worker
        self._label_channel_map: dict = {}   # speaker_id -> 'mic'|'system', set by worker
        self._label_audio_path: Path | None = None  # audio file for profile update

        # Window properties
        self.set_default_size(300, 150)
        self.set_keep_above(True)
        self.set_resizable(False)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Load CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(_CSS)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_margin_top(12)
        vbox.set_margin_bottom(12)
        vbox.set_margin_start(20)
        vbox.set_margin_end(20)

        # Timer
        self._timer_label = Gtk.Label(label="00:00:00")
        self._timer_label.get_style_context().add_class("timer-label")
        vbox.pack_start(self._timer_label, False, False, 0)

        # File size
        self._size_label = Gtk.Label(label="0 KB")
        self._size_label.get_style_context().add_class("size-label")
        vbox.pack_start(self._size_label, False, False, 0)

        # Status
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.set_line_wrap(True)
        self._status_label.set_max_width_chars(40)
        self._status_label.get_style_context().add_class("status-label")
        vbox.pack_start(self._status_label, False, False, 4)

        # Button
        self._button = Gtk.Button(label="● Record")
        self._button.get_style_context().add_class("record-btn")
        self._button.connect("clicked", self._on_button_clicked)
        vbox.pack_start(self._button, False, False, 4)

        # Action buttons (shown after transcription completes)
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        action_box.set_halign(Gtk.Align.CENTER)

        self._open_transcript_btn = Gtk.Button(label="Open Transcript")
        self._open_transcript_btn.get_style_context().add_class("action-btn")
        self._open_transcript_btn.connect("clicked", self._on_open_transcript)
        action_box.pack_start(self._open_transcript_btn, False, False, 0)

        self._open_folder_btn = Gtk.Button(label="Open Folder")
        self._open_folder_btn.get_style_context().add_class("action-btn")
        self._open_folder_btn.connect("clicked", self._on_open_folder)
        action_box.pack_start(self._open_folder_btn, False, False, 0)

        vbox.pack_start(action_box, False, False, 0)

        # Alignment model prompt (shown when model is missing)
        self._alignment_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._alignment_box.set_halign(Gtk.Align.CENTER)

        self._alignment_label = Gtk.Label()
        self._alignment_label.set_line_wrap(True)
        self._alignment_label.set_max_width_chars(35)
        self._alignment_label.get_style_context().add_class("status-label")
        self._alignment_box.pack_start(self._alignment_label, False, False, 0)

        align_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        align_btn_box.set_halign(Gtk.Align.CENTER)

        self._download_btn = Gtk.Button(label="Download & Continue")
        self._download_btn.get_style_context().add_class("record-btn")
        self._download_btn.connect("clicked", self._on_alignment_download)
        align_btn_box.pack_start(self._download_btn, False, False, 0)

        self._skip_align_btn = Gtk.Button(label="Skip Alignment")
        self._skip_align_btn.get_style_context().add_class("action-btn")
        self._skip_align_btn.connect("clicked", self._on_alignment_skip)
        align_btn_box.pack_start(self._skip_align_btn, False, False, 0)

        self._alignment_box.pack_start(align_btn_box, False, False, 0)
        vbox.pack_start(self._alignment_box, False, False, 4)

        # Speaker labeling prompt (shown after transcription if 2+ speakers)
        self._label_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        label_header = Gtk.Label(label="Assign names to speakers:")
        label_header.set_line_wrap(True)
        label_header.set_max_width_chars(35)
        label_header.get_style_context().add_class("status-label")
        self._label_box.pack_start(label_header, False, False, 0)

        # Scrollable area for speaker rows (populated dynamically)
        self._label_scroll = Gtk.ScrolledWindow()
        self._label_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._label_scroll.set_min_content_height(60)
        self._label_scroll.set_max_content_height(200)
        self._label_rows_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._label_scroll.add(self._label_rows_box)
        self._label_box.pack_start(self._label_scroll, True, True, 0)

        label_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        label_btn_box.set_halign(Gtk.Align.CENTER)

        self._label_apply_btn = Gtk.Button(label="Apply & Continue")
        self._label_apply_btn.get_style_context().add_class("record-btn")
        self._label_apply_btn.connect("clicked", self._on_label_apply)
        label_btn_box.pack_start(self._label_apply_btn, False, False, 0)

        self._label_skip_btn = Gtk.Button(label="Skip")
        self._label_skip_btn.get_style_context().add_class("action-btn")
        self._label_skip_btn.connect("clicked", self._on_label_skip)
        label_btn_box.pack_start(self._label_skip_btn, False, False, 0)

        self._label_box.pack_start(label_btn_box, False, False, 0)
        vbox.pack_start(self._label_box, False, False, 4)

        # Download progress bar (pulsing, shown during model downloads)
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_pulse_step(0.05)
        vbox.pack_start(self._progress_bar, False, False, 2)

        self.add(vbox)
        self.connect("destroy", self._on_destroy)

        # Periodic UI update (every 500ms)
        self._poll_id = GLib.timeout_add(500, self._poll_status)

    # ── Button handler ──────────────────────────────────────────────────

    def _on_button_clicked(self, _widget):
        if self._state == _State.IDLE or self._state == _State.DONE or self._state == _State.ERROR:
            self._start_recording()
        elif self._state == _State.RECORDING:
            self._stop_recording()

    def _on_open_transcript(self, _widget):
        if self._last_pdf and self._last_pdf.exists():
            subprocess.Popen(["xdg-open", str(self._last_pdf)])
        elif self._last_output:
            txt_path = self._last_output.with_suffix(".txt")
            if txt_path.exists():
                subprocess.Popen(["xdg-open", str(txt_path)])

    def _on_open_folder(self, _widget):
        if self._last_output:
            folder = self._last_output.parent
            subprocess.Popen(["xdg-open", str(folder)])

    def _on_alignment_download(self, _widget):
        """User chose 'Download & Continue' for the missing alignment model."""
        self._alignment_choice = "download"
        self._alignment_box.hide()
        self.resize(300, 150)
        self.set_resizable(False)
        self._alignment_event.set()

    def _on_alignment_skip(self, _widget):
        """User chose 'Skip Alignment' for the missing alignment model."""
        self._alignment_choice = "skip"
        self._alignment_box.hide()
        self.resize(300, 150)
        self.set_resizable(False)
        self._alignment_event.set()

    def _on_label_apply(self, _widget):
        """User clicked 'Apply & Continue' on the speaker labeling dialog."""
        label_map = {}
        for sp, entry in zip(self._label_speakers, self._label_entries):
            new_name = entry.get_text().strip()
            if new_name and new_name != sp.id:
                label_map[sp.id] = new_name
        self._label_result = label_map if label_map else None
        self._label_box.hide()
        self._cleanup_label_clips()
        self.resize(300, 150)
        self.set_resizable(False)
        self._label_event.set()

        # Update voice profiles in background with confirmed labels
        if self._label_result and self._label_audio_path:
            import threading as _threading
            _threading.Thread(
                target=self._update_voice_profiles,
                args=(self._label_result,),
                daemon=True,
            ).start()

    def _on_label_skip(self, _widget):
        """User clicked 'Skip' on the speaker labeling dialog."""
        self._label_result = None
        self._label_box.hide()
        self._cleanup_label_clips()
        self.resize(300, 150)
        self.set_resizable(False)
        self._label_event.set()

    def _on_label_play(self, _widget, clip_path):
        """Play a speaker audio clip."""
        try:
            from meet.label import play_clip
            proc = play_clip(clip_path)
            # Don't block the UI — fire and forget
        except Exception:
            pass

    def _cleanup_label_clips(self):
        """Remove temporary audio clips."""
        for clip in self._label_temp_clips:
            try:
                clip.unlink(missing_ok=True)
            except Exception:
                pass
        self._label_temp_clips.clear()

    def _update_voice_profiles(self, confirmed_label_map: dict):
        """Background task: update voice profiles with confirmed speaker labels."""
        if not self._label_audio_path or not confirmed_label_map:
            return
        try:
            from meet.voiceprint import update_profiles_from_confirmed_labels
            # We need the original (pre-relabel) segments, stored on the transcript
            # that was passed to _do_label_speakers.  Retrieve from the saved JSON.
            from meet.label import _find_session_files, _load_transcript
            files = _find_session_files(self._label_audio_path.parent)
            transcript_json = files.get("json")
            if not transcript_json or not transcript_json.exists():
                return
            transcript = _load_transcript(transcript_json)
            update_profiles_from_confirmed_labels(
                self._label_audio_path,
                transcript.segments,
                confirmed_label_map,
                self._label_channel_map,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Voice profile update failed: %s", exc
            )

    def _build_label_rows(self, speakers, wav_path, auto_matches=None):
        """Build the per-speaker label rows in the GTK label dialog.

        Called from GTK main thread via GLib.idle_add.

        Args:
            speakers: List of SpeakerInfo objects.
            wav_path: Path to audio for clip playback.
            auto_matches: Optional dict of speaker_id -> SpeakerMatch with
                          auto-recognized names and confidence scores.
        """
        from meet.label import extract_speaker_clip

        if auto_matches is None:
            auto_matches = {}

        # Clear previous rows
        for child in self._label_rows_box.get_children():
            self._label_rows_box.remove(child)
        self._label_entries.clear()
        self._label_temp_clips.clear()

        for sp in speakers:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

            # Speaker ID label
            id_label = Gtk.Label(label=f"{sp.id}:")
            id_label.set_width_chars(10)
            id_label.set_xalign(1.0)
            row.pack_start(id_label, False, False, 0)

            # Text entry for new name — pre-fill if auto-recognized
            entry = Gtk.Entry()
            match = auto_matches.get(sp.id)
            if match:
                entry.set_text(match.name)
                pct = int(match.confidence * 100)
                entry.set_tooltip_text(f"Auto-recognized: {match.name} ({pct}% confidence)")
            else:
                entry.set_placeholder_text(sp.id)
            entry.set_width_chars(14)
            row.pack_start(entry, True, True, 0)
            self._label_entries.append(entry)

            # Play button (if audio available)
            if wav_path and wav_path.exists():
                try:
                    clip_path = extract_speaker_clip(wav_path, sp)
                    self._label_temp_clips.append(clip_path)
                    play_btn = Gtk.Button(label="Play")
                    play_btn.get_style_context().add_class("action-btn")
                    play_btn.connect("clicked", self._on_label_play, clip_path)
                    row.pack_start(play_btn, False, False, 0)
                except Exception:
                    pass

            self._label_rows_box.pack_start(row, False, False, 0)

        self._label_box.show_all()

    def _attempt_download(self, lang: str, config) -> bool:
        """Download alignment model with retry prompt on failure.

        Called from the worker thread.  On network errors the user is
        shown a Retry / Skip Alignment prompt and can keep retrying
        indefinitely.

        Returns True when ready to proceed (model downloaded or user
        chose to skip alignment).  Returns False if the prompt was
        cancelled unexpectedly (caller should abort).
        """
        from meet.transcribe import download_alignment_model

        while True:
            GLib.idle_add(self._set_state, _State.DOWNLOADING_MODEL)
            try:
                download_alignment_model(
                    lang,
                    progress_callback=lambda msg: GLib.idle_add(
                        self._status_label.set_text, msg
                    ),
                )
                return True  # download succeeded
            except Exception as dl_exc:
                # Show retry / skip prompt
                self._alignment_event.clear()
                self._alignment_choice = None
                err_text = str(dl_exc)
                # Truncate very long exception messages
                if len(err_text) > 120:
                    err_text = err_text[:117] + "..."

                def _show_retry(msg=err_text):
                    self._alignment_label.set_text(
                        f"Download failed:\n{msg}"
                    )
                    self._download_btn.set_label("Retry")
                    self._set_state(_State.AWAITING_ALIGNMENT)

                GLib.idle_add(_show_retry)
                self._alignment_event.wait()

                # Reset button label for future prompts
                GLib.idle_add(
                    self._download_btn.set_label, "Download & Continue"
                )

                if self._alignment_choice == "download":
                    continue  # retry the download
                elif self._alignment_choice == "skip":
                    config.skip_alignment = True
                    return True
                else:
                    return False  # unexpected — caller should abort

    # ── Recording lifecycle ─────────────────────────────────────────────

    def _start_recording(self):
        from meet.capture import create_session, check_prerequisites

        issues = check_prerequisites()
        if issues:
            self._set_error("Prerequisites failed: " + "; ".join(issues))
            return

        self._session = create_session(**self._capture_kwargs)
        self._session.start()
        self._last_output = None
        self._last_pdf = None
        self._error_msg = None
        self._set_state(_State.RECORDING)

    def _stop_recording(self):
        """Start the drain + stop + transcribe pipeline in a background thread."""
        self._set_state(_State.DRAINING)
        self._drain_remaining = DRAIN_SECONDS
        self._worker_thread = threading.Thread(
            target=self._drain_stop_transcribe, daemon=True
        )
        self._worker_thread.start()

    def _drain_stop_transcribe(self):
        """Background thread: drain buffer, stop recording, transcribe, summarize, generate PDF."""
        output = self._do_drain()
        if output is None:
            return

        config, transcript = self._do_transcribe(output)
        if transcript is None:
            return

        transcript = self._do_label_speakers(output, transcript)

        # Save transcript (or re-save with updated labels)
        transcript.save(output.parent, basename=output.stem)

        self._do_post_process(output, transcript)
        self._do_sync(output)
        GLib.idle_add(self._set_state, _State.DONE)

    def _do_drain(self):
        """Drain the recording buffer, stop capture, and return the output path.

        Returns the output Path on success, or None if recording failed.
        """
        for remaining in range(DRAIN_SECONDS, 0, -1):
            self._drain_remaining = remaining
            time.sleep(1)
        self._drain_remaining = 0

        session = self._session
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            GLib.idle_add(self._set_error, "No audio was recorded")
            return None

        self._last_output = output
        return output

    def _do_transcribe(self, output):
        """Prepare GPU, check alignment model, and run transcription.

        Returns (config, transcript) on success, or (None, None) on failure.
        """
        from meet.transcribe import (
            TranscriptionConfig, transcribe as do_transcribe,
            AlignmentModelMissing, ensure_gpu_available,
            check_alignment_model_cached,
            ALIGNMENT_MODELS, _LANG_NAMES, _MODEL_SIZES,
        )

        config = TranscriptionConfig(**self._transcribe_kwargs)
        if not config.hf_token:
            GLib.idle_add(
                self._set_error,
                "HF_TOKEN not set — diarization won't work.\n"
                "Add 'export HF_TOKEN=hf_...' to ~/.profile and re-login.",
            )
            return None, None

        # ── Prepare GPU (unload Ollama) ──
        GLib.idle_add(self._set_state, _State.PREPARING_GPU)
        ensure_gpu_available(
            progress_callback=lambda msg: GLib.idle_add(
                self._status_label.set_text, msg
            )
        )

        # ── Pre-flight: check alignment model cache (when language is known) ──
        preflight_lang = config.language if config.language != "auto" else None
        if preflight_lang and preflight_lang in ALIGNMENT_MODELS:
            if not config.skip_alignment and not check_alignment_model_cached(preflight_lang):
                lang_name = _LANG_NAMES.get(preflight_lang, preflight_lang)
                size = _MODEL_SIZES.get(preflight_lang, "unknown size")
                self._alignment_lang = lang_name

                self._alignment_event.clear()
                self._alignment_choice = None

                def _show_preflight_prompt():
                    self._alignment_label.set_text(
                        f"Alignment model for {lang_name} not downloaded ({size})."
                    )
                    self._set_state(_State.AWAITING_ALIGNMENT)

                GLib.idle_add(_show_preflight_prompt)
                self._alignment_event.wait()

                if self._alignment_choice == "download":
                    if not self._attempt_download(preflight_lang, config):
                        GLib.idle_add(self._set_error, "Download cancelled")
                        return None, None
                elif self._alignment_choice == "skip":
                    config.skip_alignment = True
                else:
                    GLib.idle_add(self._set_error, "Alignment model prompt cancelled")
                    return None, None

        # ── Transcribe (with alignment model handling) ──
        GLib.idle_add(self._set_state, _State.TRANSCRIBING)
        transcript = None

        try:
            transcript = do_transcribe(output, config)
        except AlignmentModelMissing as exc:
            lang_name = _LANG_NAMES.get(exc.lang, exc.lang)
            size = _MODEL_SIZES.get(exc.lang, "unknown size")
            self._alignment_lang = lang_name

            self._alignment_event.clear()
            self._alignment_choice = None

            def _show_prompt():
                self._alignment_label.set_text(
                    f"Alignment model for {lang_name} not downloaded ({size})."
                )
                self._set_state(_State.AWAITING_ALIGNMENT)

            GLib.idle_add(_show_prompt)
            self._alignment_event.wait()

            if self._alignment_choice == "download":
                if not self._attempt_download(exc.lang, config):
                    GLib.idle_add(self._set_error, "Download cancelled")
                    return None, None

                GLib.idle_add(self._set_state, _State.TRANSCRIBING)
                try:
                    transcript = do_transcribe(output, config)
                except Exception as retry_exc:
                    GLib.idle_add(self._set_error, f"Transcription failed: {retry_exc}")
                    return None, None

            elif self._alignment_choice == "skip":
                GLib.idle_add(self._set_state, _State.TRANSCRIBING)
                config.skip_alignment = True
                try:
                    transcript = do_transcribe(output, config)
                except Exception as skip_exc:
                    GLib.idle_add(self._set_error, f"Transcription failed: {skip_exc}")
                    return None, None
            else:
                GLib.idle_add(self._set_error, "Alignment model prompt cancelled")
                return None, None

        except Exception as exc:
            GLib.idle_add(self._set_error, f"Transcription failed: {exc}")
            return None, None

        return config, transcript

    def _do_label_speakers(self, output, transcript):
        """Optionally show speaker labeling dialog and relabel transcript.

        Returns the (possibly relabeled) transcript.
        """
        if len(transcript.speakers) < 2:
            return transcript

        from meet.label import (
            get_speakers as _get_speakers,
            find_session_files,
            relabel_transcript_in_memory,
        )

        try:
            transcript.save(output.parent, basename=output.stem)

            spk_infos = _get_speakers(output.parent)
            if spk_infos:
                session_files = find_session_files(output.parent)
                wav_path = session_files.get("wav")

                # Build channel map: speaker_id -> 'mic' | 'system'
                from meet.audio import read_stereo_channels, compute_speaker_channel_energy
                channel_map: dict[str, str] = {}
                if wav_path and wav_path.exists():
                    stereo = read_stereo_channels(wav_path)
                    if stereo is not None:
                        mic_ratio = compute_speaker_channel_energy(
                            stereo.mic, stereo.system, transcript.segments, stereo.sample_rate
                        )
                        for spk_id, ratio in mic_ratio.items():
                            channel_map[spk_id] = "mic" if ratio > 0.5 else "system"

                # Run voice identification against profile database
                auto_matches: dict = {}
                if wav_path and wav_path.exists():
                    try:
                        from meet.voiceprint import identify_speakers
                        auto_matches = identify_speakers(
                            wav_path,
                            transcript.segments,
                            transcript.speakers,
                            channel_map,
                        )
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning(
                            "Voice identification failed: %s", exc
                        )

                self._label_speakers = spk_infos
                self._label_auto_matches = auto_matches
                self._label_channel_map = channel_map
                self._label_audio_path = wav_path

                self._label_event.clear()
                self._label_result = None

                def _show_label_dialog(
                    _spk_infos=spk_infos,
                    _wav_path=wav_path,
                    _auto_matches=auto_matches,
                ):
                    self._build_label_rows(_spk_infos, _wav_path, _auto_matches)
                    self._set_state(_State.LABELING_SPEAKERS)

                GLib.idle_add(_show_label_dialog)
                self._label_event.wait()

                if self._label_result:
                    transcript = relabel_transcript_in_memory(
                        transcript, self._label_result,
                    )
                    import json as _json
                    session_json = session_files.get("session")
                    if session_json and session_json.exists():
                        try:
                            meta = _json.loads(session_json.read_text(encoding="utf-8"))
                            meta["speaker_labels"] = self._label_result
                            session_json.write_text(
                                _json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
        except Exception:
            pass  # labeling is optional; don't fail the pipeline

        return transcript

    def _do_post_process(self, output, transcript):
        """Run summarization and PDF generation after transcription."""
        from meet.transcribe import post_process

        if self._summarize:
            GLib.idle_add(self._set_state, _State.SUMMARIZING)

        result = post_process(
            transcript, output.parent, output.stem,
            summarize=self._summarize,
            summary_backend=self._summary_backend,
            summary_model=self._summary_model,
            progress_callback=lambda msg: GLib.idle_add(
                self._status_label.set_text, msg
            ),
        )
        if result["pdf"]:
            self._last_pdf = result["pdf"]

    def _do_sync(self, output: Path) -> None:
        """If this is a scheduled meeting and sync is configured, push artifacts."""
        try:
            from meet.sync import maybe_sync_session
            maybe_sync_session(
                output.parent,
                progress_callback=lambda msg: GLib.idle_add(
                    self._status_label.set_text, msg
                ),
            )
        except Exception:
            pass  # sync is best-effort — never fail the main pipeline

    # ── State management ────────────────────────────────────────────────

    def _set_state(self, state):
        self._state = state
        ctx = self._button.get_style_context()

        # Remove all custom button classes
        for cls in ("record-btn", "stop-btn", "disabled-btn"):
            ctx.remove_class(cls)

        # Remove all status classes
        sctx = self._status_label.get_style_context()
        for cls in ("status-label", "status-recording", "status-draining",
                     "status-transcribing", "status-summarizing",
                     "status-preparing", "status-downloading",
                     "status-awaiting", "status-labeling",
                     "status-done", "status-error"):
            sctx.remove_class(cls)

        # Hide action buttons by default; only hide alignment prompt
        # if we're NOT entering the awaiting-alignment state
        self._open_transcript_btn.hide()
        self._open_folder_btn.hide()
        if state != _State.AWAITING_ALIGNMENT:
            self._alignment_box.hide()
        if state != _State.LABELING_SPEAKERS:
            self._label_box.hide()
        if state != _State.DOWNLOADING_MODEL:
            self._progress_bar.hide()

        if state == _State.IDLE:
            self._button.set_label("● Record")
            ctx.add_class("record-btn")
            self._button.set_sensitive(True)
            self._status_label.set_text("Ready")
            sctx.add_class("status-label")
            self._timer_label.set_text("00:00:00")
            self._size_label.set_text("0 KB")

        elif state == _State.RECORDING:
            self._button.set_label("■ Stop")
            ctx.add_class("stop-btn")
            self._button.set_sensitive(True)
            self._status_label.set_text("Recording...")
            sctx.add_class("status-recording")

        elif state == _State.DRAINING:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text(f"Flushing buffer... {DRAIN_SECONDS}s")
            sctx.add_class("status-draining")

        elif state == _State.PREPARING_GPU:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text("Preparing GPU...")
            sctx.add_class("status-preparing")

        elif state == _State.DOWNLOADING_MODEL:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            lang_name = self._alignment_lang or "model"
            self._status_label.set_text(f"Downloading alignment model for {lang_name}...")
            sctx.add_class("status-downloading")
            self._progress_bar.show()
            self._progress_bar.pulse()

        elif state == _State.TRANSCRIBING:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text("Transcribing...")
            sctx.add_class("status-transcribing")

        elif state == _State.AWAITING_ALIGNMENT:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            lang_name = self._alignment_lang or "language"
            self._status_label.set_text(f"Alignment model missing for {lang_name}")
            sctx.add_class("status-awaiting")
            # Show the alignment prompt and grow window to fit
            self._alignment_box.show_all()
            self.set_resizable(True)
            self.resize(300, 280)

        elif state == _State.SUMMARIZING:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text("Generating summary...")
            sctx.add_class("status-summarizing")

        elif state == _State.LABELING_SPEAKERS:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text("Assign names to speakers")
            sctx.add_class("status-labeling")
            self._label_box.show_all()
            self.set_resizable(True)
            self.resize(340, 350)

        elif state == _State.DONE:
            self._button.set_label("● Record")
            ctx.add_class("record-btn")
            self._button.set_sensitive(True)
            if self._last_output:
                # Prefer showing PDF if it exists, otherwise .txt
                pdf_path = self._last_output.with_suffix(".pdf")
                txt_path = self._last_output.with_suffix(".txt")
                if self._last_pdf and self._last_pdf.exists():
                    self._status_label.set_text(f"Done — {self._last_pdf.name}")
                    self._open_transcript_btn.set_label("Open PDF")
                    self._open_transcript_btn.show()
                elif txt_path.exists():
                    self._status_label.set_text(f"Done — {txt_path.name}")
                    self._open_transcript_btn.set_label("Open Transcript")
                    self._open_transcript_btn.show()
                else:
                    self._status_label.set_text("Done — transcript saved")
                self._open_folder_btn.show()
            else:
                self._status_label.set_text("Done")
            sctx.add_class("status-done")

        elif state == _State.ERROR:
            self._button.set_label("● Record")
            ctx.add_class("record-btn")
            self._button.set_sensitive(True)
            self._status_label.set_text(self._error_msg or "Error")
            sctx.add_class("status-error")

    def _set_error(self, msg: str):
        self._error_msg = msg
        self._set_state(_State.ERROR)

    # ── Periodic UI update ──────────────────────────────────────────────

    def _poll_status(self) -> bool:
        """Called every 500ms by GLib timer. Returns True to keep running."""
        if self._state == _State.RECORDING:
            if self._session:
                status = self._session.status()
                self._timer_label.set_text(fmt_elapsed(status.elapsed_seconds))
                self._size_label.set_text(fmt_size(status.file_size_bytes))

                if status.failed:
                    reason = status.fail_reason or "unknown error"
                    self._set_error(f"Recording failed: {reason}")

        elif self._state == _State.DRAINING:
            if self._session:
                status = self._session.status()
                self._timer_label.set_text(fmt_elapsed(status.elapsed_seconds))
                self._size_label.set_text(fmt_size(status.file_size_bytes))
            remaining = self._drain_remaining
            sctx = self._status_label.get_style_context()
            self._status_label.set_text(f"Flushing buffer... {remaining}s")

        elif self._state == _State.DOWNLOADING_MODEL:
            self._progress_bar.pulse()

        return True  # keep polling

    # ── Cleanup ─────────────────────────────────────────────────────────

    def _on_destroy(self, _widget):
        if self._poll_id:
            GLib.source_remove(self._poll_id)
            self._poll_id = None
        # If still recording, try to stop gracefully
        if self._session and self._state in (_State.RECORDING, _State.DRAINING):
            try:
                self._session.stop()
            except Exception:
                pass
        Gtk.main_quit()


# ─── Public entry point ─────────────────────────────────────────────────────

def launch(
    *,
    output_dir: str | None = None,
    model: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: str = "auto",
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    virtual_sink: bool = False,
    mic: str | None = None,
    monitor: str | None = None,
    summarize: bool = True,
    summary_backend: str | None = None,
    summary_model: str | None = None,
) -> None:
    """Launch the Meet Recorder GTK3 window.

    Accepts the same options as ``meet run`` so the CLI can pass them through.
    """
    capture_kwargs = {
        "output_dir": output_dir,
        "mic": mic,
        "monitor": monitor,
        "virtual_sink": virtual_sink,
    }

    transcribe_kwargs = {
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "language": language,
        "hf_token": hf_token,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }

    win = MeetRecorderWindow(
        capture_kwargs, transcribe_kwargs,
        summarize=summarize, summary_backend=summary_backend,
        summary_model=summary_model,
    )
    win.show_all()
    # Hide widgets that should only appear on demand
    win._alignment_box.hide()
    win._label_box.hide()
    win._progress_bar.hide()
    win._open_transcript_btn.hide()
    win._open_folder_btn.hide()
    Gtk.main()
