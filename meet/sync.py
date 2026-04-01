"""Meeting sync module — push meeting artifacts to a configured Git repository.

Detects whether a completed meeting matches a configured schedule based on
start time, then copies the relevant files into a local clone of the target
repository and pushes to GitHub via gh CLI.

Schedule and repo config: ~/.config/meet/sync_config.json
Repo clone:               ~/.local/share/meet/<repo-name>/
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

SYNC_CONFIG_PATH = Path.home() / ".config" / "meet" / "sync_config.json"
CLONE_BASE_DIR = Path.home() / ".local" / "share" / "meet"
MEETINGS_SUBDIR = "meetings"

# ─── Files to push (by suffix) ───────────────────────────────────────────────

PUSH_SUFFIXES = {".md", ".txt", ".pdf", ".srt", ".json"}
# Exclude session metadata and large raw files
EXCLUDE_PATTERNS = {
    ".session.json",
    ".summary.meta.json",
    ".ffmpeg.log",
}


# ─── Default schedule config ─────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "repo_url": "",
    "meetings": [],
}

EXAMPLE_CONFIG: dict[str, Any] = {
    "repo_url": "https://github.com/yourorg/meetings.git",
    "meetings": [
        {
            "name": "Weekly Sync",
            "folder": "weekly-sync",
            "days": [0],
            "hour_utc": 14,
            "window_minutes": 60,
        },
        {
            "name": "Dev Standup",
            "folder": "dev-standup",
            "days": [0, 1, 2, 3, 4],
            "hour_utc": 8,
            "window_minutes": 60,
        },
    ],
}


def load_sync_config() -> dict[str, Any]:
    """Load sync config from disk. Returns empty defaults if not found."""
    if not SYNC_CONFIG_PATH.exists():
        return DEFAULT_CONFIG
    try:
        return json.loads(SYNC_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load sync config: %s — using defaults", exc)
        return DEFAULT_CONFIG


def save_sync_config(config: dict[str, Any]) -> None:
    """Save sync config to disk."""
    SYNC_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def is_sync_configured() -> bool:
    """Return True if a repo_url is configured for sync."""
    config = load_sync_config()
    return bool(config.get("repo_url"))


def _repo_name_from_url(repo_url: str) -> str:
    """Extract a directory name from a Git URL.

    e.g. 'https://github.com/org/my-repo.git' -> 'my-repo'
    """
    name = repo_url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "sync-repo"


# ─── Schedule matching ────────────────────────────────────────────────────────

@dataclass
class MeetingMatch:
    name: str
    folder: str


def detect_meeting_type(session_dir: Path) -> MeetingMatch | None:
    """Check if a session matches any configured scheduled meeting.

    Reads started_at from session.json and compares against the schedule.

    Returns a MeetingMatch if matched, None otherwise.
    """
    session_json = _find_session_json(session_dir)
    if not session_json:
        return None

    try:
        meta = json.loads(session_json.read_text(encoding="utf-8"))
        started_at_str = meta.get("started_at")
        if not started_at_str:
            return None
        # Parse ISO format — may or may not have timezone info
        started_at = datetime.fromisoformat(started_at_str)
        # Treat naive datetimes as local time, convert to UTC
        if started_at.tzinfo is None:
            import time as _time
            local_offset = timedelta(seconds=-_time.timezone)
            started_at = started_at - local_offset
            started_at = started_at.replace(tzinfo=timezone.utc)
        else:
            started_at = started_at.astimezone(timezone.utc)
    except Exception as exc:
        log.debug("Could not parse session start time: %s", exc)
        return None

    config = load_sync_config()
    weekday = started_at.weekday()   # 0=Monday
    hour = started_at.hour
    minute = started_at.minute
    meeting_minutes = hour * 60 + minute

    for m in config.get("meetings", []):
        if weekday not in m.get("days", []):
            continue
        scheduled_minutes = m["hour_utc"] * 60
        window = m.get("window_minutes", 60)
        if abs(meeting_minutes - scheduled_minutes) <= window:
            return MeetingMatch(name=m["name"], folder=m["folder"])

    return None


def _find_session_json(session_dir: Path) -> Path | None:
    """Find the session.json file in a session directory."""
    matches = list(session_dir.glob("*.session.json"))
    return matches[0] if matches else None


# ─── Repo management ──────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising on non-zero exit if check=True."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result


def _get_clone_dir() -> Path:
    """Return the local clone directory for the configured repo."""
    config = load_sync_config()
    repo_url = config.get("repo_url", "")
    if not repo_url:
        raise RuntimeError(
            "No repo_url configured for sync. "
            f"Set it in {SYNC_CONFIG_PATH} or run: meet sync --init-config"
        )
    return CLONE_BASE_DIR / _repo_name_from_url(repo_url)


def ensure_repo_cloned(progress_callback=None) -> Path:
    """Ensure the configured repo is cloned locally. Clone if not present, pull if it is.

    Returns the path to the local clone.
    """
    config = load_sync_config()
    repo_url = config.get("repo_url", "")
    if not repo_url:
        raise RuntimeError(
            "No repo_url configured for sync. "
            f"Set it in {SYNC_CONFIG_PATH} or run: meet sync --init-config"
        )

    clone_dir = CLONE_BASE_DIR / _repo_name_from_url(repo_url)

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            log.info(msg)

    if not clone_dir.exists():
        _log(f"Cloning {repo_url}...")
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(clone_dir)])
        _log("Clone complete.")
    else:
        # Pull latest
        _run(["git", "pull", "--ff-only"], cwd=clone_dir, check=False)

    return clone_dir


def _collect_files(session_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (source_path, dest_filename) pairs to push from a session directory.

    Files are renamed to descriptive names (summary.md, transcript.txt, etc.)
    since the date lives in the parent folder name.
    """
    result = []
    for f in sorted(session_dir.iterdir()):
        if not f.is_file():
            continue
        if any(f.name.endswith(pat) for pat in EXCLUDE_PATTERNS):
            continue
        if f.suffix not in PUSH_SUFFIXES:
            continue

        # Map to a descriptive destination filename
        if f.name.endswith(".summary.md") or (f.suffix == ".md" and ".summary." not in f.name):
            dest_name = "summary.md"
        elif f.suffix == ".txt":
            dest_name = "transcript.txt"
        elif f.suffix == ".pdf":
            dest_name = "transcript.pdf"
        elif f.suffix == ".srt":
            dest_name = "transcript.srt"
        elif f.suffix == ".json":
            dest_name = "transcript.json"
        else:
            dest_name = f.name  # fallback: keep original name

        result.append((f, dest_name))
    return result


def _date_from_session(session_dir: Path) -> str:
    """Extract YYYY-MM-DD from the session folder name or session.json."""
    # Try parsing from folder name: meeting-YYYYMMDD-HHMMSS_*
    name = session_dir.name
    parts = name.split("-")
    if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
        d = parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    # Fallback: read from session.json
    sj = _find_session_json(session_dir)
    if sj:
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
            started = meta.get("started_at", "")[:10]
            if len(started) == 10:
                return started
        except Exception:
            pass

    return datetime.now().strftime("%Y-%m-%d")


def _ensure_readme(meetings_dir: Path) -> None:
    """Create meetings/README.md from the sync config if it doesn't exist."""
    readme = meetings_dir / "README.md"
    if readme.exists():
        return

    config = load_sync_config()
    meetings = config.get("meetings", [])
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    lines = [
        "# Meeting Notes\n\n",
        "Automatically published meeting transcripts, summaries, and PDFs.\n\n",
    ]

    if meetings:
        lines.append("## Recurring Meetings\n\n")
        lines.append("| Folder | Meeting | Schedule |\n")
        lines.append("|--------|---------|----------|\n")
        for m in meetings:
            days = ", ".join(day_names[d] for d in m.get("days", []))
            lines.append(
                f"| `{m['folder']}/` | {m['name']} | {days} {m['hour_utc']:02d}:00 UTC |\n"
            )
        lines.append("\n")

    lines.append(
        "## File Types\n\n"
        "| Extension | Contents |\n"
        "|-----------|----------|\n"
        "| `.md` | AI-generated meeting summary |\n"
        "| `.txt` | Full transcript with timestamps |\n"
        "| `.pdf` | Formatted PDF |\n"
        "| `.srt` | Subtitle/caption file |\n"
        "| `.json` | Full transcript data (programmatic access) |\n"
    )

    readme.write_text("".join(lines), encoding="utf-8")


def sync_session(
    session_dir: Path,
    meeting_type: MeetingMatch,
    progress_callback=None,
) -> list[Path]:
    """Copy session files into the configured repo and push to GitHub.

    Args:
        session_dir: Path to the local meeting session directory.
        meeting_type: MeetingMatch identifying folder and name.
        progress_callback: Optional callable(str) for status messages.

    Returns:
        List of files that were pushed.

    Raises:
        RuntimeError: If git operations fail or sync is not configured.
    """
    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            log.info(msg)

    repo = ensure_repo_cloned(progress_callback=progress_callback)

    date_str = _date_from_session(session_dir)
    # Each meeting gets its own folder: date first so folders sort chronologically
    target_dir = repo / MEETINGS_SUBDIR / f"{date_str}_{meeting_type.folder}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure README exists
    _ensure_readme(repo / MEETINGS_SUBDIR)

    # Collect and copy files with descriptive names
    source_files = _collect_files(session_dir)
    if not source_files:
        raise RuntimeError(f"No pushable files found in {session_dir}")

    copied: list[Path] = []
    for src, dest_name in source_files:
        dest = target_dir / dest_name
        shutil.copy2(src, dest)
        copied.append(dest)
        _log(f"  Staged: {dest.relative_to(repo)}")

    # Also stage the README
    _run(["git", "add", str((repo / MEETINGS_SUBDIR / "README.md").relative_to(repo))],
         cwd=repo)

    # Stage all copied files
    rel_paths = [str(p.relative_to(repo)) for p in copied]
    _run(["git", "add"] + rel_paths, cwd=repo)

    # Check if there's anything to commit
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    if not status.stdout.strip():
        _log("  Nothing to commit — already up to date.")
        return copied

    # Commit
    commit_msg = f"meetings: add {meeting_type.name} {date_str}"
    _run(["git", "commit", "-m", commit_msg], cwd=repo)
    _log(f"  Committed: {commit_msg}")

    # Push
    _log("  Pushing...")
    _run(["git", "push"], cwd=repo)
    _log(f"  Pushed {len(copied)} file(s).")

    return copied


def maybe_sync_session(
    session_dir: Path,
    progress_callback=None,
) -> MeetingMatch | None:
    """Detect meeting type and sync if it matches a scheduled meeting.

    This is the main entry point called from the GUI pipeline.
    If sync is not configured (no repo_url), silently returns None.

    Returns the MeetingMatch if synced, None if not a scheduled meeting.
    """
    if not is_sync_configured():
        return None

    match = detect_meeting_type(session_dir)
    if match is None:
        return None

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    _log(f"Scheduled meeting detected: {match.name} — syncing...")
    try:
        sync_session(session_dir, match, progress_callback=progress_callback)
    except Exception as exc:
        _log(f"Sync failed (meeting saved locally): {exc}")
        log.exception("Sync failed for %s", session_dir)

    return match
