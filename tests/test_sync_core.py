"""Tests for the core sync logic in millet.sync (0.13.0 review fixes).

Covers the previously untested operationally sensitive paths:

* credential redaction in errors and progress messages
* folder-slug validation (path traversal via --meeting-type / config)
* maybe_sync_session failure semantics (no more "failed but reported synced")
* DST-aware naive-timestamp conversion + midnight window wrap
* _collect_files collision handling
* git pull --rebase failure leaves no half-applied rebase behind
* sync_session end-to-end against a local bare git repo
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest

from millet import sync

# ─── _redact ─────────────────────────────────────────────────────────────────


def test_redact_strips_userinfo():
    assert (
        sync._redact("Cloning https://user:tok3n@github.com/org/repo.git...")
        == "Cloning https://***@github.com/org/repo.git..."
    )


def test_redact_leaves_clean_urls_alone():
    s = "Cloning https://github.com/org/repo.git..."
    assert sync._redact(s) == s


def test_run_redacts_credentials_in_error():
    with pytest.raises(RuntimeError) as ei:
        sync._run(["false", "https://user:s3cret@github.com/org/repo.git"])
    msg = str(ei.value)
    assert "s3cret" not in msg
    assert "://***@" in msg


def test_run_network_redacts_credentials_in_error(monkeypatch):
    import socket

    # Keep the DNS gate from waiting on real lookups.
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(RuntimeError) as ei:
        sync._run_network(
            ["false", "https://user:s3cret@github.com/org/repo.git"],
            retries=1, delay=0,
        )
    msg = str(ei.value)
    assert "s3cret" not in msg
    assert "://***@" in msg


# ─── _validate_folder_slug ───────────────────────────────────────────────────


@pytest.mark.parametrize("good", ["weekly-sync", "dev-standup", "a", "Team_1.x"])
def test_folder_slug_accepts_safe_segments(good):
    assert sync._validate_folder_slug(good, "test") == good


@pytest.mark.parametrize(
    "bad", ["../../etc", "a/b", "/abs", "", "-lead", "..", "a\\b", "x" * 70]
)
def test_folder_slug_rejects_traversal_and_junk(bad):
    with pytest.raises(RuntimeError, match="Invalid meeting folder"):
        sync._validate_folder_slug(bad, "test")


def test_sync_session_rejects_traversal_folder(tmp_path):
    """sync_session must refuse a hostile folder before doing ANY git or
    filesystem work — previously ../../foo copied artifacts outside the
    clone."""
    match = sync.MeetingMatch(name="Evil", folder="../../evil")
    with pytest.raises(RuntimeError, match="Invalid meeting folder"):
        sync.sync_session(tmp_path, match)


def test_cli_rejects_traversal_meeting_type(tmp_path):
    from click.testing import CliRunner

    from millet.cli.sync import sync as sync_cmd

    result = CliRunner().invoke(
        sync_cmd, ["--meeting-type", "../../evil", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "Invalid meeting folder" in result.output


# ─── maybe_sync_session ──────────────────────────────────────────────────────


def _candidate():
    return sync.SyncCandidate(
        match=sync.MeetingMatch(name="Weekly", folder="weekly"),
        team_members_found=[],
    )


def test_maybe_sync_returns_none_when_push_fails(monkeypatch, tmp_path):
    """A failed sync must NOT be reported as synced (GUI auto-sync treated
    the returned match as success)."""
    monkeypatch.setattr(sync, "is_sync_configured", lambda team=None: True)
    monkeypatch.setattr(
        sync, "check_sync_candidate", lambda sd, team=None: _candidate()
    )

    def boom(*a, **k):
        raise RuntimeError("push rejected")

    monkeypatch.setattr(sync, "sync_session", boom)
    assert sync.maybe_sync_session(tmp_path) is None


def test_maybe_sync_returns_match_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "is_sync_configured", lambda team=None: True)
    monkeypatch.setattr(
        sync, "check_sync_candidate", lambda sd, team=None: _candidate()
    )
    monkeypatch.setattr(sync, "sync_session", lambda *a, **k: [tmp_path / "x"])
    match = sync.maybe_sync_session(tmp_path)
    assert match is not None and match.folder == "weekly"


# ─── detect_meeting_type: DST + midnight wrap ───────────────────────────────


def _write_session(tmp_path: Path, started_at: str, title: str = "") -> Path:
    sdir = tmp_path / "session"
    sdir.mkdir(exist_ok=True)
    meta = {"started_at": started_at}
    if title:
        meta["title"] = title
    (sdir / "x.session.json").write_text(json.dumps(meta), encoding="utf-8")
    return sdir


@pytest.fixture
def ny_tz(monkeypatch):
    """Run the test in America/New_York so DST is observable."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_naive_timestamp_converted_dst_aware(ny_tz, monkeypatch, tmp_path):
    """2026-07-01 (EDT, UTC-4): naive 10:00 local == 14:00 UTC.  The old
    time.timezone arithmetic used the winter offset (UTC-5) year-round,
    computing 15:00 UTC and missing a 14:00-UTC schedule window."""
    sdir = _write_session(tmp_path, "2026-07-01T10:00:00.000000")
    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {
            "repo_url": "x",
            "meetings": [{
                "name": "W", "folder": "w",
                "days": [2],  # 2026-07-01 is a Wednesday
                "hour_utc": 14, "window_minutes": 30,
            }],
        },
    )
    match = sync.detect_meeting_type(sdir)
    assert match is not None and match.folder == "w"


def test_window_wraps_midnight(monkeypatch, tmp_path):
    """23:40 UTC is 20 minutes from an hour_utc=0 schedule, not 1420."""
    sdir = _write_session(tmp_path, "2026-06-30T23:40:00+00:00")
    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {
            "repo_url": "x",
            "meetings": [{
                "name": "M", "folder": "midnight",
                "days": [1],  # 2026-06-30 is a Tuesday
                "hour_utc": 0, "window_minutes": 30,
            }],
        },
    )
    match = sync.detect_meeting_type(sdir)
    assert match is not None and match.folder == "midnight"


# ─── _collect_files collisions ───────────────────────────────────────────────


def test_collect_files_keeps_original_name_on_collision(tmp_path):
    sdir = tmp_path / "s"
    sdir.mkdir()
    (sdir / "meeting.summary.md").write_text("real summary")
    (sdir / "notes.md").write_text("stray notes")
    (sdir / "a.txt").write_text("transcript a")
    (sdir / "b.txt").write_text("transcript b")

    pairs = sync._collect_files(sdir)
    dests = [d for _, d in pairs]
    # No destination is used twice — nothing gets silently clobbered.
    assert len(dests) == len(set(dests)), dests
    assert "summary.md" in dests
    assert "transcript.txt" in dests
    # The colliding files kept their original names.
    assert "notes.md" in dests
    assert "b.txt" in dests


# ─── ensure_repo_cloned: rebase abort on pull failure ────────────────────────


def test_pull_failure_aborts_rebase(monkeypatch, tmp_path):
    clone = tmp_path / "clone"
    (clone / ".git" / "info").mkdir(parents=True)

    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {"repo_url": "https://x/r.git"},
    )
    monkeypatch.setattr(sync, "_clone_dir_for", lambda url, team: clone)

    run_calls: list[list[str]] = []
    real_completed = subprocess.CompletedProcess

    def fake_run(cmd, cwd=None, check=True):
        run_calls.append(list(cmd))
        return real_completed(cmd, 0, stdout="", stderr="")

    def fake_run_network(cmd, cwd=None, **kw):
        raise RuntimeError("rebase conflict")

    monkeypatch.setattr(sync, "_run", fake_run)
    monkeypatch.setattr(sync, "_run_network", fake_run_network)

    with pytest.raises(RuntimeError, match="rebase conflict"):
        sync.ensure_repo_cloned()

    assert ["git", "rebase", "--abort"] in run_calls, (
        "a failed pull --rebase must abort any half-applied rebase so the "
        "next sync isn't wedged at the uncommitted-changes guard"
    )


# ─── sync_session end-to-end (local git) ────────────────────────────────────


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


@pytest.fixture
def git_identity(monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")
    # Never pick up the developer's git config / credential helpers.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def local_remote(tmp_path, git_identity):
    """A seeded local bare repo usable as a sync target."""
    bare = tmp_path / "remote.git"
    _git("init", "--bare", "--initial-branch=main", str(bare))
    seed = tmp_path / "seed"
    _git("clone", str(bare), str(seed))
    (seed / "README.md").write_text("seed\n")
    _git("add", ".", cwd=seed)
    _git("commit", "-m", "init", cwd=seed)
    _git("push", "origin", "HEAD:main", cwd=seed)
    return bare


def _make_session(tmp_path: Path, name: str, session_id: str) -> Path:
    sdir = tmp_path / name
    sdir.mkdir()
    (sdir / f"{name}.session.json").write_text(
        json.dumps({
            "started_at": "2026-07-06T10:00:00+00:00",
            "session_id": session_id,
        }),
        encoding="utf-8",
    )
    (sdir / f"{name}.summary.md").write_text("# Summary\n")
    (sdir / f"{name}.txt").write_text("transcript\n")
    return sdir


def test_sync_session_pushes_to_local_remote(monkeypatch, tmp_path, local_remote):
    monkeypatch.setattr(sync, "CLONE_BASE_DIR", tmp_path / "clones")
    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {
            "repo_url": str(local_remote), "meetings": [],
        },
    )

    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01TESTULID")
    match = sync.MeetingMatch(name="Weekly", folder="weekly")
    copied = sync.sync_session(sdir, match, progress_callback=lambda m: None)
    assert copied

    # Verify the remote actually received the commit + files.
    check = tmp_path / "check"
    _git("clone", str(local_remote), str(check))
    meeting_dir = check / "meetings" / "2026-07-06_weekly"
    assert meeting_dir.is_dir()
    assert (meeting_dir / "summary.md").read_text() == "# Summary\n"
    assert (meeting_dir / "transcript.txt").read_text() == "transcript\n"
    # The local-only session-id marker is never pushed.
    assert not (meeting_dir / sync.SESSION_ID_MARKER).exists()
    # Re-sync of the same session is idempotent (no crash, same folder).
    copied2 = sync.sync_session(sdir, match, progress_callback=lambda m: None)
    assert copied2


def test_sync_session_disambiguates_different_session(
    monkeypatch, tmp_path, local_remote
):
    """Two distinct sessions mapping to the same date+folder must not
    overwrite each other (collision guard, 0.12.5)."""
    monkeypatch.setattr(sync, "CLONE_BASE_DIR", tmp_path / "clones")
    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {
            "repo_url": str(local_remote), "meetings": [],
        },
    )
    match = sync.MeetingMatch(name="Weekly", folder="weekly")

    s1 = _make_session(tmp_path, "meeting-20260706-100000", "01AAAAAAAAAA")
    sync.sync_session(s1, match, progress_callback=lambda m: None)

    s2dir = tmp_path / "second"
    s2dir.mkdir()
    s2 = _make_session(s2dir, "meeting-20260706-110000", "01BBBBBBBBBB")
    # Same date prefix in the folder name → same base target dir.
    sync.sync_session(s2, match, progress_callback=lambda m: None)

    clone = sync._clone_dir_for(str(local_remote), None)
    base = clone / "meetings" / "2026-07-06_weekly"
    assert base.is_dir()
    suffixed = list(clone.glob("meetings/2026-07-06_weekly-*"))
    assert suffixed, "second session must land in a disambiguated folder"


# ─── attachments passthrough ─────────────────────────────────────────────────


def _attach(sdir: Path, name: str, content: bytes = b"data") -> Path:
    adir = sdir / sync.ATTACHMENTS_SUBDIR
    adir.mkdir(exist_ok=True)
    p = adir / name
    p.write_bytes(content)
    return p


def test_collect_attachments_bypasses_allowlist_and_rename_map(tmp_path):
    """Attachment names are the user's and must survive verbatim — the
    suffix-keyed rename map would turn slides.pdf into transcript.pdf and
    PUSH_SUFFIXES would drop images and office documents entirely."""
    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01ATTACH")
    _attach(sdir, "slides.pdf")
    _attach(sdir, "diagram.png")
    _attach(sdir, "agenda.pptx")

    dests = dict((d, s) for s, d in sync._collect_files(sdir))
    # The session's own transcript still gets its descriptive name.
    assert "transcript.txt" in dests
    # Attachments keep theirs, under the subdir, whatever the suffix.
    assert "attachments/slides.pdf" in dests
    assert "attachments/diagram.png" in dests
    assert "attachments/agenda.pptx" in dests
    # Nothing was renamed into the pipeline's namespace.
    assert "transcript.pdf" not in dests


def test_collect_attachments_skips_symlinks_dotfiles_and_subdirs(tmp_path):
    """These pairs get copied into a git clone: a symlink could point
    anywhere on the host."""
    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01ATTACH")
    _attach(sdir, "keep.pdf")
    _attach(sdir, ".hidden")
    outside = tmp_path / "secret.txt"
    outside.write_text("ssh key")
    (sdir / sync.ATTACHMENTS_SUBDIR / "link.txt").symlink_to(outside)
    (sdir / sync.ATTACHMENTS_SUBDIR / "nested").mkdir()

    dests = [d for _, d in sync._collect_files(sdir)]
    assert "attachments/keep.pdf" in dests
    assert not any(
        d.endswith(("link.txt", ".hidden", "nested")) for d in dests
    ), dests


def test_collect_files_unchanged_without_attachments_dir(tmp_path):
    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01ATTACH")
    dests = [d for _, d in sync._collect_files(sdir)]
    assert dests == ["summary.md", "transcript.txt"]


def test_collect_attachments_caps_count(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "MAX_ATTACHMENTS", 2)
    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01ATTACH")
    for i in range(5):
        _attach(sdir, f"f{i}.png")

    attached = [d for _, d in sync._collect_files(sdir) if d.startswith("attachments/")]
    assert attached == ["attachments/f0.png", "attachments/f1.png"]


def test_collect_attachments_caps_total_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(sync, "MAX_ATTACHMENTS_BYTES", 20)
    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01ATTACH")
    _attach(sdir, "a.png", b"x" * 15)
    _attach(sdir, "b.png", b"x" * 15)

    attached = [d for _, d in sync._collect_files(sdir) if d.startswith("attachments/")]
    assert attached == ["attachments/a.png"]


def test_sync_session_pushes_attachments_subdir(monkeypatch, tmp_path, local_remote):
    monkeypatch.setattr(sync, "CLONE_BASE_DIR", tmp_path / "clones")
    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {
            "repo_url": str(local_remote), "meetings": [],
        },
    )

    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01TESTULID")
    _attach(sdir, "slides.pdf", b"%PDF-1.4 slides")
    _attach(sdir, "photo of board.png", b"\x89PNG board")
    match = sync.MeetingMatch(name="Weekly", folder="weekly")
    sync.sync_session(sdir, match, progress_callback=lambda m: None)

    check = tmp_path / "check"
    _git("clone", str(local_remote), str(check))
    meeting_dir = check / "meetings" / "2026-07-06_weekly"
    assert (meeting_dir / "attachments" / "slides.pdf").read_bytes() == b"%PDF-1.4 slides"
    assert (
        meeting_dir / "attachments" / "photo of board.png"
    ).read_bytes() == b"\x89PNG board"
    # The pipeline's own artifacts are untouched by the attachment pass.
    assert (meeting_dir / "transcript.txt").read_text() == "transcript\n"
    assert not (meeting_dir / "transcript.pdf").exists()


# ─── sanity: dates used above ────────────────────────────────────────────────


def test_fixture_dates_are_correct_weekdays():
    assert datetime(2026, 7, 1).weekday() == 2   # Wednesday
    assert datetime(2026, 6, 30).weekday() == 1  # Tuesday
