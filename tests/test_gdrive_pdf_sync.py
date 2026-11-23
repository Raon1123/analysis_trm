"""Tests for scripts/gdrive_pdf_sync.sh and .claude/hooks/gdrive_pdf_sync_hook.sh (no network).

A wrapper named ``rclone`` is put first on PATH. It records argv to a log, refuses any argument that
names a real remote (``gdrive:``), can simulate failures / md5 corruption, and otherwise delegates to
the real rclone binary operating on *local* directories only -- so filter semantics are rclone's own,
but nothing ever leaves the machine.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gdrive_pdf_sync.sh"
HOOK = REPO / ".claude" / "hooks" / "gdrive_pdf_sync_hook.sh"
REAL_RCLONE = shutil.which("rclone")

pytestmark = pytest.mark.skipif(REAL_RCLONE is None, reason="rclone binary not installed")

STUB = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_RCLONE_ARGV"
for a in "$@"; do case "$a" in gdrive:*) echo "stub: refusing real remote $a" >&2; exit 99 ;; esac; done
if [ -n "${FAKE_RCLONE_FAIL:-}" ] && [ "$1" = "$FAKE_RCLONE_FAIL" ]; then echo "stub: simulated failure" >&2; exit 7; fi
if [ "$1" = "md5sum" ] && [ -n "${FAKE_RCLONE_CORRUPT:-}" ] && [ "$2" = "$FAKE_RCLONE_CORRUPT" ]; then
    "$REAL_RCLONE" "$@" | sed '1s/^[0-9a-f]\{32\}/00000000000000000000000000000000/'
    exit "${PIPESTATUS[0]}"
fi
exec "$REAL_RCLONE" "$@"
"""

INCLUDED = ["r1/30_tex/main.pdf", "r1/a.pdf", "r2/ok.pdf"]
EXCLUDED = [
    "r1/40_verify/before/x/m.pdf",
    "r1/__pycache__/z.pdf",
    "r1/scratch_build/s.pdf",
    "r1/.build/b.pdf",
    "99_Archive/o.pdf",
    "r2/99_Archive/deep.pdf",
    "13_status-reports/s.pdf",
    "r2/_before-20261103/m.pdf",
    "r2/verify/before-r2/m.r1.pdf",
    "r1/20_design/fixtures/pages.fail.pdf",
    "r2/k.before-1.pdf",
    "r1/30_tex/main.tex",
    "r1/30_tex/main.log",
]


@pytest.fixture()
def env(tmp_path: Path):
    src, dst, mon, bindir = (tmp_path / d for d in ("src", "dst", "mon", "bin"))
    for d in (src, dst, bindir):
        d.mkdir()
    for rel in INCLUDED + EXCLUDED:
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content of {rel}\n")
    native = tmp_path / "DRIVE_NATIVE.txt"
    native.write_text("# comment\n23_reports/13_status-reports\n11_governance/elsewhere\n")
    stub = bindir / "rclone"
    stub.write_text(STUB)
    stub.chmod(0o755)
    argv_log = tmp_path / "argv.log"
    argv_log.touch()
    e = dict(os.environ)
    e.update(
        PATH=f"{bindir}:{e['PATH']}",
        REAL_RCLONE=REAL_RCLONE,
        FAKE_RCLONE_ARGV=str(argv_log),
        GDRIVE_PDF_SYNC_SRC=str(src),
        GDRIVE_PDF_SYNC_REMOTE=str(dst),
        GDRIVE_PDF_SYNC_MONDIR=str(mon),
        GDRIVE_PDF_SYNC_LOCK=str(tmp_path / "sync.lock"),
        GDRIVE_PDF_SYNC_NATIVE=str(native),
        GDRIVE_PDF_SYNC_LOCK_WAIT="2",
        CUDA_VISIBLE_DEVICES="",
    )
    e.pop("GDRIVE_PDF_SYNC", None)
    return dict(src=src, dst=dst, mon=mon, env=e, argv=argv_log, tmp=tmp_path)


def run(env, *args, **kw):
    return subprocess.run(["bash", str(SCRIPT), *args], env=kw.pop("e", env["env"]),
                          capture_output=True, text=True, timeout=60, **kw)


def remote_files(env):
    return sorted(str(p.relative_to(env["dst"])) for p in env["dst"].rglob("*") if p.is_file())


def log_lines(env):
    log = env["mon"] / "gdrive_pdf_sync.log.jsonl"
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_dry_run_lists_only_included_pdfs_and_transfers_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stderr
    rec = log_lines(env)[-1]
    assert rec["mode"] == "dry-run" and rec["rc"] == 0
    assert sorted(rec["files"]) == sorted(INCLUDED)
    assert rec["n_checked"] == len(INCLUDED)
    assert remote_files(env) == []
    assert not (env["mon"] / ".gdrive_pdf_sync.last").exists()  # dry-run never advances markers
    for ex in EXCLUDED:
        assert ex not in r.stdout


def test_copy_is_idempotent_and_writes_log_and_markers(env):
    r1 = run(env, "--verify")
    assert r1.returncode == 0, r1.stderr
    assert remote_files(env) == sorted(INCLUDED)
    r2 = run(env, "--verify")
    assert r2.returncode == 0, r2.stderr
    recs = log_lines(env)
    assert [x["n_transferred"] for x in recs] == [len(INCLUDED), 0]
    assert recs[-1]["files"] == [] and recs[-1]["verify"] == f"ok:{len(INCLUDED)}"
    assert set(recs[-1]) >= {"ts", "mode", "n_transferred", "n_checked", "rc", "files"}
    assert (env["mon"] / ".gdrive_pdf_sync.last").read_text().startswith(recs[-1]["ts"])
    assert (env["mon"] / ".gdrive_pdf_sync.last_full").exists()


def test_changed_file_is_retransferred(env):
    assert run(env).returncode == 0
    time.sleep(1.1)
    (env["src"] / "r1/a.pdf").write_text("rebuilt\n")
    run(env)
    assert log_lines(env)[-1]["files"] == ["r1/a.pdf"]
    assert (env["dst"] / "r1/a.pdf").read_text() == "rebuilt\n"


def test_path_restricts_to_dir_and_to_file(env):
    r = run(env, "--path", str(env["src"] / "r1/30_tex"))
    assert r.returncode == 0, r.stderr
    assert remote_files(env) == ["r1/30_tex/main.pdf"]
    assert not (env["mon"] / ".gdrive_pdf_sync.last_full").exists()  # scoped run: full marker untouched
    r = run(env, "--path", "r2/ok.pdf")  # relative to the source tree
    assert r.returncode == 0, r.stderr
    assert remote_files(env) == ["r1/30_tex/main.pdf", "r2/ok.pdf"]
    # an excluded file named explicitly is still excluded
    run(env, "--path", "r2/k.before-1.pdf")
    assert "r2/k.before-1.pdf" not in remote_files(env)


def test_path_outside_tree_is_rejected(env):
    outside = env["tmp"] / "elsewhere"
    outside.mkdir()
    r = run(env, "--path", str(outside))
    assert r.returncode == 2
    assert env["argv"].read_text() == ""


def test_verify_mismatch_returns_1_and_keeps_markers(env):
    e = dict(env["env"], FAKE_RCLONE_CORRUPT=str(env["dst"]))
    r = run(env, "--verify", e=e)
    assert r.returncode == 1
    assert "verify FAIL" in r.stderr
    rec = log_lines(env)[-1]
    assert rec["rc"] == 1 and rec["verify"].startswith("mismatch:1/")
    assert not (env["mon"] / ".gdrive_pdf_sync.last").exists()


def test_rclone_failure_is_nonzero_and_logged(env):
    e = dict(env["env"], FAKE_RCLONE_FAIL="copy")
    r = run(env, e=e)
    assert r.returncode == 7
    assert log_lines(env)[-1]["rc"] == 7
    assert not (env["mon"] / ".gdrive_pdf_sync.last").exists()


def test_lock_prevents_overlap(env):
    lock = env["env"]["GDRIVE_PDF_SYNC_LOCK"]
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)  # simulate a concurrent run holding the lock
    try:
        r = run(env, "--no-wait")
        assert r.returncode == 75
        r = run(env)  # waits GDRIVE_PDF_SYNC_LOCK_WAIT=2 s, then gives up
        assert r.returncode == 75
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert env["argv"].read_text() == ""  # rclone never invoked while the lock was held
    assert run(env, "--no-wait").returncode == 0


def test_disable_switch(env):
    r = run(env, e=dict(env["env"], GDRIVE_PDF_SYNC="0"))
    assert r.returncode == 0 and "disabled" in r.stdout
    assert env["argv"].read_text() == ""


def test_script_has_no_destructive_rclone_verbs():
    code = "\n".join(l for l in SCRIPT.read_text().splitlines() if not l.lstrip().startswith("#"))
    for verb in ("sync", "delete", "deletefile", "purge", "rmdir", "rmdirs", "move", "moveto", "dedupe", "cleanup"):
        assert f"rclone {verb}" not in code


# ---- hook ------------------------------------------------------------------------------------------

def hook(env, payload: dict):
    e = dict(env["env"], GDRIVE_PDF_SYNC_HOOK_DRYSPAWN="1")
    t0 = time.monotonic()
    r = subprocess.run(["bash", str(HOOK)], input=json.dumps(payload), env=e,
                       capture_output=True, text=True, timeout=10)
    return r, time.monotonic() - t0


@pytest.mark.skipif(not HOOK.exists(), reason="hook not installed")
def test_hook_spawns_only_when_pdf_newer_than_full_marker(env):
    stop = {"hook_event_name": "Stop"}
    r, _ = hook(env, stop)  # no marker yet -> spawn
    assert r.returncode == 0 and "spawn" in r.stdout
    assert run(env).returncode == 0  # full-tree run writes .last_full
    r, dt = hook(env, stop)
    assert r.returncode == 0 and "spawn" not in r.stdout
    assert dt < 1.0
    time.sleep(1.1)
    (env["src"] / "r2/ok.pdf").write_text("new build\n")
    r, _ = hook(env, stop)
    assert "spawn" in r.stdout
    # an excluded PDF being newer does not trigger
    assert run(env).returncode == 0
    time.sleep(1.1)
    (env["src"] / "r1/40_verify/before/x/m.pdf").write_text("snapshot\n")
    r, _ = hook(env, stop)
    assert "spawn" not in r.stdout


@pytest.mark.skipif(not HOOK.exists(), reason="hook not installed")
def test_hook_posttooluse_ignores_non_build_commands(env):
    post = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": "ls -la"}}
    r, _ = hook(env, post)  # PDFs are newer than the (absent) marker, but the command is not a build
    assert r.returncode == 0 and "spawn" not in r.stdout
    post["tool_input"]["command"] = "bash lab/23_reports/11_lab-reports/x/30_tex/build.sh"
    r, _ = hook(env, post)
    assert r.returncode == 0 and "spawn" in r.stdout
