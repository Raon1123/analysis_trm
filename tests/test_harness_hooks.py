"""Tests for tools/harness_hooks (run each script as a subprocess)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parent.parent / "tools" / "harness_hooks"


def run(script, stdin, env_extra=None, project=None):
    env = dict(os.environ)
    env.pop("HARNESS_HOOKS", None)
    env["HARNESS_GUARD_FAKE_WORKERS"] = ""
    if project is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project)
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, str(HOOKS / script)], input=stdin, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def guard(cmd, env_extra=None, project=None):
    payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": cmd}})
    return run("bash_guard.py", payload, env_extra, project)


def rule_of(out):
    if not out:
        return None
    h = out["hookSpecificOutput"]
    assert h["permissionDecision"] == "deny" and "updatedInput" not in h
    r = h["permissionDecisionReason"]
    assert r.startswith("[harness-guard:")
    return r[len("[harness-guard:"):r.index("]")], r


@pytest.mark.parametrize("cmd,rule", [
    ("rg -t py foo src", "rg"),
    ("rtk rg -t py foo 2>/dev/null", "rg"),
    ("rtk rg foo", "rg"),
    ("rtk proxy rg foo", "rg"),
    ("cd x && rtk rg foo", "rg"),
    ("FOO=1 rtk rg foo", "rg"),
    ("x | rtk rg foo", "rg"),
    ("uv run pytest tests/", "pytest-gpu"),
    ("rtk uv run --project . pytest -q", "pytest-gpu"),
    ("pytest tests/", "pytest-gpu"),
    ("uv run python -m pytest tests", "pytest-gpu"),
    ("uv run --extra dev pytest tests", "pytest-gpu"),
    ("rg --files", "rg"),
    ("rg --files | grep x", "rg"),
    ("rg -g '*.py' foo", "rg"),
    ("rg --glob=*.py foo .", "rg"),
    ("rg -tpy foo", "rg"),
    ("rg -nU foo", "rg"),
    ("rg -uu foo", "rg"),
    ("rg --hidden foo", "rg"),
    ("uv run --upgrade-package numpy pytest tests", "pytest-gpu"),
])
def test_rules_fire(cmd, rule):
    got = rule_of(guard(cmd))
    assert got and got[0] == rule
    assert "guard-ok:" + rule in got[1]


def test_queue_dup_fires_with_workers():
    env = {"HARNESS_GUARD_FAKE_WORKERS": "111,222"}
    for cmd in ("scripts/queue_run.sh", "GPUS='4 5' nohup scripts/queue_run.sh &", "bash scripts/queue_run.sh"):
        got = rule_of(guard(cmd, env))
        assert got and got[0] == "queue-dup" and "111" in got[1], cmd


@pytest.mark.parametrize("cmd", [
    "rg -t py foo  # guard-ok:rg",
    "uv run pytest tests/ # guard-ok:pytest-gpu",
])
def test_bypass_markers(cmd):
    assert guard(cmd) == {}


def test_bypass_queue_dup():
    assert guard("scripts/queue_run.sh # guard-ok:queue-dup", {"HARNESS_GUARD_FAKE_WORKERS": "5"}) == {}


def test_marker_is_per_rule():
    assert rule_of(guard("rg -t py foo # guard-ok:pytest-gpu"))[0] == "rg"


@pytest.mark.parametrize("cmd", [
    "uv run python x.py",
    "rtk grep -rn foo .",
    'echo "python3 is old"',
    "cat <<'EOF'\npython3 x\nEOF",
    "cat <<'EOF'\nrg -t py foo\nEOF",
    ".venv/bin/python x.py",
    "python3 --version",
    "python -V",
    "python3 -c 1",
    "python3 x.py",
    "python x.py",
    "FOO=1 rtk python3 x.py",
    "python3 - <<'EOF'\nprint(1)\nEOF",
    "python3 --version 2>&1",
    "python3 -V 2>/dev/null",
    'N="$(grep -cE "rg |python3 -" f)"; echo $N',
    "git commit -m \"$(cat <<'EOF'\nfix\" odd quote\nrg -t py foo\npytest tests\nEOF\n)\"",
    "uv run --extra pytest python x.py",
    "uv run --upgrade-package pytest python x.py",
    "uv run -P pytest python x.py",
    "uv run --reinstall-package pytest python x.py",
    "uv run --extra-index-url https://x/pytest python x.py",
    "uv run --index-url https://x/pytest python x.py",
    "uv run --refresh-package pytest python x.py",
    "uv run --no-binary-package pytest python x.py",
    "uv run -w pytest python x.py",
    "uv run --config-setting pytest=1 python x.py",
    'echo "${var:-"default; rg -t py x"}"',
    'echo "${var:-x}"; rg -t py foo',
    'echo "unterminated; rg -t py foo',
    "scripts/queue_run.sh dry-run",
    "GPUS='4 5' scripts/queue_run.sh dry-run 2>&1 | head -12",
    'CUDA_VISIBLE_DEVICES="" uv run pytest tests/',
    "grep rg file",
    "rtk git grep foo",
    'git commit -m "use rg"',
    "echo 'a; rg -t py b'",
    "ls # python3 is old",
    "scripts/queue_run.sh status",
    "/usr/bin/python3 x.py",
])
def test_non_firing(cmd):
    assert guard(cmd, {"HARNESS_GUARD_FAKE_WORKERS": "111"}) == {}


def test_queue_start_variants_still_fire():
    env = {"HARNESS_GUARD_FAKE_WORKERS": "9"}
    for cmd in ("scripts/queue_run.sh", "scripts/queue_run.sh start", "bash scripts/queue_run.sh &"):
        assert rule_of(guard(cmd, env))[0] == "queue-dup", cmd


def test_oversize_command_skips_rules():
    assert guard("rg -t py foo " + "x" * 100001) == {}
    assert guard("echo '" + "a" * 150000 + "'; rg -t py foo") == {}


def test_json_output_is_ascii():
    p = subprocess.run([sys.executable, str(HOOKS / "session_brief.py")], input=b'{"source":"startup"}',
                       stdout=subprocess.PIPE, timeout=20,
                       env=dict(os.environ, CLAUDE_PROJECT_DIR=str(HOOKS)))
    assert p.stdout.decode("ascii")
    assert json.loads(p.stdout)["hookSpecificOutput"]["additionalContext"].endswith("근거가 아니다.")


def test_queue_start_without_workers_is_fine():
    assert guard("scripts/queue_run.sh") == {}


def test_non_bash_tool_and_malformed():
    assert run("bash_guard.py", json.dumps({"tool_name": "Write", "tool_input": {"command": "rg -t py x"}})) == {}
    for bad in ("", "not json", "[1,2]", '{"tool_input": 5}', '{"tool_name":"Bash","tool_input":{"command":7}}'):
        assert run("bash_guard.py", bad) == {}
        assert run("sh_syntax_check.py", bad) == {}
    for bad in ("", "not json", "[1,2]"):
        assert run("session_brief.py", bad) == {}


def test_disable_switches(tmp_path):
    assert guard("rg -t py foo", {"HARNESS_HOOKS": "0"}) == {}
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "harness_hooks.disabled").write_text("")
    assert guard("rg -t py foo", project=tmp_path) == {}
    out = run("session_brief.py", '{"source":"startup"}', project=tmp_path)
    assert out == {}


STATE = """---
updated: 2026-01-01 10:00
---
# STATE
## 0. x
| not | a track |
## 1. tracks

| code | track | recent | next | doc |
|---|---|---|---|---|
| **AAA** | first | 11-06 10:00 **needs-decisions**: long prose | n | `31_handoffs/11_lab-handoffs/2026-11-06_aaa.md` |
| **BBB** | second | 10-27 09:15 **EXP-020** done **ready-for-resume** | n | `31_handoffs/x/bbb.md` |
| ~~CCC~~ | struck | 09-01 01:01 **DONE** | n | `31_handoffs/x/ccc.md` |
| DDD | no status | none | n | none |
## 2. next
| ZZZ | outside | 01-01 01:01 | | |
"""


def make_project(tmp_path, state=STATE):
    if state is not None:
        d = tmp_path / "lab" / "11_governance" / "11_lab-root"
        d.mkdir(parents=True)
        (d / "STATE.md").write_text(state, encoding="utf-8")
    q = tmp_path / "scripts" / "queue"
    for name, n in (("jobs", 3), ("processing", 1), ("done", 2)):
        (q / name).mkdir(parents=True)
        for i in range(n):
            (q / name / ("%d.job" % i)).write_text("x")
    return tmp_path


def brief(project, source="startup"):
    out = run("session_brief.py", json.dumps({"hook_event_name": "SessionStart", "source": source}), project=project)
    h = out["hookSpecificOutput"]
    assert h["hookEventName"] == "SessionStart"
    return h["additionalContext"]


def test_session_brief(tmp_path):
    b = brief(make_project(tmp_path))
    assert len(b) <= 6000
    assert b.startswith("[session-brief] ") and "source=startup" in b
    assert "AAA | needs-decisions | 11-06 10:00 | 31_handoffs/11_lab-handoffs/2026-11-06_aaa.md" in b
    assert "BBB | ready-for-resume | 10-27 09:15 | 31_handoffs/x/bbb.md" in b
    assert "DDD | ? | n/a | n/a" in b
    assert "CCC" not in b and "ZZZ" not in b and "long prose" not in b
    assert "STALE" in b
    assert "jobs=3 processing=1 done=2 failed=n/a" in b
    assert b.rstrip().endswith("이 brief는 요약이지 근거가 아니다.")
    assert "source=compact" in brief(tmp_path, "compact")


def test_session_brief_missing_state_and_queue(tmp_path):
    b = brief(tmp_path)
    assert "STATE.md: n/a" in b and "jobs=n/a" in b


def test_session_brief_truncates(tmp_path):
    rows = "\n".join("| **T%03d** | t | 01-01 01:01 **in-progress** | n | `31_handoffs/%s.md` |" % (i, "p" * 150)
                     for i in range(100))
    b = brief(make_project(tmp_path, STATE.replace("| DDD", rows + "\n| DDD")))
    assert len(b) <= 6000
    assert "more tracks truncated" in b
    assert "queue: " in b and "GPU 4-7" in b
    assert b.rstrip().endswith("이 brief는 요약이지 근거가 아니다.")


def sh_check(path):
    return run("sh_syntax_check.py", json.dumps({"tool_name": "Write", "tool_input": {"file_path": str(path)}}))


def test_sh_syntax_check(tmp_path):
    good = tmp_path / "good.sh"
    good.write_text("#!/bin/bash\necho hi\n")
    assert sh_check(good) == {}
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/bash\nif then fi (\n")
    out = sh_check(bad)
    assert out["decision"] == "block" and out["reason"].startswith("[harness:bash -n] ")
    assert str(bad) in out["reason"]
    txt = tmp_path / "x.txt"
    txt.write_text("if then fi (")
    assert sh_check(txt) == {}
    assert sh_check(tmp_path / "missing.sh") == {}


@pytest.mark.parametrize("cmd", [
    "rg foo .", 'rg -n "x" dir', "rg -i foo", "rtk grep foo", 'echo "rtk rg"', "git log --grep x", "rg -e -t foo", "rg foo -- -g",
    "rg -A 3 foo", "rg -l foo | head", "echo --files | rg foo",
])
def test_rg_plain_passes(cmd):
    assert guard(cmd) == {}


def test_rg_deny_wording():
    got = rule_of(guard("rg --files"))
    assert got and "rewritten to `rtk grep`" in got[1] and "pipe-sink" in got[1] and "ripgrep-only option `--files`" in got[1]
    assert "git ls-files" in got[1] and "guard-ok:rg" in got[1]


def _registered_commands():
    """(source, hook script name, command) for every harness_hooks command in both registrations."""
    root = HOOKS.parent.parent
    out = []
    for rel in (".claude/settings.json", "tools/harness_hooks/settings_snippet.json"):
        data = json.loads((root / rel).read_text(encoding="utf-8"))
        for groups in data["hooks"].values():
            for g in groups:
                for h in g["hooks"]:
                    c = h.get("command", "")
                    if "tools/harness_hooks/" in c:
                        name = c.split("tools/harness_hooks/")[1].split('"')[0]
                        out.append((rel, name, c))
    assert len(out) == 6, out
    return out


def _bash_c(cmd, project, stdin=""):
    env = dict(os.environ)
    env.pop("HARNESS_HOOKS", None)
    env["CLAUDE_PROJECT_DIR"] = str(project)
    return subprocess.run(["bash", "-c", cmd], input=stdin, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, timeout=20)


@pytest.mark.parametrize("case", ["present", "missing", "unreadable"])
@pytest.mark.parametrize("src,name,cmd", _registered_commands())
def test_registered_command_robust(tmp_path, case, src, name, cmd):
    """Registration must exit 0 silently when the hook script is missing/unreadable (m6)."""
    import shutil
    if case == "unreadable" and os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    d = tmp_path / "tools" / "harness_hooks"
    d.mkdir(parents=True)
    if case != "missing":
        shutil.copy(HOOKS / name, d / name)
        shutil.copy(HOOKS / "_common.py", d / "_common.py")
    if case == "unreadable":
        (d / name).chmod(0)
    payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                          "tool_input": {"command": "rtk rg foo"}})
    try:
        p = _bash_c(cmd, tmp_path, payload if name == "bash_guard.py" else "")
    finally:
        if case == "unreadable":
            (d / name).chmod(0o644)
    if case == "present" and name == "bash_guard.py":
        assert p.returncode == 0, p.stderr
        assert json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    elif case != "present":
        assert p.returncode == 0 and p.stdout == "", (p.returncode, p.stdout, p.stderr)
