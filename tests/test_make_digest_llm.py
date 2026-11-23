"""WP1 tests for make_digest.py (NAS module, loaded by path). No real claude calls;
all output goes to tmp dirs."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

MOD = Path("/mnt/ayp/agents/trm/93_monitoring/11_lab-monitoring/15_research-dashboard/make_digest.py")


@pytest.fixture()
def md(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_digest_under_test", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DIGEST_DIR = tmp_path / "digest"
    m.REJECTED_DIR = m.DIGEST_DIR / "_rejected"
    monkeypatch.setattr(m, "LAB", tmp_path)          # rejected paths are relative to LAB
    return m


def _cp(rc=0, out=b"", err=b""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def test_failure_reason_has_rc_stdout_tail_version(md, monkeypatch):
    def fake(cmd, **kw):
        if "--version" in cmd:
            return _cp(0, b"9.9.9 (Claude Code)\n")
        return _cp(1, b"Some limit message\nline2", b"")
    monkeypatch.setattr(md.subprocess, "run", fake)
    text, reason = md.run_claude("p", "m")
    assert text is None
    assert "rc=1" in reason and "Some limit message line2" in reason
    assert "9.9.9" in reason and "\n" not in reason


def test_version_unknown_on_failure(md, monkeypatch):
    def fake(cmd, **kw):
        if "--version" in cmd:
            raise subprocess.TimeoutExpired(cmd, 5)
        return _cp(2, b"", b"bad\nthing")
    monkeypatch.setattr(md.subprocess, "run", fake)
    _, reason = md.run_claude("p", "m")
    assert "cli_version=unknown" in reason and "\n" not in reason


def test_success_path(md, monkeypatch):
    monkeypatch.setattr(md.subprocess, "run", lambda cmd, **kw: _cp(0, b"preamble\n# Title\nbody\n"))
    text, reason = md.run_claude("p", "m")
    assert reason is None and text.startswith("# ")


def _setup_main(md, monkeypatch, llm_outputs, fails_seq):
    prompts = []
    outs = iter(llm_outputs)
    fseq = iter(fails_seq)

    def fake_claude(prompt, material):
        prompts.append(prompt)
        return next(outs)

    monkeypatch.setattr(md, "run_claude", fake_claude)
    monkeypatch.setattr(md, "collect", lambda *a: [("x", "x.md", "text")])
    monkeypatch.setattr(md, "load_withheld", lambda: (
        lambda t, m: (t, 0), type("M", (), {"matches": lambda s, x: False})(), "[열람 보류]"))
    monkeypatch.setattr(md, "postcheck",
                        lambda body, mat, w: (next(fseq), {"anchored": 3, "unanchored": 0, "sources": 1}))
    monkeypatch.setattr(md.sys, "argv", ["make_digest.py", "--daily", "2026-11-01"])
    return prompts


def test_retry_then_pass(md, monkeypatch):
    prompts = _setup_main(md, monkeypatch, [("# A\nfirst", None), ("# B\nsecond", None)],
                          [["id `X-1`가 자료에 없음 (토큰 PASS)"], []])
    assert md.main() == 0
    out = (md.DIGEST_DIR / "2026-11-01.md").read_text(encoding="utf-8")
    assert 'mode: "llm"' in out and "postcheck_retries: 1" in out and "second" in out
    assert len(prompts) == 2 and "id `X-1`가 자료에 없음 (토큰 PASS)" in prompts[1]
    assert "id `X-1`" not in prompts[0]


def test_retry_fails_twice(md, monkeypatch):
    prompts = _setup_main(md, monkeypatch, [("# A\nfirst", None), ("# B\nsecond", None)],
                          [["f1"], ["f2"]])
    assert md.main() == 0
    assert len(prompts) == 2
    out = (md.DIGEST_DIR / "2026-11-01.md").read_text(encoding="utf-8")
    assert 'mode: "fallback"' in out and "fallback digest" in out
    rej = sorted(p.name for p in md.REJECTED_DIR.iterdir())
    assert rej == ["2026-11-01.md", "2026-11-01.retry.md"]
    assert "first" in (md.REJECTED_DIR / rej[0]).read_text(encoding="utf-8")
    assert "second" in (md.REJECTED_DIR / rej[1]).read_text(encoding="utf-8")


def test_retry_call_fails(md, monkeypatch):
    _setup_main(md, monkeypatch, [("# A\nfirst", None), (None, "claude CLI rc=1: boom")], [["f1"]])
    assert md.main() == 0
    out = (md.DIGEST_DIR / "2026-11-01.md").read_text(encoding="utf-8")
    line = [l for l in out.splitlines() if l.startswith("> ⚠")][0]
    assert "사후 검사 실패" in line and "재생성 호출도 실패" in line and "boom" in line
    assert [p.name for p in md.REJECTED_DIR.iterdir()] == ["2026-11-01.md"]


def test_no_llm(md, monkeypatch):
    monkeypatch.setattr(md, "collect", lambda *a: [("x", "x.md", "text")])
    monkeypatch.setattr(md, "run_claude", lambda *a: pytest.fail("LLM called"))
    monkeypatch.setattr(md.sys, "argv", ["make_digest.py", "--daily", "2026-11-01", "--no-llm"])
    assert md.main() == 0
    out = (md.DIGEST_DIR / "2026-11-01.md").read_text(encoding="utf-8")
    assert 'mode: "fallback"' in out and "--no-llm" in out
