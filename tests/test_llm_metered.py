"""Tests for the weekly-digest metered fallback (NAS script llm_metered.py). No network, no real client."""
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path("/mnt/ayp/agents/trm/93_monitoring/11_lab-monitoring/15_research-dashboard/llm_metered.py")
KEY = "sk-ant-SENTINEL-do-not-leak-0123456789"
NOW = datetime(2026, 11, 15, 3, 0, tzinfo=timezone.utc)


def _load():
    spec = importlib.util.spec_from_file_location("llm_metered_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeClient:
    sdk_version = "fake"

    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def send(self, params):
        self.calls.append(params)
        if self.exc:
            raise self.exc
        return self.reply


def ok_reply(text="# 주간 요약\n본문", stop="end_turn"):
    return {"id": "msg_1", "_request_id": "req_abc", "model": "claude-sonnet-5-5", "stop_reason": stop,
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
            "usage": {"input_tokens": 10000, "output_tokens": 5000,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    m = _load()
    m.LEDGER_PATH = tmp_path / "cache" / "metered_calls.jsonl"
    m._now = lambda: NOW
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    constructed = []
    holder = {"client": FakeClient(ok_reply())}

    def make_client():
        constructed.append(1)
        return holder["client"]

    m.make_client = make_client
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO("프롬프트 <material>x</material>".encode()), encoding="utf-8"))
    return m, holder, constructed


def run(m, capsys, *argv):
    rc = m.main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def ledger(m):
    if not m.LEDGER_PATH.exists():
        return []
    recs = [json.loads(x) for x in m.LEDGER_PATH.read_text(encoding="utf-8").splitlines()]
    return [r for r in recs if r.get("outcome") != "attempt"]


def raw_lines(m):
    return m.LEDGER_PATH.read_text(encoding="utf-8").splitlines() if m.LEDGER_PATH.exists() else []


def seed(m, n, ts):
    m.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with m.LEDGER_PATH.open("a") as f:
        for _ in range(n):
            f.write(json.dumps({"ts": ts, "kind": "weekly", "outcome": "ok"}) + "\n")


def no_key_leak(out, err, m):
    blob = out + err + (m.LEDGER_PATH.read_text(encoding="utf-8") if m.LEDGER_PATH.exists() else "")
    assert KEY not in blob


def test_ok_path(env, capsys):
    m, holder, constructed = env
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "2026-W46")
    assert rc == 0 and out == "# 주간 요약\n본문\n"
    (p,) = holder["client"].calls
    assert p["model"] == "claude-sonnet-5-5" and p["max_tokens"] == m.MAX_TOKENS and len(p["messages"]) == 1
    (rec,) = ledger(m)
    assert rec["outcome"] == "ok" and rec["input_tokens"] == 10000 and rec["output_tokens"] == 5000
    assert rec["kind"] == "weekly" and rec["label"] == "2026-W46" and rec["request_id"] == "req_abc"
    assert rec["ts"].startswith("2026-11-15")
    # sonnet-5-5 is priced in prices.py (batch 1/5 -> standard 2/10): 10000*2e-6 + 5000*10e-6 = 0.07
    assert rec["est_cost_usd"] == pytest.approx(0.07)
    no_key_leak(out, err, m)


def test_kind_daily_refused(env, capsys):
    m, holder, constructed = env
    rc, out, err = run(m, capsys, "--kind", "daily", "--label", "2026-11-10")
    assert rc == 2 and out == "" and "daily" in err and not constructed and ledger(m) == []


def test_missing_label_is_usage_error(env, capsys):
    m, _, constructed = env
    rc, *_ = run(m, capsys, "--kind", "weekly")
    assert rc == 2 and not constructed


def test_cap_reached_this_month(env, capsys):
    m, holder, constructed = env
    seed(m, m.MONTHLY_CAP, "2026-11-03T00:00:00+00:00")
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "2026-W46")
    assert rc == 3 and out == "" and not constructed and len(ledger(m)) == m.MONTHLY_CAP
    assert "cap" in err


def test_last_month_does_not_count(env, capsys):
    m, holder, constructed = env
    seed(m, m.MONTHLY_CAP, "2026-10-31T23:59:59+00:00")
    rc, out, _ = run(m, capsys, "--kind", "weekly", "--label", "2026-W46")
    assert rc == 0 and constructed and len(ledger(m)) == m.MONTHLY_CAP + 1


def test_failed_attempts_count_toward_cap(env, capsys):
    m, holder, constructed = env
    seed(m, m.MONTHLY_CAP - 1, "2026-11-01T00:00:00+00:00")
    assert run(m, capsys, "--kind", "weekly", "--label", "a")[0] == 0     # 6th attempt allowed
    m_stdin = io.TextIOWrapper(io.BytesIO(b"again"), encoding="utf-8")
    import sys
    sys.stdin = m_stdin
    assert run(m, capsys, "--kind", "weekly", "--label", "b")[0] == 3     # 7th refused


def test_missing_key(env, capsys, monkeypatch):
    m, holder, constructed = env
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 4 and out == "" and not constructed and ledger(m) == [] and "ANTHROPIC_API_KEY" in err


def test_sdk_not_importable(env, capsys):
    m, holder, constructed = env

    def boom():
        raise ImportError("No module named anthropic")

    m.make_client = boom
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 4 and ledger(m) == [] and "SDK" in err


def test_api_error(env, capsys):
    m, holder, constructed = env

    class Boom(Exception):
        status_code = 529

    holder["client"] = FakeClient(exc=Boom(f"overloaded; auth header was {KEY}"))
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 5 and out == ""
    (rec,) = ledger(m)
    assert rec["outcome"] == "api_error" and rec["est_cost_usd"] is None and rec["http_status"] == 529
    no_key_leak(out, err, m)
    assert "<key>" in err                                                  # scrubbed, not dropped


def test_max_tokens_truncation(env, capsys):
    m, holder, constructed = env
    holder["client"] = FakeClient(ok_reply(text="cut off mid-", stop="max_tokens"))
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 6 and out == ""
    (rec,) = ledger(m)
    assert rec["outcome"] == "max_tokens" and rec["stop_reason"] == "max_tokens" and rec["output_tokens"] == 5000
    assert rec["est_cost_usd"] == pytest.approx(0.07)


def test_empty_text(env, capsys):
    m, holder, constructed = env
    holder["client"] = FakeClient(ok_reply(text="  \n"))
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 6 and out == ""
    assert ledger(m)[0]["outcome"] == "empty"


def test_refusal_is_not_output(env, capsys):
    m, holder, constructed = env
    holder["client"] = FakeClient(ok_reply(text="I can't", stop="refusal"))
    rc, out, _ = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 6 and out == "" and ledger(m)[0]["outcome"] == "bad_stop"


def test_dry_run(env, capsys):
    m, holder, constructed = env
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x", "--dry-run")
    assert rc == 0 and out == "" and not constructed and ledger(m) == []
    assert "chars=" in err and "claude-sonnet-5-5" in err and str(m.MAX_TOKENS) in err and "cap=ok" in err
    no_key_leak(out, err, m)


def test_dry_run_reports_cap_without_failing(env, capsys):
    m, holder, constructed = env
    seed(m, m.MONTHLY_CAP, "2026-11-02T00:00:00+00:00")
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "x", "--dry-run")
    assert rc == 0 and "cap=REACHED" in err and not constructed and len(ledger(m)) == m.MONTHLY_CAP


def test_unpriced_model_gives_null_cost(env, capsys):
    m, holder, constructed = env
    m.MODEL = "claude-not-in-prices"
    rc, *_ = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 0 and ledger(m)[0]["est_cost_usd"] is None


def test_empty_stdin_is_usage_error(env, capsys, monkeypatch):
    m, holder, constructed = env
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"  \n"), encoding="utf-8"))
    rc, *_ = run(m, capsys, "--kind", "weekly", "--label", "x")
    assert rc == 2 and not constructed


# ---- fail-closed cap accounting (verifier r1 FAIL -> fixes) -----------------------------------------------------
def test_attempt_line_written_before_send(env, capsys):
    m, holder, _ = env
    seen = []
    orig = holder["client"].send

    def send(params):
        seen.append([json.loads(x).get("outcome") for x in raw_lines(m)])
        return orig(params)

    holder["client"].send = send
    assert run(m, capsys, "--kind", "weekly", "--label", "w")[0] == 0
    assert seen == [["attempt"]]
    recs = [json.loads(x) for x in raw_lines(m)]
    assert [r.get("outcome") for r in recs] == ["attempt", "ok"] and recs[1]["phase"] == "result"
    assert m.count_month(NOW) == 1


@pytest.mark.parametrize("bad", ["not json", '{"kind": "weekly"}', '{"ts": 5}', '{"ts": "2026-11-0', "[1, 2]"])
def test_corrupt_lines_count_toward_cap(env, capsys, bad):
    m, _, constructed = env
    m.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.LEDGER_PATH.write_text("\n".join([bad] * m.MONTHLY_CAP) + "\n", encoding="utf-8")
    rc, out, _ = run(m, capsys, "--kind", "weekly", "--label", "w")
    assert rc == 3 and out == "" and not constructed


def test_truncated_last_line_counts(env, capsys):
    m, _, constructed = env
    seed(m, m.MONTHLY_CAP - 1, "2026-11-01T00:00:00+00:00")
    with m.LEDGER_PATH.open("a") as f:
        f.write('{"ts": "2026-11-0')
    assert run(m, capsys, "--kind", "weekly", "--label", "w")[0] == 3 and not constructed


def test_unreadable_ledger_refuses(env, capsys):
    m, _, constructed = env
    m.LEDGER_PATH.mkdir(parents=True)          # a directory where the file should be
    rc, out, err = run(m, capsys, "--kind", "weekly", "--label", "w")
    assert rc == 3 and out == "" and not constructed and "fail closed" in err


def test_attempt_write_failure_refuses(env, capsys):
    m, holder, _ = env
    m.append_ledger = lambda rec: False
    rc, out, _ = run(m, capsys, "--kind", "weekly", "--label", "w")
    assert rc == 3 and out == "" and holder["client"].calls == []


def test_real_client_disables_retries():
    src = _load().__file__
    assert "Anthropic(max_retries=0)" in open(src, encoding="utf-8").read()


# ---- verifier r2 minors -----------------------------------------------------------------------------------------
def test_partial_last_line_never_swallows_attempts(env, capsys):
    import sys
    m, holder, _ = env
    seed(m, 4, "2026-11-01T00:00:00+00:00")
    with m.LEDGER_PATH.open("a") as f:
        f.write('{"ts": "2026-11-0')                 # truncated, no trailing newline -> counts as the 5th
    rcs = []
    for _ in range(4):
        sys.stdin = io.TextIOWrapper(io.BytesIO(b"p"), encoding="utf-8")
        rcs.append(run(m, capsys, "--kind", "weekly", "--label", "w")[0])
    assert rcs == [0, 3, 3, 3] and len(holder["client"].calls) == 1


@pytest.mark.parametrize("bad", ['{"ts": "0001-01-01T00:00:00+01:00"}', "[" * 5000])
def test_pathological_lines_count_not_crash(env, capsys, bad):
    m, _, constructed = env
    m.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.LEDGER_PATH.write_text("\n".join([bad] * m.MONTHLY_CAP) + "\n", encoding="utf-8")
    assert run(m, capsys, "--kind", "weekly", "--label", "w")[0] == 3 and not constructed


def test_surrogate_label_does_not_crash(env, capsys):
    m, _, _ = env
    assert run(m, capsys, "--kind", "weekly", "--label", "w\udcff")[0] in (0, 3)
