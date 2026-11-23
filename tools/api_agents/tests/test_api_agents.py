"""Offline tests for tools/api_agents (no network, no key). Real deny sources are read (fail-closed builder)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from tools.api_agents import common, run, sync
from tools.api_agents.common import TOOL_NAMES, agent_params, load_agent_defs, worker_order

PILOT_LIST = "38_api-batch/11_batches/2026-11-11_docaudit-pilot2-living/shared/files.txt"


@pytest.fixture(autouse=True)
def _redirect_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "LEDGER_PATH", tmp_path / "ledger.jsonl")


class _Res:
    def __init__(self, prefix):
        self.prefix, self.items, self.calls = prefix, {}, []

    def create(self, **p):
        i = f"{self.prefix}_{len(self.items) + 1}"
        o = types.SimpleNamespace(id=i, version=1, name=p["name"], metadata=p.get("metadata"), archived_at=None, p=p)
        self.items[i] = o; self.calls.append(("create", i)); return o

    def update(self, i, version=None, **p):
        o = self.items[i]
        if version is not None and version != o.version:
            raise RuntimeError("409 version mismatch")
        o.version += 1; o.p = p; o.name = p.get("name", o.name); o.metadata = p.get("metadata", o.metadata)
        self.calls.append(("update", i)); return o

    def list(self, limit=None):
        return list(self.items.values())


class FakeClient:
    def __init__(self):
        self.beta = types.SimpleNamespace(agents=_Res("agent"), environments=_Res("env"))


def test_defs_shape():
    d = load_agent_defs()
    assert set(d) >= {"trm-reader", "trm-checker", "trm-lead"}
    order = worker_order(d)
    assert order.index("trm-lead") > order.index("trm-reader")
    # C1 configuration (user decision 2026-11-17, lab/38_api-batch/14_managed-agents/COST_PLAN_2026-11-17.md):
    # the reader is the cheaper seat, the checker is the judgment seat and must not be weaker than the reader.
    assert d["trm-reader"]["model"] == {"id": "claude-sonnet-5-5", "effort": "medium"}
    assert d["trm-checker"]["model"] == {"id": "claude-opus-5-5", "effort": "medium"}
    assert d["trm-lead"]["model"] == {"id": "claude-fable-5-2", "effort": "high"}
    lock = {"agents": {"trm-reader": {"id": "a1", "version": 3}, "trm-checker": {"id": "a2", "version": 1}}}
    p = agent_params(d["trm-lead"], lock)
    ma = p["multiagent"]
    assert ma["type"] == "multiagent_20261001"
    for m in ("subagents", "workflows"):
        assert ma[m]["inline_agents"] == {"type": "disabled"}          # workflow threads cannot fall back to the lead's model
        assert ma[m]["predefined_agents"] == [{"type": "agent", "id": "a1", "version": 3},
                                              {"type": "agent", "id": "a2", "version": 1}]
    cfg = {c["name"]: c["enabled"] for c in p["tools"][0]["configs"]}
    assert set(cfg) == set(TOOL_NAMES) and not cfg["web_fetch"] and not cfg["web_search"]
    assert p["tools"][0]["default_config"]["enabled"] is False
    r = agent_params(d["trm-reader"], lock)
    assert "multiagent" not in r and {k for k, v in {c["name"]: c["enabled"] for c in r["tools"][0]["configs"]}.items() if v} \
        == {"read", "glob", "grep", "write", "edit"}


def test_lead_needs_workers_in_lock():
    d = load_agent_defs()
    with pytest.raises(common.Halt):
        agent_params(d["trm-lead"], {"agents": {}})


def test_sync_idempotent_and_updates_in_place(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    c = FakeClient()
    r1 = sync.sync(c, lock_path=lock_path, log=lambda *_: None)
    assert len(r1["changes"]) == 4                                     # env + reader + checker + lead
    lock = json.loads(lock_path.read_text())
    assert lock["agents"]["trm-lead"]["version"] == 1
    r2 = sync.sync(c, lock_path=lock_path, log=lambda *_: None)
    assert r2["changes"] == []
    real = common.load_agent_defs

    def edited():
        d = real(); d["trm-reader"]["system"] += "\nExtra line.\n"; return d
    monkeypatch.setattr(sync, "load_agent_defs", edited)
    r3 = sync.sync(c, lock_path=lock_path, log=lambda *_: None)
    assert ("agent", "trm-reader") in r3["changes"] and ("agent", "trm-lead") in r3["changes"]   # lead re-pins the worker
    lock = json.loads(lock_path.read_text())
    assert lock["agents"]["trm-reader"]["version"] == 2 and lock["agents"]["trm-lead"]["version"] == 2
    lead = c.beta.agents.items[lock["agents"]["trm-lead"]["id"]]
    assert lead.p["multiagent"]["workflows"]["predefined_agents"][0]["version"] == 2
    assert sum(1 for k, _ in c.beta.agents.calls if k == "create") == 3   # never re-created


def test_sync_adopts_remote_by_key(tmp_path):
    c = FakeClient()
    sync.sync(c, lock_path=tmp_path / "a.json", log=lambda *_: None)
    r = sync.sync(c, lock_path=tmp_path / "lost.json", log=lambda *_: None)    # lock lost: adopt, do not duplicate
    assert sum(1 for k, _ in c.beta.agents.calls if k == "create") == 3
    assert all(k == "update" for k, _ in c.beta.agents.calls[3:])
    assert r["changes"]


def test_follow_gate():
    f = run.Follow()
    f.on({"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "e0"})
    assert f.done is None                                              # idle before the kickoff does not count
    f.on({"type": "user.message", "id": "e1"})
    f.on({"type": "workflow_run.created", "workflow_run_id": "wrun_1", "id": "e2"})
    assert f.on({"type": "agent.tool_use", "evaluated_permission": "ask", "id": "e3"}) == [("deny", "e3")]
    assert f.on({"type": "agent.tool_use", "evaluated_permission": "allow", "id": "e4"}) == []
    f.on({"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "e5"})
    assert f.done is None                                              # a run is still open
    f.on({"type": "session.status_idle", "stop_reason": {"type": "retries_exhausted"}, "id": "e6"})
    assert f.done is None
    f.on({"type": "workflow_run.status_ended", "workflow_run_id": "wrun_1", "result": {"type": "completed"}, "id": "e7"})
    f.on({"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "e8"})
    assert f.done == "end_turn" and "wrun_1" in f.ended_runs
    g = run.Follow(); g.on({"type": "user.message", "id": "x"})
    g.on({"type": "session.status_idle", "stop_reason": {"type": "budget_reached"}, "id": "y"})
    assert g.done == "budget_reached"
    g.on({"type": "session.status_running", "id": "z"})                # budget raised: resumed, not done
    assert g.done is None


def test_render_and_ipv4():
    assert run.render("a {{X}} b", {"X": 1}) == "a 1 b"
    with pytest.raises(common.Halt):
        run.render("a {{Y}}", {"X": 1})
    assert common.ipv4_hits("host 10.1.2.3 and 127.0.0.1 and v1.2.3") == ["10.1.2.3"]


def test_prepare_small(tmp_path):
    rels = run.read_list(common.LAB_ROOT / PILOT_LIST)[:2]
    spec = tmp_path / "s.yaml"
    spec.write_text(json.dumps({
        "name": "unit-prepare", "task": "docaudit", "lead": "trm-lead", "environment": "trm-offline",
        "budget_usd": 1, "deadline_min": 5, "files": {"select": rels},
        "context": [{"path": "38_api-batch/11_batches/2026-11-11_docaudit-pilot2-living/shared/header.md",
                     "mount": "context/header.md"}],
        "vars": {"SHARD_TOKENS": 40000, "MAX_PARALLEL": 4, "MAX_FINDINGS_PER_FILE": 8, "RUN_NOTE": "unit"}}))
    rd = run.prepare(spec, runs_dir=tmp_path / "runs", today="2026-01-01")
    corpus = json.loads((rd / "corpus.json").read_text())
    assert corpus["n_files"] + corpus["n_skipped"] == 2
    k = (rd / "kickoff.md").read_text()
    assert "{{" not in k and "/mnt/session/uploads/context/header.md" in k
    assert run.reverify(rd)["n_files"] == corpus["n_files"]
    (rd / "kickoff.md").write_text(k + "x")
    with pytest.raises(common.Halt):
        run.reverify(rd)


def test_prepare_refuses_non_lab_kind(tmp_path):
    b = common.builder()
    ent, skip = common.scan_lab_file(b, "38_api-batch/ledger.jsonl")          # L1: seat input must be lab .md/.txt
    assert ent is None and skip["layer"] == "L1"
    for bad in ("../etc/hostname", "/etc/hostname", "11_governance/../11_governance/README.md", "./x.md", ""):
        ent, skip = common.scan_lab_file(b, bad)
        assert ent is None and skip["layer"] == "path"


def test_scan_hit_refuses():
    from tools.api_batch.pack import PackError
    b = common.builder()
    with pytest.raises(PackError):
        common.scan_text(b, "kickoff", "token sk-ant-api03-ABCDEFGHIJKLMNOP here")
    with pytest.raises(PackError):
        common.scan_text(b, "kickoff", "server at 10.20.30.40")


def _spec(tmp_path, **over):
    rels = run.read_list(common.LAB_ROOT / PILOT_LIST)[:2]
    d = {"name": "unit-prepare", "task": "docaudit", "lead": "trm-lead", "environment": "trm-offline",
         "budget_usd": 1, "deadline_min": 5, "files": {"select": rels},
         "context": [{"path": "38_api-batch/11_batches/2026-11-11_docaudit-pilot2-living/shared/header.md",
                      "mount": "context/header.md"}],
         "vars": {"SHARD_TOKENS": 40000, "MAX_PARALLEL": 4, "MAX_FINDINGS_PER_FILE": 8, "RUN_NOTE": "unit"}}
    d.update(over)
    p = tmp_path / f"s{len(list(tmp_path.glob('s*.yaml')))}.yaml"
    p.write_text(json.dumps(d))
    return p


@pytest.mark.parametrize("over", [
    {"context": [{"path": "/etc/hostname", "mount": "context/h.md"}]},
    {"context": [{"path": "../../etc/hostname", "mount": "context/h.md"}]},
    {"context": [{"path": "38_api-batch/ledger.jsonl", "mount": "context/h.md"}]},          # not .md/.txt
    {"context": [{"path": "38_api-batch/11_batches/2026-11-11_docaudit-pilot2-living/shared/header.md",
                  "mount": "context/.."}]},
    {"context": [{"path": "38_api-batch/11_batches/2026-11-11_docaudit-pilot2-living/shared/header.md",
                  "mount": "context/files.txt"}]},
    {"task": "../tasks/docaudit"},
    {"files": {"select": ["../outside.md", "/etc/passwd"]}},
    {"files": {"list": "../x.txt"}},
])
def test_prepare_rejects_bad_specs(tmp_path, over):
    with pytest.raises(common.Halt):
        run.prepare(_spec(tmp_path, **over), runs_dir=tmp_path / "runs", today="2026-01-01")


class _Files:
    def __init__(self):
        self.sent = []

    def upload(self, file, expires_in_seconds=None):
        self.sent.append(file[1]); return types.SimpleNamespace(id=f"file_{len(self.sent)}")


def test_upload_binds_scanned_bytes(tmp_path):
    rd = run.prepare(_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    corpus = json.loads((rd / "corpus.json").read_text())
    c = types.SimpleNamespace(files=_Files())
    res = run.upload_all(c, rd, corpus)
    assert len(res) == corpus["n_files"] + 2 and len(c.files.sent) == len(res)
    assert {r["mount_path"] for r in res} == {"/lab/" + f["rel"] for f in corpus["files"]} | {
        "/context/header.md", "/context/files.txt"}
    # restart: finished uploads are reused only when their recorded sha matches; resources come from corpus.json
    ups = json.loads((rd / "uploads.json").read_text())
    ups["lab/stray.md"] = {"file_id": "file_x", "sha256": "0" * 64, "bytes": 1}
    (rd / "uploads.json").write_text(json.dumps(ups))
    res2 = run.upload_all(c, rd, corpus)
    assert len(c.files.sent) == len(res) and "/lab/stray.md" not in {r["mount_path"] for r in res2}
    first = next(iter(k for k in ups if k.startswith("lab/") and k != "lab/stray.md"))
    ups[first]["sha256"] = "1" * 64
    (rd / "uploads.json").write_text(json.dumps(ups))
    with pytest.raises(common.Halt):
        run.upload_all(c, rd, corpus)                                  # stale entry halts
    # bytes on disk differ from the scanned bytes (simulated by a changed scanned sha): halt before sending
    (rd / "uploads.json").unlink()
    c2 = types.SimpleNamespace(files=_Files())
    corpus["files"][0]["sha256"] = "2" * 64
    with pytest.raises(common.Halt):
        run.upload_all(c2, rd, corpus)
    assert c2.files.sent == []
    corpus = json.loads((rd / "corpus.json").read_text())
    (rd / "files.txt").write_text((rd / "files.txt").read_text() + "/mnt/session/uploads/lab/extra.md\n")
    with pytest.raises(common.Halt):
        run.upload_all(types.SimpleNamespace(files=_Files()), rd, corpus)


class _Stream:
    def __init__(self, evs):
        self.evs = evs

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.evs)

    def close(self):
        pass


class _HTTPError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}"); self.status_code = code


class _Events:
    def __init__(self, fail_first_message=False, history=None, fail_first_deny=False, message_error=None):
        self.history, self.pending, self.sends = list(history or []), [], []
        self.fail, self.fail_deny, self.message_error = fail_first_message, fail_first_deny, message_error
        self.attempts = []

    def stream(self, sid):
        evs, self.pending = self.pending, []
        return _Stream(evs)

    def send(self, sid, events):
        for e in events:
            self.attempts.append(e["type"])
            if e["type"] == "user.message" and self.message_error:
                raise self.message_error
            if e["type"] == "user.message" and self.fail:
                self.fail = False
                raise RuntimeError("connection reset")
            if e["type"] == "user.tool_confirmation" and self.fail_deny:
                self.fail_deny = False
                raise RuntimeError("connection reset")
            self.sends.append(e)
            if e["type"] == "user.tool_confirmation":          # the server lists client events too
                self.history.append({"type": "user.tool_confirmation", "tool_use_id": e["tool_use_id"],
                                     "id": f"c{len(self.sends)}"})
            if e["type"] == "user.message":
                new = [{"type": "user.message", "id": f"u{len(self.sends)}"},
                       {"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": f"i{len(self.sends)}"}]
                self.history += new; self.pending += new

    def list(self, sid):
        return list(self.history)


def _client(events):
    ses = types.SimpleNamespace(events=events, retrieve=lambda sid: types.SimpleNamespace(status="idle"))
    return types.SimpleNamespace(beta=types.SimpleNamespace(sessions=ses))


def test_follow_retries_failed_kickoff_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    ev = _Events(fail_first_message=True)
    st = run.follow(_client(ev), tmp_path, "sesn_x", "hello", run.time.time() + 30)
    assert st.done == "end_turn"
    assert [e["type"] for e in ev.sends] == ["user.message"]          # resent after the failure, exactly once
    assert (tmp_path / "kickoff_sent").exists()


def test_follow_reanswers_lost_deny_on_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    hist = [{"type": "user.message", "id": "u0"},
            {"type": "agent.tool_use", "evaluated_permission": "ask", "id": "a1"},
            {"type": "session.status_idle", "stop_reason": {"type": "requires_action"}, "id": "r1"}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist))   # seen, never answered
    (tmp_path / "kickoff_sent").write_text("x\n")
    ev = _Events(history=hist)
    orig = ev.send

    def send(sid, events):
        orig(sid, events)
        if events[0]["type"] == "user.tool_confirmation":
            done = {"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "i9"}
            ev.history.append(done); ev.pending.append(done)
    ev.send = send
    st = run.follow(_client(ev), tmp_path, "sesn_x", None, run.time.time() + 30)
    denies = [e for e in ev.sends if e["type"] == "user.tool_confirmation"]
    assert [d["tool_use_id"] for d in denies] == ["a1"] and denies[0]["result"] == "deny"
    assert st.done == "end_turn"


def test_follow_stops_on_refusal_idle():
    f = run.Follow(); f.on({"type": "user.message", "id": "u"})
    f.on({"type": "session.status_idle", "stop_reason": {"type": "refusal"}, "id": "x"})
    assert f.done == "refusal"


def test_follow_history_idle_but_session_running_is_not_done():
    f = run.Follow(); f.on({"type": "user.message", "id": "u"})
    f.on({"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "x"})
    assert f.done == "end_turn"
    f.on({"type": "session.status_running", "id": "y"})
    assert f.done is None


def _ask_history():
    return [{"type": "user.message", "id": "u0"},
            {"type": "agent.tool_use", "evaluated_permission": "ask", "id": "a1"},
            {"type": "session.status_idle", "stop_reason": {"type": "requires_action"}, "id": "r1"}]


def _end_after_deny(ev):
    orig = ev.send

    def send(sid, events):
        orig(sid, events)
        if events[0]["type"] == "user.tool_confirmation":
            done = {"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "i9"}
            ev.history.append(done); ev.pending.append(done)
    ev.send = send


def test_follow_retries_deny_whose_send_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    (tmp_path / "kickoff_sent").write_text("x\n")
    ev = _Events(history=_ask_history(), fail_first_deny=True)        # the ask is new to this process
    _end_after_deny(ev)
    st = run.follow(_client(ev), tmp_path, "sesn_x", None, run.time.time() + 30)
    assert ev.attempts.count("user.tool_confirmation") == 2           # failed once, sent again after reconnect
    assert [e["tool_use_id"] for e in ev.sends if e["type"] == "user.tool_confirmation"] == ["a1"]
    assert st.done == "end_turn"


def _fake_clock(monkeypatch):
    clock, slept = [1000.0], []
    monkeypatch.setattr(run.time, "time", lambda: clock[0])

    def sleep(s):
        slept.append(s); clock[0] += s
    monkeypatch.setattr(run.time, "sleep", sleep)
    return clock, slept


def test_follow_stops_on_permanent_client_error(tmp_path, monkeypatch):
    clock, _ = _fake_clock(monkeypatch)
    ev = _Events(message_error=_HTTPError(400))
    st = run.follow(_client(ev), tmp_path, "sesn_x", "hello", clock[0] + 30)
    assert st.done == "client_error_400" and ev.attempts == ["user.message"]
    assert not (tmp_path / "kickoff_sent").exists()
    ev = _Events(message_error=_HTTPError(429))                        # rate limit: retried until the deadline
    st = run.follow(_client(ev), tmp_path, "sesn_x", "hello", clock[0] + 5)
    assert st.done == "deadline" and len(ev.attempts) == 2


def test_follow_backoff_never_sleeps_past_deadline(tmp_path, monkeypatch):
    clock, slept = _fake_clock(monkeypatch)
    ev = _Events(message_error=RuntimeError("reset"))
    st = run.follow(_client(ev), tmp_path, "sesn_x", "hello", clock[0] + 5)
    assert slept == [2, 3] and st.done == "deadline" and clock[0] == 1005.0   # 2 s, then clamped to the 3 s left


# ------------------------------------------------------------------ start / watch (fake client, no network)
class _Sessions:
    def __init__(self, events):
        self.events, self.created = events, []

    def create(self, **p):
        self.created.append(p); return types.SimpleNamespace(id="sesn_unit")

    def retrieve(self, sid):
        return types.SimpleNamespace(status="idle")


def _start_client():
    ev = _Events()
    return types.SimpleNamespace(files=_Files(), beta=types.SimpleNamespace(sessions=_Sessions(ev))), ev


FAKE_LOCK = {"agents": {"trm-lead": {"id": "agent_lead", "version": 2}},
             "environments": {"trm-offline": {"id": "env_x"}}}


@pytest.fixture
def _no_network_tail(monkeypatch):
    monkeypatch.setattr(run, "read_lock", lambda: FAKE_LOCK)
    monkeypatch.setattr(run, "collect", lambda client, rd: {})
    monkeypatch.setattr(run.time, "sleep", lambda s: None)


@pytest.mark.parametrize("tamper", ["spec", "deny", "kickoff_scan", "file"])
def test_start_halts_before_any_egress(tmp_path, monkeypatch, _no_network_tail, tamper):
    rd = run.prepare(_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    if tamper == "spec":                                               # title / budget cap come from spec.yaml
        (rd / "spec.yaml").write_text((rd / "spec.yaml").read_text().replace('"budget_usd": 1', '"budget_usd": 4000'))
    elif tamper == "deny":
        c = json.loads((rd / "corpus.json").read_text()); c["deny_sha256"] = "0" * 64
        (rd / "corpus.json").write_text(json.dumps(c))
    elif tamper == "kickoff_scan":                                     # a deny rule that now matches the kickoff
        real = run.scan_text

        def scan(b, name, text):
            if name == "kickoff":
                from tools.api_batch.pack import PackError
                raise PackError("L3b: simulated hit")
            return real(b, name, text)
        monkeypatch.setattr(run, "scan_text", scan)
    elif tamper == "file":
        c = json.loads((rd / "corpus.json").read_text()); c["files"][0]["sha256"] = "3" * 64
        (rd / "corpus.json").write_text(json.dumps(c))
    client, ev = _start_client()
    with pytest.raises(Exception):
        run.start(client, rd, AUTH)
    assert client.files.sent == [] and client.beta.sessions.created == [] and ev.attempts == []
    assert not (rd / "session.json").exists()


AUTH = "unit test: user approval quoted here, 2026-01-01"


@pytest.mark.parametrize("auth", [None, "", "   ", "ok"])
def test_start_needs_a_quoted_authorization(tmp_path, _no_network_tail, auth):
    rd = run.prepare(_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    client, ev = _start_client()
    with pytest.raises(common.Halt):
        run.start(client, rd, auth)
    assert client.files.sent == [] and client.beta.sessions.created == [] and not common.LEDGER_PATH.exists()


def test_start_ok_path_sends_pinned_values(tmp_path, _no_network_tail):
    rd = run.prepare(_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    corpus = json.loads((rd / "corpus.json").read_text())
    assert len(corpus["spec_sha256"]) == 64
    client, ev = _start_client()
    assert run.start(client, rd, AUTH) == 0
    rows = [json.loads(x) for x in common.LEDGER_PATH.read_text().splitlines()]
    assert [r["basis"] for r in rows if r["event"] == "authorized"] == [AUTH]   # the run's own approval, not a constant
    (p,) = client.beta.sessions.created
    assert p["title"] == "trm unit-prepare" and p["budget"]["max_list_cost"]["amount"] == "100"
    assert p["agent"] == {"type": "agent", "id": "agent_lead", "version": 2}
    assert len(p["resources"]) == corpus["n_files"] + 2 == len(client.files.sent)
    assert [e["content"][0]["text"] for e in ev.sends if e["type"] == "user.message"] == [(rd / "kickoff.md").read_text()]
    with pytest.raises(common.Halt):
        run.start(client, rd, AUTH)                                    # one session per run dir


def test_collect_labels_threads_without_lock(tmp_path, monkeypatch):
    (tmp_path / "session.json").write_text(json.dumps({"id": "sesn_unit"}))
    threads = [types.SimpleNamespace(to_dict=lambda d=d: d) for d in (
        {"agent": {"type": "agent", "id": "agent_old_lead", "name": "TRM Lead", "version": 2, "model": {"id": "m"}}},
        {"agent": {"type": "agent", "id": "agent_old_reader", "name": "TRM Reader", "version": 2, "model": {"id": "m"}},
         "workflow_run_id": "wrun_1"})]
    ses = types.SimpleNamespace(retrieve=lambda sid: types.SimpleNamespace(to_dict=lambda: {"status": "idle", "usage": {}}),
                                threads=types.SimpleNamespace(list=lambda sid: threads))
    client = types.SimpleNamespace(beta=types.SimpleNamespace(sessions=ses, files=types.SimpleNamespace(
        list=lambda **k: [types.SimpleNamespace(id="file_o", filename="REPORT.md", size_bytes=1)])),
        files=types.SimpleNamespace(download=lambda fid: types.SimpleNamespace(
            write_to_file=lambda p: Path(p).write_text("x"))))
    monkeypatch.setattr(run, "read_lock", lambda: {"agents": {}, "environments": {}})
    s = run.collect(client, tmp_path)
    assert s["threads_by_agent"] == {"TRM Lead@v2|m|thread": 1, "TRM Reader@v2|m|run": 1}
    assert s["n_run_threads"] == 1 and s["inline_threads"] == 0


def test_watch_sends_a_never_sent_kickoff_once(tmp_path, _no_network_tail):
    rd = run.prepare(_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    (rd / "session.json").write_text(json.dumps({"id": "sesn_unit"}))
    client, ev = _start_client()
    assert run.watch(client, rd) == 0
    assert ev.attempts == ["user.message"] and (rd / "kickoff_sent").exists()
    rd2 = run.prepare(_spec(tmp_path, name="unit-prepare-two"), runs_dir=tmp_path / "runs", today="2026-01-01")
    (rd2 / "session.json").write_text(json.dumps({"id": "sesn_unit"}))
    client2, ev2 = _start_client()
    ev2.history = [{"type": "user.message", "id": "u0"},           # the send landed; only the marker was lost
                   {"type": "session.status_idle", "stop_reason": {"type": "end_turn"}, "id": "i0"}]
    assert run.watch(client2, rd2) == 0
    assert ev2.attempts == [] and "found in history" in (rd2 / "kickoff_sent").read_text()


class _Untouchable:
    def __getattr__(self, k):
        raise AssertionError(f"client used before the scan finished: {k}")


@pytest.mark.parametrize("field,value", [("agent_name", "sk-ant-api03-ABCDEFGHIJKLMNOP"),
                                         ("agent_description", "lives at 10.20.30.40"),
                                         ("env_name", "env 10.20.30.40")])
def test_sync_scans_names_before_any_call(tmp_path, monkeypatch, field, value):
    from tools.api_batch.pack import PackError
    ra, re_ = common.load_agent_defs, common.load_env_defs

    def agents():
        d = ra()
        if field == "agent_name":
            d["trm-reader"]["name"] = value
        if field == "agent_description":
            d["trm-reader"]["description"] = value
        return d

    def envs():
        d = re_()
        if field == "env_name":
            d["trm-offline"]["name"] = value
        return d
    monkeypatch.setattr(sync, "load_agent_defs", agents)
    monkeypatch.setattr(sync, "load_env_defs", envs)
    with pytest.raises(PackError):
        sync.sync(_Untouchable(), lock_path=tmp_path / "l.json", log=lambda *_: None)
    assert not (tmp_path / "l.json").exists()


def test_scan_lab_file_ipv4_refuses(tmp_path, monkeypatch):
    lab = tmp_path / "lab"; (lab / "x").mkdir(parents=True)
    (lab / "x" / "a.md").write_text("# t\nhost 10.20.30.40\n")
    monkeypatch.setattr(common, "LAB_ROOT", lab)

    class B:
        def build_verify_only(self, *a, **k):
            return None
    ent, skip = common.scan_lab_file(B(), "x/a.md")
    assert ent is None and skip["layer"] == "IPV4"


# ------------------------------------------------------------------ shared worker text (inline context, A4)
HEADER = "38_api-batch/11_batches/2026-11-11_docaudit-pilot2-living/shared/header.md"
HEADER_MOUNT = "/mnt/session/uploads/context/header.md"
INLINE_NOTE = ("Every worker in this run already has the context file " + HEADER_MOUNT + " and the rules block for "
               "every agent in its system prompt. Briefs must not repeat them and must not tell workers to read the "
               "context file. A brief carries only that worker's input paths, its output path and output format, the "
               "current date, and anything specific to its shard.")
PLAIN_NOTE = ("Every brief also carries the path of the context file (" + HEADER_MOUNT + "), which the worker reads "
              "first, and the rules block for every agent, word for word.")


def _inline_spec(tmp_path, **over):
    d = {"context": [{"path": HEADER, "mount": "context/header.md", "inline": True}],
         "vars": {"SHARD_TOKENS": 40000, "MAX_PARALLEL": 4, "MAX_FINDINGS_PER_FILE": 5, "RUN_NOTE": "unit"}}
    d.update(over)
    return _spec(tmp_path, **d)


def test_split_worker_rules_and_substitution():
    t = "head {{X}}\n<!-- WORKER_RULES_BEGIN -->\n- rule {{X}}\n- second\n<!-- WORKER_RULES_END -->\ntail\n"
    body, rules = run.split_worker_rules(t)
    assert body == "head {{X}}\n- rule {{X}}\n- second\ntail\n"            # marker lines gone, content kept
    assert rules == "- rule {{X}}\n- second\n" and run.render(rules, {"X": 5}) == "- rule 5\n- second\n"
    assert run.split_worker_rules("no markers {{X}}\n") == ("no markers {{X}}\n", None)
    for bad in ("<!-- WORKER_RULES_END -->\nx\n<!-- WORKER_RULES_BEGIN -->\n",
                "<!-- WORKER_RULES_BEGIN -->\nx\n<!-- WORKER_RULES_BEGIN -->\ny\n<!-- WORKER_RULES_END -->\n",
                "<!-- WORKER_RULES_BEGIN -->\nx\n",
                "text <!-- WORKER_RULES_BEGIN --> inline\nx\n<!-- WORKER_RULES_END -->\n",
                "<!-- WORKER_RULES_BEGIN -->\n\n<!-- WORKER_RULES_END -->\n"):
        with pytest.raises(common.Halt):
            run.split_worker_rules(bad)
    body, rules = run.split_worker_rules((run.TASKS_DIR / "docaudit.md").read_text())
    assert "WORKER_RULES" not in body and "{{MAX_FINDINGS_PER_FILE}}" in rules and "{{WORKER_BRIEFS}}" in body


def test_prepare_inline_writes_worker_system_and_sha(tmp_path):
    rd = run.prepare(_inline_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    corpus = json.loads((rd / "corpus.json").read_text())
    raw = (rd / "worker_system.md").read_bytes()
    assert corpus["worker_system_sha256"] == common.sha256_bytes(raw)
    assert [c["inline"] for c in corpus["context"]] == [True]
    ws = raw.decode()
    header = (common.LAB_ROOT / HEADER).read_text().strip("\n")
    ctx_at, rules_at = ws.index(f"# Context file: {HEADER_MOUNT}\n\n" + header), ws.index("\n# Rules for every agent\n")
    assert ctx_at < rules_at                                              # context first, then the rules
    assert "At most 5 findings per file" in ws and "{{" not in ws and "WORKER_RULES" not in ws
    _, rules = run.split_worker_rules((run.TASKS_DIR / "docaudit.md").read_text())
    assert run.render(rules, {"MAX_FINDINGS_PER_FILE": 5}).strip("\n") in ws
    assert run.reverify(rd)["worker_system_sha256"] == corpus["worker_system_sha256"]
    (rd / "worker_system.md").write_text(ws + "x")
    with pytest.raises(common.Halt):
        run.reverify(rd)
    (rd / "worker_system.md").unlink()
    with pytest.raises(common.Halt):
        run.reverify(rd)


def test_kickoff_wording_both_cases(tmp_path):
    rd = run.prepare(_inline_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    k = (rd / "kickoff.md").read_text()
    assert INLINE_NOTE in k and "word for word" not in k and "WORKER_RULES" not in k
    rd2 = run.prepare(_spec(tmp_path, name="unit-plain"), runs_dir=tmp_path / "runs", today="2026-01-01")
    k2 = (rd2 / "kickoff.md").read_text()
    assert PLAIN_NOTE in k2 and "already has" not in k2 and "WORKER_RULES" not in k2
    assert not (rd2 / "worker_system.md").exists()
    assert json.loads((rd2 / "corpus.json").read_text())["worker_system_sha256"] is None
    assert k.replace(INLINE_NOTE, PLAIN_NOTE).replace("At most 5", "At most 8") == k2   # only the note (and var) differ


def test_marker_free_template_kickoff_unchanged(tmp_path, monkeypatch):
    """Markers are stripped from the kickoff: a template without them gives the same kickoff (no inline context)."""
    tasks = tmp_path / "tasks"; tasks.mkdir()
    src = (run.TASKS_DIR / "docaudit.md").read_text()
    (tasks / "docaudit.md").write_text("".join(ln for ln in src.splitlines(keepends=True) if "WORKER_RULES" not in ln))
    rd = run.prepare(_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    monkeypatch.setattr(run, "TASKS_DIR", tasks)
    rd2 = run.prepare(_spec(tmp_path, name="unit-unmarked"), runs_dir=tmp_path / "runs", today="2026-01-01")
    assert (rd / "kickoff.md").read_bytes() == (rd2 / "kickoff.md").read_bytes()
    assert not (rd2 / "worker_system.md").exists()
    with pytest.raises(common.Halt):                                      # inline context needs a marked rules block
        run.prepare(_inline_spec(tmp_path, name="unit-unmarked-inline"), runs_dir=tmp_path / "runs", today="2026-01-01")


@pytest.mark.parametrize("over", [
    {"context": [{"path": HEADER, "mount": "context/header.md", "inline": "yes"}]},
    {"vars": {"SHARD_TOKENS": 1, "MAX_PARALLEL": 1, "MAX_FINDINGS_PER_FILE": 5, "RUN_NOTE": "", "WORKER_BRIEFS": "x"}},
])
def test_prepare_inline_rejects(tmp_path, over):
    with pytest.raises(common.Halt):
        run.prepare(_inline_spec(tmp_path, **over), runs_dir=tmp_path / "runs", today="2026-01-01")
    assert not (tmp_path / "runs").exists()


def _tmp_lab(tmp_path, monkeypatch, context_text):
    from tools.api_batch.deny import default_deny_files
    from tools.api_batch.pack import PackBuilder
    lab = tmp_path / "lab"; (lab / "x").mkdir(parents=True)
    (lab / "x" / "README.md").write_text("# Index\n\nThe registry lists three entries.\n")
    (lab / "x" / "context.md").write_text(context_text)
    monkeypatch.setattr(common, "LAB_ROOT", lab)
    monkeypatch.setattr(run, "LAB_ROOT", lab)
    monkeypatch.setattr(run, "builder", lambda: PackBuilder.from_files(default_deny_files(), lab_root=lab))
    p = tmp_path / "tmp-lab.yaml"
    p.write_text(json.dumps({
        "name": "unit-tmp-lab", "task": "docaudit", "lead": "trm-lead", "environment": "trm-offline",
        "budget_usd": 1, "deadline_min": 5, "files": {"select": ["x/README.md"]},
        "context": [{"path": "x/context.md", "mount": "context/context.md", "inline": True}],
        "vars": {"SHARD_TOKENS": 40000, "MAX_PARALLEL": 4, "MAX_FINDINGS_PER_FILE": 5, "RUN_NOTE": "unit"}}))
    return p


def test_prepare_inline_context_clean_control(tmp_path, monkeypatch):
    rd = run.prepare(_tmp_lab(tmp_path, monkeypatch, "# Context\n\nPlain project notes.\n"),
                     runs_dir=tmp_path / "runs", today="2026-01-01")
    assert "Plain project notes." in (rd / "worker_system.md").read_text()


def test_prepare_inline_context_denied_string_fails(tmp_path, monkeypatch):
    spec = _tmp_lab(tmp_path, monkeypatch, "# Context\n\ntoken sk-ant-api03-ABCDEFGHIJKLMNOP here\n")
    with pytest.raises(common.Halt, match="context file refused"):
        run.prepare(spec, runs_dir=tmp_path / "runs", today="2026-01-01")
    assert not (tmp_path / "runs").exists()


def test_prepare_scans_worker_system_text(tmp_path, monkeypatch):
    from tools.api_batch.pack import PackError
    real, seen = run.scan_text, []

    def spy(b, name, text):
        seen.append((name, text)); return real(b, name, text)
    monkeypatch.setattr(run, "scan_text", spy)
    rd = run.prepare(_inline_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    assert ("worker_system", (rd / "worker_system.md").read_text()) in seen

    def deny_ws(b, name, text):
        if name == "worker_system":
            raise PackError("L3b: simulated hit")
        return real(b, name, text)
    monkeypatch.setattr(run, "scan_text", deny_ws)
    with pytest.raises(PackError):
        run.prepare(_inline_spec(tmp_path, name="unit-denied"), runs_dir=tmp_path / "runs", today="2026-01-01")
    assert not (tmp_path / "runs" / "2026-01-01_unit-denied").exists()


def test_agent_params_unchanged_without_worker_system(tmp_path):
    d = load_agent_defs()
    lock = {"agents": {"trm-reader": {"id": "a1", "version": 3}, "trm-checker": {"id": "a2", "version": 1}}}
    for k, v in d.items():
        p = agent_params(v, lock)
        assert p["system"] == v["system"]
        assert common.params_sha(p) == common.params_sha(agent_params(v, lock, None)) \
            == common.params_sha(agent_params(v, lock, ""))
    la, lb = tmp_path / "a.json", tmp_path / "b.json"
    sync.sync(FakeClient(), lock_path=la, log=lambda *_: None)
    sync.sync(FakeClient(), lock_path=lb, log=lambda *_: None, worker_system=None, lead="trm-lead")
    sa, sb = json.loads(la.read_text())["agents"], json.loads(lb.read_text())["agents"]
    assert {k: v["sha256"] for k, v in sa.items()} == {k: v["sha256"] for k, v in sb.items()}
    assert not any("worker_system_sha256" in v for v in sa.values())


def _ws_client():
    ev = _Events()
    beta = types.SimpleNamespace(agents=_Res("agent"), environments=_Res("env"), sessions=_Sessions(ev))
    return types.SimpleNamespace(files=_Files(), beta=beta), ev


@pytest.fixture
def _ws_lock(tmp_path, monkeypatch):
    import functools
    lp = tmp_path / "lock.json"
    monkeypatch.setattr(run, "read_lock", lambda: common.read_lock(lp))
    monkeypatch.setattr(run, "sync_defs", functools.partial(sync.sync, lock_path=lp))
    monkeypatch.setattr(run, "collect", lambda client, rd: {})
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    return lp


def test_start_inline_syncs_workers_then_lead(tmp_path, _ws_lock):
    c, ev = _ws_client()
    sync.sync(c, lock_path=_ws_lock, log=lambda *_: None)                # plain v1 agents, as after a normal sync
    rd = run.prepare(_inline_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    ws = (rd / "worker_system.md").read_text()
    sha = json.loads((rd / "corpus.json").read_text())["worker_system_sha256"]
    assert run.start(c, rd, AUTH) == 0
    lock = json.loads(_ws_lock.read_text())["agents"]
    d = load_agent_defs()
    for k in ("trm-reader", "trm-checker"):
        a = c.beta.agents.items[lock[k]["id"]]
        assert a.p["system"] == d[k]["system"] + common.WORKER_SYSTEM_SEP + ws
        assert lock[k]["version"] == 2 and lock[k]["worker_system_sha256"] == sha
    lead = c.beta.agents.items[lock["trm-lead"]["id"]]
    assert lead.p["system"] == d["trm-lead"]["system"] and "worker_system_sha256" not in lock["trm-lead"]
    for m in ("subagents", "workflows"):
        assert [r["version"] for r in lead.p["multiagent"][m]["predefined_agents"]] == [2, 2]
    ups = [i for kind, i in c.beta.agents.calls if kind == "update"]
    assert set(ups[:2]) == {lock["trm-reader"]["id"], lock["trm-checker"]["id"]} \
        and ups[2:] == [lock["trm-lead"]["id"]]                           # workers first, then the lead
    (p,) = c.beta.sessions.created
    assert p["agent"] == {"type": "agent", "id": lock["trm-lead"]["id"], "version": 2}
    assert json.loads((rd / "session.json").read_text())["worker_system_sha256"] == sha
    assert [e["content"][0]["text"] for e in ev.sends if e["type"] == "user.message"] == [(rd / "kickoff.md").read_text()]
    # a second run with the same shared text reuses those versions (one cached prefix across runs, no new versions)
    rd2 = run.prepare(_inline_spec(tmp_path, name="unit-inline-two"), runs_dir=tmp_path / "runs", today="2026-01-01")
    c.beta.sessions.created.clear()
    assert run.start(c, rd2, AUTH) == 0 and len(c.beta.agents.calls) == 6
    assert c.beta.sessions.created[0]["agent"]["version"] == 2


def test_start_plain_refuses_workers_left_with_shared_text(tmp_path, _ws_lock):
    c, _ = _ws_client()
    sync.sync(c, lock_path=_ws_lock, log=lambda *_: None)
    assert run.start(c, run.prepare(_inline_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01"), AUTH) == 0
    rd = run.prepare(_spec(tmp_path, name="unit-plain-after"), runs_dir=tmp_path / "runs", today="2026-01-01")
    n_up, n_calls, n_ses = len(c.files.sent), len(c.beta.agents.calls), len(c.beta.sessions.created)
    with pytest.raises(common.Halt, match="tools.api_agents.sync"):
        run.start(c, rd, AUTH)
    assert (len(c.files.sent), len(c.beta.agents.calls), len(c.beta.sessions.created)) == (n_up, n_calls, n_ses)
    sync.sync(c, lock_path=_ws_lock, log=lambda *_: None)                # back to plain prompts, lead re-pins
    lock = json.loads(_ws_lock.read_text())["agents"]
    assert not any("worker_system_sha256" in v for v in lock.values())
    assert c.beta.agents.items[lock["trm-reader"]["id"]].p["system"] == load_agent_defs()["trm-reader"]["system"]
    assert run.start(c, rd, AUTH) == 0


def test_start_inline_halts_before_egress_on_tampered_worker_system(tmp_path, _ws_lock):
    c, ev = _ws_client()
    sync.sync(c, lock_path=_ws_lock, log=lambda *_: None)
    rd = run.prepare(_inline_spec(tmp_path), runs_dir=tmp_path / "runs", today="2026-01-01")
    (rd / "worker_system.md").write_text((rd / "worker_system.md").read_text() + "extra\n")
    n_calls = len(c.beta.agents.calls)
    with pytest.raises(common.Halt):
        run.start(c, rd, AUTH)
    assert len(c.beta.agents.calls) == n_calls and c.files.sent == [] and c.beta.sessions.created == []
    assert ev.attempts == [] and not (rd / "session.json").exists()


def test_worker_briefs_variants():
    one = [{"mount": "context/header.md", "inline": True}]
    assert run.worker_briefs(one) == INLINE_NOTE
    assert run.worker_briefs([{"mount": "context/header.md", "inline": False}]) == PLAIN_NOTE
    mixed = run.worker_briefs(one + [{"mount": "context/registry.md", "inline": False}])
    assert mixed.startswith(INLINE_NOTE) and "(/mnt/session/uploads/context/registry.md) are not in the system prompt" in mixed
    assert "context files /mnt/session/uploads/context/a.md, /mnt/session/uploads/context/b.md" in run.worker_briefs(
        [{"mount": "context/a.md", "inline": True}, {"mount": "context/b.md", "inline": True}])


def _fake_run(tmp_path, name, findings, sha="same", cost="200"):
    rd = tmp_path / "runs" / name
    (rd / "outputs").mkdir(parents=True)
    (rd / "corpus.json").write_text(json.dumps({"files": [{"rel": "doc/a.md"}], "files_txt_sha256": sha}))
    (rd / "summary.json").write_text(json.dumps({"list_cost": {"amount": cost, "currency": "USD"}}))
    (rd / "outputs" / "findings.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in findings))
    return rd


def test_merge_unions_findings_and_marks_agreement(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    (lab / "doc").mkdir(parents=True)
    (lab / "doc" / "a.md").write_text("alpha line one\nbeta line two is stale\ngamma line three\n", encoding="utf-8")
    monkeypatch.setattr(run, "LAB_ROOT", lab)
    f = lambda quote, sev, claim: {"file": f"{run.UPLOADS}/lab/doc/a.md", "quote": quote, "severity": sev, "claim": claim}
    a = _fake_run(tmp_path, "r1", [f("beta line two is stale", "low", "A"), f("alpha line one", "low", "only in r1")])
    b = _fake_run(tmp_path, "r2", [f("line two is stale", "medium", "B"), f("gamma line three", "low", "only in r2")])
    info = run.merge([a, b], tmp_path / "merged")
    rows = [json.loads(x) for x in (tmp_path / "merged" / "outputs" / "findings.jsonl").read_text().splitlines()]
    assert info["n_merged"] == 3 and info["n_in_every_run"] == 1 and info["n_in_one_run"] == 2
    assert info["list_cost_cents_total"] == 400
    both = [r for r in rows if r["n_runs"] == 2]
    assert len(both) == 1 and both[0]["severity"] == "medium" and both[0]["found_in"] == ["r1", "r2"]
    assert [r["quote"] for r in rows][0] == "alpha line one"                  # sorted by file, then located line
    assert json.loads((tmp_path / "merged" / "postcheck.json").read_text())["counts"]["quote_found"] == 3


def test_merge_refuses_different_corpora_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "LAB_ROOT", tmp_path / "lab")
    a = _fake_run(tmp_path, "r1", [])
    b = _fake_run(tmp_path, "r2", [], sha="other")
    with pytest.raises(common.Halt):
        run.merge([a, b], tmp_path / "m1")
    with pytest.raises(common.Halt):
        run.merge([a, a], tmp_path / "m2")
    c = _fake_run(tmp_path, "r3", [])
    run.merge([a, c], tmp_path / "m3")
    with pytest.raises(common.Halt):
        run.merge([a, c], tmp_path / "m3")                                    # never overwrite a merged result


def test_prepare_tag_gives_a_second_run_dir(tmp_path):
    rels = run.read_list(common.LAB_ROOT / PILOT_LIST)[:2]
    spec = tmp_path / "s.yaml"
    spec.write_text(json.dumps({"name": "tagtest", "task": "docaudit", "lead": "trm-lead", "environment": "trm-offline",
                                    "budget_usd": 1, "deadline_min": 5, "files": {"select": rels},
                                    "context": [], "vars": {"SHARD_TOKENS": 8000, "MAX_PARALLEL": 2,
                                                            "MAX_FINDINGS_PER_FILE": 5, "RUN_NOTE": ""}}))
    a = run.prepare(spec, runs_dir=tmp_path / "runs", today="2026-01-01")
    b = run.prepare(spec, runs_dir=tmp_path / "runs", today="2026-01-01", tag="r2")
    assert a.name == "2026-01-01_tagtest" and b.name == "2026-01-01_tagtest-r2"
    with pytest.raises(common.Halt):
        run.prepare(spec, runs_dir=tmp_path / "runs", today="2026-01-01", tag="Bad Tag")


def test_postcheck_resolves_counter_quote_in_mounted_context(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    (lab / "doc").mkdir(parents=True); (lab / "shared").mkdir()
    (lab / "doc" / "a.md").write_text("alpha line one\n", encoding="utf-8")
    (lab / "shared" / "header.md").write_text("EXP-003 superseded\n", encoding="utf-8")
    monkeypatch.setattr(run, "LAB_ROOT", lab)
    row = {"file": f"{run.UPLOADS}/lab/doc/a.md", "quote": "alpha line one",
           "counter_file": f"{run.UPLOADS}/context/header.md", "counter_quote": "EXP-003 superseded"}
    rd = _fake_run(tmp_path, "ctx", [row, {**row, "counter_quote": "not in the context"}])
    (rd / "corpus.json").write_text(json.dumps({"files": [{"rel": "doc/a.md"}], "files_txt_sha256": "same",
                                                "context": [{"rel": "shared/header.md", "mount": "context/header.md"}]}))
    counts = run.postcheck(rd)["counts"]
    assert counts["counter_checked"] == 2 and counts["counter_found"] == 1


def test_retire_archives_everything_and_empties_lock(tmp_path):
    lock_path = tmp_path / "lock.json"
    common.write_lock({"agents": {"a": {"id": "agent_1"}, "b": {"id": "agent_2"}},
                       "environments": {"e": {"id": "env_1"}}}, lock_path)
    calls = []
    arch = lambda kind: (lambda i: (calls.append((kind, i)), types.SimpleNamespace(id=i, archived_at="t"))[1])
    client = types.SimpleNamespace(beta=types.SimpleNamespace(agents=types.SimpleNamespace(archive=arch("agent")),
                                                              environments=types.SimpleNamespace(archive=arch("env"))))
    with pytest.raises(common.Halt):
        sync.retire(client, "short", lock_path=lock_path, log=lambda *_: None)
    assert calls == []
    r = sync.retire(client, "user 2026-01-01: 'archive them'", lock_path=lock_path, log=lambda *_: None)
    assert calls == [("agent", "agent_1"), ("agent", "agent_2"), ("env", "env_1")] and len(r["archived"]) == 3
    assert common.read_lock(lock_path) == {"agents": {}, "environments": {}}
    rows = [json.loads(x) for x in common.LEDGER_PATH.read_text().splitlines()]
    assert [x["event"] for x in rows] == ["agent_archived", "agent_archived", "env_archived", "lock_cleared"]
    assert rows[-1]["previous"]["agents"]["a"]["id"] == "agent_1"

