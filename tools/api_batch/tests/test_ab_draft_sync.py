"""draft-sync (2026-10-30): sync drafting through the batch-lane rails. FakeClient only; no network, no key."""
import json

import pytest

from tools.api_batch import cli, common, draft, ops, prices
from tools.api_batch.common import Halt
from tools.api_batch import manifest as M
from tools.api_batch.fake_client import FakeApiError, FakeClient
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder

SPEC = """name: draft-test
model: claude-sonnet-5-5
effort: medium
max_tokens: 1000
system: shared/system.md
sections:
  - {id: intro, prompt: shared/intro.md, data: [notes.md]}
  - {id: results, prompt: shared/results.md, data: [notes.md, table.csv]}
"""


@pytest.fixture
def env(tmp_path, deny_files, lab):
    root = tmp_path / "root"
    pd = root / "11_batches" / "2026-10-30_draft-test"
    (pd / "shared").mkdir(parents=True)
    (pd / "spec.yaml").write_text(SPEC, encoding="utf-8")
    (pd / "shared" / "system.md").write_text("You draft report sections. Output markdown.\n", encoding="utf-8")
    (pd / "shared" / "intro.md").write_text("Write the introduction from the notes.\n", encoding="utf-8")
    (pd / "shared" / "results.md").write_text("Write the results paragraph from the table.\n", encoding="utf-8")
    (lab / "notes.md").write_text("# Notes\nThe model reaches acc 0.9 at k=3 — 한글 포함.\n", encoding="utf-8")
    (lab / "table.csv").write_text("k,acc\n3,0.9\n5,0.7\n", encoding="utf-8")
    builder = PackBuilder.from_files(deny_files, lab_root=lab)
    led = Ledger(root / "ledger.jsonl")
    return type("E", (), dict(root=root, pd=pd, lab=lab, builder=builder, led=led, deny_files=deny_files))


def _plan(env):
    return draft.plan(env.pd / "spec.yaml", env.builder, env.led)


def _approve(env, usd=None):
    sp = draft.sync_plan_for_approve(env.pd)
    return ops.approve(None, sp["hard_worst_usd"] if usd is None else usd, env.led, env={}, sync_plan=sp)


def _ready(env):
    ops.selftest(env.led, runner=lambda: 0)
    d = _plan(env)
    _approve(env)
    return d


class ScriptedClient(FakeClient):
    """FakeClient whose streamed reply (usage / model / text / exception) is set per test."""

    def __init__(self, out_mult=0.0, model=None, text=None, raise_on=None, stop_reason=None, out_mult_on=None):
        super().__init__()
        self.out_mult, self.model, self.text, self.raise_on = out_mult, model, text, raise_on
        self.stop_reason, self.out_mult_on = stop_reason, out_mult_on or {}

    def send_stream(self, request):
        if self.raise_on and request["custom_id"] in self.raise_on:
            self.calls.append("send_stream")
            self.sync_sent.append(request)
            raise FakeApiError("simulated 529 overloaded")
        msg = super().send_stream(request)
        mult = self.out_mult_on.get(request["custom_id"], self.out_mult)
        if mult:
            msg["usage"]["output_tokens"] = int(mult * request["params"]["max_tokens"])
        if self.stop_reason:
            msg["stop_reason"] = self.stop_reason
        if self.model:
            msg["model"] = self.model
        if self.text:
            msg["content"] = [{"type": "text", "text": self.text}]
        return msg


# ------------------------------------------------------------------------------------- plan
def test_plan_artifacts_standard_price_and_ledger(env):
    d = _plan(env)
    assert d["use"] == "draft-sync" and d["n_requests"] == 2 and d["sections"] == ["intro", "results"]
    for f in ("manifest.jsonl", "plan.json", "pack/pack_manifest.json"):
        assert (env.pd / f).exists()
    rows = [json.loads(ln) for ln in (env.pd / "manifest.jsonl").read_text().splitlines()]
    # cost formula 2: every block up to the last breakpoint at the 5m write price, the rest at p_in
    batch = 0.0
    for r in rows:
        t_ctx, t_var, ttl = M.cache_split(r["params"])
        assert ttl == "5m" and t_ctx > 0
        batch += prices.hard_worst_request(r["params"]["model"], t_ctx, t_var, r["params"]["max_tokens"], ttl)
    assert draft.SYNC_PRICE_FACTOR == 2.0 and d["est_usd_hard_worst"] == pytest.approx(2.0 * batch, abs=1e-6)
    assert d["cost_formula"] == 2 and d["cache_ttl"] == "5m"
    assert all("fallbacks" not in r["params"] for r in rows)
    # default cache_ttl 5m: breakpoints on the system prompt and on notes.md (the data document both sections share)
    for r in rows:
        p = r["params"]
        assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
        content = p["messages"][0]["content"]
        assert [b.get("title") for b in content if b.get("cache_control")] == ["notes.md"]
        assert "cache_control" not in content[-1]                              # the section prompt is the tail
        assert r["meta"]["cache_after"] == ["system", "notes.md"]
    assert all(s["mode"] == "verify-only" for s in d["scan"]) and d["masked_blocks"] == 0
    p = env.led.last("planned")
    assert p["plan_hash"] == d["plan_hash"] and p["lane"] == "draft-sync" and p["est_usd_hard_worst"] == d["est_usd_hard_worst"]
    assert _plan(env)["plan_hash"] == d["plan_hash"]                            # deterministic


def test_plan_cache_ttl_none_matches_the_pre_breakpoint_formula(env):
    (env.pd / "spec.yaml").write_text(SPEC + "cache_ttl: none\n", encoding="utf-8")
    d = _plan(env)
    rows = [json.loads(ln) for ln in (env.pd / "manifest.jsonl").read_text().splitlines()]
    assert all("cache_control" not in json.dumps(r["params"]) for r in rows)
    old = sum(draft.request_hard_worst(r["params"], 1) for r in rows)
    assert d["est_usd_hard_worst"] == pytest.approx(old, abs=1e-6)          # formulas 1 and 2 agree without breakpoints


def test_old_formula1_plan_still_loads(env):
    """A plan written before cost formula 2 (no `cost_formula` key, formula-1 plan_hash) must still re-derive."""
    (env.pd / "spec.yaml").write_text(SPEC + "cache_ttl: none\n", encoding="utf-8")
    _plan(env)
    pj = env.pd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    worst = d["est_usd_hard_worst"]
    d.pop("cost_formula")
    d["plan_hash"] = draft.compute_plan_hash(d["manifest_sha256"], worst)     # how the old code hashed
    pj.write_text(json.dumps(d), encoding="utf-8")
    d2, _rows, w2 = draft.load_plan(env.pd)
    assert d2["plan_hash"] == d["plan_hash"] and w2 == pytest.approx(worst, abs=1e-6)


def test_shared_depths():
    P = lambda x: type("P", (), {"pack_sha256": x})()              # noqa: E731
    a, b, c = P("a"), P("b"), P("c")
    assert draft._shared_depths([[a, b], [a, b, c], [a, c]]) == (1, [2, 2, 1])
    assert draft._shared_depths([[a]]) == (0, [0])
    assert draft._shared_depths([[a], [b]]) == (0, [0, 0])


def test_plan_refuses_deny_hit_in_data_without_masking(env):
    (env.lab / "notes.md").write_text("# Notes\nEXP-013 outcomes here\n", encoding="utf-8")
    with pytest.raises(Halt, match="L3b|L2"):
        _plan(env)
    assert not (env.pd / "plan.json").exists() and env.led.rows("planned") == []
    (env.lab / "notes.md").write_text("clean\n", encoding="utf-8")
    (env.lab / "per_call.csv").write_text("a,b\n1,2\n", encoding="utf-8")         # denied path glob
    (env.pd / "spec.yaml").write_text(SPEC.replace("table.csv", "per_call.csv"), encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)
    assert env.led.rows("planned") == []


def test_plan_refuses_deny_hit_in_prompt_and_bad_data_kind(env):
    (env.pd / "shared" / "intro.md").write_text("summarise the clean_r1 cohort\n", encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)
    (env.pd / "shared" / "intro.md").write_text("fine\n", encoding="utf-8")
    (env.lab / "blob.bin").write_bytes(b"x")
    (env.pd / "spec.yaml").write_text(SPEC.replace("table.csv", "blob.bin"), encoding="utf-8")
    with pytest.raises(Halt, match="extension"):
        _plan(env)


def test_plan_refuses_unregistered_model(env):
    (env.pd / "spec.yaml").write_text(SPEC.replace("claude-sonnet-5-5", "claude-nope-9"), encoding="utf-8")
    with pytest.raises(Halt, match="prices.py"):
        _plan(env)
    assert env.led.rows("planned") == []


def test_plan_refuses_over_budget_and_over_limits(env):
    (env.pd / "spec.yaml").write_text(SPEC + "budget_usd: 0.001\n", encoding="utf-8")
    with pytest.raises(ops.Refused, match="cap"):
        _plan(env)
    (env.pd / "spec.yaml").write_text(SPEC, encoding="utf-8")
    env.led.append("approved", manifest_sha256="other", est_usd_hard_worst=189.999)   # day/month nearly exhausted
    with pytest.raises(ops.Refused, match="committed"):
        _plan(env)
    assert env.led.rows("planned") == []


# ------------------------------------------------------------------------------------- approve / run gates
def test_run_refused_without_approval_or_tests_passed(env):
    _plan(env)
    fake = FakeClient()
    with pytest.raises(ops.Refused, match="tests_passed"):
        draft.run(env.pd, env.led, fake, env.builder)
    ops.selftest(env.led, runner=lambda: 0)
    with pytest.raises(ops.Refused, match="no matching user approval"):
        draft.run(env.pd, env.led, fake, env.builder)
    with pytest.raises(ops.Refused):                                            # approval below hard-worst
        _approve(env, usd=0.000001)
    assert fake.sync_sent == [] and env.led.rows("sync_sent") == []


def test_approval_with_understated_worst_is_refused_at_run(env):
    ops.selftest(env.led, runner=lambda: 0)
    d = _plan(env)
    ops.approve(None, d["est_usd_hard_worst"], env.led, env={},
                sync_plan={"plan_hash": d["plan_hash"], "hard_worst_usd": d["est_usd_hard_worst"] / 10})
    fake = FakeClient()
    with pytest.raises(ops.Refused, match="approval records hard-worst"):
        draft.run(env.pd, env.led, fake, env.builder)
    assert fake.sync_sent == []


def test_plan_hash_or_manifest_tamper_refused(env):
    _ready(env)
    pj = env.pd / "plan.json"
    orig = pj.read_text()
    for field, val in (("plan_hash", "0" * 64), ("est_usd_hard_worst", 0.0001)):
        d = json.loads(orig)
        d[field] = val
        pj.write_text(json.dumps(d))
        with pytest.raises(ops.Refused, match="tampered"):
            draft.run(env.pd, env.led, FakeClient(), env.builder)
        with pytest.raises(ops.Refused, match="tampered"):
            draft.sync_plan_for_approve(env.pd)
    pj.write_text(orig)
    mf = env.pd / "manifest.jsonl"
    mf.write_text(mf.read_text().replace('"max_tokens":1000', '"max_tokens":999'))
    with pytest.raises(Halt):
        draft.run(env.pd, env.led, FakeClient(), env.builder)
    assert env.led.rows("sync_sent") == []


def test_run_rescans_against_a_grown_deny_source(env, deny_files):
    _ready(env)
    deny_files[1].write_text(deny_files[1].read_text() + "  - id: new\n    status: withheld\n    patterns: ['acc 0\\.9']\n",
                             encoding="utf-8")
    fresh = PackBuilder.from_files(deny_files, lab_root=env.lab)
    fake = FakeClient()
    with pytest.raises(Halt, match="L3b"):
        draft.run(env.pd, env.led, fake, fresh)
    assert fake.sync_sent == [] and env.led.rows("sync_sent") == []


# ------------------------------------------------------------------------------------- run
def test_run_happy_path_one_approval_n_sections_then_consumed(env):
    d = _ready(env)
    fake = FakeClient()
    r = draft.run(env.pd, env.led, fake, env.builder)
    assert r["halted"] is None and [s["rc"] for s in r["sections"]] == [0, 0] and len(fake.sync_sent) == 2
    ev = [(x["event"], x.get("custom_id")) for x in env.led.rows() if x["event"] in ("sync_sent", "sync_done")]
    assert ev == [("sync_sent", "draft-intro"), ("sync_done", "draft-intro"),
                  ("sync_sent", "draft-results"), ("sync_done", "draft-results")]
    aid = env.led.last("approved")["approval_id"]
    assert all(x["approval_id"] == aid and x["manifest_sha256"] == d["plan_hash"] for x in env.led.rows("sync_sent"))
    for sid in ("intro", "results"):
        body = (env.pd / "out" / f"{sid}.md").read_text(encoding="utf-8")
        hdr = body.split("\n---\n", 1)[0]
        assert "status: DRAFT (api-batch output — not evidence)" in hdr and "model: claude-sonnet-5-5" in hdr
        assert "usage: {" in hdr and "cost_est_usd:" in hdr and "l5: pass" in hdr and body.rstrip().endswith("fake sync")
        assert json.loads((env.pd / "raw" / f"{sid}.json").read_text())["id"] == "msg_fake_sync"
    assert draft.run(env.pd, env.led, fake, env.builder)["nothing_to_do"]       # all done: no approval needed
    (env.pd / "out" / "results.md").unlink()                                    # one section pending again
    with pytest.raises(ops.Refused, match="consumed"):                          # one approval = one run
        draft.run(env.pd, env.led, fake, env.builder)
    assert len(fake.sync_sent) == 2
    assert env.led.committed_usd() == pytest.approx(d["est_usd_hard_worst"])


def test_run_halts_before_send_that_could_exceed_approved_amount(env):
    _ready(env)                                                                 # approved = exactly the plan hard-worst
    fake = ScriptedClient(out_mult=1.5)                                         # section 1 bills past its own worst
    r = draft.run(env.pd, env.led, fake, env.builder)
    assert len(fake.sync_sent) == 1 and r["halted"].startswith("budget") and r["spent_usd_est"] <= r["approved_usd"]
    assert [x["custom_id"] for x in env.led.rows("sync_sent")] == ["draft-intro"]
    assert "halted before send" in env.led.last("error")["error"]
    assert not (env.pd / "out" / "results.md").exists()


def test_run_halts_after_send_when_spend_exceeds_approval(env):
    _ready(env)
    fake = ScriptedClient(out_mult=50)
    r = draft.run(env.pd, env.led, fake, env.builder)
    assert len(fake.sync_sent) == 1 and "> approved" in r["halted"] and r["spent_usd_est"] > r["approved_usd"]


def test_send_failure_records_rc_and_consumes_approval(env):
    _ready(env)
    fake = ScriptedClient(raise_on={"draft-intro"})
    r = draft.run(env.pd, env.led, fake, env.builder)
    assert r["sections"] == [{"section": "intro", "rc": 1}] and "send failed" in r["halted"]
    done = env.led.last("sync_done")
    assert done["rc"] == 1 and "FakeApiError" in done["error"] and len(fake.sync_sent) == 1
    with pytest.raises(ops.Refused, match="consumed"):
        draft.run(env.pd, env.led, FakeClient(), env.builder)
    _approve(env)                                                               # a new approval retries
    r2 = draft.run(env.pd, env.led, FakeClient(), env.builder)
    assert [s["rc"] for s in r2["sections"]] == [0, 0]
    fake3 = FakeClient()                                                        # all done: no approval needed
    r3 = draft.run(env.pd, env.led, fake3, env.builder)
    assert fake3.calls == [] and r3["nothing_to_do"] and all(s.get("skipped") for s in r3["sections"])


def test_model_fallback_is_quarantined_and_stops(env):
    _ready(env)
    r = draft.run(env.pd, env.led, ScriptedClient(model="claude-haiku-5-5"), env.builder)
    assert r["sections"][0]["rc"] == 2 and "fallback" in r["halted"] and len(r["sections"]) == 1
    assert (env.pd / "quarantine" / "intro.md").exists() and not (env.pd / "out" / "intro.md").exists()
    _approve(env)
    r2 = draft.run(env.pd, env.led, ScriptedClient(model="claude-sonnet-5-5-20261001"), env.builder)
    assert [s["rc"] for s in r2["sections"]] == [0, 0]                         # dated snapshot of the same model


def test_l5_hit_in_output_is_quarantined(env):
    _ready(env)
    r = draft.run(env.pd, env.led, ScriptedClient(text="the cohort EXP-013 reached 0.9"), env.builder)
    assert [s["rc"] for s in r["sections"]] == [3, 3] and all(s["l5"] == "quarantined" for s in r["sections"])
    assert not (env.pd / "out" / "intro.md").exists()
    assert "l5: quarantined" in (env.pd / "quarantine" / "intro.md").read_text(encoding="utf-8")


def test_spec_rejects_unknown_keys_and_bad_section_ids(env):
    for bad in (SPEC + "fallbacks: [claude-haiku-5-5]\n", SPEC.replace("id: intro", "id: ../x"),
                SPEC.replace("id: results", "id: intro"), SPEC.replace("effort: medium\n", "")):
        (env.pd / "spec.yaml").write_text(bad, encoding="utf-8")
        with pytest.raises(Halt):
            _plan(env)


# ------------------------------------------------------------------------------------- cli
def test_cli_plan_user_approve_run(env, monkeypatch, capsys):
    argv = ["--batch-root", str(env.root), "--lab-root", str(env.lab), *sum((["--deny-file", str(f)] for f in env.deny_files), [])]
    assert cli.main([*argv, "draft-sync", "plan", str(env.pd / "spec.yaml")]) == 0
    out = capsys.readouterr().out
    usd = out.split("--usd ")[1].split()[0]                                     # the printed command must be approvable
    worst = json.loads((env.pd / "plan.json").read_text())["est_usd_hard_worst"]
    for k in common.SESSION_MARKERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert cli.main([*argv, "approve", str(env.pd), "--usd", "1"]) == 3          # session guard kept
    assert env.led.rows("approved") == []
    monkeypatch.delenv("CLAUDECODE")
    assert cli.main([*argv, "approve", str(env.pd), "--usd", "1", "--for-probe-sync"]) == 3
    assert cli.main([*argv, "approve", str(env.pd), "--usd", usd]) == 0
    a = env.led.last("approved")
    assert a["lane"] == "sync" and a["est_usd_hard_worst"] == pytest.approx(worst)  # from plan.json, not typed
    ops.selftest(env.led, runner=lambda: 0)
    fake = FakeClient()
    assert cli.main([*argv, "draft-sync", "run", str(env.pd)], client=fake) == 0 and len(fake.sync_sent) == 2
    assert cli.main([*argv, "draft-sync", "run", str(env.pd)], client=fake) == 0   # all done: nothing to do
    assert "nothing to do" in capsys.readouterr().out and len(fake.sync_sent) == 2
    n_rows, committed = len(env.led.rows()), env.led.committed_usd()
    assert cli.main([*argv, "approve", str(env.pd), "--usd", usd]) == 3            # approve of an all-done plan refused
    assert len(env.led.rows()) == n_rows and env.led.committed_usd() == pytest.approx(committed)


# ------------------------------------------------------------------------------------- streaming / re-run / model id
def test_run_sends_over_the_streaming_path_only(env):
    _ready(env)
    fake = FakeClient()
    draft.run(env.pd, env.led, fake, env.builder)
    assert fake.calls == ["send_stream", "send_stream"]                        # never the non-streaming send_sync
    with pytest.raises(FakeApiError, match="stream"):
        fake.send_stream({"custom_id": "x", "params": {"model": "m", "max_tokens": 1, "stream": True}})


class _Rec:
    def __init__(self, d):
        self.d = d

    def model_dump(self, mode=None):
        assert mode == "json"
        return dict(self.d)


class _StubMessages:
    """Stands in for anthropic.Anthropic().messages (no network): records which entry point was used."""

    def __init__(self):
        self.seen = []

    def create(self, **kw):
        self.seen.append(("create", kw))
        return _Rec({"id": "msg_c", "model": kw["model"]})

    def stream(self, **kw):
        self.seen.append(("stream", kw))
        outer = self

        class _Mgr:
            def __enter__(self):
                outer.seen.append(("enter", None))
                return self

            def __exit__(self, *a):
                outer.seen.append(("exit", None))
                return False

            def get_final_message(self):
                return _Rec({"id": "msg_s", "model": kw["model"], "usage": {"output_tokens": 7}})
        return _Mgr()


def test_real_client_send_stream_and_unchanged_send_sync_without_network():
    from tools.api_batch.client import RealClient
    rc = object.__new__(RealClient)                                            # skip __init__: no SDK import, no key
    stub = _StubMessages()
    rc._c = type("C", (), {"messages": stub})()
    params = {"model": "claude-sonnet-5-5", "max_tokens": 64000, "messages": [{"role": "user", "content": "x"}]}
    out = rc.send_stream({"custom_id": "a", "params": params})
    assert out == {"id": "msg_s", "model": "claude-sonnet-5-5", "usage": {"output_tokens": 7}} and isinstance(out, dict)
    assert stub.seen == [("stream", params), ("enter", None), ("exit", None)]   # context manager closed
    stub.seen.clear()
    assert rc.send_sync({"custom_id": "a", "params": params}) == {"id": "msg_c", "model": "claude-sonnet-5-5"}
    assert stub.seen == [("create", params)]                                   # probe-sync path: still messages.create
    with pytest.raises(ValueError, match="stream"):
        rc.send_stream({"custom_id": "a", "params": {**params, "stream": True}})


def test_probe_sync_path_still_uses_send_sync():
    from tools.api_batch import ops as _ops
    import inspect
    src = inspect.getsource(_ops.probe_sync)
    assert "client.send_sync(" in src and "send_stream" not in src


def test_rerun_with_everything_done_is_a_noop_for_approval_and_committed(env):
    d = _ready(env)
    draft.run(env.pd, env.led, FakeClient(), env.builder)
    n_rows, committed = len(env.led.rows()), env.led.committed_usd()
    assert committed == pytest.approx(d["est_usd_hard_worst"])
    fake = FakeClient()
    r = draft.run(env.pd, env.led, fake, env.builder)                          # no new approval: not "consumed"
    assert r["nothing_to_do"] and r["halted"] is None and r["approval_id"] is None and fake.calls == []
    assert len(env.led.rows()) == n_rows and env.led.committed_usd() == pytest.approx(committed)
    with pytest.raises(ops.Refused, match="nothing to do"):
        draft.sync_plan_for_approve(env.pd, env.led)                           # approve path refuses before a row
    assert len(env.led.rows()) == n_rows and env.led.committed_usd() == pytest.approx(committed)
    (env.pd / "out" / "results.md").unlink()                                   # a missing output is pending again
    assert [r["custom_id"] for r in draft.pending_sections(env.pd, env.led, d["plan_hash"], draft.load_plan(env.pd)[1])] \
        == ["draft-results"]
    assert draft.sync_plan_for_approve(env.pd, env.led)["plan_hash"] == d["plan_hash"]


def test_rerun_nothing_to_do_does_not_consume_an_existing_approval(env):
    d = _ready(env)
    draft.run(env.pd, env.led, FakeClient(), env.builder)
    _approve(env)                                                              # a stray approval (no ledger check)
    stray = env.led.last("approved")["approval_id"]
    draft.run(env.pd, env.led, FakeClient(), env.builder)
    assert stray not in {x.get("approval_id") for x in env.led.rows("sync_sent", manifest_sha256=d["plan_hash"])}


@pytest.mark.parametrize("returned,fb", [
    ("claude-sonnet-5-5", False),                     # exact
    ("claude-sonnet-5-5-20261001", False),            # dated snapshot of the requested alias
    ("claude-sonnet-5-50", True),                     # prefix match, different model
    ("claude-sonnet-5-5-1", True),                    # prefix match, different (point) model
    ("claude-sonnet-5-5-mini", True),
    ("claude-sonnet-5-5-2026100", True),              # 7 digits is not a snapshot date
    ("claude-sonnet-5-5-20261001-x", True),
    ("claude-haiku-5-5", True),
    ("claude-sonnet-5", True),                        # shorter id
    ("", True), (None, True),
])
def test_model_id_check_is_exact_or_dated_snapshot(returned, fb):
    assert draft.is_fallback("claude-sonnet-5-5", returned) is fb


def test_prefix_but_different_model_is_rc2_in_run(env):
    _ready(env)
    r = draft.run(env.pd, env.led, ScriptedClient(model="claude-sonnet-5-5-1"), env.builder)
    assert r["sections"][0]["rc"] == 2 and "fallback" in r["halted"] and len(r["sections"]) == 1
    assert (env.pd / "quarantine" / "intro.md").exists() and not (env.pd / "out" / "intro.md").exists()


# ------------------------------------------------------------------------------------- verifier r1 gaps (NB-1, NB-2)
def test_run_reapplies_deny_path_glob_added_after_plan(env, deny_files):
    """NB-1: a deny `paths:` glob added after plan refuses at run (content of notes.md stays clean)."""
    d = _ready(env)
    assert all("notes.md" in r["meta"]["source_paths"] for r in draft.load_plan(env.pd)[1])
    deny_files[1].write_text(deny_files[1].read_text() + "  - id: late-glob\n    status: withheld\n    kind: sealed-artifact\n"
                             "    patterns: ['zz_never_in_text_zz']\n    paths: ['**/notes.md']\n", encoding="utf-8")
    fresh = PackBuilder.from_files(deny_files, lab_root=env.lab)
    fake = FakeClient()
    with pytest.raises(ops.Refused, match="deny glob"):
        draft.run(env.pd, env.led, fake, fresh)
    assert fake.sync_sent == [] and env.led.rows("sync_sent") == []
    assert draft.run(env.pd, env.led, FakeClient(), env.builder)["plan_hash"] == d["plan_hash"]   # old deny: runs


def test_run_rechecks_day_month_limits_after_approve(env):
    """_limits_ok at run: committed spend that grew after approve refuses before any send."""
    _ready(env)
    env.led.append("approved", manifest_sha256="other", est_usd_hard_worst=189.999)
    fake = FakeClient()
    with pytest.raises(ops.Refused, match="committed"):
        draft.run(env.pd, env.led, fake, env.builder)
    assert fake.sync_sent == [] and env.led.rows("sync_sent") == []


def _rederive_with_fallbacks(pd):
    """Consistent tamper: add `fallbacks` to one manifest row and re-derive every hash plan.json carries."""
    mf, pj = pd / "manifest.jsonl", pd / "plan.json"
    rows = [json.loads(ln) for ln in mf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows[0]["params"]["fallbacks"] = ["claude-haiku-5-5"]
    rows[0]["params_sha256"] = common.sha256_text(common.canon_json(rows[0]["params"]))
    data = ("\n".join(common.canon_json(r) for r in rows) + "\n").encode("utf-8")
    mf.write_bytes(data)
    msha = common.sha256_bytes(data)
    worst = sum(draft.request_hard_worst(r["params"]) for r in rows)
    d = json.loads(pj.read_text(encoding="utf-8"))
    d.update(manifest_sha256=msha, est_usd_hard_worst=round(worst, 6),
             plan_hash=draft.compute_plan_hash(msha, worst, draft.COST_FORMULA))
    pj.write_text(json.dumps(d), encoding="utf-8")
    return d, worst


def test_manifest_fallbacks_refused_at_load_and_run_even_when_rederived(env):
    """_no_fallback: a re-derived manifest carrying `fallbacks` passes every hash check but is still refused."""
    ops.selftest(env.led, runner=lambda: 0)
    _plan(env)
    d, worst = _rederive_with_fallbacks(env.pd)
    with pytest.raises(ops.Refused, match="fallback"):
        draft.load_plan(env.pd)
    with pytest.raises(ops.Refused, match="fallback"):
        draft.sync_plan_for_approve(env.pd)
    ops.approve(None, round(worst, 6), env.led, env={},                    # even with a matching approval row
                sync_plan={"plan_hash": d["plan_hash"], "hard_worst_usd": round(worst, 6)})
    fake = FakeClient()
    with pytest.raises(ops.Refused, match="fallback"):
        draft.run(env.pd, env.led, fake, env.builder)
    assert fake.sync_sent == [] and env.led.rows("sync_sent") == []


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_non_success_stop_is_rc4_pending_and_cli_exit_5(env, stop, capsys):
    """rc=4 path: a max_tokens / refusal reply is recorded rc 4, the section stays pending, CLI exits 5."""
    d = _ready(env)
    r = draft.run(env.pd, env.led, ScriptedClient(stop_reason=stop), env.builder)
    assert [s["rc"] for s in r["sections"]] == [4, 4] and all(s["result_status"] == stop for s in r["sections"])
    assert [x["rc"] for x in env.led.rows("sync_done")] == [4, 4]
    rows = draft.load_plan(env.pd)[1]
    assert [x["custom_id"] for x in draft.pending_sections(env.pd, env.led, d["plan_hash"], rows)] == \
        ["draft-intro", "draft-results"]
    assert f"result_status: {stop}" in (env.pd / "out" / "intro.md").read_text(encoding="utf-8")
    _approve(env)
    argv = ["--batch-root", str(env.root), "--lab-root", str(env.lab), *sum((["--deny-file", str(f)] for f in env.deny_files), [])]
    assert cli.main([*argv, "draft-sync", "run", str(env.pd)], client=ScriptedClient(stop_reason=stop)) == 5


def test_last_section_overspend_halts_after_send(env):
    """post-send budget halt: the pre-send check passes for the last section, its billed cost then exceeds the approval."""
    _ready(env)
    fake = ScriptedClient(out_mult_on={"draft-results": 50})
    r = draft.run(env.pd, env.led, fake, env.builder)
    assert len(fake.sync_sent) == 2 and r["spent_usd_est"] > r["approved_usd"]
    assert r["halted"] is not None and r["halted"].startswith("budget: spent")
    err = env.led.last("error")
    assert err["custom_id"] == "draft-results" and err["error"].startswith("budget: spent")
    assert "halted before send" not in err["error"]


def test_batch_lane_approval_does_not_authorise_draft_sync_run(env):      # W1 hardening (2026-11-03)
    ops.selftest(env.led, runner=lambda: 0)
    d = _plan(env)
    env.led.append("approved", batch_dir=str(env.pd), manifest_sha256=d["plan_hash"], est_usd_hard_worst=d["est_usd_hard_worst"],
                   approval={"by": "user-cli", "usd": 1000.0, "manifest_sha256": d["plan_hash"], "ts": ops.now()},
                   ts=ops.now(), lane="batch", approval_id="batchlane01")
    fake = FakeClient()
    with pytest.raises(ops.Refused, match="no matching user approval"):
        draft.run(env.pd, env.led, fake, env.builder)
    assert fake.sync_sent == []


def test_formula1_plan_hash_is_the_pre_change_format():
    """Golden: the exact canonical JSON the pre-2026-11-05 code hashed (no cost_formula key)."""
    from tools.api_batch.common import sha256_text
    legacy = '{"hard_worst_usd":0.123457,"manifest_sha256":"abc","sync_price_factor":2.0,"use":"draft-sync"}'
    assert draft.compute_plan_hash("abc", 0.1234567) == sha256_text(legacy)
    assert draft.compute_plan_hash("abc", 0.1234567, 2) != sha256_text(legacy)


def test_run_on_a_formula1_plan(env):
    (env.pd / "spec.yaml").write_text(SPEC + "cache_ttl: none\n", encoding="utf-8")
    ops.selftest(env.led, runner=lambda: 0)
    _plan(env)
    pj = env.pd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    d.pop("cost_formula")
    d["plan_hash"] = draft.compute_plan_hash(d["manifest_sha256"], d["est_usd_hard_worst"])
    pj.write_text(json.dumps(d), encoding="utf-8")
    _approve(env)
    r = draft.run(env.pd, env.led, FakeClient(), env.builder)
    assert r["halted"] is None and [x["rc"] for x in r["sections"]] == [0, 0]
