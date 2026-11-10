import json
from pathlib import Path

import pytest

from tools.api_batch import cli, ops
from tools.api_batch.common import Halt, sha256_file
from tools.api_batch.fake_client import FakeApiError, FakeClient
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder

PROBE_SPEC = """name: p0-probe
use: probe
budget: {cap_usd_hard_worst: 0.5}
probe_requests:
  - {id: probe-haiku-s1, model: claude-haiku-5-5, effort: low, max_tokens: 1024, var: p0_haiku_short.txt}
  - {id: probe-sonnet-a-s1, model: claude-sonnet-5-5, effort: medium, max_tokens: 2048, cache_ttl: 1h, prefix: p0_sonnet_prefix.txt, var: p0_sonnet_var_a.txt, schema: probe}
  - {id: probe-sonnet-b-s1, model: claude-sonnet-5-5, effort: medium, max_tokens: 2048, cache_ttl: 1h, prefix: p0_sonnet_prefix.txt, var: p0_sonnet_var_b.txt, schema: probe}
  - {id: probe-fable-s1, model: claude-fable-5-1, effort: low, max_tokens: 4096, var: p0_fable_short.txt, schema: probe}
"""


@pytest.fixture
def env(tmp_path, deny_files, lab):
    root = tmp_path / "root"
    bd = root / "11_batches" / "2026-10-27_p0-probe"
    bd.mkdir(parents=True)
    (bd / "spec.yaml").write_text(PROBE_SPEC, encoding="utf-8")
    builder = PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    led = Ledger(root / "ledger.jsonl")
    return type("E", (), dict(root=root, bd=bd, builder=builder, led=led, deny_files=deny_files))


def planned(env):
    return ops.plan(env.bd, env.builder, env.led)


def approved(env, usd=1.0):
    d = planned(env)
    ops.approve(env.bd, usd, env.led, env={})
    return d


def test_plan_artifacts_and_determinism(env):
    d = planned(env)
    assert d["n_requests"] == 4 and 0 < d["est_usd_hard_worst"] <= 0.5
    for f in ("manifest.jsonl", "plan.json", "pack/pack_manifest.json"):
        assert (env.bd / f).exists()
    row = env.led.last("planned")
    assert row["manifest_sha256"] == d["manifest_sha256"] and row["pack_py_sha256"]
    assert ops.plan(env.bd, env.builder, env.led)["manifest_sha256"] == d["manifest_sha256"]


def test_plan_refuses_over_cap_and_unknown_model(env):
    env.bd.joinpath("spec.yaml").write_text(PROBE_SPEC.replace("cap_usd_hard_worst: 0.5", "cap_usd_hard_worst: 0.01"))
    with pytest.raises(Halt):
        planned(env)
    env.bd.joinpath("spec.yaml").write_text(PROBE_SPEC.replace("claude-haiku-5-5", "claude-nope-1"))
    with pytest.raises(Halt):
        planned(env)


def test_approve_refuses_inside_claude_code_session(env, monkeypatch):
    planned(env)
    for marker in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID"):
        for k in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(marker, "1")
        with pytest.raises(ops.Refused, match="Claude Code session"):
            ops.approve(env.bd, 1.0, env.led)
        assert cli.main(["--batch-root", str(env.root), "approve", str(env.bd), "--usd", "1.0"]) == 3
        assert env.led.rows("approved") == []
    for k in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(k, raising=False)
    assert cli.main(["--batch-root", str(env.root), "approve", str(env.bd), "--usd", "1.0"]) == 0
    assert env.led.last("approved")["approval"]["by"] == "user-cli"


def test_approve_amount_and_limits(env):
    d = planned(env)
    with pytest.raises(ops.Refused):
        ops.approve(env.bd, d["est_usd_hard_worst"] / 2, env.led, env={})
    env.led.append("approved", manifest_sha256="other", est_usd_hard_worst=189.99)    # day already near the $190 cap (2026-10-30)
    with pytest.raises(ops.Refused):
        ops.approve(env.bd, 1.0, env.led, env={})


def test_submit_requires_matching_approval_and_is_idempotent(env):
    planned(env)
    fake = FakeClient()
    with pytest.raises(ops.Refused):
        ops.submit(env.bd, env.led, fake)
    assert fake.calls == []
    ops.approve(env.bd, 1.0, env.led, env={})
    row = ops.submit(env.bd, env.led, fake)
    assert row["batch_id"].startswith("msgbatch_fake") and fake.calls == ["create"]
    with pytest.raises(ops.Refused):                                      # same manifest again
        ops.submit(env.bd, env.led, fake)
    assert fake.calls == ["create"]
    # editing the manifest after approval invalidates it
    (env.bd / "manifest.jsonl").write_text((env.bd / "manifest.jsonl").read_text().replace('"max_tokens":1024', '"max_tokens":1025'))
    with pytest.raises(Halt):
        ops.submit(env.bd, env.led, FakeClient())


class CrashLedger(Ledger):
    def append(self, event, **kw):
        if event == "submitted":
            raise RuntimeError("simulated crash after create, before the ledger row")
        return super().append(event, **kw)


def test_crash_between_create_and_ledger_row_recovers_without_resubmit(env):
    approved(env)
    fake = FakeClient()
    with pytest.raises(RuntimeError):
        ops.submit(env.bd, CrashLedger(env.led.path), fake)
    assert len(fake.batches) == 1 and env.led.rows("submitted") == [] and len(env.led.rows("submitting")) == 1
    with pytest.raises(ops.Refused):                                      # no automatic resubmission
        ops.submit(env.bd, env.led, fake)
    assert len(fake.batches) == 1
    r = ops.submit(env.bd, env.led, fake, recover=True)
    assert r["recovered"] and r["batch_id"] == "msgbatch_fake_0001"
    with pytest.raises(ops.Refused):
        ops.submit(env.bd, env.led, fake, recover=True)


def test_recover_ambiguous_stops(env):
    approved(env)
    fake = FakeClient()
    with pytest.raises(RuntimeError):
        ops.submit(env.bd, CrashLedger(env.led.path), fake)
    fake.create([{"custom_id": f"q{i}", "params": {"model": "m", "max_tokens": 1}} for i in range(4)])   # a second 4-request batch
    with pytest.raises(ops.Refused, match="exactly 1"):
        ops.submit(env.bd, env.led, fake, recover=True)


def submitted(env):
    approved(env)
    fake = FakeClient(seed=7)
    bid = ops.submit(env.bd, env.led, fake)["batch_id"]
    return fake, bid


def test_collect_shuffled_refusal_expired_and_idempotent(env):
    fake, bid = submitted(env)
    with pytest.raises(ops.Refused):                                     # not ended yet
        ops.collect(env.bd, env.led, fake, env.root, env.builder)
    fake.finish(bid, {"probe-haiku-s1": "refusal", "probe-fable-s1": "expired", "probe-sonnet-a-s1": "cache_write",
                      "probe-sonnet-b-s1": "cache_hit"})
    ops.status(env.bd, env.led, fake)
    out = ops.collect(env.bd, env.led, fake, env.root, env.builder)
    assert out["counts"] == {"refusal": 1, "expired": 1, "succeeded": 2} and out["fetched"]
    ids = [json.loads(l)["custom_id"] for l in (env.bd / "raw/results.jsonl").read_text().splitlines()]
    assert sorted(ids) == sorted(["probe-haiku-s1", "probe-fable-s1", "probe-sonnet-a-s1", "probe-sonnet-b-s1"])
    inbox = [json.loads(l) for l in (env.root / "INBOX.jsonl").read_text().splitlines()]
    assert {r["custom_id"]: r["result_status"] for r in inbox}["probe-haiku-s1"] == "refusal"
    item = (env.bd / "items/probe-sonnet-a-s1.md").read_text()
    assert "status: DRAFT (api-batch output — not evidence)" in item and "l5: pass" in item and "model_returned: claude-sonnet-5-5" in item
    col = env.led.last("collected")
    assert col["released_usd"] > 0 and col["refusals"] == {"other": 1} and col["r_hit_measured"] == pytest.approx(0.5)
    u = col["usage_totals"]
    assert col["cache_coverage"] == round(u["cache_read_input_tokens"] / (u["cache_read_input_tokens"]
                                          + u["cache_creation_input_tokens"] + u["input_tokens"]), 6) < col["r_hit_measured"]
    assert env.led.rows("deleted") and "delete" in fake.calls
    # re-run: no new fetch, no second collected/deleted row
    n_rows = len(env.led.rows())
    calls = list(fake.calls)
    out2 = ops.collect(env.bd, env.led, fake, env.root, env.builder)
    assert not out2["fetched"] and fake.calls == calls and len(env.led.rows()) == n_rows
    ops.collect(env.bd, env.led, None, env.root, env.builder, offline=True)
    assert len(env.led.rows()) == n_rows


@pytest.mark.parametrize("kw", [dict(drop=["probe-haiku-s1"]), dict(duplicate=["probe-haiku-s1"])])
def test_collect_missing_or_duplicate_ids_do_not_touch_inbox(env, kw):
    fake, bid = submitted(env)
    fake.finish(bid, **kw)
    with pytest.raises(ops.Refused):
        ops.collect(env.bd, env.led, fake, env.root, env.builder)
    assert not (env.root / "INBOX.jsonl").exists() and env.led.rows("error") and not env.led.rows("collected")
    assert not (env.bd / "raw/results.jsonl").exists()


def test_l5_quarantines_denied_output(env):
    fake, bid = submitted(env)
    fake.finish(bid, texts={"probe-haiku-s1": "the cohort EXP-013 reached 0.9"})
    ops.collect(env.bd, env.led, fake, env.root, env.builder)
    assert (env.bd / "quarantine/probe-haiku-s1.md").exists() and not (env.bd / "items/probe-haiku-s1.md").exists()
    ids = [json.loads(l)["custom_id"] for l in (env.root / "INBOX.jsonl").read_text().splitlines()]
    assert "probe-haiku-s1" not in ids and len(ids) == 3


def test_parse_error_and_max_tokens_recorded(env):
    fake, bid = submitted(env)
    fake.finish(bid, {"probe-sonnet-a-s1": "max_tokens"}, texts={"probe-fable-s1": "not json"})
    out = ops.collect(env.bd, env.led, fake, env.root, env.builder)
    assert out["counts"].get("max_tokens") == 1 and out["counts"].get("parse_error") == 1


def test_cancel_and_fallbacks_create_error(env):
    fake, bid = submitted(env)
    ops.cancel(env.bd, env.led, fake)
    assert env.led.rows("canceled")
    f2 = FakeClient()
    with pytest.raises(FakeApiError):
        f2.create([{"custom_id": "a", "params": {"model": "m", "max_tokens": 1, "fallbacks": []}}])
    with pytest.raises(FakeApiError):
        f2.create([{"custom_id": "a", "params": {"model": "m", "max_tokens": 1, "stream": True}}])


def test_probe_sync_needs_double_approval(env):
    planned(env)
    fake = FakeClient()
    ops.approve(env.bd, 1.0, env.led, env={})
    d = json.loads((env.bd / "plan.json").read_text())
    ok = 2 * d["est_usd_hard_worst"] <= 1.0
    if not ok:                                                            # approval of 1x is not enough for standard price
        with pytest.raises(ops.Refused):
            ops.probe_sync(env.bd, env.led, fake)
    ops.approve(env.bd, 2.0, env.led, env={}, sync_factor=2.0)
    assert len(ops.probe_sync(env.bd, env.led, fake, ["probe-haiku-s1"])) == 1
    assert len(env.led.rows("sync_sent")) == 1 and env.led.rows("sync_done")[0]["cost_est_usd"] >= 0


def test_probe_sync_one_approval_one_send_nb2(env):
    d = planned(env)
    fake = FakeClient()
    ops.approve(env.bd, 2.0, env.led, env={}, sync_factor=2.0)
    assert len(ops.probe_sync(env.bd, env.led, fake)) == 4 and len(fake.sync_sent) == 4
    with pytest.raises(ops.Refused, match="consumed"):                   # same approval, same inputs: refused
        ops.probe_sync(env.bd, env.led, fake)
    with pytest.raises(ops.Refused, match="consumed"):                   # a different --ids subset is not a loophole
        ops.probe_sync(env.bd, env.led, fake, ["probe-haiku-s1"])
    assert len(fake.sync_sent) == 4
    ids = {r["approval_id"] for r in env.led.rows("sync_sent")}
    assert len(ids) == 1 and None not in ids                             # every send names the approval it used
    ops.approve(env.bd, 2.0, env.led, env={}, sync_factor=2.0)           # new approval -> exactly one more send
    assert len(ops.probe_sync(env.bd, env.led, fake, ["probe-haiku-s1"])) == 1
    assert len(env.led.rows("sync_sent")) == 5
    # committed_usd: the batch-lane 1x approval counts once (max), each sync approval counts on its own
    w = d["est_usd_hard_worst"]
    assert env.led.committed_usd() == pytest.approx(2 * (2 * w))


def test_committed_usd_upgraded_batch_approval_is_max_not_sum(env):
    d = planned(env)
    w = d["est_usd_hard_worst"]
    ops.approve(env.bd, 1.0, env.led, env={})
    ops.approve(env.bd, 1.0, env.led, env={})                            # repeated 1x batch approval, one submit possible
    assert env.led.committed_usd() == pytest.approx(w)
    env.led.append("approved", manifest_sha256=d["manifest_sha256"], est_usd_hard_worst=2 * w)    # old-shape row, larger worst
    assert env.led.committed_usd() == pytest.approx(2 * w)               # replaces, does not add (and is not under-counted)


def test_lane_isolation_probe_approval_does_not_authorise_submit_and_vice_versa(env):
    d = planned(env); w = d["est_usd_hard_worst"]
    ops.approve(env.bd, round(2 * w + 0.01, 2), env.led, env={}, sync_factor=2.0)     # probe-sync lane only
    with pytest.raises(ops.Refused, match="no matching user approval"):
        ops.submit(env.bd, env.led, FakeClient())
    ops.approve(env.bd, round(2 * w + 0.01, 2), env.led, env={})                       # batch lane at >= 2x
    ops.submit(env.bd, env.led, FakeClient())                                           # batch approval satisfies submit
    env.led.path.write_text("")
    d = planned(env)
    ops.approve(env.bd, round(2 * w + 0.01, 2), env.led, env={})                       # batch lane only
    with pytest.raises(ops.Refused, match="no matching user approval"):
        ops.probe_sync(env.bd, env.led, FakeClient())


def test_plan_json_dollar_figure_tamper_refused(env):
    planned(env)
    pj = env.bd / "plan.json"
    d = json.loads(pj.read_text()); d["est_usd_hard_worst"] = 0.0001; pj.write_text(json.dumps(d))
    with pytest.raises(ops.Refused, match="re-derived"):
        ops.approve(env.bd, 0.001, env.led, env={})


def test_plan_json_small_dollar_gap_refused_at_1e6_tolerance(env):               # W2 hardening (2026-11-03)
    planned(env)
    pj = env.bd / "plan.json"
    d = json.loads(pj.read_text()); d["est_usd_hard_worst"] = d["est_usd_hard_worst"] - 5e-6; pj.write_text(json.dumps(d))   # inside the old 1e-5 tolerance
    with pytest.raises(ops.Refused, match="re-derived"):
        ops.approve(env.bd, 1.0, env.led, env={})
