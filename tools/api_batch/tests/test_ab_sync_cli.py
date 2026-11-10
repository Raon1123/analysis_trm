import json
import sys
import types

import pytest

from tools.api_batch import cli, common, ops
from tools.api_batch.common import Halt, sha256_text
from tools.api_batch.ledger import Ledger


@pytest.fixture
def led(tmp_path):
    return Ledger(tmp_path / "scratch" / "ledger.jsonl")


@pytest.fixture
def inputs(lab):
    r, p = lab / "RESULTS.md", lab / "PREREG.md"
    r.write_text("# RESULTS\nacc 0.5 — 한글 포함\n", encoding="utf-8")
    p.write_text("# PREREG\nplan text\n", encoding="utf-8")
    return r, p


class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, plan_path, out, budget):
        self.calls.append((plan_path, out, budget))
        return 0


def test_selftest_writes_row_only_on_pass(led):
    assert ops.selftest(led, runner=lambda: 1) == 1 and led.rows("tests_passed") == []
    assert ops.selftest(led, runner=lambda: 0) == 0
    r = led.last("tests_passed")
    assert r["pack_py_sha256"] and set(r["trm_mcp"]) == {"withheld.py", "research_digest.py"}
    assert set(r["frozen_doc_lint"]) == {"doc_lint.py", "doc_lint_prompt.md", "doc_lint_schema.json"}
    assert any(n.startswith("test_ab_") for n in r["test_files"])
    assert ops.tests_passed_current(led) == (True, "ok")


def test_tests_passed_goes_stale(led):
    ops.selftest(led, runner=lambda: 0)
    row = dict(led.last("tests_passed"))
    for k in ("pack_py_sha256", "deny_py_sha256"):
        led.path.write_text(json.dumps({**row, k: "0" * 64}) + "\n")
        ok, why = ops.tests_passed_current(led)
        assert not ok and k in why


def test_doclint_sync_lapses_without_tests_passed(builder, led, inputs):
    spy = Spy()
    with pytest.raises(ops.Refused, match="tests_passed"):
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)
    assert spy.calls == [] and led.rows("scanned") == []


def test_doclint_sync_needs_verifier_ref_and_clean_scan(builder, led, inputs, lab):
    ops.selftest(led, runner=lambda: 0)
    spy = Spy()
    with pytest.raises(ops.Refused, match="verifier"):
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="", runner=spy)
    dirty = lab / "DIRTY.md"
    dirty.write_text("this mentions EXP-013 outcomes", encoding="utf-8")
    with pytest.raises(ops.Refused, match="scan"):
        ops.doclint_sync(dirty, inputs[1], builder, led, p0a_verifier_ref="ref", runner=spy)
    assert spy.calls == [] and led.last("scanned")["exit_code"] == 1       # scan row written, doc_lint.py never called


def test_doclint_sync_happy_path_uses_scanned_copies(builder, led, inputs):
    ops.selftest(led, runner=lambda: 0)
    plan = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="VERIFY-P0A-r1", plan_only=True)
    with pytest.raises(ops.Refused):                                       # no approval keyed by the plan hash yet
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="VERIFY-P0A-r1", runner=Spy())
    ops.approve(None, plan["hard_worst_usd"] + 0.01, led, env={}, sync_plan={"plan_hash": plan["plan_hash"], "hard_worst_usd": plan["hard_worst_usd"]})
    spy = Spy()
    out = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="VERIFY-P0A-r1", runner=spy)
    assert out["rc"] == 0 and out["exposure_incident"] is None and len(spy.calls) == 1
    srow = [r for r in led.rows("scanned")][-1]
    res_text = inputs[0].read_text(encoding="utf-8")
    assert srow["files"][0]["text_sha256"] == sha256_text(res_text)        # doc_lint.py:130-131 method
    plan_json = json.loads(open(spy.calls[0][0]).read())
    from pathlib import Path
    assert Path(plan_json[0]["results"]).read_text(encoding="utf-8") == res_text and plan_json[0]["results"] != str(inputs[0])
    ev = [r["event"] for r in led.rows()]
    assert ev.index("sync_sent") > max(i for i, e in enumerate(ev) if e == "scanned") and ev[-1] == "sync_done"


def test_doclint_sync_flags_exposure_incident(builder, led, inputs):
    ops.selftest(led, runner=lambda: 0)
    plan = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", plan_only=True)
    ops.approve(None, 5.0, led, env={}, sync_plan={"plan_hash": plan["plan_hash"], "hard_worst_usd": plan["hard_worst_usd"]})

    def bad_runner(plan_path, out, budget):
        out.mkdir(parents=True)
        (out / "ledger.jsonl").write_text(json.dumps({"event": "sent", "results_sha256": "dead", "prereg_sha256": "beef"}) + "\n")
        return 0
    r = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=bad_runner)
    assert r["exposure_incident"] and any("EXPOSURE" in e["error"] for e in led.rows("error"))


def test_scan_cli_exit_codes(deny_files, tmp_path, inputs, lab):
    argv = ["--batch-root", str(tmp_path / "b"), "--lab-root", str(lab), *sum((["--deny-file", str(f)] for f in deny_files), [])]
    assert cli.main([*argv, "scan", "--verify-only", str(inputs[0])]) == 0
    bad = lab / "bad.md"
    bad.write_text("EXP 013 results")
    assert cli.main([*argv, "scan", "--verify-only", str(bad)]) == 1
    rows = Ledger(tmp_path / "b" / "ledger.jsonl").rows("scanned")
    assert [r["exit_code"] for r in rows] == [0, 1]
    # deny source absent -> halt (rc 3), no scanned row
    assert cli.main(["--batch-root", str(tmp_path / "b"), "--lab-root", str(lab), "--deny-file", str(tmp_path / "nope.yaml"), "scan", "--verify-only", str(bad)]) == 3
    assert len(Ledger(tmp_path / "b" / "ledger.jsonl").rows("scanned")) == 2


def test_tools_path_assertion(monkeypatch):
    common.check_tools_path()                                              # real environment passes
    bogus = types.ModuleType("tools")
    bogus.__path__ = ["/nonexistent/site-packages/tools"]
    monkeypatch.setitem(sys.modules, "tools", bogus)
    with pytest.raises(Halt):
        common.check_tools_path()
    assert cli.main(["selftest"]) == 2
    regular = types.ModuleType("tools")
    regular.__path__ = [str(common.TOOLS_DIR)]
    regular.__file__ = "/site-packages/tools/__init__.py"
    monkeypatch.setitem(sys.modules, "tools", regular)
    with pytest.raises(Halt):
        common.check_tools_path()


def test_session_marker_detection():
    assert common.in_claude_code_session({"CLAUDECODE": "1"}) == "CLAUDECODE"
    assert common.in_claude_code_session({"CLAUDE_CODE_SESSION_ID": "abc"}) == "CLAUDE_CODE_SESSION_ID"
    assert common.in_claude_code_session({"CLAUDECODE": "", "PATH": "/bin"}) is None


def test_cli_stubs_and_new(tmp_path):
    assert cli.main(["--batch-root", str(tmp_path), "triage"]) == 4
    assert cli.main(["--batch-root", str(tmp_path), "new", "my-slug", "--use", "U3"]) == 0
    from tools.api_batch.spec import load_spec
    d = next((tmp_path / "11_batches").iterdir())
    assert load_spec(d / "spec.yaml")["name"] == "my-slug"


def _approve_sync(led, plan, usd=5.0):
    return ops.approve(None, usd, led, env={}, sync_plan={"plan_hash": plan["plan_hash"], "hard_worst_usd": plan["hard_worst_usd"]})


def test_doclint_sync_second_send_needs_new_approval_nb1(builder, led, inputs):
    ops.selftest(led, runner=lambda: 0)
    plan = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", plan_only=True)
    _approve_sync(led, plan)
    spy = Spy()
    assert ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)["rc"] == 0
    with pytest.raises(ops.Refused, match="consumed"):                    # same inputs, same approval
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)
    assert len(spy.calls) == 1 and len(led.rows("sync_sent")) == 1
    assert led.last("sync_sent")["approval_id"] == led.last("approved")["approval_id"]
    _approve_sync(led, plan)                                              # a new approval buys exactly one more send
    ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)
    assert len(spy.calls) == 2
    with pytest.raises(ops.Refused, match="consumed"):
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)
    assert led.committed_usd() == pytest.approx(2 * plan["hard_worst_usd"])      # sync lane: one count per approval


def test_failed_runner_still_consumes_the_approval(builder, led, inputs):
    ops.selftest(led, runner=lambda: 0)
    plan = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", plan_only=True)
    _approve_sync(led, plan)
    r = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=lambda *a: 1)
    assert r["rc"] == 1
    with pytest.raises(ops.Refused, match="consumed"):
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=Spy())


def test_doclint_sync_stale_tests_passed_end_to_end_nb5(builder, led, inputs):
    """One test for NB-5 and the 'pack.py sha differs -> doclint-sync refuses' PARTIAL item: the stored
    pack_py_sha256 no longer equals the current pack.py sha, so doclint_sync() itself must refuse before any scan."""
    ops.selftest(led, runner=lambda: 0)
    row = dict(led.last("tests_passed"))
    led.path.write_text(json.dumps({**row, "pack_py_sha256": "0" * 64}) + "\n", encoding="utf-8")
    spy = Spy()
    with pytest.raises(ops.Refused, match="pack_py_sha256"):
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)
    assert spy.calls == [] and led.rows("scanned") == [] and led.rows("sync_sent") == []


def test_approve_sync_plan_needs_worst_nb3(tmp_path, led, monkeypatch, capsys):
    with pytest.raises(ops.Refused, match="sync-worst"):
        ops.approve(None, 5.0, led, env={}, sync_plan={"plan_hash": "h", "hard_worst_usd": 0.0})
    for k in common.SESSION_MARKERS:                                      # so the refusal below is NB-3, not the session guard
        monkeypatch.delenv(k, raising=False)
    argv = ["--batch-root", str(tmp_path / "b"), "approve", "--usd", "5", "--sync-plan-hash", "abc"]
    assert cli.main(argv) == 3 and "--sync-worst" in capsys.readouterr().err
    assert Ledger(tmp_path / "b" / "ledger.jsonl").rows("approved") == []
    assert cli.main([*argv, "--sync-worst", "1.5"]) == 0
    assert Ledger(tmp_path / "b" / "ledger.jsonl").last("approved")["est_usd_hard_worst"] == 1.5


def test_batch_lane_approval_does_not_authorise_doclint_sync(builder, led, inputs):      # W1 hardening (2026-11-03)
    ops.selftest(led, runner=lambda: 0)
    plan = ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", plan_only=True)
    led.append("approved", batch_dir="x", manifest_sha256=plan["plan_hash"], est_usd_hard_worst=plan["hard_worst_usd"],
               approval={"by": "user-cli", "usd": 1000.0, "manifest_sha256": plan["plan_hash"], "ts": ops.now()},
               ts=ops.now(), lane="batch", approval_id="batchlane02")
    spy = Spy()
    with pytest.raises(ops.Refused, match="no matching user approval"):
        ops.doclint_sync(*inputs, builder, led, p0a_verifier_ref="r", runner=spy)
    assert spy.calls == [] and led.rows("sync_sent") == []
