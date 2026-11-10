"""allow_files: sha-pinned verify-only exception for the seat-doclint lane (decision 2026-10-27). Only the mechanism is
tested here (the real entries are user-approved and live in API_BATCH_DENY.yaml)."""
import json

import pytest
import yaml

from tools.api_batch import cli, deny, ops
from tools.api_batch.common import Halt, sha256_bytes
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder, PackError

RES = "---\nexp_id: EXP-016\n---\n# RESULTS\nsee per_call.csv and the EXP-016 table\n"
PRE = "# PREREG\nplan text, nothing denied\n"
SPEC = """name: p1-lint
use: seat-doclint
model: claude-opus-5-5
n: 1
inputs:
  results: 23_reports/RESULTS.md
  prereg: 23_reports/PREREG.md
budget: {cap_usd_hard_worst: 1.30}
"""
RELP = "23_reports/RESULTS.md"


def entry(path=RELP, sha=None, lanes=("seat-doclint",)):
    return {"path": path, "sha256": sha or sha256_bytes(RES.encode("utf-8")), "lanes": list(lanes),
            "reason": "test exception", "approved_by": "user", "approved_on": "2026-10-27"}


@pytest.fixture
def env(tmp_path, deny_files, lab):
    (lab / "23_reports").mkdir()
    (lab / "23_reports/RESULTS.md").write_text(RES, encoding="utf-8")
    (lab / "23_reports/PREREG.md").write_text(PRE, encoding="utf-8")
    bd = tmp_path / "root/11_batches/b"
    bd.mkdir(parents=True)
    (bd / "spec.yaml").write_text(SPEC, encoding="utf-8")

    def mk(entries):
        doc = yaml.safe_load(deny_files[1].read_text(encoding="utf-8"))
        doc["allow_files"] = entries
        deny_files[1].write_text(yaml.safe_dump(doc), encoding="utf-8")
        return PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    return type("E", (), dict(mk=mk, bd=bd, lab=lab, led=Ledger(tmp_path / "root/ledger.jsonl"), deny_files=deny_files))


def test_baseline_refuses_without_entry(env):
    b = env.mk([])
    with pytest.raises(PackError, match="L2"):
        b.build_verify_only("seat_input", "RESULTS.md", RES, origin=env.lab / RELP, lane="seat-doclint")


def test_allowed_pass_records_entry_and_hits(env):
    b = env.mk([entry()])
    pk = b.build_verify_only("seat_input", "RESULTS.md", RES, origin=env.lab / RELP, lane="seat-doclint")
    assert pk.sealed and pk.report["allowed_by"]["path"] == RELP
    kinds = {(h["layer"], h["kind"]) for h in pk.report["allowed_hits"]}
    assert ("L2", "frontmatter") in kinds and ("L3b", "body") in kinds
    assert any("per_call" in h["hit"] for h in pk.report["allowed_hits"])
    assert pk.manifest_entry()["report"]["allowed_by"]["sha256"] == entry()["sha256"]


def test_sha_changed_refuses_with_reason(env):
    b = env.mk([entry(sha="0" * 64)])
    with pytest.raises(PackError, match="different sha256"):
        b.build_verify_only("seat_input", "RESULTS.md", RES, origin=env.lab / RELP, lane="seat-doclint")
    # file edited after the entry was written
    b = env.mk([entry()])
    (env.lab / RELP).write_text(RES + "one more line\n", encoding="utf-8")
    with pytest.raises(PackError, match="different sha256"):
        b.build_verify_only("seat_input", "RESULTS.md", (env.lab / RELP).read_text(encoding="utf-8"),
                            origin=env.lab / RELP, lane="seat-doclint")


def test_other_lane_or_no_lane_or_other_path_keeps_refusal(env):
    b = env.mk([entry()])
    with pytest.raises(PackError):
        b.build_verify_only("seat_input", "RESULTS.md", RES, origin=env.lab / RELP, lane=None)
    with pytest.raises(PackError):
        b.build_verify_only("seat_input", "RESULTS.md", RES, origin=env.lab / RELP, lane="U1")
    (env.lab / "23_reports/OTHER.md").write_text(RES, encoding="utf-8")        # same bytes, other path
    with pytest.raises(PackError):
        b.build_verify_only("seat_input", "OTHER.md", RES, origin=env.lab / "23_reports/OTHER.md", lane="seat-doclint")
    b2 = env.mk([entry(lanes=["seat-doclint"])])
    with pytest.raises(PackError):                                              # origin-less (spec text) never excepted
        b2.build_verify_only("spec_text", "x", RES, lane="seat-doclint")


def test_plan_carries_allowed_by_and_hits_other_lane_scan_refuses(env):
    b = env.mk([entry()])
    d = ops.plan(env.bd, b, env.led)
    assert d["allowed_files"][0]["allowed_by"]["path"] == RELP and d["allowed_files"][0]["allowed_hits"]
    sc = [x for x in d["scan"] if x.get("allowed_by")]
    assert len(sc) == 1 and sc[0]["allowed_hits"]
    pr = env.led.last("planned")
    assert pr["allowed_files"][0]["path"] == RELP and pr["allowed_files"][0]["n_hits"] >= 2
    # the stored plan.json keeps it too
    assert json.loads((env.bd / "plan.json").read_text())["allowed_files"]
    # same file through a non-doclint scan -> refused; with the lane -> pass and recorded in the scanned row
    p = env.lab / RELP
    assert ops.scan_verify_only([p], b, env.led) == 1
    assert ops.scan_verify_only([p], b, env.led, lane="seat-doclint") == 0
    row = env.led.last("scanned")
    assert row["lane"] == "seat-doclint" and row["files"][0]["allowed_by"]["path"] == RELP and row["files"][0]["allowed_hits"]


def test_plan_refuses_when_sha_stale(env):
    b = env.mk([entry(sha="1" * 64)])
    with pytest.raises(Halt):
        ops.plan(env.bd, b, env.led)
    assert not (env.bd / "manifest.jsonl").exists()


def test_deny_sha_changes_with_entries(env):
    a = env.mk([]).deny.sha256
    assert env.mk([entry()]).deny.sha256 != a


@pytest.mark.parametrize("bad", [
    {"path": "/abs/x.md"}, {"path": "../x.md"}, {"sha256": "ABC"}, {"sha256": "a" * 63}, {"lanes": []}, {"lanes": ["U1"]},
    {"lanes": "seat-doclint"}, {"reason": ""}, {"approved_by": "claude"}, {"approved_on": "yesterday"}])
def test_loader_validates_entries(env, bad):
    with pytest.raises(Halt, match="allow_files"):
        env.mk([{**entry(), **bad}])


def test_loader_rejects_missing_or_extra_keys_and_non_list(env):
    e = entry(); e.pop("reason")
    with pytest.raises(Halt, match="allow_files"):
        env.mk([e])
    with pytest.raises(Halt, match="allow_files"):
        env.mk([{**entry(), "note": "x"}])
    with pytest.raises(Halt, match="allow_files"):
        env.mk({"path": "x"})
    assert env.mk(None).deny.allow_files == []


def test_cli_scan_lane_flag(env, capsys):
    env.mk([entry()])
    base = ["--batch-root", str(env.bd.parent.parent), "--lab-root", str(env.lab), "--ledger", str(env.led.path)]
    for f in env.deny_files:
        base += ["--deny-file", str(f)]
    p = str(env.lab / RELP)
    assert cli.main([*base, "scan", "--verify-only", p]) == 1
    assert cli.main([*base, "scan", "--verify-only", "--lane", "seat-doclint", p]) == 0
