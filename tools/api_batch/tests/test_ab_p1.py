"""P1: batch-path doc-lint shadow (design §8.8): spec form `use: seat-doclint`, verify-only inputs, frozen build_params."""
import json

import pytest

from tools.api_batch import cli, ops
from tools.api_batch.common import Halt, canon_json, sha256_text
from tools.api_batch.doclint_bridge import doc_lint_module
from tools.api_batch.fake_client import FakeClient
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder
from tools.api_batch.spec import SpecError, validate_spec

RES = "# RESULTS\nacc 0.5 — 한글 포함\nthe table says 12 rows\n"
PRE = "# PREREG\nplan text\n"
SPEC = """name: p1-lint
use: seat-doclint
model: claude-opus-5-5
n: 3
inputs:
  results: 23_reports/RESULTS.md
  prereg: 23_reports/PREREG.md
budget: {cap_usd_hard_worst: 1.30}
"""


@pytest.fixture
def env(tmp_path, deny_files, lab):
    (lab / "23_reports").mkdir()
    (lab / "23_reports/RESULTS.md").write_text(RES, encoding="utf-8")
    (lab / "23_reports/PREREG.md").write_text(PRE, encoding="utf-8")
    root = tmp_path / "root"
    bd = root / "11_batches" / "2026-10-27_p1"
    bd.mkdir(parents=True)
    (bd / "spec.yaml").write_text(SPEC, encoding="utf-8")
    builder = PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    return type("E", (), dict(root=root, bd=bd, builder=builder, led=Ledger(root / "ledger.jsonl"), lab=lab))


def test_plan_builds_frozen_requests_with_design_custom_ids(env):
    d = ops.plan(env.bd, env.builder, env.led)
    dl = doc_lint_module()
    rows = [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]
    src8 = sha256_text(RES)[:8]
    assert sorted(r["custom_id"] for r in rows) == [f"lint-{src8}-o-s{j}" for j in (1, 2, 3)]
    assert d["n_requests"] == 3 and 0 < d["est_usd_hard_worst"] <= 1.30
    for r in rows:                                                   # bytes are exactly the frozen build_params of the scanned texts
        assert r["params"] == dl.build_params("claude-opus-5-5", RES, PRE)
    assert d["lint_inputs"]["results_sha256"] == sha256_text(RES) and d["lint_inputs"]["prereg_sha256"] == sha256_text(PRE)
    assert {p["mode"] for p in d["scan"] if p["kind"] == "seat_input"} == {"verify-only"}
    assert env.led.last("planned")["manifest_sha256"] == d["manifest_sha256"]


@pytest.mark.parametrize("which,body", [("RESULTS.md", "this mentions EXP-013 outcomes"), ("PREREG.md", "run fig1_tf_z_iter_k3_s2 here"),
                                        ("RESULTS.md", "---\nexp_id: EXP-016\n---\nbody")])
def test_plan_refuses_denied_input_and_writes_nothing(env, which, body):
    (env.lab / "23_reports" / which).write_text(body, encoding="utf-8")
    with pytest.raises(Halt):
        ops.plan(env.bd, env.builder, env.led)
    assert not (env.bd / "manifest.jsonl").exists() and not (env.bd / "plan.json").exists() and env.led.rows("planned") == []


def test_plan_refuses_input_outside_lab(env, tmp_path):
    out = tmp_path / "x.md"
    out.write_text(RES, encoding="utf-8")
    (env.bd / "spec.yaml").write_text(SPEC.replace("23_reports/RESULTS.md", str(out)), encoding="utf-8")
    with pytest.raises(Halt, match="L1"):
        ops.plan(env.bd, env.builder, env.led)


def test_plan_refuses_over_cap(env):
    (env.bd / "spec.yaml").write_text(SPEC.replace("1.30", "0.5"), encoding="utf-8")
    with pytest.raises(ops.Refused, match="cap"):
        ops.plan(env.bd, env.builder, env.led)


@pytest.mark.parametrize("old,new", [("claude-opus-5-5", "claude-sonnet-5-5"), ("n: 3", "n: 4"), ("n: 3", "n: 0"),
                                     ("inputs:\n  results: 23_reports/RESULTS.md\n  prereg: 23_reports/PREREG.md\n", "inputs: {results: a}\n")])
def test_spec_form_is_strict(old, new):
    import yaml
    with pytest.raises(SpecError):
        validate_spec(yaml.safe_load(SPEC.replace(old, new)))
    assert validate_spec(yaml.safe_load(SPEC))["n"] == 3


def test_batch_path_end_to_end_with_quote_checks(env):
    d = ops.plan(env.bd, env.builder, env.led)
    ops.approve(env.bd, 1.30, env.led, env={})
    fake = FakeClient(seed=3)
    bid = ops.submit(env.bd, env.led, fake)["batch_id"]
    cids = sorted(json.loads(ln)["custom_id"] for ln in (env.bd / "manifest.jsonl").read_text().splitlines())
    f = lambda q: json.dumps({"findings": [{"category": "x", "severity": "low", "location": "l", "quote_source": "RESULTS", "quote": q,
                                            "description": "d", "suggested_fix": "s"}]})
    fake.finish(bid, texts={cids[0]: f("the table says 12 rows"), cids[1]: f("the table   says\n12 rows"), cids[2]: f("not in the text")})
    ops.collect(env.bd, env.led, fake, env.root, env.builder)
    heads = {c: (env.bd / "items" / f"{c}.md").read_text() for c in cids}
    assert "quote_checks: findings=1 exact=1 ws_normalized=1" in heads[cids[0]]
    assert "quote_checks: findings=1 exact=0 ws_normalized=1" in heads[cids[1]]
    assert "quote_checks: findings=1 exact=0 ws_normalized=0" in heads[cids[2]]
    assert d["manifest_sha256"] == env.led.last("collected")["manifest_sha256"]
    # a probe item has no quote_checks line
    assert env.led.last("collected")["counts"] == {"succeeded": 3}


def test_doclint_batch_plan_via_cli(env, deny_files):
    argv = ["--batch-root", str(env.root), "--lab-root", str(env.lab), *sum((["--deny-file", str(f)] for f in deny_files), [])]
    assert cli.main([*argv, "plan", str(env.bd)]) == 0
    (env.lab / "23_reports/RESULTS.md").write_text("EXP 013 leaked", encoding="utf-8")
    assert cli.main([*argv, "plan", str(env.bd)]) == 3
