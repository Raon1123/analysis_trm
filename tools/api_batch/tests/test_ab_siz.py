"""SIZ-map / SIZ-reduce (state/iteration/z synthesis, PREREG §9 S2/S3): map = one request per S2 target cell, one
model per spec (Sonnet/Opus twins on the same packets); reduce = one request per question over both collected maps.
Synthetic S1 fixtures (tests/fixtures/state_iter_z_synthesis) only, tmp lab/ledger; the real staged templates."""
import json
import shutil

import pytest
import yaml

from tools.api_batch import manifest as M
from tools.api_batch import ops
from tools.api_batch.common import REPO_ROOT, Halt
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder
from tools.api_batch.spec import SpecError, load_spec, validate_spec
from tools.state_iter_z_synthesis import packets as P

TPL_REL = f"lab/{P.SYNTH_REL}/S23_templates"
TPL = REPO_ROOT / TPL_REL
FIX = REPO_ROOT / "tests/fixtures/state_iter_z_synthesis"
# mirror of the three active API_BATCH_DENY.yaml families (sealed-dirs / preds / z-snapshots)
ACTIVE_DENY_MIRROR = """version: 1
families:
  - id: sealed-dirs
    status: withheld
    kind: sealed-artifact
    exp_ids: []
    patterns: ['(?<![A-Za-z0-9])sealed/']
    paths: ['**/sealed/**']
  - id: preds
    status: withheld
    kind: sealed-artifact
    exp_ids: []
    patterns: ['(?<![A-Za-z0-9])preds(?![A-Za-z0-9])']
    paths: ['**/preds', '**/preds/**', '**/preds.*']
  - id: z-snapshots
    status: withheld
    kind: sealed-artifact
    exp_ids: []
    patterns: ['z_snapshots']
    paths: ['**/z_snapshots/**']
"""
RELEASED_ONLY = """version: 1
families:
  - id: released-fixture
    status: released
    exp_ids: [EXP-016]
    patterns: []
"""
SHARED = {"map": ["shared/siz_rules_header.md", "shared/siz_map_system.md"],
          "reduce": ["shared/siz_rules_header.md", "shared/siz_reduce_system.md"]}


@pytest.fixture
def env(tmp_path):
    lab = tmp_path / "lab"
    w, d = tmp_path / "W.yaml", tmp_path / "D.yaml"
    w.write_text(RELEASED_ONLY, encoding="utf-8")
    d.write_text(ACTIVE_DENY_MIRROR, encoding="utf-8")
    s1 = lab / P.SYNTH_REL
    s1.mkdir(parents=True)
    shutil.copyfile(FIX / "S1_table.fixture.csv", s1 / "S1_table.csv")
    shutil.copyfile(FIX / "S1_cells.fixture.csv", s1 / "S1_cells.csv")
    (lab / P.SYNTH_REL / "S23_templates").mkdir()
    shutil.copyfile(TPL / "HASHES.sha256", lab / P.SYNTH_REL / "S23_templates/HASHES.sha256")
    root = lab / "38_api-batch"
    bds = {"sonnet": root / "11_batches/2099-01-01_siz-map-sonnet", "opus": root / "11_batches/2099-01-01_siz-map-opus",
           "reduce": root / "11_batches/2099-01-02_siz-reduce-fable"}
    for k, bd in bds.items():
        spec = f"batch/map_{k}.spec.yaml" if k != "reduce" else "batch/reduce_fable.spec.yaml"
        P.stage([f"{TPL_REL}/{r}" for r in [spec, *SHARED["map" if k != "reduce" else "reduce"]]], bd, TPL / "HASHES.sha256")
    rs = yaml.safe_load((bds["reduce"] / "spec.yaml").read_text(encoding="utf-8"))
    rs["map_batches"] = {"claude-sonnet-5-5": f"38_api-batch/11_batches/{bds['sonnet'].name}",
                         "claude-opus-5-5": f"38_api-batch/11_batches/{bds['opus'].name}"}
    (bds["reduce"] / "spec.yaml").write_text(yaml.safe_dump(rs, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def builder(bd):
        return PackBuilder.from_files([w, d], lab_root=lab, batch_dir=bd)

    def plan(k):
        return ops.plan(bds[k], builder(bds[k]), led)
    led = Ledger(root / "ledger.jsonl")
    return type("E", (), dict(lab=lab, s1=s1, root=root, bds=bds, plan=staticmethod(plan), led=led))


def rows_of(bd):
    return M.load_manifest(bd / "manifest.jsonl")[0]


def fake_collect(env, *, skip=None, status="succeeded", l5="pass"):
    inbox = []
    for k in ("sonnet", "opus"):
        bd = env.bds[k]
        (bd / "items").mkdir(exist_ok=True)
        for r in rows_of(bd):
            cid = r["custom_id"]
            if cid == skip:
                continue
            (bd / "items" / f"{cid}.md").write_text(
                f"---\nstatus: DRAFT (api-batch output — not evidence)\nuse: SIZ-map\ncustom_id: {cid}\n"
                f"model_requested: {r['params']['model']}\nl5: {l5}\n---\n"
                + json.dumps({"cell_id": r["meta"]["cell_id"], "scope_mismatches": [], "candidate_sentences": [],
                              "rule_findings": []}) + "\n", encoding="utf-8")
            inbox.append({"custom_id": cid, "batch_id": f"msgbatch_{k}", "use": "SIZ-map", "result_status": status,
                          "item": f"items/{cid}.md"})
    (env.root / "INBOX.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inbox), encoding="utf-8")


def map_packets():
    return P.build_map_packets((FIX / "S1_table.fixture.csv").read_text(encoding="utf-8"),
                               (FIX / "S1_cells.fixture.csv").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ spec
def test_template_specs_load():
    for rel in ("batch/map_sonnet.spec.yaml", "batch/map_opus.spec.yaml", "batch/reduce_fable.spec.yaml"):
        assert load_spec(TPL / rel)["use"] in ("SIZ-map", "SIZ-reduce")


def test_spec_one_model_per_spec_and_fixed_fields():
    s = yaml.safe_load((TPL / "batch/map_sonnet.spec.yaml").read_text(encoding="utf-8"))
    r = yaml.safe_load((TPL / "batch/reduce_fable.spec.yaml").read_text(encoding="utf-8"))
    for bad in ({"model": ["claude-sonnet-5-5", "claude-opus-5-5"]}, {"model": "claude-haiku-5-5"},
                {"model": "claude-fable-5-1"}, {"data": {"s1_table": "x"}}, {"packets": {**s["packets"], "function": "os.system"}},
                {"packets": {**s["packets"], "classes": ["A-VERDICT"]}}, {"prompt": {"system_template": "tools/x.md"}}):
        with pytest.raises(SpecError):
            validate_spec({**s, **bad})
    for key in ("templates_hashes", "packets", "output_schema"):
        with pytest.raises(SpecError):
            validate_spec({k: v for k, v in s.items() if k != key})
    for bad in ({"digest": True}, {"group_by": ["q", "s"]}, {"exclude_q_tokens": ["Q-NONE"]}, {"model": "claude-opus-5-5"},
                {"map_batches": {"claude-sonnet-5-5": "a"}}, {"map_batches": {m: "same" for m in P.MAP_MODELS}},
                {"data": {"s1_cells": "c", "s1_table": "t"}}):
        with pytest.raises(SpecError):
            validate_spec({**r, **bad})


# ------------------------------------------------------------------------------------------------ map
def test_map_plan_twins_share_packets(env):
    ds, do = env.plan("sonnet"), env.plan("opus")
    want = map_packets()
    assert ds["use"] == do["use"] == "SIZ-map" and ds["n_requests"] == do["n_requests"] == len(want)
    assert ds["siz"]["cells"] == [p["cell_id"] for p in want] and ds["siz"]["n_locked"] == sum(p["mode"] == "LOCKED" for p in want)
    assert ds["est_usd_hard_worst"] <= ds["cap_usd"] == 5.0 and do["est_usd_hard_worst"] <= do["cap_usd"] == 10.0
    rs, ro = rows_of(env.bds["sonnet"]), rows_of(env.bds["opus"])
    assert {r["custom_id"] for r in rs}.isdisjoint({r["custom_id"] for r in ro})        # shared INBOX: ids must not collide
    for a, b in zip(rs, ro):
        assert a["custom_id"].startswith("sizm-") and a["meta"]["cell_id"] == b["meta"]["cell_id"]
        assert (a["params"]["model"], b["params"]["model"]) == P.MAP_MODELS
        assert {k: v for k, v in a["params"].items() if k != "model"} == {k: v for k, v in b["params"].items() if k != "model"}
        assert a["pack_sha256s"] == b["pack_sha256s"]
        assert {k: v for k, v in a["meta"].items() if k != "model"} == {k: v for k, v in b["meta"].items() if k != "model"}
    r = rs[0]
    blocks = r["params"]["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["document", "text"] and blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[0]["source"]["data"] == (TPL / "shared/siz_rules_header.md").read_text(encoding="utf-8")
    assert blocks[1]["text"].startswith("# CELL PACKET ") and blocks[1]["text"] == want[0]["text"]
    assert r["params"]["system"][0]["text"] == (TPL / "shared/siz_map_system.md").read_text(encoding="utf-8")
    assert r["params"]["output_config"] == {"effort": "medium", "format": {
        "type": "json_schema", "schema": json.loads((TPL / "schemas/siz_review_note.json").read_text(encoding="utf-8"))}}
    assert r["params"]["max_tokens"] == 8000 and r["meta"]["header_witness_sha256"] == ds["siz"]["template_sha256"]["header"]
    scan = {p["name"]: p for p in ds["scan"]}
    assert scan["S1_table.csv"]["mode"] == scan["S1_cells.csv"]["mode"] == "verify-only" and ds["masked_blocks"] == 0
    assert [x["n_requests"] for x in env.led.rows("planned")] == [len(want), len(want)]


def test_map_refuses_missing_s1_file(env):
    (env.s1 / "S1_cells.csv").unlink()
    with pytest.raises(Halt, match="not found"):
        env.plan("sonnet")
    assert not (env.bds["sonnet"] / "manifest.jsonl").exists() and not env.led.rows("planned")


def test_map_refuses_wrong_q(env):
    f = env.s1 / "S1_cells.csv"
    f.write_text(f.read_text(encoding="utf-8").replace("Q-Z7,CL-0006", "Q-X9,CL-0006"), encoding="utf-8")
    with pytest.raises(Halt, match="packets.py refused"):
        env.plan("sonnet")
    assert not env.led.rows("planned")


def test_map_refuses_deny_hit_in_s1(env):
    f = env.s1 / "S1_table.csv"
    f.write_text(f.read_text(encoding="utf-8") + "# see runs/z_snapshots for the raw states\n", encoding="utf-8")
    with pytest.raises(Halt, match="L3b"):
        env.plan("sonnet")
    assert not (env.bds["sonnet"] / "manifest.jsonl").exists() and not env.led.rows("planned")


def test_map_refuses_header_edited_after_witness(env):
    h = env.bds["sonnet"] / "shared/siz_rules_header.md"
    h.write_text(h.read_text(encoding="utf-8") + "\nextra rule\n", encoding="utf-8")
    with pytest.raises(Halt, match="differs"):
        env.plan("sonnet")


@pytest.mark.parametrize("mode,msg", [("sha", "packets generator .* sha256 differs"), ("drop", "packets generator .* not listed")])
def test_plan_refuses_packets_py_not_witnessed(env, mode, msg):
    """W9: plan binds tools/state_iter_z_synthesis/packets.py to S23_templates/HASHES.sha256 (changed or unlisted ->
    refused before any packet is built, nothing written)."""
    gen = "tools/state_iter_z_synthesis/packets.py"
    hp = env.lab / P.SYNTH_REL / "S23_templates/HASHES.sha256"
    lines = hp.read_text(encoding="utf-8").splitlines(keepends=True)
    assert sum(ln.rstrip("\n").endswith(f"  {gen}") for ln in lines) == 1
    out = []
    for ln in lines:
        if ln.rstrip("\n").endswith(f"  {gen}"):
            if mode == "drop":
                continue
            ln = f"{'0' * 64}  {gen}\n"
        out.append(ln)
    hp.write_text("".join(out), encoding="utf-8")
    with pytest.raises(Halt, match=msg):
        env.plan("sonnet")
    assert not (env.bds["sonnet"] / "manifest.jsonl").exists() and not env.led.rows("planned")


def test_plan_records_the_generator_witness(env):
    d = env.plan("sonnet")
    gen = "tools/state_iter_z_synthesis/packets.py"
    assert d["siz"]["template_sha256"]["generator"] == P.read_hashes(TPL / "HASHES.sha256")[gen] == P.sha256_file(REPO_ROOT / gen)


# ------------------------------------------------------------------------------------------------ reduce
def test_reduce_one_request_per_question(env):
    env.plan("sonnet"), env.plan("opus")
    fake_collect(env)
    d = env.plan("reduce")
    rps = P.build_reduce_packets((FIX / "S1_cells.fixture.csv").read_text(encoding="utf-8"))
    built = [r["q"] for r in rps if r["status"] == "built"]
    assert d["use"] == "SIZ-reduce" and d["n_requests"] == len(built) and d["siz"]["questions"] == built
    assert {s["question_token"] for s in d["siz"]["skipped"]} <= set(P.REDUCE_EXCLUDED) and d["siz"]["skipped"]
    assert d["est_usd_hard_worst"] <= d["cap_usd"] == 10.0
    rows = rows_of(env.bds["reduce"])
    for r, rp in zip(rows, [x for x in rps if x["status"] == "built"]):
        assert r["custom_id"].startswith("sizr-") and r["meta"]["q"] == rp["q"] and r["params"]["model"] == P.REDUCE_MODEL
        assert r["params"]["max_tokens"] == 16000 and r["params"]["output_config"]["effort"] == "high"
        c = r["params"]["messages"][0]["content"]
        assert c[0]["title"] == "siz_rules_header" and c[1]["source"]["data"] == rp["text"] and c[-1]["type"] == "text"
        assert r["meta"]["n_memos"] == 2 * len(rp["targets"]) == len(c) - 3
        assert not any("cache_control" in b for b in c)
        assert "digest" not in json.dumps(r["meta"])
        body = json.dumps(r["params"], ensure_ascii=False)
        assert "S1_table ROWS" not in body and all(f"\"q\": \"{q}\"" not in body for q in built if q != rp["q"])


def test_reduce_scopes_refuse_bad_or_excluded_q(env):
    env.plan("sonnet"), env.plan("opus")
    fake_collect(env)
    sp = env.bds["reduce"] / "spec.yaml"
    base = yaml.safe_load(sp.read_text(encoding="utf-8"))
    skipped_q = P.build_reduce_packets((FIX / "S1_cells.fixture.csv").read_text(encoding="utf-8"))
    skipped_q = next(r["q"] for r in skipped_q if r["status"] != "built")
    for bad in (["Q-X9"], [skipped_q]):
        sp.write_text(yaml.safe_dump({**base, "scopes": bad}), encoding="utf-8")
        with pytest.raises(Halt, match="scopes"):
            env.plan("reduce")
    sp.write_text(yaml.safe_dump({**base, "scopes": ["Q-I2"]}), encoding="utf-8")
    assert env.plan("reduce")["siz"]["questions"] == ["Q-I2"]


def test_reduce_refuses_incomplete_maps(env):
    env.plan("sonnet"), env.plan("opus")
    fake_collect(env, skip=rows_of(env.bds["opus"])[0]["custom_id"])
    with pytest.raises(Halt, match="not fully collected"):
        env.plan("reduce")
    fake_collect(env, status="refusal")
    with pytest.raises(Halt, match="not fully collected"):
        env.plan("reduce")
    fake_collect(env, l5="quarantined")
    with pytest.raises(Halt, match="l5"):
        env.plan("reduce")
    assert not (env.bds["reduce"] / "manifest.jsonl").exists()


def test_reduce_refuses_unplanned_map(env):
    env.plan("sonnet"), env.plan("opus")
    fake_collect(env)
    (env.bds["opus"] / "plan.json").unlink()
    with pytest.raises(Halt, match="not planned"):
        env.plan("reduce")


def test_reduce_refuses_non_twin_or_swapped_maps(env):
    sp = env.bds["opus"] / "spec.yaml"
    s = yaml.safe_load(sp.read_text(encoding="utf-8"))
    s["packets"]["r3_ii_cite"] = "abl_tfb compute-matched definition"              # different packets than the twin
    sp.write_text(yaml.safe_dump(s), encoding="utf-8")
    env.plan("sonnet"), env.plan("opus")
    fake_collect(env)
    with pytest.raises(Halt, match="differ beyond the model"):
        env.plan("reduce")
    rp = env.bds["reduce"] / "spec.yaml"
    r = yaml.safe_load(rp.read_text(encoding="utf-8"))
    r["map_batches"] = dict(zip(P.MAP_MODELS, reversed(list(r["map_batches"].values()))))
    rp.write_text(yaml.safe_dump(r), encoding="utf-8")
    with pytest.raises(Halt, match="is not a claude-sonnet-5-5 batch"):
        env.plan("reduce")


def test_reduce_with_cache_ttl_puts_the_breakpoint_on_the_header(env):
    env.plan("sonnet"), env.plan("opus")
    fake_collect(env)
    sp = env.bds["reduce"] / "spec.yaml"
    rs = yaml.safe_load(sp.read_text(encoding="utf-8"))
    rs["cache_ttl"] = "5m"
    sp.write_text(yaml.safe_dump(rs, allow_unicode=True), encoding="utf-8")
    env.plan("reduce")
    for r in rows_of(env.bds["reduce"]):
        c = r["params"]["messages"][0]["content"]
        assert c[0]["title"] == "siz_rules_header" and c[0]["cache_control"] == {"type": "ephemeral"}
        assert not any("cache_control" in b for b in c[1:]) and "cache_control" not in json.dumps(r["params"]["system"])
        assert r["meta"]["cache_after"] == ["siz_rules_header"]
