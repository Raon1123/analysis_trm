"""jev lane (2026-11-05, plan api-joyful-island D): TypeSafe Jev typed judgments through the batch-lane rails.
FakeJevClient only; no network, no key, no typesafe-sdk import."""
import json
import sys

import pytest

from tools.api_batch import cli, jev, ops
from tools.api_batch.common import Halt
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder

HEAD = """name: jev-test
model: jev-1.14.0
items:
  - {id: a1, state: [notes.md]}
  - {id: a2, state: [notes.md, table.csv]}
"""
QUESTIONS = """is_numeric:
  type: noul
  instructions: Does the document report a numeric accuracy?
kind:
  type: choice
  instructions: What kind of document is this?
  criteria:
    notes: Free-form notes.
    table: A data table.
    other: null
clarity:
  type: score
  instructions: How clearly is the claim stated?
  criteria: [Unclear., Partly clear., Fully clear.]
"""


def spec_text(head=HEAD, questions=QUESTIONS):
    return head + "questions:\n" + "".join("  " + ln + "\n" for ln in questions.splitlines())


SPEC = spec_text()


@pytest.fixture
def env(tmp_path, deny_files, lab):
    root = tmp_path / "root"
    pd = root / "11_batches" / "2026-11-05_jev-test"
    pd.mkdir(parents=True)
    (pd / "spec.yaml").write_text(SPEC, encoding="utf-8")
    (lab / "notes.md").write_text("# Notes\nThe model reaches acc 0.9 at k=3 — 한글 포함.\n", encoding="utf-8")
    (lab / "table.csv").write_text("k,acc\n3,0.9\n5,0.7\n", encoding="utf-8")
    builder = PackBuilder.from_files(deny_files, lab_root=lab)
    led = Ledger(root / "ledger.jsonl")
    return type("E", (), dict(root=root, pd=pd, lab=lab, builder=builder, led=led, deny_files=deny_files))


def _plan(env):
    return jev.plan(env.pd / "spec.yaml", env.builder, env.led)


def _approve(env, usd=None):
    sp = jev.sync_plan_for_approve(env.pd)
    return ops.approve(None, sp["hard_worst_usd"] if usd is None else usd, env.led, env={}, sync_plan=sp)


def _ready(env):
    ops.selftest(env.led, runner=lambda: 0)
    d = _plan(env)
    _approve(env)
    return d


def _rows(env):
    return [json.loads(x) for x in (env.pd / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ plan
def test_plan_fanout_one_request_per_item(env):
    d = _plan(env)
    rows = _rows(env)
    assert d["use"] == "jev" and d["n_requests"] == 2 and d["n_questions"] == 3
    assert [r["custom_id"] for r in rows] == ["jev-a1-r1", "jev-a2-r1"]
    p = rows[1]["params"]
    assert set(p) == {"model", "state", "questions"} and p["model"] == "jev-1.14.0"
    assert list(p["state"]) == ["doc1", "doc2"]                      # neutral names, no file names
    assert "notes.md" not in json.dumps(p) and "table.csv" not in json.dumps(p)
    assert p["questions"]["kind"]["criteria"] == {"notes": "Free-form notes.", "table": "A data table.", "other": None}
    assert p["questions"]["clarity"]["criteria"] == ["Unclear.", "Partly clear.", "Fully clear."]
    assert sorted(rows[1]["meta"]["source_paths"]) == ["notes.md", "table.csv"]
    planned = env.led.rows("planned")
    assert planned[-1]["lane"] == "jev" and planned[-1]["plan_hash"] == d["plan_hash"]


def test_plan_hard_worst_matches_formula(env):
    d = _plan(env)
    rows = _rows(env)
    from tools.api_batch.common import est_tokens
    tot = 0.0
    for r in rows:
        p = r["params"]
        s = sum(est_tokens(v) for v in p["state"].values())
        q = sum(est_tokens(t) for t in jev._q_leaves(p["questions"]))
        tot += (s * len(p["questions"]) + q) * jev.JEV_PRICE_IN_PER_MTOK * jev.JEV_PRICE_SAFETY / 1e6
    assert d["est_usd_hard_worst"] == pytest.approx(max(tot, 1e-6), abs=1e-6)


def test_repeats_multiply_requests(env):
    (env.pd / "spec.yaml").write_text(SPEC + "repeats: 3\n", encoding="utf-8")
    d = _plan(env)
    assert d["n_requests"] == 6
    ids = [r["custom_id"] for r in _rows(env)]
    assert ids[:3] == ["jev-a1-r1", "jev-a1-r2", "jev-a1-r3"]
    assert len({r["params_sha256"] for r in _rows(env)[:3]}) == 1     # identical input -> stability measure


@pytest.mark.parametrize("bad", [
    SPEC.replace("jev-1.14.0", "jev-latest"),                         # alias refused
    SPEC.replace("jev-1.14.0", "claude-sonnet-5-5"),
    SPEC + "repeats: 9\n",
    SPEC + "extra: 1\n",
    SPEC.replace("{id: a2,", "{id: a1,"),                              # duplicate id
    SPEC.replace("{id: a2,", "{id: A_2,"),
])
def test_bad_spec_refused(env, bad):
    (env.pd / "spec.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)
    assert not (env.pd / "manifest.jsonl").exists()


@pytest.mark.parametrize("bad", [
    "q1: {type: box, instructions: x}",                                 # Box not supported
    "Q1: {type: noul, instructions: x}",                                # qid shape
    "q1: {type: noul, instructions: ''}",
    "q1: {type: choice, instructions: x, criteria: {'Bad Label': y}}",  # labels are code-controlled tokens
    "q1: {type: choice, instructions: x}",
    "q1: {type: score, instructions: x, criteria: [a,b,c,d,e,f,g,h,i,j,k]}",
    "q1: {type: noul, instructions: x, criteria: {maybe: y}}",
    "q1: {type: noul, instructions: x, extra: 1}",
])
def test_bad_questions_refused(env, bad):
    (env.pd / "spec.yaml").write_text(spec_text(questions=bad), encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)


def test_denied_text_in_question_refused(env):
    # a question naming a withheld exp_id (EXP-013) or token (clean_r1) must not leave the machine
    (env.pd / "spec.yaml").write_text(spec_text(questions=QUESTIONS.replace("What kind of document", "What EXP-013 outcome")), encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)
    (env.pd / "spec.yaml").write_text(spec_text(questions=QUESTIONS.replace("A data table.", "A clean_r1 table.")), encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)


def test_denied_state_refused(env):
    (env.lab / "notes.md").write_text("fig1_mlp_z_iter_k3_s4 reaches 0.9\n", encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)


def test_state_extension_refused(env):
    (env.lab / "x.png").write_bytes(b"\x89PNG")
    (env.pd / "spec.yaml").write_text(SPEC.replace("[notes.md]", "[x.png]"), encoding="utf-8")
    with pytest.raises(Halt):
        _plan(env)


def test_budget_cap_refuses(env):
    (env.pd / "spec.yaml").write_text(SPEC + "budget_usd: 0.0000001\n", encoding="utf-8")
    with pytest.raises(Halt, match="hard-worst"):
        _plan(env)


# ------------------------------------------------------------------ validator (single-path invariant)
def _params_packs(env):
    _plan(env)
    p = _rows(env)[0]["params"]
    packs = [env.builder.build_spec_text(f"t{i}", t) for i, t in enumerate(jev._text_leaves(p))]
    return p, packs


def test_validator_accepts_planned(env):
    p, packs = _params_packs(env)
    jev.validate_params(p, packs)


def test_validator_rejects_non_pack_text(env):
    p, packs = _params_packs(env)
    p = json.loads(json.dumps(p))
    p["questions"]["is_numeric"]["instructions"] = "Injected text not from any Pack."
    with pytest.raises(jev.JevError, match="not the bytes"):
        jev.validate_params(p, packs)


def test_validator_rejects_unused_pack(env):
    p, packs = _params_packs(env)
    with pytest.raises(jev.JevError, match="do not appear"):
        jev.validate_params(p, packs + [env.builder.build_spec_text("extra", "An unused text.")])


@pytest.mark.parametrize("mut", [
    lambda p: p.update(images=[1]),
    lambda p: p.update(model="jev-latest"),
    lambda p: p.update(state={}),
    lambda p: p["questions"]["kind"]["criteria"].update({"Bad Label": None}),
    lambda p: p["questions"]["is_numeric"].update(type="box"),
])
def test_validator_rejects_shape(env, mut):
    p, packs = _params_packs(env)
    p = json.loads(json.dumps(p))
    mut(p)
    with pytest.raises(jev.JevError):
        jev.validate_params(p, packs)


def test_validator_rejects_unsealed(env):
    p, packs = _params_packs(env)
    with pytest.raises(jev.JevError):
        jev.validate_params(p, [])


def test_validator_size_caps(env, monkeypatch):
    p, packs = _params_packs(env)
    monkeypatch.setattr(jev, "MAX_QUESTION_TOKENS", 5)
    with pytest.raises(jev.JevError, match="too large"):
        jev.validate_params(p, packs)


# ------------------------------------------------------------------ normalize
def test_normalize_fake_roundtrip(env):
    _plan(env)
    p = _rows(env)[1]["params"]
    got = jev.FakeJevClient().send(p)
    n = jev.normalize(got["raw"], p)
    assert n["is_numeric"]["value"] == pytest.approx(0.8)
    assert n["kind"]["value"] == "notes"
    assert n["clarity"]["value"] == 2.0


@pytest.mark.parametrize("mut,msg", [
    (lambda a: a.pop("kind"), "answers"),
    (lambda a: a.update(extra={"type": "noul", "noul": 0.1}), "answers"),
    (lambda a: a["is_numeric"].update(noul=1.5), "noul"),
    (lambda a: a["is_numeric"].update(type="choice"), "type"),
    (lambda a: a["kind"].update(choice="invented"), "choice"),
    (lambda a: a["kind"]["probabilities"].update(invented=0.1), "probabilities"),
    (lambda a: a["clarity"].update(score=7), "score"),
    (lambda a: a["clarity"]["probabilities"].update({"9": 0.1}), "probabilities"),
])
def test_normalize_rejects(env, mut, msg):
    _plan(env)
    p = _rows(env)[0]["params"]
    raw = jev.FakeJevClient().send(p)["raw"]
    mut(raw["answers"])
    with pytest.raises(jev.JevError, match=msg):
        jev.normalize(raw, p)


# ------------------------------------------------------------------ run gating
def test_run_happy_path(env):
    d = _ready(env)
    c = jev.FakeJevClient()
    r = jev.run(env.pd, env.led, c, env.builder)
    assert r["halted"] is None and [x["rc"] for x in r["requests"]] == [0, 0]
    assert len(c.calls) == 2 and "questions" in c.calls[0]
    out = [json.loads(x) for x in (env.pd / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [o["item"] for o in out] == ["a1", "a2"] and out[0]["answers"]["kind"]["value"] == "notes"
    assert (env.pd / "raw" / "jev-a1-r1.json").exists()
    sent, done = env.led.rows("sync_sent", lane="jev"), env.led.rows("sync_done", lane="jev")
    assert len(sent) == 2 and len(done) == 2 and all(x["plan_hash"] == d["plan_hash"] for x in sent)
    assert done[0]["cost_est_usd"] > 0 and done[0]["usage"]["input_tokens"] > 0
    # all done -> nothing to send, and approving again is refused
    assert jev.run(env.pd, env.led, c, env.builder)["nothing_to_do"] is True
    with pytest.raises(Halt, match="nothing to do"):
        jev.sync_plan_for_approve(env.pd, env.led)


def test_run_needs_approval(env):
    ops.selftest(env.led, runner=lambda: 0)
    _plan(env)
    with pytest.raises(Halt, match="approval"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)


def test_run_needs_tests_passed(env):
    _plan(env)
    _approve(env)
    with pytest.raises(Halt, match="lapses"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)


def test_one_approval_one_run(env):
    _ready(env)
    jev.run(env.pd, env.led, jev.FakeJevClient(fail=True), env.builder)          # consumes the approval
    with pytest.raises(Halt, match="consumed"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)


def test_run_send_failure_halts(env):
    _ready(env)
    r = jev.run(env.pd, env.led, jev.FakeJevClient(fail=True), env.builder)
    assert r["halted"].startswith("send failed") and [x["rc"] for x in r["requests"]] == [1]
    assert not (env.pd / "out" / "results.jsonl").exists()


def test_run_model_mismatch_halts(env):
    _ready(env)
    r = jev.run(env.pd, env.led, jev.FakeJevClient(model_override="jev-1.15.0"), env.builder)
    assert [x["rc"] for x in r["requests"]] == [2] and "model mismatch" in r["halted"]
    assert not (env.pd / "out" / "results.jsonl").exists()


def test_run_bad_response_rc4(env):
    _ready(env)
    r = jev.run(env.pd, env.led, jev.FakeJevClient(bad_answer=True), env.builder)
    assert [x["rc"] for x in r["requests"]] == [4, 4] and r["halted"] is None
    assert not (env.pd / "out" / "results.jsonl").exists()


def test_run_resumes_only_pending(env):
    _ready(env)
    jev.run(env.pd, env.led, jev.FakeJevClient(bad_answer=True), env.builder)   # both rc 4 -> still pending
    _approve(env)
    c = jev.FakeJevClient()
    r = jev.run(env.pd, env.led, c, env.builder)
    assert [x["rc"] for x in r["requests"]] == [0, 0] and len(c.calls) == 2


def test_tampered_manifest_refused(env):
    _ready(env)
    m = env.pd / "manifest.jsonl"
    m.write_text(m.read_text(encoding="utf-8").replace("Free-form notes.", "Free-form notez."), encoding="utf-8")
    with pytest.raises(Halt, match="manifest"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)


def test_tampered_plan_worst_refused(env):
    _ready(env)
    pj = env.pd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    d["est_usd_hard_worst"] = 0.000001
    pj.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(Halt, match="tampered"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)


def test_deny_grown_after_plan_refuses_run(env, tmp_path):
    _ready(env)
    extra = tmp_path / "extra_deny.yaml"
    extra.write_text("version: 1\nfamilies:\n  - id: late\n    status: withheld\n    exp_ids: []\n"
                     "    patterns: ['Free-form notes']\n", encoding="utf-8")
    b2 = PackBuilder.from_files([*env.deny_files, extra], lab_root=env.lab)
    c = jev.FakeJevClient()
    with pytest.raises(Halt):
        jev.run(env.pd, env.led, c, b2)
    assert c.calls == [] and env.led.rows("sync_sent", lane="jev") == []


def test_budget_halt_mid_run(env, monkeypatch):
    _ready(env)
    # usage reported far above the plan estimate: the next request could exceed the approval
    orig = jev.FakeJevClient.send

    def send(self, params):
        got = orig(self, params)
        got["raw"]["usage"]["input_tokens"] = 10**9
        return got
    monkeypatch.setattr(jev.FakeJevClient, "send", send)
    r = jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)
    assert len(r["requests"]) == 1 and "exceed" in r["halted"]


# ------------------------------------------------------------------ CLI / approve / isolation
def _cli(env, *args, client=None):
    return cli.main(["--batch-root", str(env.root), "--lab-root", str(env.lab),
                     *sum([["--deny-file", str(f)] for f in env.deny_files], []), *args], client=client)


def test_cli_plan_and_run(env, monkeypatch):
    assert _cli(env, "jev", "plan", str(env.pd / "spec.yaml")) == 0
    led = Ledger(env.root / "ledger.jsonl")
    ops.selftest(led, runner=lambda: 0)
    for k in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(k, raising=False)
    d = json.loads((env.pd / "plan.json").read_text(encoding="utf-8"))
    assert _cli(env, "approve", str(env.pd), "--usd", str(d["est_usd_hard_worst"] + 1e-6)) == 0
    appr = led.rows("approved")[-1]
    assert appr["lane"] == "sync" and appr["manifest_sha256"] == d["plan_hash"]
    assert _cli(env, "jev", "run", str(env.pd), client=jev.FakeJevClient()) == 0
    assert _cli(env, "jev", "run", str(env.pd)) == 0          # nothing to do: no RealJevClient constructed


def test_cli_approve_refused_in_session(env, monkeypatch):
    _plan(env)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert _cli(env, "approve", str(env.pd), "--usd", "1") == 3


def test_no_sdk_import_offline(env):
    _ready(env)
    jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)
    assert "typesafe_sdk" not in sys.modules


def test_capabilities_table():
    assert jev.CAPABILITIES["jev"]["batch"] is False and jev.CAPABILITIES["jev"]["prefix_cache"] is False
    assert jev.CAPABILITIES["jev"]["output_kind"] == "typed_judgment"


# ------------------------------------------------------------------ r2 (VERIFY-JEV-D-r1 F1-F9)
def _grown(env, tmp_path, pattern=None, path=None):
    extra = tmp_path / "late_deny.yaml"
    fam = "version: 1\nfamilies:\n  - id: late\n    status: withheld\n    exp_ids: []\n"
    fam += f"    patterns: ['{pattern or 'zzz_nomatch_zzz'}']\n"     # load_deny refuses a family with no patterns
    if path:
        fam += f"    paths: ['{path}']\n"
    extra.write_text(fam, encoding="utf-8")
    return PackBuilder.from_files([*env.deny_files, extra], lab_root=env.lab)


@pytest.mark.parametrize("pattern", ["(?<![a-z])other(?![a-z])", "is_numeric", "doc2"])
def test_run_rescans_identifiers(env, tmp_path, pattern):
    # F1: a deny pattern added after plan that matches only a choice label / qid / state key blocks the send
    _ready(env)
    c = jev.FakeJevClient()
    with pytest.raises(Halt):
        jev.run(env.pd, env.led, c, _grown(env, tmp_path, pattern=pattern))
    assert c.calls == [] and env.led.rows("sync_sent", lane="jev") == []


def test_run_reapplies_deny_path_glob_added_after_plan(env, tmp_path):
    # F3: only the source path matches (no text does)
    _ready(env)
    c = jev.FakeJevClient()
    b2 = _grown(env, tmp_path, path="**/table.csv")                 # builds: the deny source itself is valid
    with pytest.raises(Halt, match="now matches the deny source"):
        jev.run(env.pd, env.led, c, b2)
    assert c.calls == [] and env.led.rows("sync_sent", lane="jev") == []


def test_run_refuses_row_without_source_paths(env):
    _ready(env)
    m = env.pd / "manifest.jsonl"
    rows = _rows(env)
    rows[0]["meta"]["source_paths"] = []
    from tools.api_batch.common import canon_json, sha256_text
    data = "\n".join(canon_json(r) for r in rows) + "\n"
    m.write_text(data, encoding="utf-8")
    pj = env.pd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    d["manifest_sha256"] = sha256_text(data)
    d["plan_hash"] = jev.compute_plan_hash(d["manifest_sha256"], d["est_usd_hard_worst"])
    pj.write_text(json.dumps(d), encoding="utf-8")
    _approve(env)
    with pytest.raises(Halt, match="source paths"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)


def test_tampered_plan_hash_refused(env):
    # F2: only plan_hash edited -> approval key no longer bound to the plan
    _ready(env)
    pj = env.pd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    d["plan_hash"] = "0" * 64
    pj.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(Halt, match="plan_hash"):
        jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)
    with pytest.raises(Halt, match="plan_hash"):
        jev.sync_plan_for_approve(env.pd)


def test_limits_checked_at_plan_and_run(env, monkeypatch):
    # F4
    monkeypatch.setattr(env.led, "committed_usd", lambda prefix=None: 10_000.0)
    with pytest.raises(Halt, match="committed"):
        _plan(env)
    monkeypatch.undo()
    _ready(env)
    monkeypatch.setattr(env.led, "committed_usd", lambda prefix=None: 10_000.0)
    c = jev.FakeJevClient()
    with pytest.raises(Halt, match="committed"):
        jev.run(env.pd, env.led, c, env.builder)
    assert c.calls == []


def test_committed_usd_counts_jev_approval(env):
    d = _ready(env)
    assert env.led.committed_usd() == pytest.approx(d["est_usd_hard_worst"], abs=1e-6)


def test_plan_calls_validator(env, monkeypatch):
    # F5: a params builder that slips in non-Pack text must be refused at plan
    orig = jev.build_params

    def bad(*a, **k):
        p = orig(*a, **k)
        p["questions"]["is_numeric"]["instructions"] += " (unscanned)"
        return p
    monkeypatch.setattr(jev, "build_params", bad)
    with pytest.raises(jev.JevError, match="not the bytes"):
        _plan(env)
    assert not (env.pd / "manifest.jsonl").exists()


def test_validator_rejects_forged_unsealed_pack(env):
    p, packs = _params_packs(env)
    forged = jev.Pack(packs[0].kind, packs[0].name, packs[0].text, packs[0].mode)
    with pytest.raises(jev.JevError, match="unsealed"):
        jev.validate_params(p, [forged, *packs[1:]])


def test_validator_rejects_non_neutral_state_key(env):
    p, packs = _params_packs(env)
    p = json.loads(json.dumps(p))
    p["state"] = {"notes.md": p["state"]["doc1"]}
    with pytest.raises(jev.JevError, match="neutral"):
        jev.validate_params(p, packs)


@pytest.mark.parametrize("mut", [
    lambda raw: raw["answers"]["is_numeric"].update(noul=True),                  # bool is not a probability
    lambda raw: raw["answers"]["kind"].update(choice=["notes"]),                  # unhashable
    lambda raw: raw["answers"]["kind"].update(probabilities=["notes"]),           # not a mapping
    lambda raw: raw["answers"]["kind"]["probabilities"].update(notes=1.7),
    lambda raw: raw["answers"]["clarity"].update(confidence="high"),
])
def test_normalize_rejects_malformed_values(env, mut):
    _plan(env)
    p = _rows(env)[0]["params"]
    raw = jev.FakeJevClient().send(p)["raw"]
    mut(raw)
    with pytest.raises(jev.JevError):
        jev.normalize(raw, p)


@pytest.mark.parametrize("reply", [
    {"raw": ["not", "an", "object"]},                                            # raw not an object
    ["not", "a", "dict"],                                                        # send() result not a dict
    "usage-list", "usage-huge", "usage-negative", "usage-bool",
])
def test_malformed_response_closes_sync_sent(env, monkeypatch, reply):
    # r2 G2/G3: every malformed reply gets sync_done rc 4 (never an unclosed sync_sent, never rc 2)
    _ready(env)
    orig = jev.FakeJevClient.send

    def send(self, params):
        if not isinstance(reply, str):
            return reply
        got = orig(self, params)
        got["raw"]["usage"] = {"usage-list": ["x"], "usage-huge": {"input_tokens": 1e999},
                               "usage-negative": {"input_tokens": -5}, "usage-bool": {"input_tokens": True}}[reply]
        return got
    monkeypatch.setattr(jev.FakeJevClient, "send", send)
    r = jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)
    assert [x["rc"] for x in r["requests"]] == [4, 4] and r["halted"] is None
    assert len(env.led.rows("sync_done", lane="jev")) == len(env.led.rows("sync_sent", lane="jev")) == 2
    assert r["spent_usd_est"] > 0                                                # unknown usage charged at hard-worst
    assert not (env.pd / "out" / "results.jsonl").exists()


def test_run_revalidates_params(env):
    # r2 G4: a manifest re-hashed consistently but carrying a non-neutral state key is refused at run
    from tools.api_batch.common import canon_json, sha256_text
    _ready(env)
    rows = _rows(env)
    for r in rows:
        r["params"]["state"] = {("notes" if k == "doc1" else k): v for k, v in r["params"]["state"].items()}
        r["params_sha256"] = sha256_text(canon_json(r["params"]))
    data = "\n".join(canon_json(r) for r in rows) + "\n"
    (env.pd / "manifest.jsonl").write_text(data, encoding="utf-8")
    pj = env.pd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    d["manifest_sha256"] = sha256_text(data)
    d["plan_hash"] = jev.compute_plan_hash(d["manifest_sha256"], d["est_usd_hard_worst"])
    pj.write_text(json.dumps(d), encoding="utf-8")
    _approve(env)
    c = jev.FakeJevClient()
    with pytest.raises(jev.JevError, match="neutral"):
        jev.run(env.pd, env.led, c, env.builder)
    assert c.calls == []


def test_normalize_exception_is_rc4(env, monkeypatch):
    _ready(env)
    monkeypatch.setattr(jev, "normalize", lambda raw, params: (_ for _ in ()).throw(TypeError("boom")))
    r = jev.run(env.pd, env.led, jev.FakeJevClient(), env.builder)
    assert [x["rc"] for x in r["requests"]] == [4, 4]
    assert len(env.led.rows("sync_done", lane="jev")) == len(env.led.rows("sync_sent", lane="jev")) == 2


@pytest.mark.parametrize("bad", ["q1\n", "ok\n"])
def test_regex_trailing_newline_refused(bad):
    # F9: fullmatch, not match-with-$
    assert not jev.QID_RE.fullmatch(bad) and not jev.LABEL_RE.fullmatch(bad) and not jev.ITEM_RE.fullmatch(bad)
