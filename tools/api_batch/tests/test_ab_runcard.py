"""runcard lane (wandb run-card prose, Stage B): spec, plan (card selection, skip rule, cohort halt, deny refusal, re-plan),
the deterministic post-check (tools/wandb_run_index/prose_check.py), the injector (prose_inject.py) and an offline
plan -> fake results -> check -> inject rehearsal on a scratch lab/ledger. Synthetic cards only; no network."""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from runcard_fixture import HEADER, ROWS, good_output, make_card
from tools.api_batch import ops
from tools.api_batch import runcard as RC
from tools.api_batch.common import REPO_ROOT, Halt
from tools.api_batch.fake_client import FakeClient
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder
from tools.api_batch.spec import SpecError, validate_spec

sys.path.insert(0, str(REPO_ROOT / "tools/wandb_run_index"))
import prose_check as PC  # noqa: E402
import prose_inject as PI  # noqa: E402
import pilot_gate as PG  # noqa: E402

KNOWN = [("aaaa1111", "a_k4_s1"), ("bbbb2222", "b_k4_s1"), ("cccc3333", "c_k4_s1"), ("dddd4444", "d_k4_s1"), ("eeee5555", "e_k4_s1"),
         ("jjjj0000", "j_k4_s1"), ("abc12345", "demo_k4_s1"), ("4s34eck3", "other_run_alpha"), ("abcdefgh", "other_run_beta"),
         ("0123abcd", "ABC_k2_ablation_H_cycles"), ("qwertyui", "fig1_tf_z_iter_k9_s1")]
KNOWN_SETS = ({i for i, _ in KNOWN}, {n for _, n in KNOWN})

SPEC = """name: rc-test
use: runcard
model: claude-sonnet-5-5
effort: low
max_tokens: 6000
cache_ttl: 1h
header: shared/runcard_header.md
prompt: {{system_template: tools/api_batch/prompts/runcard.md}}
output_schema: runcard_prose
cards: {{root: 21_experiments/12_run-cards, cohorts: {cohorts}}}
budget: {{cap_usd_hard_worst: 5.0}}
"""


def put(lab, project, text_or_kw):
    kw = text_or_kw
    text = make_card(project=project, **kw)
    cid = kw.get("id", "abc12345")
    p = lab / "21_experiments/12_run-cards" / project / f"{kw.get('name', 'demo_k4_s1')}__{cid}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def env(tmp_path, deny_files, lab):
    put(lab, "Demo", dict(id="aaaa1111", name="a_k4_s1"))
    put(lab, "Demo", dict(id="bbbb2222", name="b_k4_s1", status="no-summary"))
    put(lab, "Demo", dict(id="cccc3333", name="c_k4_s1", rows={}))                      # empty stats -> skipped
    put(lab, "Old", dict(id="dddd4444", name="d_k4_s1", cohort="legacy", tier="constructed-unverified"))
    put(lab, "Arc", dict(id="eeee5555", name="e_k4_s1", cohort="other"))
    idx = lab / PC.INDEX_REL
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text("".join(json.dumps({"id": i, "name": n}) + "\n" for i, n in KNOWN), encoding="utf-8")
    (lab / "21_experiments/12_run-cards/_meta").mkdir(parents=True)
    (lab / "21_experiments/12_run-cards/_meta/README.md").write_text("meta", encoding="utf-8")
    root = tmp_path / "root"
    bd = root / "11_batches" / "2026-11-10_rc"
    (bd / "shared").mkdir(parents=True)
    (bd / "shared/runcard_header.md").write_text(HEADER, encoding="utf-8")
    (bd / "spec.yaml").write_text(SPEC.format(cohorts="[clean, other]"), encoding="utf-8")
    builder = PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    return type("E", (), dict(root=root, bd=bd, lab=lab, builder=builder, led=Ledger(lab / "38_api-batch" / "ledger.jsonl"), deny=deny_files))


def plan(env):
    return ops.plan(env.bd, env.builder, env.led)


# ----------------------------------------------------------------------------------------------- spec
def test_spec_validation():
    ok = yaml.safe_load(SPEC.format(cohorts="[clean]"))
    assert validate_spec(copy.deepcopy(ok))["use"] == "runcard"
    for mut in (lambda s: s["cards"].update(cohorts=["clean", "nope"]), lambda s: s["cards"].update(cohorts=[]),
                lambda s: s.update(cache_ttl="none"), lambda s: s.update(header="../x.md"),
                lambda s: s["cards"].update(extra=1), lambda s: s.pop("output_schema"), lambda s: s.update(max_tokens=0),
                lambda s: s.update(model="claude-nope")):
        bad = copy.deepcopy(ok)
        mut(bad)
        with pytest.raises(SpecError):
            validate_spec(bad)


# ----------------------------------------------------------------------------------------------- plan
def test_plan_selects_cohorts_skips_empty_and_replans_identically(env):
    d = plan(env)
    assert d["n_requests"] == 3 and d["runcard"]["n_selected"] == 3                      # a, b (clean) + e (other)
    assert d["runcard"]["selected_by_cohort"] == {"clean": 2, "other": 1}
    assert [s["id"] for s in d["runcard"]["skipped"]] == ["cccc3333"] and d["runcard"]["n_skipped_no_stats"] == 1
    rows = [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]
    r = rows[0]
    assert r["meta"]["card_sha256"] and r["meta"]["card_rel"].startswith("21_experiments/12_run-cards/")
    blocks = r["params"]["messages"][0]["content"]
    assert blocks[0]["type"] == "document" and blocks[0].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[1]["type"] == "text" and "cache_control" not in blocks[1] and "Run card" in blocks[1]["text"]
    assert r["params"]["output_config"]["effort"] == "low" and r["params"]["max_tokens"] == 6000
    assert d["est_usd_hard_worst"] > 0 and d["masked_blocks"] == 0
    assert plan(env)["manifest_sha256"] == d["manifest_sha256"]                         # re-plan = same command, same hash


def test_replan_picks_up_changed_cards(env):
    d0 = plan(env)
    put(env.lab, "Demo", dict(id="ffff6666", name="f_k4_s1"))
    d1 = plan(env)
    assert d1["n_requests"] == d0["n_requests"] + 1 and d1["manifest_sha256"] != d0["manifest_sha256"]


def test_unknown_cohort_anywhere_halts(env):
    put(env.lab, "Demo", dict(id="gggg7777", name="g_k4_s1", cohort="unknown"))
    with pytest.raises(Halt, match="cohort"):
        plan(env)


def test_id_filename_mismatch_halts(env):
    p = put(env.lab, "Demo", dict(id="hhhh8888", name="h_k4_s1"))
    p.rename(p.with_name("h_k4_s1__zzzz0000.md"))
    with pytest.raises(Halt, match="does not match"):
        plan(env)


def test_deny_hit_refuses_whole_plan_nothing_masked(env):
    q = put(env.lab, "Demo", dict(id="iiii9999", name="i_k4_s1"))
    q.write_text(q.read_text().replace("## Related documents", "see per_call.csv\n\n## Related documents"))
    with pytest.raises(Halt, match="L3b|L2"):
        plan(env)
    assert not (env.bd / "plan.json").exists()


def test_card_sha_is_stable_across_injection_and_changes_with_content():
    t = make_card()
    filled = make_card(region="\nDRAFT text")
    assert RC.card_sha(t) == RC.card_sha(filled)
    assert RC.card_sha(t) != RC.card_sha(t.replace("seed: 1", "seed: 2"))
    with pytest.raises(RC.RunCardError):
        RC.card_sha(t.replace(RC.MARK_E, ""))


# ----------------------------------------------------------------------------------------------- post-check
def fails(out, **kw):
    return PC.check_output(make_card(**kw), out, KNOWN_SETS)


def req(kind, **kw):
    return PC.Card(make_card(**kw)).required_sentences()[kind]


def has(fl, tag):
    return any(tag in f for f in fl)


def test_good_output_passes():
    assert fails(good_output()) == []


def test_numbers_must_match_table_cells():
    o = good_output()
    o["what_the_numbers_show"][1]["value"] = "0.99"
    assert has(fails(o), "R2")
    o = good_output(); o["what_the_numbers_show"][0]["step"] = "4883"
    assert has(fails(o), "R2")
    o = good_output(); o["what_the_numbers_show"][0]["key"] = "train/nonexistent"
    assert has(fails(o), "not in the stats table")
    o = good_output(); o["what_the_numbers_show"][0]["value"] = "0.0039"          # rounded
    assert has(fails(o), "R2")
    o = good_output(); o["what_the_numbers_show"][2]["value"] = "0.00000005"        # reformatted 5e-08
    assert has(fails(o), "R2")


def test_free_text_numbers_derived_or_invented_are_rejected():
    for s in ("It rose to 0.5 at step 4882.", "probe/test_exact was 0.00390625 at step 4882, a 99% change.",
              "probe/test_exact went from 0.00390625 to 1, roughly 256 times", "n was 1,000 points.",
              "probe/test_exact was 0.00390625 at step 4882 and 4883."):
        o = good_output(); o["numbers_summary"] = s
        assert fails(o), s
    o = good_output(); o["what_this_run_is"] = "Run demo_k4_s1 with seed 1 and k 4 started 2026-07-22."
    assert fails(o) == []
    o = good_output(); o["what_this_run_is"] = "Run demo_k4_s1 with seed 2."
    assert has(fails(o), "R2")                                                              # 2 is not a card number
    o = good_output(); o["what_this_run_is"] = "Run demo_k4_s1, 30 points."                 # 30 only occurs inside a sha256 hex string
    assert has(fails(o), "'30'")
    c = PC.Card(make_card())
    assert "30" not in c.nums and "75186" not in c.nums and "99" not in c.nums and "244100" in c.nums
    assert "0.00390625" in c.nums and "5e-08" in c.nums and "4882" in c.nums


def test_ordinary_wording_is_not_over_rejected():
    o = good_output()
    o["what_this_run_is"] = "Run demo_k4_s1; H and L cycles are not listed; min/max and first/last are table columns."
    o["caveats"] = [{"kind": "other", "text": "The learning rate key train/lr is a working gloss, not verified against pretrain.py."}]
    o["numbers_summary"] = "probe/test_exact was 0.00390625 at step 4882 and 1 at step 244100, logged over 50 points."
    assert has(fails(o), "'50'") and not has(fails(o), "R5") and not has(fails(o), "R3")      # only the invented-count '50' is rejected


def test_summary_numbers_must_come_from_listed_entries():
    o = good_output(); o["numbers_summary"] = "train/lm_loss was 2.54363 at step 1."          # in the card, but not an entry
    assert has(fails(o), "numbers_summary")


def test_ids_paths_urls():
    o = good_output(); o["see_also"] = ["EXP-099_other.md"]
    assert has(fails(o), "see_also")
    o = good_output(); o["what_this_run_is"] = "Sibling run other_k5_s9 exists."
    assert has(fails(o), "R3")
    o = good_output(); o["run_id"] = "zzzz0000"
    assert has(fails(o), "run_id")
    url = "https://wandb.ai/someone/Demo/runs/abc12345"
    o = good_output(); o["what_this_run_is"] = f"Run page: {url}"
    assert fails(o) == []                                                                   # api-confirmed card URL: allowed
    o = good_output(extra_caveats=("link_tier",)); o["caveats"][0]["text"] = f"URL {url} is not api-confirmed."
    assert has(fails(o, tier="constructed-unverified"), "R4")                               # constructed tier: no URL at all
    o = good_output(); o["what_this_run_is"] = "Run page: https://wandb.ai/other/runs/x"
    assert has(fails(o), "R4")


def test_forbidden_words_and_api_confirmed_exemption():
    for w in ("This confirms the hypothesis.", "The result is proven.", "Loss fell because of EMA.", "H-023 is falsified.",
              "A strong result.", "The run collapsed."):
        o = good_output(); o["numbers_summary"] = f"probe/test_exact was 0.00390625 at step 4882. {w}"
        assert has(fails(o), "R5"), w
    o = good_output(); o["caveats"] = [{"kind": "other", "text": "Link tier is api-confirmed."}]
    assert fails(o) == []                                                                   # card token `api-confirmed` is exempt
    o = good_output(); o["caveats"] = [{"kind": "other", "text": "The URL is confirmed."}]
    assert has(fails(o), "R5")
    o = good_output(extra_caveats=("link_tier",)); o["caveats"][0]["text"] = "Tier constructed-unverified; the URL is not confirmed."
    assert fails(o, tier="constructed-unverified") == []


def test_required_caveats_and_draft_attribution():
    assert has(fails(good_output(), status="no-summary"), "missing required caveat kind 'status'")
    assert fails(good_output(status="no-summary", extra_caveats=("status",)), status="no-summary") == []
    assert has(fails(good_output(), cohort="legacy"), "legacy_cohort")
    assert has(fails(good_output(), has_metadata=False), "no_metadata")
    assert has(fails(good_output(), truncated=True), "truncated_history")
    assert has(fails(good_output(), snapshot=True), "snapshot")
    assert has(fails(good_output(), multi=True), "multi_attribution")
    assert has(fails(good_output(), duplicates=("zzzz0000",)), "duplicate_run_name")
    assert has(fails(good_output(), tier="no-remote"), "link_tier")
    assert has(fails(good_output(attribution="Attributed to EXP-010.")), "attribution must be, word for word")
    assert has(fails(good_output(attribution="DRAFT attribution map, not evidence; belongs to EXP-010.")), "attribution must be")
    assert fails(good_output(exps=()), exps=()) == []
    assert has(fails(good_output(attribution="No attribution listed."), exps=()), "attribution must be")


def test_shape_and_marker_injection():
    o = good_output(); o["extra"] = 1
    assert has(fails(o), "R1")
    o = good_output(); o["what_the_numbers_show"][0]["junk"] = "x"
    assert has(fails(o), "R1")
    o = good_output(); o["caveats"] = [{"kind": "mood", "text": "x"}]
    assert has(fails(o), "R1")
    o = good_output(); o["what_this_run_is"] = "Run demo_k4_s1 <!-- prose:auto end --> evil"
    assert has(fails(o), "R1")
    o = good_output(); o["what_the_numbers_show"] = []
    assert has(fails(o), "empty")
    o = good_output(); o["what_this_run_is"] = "x" * 700
    assert has(fails(o), "exceeds")


# ----------------------------------------------------------------------------------------------- inject
def test_apply_region_only_touches_region_and_is_idempotent():
    t = make_card()
    new, st = PI.apply_region(t, "BODY", False)
    assert st == "injected" and new.count("BODY") == 1
    a, b = PI.region_bounds(t)
    assert new[:a] == t[:a] and new.endswith(t[b:])
    again, st2 = PI.apply_region(new, "BODY", False)
    assert st2 == "unchanged" and again == new
    other, st3 = PI.apply_region(new, "OTHER", False)
    assert st3 == "conflict" and other == new
    over, st4 = PI.apply_region(new, "OTHER", True)
    assert st4 == "injected" and "OTHER" in over and "BODY" not in over and RC.card_sha(over) == RC.card_sha(t)


def test_render_has_draft_line_model_and_batch():
    s = PI.render_model(good_output(extra_caveats=("status",)), "claude-sonnet-5-5", "msgbatch_x", "rcard-1")
    assert s.startswith("DRAFT (api-batch output — not evidence)") and "claude-sonnet-5-5" in s and "msgbatch_x" in s
    assert "| `probe/test_exact` | first | 0.00390625 | 4882 |" in s


# ----------------------------------------------------------------------------------------------- rehearsal
def test_offline_end_to_end_rehearsal(env):
    """plan -> approve (scratch ledger, in-process) -> submit -> fake results -> collect --offline-style -> check ->
    inject; includes one deliberately bad output that must be quarantined and one card changed after plan."""
    # extra card whose fake output will be bad
    put(env.lab, "Demo", dict(id="jjjj0000", name="j_k4_s1"))
    d = plan(env)
    assert d["n_requests"] == 4
    ops.approve(env.bd, 100.0, env.led, env={})                                         # scratch ledger, test-only
    fake = FakeClient()
    ops.submit(env.bd, env.led, fake)
    rows = [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]
    texts, bad_cid, drift_cid = {}, None, None
    for r in rows:
        m = r["meta"]
        st = "no-summary" if m["run_id"] == "bbbb2222" else "ended"
        out = good_output(m["run_id"], m["cohort"], st, extra_caveats=("status",) if st != "ended" else ())
        out["what_this_run_is"] = f"Run named in card {m['run_id']}, status {st}."
        if m["run_id"] == "jjjj0000":
            out["numbers_summary"] = "probe/test_exact rose 256-fold, which confirms the hypothesis."
            bad_cid = r["custom_id"]
        texts[r["custom_id"]] = json.dumps(out)
    bid = env.led.last("submitted")["batch_id"]
    fake.finish(bid, texts=texts)
    ops.collect(env.bd, env.led, fake, env.root, env.builder, delete=False)
    chk = PC.check_batch(env.bd, env.lab)
    q = {x["run_id"] for x in chk["quarantined"]}
    assert q == {"jjjj0000"}
    assert [x["run_id"] for x in chk["quarantined"] if x["custom_id"] == bad_cid] == ["jjjj0000"]
    assert {x["run_id"] for x in chk["accepted"]} == {"aaaa1111", "bbbb2222", "eeee5555"}
    before = {p: p.read_bytes() for p in (env.lab / "21_experiments/12_run-cards").rglob("*.md")}
    rep = PI.inject(env.bd, env.lab)
    assert {x["run_id"] for x in rep["quarantined"]} == q and rep["fixed_fill"][0]["run_id"] == "cccc3333"
    after = {p: p.read_bytes() for p in before}
    changed = {p.name.split("__")[1][:-3] for p in before if before[p] != after[p]}
    assert changed == ({x["run_id"] for x in chk["accepted"]} | {"cccc3333"})            # quarantined + non-selected untouched
    for p in before:
        if before[p] != after[p]:
            o, n = before[p].decode(), after[p].decode()
            a, b = PI.region_bounds(o)
            na, nb = PI.region_bounds(n)
            assert n[:na] == o[:a] and n[nb:] == o[b:]                                   # outside the region: byte-identical
            assert RC.card_sha(n) == RC.card_sha(o)
    assert PI.inject(env.bd, env.lab)["injected"] == [] and not PI.inject(env.bd, env.lab)["conflict"]   # idempotent
    # a card changed after plan is refused (drift), never injected
    victim = next(iter(chk["accepted"]))
    vp = env.lab / victim["card_rel"]
    vp.write_text(vp.read_text().replace("seed: 1", "seed: 9"))
    edited = vp.read_text()
    rep2 = PI.inject(env.bd, env.lab, overwrite=True)
    assert has_r7({"quarantined": rep2["quarantined"]}, victim["run_id"]) and victim["run_id"] not in {x["run_id"] for x in rep2["injected"]}
    assert vp.read_text() == edited                                                          # left exactly as the user's edit


def has_r7(chk, rid):
    return any(x["run_id"] == rid and any("R7 card changed" in r for r in x["reasons"]) for x in chk["quarantined"])


def test_cli_exit_codes(env, capsys):
    plan(env)
    assert PC.main([str(env.bd), "--lab-root", str(env.lab)]) == 3                      # nothing collected -> missing
    assert PC.main([str(env.bd / "nope"), "--lab-root", str(env.lab)]) == 2


# ----------------------------------------------------------------------------------------------- r2: subset, truncated, snapshot, multi
def test_pack_text_is_blanked_card(env):
    p = put(env.lab, "Demo", dict(id="kkkk1010", name="k_k4_s1", region="\nOLD PROSE SHOULD NOT BE SENT"))
    (env.bd / "spec.yaml").write_text(SPEC.format(cohorts="[clean, other]").replace("cohorts: [clean, other]}", "cohorts: [clean, other], include_injected: true}"))
    plan(env)
    row = next(json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines() if "kkkk1010" in ln)
    tail = row["params"]["messages"][0]["content"][1]["text"]
    assert "OLD PROSE" not in tail and tail == RC.blanked_text(p.read_text())


def ids_spec(env, ids, extra=""):
    (env.bd / "ids.txt").write_text("# pilot\n" + "".join(f"{i}  # c\n" for i in ids), encoding="utf-8")
    (env.bd / "spec.yaml").write_text(SPEC.format(cohorts="[clean, other]").replace(
        "cards: {root: 21_experiments/12_run-cards, cohorts: [clean, other]}",
        "cards: {root: 21_experiments/12_run-cards, ids_file: ids.txt%s}" % extra), encoding="utf-8")


def test_ids_file_selection_across_cohorts_and_skips(env):
    ids_spec(env, ["aaaa1111", "dddd4444", "cccc3333"])                               # clean, legacy, empty-stats
    d = plan(env)
    assert d["n_requests"] == 2 and d["runcard"]["ids_file_sha256"] and d["runcard"]["n_ids"] == 3
    assert [s["id"] for s in d["runcard"]["skipped"]] == ["cccc3333"] and d["runcard"]["skipped"][0]["reason"] == "no_stats_rows"
    ids_spec(env, ["aaaa1111", "nonexistent1"])
    with pytest.raises(Halt, match="not found"):
        plan(env)
    ids_spec(env, ["aaaa1111", "aaaa1111"])
    with pytest.raises(Halt, match="duplicate"):
        plan(env)
    ids_spec(env, ["not an id"])
    with pytest.raises(Halt, match="not a run id"):
        plan(env)


def test_history_empty_is_skipped_with_reason(env):
    put(env.lab, "Demo", dict(id="llll1212", name="l_k4_s1", rows=ROWS, history_empty=True))
    d = plan(env)
    assert {s["id"]: s["reason"] for s in d["runcard"]["skipped"]}["llll1212"] == "history_empty"
    assert "llll1212" not in (env.bd / "manifest.jsonl").read_text()


def test_idsfile_rerun_does_not_resend_injected(env):
    ids_spec(env, ["aaaa1111", "bbbb2222"])
    p = env.lab / "21_experiments/12_run-cards/Demo/a_k4_s1__aaaa1111.md"
    p.write_text(make_card(id="aaaa1111", name="a_k4_s1", project="Demo", region="\nDRAFT injected"))
    d = plan(env)
    assert d["n_requests"] == 1 and d["runcard"]["n_skipped_already_injected"] == 1
    assert d["runcard"]["skipped_already_injected"][0]["id"] == "aaaa1111" and not d["runcard"]["skipped"]
    ids_spec(env, ["aaaa1111", "bbbb2222"], ", include_injected: true")
    assert plan(env)["n_requests"] == 2


def test_cohort_replan_after_pilot_inject_drops_injected_ids(env):
    d0 = plan(env)
    p = env.lab / "21_experiments/12_run-cards/Demo/a_k4_s1__aaaa1111.md"
    new, st = PI.apply_region(p.read_text(), "DRAFT (api-batch output - not evidence) pilot prose", False)
    p.write_text(new)                                                                   # as if the pilot's accepted prose was injected
    d1 = plan(env)
    assert d1["n_requests"] == d0["n_requests"] - 1 and d1["runcard"]["n_skipped_already_injected"] == 1
    assert d1["runcard"]["skipped_already_injected"][0]["id"] == "aaaa1111"
    assert "aaaa1111" not in (env.bd / "manifest.jsonl").read_text()
    (env.bd / "spec.yaml").write_text(SPEC.format(cohorts="[clean, other]").replace("cohorts: [clean, other]}", "cohorts: [clean, other], include_injected: true}"))
    assert plan(env)["n_requests"] == d0["n_requests"]                                  # explicit opt-in resends (pack text stays blanked)


def test_spec_ids_file_validation():
    base = yaml.safe_load(SPEC.format(cohorts="[clean]"))
    ok = copy.deepcopy(base); ok["cards"] = {"root": "21_experiments/12_run-cards", "ids_file": "ids.txt"}
    assert validate_spec(ok)
    for c in ({"root": "x"}, {"root": "x", "ids_file": "../up.txt"}, {"root": "x", "ids_file": "/abs.txt"},
              {"root": "x", "cohorts": ["clean"], "ids_file": "a.txt", "include_injected": "yes"}, {"root": "x", "cohorts": ["clean"], "zz": 1}):
        bad = copy.deepcopy(base); bad["cards"] = c
        with pytest.raises(SpecError):
            validate_spec(bad)


def test_rerun_ids_from_postcheck_excludes_accepted(env, tmp_path):
    put(env.lab, "Demo", dict(id="jjjj0000", name="j_k4_s1"))
    plan(env)
    ops.approve(env.bd, 100.0, env.led, env={})
    fake = FakeClient()
    ops.submit(env.bd, env.led, fake)
    rows = [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]
    texts = {}
    for r in rows:
        m = r["meta"]
        o = good_output(m["run_id"], m["cohort"], "ended", extra_caveats=("status",) if m["run_id"] == "bbbb2222" else ())
        if m["run_id"] in ("jjjj0000",):
            o["numbers_summary"] = "This proves it."
        texts[r["custom_id"]] = json.dumps(o)
    fake.finish(env.led.last("submitted")["batch_id"], texts=texts, outcomes={rows[0]["custom_id"]: "errored"})
    ops.collect(env.bd, env.led, fake, env.root, env.builder, delete=False)
    out = tmp_path / "rerun" / "ids.txt"
    assert PC.main([str(env.bd), "--lab-root", str(env.lab), "--rerun-ids", str(out)]) == 3
    got = {ln.split("#")[0].strip() for ln in out.read_text().splitlines() if ln.split("#")[0].strip()}
    chk = PC.check_batch(env.bd, env.lab)
    assert got == {x["run_id"] for x in chk["quarantined"] + chk["missing"]} and not got & {x["run_id"] for x in chk["accepted"]}
    assert "jjjj0000" in got and RC.read_ids(out)


def test_truncated_snapshot_multi_rules():
    tr = req("truncated_history", truncated=True)
    assert "292" in tr and "264" in tr and "final step" in tr
    o = good_output(truncated=True, extra_caveats=("truncated_history",))
    assert fails(o, truncated=True) == []
    # B3 variants
    for bad in ("summary_last_step 292 and history_last_step 264; the last value is the final value.", "History is short.",
                "Local history ends at step 264, before the summary step 292; the last values quoted here are the run's final step."):
        o2 = good_output(truncated=True, extra_caveats=("truncated_history",), texts={"truncated_history": bad})
        assert has(fails(o2, truncated=True), "verbatim"), bad
    for w in ("The final value at the end of the run was reported.", "The run finished.", "Training completed.", "The loss converged.", "It ended at step 264."):
        o3 = good_output(truncated=True, extra_caveats=("truncated_history",)); o3["numbers_summary"] += " " + w
        assert has(fails(o3, truncated=True), "R5"), w
    o4 = good_output(truncated=True, extra_caveats=("truncated_history",)); o4["numbers_summary"] = "probe/test_exact was 1 at step 244100."
    assert has(fails(o4, truncated=True), "local history")                              # last values quoted without saying so
    # B4 variants
    sn = req("snapshot", snapshot=True, status="running?")
    assert "2026-11-10T11:11:34+00:00" in sn
    def snap(texts=None, **kw):
        o_ = good_output(status="running?", snapshot=True, extra_caveats=("status", "snapshot"), texts={"snapshot": sn, **(texts or {})})
        o_.update(kw)
        return fails(o_, snapshot=True, status="running?")
    assert snap() == []
    for bad in ("Run is fine.", "Partial snapshot, but the run finished.", "Partial snapshot taken at 2026-11-10T11:11:34+00:00; the run may still have been in progress."):
        assert has(snap({"snapshot": bad}), "verbatim"), bad
    for w in ("The run finished and completed training.", "These are the final values.", "Training ended.", "The run converged."):
        fl = snap(what_this_run_is="Run demo_k4_s1 in project Demo. " + w)
        assert has(fl, "R5"), w
    assert has(snap(what_this_run_is="Run demo_k4_s1 in project Demo, status ended."), "'ended'")
    # multi-attribution exact sentence
    mt = req("multi_attribution", multi=True)
    assert "EXP-010, EXP-015, EXP-023" in mt and mt.endswith("this is unresolved here.")
    assert fails(good_output(multi=True, extra_caveats=("multi_attribution",)), multi=True) == []
    for bad in ("Unresolved.", "The attribution is unresolved between EXP-010, EXP-015 and EXP-023.", "This is an EXP-015 run; unresolved."):
        assert has(fails(good_output(multi=True, extra_caveats=("multi_attribution",), texts={"multi_attribution": bad}), multi=True), "verbatim"), bad
    # attribution is one fixed sentence: resolving it, listing ids, or dropping 'unresolved' is rejected
    for bad in ("DRAFT: EXP-010, EXP-015, EXP-023.", "DRAFT attribution map, not evidence; unresolved: EXP-010, EXP-015, EXP-023.",
                "DRAFT attribution map, not evidence; the script prints the listed ids below.",       # single-attribution sentence on a multi card
                "DRAFT attribution map, not evidence; the map lists several experiments and is unresolved here; the script prints the listed ids below. Most likely EXP-015."):
        assert has(fails(good_output(multi=True, extra_caveats=("multi_attribution",), attribution=bad), multi=True), "attribution must be"), bad


# --- verifier 16_STAGE_B_VERIFY-r1 adversarial cases, permanent -----------------------------------------------------------
def txt(field, value, **kw):
    o = good_output(); o[field] = value
    return fails(o, **kw)


def test_b1_foreign_run_ids_and_names_any_shape():
    assert has(txt("what_this_run_is", "Sibling 4s34eck3 exists."), "run id '4s34eck3'")          # digit-leading id
    assert has(txt("what_this_run_is", "Sibling abcdefgh exists."), "run id 'abcdefgh'")          # all-letter id
    assert has(txt("what_this_run_is", "See 0123abcd."), "0123abcd")
    assert has(txt("what_this_run_is", "Compare with other_run_alpha."), "run name 'other_run_alpha'")
    assert has(txt("what_this_run_is", "Compare with fig1_tf_z_iter_k9_s1 here."), "fig1_tf_z_iter_k9_s1")
    assert has(txt("what_this_run_is", "An unknown 8-char id zx9y8w7v."), "zx9y8w7v")            # generic 8-char token with a digit
    assert txt("what_this_run_is", "Run abc12345 (demo_k4_s1) in project Demo.") == []            # its own id and name are fine
    o = good_output(); o["caveats"] = [{"kind": "other", "text": "Duplicate qwertyui."}]
    assert has(fails(o), "qwertyui")


def test_b2_claim_vocabulary_en_and_ko():
    for w in ("This is evidence for a hypothesis.", "The values are consistent with the claim.", "The model generalizes.", "The model generalises.",
              "The table shows that loss fell.", "This indicates a change.", "This suggests overfitting.", "Loss fell due to the schedule.",
              "Loss fell because of the schedule.", "It demonstrates progress.", "A perfect result.", "The run failed.", "It memorizes the data.",
              "The run is solved.", "It reveals structure.", "Accuracy rose, hence learning.", "This explains the gap.", "Thus accuracy rose.",
              "This leads to a plateau."):
        assert has(txt("numbers_summary", "probe/test_exact was 1 at step 244100. " + w), "R5"), w
    for w in ("가설을 입증한다.", "가설을 반증한다.", "결과가 지지한다.", "시사한다.", "증명됨.", "확인되었다.", "뒷받침한다."):
        fl = txt("numbers_summary", "probe/test_exact was 1 at step 244100. " + w)
        assert has(fl, "outside printable ASCII"), w                                                  # Hangul rejected by the character allow-list
    assert txt("numbers_summary", "probe/test_exact was 1 at step 244100.") == []
    assert PC.FORBIDDEN_FILE.name == "prose_forbidden.txt" and PC.EXEMPT_PHRASES


def test_fine_outputs_are_not_rejected():
    o = good_output()
    o["what_this_run_is"] = "Run demo_k4_s1 in project Demo, arch preset trm, k 4, seed 1, started 2026-07-22, status ended."
    o["caveats"] = [{"kind": "other", "text": "Metric labels are repeated as printed: working gloss from the key name / code reference; not verified against pretrain.py."}]
    o["numbers_summary"] = "probe/test_exact was 0.00390625 at step 4882 and 1 at step 244100."
    assert fails(o) == []
    o["caveats"] = [{"kind": "other", "text": "Definition: assumed extension of a verified definition - not independently checked."}]
    assert fails(o) == []
    o["caveats"] = [{"kind": "other", "text": "Definition label: verified on this run only."}]
    assert fails(o) == []


def test_loader_rejects_bad_vocab_file(tmp_path):
    f = tmp_path / "v.txt"
    f.write_text("[en]\n(unclosed\n", encoding="utf-8")
    with pytest.raises(Exception):
        PC.load_forbidden(f)
    f.write_text("[zz]\nx\n", encoding="utf-8")
    with pytest.raises(ValueError):
        PC.load_forbidden(f)
    f.write_text("[exempt]\nx\n", encoding="utf-8")
    with pytest.raises(ValueError):
        PC.load_forbidden(f)


def test_card_label_constants_are_exempt():
    sys.path.insert(0, str(REPO_ROOT / "tools/wandb_run_index"))
    import importlib.util
    sp = importlib.util.spec_from_file_location("wri_common", REPO_ROOT / "tools/wandb_run_index/common.py")
    cm = importlib.util.module_from_spec(sp); sp.loader.exec_module(cm)
    for label in cm.LABELS:
        assert any(e in label for e in PC.EXEMPT_PHRASES), label


def test_fixed_fill_wording_for_history_empty():
    s = PI.render_fixed("ended", "legacy", "history_empty")
    assert "history_empty" in s and s.startswith("DRAFT (script-filled")
    assert "empty stats table" in PI.render_fixed("ended", "legacy")


def test_pilot_ids_strata(tmp_path):
    import pilot_ids as PL
    root = tmp_path / "cards"
    spec = []
    n = 0
    def mk(**kw):
        nonlocal n
        n += 1
        kw.setdefault("id", f"id{n:06d}"); kw.setdefault("name", f"run_{n}"); kw.setdefault("project", "P")
        p = root / "P" / f"{kw['name']}__{kw['id']}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(make_card(**kw), encoding="utf-8")
    for i in range(12):
        mk(cohort="clean"); mk(cohort="legacy")
    for i in range(4):
        mk(cohort="other"); mk(truncated=True); mk(multi=True); mk(z=True)
        mk(name=f"pp_ltf_g7_lay2_k5_s{4 + i % 3}", multi=True)
    mk(snapshot=True, status="running?"); mk(status="no-summary"); mk(status="no-summary", tier="no-remote")
    mk(history_empty=True, rows={})
    mk(tier="constructed-unverified")
    feats = PL.features(root)
    assert all(not f["id"].startswith("zz") for f in feats)
    sel1, why1 = PL.select(feats, 7, 24)
    sel2, _ = PL.select(feats, 7, 24)
    assert [f["id"] for f in sel1] == [f["id"] for f in sel2] and len(sel1) == 24
    sel3, _ = PL.select(feats, 8, 24)
    assert [f["id"] for f in sel3] != [f["id"] for f in sel1]
    c = lambda pred: sum(1 for f in sel1 if pred(f))  # noqa: E731
    assert c(lambda f: f["cohort"] == "clean") >= 6 and c(lambda f: f["cohort"] == "legacy") >= 6 and c(lambda f: f["cohort"] == "other") >= 3
    assert c(lambda f: f["trunc"]) >= 3 and c(lambda f: f["snap"]) >= 1 and c(lambda f: f["multi"]) >= 4
    assert c(lambda f: f["multi"] and f["pp"]) >= 2 and c(lambda f: f["status"] == "no-summary") >= 2
    assert c(lambda f: f["tier"] != "api-confirmed") >= 2 and c(lambda f: f["z"]) >= 2
    assert max(f["rows"] for f in feats) == max(f["rows"] for f in sel1)


# ----------------------------------------------------------------------------------------------- pilot gate
def collect_all(env, mutate=None, outcomes=None):
    plan(env)
    ops.approve(env.bd, 100.0, env.led, env={})
    fake = FakeClient()
    ops.submit(env.bd, env.led, fake)
    rows = [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]
    texts = {}
    for r in rows:
        m = r["meta"]
        st = "no-summary" if m["run_id"] == "bbbb2222" else "ended"
        o = good_output(m["run_id"], m["cohort"], st, extra_caveats=("status",) if st != "ended" else ())
        o["what_this_run_is"] = f"Run named in card {m['run_id']}."
        if mutate:
            mutate(m["run_id"], o)
        texts[r["custom_id"]] = json.dumps(o)
    fake.finish(env.led.last("submitted")["batch_id"], texts=texts, outcomes=outcomes or {})
    ops.collect(env.bd, env.led, fake, env.root, env.builder, delete=False)
    return rows


def gate_rc(env, capsys):
    rc = PG.main([str(env.bd), "--lab-root", str(env.lab)])
    return rc, capsys.readouterr().out


def edit_item(env, cid, **repl):
    ip = env.bd / "items" / f"{cid}.md"
    t = ip.read_text()
    for k, v in repl.items():
        t = re.sub(rf"^{k}: .*$", f"{k}: {v}", t, flags=re.M)
    ip.write_text(t)


import re  # noqa: E402


def test_gate_passes_on_clean_pilot_and_prints_numbers(env, capsys):
    collect_all(env)
    rc, out = gate_rc(env, capsys)
    assert rc == 0 and "PASS (exit 0)" in out and "G4" in out and "0.000" in out and "G5" in out and "histogram" in out


def test_gate_fails_on_each_rule(env, capsys):
    rows = collect_all(env, outcomes=None)
    c0 = rows[0]["custom_id"]
    edit_item(env, c0, stop_reason="max_tokens", result_status="max_tokens")
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G2" in out
    edit_item(env, c0, stop_reason="end_turn", result_status="succeeded", usage='{"output_tokens": 4000, "input_tokens": 1}')
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G5" in out and "PASS  G2" in out
    edit_item(env, c0, usage='{"output_tokens": 100}', model_returned="claude-opus-5-5")
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G3" in out
    edit_item(env, c0, model_returned="claude-sonnet-5-5", cost_est_usd="99.0")
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G6" in out
    edit_item(env, c0, cost_est_usd="0.001")
    (env.bd / "items" / f"{rows[1]['custom_id']}.md").unlink()
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G1" in out


def test_gate_missing_usage_or_cost_fails_not_passes(env, capsys):
    rows = collect_all(env)
    c0 = rows[0]["custom_id"]
    ip = env.bd / "items" / f"{c0}.md"
    keep = ip.read_text()
    ip.write_text(re.sub(r"^usage: .*\n", "", keep, flags=re.M))                       # usage field gone: must FAIL G5, never read as 0 tokens
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G5" in out and "1 items without usage" in out
    ip.write_text(re.sub(r"^cost_est_usd: .*\n", "", keep, flags=re.M))                # cost gone: FAIL G6
    rc, out = gate_rc(env, capsys)
    assert rc == 1 and "FAIL  G6" in out and "1 items without cost_est_usd" in out
    ip.write_text(keep)
    assert gate_rc(env, capsys)[0] == 0


def test_gate_quarantine_rate_threshold_and_histogram(env, capsys):
    collect_all(env, mutate=lambda rid, o: o.update(numbers_summary="This proves it.") if rid != "aaaa1111" else None)
    rc, out = gate_rc(env, capsys)                                                    # 2 of 3 quarantined > 0.20
    assert rc == 1 and "FAIL  G4" in out and "R5 numbers_summary: forbidden word '…'" in out


def test_gate_error_exit_and_prose_check_exit_codes(env, capsys):
    assert PG.main([str(env.bd / "nope"), "--lab-root", str(env.lab)]) == 2
    plan(env)
    assert PC.main([str(env.bd), "--lab-root", str(env.lab)]) == 3                    # ran, items missing
    (env.lab / PC.INDEX_REL).unlink()
    assert PC.main([str(env.bd), "--lab-root", str(env.lab)]) == 2                    # cannot run (no run index) = error, not 3


# ----------------------------------------------------------------------------------------------- incident 2026-11-10: in-flight re-plan
def test_plan_refuses_inflight_batch_dir_and_collected_raw(env):
    plan(env)
    ops.approve(env.bd, 100.0, env.led, env={})
    fake = FakeClient()
    ops.submit(env.bd, env.led, fake)
    before = (env.bd / "manifest.jsonl").read_bytes()
    with pytest.raises(Halt, match="not yet collected"):
        plan(env)                                                                     # submitted, not collected: refuse
    assert (env.bd / "manifest.jsonl").read_bytes() == before                         # nothing overwritten
    # collected: raw/ holds results -> still refused (new dir required)
    rows = [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]
    texts = {r["custom_id"]: json.dumps(good_output(r["meta"]["run_id"], r["meta"]["cohort"], "no-summary" if r["meta"]["run_id"] == "bbbb2222" else "ended",
                                                    extra_caveats=("status",) if r["meta"]["run_id"] == "bbbb2222" else ())) for r in rows}
    fake.finish(env.led.last("submitted")["batch_id"], texts=texts)
    ops.collect(env.bd, env.led, fake, env.root, env.builder, delete=False)
    with pytest.raises(Halt, match="raw/"):
        plan(env)
    assert (env.bd / "manifest.jsonl").read_bytes() == before
    # a fresh directory (same spec) plans fine
    nd = env.root / "11_batches" / "2026-11-10_rc2"
    shutil.copytree(env.bd / "shared", nd / "shared")
    (nd / "spec.yaml").write_text((env.bd / "spec.yaml").read_text())
    assert ops.plan(nd, env.builder, env.led)["n_requests"] == 3


def sent_record(env, tamper=False):
    """Simulate the incident: copy the as-sent record, then let the top-level plan.json/manifest disagree with it."""
    bd = env.bd
    (bd / "as_sent").mkdir()
    for n in ("manifest.jsonl", "plan.json"):
        shutil.copy(bd / n, bd / "as_sent" / n)
    (bd / "plan.json").write_text((bd / "plan.json").read_text().replace('"manifest_sha256": "', '"manifest_sha256": "0'))   # top level now wrong
    if tamper:                                                                         # a self-consistent as_sent that the ledger never saw
        import tools.api_batch.manifest as M
        mp = bd / "as_sent" / "manifest.jsonl"
        rows = [json.loads(ln) for ln in mp.read_text().splitlines()]
        rows[0]["meta"]["card_sha256"] = "f" * 64
        mp.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
        msha = M.load_manifest(mp)[1]
        pp = bd / "as_sent" / "plan.json"
        d = json.loads(pp.read_text()); d["manifest_sha256"] = msha
        pp.write_text(json.dumps(d))


def test_check_inject_gate_use_the_as_sent_record(env, capsys):
    collect_all(env)
    sent_record(env)
    res = PC.check_batch(env.bd, env.lab)
    assert res["manifest_source"] == "as_sent" and len(res["accepted"]) == 3 and not res["quarantined"]
    rc, out = gate_rc(env, capsys)
    assert rc == 0 and "(as_sent)" in out
    assert PI.inject(env.bd, env.lab, dry_run=True)["injected"]
    # card changed after it was sent -> R7 and inject refuses, judged against the AS-SENT card_sha256
    cp = next((env.lab / "21_experiments/12_run-cards/Demo").glob("a_k4_s1__*.md"))
    cp.write_text(cp.read_text().replace("seed: 1", "seed: 9"))
    res = PC.check_batch(env.bd, env.lab)
    assert has_r7(res, "aaaa1111")
    assert "aaaa1111" not in {x["run_id"] for x in PI.inject(env.bd, env.lab, overwrite=True)["injected"]}


def test_as_sent_manifest_must_match_the_ledger_submitted_row(env):
    collect_all(env)
    sent_record(env, tamper=True)
    with pytest.raises(ValueError, match="ledger submitted manifest"):
        PC.check_batch(env.bd, env.lab)


def test_top_level_manifest_must_match_its_plan_without_as_sent(env):
    collect_all(env)
    (env.bd / "plan.json").write_text((env.bd / "plan.json").read_text().replace('"manifest_sha256": "', '"manifest_sha256": "0'))
    with pytest.raises(ValueError, match="does not match"):
        PC.check_batch(env.bd, env.lab)


def test_rule_set_selected_per_request_and_unknown_is_an_error():
    assert PC.rules_of({"rules": "r4"}) == "r4"
    assert PC.rules_of({"prompt_sha256": "b4acc404f8790d79920ffc66a96eafd594b9a501feae3d36e83afb580ae8118d"}) == "r4-asent"
    with pytest.raises(ValueError, match="unknown rule set"):
        PC.rules_of({"prompt_sha256": "0" * 64, "run_id": "x"})
    card = make_card()
    for text, strict_ok in (("Run demo_k4_s1 in project Demo, the status was ended.", False), ("Run demo_k4_s1 in project Demo, recorded as `ended`.", False),
                            ("Run demo_k4_s1 in project Demo, status ended.", True)):
        o = good_output(); o["what_this_run_is"] = text
        assert (PC.check_output(card, o, KNOWN_SETS, "r4") == []) == strict_ok, text
        assert PC.check_output(card, o, KNOWN_SETS, "r4-asent") == [], text
    o = good_output(); o["what_this_run_is"] = "Run demo_k4_s1 in project Demo, it ended at step 244100."
    assert PC.check_output(card, o, KNOWN_SETS, "r4-asent")                      # loose rule still rejects bare 'ended'


def test_step_with_leading_at_only_under_the_as_sent_rule_set():
    card = make_card()
    o = good_output()
    for e in o["what_the_numbers_show"]:
        e["step"] = "@" + e["step"]
    assert PC.check_output(card, o, KNOWN_SETS, "r4-asent") == []
    assert any("R2" in f for f in PC.check_output(card, o, KNOWN_SETS, "r4"))
    body = PI.render_model(o, "m", "b", "c", card)
    assert "| 4882 |" in body and "@4882" not in body
    o2 = good_output(); o2["what_the_numbers_show"][0]["step"] = "@4883"
    assert PC.check_output(card, o2, KNOWN_SETS, "r4-asent")


# ----------------------------------------------------------------------------------------------- r5 holes
def tamper_top_level(bd):
    """Replace the top-level manifest/plan by a self-consistent record that differs from what was submitted."""
    import tools.api_batch.manifest as M
    mp = bd / "manifest.jsonl"
    rows = [json.loads(ln) for ln in mp.read_text().splitlines()]
    rows[0]["meta"]["card_sha256"] = "f" * 64
    mp.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    pp = bd / "plan.json"
    d = json.loads(pp.read_text()); d["manifest_sha256"] = M.load_manifest(mp)[1]
    pp.write_text(json.dumps(d))


def test_fail_closed_when_sent_record_missing_or_mismatched(env, capsys):
    collect_all(env)
    assert PC.check_batch(env.bd, env.lab)["manifest_source"] == "top"                  # top-level == submitted: fine
    tamper_top_level(env.bd)                                                           # re-planned after submit, no as_sent
    with pytest.raises(ValueError, match="fail closed"):
        PC.check_batch(env.bd, env.lab)
    assert PC.main([str(env.bd), "--lab-root", str(env.lab)]) == 2
    assert PG.main([str(env.bd), "--lab-root", str(env.lab)]) == 2
    assert PI.main([str(env.bd), "--lab-root", str(env.lab)]) == 2
    capsys.readouterr()


def test_fail_closed_without_a_ledger(env):
    collect_all(env)
    env.led.path.unlink()
    with pytest.raises(ValueError, match="ledger"):
        PC.check_batch(env.bd, env.lab)


def test_plan_guard_resolves_relative_batch_dir_from_another_cwd(env, monkeypatch):
    plan(env)
    ops.approve(env.bd, 100.0, env.led, env={})
    ops.submit(env.bd, env.led, FakeClient())
    other = env.root
    monkeypatch.chdir(other)
    rel = Path("11_batches/2026-11-10_rc")
    assert not rel.is_absolute()
    with pytest.raises(Halt, match="not yet collected"):
        ops.plan(rel, env.builder, env.led)
    # a ledger row recorded repo-relative reads the same directory as the absolute one, whatever the cwd
    from tools.api_batch.common import canon_batch_dir
    assert canon_batch_dir("lab/38_api-batch") == (REPO_ROOT / "lab/38_api-batch").resolve()
    monkeypatch.chdir("/")
    assert canon_batch_dir("lab/38_api-batch") == (REPO_ROOT / "lab/38_api-batch").resolve()


def test_inject_dry_run_writes_nothing_into_the_batch_dir(env, tmp_path):
    collect_all(env)
    snap = lambda: {str(p.relative_to(env.bd)): p.read_bytes() for p in env.bd.rglob("*") if p.is_file()}   # noqa: E731
    cards = lambda: {str(p): p.read_bytes() for p in (env.lab / "21_experiments/12_run-cards").rglob("*.md")}   # noqa: E731
    before, cbefore = snap(), cards()
    rep = PI.inject(env.bd, env.lab, dry_run=True)
    assert rep["injected"] and snap() == before and cards() == cbefore
    assert PI.main([str(env.bd), "--lab-root", str(env.lab), "--dry-run"]) == 0 and snap() == before
    out = tmp_path / "rep" / "r.json"
    PI.main([str(env.bd), "--lab-root", str(env.lab), "--dry-run", "--report", str(out)])
    assert out.is_file() and snap() == before                                          # report only where the caller asked
    PI.inject(env.bd, env.lab)
    assert (env.bd / "inject_report.json").is_file()                                   # a real run still writes it


@pytest.mark.parametrize("text", ["See https://example.org/x.", "ftp://x.y/z", "file:///etc/passwd", "www.example.com", "WWW.Example.com",
                                  "mailto:a@b.co", "wandb.ai/raon1123/p/runs/x", "evil.example.com", "see notes.md", "paper.pdf",
                                  "a.b.c.dev", "http://localhost/x", "data://x"])
def test_link_like_text_is_rejected(text):
    o = good_output(); o["what_this_run_is"] += " " + text
    assert any(f.startswith("R4") or f.startswith("R3") for f in fails(o)), text


def test_dotted_names_printed_in_the_card_are_allowed():
    o = good_output()
    assert fails(o) == [] and o["see_also"] == ["EXP-010_demo.md"]                  # printed dotted name, in see_also: fine
    o["what_this_run_is"] += " Also abc.def."
    assert any("dotted domain" in f for f in fails(o))


def test_prose_check_and_gate_write_into_the_batch_dir_only_with_write(env, tmp_path, capsys):
    collect_all(env)
    snap = lambda: {str(p.relative_to(env.bd)): p.read_bytes() for p in env.bd.rglob("*") if p.is_file()}   # noqa: E731
    before = snap()
    assert PC.main([str(env.bd), "--lab-root", str(env.lab)]) == 0
    assert PG.main([str(env.bd), "--lab-root", str(env.lab)]) == 0
    assert snap() == before and not (env.bd / "postcheck.json").exists() and not (env.bd / "pilot_gate.txt").exists()
    out = tmp_path / "r" / "pc.json"
    assert PC.main([str(env.bd), "--lab-root", str(env.lab), "--report", str(out)]) == 0
    assert json.loads(out.read_text())["accepted"] and snap() == before
    gout = tmp_path / "r" / "gate.txt"
    PG.main([str(env.bd), "--lab-root", str(env.lab), "--report", str(gout)])
    assert "PASS (exit 0)" in gout.read_text() and snap() == before
    assert PC.main([str(env.bd), "--lab-root", str(env.lab), "--write"]) == 0 and (env.bd / "postcheck.json").is_file()
    assert PG.main([str(env.bd), "--lab-root", str(env.lab), "--write"]) == 0 and (env.bd / "pilot_gate.txt").is_file()
    capsys.readouterr()
