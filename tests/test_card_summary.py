"""r5: the script-rendered Summary section of the run cards (tools/wandb_run_index/common.py render_summary) and its checks over
EVERY real card (not a sample): numbers equal the stats table, no interpretive / status / verdict word outside the fixed
templates, 1,283 cards, link.url on exactly 1,221, no broken related links."""
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests._wri_import import wri  # noqa: E402

C = wri("common")  # not a bare `import common`: dataset/ has its own module of that name

CARDS = REPO / "lab/21_experiments/12_run-cards"
FILES = sorted(p for p in CARDS.glob("*/*.md") if p.parent.name != "_meta") if CARDS.is_dir() else []

STATS = {"status": "ok", "keys": {
    "probe/test_exact": {"n": 3, "first": 0.0039, "first_step": 10, "last": 1, "last_step": 300, "min": 0.0039, "min_step": 10, "max": 1, "max_step": 300},
    "train/lm_loss": {"n": 3, "first": 2.5, "first_step": 1, "last": 0.25, "last_step": 300, "min": 0.1, "min_step": 200, "max": 7.5, "max_step": 50},
    "zz/other": {"n": 1, "first": 3, "first_step": 5, "last": 3, "last_step": 5, "min": 3, "min_step": 5, "max": 3, "max_step": 5}}}


def fm_of(**kw):
    fm = {"id": "abc12345", "name": "demo_k4_s1", "project": "Demo", "started": "2026-07-22T10:29:14Z", "status": "ended", "has_metadata": True,
          "history_truncated": False, "snapshot": False, "multi_attribution": False, "cohort": "clean",
          "link": {"tier": "api-confirmed", "url_status": "confirmed", "url": "https://wandb.ai/x/Demo/runs/abc12345"},
          "config": {"arch_preset": "trm", "arch_name": None, "H_cycles": 3, "L_cycles": 6, "k": 4, "seed": 1,
                     "data_paths": ["data/sigma_k_10/4"], "ema": True, "log_z_dynamics": False},
          "attribution": {"exps": ["EXP-010"], "hyps": [], "exp_attribution": [{"id": "EXP-010", "source": "map", "confidence": "registry"}],
                          "hyp_attribution": []}, "duplicates": []}
    fm.update(kw)
    return fm


def test_summary_parts_basic():
    p = C.summary_parts(fm_of(), STATS)
    assert p["config"] == ("Run `demo_k4_s1` id `abc12345` in project `Demo`; model variant `trm`; k `4`; seed `1`; H cycles `3`; L cycles `6`; "
                           "data path `data/sigma_k_10/4`; ema `true`; log_z_dynamics `false`; started `2026-07-22`; local status `ended`.")
    assert p["metrics"][0] == "- `probe/test_exact`: first 0.0039 (@10), last 1 (@300)"              # min/max equal first/last: omitted
    assert p["metrics"][1] == "- `train/lm_loss`: first 2.5 (@1), last 0.25 (@300), min 0.1 (@200), max 7.5 (@50)"
    assert p["metrics"][2].startswith("- `zz/other`")                                                  # unknown keys after the priority list
    assert p["caveats"] == [] and p["attribution"].startswith("DRAFT attribution map, not evidence: EXP-010 (source map, confidence registry)")


def test_summary_caveat_templates_follow_the_flags():
    fm = fm_of(status="no-summary", has_metadata=False, history_truncated=True, history_last_step=264, summary_last_step=292,
               snapshot=True, snapshot_utc="2026-11-10T11:11:34+00:00", multi_attribution=True, cohort="legacy",
               duplicates=[{"id": "zzzz0000"}], link={"tier": "constructed-unverified", "url_status": "x", "url": "u"},
               attribution={"exps": ["EXP-010", "EXP-015"], "hyps": ["H-1"], "exp_attribution": [], "hyp_attribution": []}, name=None)
    p = C.summary_parts(fm, STATS)
    assert p["config"].startswith("Run with no name id `abc12345`")
    assert p["caveats"] == [C.T_NO_SUMMARY, C.T_NO_META,
                            "Local history ends at step 264, before the summary step 292; the last values quoted here are not the run's final step.",
                            "Partial snapshot taken at 2026-11-10T11:11:34+00:00; the run may still have been in progress, so these values are not final.",
                            "Other runs share this project and run name: zzzz0000.",
                            "The attribution map assigns this run to several experiments (EXP-010, EXP-015); this is unresolved here.",
                            "The link tier is constructed-unverified; the URL is not api-confirmed.", C.T_LEGACY_GENERIC]
    assert "EXP-010; EXP-015; H-1" in p["attribution"].replace(" (source None, confidence None)", "")
    assert C.summary_parts(fm_of(link={"tier": "no-remote", "url": None}), STATS)["caveats"] == [C.T_TIER_OFFLINE]
    assert C.summary_parts(fm_of(history_empty=True), {"status": "ok", "keys": {}})["metrics"] == []
    assert C.summary_parts(fm_of(attribution={"exps": [], "hyps": []}), STATS)["attribution"] == C.T_ATTR_NONE
    assert "killed" not in C.T_NO_SUMMARY and "crashed" not in C.T_NO_SUMMARY
    assert "local run directory" in C.T_NO_SUMMARY and "local files do not record an end state" in C.T_NO_SUMMARY


def test_render_summary_is_a_pure_function():
    a = C.render_summary(fm_of(), STATS)
    assert a == C.render_summary(fm_of(), STATS) and a[0] == C.SUMMARY_TITLE


needs_cards = pytest.mark.skipif(not FILES, reason="run cards not present")


def parse_card(path):
    t = path.read_text(encoding="utf-8")
    fm = yaml.safe_load(t[4:t.index("\n---\n", 4)])
    summ = t[t.index(C.SUMMARY_TITLE):t.index("\n## Symbols")]
    sec = t.split("## Precomputed stats", 1)[1].split("\n## ", 1)[0]
    table = {}
    for ln in sec.splitlines():
        if ln.startswith("| `"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            table[c[0].strip("`")] = dict(zip(("first", "last", "min", "max"), c[2:6]))
    return t, fm, summ, table


@needs_cards
def test_every_card_has_the_summary_and_its_numbers_equal_the_stats_table():
    n = nm = 0
    for p in FILES:
        t, fm, summ, table = parse_card(p)
        n += 1
        lines = [ln for ln in summ.splitlines() if ln.startswith("- `")]
        assert {re.match(r"- `([^`]+)`", ln).group(1) for ln in lines} == set(table), p.name        # every table key, no other
        order = [re.match(r"- `([^`]+)`", ln).group(1) for ln in lines]
        pri = [k for k in C.SUMMARY_PRIORITY if k in table]
        assert order[:len(pri)] == pri and order[len(pri):] == sorted(order[len(pri):]), p.name       # fixed order
        for ln in lines:
            key = re.match(r"- `([^`]+)`", ln).group(1)
            got = dict(re.findall(r"(first|last|min|max) (\S+ \(@\S+?\))(?:,|$)", ln))
            assert got["first"] == table[key]["first"] and got["last"] == table[key]["last"], (p.name, key)
            for st in ("min", "max"):
                if st in got:
                    assert got[st] == table[key][st], (p.name, key, st)                              # identical string, no re-rounding
                else:
                    v = table[key][st].split(" ")[0]
                    assert v in (table[key]["first"].split(" ")[0], table[key]["last"].split(" ")[0]), (p.name, key, st)
            nums_line = re.findall(r"-?\d[\w.+\-]*", ln.split(":", 1)[1])
            cells = " ".join(" ".join(table[key].values()).replace("(@", " ").replace(")", " ").split())
            assert all(x in cells.split() for x in nums_line), (p.name, key, nums_line)               # every number is a table token
        nm += len(lines)
    assert n == 1283 and nm > 10000, (n, nm)


@needs_cards
def test_rendered_summary_has_no_interpretive_status_or_verdict_word_outside_the_templates():
    PC = wri("prose_check")
    templ = [v for k, v in vars(C).items() if k.startswith("T_") and isinstance(v, str)]
    hits = []
    for p in FILES:
        t, fm, summ, table = parse_card(p)
        s = summ
        for tp in templ:                                         # fixed caveat templates, with their {x} filled by card fields
            pat = re.escape(tp)
            pat = re.sub(r"\\\{[a-z_]+\\\}", r".+?", pat)
            s = re.sub(pat, " ", s)
        s = re.sub(r"`[^`]*`", " ", s)                            # backticked card values: names, paths, keys, status value
        for lab in ("local status", "local exit_code", "wandb API api_state"):   # fixed field labels of the config sentence
            s = s.replace(lab, " ")
        s = re.sub(r"DRAFT attribution map, not evidence: [^\n]*", " ", s)
        s = re.sub(r"\(source [^)]*\)", " ", s)
        for rx in (PC.FORBIDDEN_RE, PC.STATUS_RE, PC.NUMWORD_RE):
            m = rx.search(s)
            if m:
                hits.append((p.name, m.group(0)))
    assert not hits, hits[:5]


@needs_cards
def test_card_corpus_invariants():
    assert len(FILES) == 1283
    with_url = 0
    for p in FILES:
        t, fm, _s, _t = parse_card(p)
        with_url += bool((fm.get("link") or {}).get("url"))
        for label, rel in re.findall(r"^- \[([^\]]+)\]\(([^)]+)\)", t.split("## Related documents", 1)[1].split("\n## ", 1)[0], re.M):
            assert (p.parent / rel).exists(), (p.name, rel)
        assert "wandb-summary.json is absent" in t if fm.get("status") == "no-summary" else True
        assert "run killed" not in t and "crashed, or never" not in t
    assert with_url == 1221


# ---------------------------------------------------------------------------------------------- r6 (verifier minors M1-M4)
def test_end_state_fields_printed_by_source_and_flag_rule():
    fm = fm_of(status="ended", exit_code=1, link={"tier": "api-confirmed", "url": "u", "api_state": "crashed"}, end_state_fields_differ=True)
    p = C.summary_parts(fm, STATS)
    assert "local status `ended`; local exit_code `1`; wandb API api_state `crashed`." in p["config"]
    assert p["caveats"] == [C.T_END_STATE] and "not reconcile" in C.T_END_STATE
    # fields absent are not printed; no flag, no sentence
    p0 = C.summary_parts(fm_of(exit_code=None), STATS)
    assert "exit_code" not in p0["config"] and "api_state" not in p0["config"] and C.T_END_STATE not in p0["caveats"]
    f = C.end_state_fields_differ
    assert f("ended", 1, "finished") and f("ended", 0, "crashed") and f("no-summary", None, "killed") and f("ended", None, "failed")
    assert not f("ended", 0, "finished") and not f("ended", None, None) and not f("no-summary", 1, None) and not f("running?", None, "running")
    assert not f("ended", True, None)                                                                  # a bool is not an exit code
    assert "end_state_fields_differ is true iff" in C.END_STATE_RULE_TEXT


def test_no_summary_sentence_is_local_and_api_state_clause_follows():
    fm = fm_of(status="no-summary", link={"tier": "api-confirmed", "url": "u", "api_state": "crashed"}, end_state_fields_differ=True)
    cav = C.summary_parts(fm, STATS)["caveats"]
    assert cav[0] == C.T_NO_SUMMARY + " The wandb API record for this run lists api_state `crashed`."
    assert cav[1] == C.T_END_STATE
    assert C.summary_parts(fm_of(status="no-summary"), STATS)["caveats"] == [C.T_NO_SUMMARY]


def test_legacy_sentence_states_only_the_cohort_rule_outcome():
    fm = fm_of(cohort="legacy", started="2026-07-10T08:00:00Z", config={"data_paths": ["data/sigma_k_10/4"]})
    s = C.summary_parts(fm, STATS)["caveats"][0]
    assert s == ("Cohort legacy per the cohort rule: data path `data/sigma_k_10/4` and start time `2026-07-10T08:00:00Z` are before "
                 "`2026-07-20T13:18:00Z`, the start of the rebuild window of that data dir (rule: cohort_rule in the frontmatter, "
                 "common.py COHORT_WINDOWS, _meta/README.md).")
    assert "comparable" not in s and "comparable" not in C.T_LEGACY_GENERIC
    assert C.summary_parts(fm_of(cohort="legacy", config={}), STATS)["caveats"] == [C.T_LEGACY_GENERIC]
    n20 = C.summary_parts(fm_of(cohort="legacy", started="2026-07-01T00:00:00Z", config={"data_paths": ["data/sigma_k/3"]}), STATS)["caveats"][0]
    assert "`data/sigma_k/3`" in n20 and "`2026-07-20T13:29:00Z`" in n20


def test_config_values_are_printed_as_the_frontmatter_prints_them():
    cfg = {"arch_preset": "trm", "ema": True, "log_z_dynamics": "True", "k": 4}
    c = C.summary_parts(fm_of(config=cfg), STATS)["config"]
    assert "ema `true`" in c and "log_z_dynamics `True`" in c            # YAML bool -> true; the string "True" stays "True"
    assert C.to_yaml({"a": "True", "b": True}).replace(" ", "") .count('a:"True"') == 1 and "b:true" in C.to_yaml({"b": True}).replace(" ", "")


@needs_cards
def test_corpus_end_state_flag_templates_and_counts():
    ended_nonzero = ended_api = union = flagged = 0
    for p in FILES:
        t, fm, summ, table = parse_card(p)
        api = (fm.get("link") or {}).get("api_state")
        want = C.end_state_fields_differ(fm["status"], fm.get("exit_code"), api)
        assert fm["end_state_fields_differ"] is want, p.name
        assert (C.T_END_STATE in summ) is want, p.name                                           # template present iff flag
        assert (C.T_NO_SUMMARY in summ) is (fm["status"] == "no-summary"), p.name
        assert (C.T_NO_META in summ) is (fm["has_metadata"] is False), p.name
        assert (C.T_LEGACY_GENERIC in summ or "Cohort legacy per the cohort rule: data path" in summ) is (fm["cohort"] == "legacy"), p.name
        assert (C.T_ATTR_NONE in summ) is (not (fm["attribution"]["exps"] or fm["attribution"]["hyps"])), p.name
        assert "comparable" not in summ
        cfg_line = [ln for ln in summ.splitlines() if ln.startswith("Run ")][0]
        assert f"local status `{fm['status']}`" in cfg_line
        assert (("local exit_code" in cfg_line) is (fm.get("exit_code") is not None)), p.name
        assert (("wandb API api_state" in cfg_line) is (api is not None)), p.name
        if fm["status"] == "ended":
            en = isinstance(fm.get("exit_code"), int) and fm["exit_code"] != 0
            ended_nonzero += en
            ended_api += api in ("crashed", "failed", "killed")
            union += en or api in ("crashed", "failed", "killed")
        flagged += want
    print("ended+nonzero exit", ended_nonzero, "ended+api crashed/failed/killed", ended_api, "union among ended", union, "flag total", flagged)
    assert flagged >= union > 0
