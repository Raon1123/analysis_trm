"""Stage-A run-card pipeline: tiering, cohort, sanitising, idempotence, legacy exclusion.

Synthetic fixtures only (tmp_path); no wandb import, no network, no real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests._wri_import import wri  # noqa: E402

# Not bare imports: dataset/ has its own module named `common`, and the two clash in one pytest process.
cards = wri("cards")
common = wri("common")
links = wri("links")

OK = {"ok": True, "n": 2, "error": None}


def _run(rid="aaa111", project="power_permutation", mode="online", name="pp_x_k3_s1", synced=False):
    return {"id": rid, "project": project, "mode": mode, "name": name, "synced_marker": synced}


# ----------------------------------------------------------------------------- tiers
def test_tier_api_confirmed_uses_api_url_and_flags_name_mismatch():
    api = {"aaa111": {"project": "power_permutation", "name": "renamed", "state": "finished",
                      "url": "https://wandb.ai/raon1123/power_permutation/runs/aaa111"}}
    t = common.assign_tier(_run(), api, {}, {"power_permutation": OK})
    assert t["tier"] == "api-confirmed" and t["url_status"] == "confirmed"
    assert t["name_mismatch"] is True and not t["errors"]


def test_tier_queue_log_when_not_in_api():
    q = {"aaa111": [("raon1123", "power_permutation", "https://wandb.ai/raon1123/power_permutation/runs/aaa111",
                     "logs/queue/0001_x.log")]}
    t = common.assign_tier(_run(), {}, q, {"power_permutation": OK})
    assert t["tier"] == "queue-log" and "absent from a complete API listing" in t["finding"]


def test_tier_constructed_and_listing_unavailable_is_not_confirmed_absent():
    t = common.assign_tier(_run(), {}, {}, {"power_permutation": {"ok": False, "n": 0, "error": "x"}})
    assert t["tier"] == "constructed-unverified" and t["url_status"] == "constructed-guess"
    assert not t.get("listing_complete_id_absent")
    t2 = common.assign_tier(_run(), {}, {}, {"power_permutation": OK})
    assert t2["listing_complete_id_absent"] is True


def test_tier_no_remote_emits_no_url_and_synced_offline_is_not_no_remote():
    t = common.assign_tier(_run(mode="offline"), {}, {}, {})
    assert t["tier"] == "no-remote" and t["url"] is None
    t2 = common.assign_tier(_run(mode="offline", synced=True), {}, {}, {"power_permutation": OK})
    assert t2["tier"] == "constructed-unverified"


def test_tier_inconsistencies_recorded():
    api = {"aaa111": {"project": "Sigma_k", "name": "pp_x_k3_s1", "state": "finished", "url": "u"}}
    assert common.assign_tier(_run(), api, {}, {"power_permutation": OK})["errors"]
    q = {"aaa111": [("raon1123", "Sigma_k", "https://wandb.ai/raon1123/Sigma_k/runs/aaa111", "logs/queue/a.log")]}
    assert common.assign_tier(_run(), {}, q, {"power_permutation": OK})["errors"]


def _links_root(tmp_path: Path, api_project: str) -> Path:
    (tmp_path / "lab/21_experiments/11_lab-experiments").mkdir(parents=True)
    (tmp_path / common.META_REL).mkdir(parents=True)
    (tmp_path / "logs/queue").mkdir(parents=True)
    (tmp_path / "logs/queue/0001_x.log").write_text("wandb: View run at nowhere\n")
    row = {"id": "aaa111", "project": "power_permutation", "wandb_project": "power_permutation",
           "name": "pp_x_k3_s1", "mode": "online", "synced_marker": False}
    (tmp_path / links.INDEX).write_text(json.dumps(row) + "\n")
    listing = {"power_permutation": [{"id": "zzz999", "project": "power_permutation", "name": "o", "url": "u"}],
               api_project: [{"id": "aaa111", "project": api_project, "name": "pp_x_k3_s1", "url": "u"}]}
    status = {p: {"ok": True, "n": 1, "error": None} for p in listing}
    (tmp_path / common.META_REL / "api_listing.json").write_text(json.dumps({"status": status, "listing": listing}))
    return tmp_path


def test_links_exit_code_nonzero_on_inconsistency(tmp_path, monkeypatch):
    root = _links_root(tmp_path, "Sigma_k")  # API lists the id under another project
    monkeypatch.setattr(sys, "argv", ["links.py", "--root", str(root)])
    assert links.main() == 3


def test_links_exit_code_zero_when_consistent(tmp_path, monkeypatch):
    root = _links_root(tmp_path, "power_permutation")
    monkeypatch.setattr(sys, "argv", ["links.py", "--root", str(root)])
    assert links.main() == 0


# ----------------------------------------------------------------------------- cohort
@pytest.mark.parametrize("dp,started,want", [
    (["data/sigma_k_10/5"], "2026-07-19T10:00:00Z", "legacy"),
    (["data/sigma_k_10/5"], "2026-07-20T13:10:10Z", "legacy"),
    (["data/sigma_k_10/5"], "2026-07-20T13:18:20Z", "unknown"),   # inside rebuild window
    (["data/sigma_k_10/5"], "2026-07-21T00:00:00Z", "clean"),
    (["data/sigma_k_10/11"], "2026-07-20T20:00:00Z", "legacy"),   # k=11 rebuilt only on 07-21
    (["data/sigma_k_10/11"], "2026-07-22T00:00:00Z", "clean"),
    (["data/sigma_k_10/13"], "2026-08-01T00:00:00Z", "unknown"),
    (["data/sigma_k/3"], "2026-05-01T00:00:00Z", "legacy"),
    (["data/sigma_k/3"], "2026-08-01T00:00:00Z", "clean"),
    (["data/power_permutation/d1/3"], "2026-10-01T00:00:00Z", "clean"),
    (["data/power_permutation/d1/3"], "2026-08-01T00:00:00Z", "unknown"),
    (["data/arc1concept-aug-1000"], "2026-02-01T00:00:00Z", "other"),
    (["data/sudoku-extreme-1k-aug-1000"], None, "other"),
    (["data/sigma_k_10/5"], None, "unknown"),
    ([], "2026-10-01T00:00:00Z", "unknown"),
    (["data/elsewhere/3"], "2026-10-01T00:00:00Z", "unknown"),
    (["data/sigma_k_10/5", "data/sigma_k/5"], "2026-07-20T13:20:00Z", "unknown"),  # disagree
])
def test_cohort_rule(dp, started, want):
    assert common.cohort(dp, started)[0] == want


def test_cohort_string_paths_and_rewritten_dir_guard():
    assert common.cohort("[data/sigma_k_10/3]", "2026-09-01T00:00:00Z")[0] == "clean"
    km = {"data/sigma_k_10/3": common.parse_started("2026-10-01T00:00:00Z")}
    assert common.cohort(["data/sigma_k_10/3"], "2026-09-01T00:00:00Z", km)[0] == "unknown"


# ----------------------------------------------------------------------------- names
def test_sanitize_and_no_project():
    assert common.card_relpath("?", "a/b c?", "x1") == "_no-project/a_b_c__x1.md"
    assert common.card_relpath("", None, "x2") == "_no-project/_unnamed__x2.md"
    assert common.card_relpath("power_permutation", "../../etc", "x3") == "power_permutation/etc__x3.md"
    assert common.collisions(["p/A__1.md", "p/a__1.md", "p/b__2.md"]) == [["p/A__1.md", "p/a__1.md"]]


# ----------------------------------------------------------------------------- cards
def _cards_root(tmp_path: Path) -> Path:
    meta = tmp_path / common.META_REL
    meta.mkdir(parents=True)
    (tmp_path / "lab/21_experiments/11_lab-experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "lab/24_audits/11_lab-audits").mkdir(parents=True)
    (tmp_path / cards.MAP).write_text("families: []\n")
    (tmp_path / "lab/21_experiments/11_lab-experiments/EXP-099_demo.md").write_text("# demo\n")
    rows, scan, lk = [], [], []
    for rid, name, coh, seed in (("c1", "pp_a_k3_s1", "clean", 1), ("c2", "pp_a_k3_s2", "clean", 2),
                                 ("l1", "fig1_tf_k3_s1", "legacy", 1)):
        rows.append({"id": rid, "name": name, "project": "power_permutation", "wandb_project": "power_permutation",
                     "cohort": coh, "cohort_reason": "t", "status": "ended", "mode": "online", "has_metadata": True,
                     "seed": seed, "sibling_key_sha": "S" if rid != "l1" else "L", "exps": ["EXP-099"], "hyps": [],
                     "confidence": ["doc"], "wandb_sha256": "h" + rid, "started_at": "2026-10-01T00:00:00Z"})
        scan.append({"id": rid, "wandb_sha256": "h" + rid, "stats": {"status": "ok", "keys": {
            "probe/test_exact": {"n": 3, "nonfinite": 0, "first": 0.1, "first_step": 1, "last": 0.9, "last_step": 3,
                                 "min": 0.1, "min_step": 1, "max": 0.95, "max_step": 2}}}})
        lk.append({"id": rid, "tier": "constructed-unverified", "url_status": "constructed-guess",
                   "url": f"https://wandb.ai/raon1123/power_permutation/runs/{rid}", "evidence": ["x"]})
    w = lambda p, xs: p.write_text("".join(json.dumps(x) + "\n" for x in xs))
    w(tmp_path / cards.INDEX, rows)
    w(meta / "scan_cache.jsonl", scan)
    w(meta / "links.jsonl", lk)
    (meta / "scan_snapshot.json").write_text(json.dumps({"snapshot_utc": "2026-11-10T00:00:00+00:00"}))
    return tmp_path


def _cards_main(monkeypatch, root, *extra):
    monkeypatch.setattr(sys, "argv", ["cards.py", "--root", str(root), "--apply", *extra])
    return cards.main()


def test_cards_idempotent_and_prose_preserved(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    assert _cards_main(monkeypatch, root) == 0
    d = root / common.CARDS_REL / "power_permutation"
    files = sorted(d.glob("*.md"))
    assert len(files) == 3
    first = {p.name: p.read_bytes() for p in files}
    c1 = d / "pp_a_k3_s1__c1.md"
    txt = c1.read_text()
    assert "url_constructed_unverified" in txt and '"url":' not in txt  # never shown as confirmed
    assert "| `probe/test_exact` | 3 |" in txt and "512-example" not in txt  # not verified outside Sigma_k_fig12
    assert "pp_a_k3_s2__c2.md" in txt  # sibling link
    assert "EXP-099_demo.md" in txt  # related doc
    assert _cards_main(monkeypatch, root) == 0
    assert {p.name: p.read_bytes() for p in sorted(d.glob("*.md"))} == first
    c1.write_text(txt.replace(common.PROSE_START + "\n", common.PROSE_START + "\nhand prose\n"))
    assert _cards_main(monkeypatch, root) == 0
    assert "hand prose" in c1.read_text()


def test_cards_exclude_legacy(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    assert _cards_main(monkeypatch, root, "--exclude-cohort", "legacy") == 0
    names = {p.name for p in (root / common.CARDS_REL / "power_permutation").glob("*.md")}
    assert names == {"pp_a_k3_s1__c1.md", "pp_a_k3_s2__c2.md"}
    assert [r["cohort"] for r in common.filter_cohort([{"cohort": "legacy"}, {"cohort": "clean"}], ["legacy"])] == ["clean"]


def test_cards_stats_unavailable_message():
    fm = {"id": "x", "name": "n", "project": "p", "link": {"tier": "no-remote", "url_status": "none"}}
    out = common.render_card(fm, {"status": "scan-error: boom", "keys": {}}, [], None, [])
    assert "Stats unavailable: `scan-error: boom`" in out


def test_build_inject_moves_hand_text_out_of_markers(tmp_path):
    build = wri("build")
    p = tmp_path / "EXP-999_x.md"
    p.write_text(f"# body\n\n{build.START}\n## §S. hand pointer\n\ntext\n\n## wandb run 목록 — old\nOLD\n{build.END}\ntail\n")
    blk = f"{build.START}\n## wandb run 목록 — new\nNEW\n{build.END}\n"
    out = build.inject(p, blk, None)
    assert "## §S. hand pointer" in out and out.index("§S") < out.index(build.START)
    assert "OLD" not in out and "NEW" in out and out.endswith("tail\n")
    p.write_text(out)
    assert build.inject(p, blk, None) == out  # fixed point


# ----------------------------------------------------------------------------- Stage-A patch r1
def _rewrite(root, mut_row=None, mut_scan=None):
    meta = root / common.META_REL
    rows = [json.loads(x) for x in open(root / cards.INDEX)]
    scan = [json.loads(x) for x in open(meta / "scan_cache.jsonl")]
    for r in rows:
        mut_row and mut_row(r)
    for r in scan:
        mut_scan and mut_scan(r)
    w = lambda p, xs: p.write_text("".join(json.dumps(x) + "\n" for x in xs))
    w(root / cards.INDEX, rows)
    w(meta / "scan_cache.jsonl", scan)


def _card(root, project, name, rid):
    return (root / common.CARDS_REL / project / f"{name}__{rid}.md").read_text()


def test_history_truncation_disclosed(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    _rewrite(root, lambda r: r.update(last_step=10, last_step_source="wandb-summary.json:_step"),
             lambda s: s.update(last_history_step=3))
    assert _cards_main(monkeypatch, root) == 0
    txt = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert 'history_truncated: true' in txt and 'summary_last_step: 10' in txt
    assert 'history_last_step: 3' in txt and 'stats_table_last_step: 3' in txt
    i_note, i_tab = txt.index("Note: the `last` values in this table come from the local `.wandb` history"), txt.index("| key | n |")
    assert i_note < i_tab and txt[i_note:i_tab].count("\n") == 2  # one line directly above the table (+blank)


def test_no_truncation_flag_when_summary_matches(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    _rewrite(root, lambda r: r.update(last_step=3, last_step_source="wandb-summary.json:_step"))
    assert _cards_main(monkeypatch, root) == 0
    txt = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert 'history_truncated: false' in txt and "Note: the `last` values" not in txt


def test_running_snapshot_marked_and_dry_run_splits_changes(tmp_path, monkeypatch, capsys):
    root = _cards_root(tmp_path)
    _rewrite(root, lambda r: r.update(status="running?") if r["id"] == "c1" else None)
    ck = root / "checkpoints/power_permutation/pp_a_k3_s1"
    (ck / "step_1").mkdir(parents=True)
    assert _cards_main(monkeypatch, root) == 0
    txt = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert 'snapshot: true' in txt and 'snapshot_utc: "2026-11-10T00:00:00+00:00"' in txt
    assert 'snapshot: false' in _card(root, "power_permutation", "pp_a_k3_s2", "c2")
    (ck / "step_2").mkdir()  # live run grows
    monkeypatch.setattr(sys, "argv", ["cards.py", "--root", str(root), "--dry-run"])
    capsys.readouterr()
    assert cards.main() == 0
    out = capsys.readouterr().out
    assert "changed 1 (running 1, unexpected 0)" in out
    # same change on an ENDED run is reported as unexpected, not hidden
    (root / "checkpoints/power_permutation/pp_a_k3_s2/step_1").mkdir(parents=True)
    assert cards.main() == 0
    assert "unexpected 1" in capsys.readouterr().out


def test_attribution_pairing_and_multi_attribution(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    (root / cards.MAP).write_text(
        "families:\n"
        "- {project: power_permutation, name_regex: '^pp_a_k3_s1$', exp_ids: [EXP-015], hypothesis_ids: [H-199], confidence: registry}\n"
        "- {project: power_permutation, name_regex: '^pp_a_k3_s1$', exp_ids: [EXP-016], hypothesis_ids: [], confidence: doc}\n"
        "- {project: power_permutation, name_regex: '^pp_a_k3_s2$', exp_ids: [EXP-015], hypothesis_ids: [], confidence: registry}\n")
    build = wri("build")
    monkeypatch.setattr(build, "PREFIX_RULES", [("power_permutation", r"^pp_a_k3_s1$", ["EXP-023"], ["H-199"], "n")])
    assert _cards_main(monkeypatch, root) == 0
    t1 = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert 'multi_attribution: true' in t1 and '"confidence"' not in t1.split("exp_attribution")[0]
    for pair in ('id: "EXP-015"\n      source: "map"\n      confidence: "registry"',
                 'id: "EXP-016"\n      source: "map"\n      confidence: "doc"',
                 'id: "EXP-023"\n      source: "prefix-rule"\n      confidence: "prefix-rule"'):
        assert pair in t1
    assert "assigns this run to several EXPs (EXP-015, EXP-016, EXP-023)" in t1
    assert "EXP-015/EXP-016 match(es) come from the map `name_regex`" in t1 and "EXP-023 comes from the generator prefix rule" in t1
    t2 = _card(root, "power_permutation", "pp_a_k3_s2", "c2")
    assert 'multi_attribution: false' in t2 and "several EXPs" not in t2


def test_verified_metric_definitions_three_tiers():
    run, assumed, mlp = "fig1_tf_noz_iter_k2_s1", "fig1_tf_noz_noiter_k5_s2", "fig1_mlp_noz_iter_k2_s1"
    for k in ("probe/test_exact", "all/exact_accuracy"):
        d = common.metric_def(k, "Sigma_k_fig12", run)
        assert "verified on this run only (tf noz_iter k2_s1" in d and common.metric_def_is_verified(k, "Sigma_k_fig12", run)
        d = common.metric_def(k, "Sigma_k_fig12", assumed)
        assert "assumed extension of a verified definition \u2014 not independently checked" in d
        assert not common.metric_def_is_verified(k, "Sigma_k_fig12", assumed)
        assert common.metric_def_tier(k, "Sigma_k_fig12", "fig1_tf_z_iter_k3_s1") == "assumed"
        for proj, nm in (("Sigma_k_fig12", mlp), ("Sigma_k_fig12", "c0A_tf_noz_iter_k5_from_k6s1"), ("Sigma_k_fig12", "fig1d_tf_noz_iter_k3_s1"),
                         ("Sigma_k_fig12", "fig1x2_tf_z_iter_k2_s1"), ("Sigma_k_fig12", None), ("power_permutation", run), ("Sigma_k_new", assumed), (None, run)):
            d = common.metric_def(k, proj, nm)
            assert "working gloss from the key name / code reference" in d and "512" not in d and "online" not in d
            assert common.metric_def_tier(k, proj, nm) == "gloss" and not common.metric_def_is_verified(k, proj, nm)
    assert "512-example" in common.metric_def("probe/test_exact", "Sigma_k_fig12", run)
    assert "512-example" in common.metric_def("probe/test_exact", "Sigma_k_fig12", assumed)   # assumed extension keeps the definition, labelled
    assert common.metric_def_is_verified("num_params", "power_permutation")
    assert all(l in (common.VERIFIED_SCOPE_SUFFIX + common.ASSUMED_SUFFIX + common.UNVERIFIED_SUFFIX) for l in common.LABELS)


def test_history_empty_vs_no_allowlisted_key(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    def empty(s): s["stats"].update(keys={}, history_records=0)
    def nokey(s): s["stats"].update(keys={}, history_records=5)
    _rewrite(root, lambda r: None, empty)
    assert _cards_main(monkeypatch, root) == 0
    t = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert "history_empty: true" in t and "The local .wandb file contains no history records" in t
    assert "No allow-listed" not in t
    _rewrite(root, lambda r: None, nokey)
    assert _cards_main(monkeypatch, root) == 0
    t = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert "history_empty" not in t and "History records exist, but no allow-listed metric key" in t
    assert "contains no history records" not in t


def test_running_snapshot_states_input_and_timestamp(tmp_path, monkeypatch):
    root = _cards_root(tmp_path)
    _rewrite(root, lambda r: r.update(status="running?") if r["id"] == "c1" else None)
    assert _cards_main(monkeypatch, root) == 0
    t = _card(root, "power_permutation", "pp_a_k3_s1", "c1")
    assert "table numbers come from the scan cache" in t and "2026-11-10T00:00:00+00:00" in t
    assert "index (`_wandb_runs_index.jsonl`)" in t and "Values will move until the run ends" in t
    assert "Snapshot: this run" not in _card(root, "power_permutation", "pp_a_k3_s2", "c2")
