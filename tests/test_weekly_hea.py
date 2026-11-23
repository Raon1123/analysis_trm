"""Tests for the WP2 weekly HEA extractor (NAS module 15_research-dashboard/weekly_hea.py).

Fixtures are a small synthetic lab tree (tests/fixtures/weekly_hea/), copied to tmp_path so mtimes can
be pinned: everything is put outside the window except the files a test wants inside it.
Window = 2026-01-05 (Mon) .. 2026-01-11 (Sun), KST.
"""
from __future__ import annotations

import builtins
import datetime as dt
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pytest

MOD_PATH = Path("/mnt/ayp/agents/trm/93_monitoring/11_lab-monitoring/15_research-dashboard/weekly_hea.py")
FIX = Path(__file__).parent / "fixtures" / "weekly_hea"
KST = dt.timezone(dt.timedelta(hours=9))
START, END = dt.date(2026, 1, 5), dt.date(2026, 1, 11)

if not MOD_PATH.exists():  # pragma: no cover
    pytest.skip("weekly_hea.py (NAS) not present", allow_module_level=True)


def _load():
    spec = importlib.util.spec_from_file_location("weekly_hea_under_test", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wh = _load()
wh._queue_ts = wh._mtime          # fixtures set landing times with os.utime; production uses ctime (see below)


def _ts(y, m, d, h=12):
    return dt.datetime(y, m, d, h, tzinfo=KST).timestamp()


@pytest.fixture()
def tree(tmp_path):
    lab = tmp_path / "lab"
    q = tmp_path / "queue"
    shutil.copytree(FIX / "lab", lab)
    shutil.copytree(FIX / "queue", q)
    old = _ts(2025, 12, 1)
    for root, _dirs, files in os.walk(tmp_path):
        for n in files:
            os.utime(os.path.join(root, n), (old, old))
    inside = _ts(2026, 1, 7)
    for rel in ("done/0001_e902_a_k5_s1.job", "done/0002_fixA_k3_s1.job", "done/0003_mystery_k1_s1.job",
                "failed/0004_fixA_k4_s2.job", "done/0005_pp_ref_k3_s1.job"):
        os.utime(q / rel, (inside, inside))
    # 0006 stays outside the window
    return lab, q


@pytest.fixture()
def result(tree):
    lab, q = tree
    return wh.build_cards(START, END, lab_root=lab, queue_root=q)


def _card(res, eid):
    hits = [c for c in res["cards"] if c["exp_id"] == eid]
    assert hits, "no card for %s; have %s" % (eid, [c["exp_id"] for c in res["cards"]])
    return hits[0]


def _all_outputs(res):
    return json.dumps(res, ensure_ascii=False) + wh.render_table(res) + wh.material_block(res)


def _excerpts(obj):
    if isinstance(obj, dict):
        if set(obj) >= {"text", "source"} and isinstance(obj.get("source"), dict):
            yield obj
        for v in obj.values():
            yield from _excerpts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _excerpts(v)


# ------------------------------------------------------------------ shapes
def test_old_generation_doc(result):
    c = _card(result, "EXP-901")
    assert [b["text"] for b in c["expected"]["branches"]] == ["S1", "S2", r"S3 \| alt"]
    quotes = [q["text"] for q in c["actual"]["quotes"]]
    assert quotes == ["- [x] 시나리오 판정: **S2** (fixture)"]          # unchecked + job-number lines excluded
    assert any("job 번호" in u for u in c["unextractable"])
    assert c["status_verbatim"]["text"].startswith("done")
    assert c["hypotheses"][0]["id"] == "H-901"
    assert c["hypotheses"][0]["statement"]["text"].startswith("Old-generation fixture hypothesis")


def test_new_generation_with_prereg_elsewhere(result):
    c = _card(result, "EXP-902")
    assert c["expected"]["source"]["file"] == "23_reports/11_lab-reports/2026-01-02_exp902-prereg.md"
    assert [b["text"] for b in c["expected"]["branches"]] == ["R-A", "R-B", "R-C"]   # null table skipped
    assert [q["text"] for q in c["actual"]["quotes"]] == ["C-FIX-1\tR-A", "C-FIX-2\tR-B"]
    assert c["actual"]["verdict_verbatim"]["text"] == "PASS"                      # r10 beats r2 (numeric)
    assert c["actual"]["verdict_verbatim"]["source"]["file"].endswith("EXP-902_RESULTS_VERIFY-r10.return.yaml")
    assert c["status_results"]["text"] == "VERIFIED-r10-PASS"
    assert c["hypotheses"][0]["id"] == "H-902"
    # witness 2026-01-04T16:00Z == 2026-01-05 01:00 KST -> inside; the 2025-12-20 entry is not
    wit = [w["text"] for w in c["activity"]["witness"]]
    assert wit == ["## W1 — prereg frozen (2026-01-04T16:00:00Z)"]
    assert {"i", "ii"} <= {x["by"] for x in c["included_by"]}


def test_doc_without_headings_is_unextractable(result):
    c = _card(result, "EXP-903")
    assert c["expected"] is None
    assert any(u.startswith("예상:") for u in c["unextractable"])
    assert any(u.startswith("실제:") for u in c["unextractable"])
    assert c["mention_only"] is True
    assert "추출 불가(" in wh.render_table(result)


def test_malformed_frontmatter_and_alias(result):
    c = _card(result, "EXP-950")                      # EXP-904 files, canonical EXP-950 via ID_ALIASES
    assert c["file_prefix"] == "EXP-904" and c["aliases"] == ["EXP-904"]
    assert any("frontmatter" in u for u in c["unextractable"])


def test_garbage_doc_does_not_raise(result):
    c = _card(result, "EXP-906")
    assert c["unextractable"]
    assert not result["stats"]["errors"]


def test_missing_lab_root_does_not_raise(tmp_path):
    res = wh.build_cards(START, END, lab_root=tmp_path / "nope", queue_root=tmp_path / "noq")
    assert res["cards"] == [] and res["unattributed_runs"] == []


# ------------------------------------------------------------------ attribution
def test_run_attribution(result):
    c902 = _card(result, "EXP-902")
    r = {n["name"]: n for n in c902["runs"]["names"]}
    assert r["e902_a_k5_s1"]["attributed_by"] == "sidecar" and r["e902_a_k5_s1"]["draft_map"] is True
    assert r["e902_a_k5_s1"]["wandb_ids"] == ["abcd1234"]
    c901 = _card(result, "EXP-901")
    assert {n["name"]: n["attributed_by"] for n in c901["runs"]["names"]} == \
        {"fixA_k3_s1": "schema", "fixA_k4_s2": "schema"}
    assert (c901["runs"]["done"], c901["runs"]["failed"]) == (1, 1)
    # `pp_ref_*` is only a reference in the schema cell -> not owned; mystery matches nothing
    assert sorted(result["unattributed_runs"]) == ["mystery_k1_s1", "pp_ref_k3_s1"]
    assert result["stats"]["landings"] == 5                   # 0006 is outside the window
    assert "귀속 불가 run 2건" in wh.render_table(result)
    assert "초안 map" in wh.render_table(result)


def test_schema_compiler_fullmatch():
    pats = wh.compile_schema_cell("`fig1_tf_<z>_<it>_k<k>_s<seed>` External comparators `pp_x_k*`")
    assert len(pats) == 1
    assert pats[0].fullmatch("fig1_tf_noz_iter_k5_s1")
    assert not pats[0].fullmatch("fig1x2_tf_noz_iter_k5_s1")
    p2 = wh.compile_schema_cell("`fig1d_{tf,mlp}_{noz,z}_iter_k{9,11}_s{1,2,3}`")[0]
    assert p2.fullmatch("fig1d_mlp_z_iter_k11_s3") and not p2.fullmatch("fig1d_tf_z_iter_k10_s1")


# ------------------------------------------------------------------ invariants
def test_sealed_files_never_opened(tree, monkeypatch):
    lab, q = tree
    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    res = wh.build_cards(START, END, lab_root=lab, queue_root=q)
    assert opened, "spy saw no reads"
    assert not [p for p in opened if "sealed" in os.path.basename(p).lower()]
    c906 = _card(res, "EXP-906")
    assert not [x for x in c906["included_by"] if "sealed" in x.get("file", "").lower()]


def test_queue_number_never_in_output(result):
    out = _all_outputs(result)
    for n in ("0001", "0002", "0003", "0004", "0005", "0006", "0123"):
        assert not re.search(r"(?<!\d)%s(?!\d)" % n, out), n


def test_table_cells_are_markdown_safe(result):
    table = wh.render_table(result)
    rows = [ln for ln in table.split("\n") if ln.startswith("|")]
    assert len(rows) == 2 + len(result["cards"])
    for ln in rows:
        cells = re.split(r"(?<!\\)\|", ln)[1:-1]
        assert len(cells) == 6, ln
        assert "\t" not in ln
    assert "…" in table                                   # H-902 statement truncated at 160 chars


def test_every_excerpt_matches_its_source_line(result, tree):
    lab, _ = tree
    seen = 0
    for ex in _excerpts(result):
        lines = (lab / ex["source"]["file"]).read_text(encoding="utf-8", errors="replace").split("\n")
        assert ex["text"] in lines[ex["source"]["line"] - 1], ex
        seen += 1
    assert seen >= 15


def test_render_is_deterministic_and_material_has_paths(result):
    assert wh.render_table(result) == wh.render_table(json.loads(json.dumps(result)))
    mb = wh.material_block(result)
    assert "verify verdict (21_experiments/11_lab-experiments/EXP-902_RESULTS_VERIFY-r10.return.yaml:3): PASS" in mb
    assert "언급만" in wh.render_table(result)


def test_cli_writes_week_json(tree, tmp_path, capsys):
    lab, q = tree
    out = tmp_path / "out"
    rc = wh.main(["--week", "2026-01-08", "--out", str(out), "--lab-root", str(lab), "--queue-root", str(q),
                  "--no-withheld"])
    assert rc == 0
    data = json.loads((out / "2026-W02.json").read_text(encoding="utf-8"))
    assert data["window"] == {"start": "2026-01-05", "end": "2026-01-11", "label": "2026-W02", "tz": "KST"}
    assert "| 가설 | 실험 |" in capsys.readouterr().out


# ---- verifier r1 FAIL fixes (2026-11-10): job-number scrub, audits-dir verify lookup, relaxed verdict line ----------
def _wh():
    import importlib.util
    from pathlib import Path as _P
    p = _P("/mnt/ayp/agents/trm/93_monitoring/11_lab-monitoring/15_research-dashboard/weekly_hea.py")
    spec = importlib.util.spec_from_file_location("weekly_hea_r1fix", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_job_numbers_scrubbed_including_plural_and_ranges():
    m = _wh()
    for raw in ["jobs 0696–0700 landed", "45 jobs 0524-0568", "job #0214", "Jobs 0701–0706, 0214–0219 done",
                "0696_fig1x2_tf_noz_iter_k7_s1.job"]:
        out = m._scrub_jobs(raw)
        assert not m.JOBNUM_RE.search(out), (raw, out)
        assert not __import__("re").search(r"\d{4}", out), (raw, out)
    tree = m._scrub_tree({"text": "jobs 0696–0700", "file": "21_experiments/0696_x.md",
                          "cards": [{"status_registry": {"text": "45 jobs 0524–0568"}}]})
    assert tree["file"] == "21_experiments/0696_x.md"                    # paths untouched
    assert "0696" not in tree["text"] and "0524" not in tree["cards"][0]["status_registry"]["text"]
    assert m.JOB_MARK in tree["text"]


def test_verify_lookup_falls_back_to_audits_dir(tmp_path):
    m = _wh()
    lab = tmp_path
    exp = lab / m.EXP_REL
    aud = lab / m.AUDITS_REL / "2026-10-31_exp020_results_verify"
    exp.mkdir(parents=True)
    aud.mkdir(parents=True)
    (aud / "VERIFY-r1.return.md").write_text("verdict: FAIL\n", encoding="utf-8")
    (aud / "VERIFY-r2.return.md").write_text("x\nverdict: PASS\n", encoding="utf-8")
    (lab / m.AUDITS_REL / "2026-10-22_exp020_prereg_verify").mkdir()      # prereg verify must not be picked
    (lab / m.AUDITS_REL / "2026-10-22_exp020_prereg_verify" / "VERIFY-r9.return.md").write_text("verdict: FAIL\n")
    (lab / m.AUDITS_REL / "2026-10-31_exp0201_results_verify").mkdir()    # a different EXP number
    rd = m._Reader(lab)
    p = m._latest_verify(rd, exp, "EXP-020", lab)
    assert p == aud / "VERIFY-r2.return.md"
    v = m._verdict_line(rd, p)
    assert v["text"] == "PASS" and v["source"]["line"] == 2
    (exp / "EXP-020_RESULTS_VERIFY-r1.return.yaml").write_text("verdict: PASS-with-minors\n", encoding="utf-8")
    assert m._latest_verify(rd, exp, "EXP-020", lab) == exp / "EXP-020_RESULTS_VERIFY-r1.return.yaml"   # local wins
    assert m._latest_verify(rd, exp, "EXP-999", lab) is None


def test_checked_verdict_line_with_parenthetical():
    m = _wh()
    assert m.CHECKED_VERDICT_RE.match("- [x] P-A 판정 (tf block, s1): S-4 (DRAFT)")
    assert m.CHECKED_VERDICT_RE.match("- [x] 시나리오 판정: S3")
    assert not m.CHECKED_VERDICT_RE.match("- [ ] 시나리오 판정: S3")


def test_hypothesis_cell_is_ids_only_with_legend():
    m = _wh()
    card = {"exp_id": "EXP-001", "hypotheses": [
        {"id": "H-022", "statement": {"text": "long statement " * 30, "source": {"file": "h.md", "line": 3}}}]}
    assert m._hyp_cell(card) == "H-022"
    leg = m._hyp_legend([card, card])
    assert leg.count("H-022") == 1 and "h.md:3" in leg and "…" in leg


def test_bare_job_ranges_scrubbed_but_dates_and_numbers_kept():
    m = _wh()
    out = m._scrub_jobs("enqueued (0579–0581) and (0585-0623); job 2026-11-03 note; lr 0.001-0.01; lines 0123:0456")
    assert "0579" not in out and "0585" not in out and "0623" not in out
    assert "2026-11-03" in out and "0.001-0.01" in out and "0123:0456" in out
    assert m._scrub_jobs("window 2026-11-02..2026-11-08, k 5-7, step 1000-2000") == \
        "window 2026-11-02..2026-11-08, k 5-7, step 1000-2000"


def test_queue_landing_time_is_ctime_not_mtime(tmp_path):
    m = _wh()
    f = tmp_path / "0001_run_a.job"
    f.write_text("x")
    os.utime(f, (1_600_000_000, 1_600_000_000))          # mtime far in the past = "enqueued long ago"
    g = tmp_path / "moved.job"
    os.rename(f, g)                                       # the queue's move into done/
    assert abs(m._queue_ts(g) - __import__("time").time()) < 120
    assert m._mtime(g) == 1_600_000_000
    assert m._queue_ts(tmp_path / "missing") is None
