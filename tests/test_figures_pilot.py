"""Tests for the four pilot figure modules (RF-01..RF-04) and summaries.py. Synthetic inputs only, no NAS."""
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tools.figures import run_one
from tools.figures.summaries import check_panel, summarize_run

CELLS_HEADER = ("figure_id,panel,series,run_name,wandb_id,project,cohort,cohort_word_en,cohort_word_ko,config,"
                "config_name_en,config_name_ko,k,seed,budget,first_reach_step,sustained_start,final_value,"
                "max_value,n_evals,n_at_or_above,value_at_budget,reached_by_budget,train_final")
NAMES = [
    ("EXP-001", "timeline", "First grid", "첫 격자"), ("EXP-002", "timeline", "Second grid", "둘째 격자"),
    ("MS-01", "separator", "card written", "카드 작성"),
    ("CFG-tf_z_iter", "config", "Block, state kept", "블록, 상태 유지"),
    ("CFG-tf_noz_iter", "config", "Block, state removed", "블록, 상태 제거"),
    ("CFG-act_noz_iter_h16", "config", "Halting bundle", "멈춤 묶음"),
    ("legacy", "data", "old data", "옛 데이터"), ("clean_d0", "data", "corrected data", "고친 데이터"),
    ("d1", "data", "second-generation data", "2세대 데이터"),
]


# ------------------------------------------------------------------ summarize_run
def test_summarize_never_reaches():
    s = summarize_run([1, 2, 3], [0.1, 0.5, 0.8], budget=2)
    assert s["first_reach_step"] is None and s["sustained_start"] is None
    assert s["n_at_or_above"] == 0 and s["final_value"] == 0.8 and s["max_value"] == 0.8
    assert s["value_at_budget"] == 0.5 and s["reached_by_budget"] is False and s["n_evals"] == 3


def test_summarize_reaches_and_holds():
    s = summarize_run([1, 2, 3, 4], [0.1, 0.9, 1.0, 1.0], budget=3)
    assert s["first_reach_step"] == 2 and s["sustained_start"] == 2 and s["n_at_or_above"] == 3
    assert s["value_at_budget"] == 1.0 and s["reached_by_budget"] is True


def test_summarize_reaches_dips_recovers():
    s = summarize_run([1, 2, 3, 4, 5], [0.95, 0.2, 0.95, 1.0, 1.0], budget=1)
    assert s["first_reach_step"] == 1 and s["sustained_start"] == 3 and s["n_at_or_above"] == 4
    assert s["reached_by_budget"] is True and s["value_at_budget"] == 0.95


def test_summarize_reaches_then_ends_below():
    s = summarize_run([1, 2, 3], [1.0, 1.0, 0.4], budget=None)
    assert s["first_reach_step"] == 1 and s["sustained_start"] is None and s["final_value"] == 0.4
    assert s["value_at_budget"] is None and s["reached_by_budget"] is True


def test_protocol_guard_raises():
    with pytest.raises(ValueError, match="two different test measures"):
        check_panel(["probe/test_exact", "all/exact_accuracy"], ["d1"], True)
    with pytest.raises(ValueError, match="per-run tags"):
        check_panel(["probe/test_exact", "probe/train_exact"], ["legacy", "clean_d0"], False)
    with pytest.raises(ValueError):
        check_panel(["phase/index"], ["d1"], True)
    assert check_panel(["probe/test_exact", "probe/train_exact"], ["legacy", "clean_d0"], True)


# ------------------------------------------------------------------ fixtures
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _names(tmp):
    p = tmp / "NAMES.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "kind", "name_en", "name_ko"])
        w.writerows(NAMES)
    return p


def _run_rows(name, wid, project, family, cohort, k, seed, steps=(0, 100000, 244100, 400000, 488200)):
    out = []
    for m, vals in (("probe/test_exact", [0.0, 0.2, 0.95, 0.5, 1.0]), ("probe/train_exact", [0.1, 0.9, 1.0, 1.0, 1.0]),
                    ("all/exact_accuracy", [0.0] * 5)):
        for s, v in zip(steps, vals):
            out.append([name, wid, project, family, cohort, "data/x", "2026-01-01", "trm_singlez", k, seed, s, m, v])
    return out


COLS = "run_name,wandb_id,project,family,cohort,data_path,started_at,arch,k,seed,step,metric,value".split(",")


def _write_cache(tmp):
    cache = tmp / "cache"
    cache.mkdir()
    ext, leg, cln = [], [], []
    spec = [("noz", 2, 1), ("noz", 5, 1), ("noz", 5, 2), ("noz", 5, 3), ("noz", 7, 1), ("noz", 7, 2), ("noz", 7, 3),
            ("z", 4, 2), ("z", 5, 1)]
    for i, (c, k, s) in enumerate(spec):
        rest = f"tf_{c}_iter_k{k}_s{s}"
        ext += _run_rows(f"fig1x2_{rest}", f"x2id{i}", "P", "ext_2x", "clean_d0" if i % 2 else "legacy", k, s)
        leg += _run_rows(f"fig1_{rest}", f"l1id{i}", "P", "legacy_1x", "legacy", k, s, steps=(0, 50000, 100000, 200000, 244100))
    cln += _run_rows("fig1_tf_noz_iter_k5_s1", "c1id0", "Q", "clean_1x", "clean_d0", 5, 1, steps=(0, 50000, 100000, 200000, 244100))
    d1 = []
    for cfg in ("e020_fig1_tf_noz_iter_ep76k", "e020_act_noz_iter_h16_ep76k"):
        for k in (5, 6, 7):
            for s in (1, 2, 3):
                d1 += _run_rows(f"{cfg}_k{k}_s{s}", f"d{cfg[5]}{k}{s}", "R", "d1_ext", "d1", k, s, steps=(0, 50000, 122050, 150000, 185516))
    files = {"ext_2x.eval.csv": ext, "legacy_1x.eval.csv": leg, "clean_1x.eval.csv": cln, "d1_ext.eval.csv": d1}
    man = [["file", "sha256", "n_rows", "n_runs"]]
    for fn, rows in files.items():
        pd.DataFrame(rows, columns=COLS).to_csv(cache / fn, index=False)
        man.append([fn, _sha(cache / fn), len(rows), 0])
    with open(cache / "MANIFEST.files.csv", "w", newline="") as f:
        csv.writer(f).writerows(man)
    metrics = {m: {"eval_set": "e", "n_examples": 512, "weights": "w", "status": "unsettled", "plain_en": "a", "plain_ko": "b"}
               for m in ("probe/test_exact", "probe/train_exact")}
    (cache / "PROTOCOLS.json").write_text(json.dumps({"protocols": {"metrics": metrics}}))
    return cache, files


def _entry(tmp, fig_id, module, inputs, protocol, cohorts):
    reg = tmp / "reg"
    reg.mkdir(exist_ok=True)
    p = reg / f"{fig_id}.yaml"
    p.write_text(yaml.safe_dump({
        "id": fig_id, "module": module, "title": "t", "langs": ["en", "ko"], "print_width_in": 6.7,
        "inputs": inputs, "protocol": protocol, "cohorts": cohorts, "report_refs": []}))
    return p


def _check_outputs(out, fig_id, n_values=None):
    for lang in ("en", "ko"):
        for ext in ("png", "pdf"):
            assert (out / f"{fig_id}.{lang}.{ext}").stat().st_size > 0
    vals = pd.read_csv(out / f"{fig_id}.values.csv")
    src = json.loads((out / f"{fig_id}.source.json").read_text())
    assert src["n_values_rows"] == len(vals)
    if n_values is not None:
        assert len(vals) == n_values
    return vals, src


PROTO = {"metric_test": "probe/test_exact", "metric_train": "probe/train_exact"}


# ------------------------------------------------------------------ timeline
def _timeline(tmp, rows):
    p = tmp / "TIMELINE.json"
    p.write_text(json.dumps({"meta": {}, "rows": rows}))
    return p


def _tl_row(rid, no, **kw):
    r = {"id": rid, "kind": "experiment", "main_row_no": no, "separator": False, "separator_date": "",
         "date_prereg": "2026-06-07", "date_run": "2026-06-07..2026-06-22", "date_verdict": "2026-06-23",
         "verdict_recorded": True}
    r.update(kw)
    return r


def _tl_rows():
    return [
        _tl_row("EXP-001", 1),
        _tl_row("EXP-002", 2, date_prereg="not applicable (post-hoc)", date_run="about 2026-07-21",
                date_verdict="no verdict recorded", verdict_recorded=False),
        {"id": "MS-01", "kind": "milestone", "main_row_no": None, "separator": True, "separator_date": "2026-05-08",
         "date_prereg": "not recorded", "date_run": "2026-05-08", "date_verdict": "no verdict recorded",
         "verdict_recorded": False},
    ]


def _build(tmp, fig_id, module, inputs, protocol, cohorts, cache):
    reg = _entry(tmp, fig_id, module, inputs, protocol, cohorts)
    out = tmp / "out"
    run_one.run(reg, out, cache)
    return out


def test_timeline_builds(tmp_path):
    names, tl = _names(tmp_path), _timeline(tmp_path, _tl_rows())
    inputs = [{"role": "file", "path": str(tl)}, {"role": "file", "path": str(names)}]
    out = _build(tmp_path, "RF-01-timeline", "tools.figures.figs.timeline", inputs, {}, [], None)
    vals, _ = _check_outputs(out, "RF-01-timeline")
    # lane 1: plan, run_start, verdict; lane 2: hollow circle only (plan literal draws nothing); one separator
    assert sorted(vals["series"]) == ["plan", "run_latest", "run_start", "separator", "verdict"]
    assert vals["source_ref"].notna().all() and vals["date"].notna().all()
    assert "TIMELINE:MS-01" in set(vals["source_ref"])


@pytest.mark.parametrize("mutate,msg", [
    (lambda rows: rows[0].update(date_run="June 7"), "EXP-001 field date_run"),
    (lambda rows: rows[2].update(separator_date=""), "MS-01"),
    (lambda rows: rows[0].update(id="EXP-999"), "EXP-999"),
])
def test_timeline_errors(tmp_path, mutate, msg):
    rows = _tl_rows()
    mutate(rows)
    names, tl = _names(tmp_path), _timeline(tmp_path, rows)
    inputs = [{"role": "file", "path": str(tl)}, {"role": "file", "path": str(names)}]
    with pytest.raises(ValueError, match=msg):
        _build(tmp_path, "RF-01-timeline", "tools.figures.figs.timeline", inputs, {}, [], None)


# ------------------------------------------------------------------ glance
GL_COLS = ("row_no,change_en,change_ko,data_generation,runs_per_setting,compute_matched_en,compute_matched_ko,"
           "scoring_word,scoring_class,scoring_source,registry_token_verbatim,registry_source,seen_en,seen_ko").split(",")


def _glance(tmp, words):
    p = tmp / "GLANCE.csv"
    classes = ["observation", "mixed", "blank"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(GL_COLS)
        for i, (word, gen) in enumerate(zip(words, ["legacy", "clean_d0", "none"])):
            w.writerow([i + 1, f"Change {i}", f"변화 {i}", gen, "3", "Yes", "예", word, classes[i], "", "", "", "", ""])
    g = tmp / "GLOSS.csv"
    with open(g, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "kind", "plain_en", "plain_ko", "not_stronger_because"])
        w.writerow(["observation", "scoring", "what the data show", "자료가 보여 주는 것", "x"])
        w.writerow(["observation (legacy); association (confound: parameter count)", "scoring", "observed on old data; an association", "옛 데이터 관측; 연관", "x"])
        w.writerow(["observation", "status", "WRONG KIND", "틀림", "x"])
    return p, g


def test_glance_builds_and_missing_gloss(tmp_path):
    names = _names(tmp_path)
    gl, gs = _glance(tmp_path, ["observation", "observation (legacy); association (confound: parameter count)", ""])
    inputs = [{"role": "file", "path": str(gl)}, {"role": "file", "path": str(gs)}, {"role": "file", "path": str(names)}]
    out = _build(tmp_path, "RF-02-glance", "tools.figures.figs.glance", inputs, {}, [], None)
    vals, _ = _check_outputs(out, "RF-02-glance", n_values=15)
    sc = vals[vals["column"] == "scoring"]
    assert list(sc["text"]) == ["what the data show", "observed on old data; an association", "–"]
    assert vals["source_ref"].notna().all()
    gl2, _ = _glance(tmp_path / "x", ["nope", "observation", ""]) if (tmp_path / "x").mkdir() is None else (None, None)
    inputs[0] = {"role": "file", "path": str(gl2)}
    with pytest.raises(ValueError, match="nope"):
        run_one.run(_entry(tmp_path, "RF-02-glance", "tools.figures.figs.glance", inputs, {}, []), tmp_path / "out2", None)


# ------------------------------------------------------------------ curves
def test_curves_budget_builds(tmp_path):
    names = _names(tmp_path)
    cache, files = _write_cache(tmp_path)
    inputs = [{"role": "cache", "family": f.split(".")[0], "file": f}
              for f in ("legacy_1x.eval.csv", "clean_1x.eval.csv", "ext_2x.eval.csv")]
    inputs.append({"role": "file", "path": str(names)})
    out = _build(tmp_path, "RF-03-curves-budget", "tools.figures.figs.curves_budget", inputs, PROTO, ["legacy", "clean_d0"], cache)
    n_expected = 2 * 5 * (9 + 9 + 1)  # two probe metrics, five evaluations, nine 2x + nine legacy 1x + one clean 1x
    vals, src = _check_outputs(out, "RF-03-curves-budget", n_values=n_expected)
    assert set(vals["metric"]) == {"probe/test_exact", "probe/train_exact"}
    assert (vals[["run_name", "wandb_id", "metric", "step"]].notna().all()).all()
    assert "protocol" in src and src["protocol"]["metric_test"] == "probe/test_exact"
    first = (out / "RF-03-curves-budget.cells.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first == CELLS_HEADER
    cells = pd.read_csv(out / "RF-03-curves-budget.cells.csv")
    assert (cells["series"] == "2x").sum() == 9 and len(cells) == 19
    assert set(cells["config"]) <= {"CFG-tf_z_iter", "CFG-tf_noz_iter", "CFG-act_noz_iter_h16"}
    assert cells["config_name_en"].notna().all() and cells["config_name_ko"].notna().all()
    assert (cells["budget"] == 244100).all()


def test_curves_d1ext_builds(tmp_path):
    names = _names(tmp_path)
    cache, _ = _write_cache(tmp_path)
    inputs = [{"role": "cache", "family": "d1_ext", "file": "d1_ext.eval.csv"}, {"role": "file", "path": str(names)}]
    out = _build(tmp_path, "RF-04-curves-d1ext", "tools.figures.figs.curves_d1ext", inputs, PROTO, ["d1"], cache)
    vals, src = _check_outputs(out, "RF-04-curves-d1ext", n_values=18 * 10)
    assert set(vals["metric"]) == {"probe/test_exact", "probe/train_exact"}
    first = (out / "RF-04-curves-d1ext.cells.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first == CELLS_HEADER
    cells = pd.read_csv(out / "RF-04-curves-d1ext.cells.csv")
    assert len(cells) == 18 and set(cells["series"]) == {"ext"} and (cells["budget"] == 122050).all()
    assert set(cells["panel"]) == {f"{c}_k{k}" for c in ("CFG-tf_noz_iter", "CFG-act_noz_iter_h16") for k in (5, 6, 7)}
    assert src["protocol"]["metric_train"] == "probe/train_exact"
