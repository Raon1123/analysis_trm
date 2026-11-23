"""Golden / invariant tests against the live lab (read-only). Run:
    cd /home/ayp/project/trm && CUDA_VISIBLE_DEVICES="" uv run pytest tools/hypothesis_trace -q
"""
import json
import os
import re
import subprocess
import sys

import pytest

import build
import frontmatter as fmod
from paths import DEFAULT_PATHS_FILE, Paths
from withheld import Withheld

HERE = os.path.dirname(os.path.abspath(__file__))
P = Paths(DEFAULT_PATHS_FILE)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("trace")
    fmod.OPENED.clear()
    build.run(str(out), None, DEFAULT_PATHS_FILE)
    opened = list(fmod.OPENED)
    traces = {}
    for fn in os.listdir(out):
        if fn.endswith(".trace.json"):
            with open(out / fn, encoding="utf-8") as fh:
                traces[json.load(fh)["h_id"]] = json.load(open(out / fn, encoding="utf-8"))
    return {"out": out, "traces": traces, "opened": opened, "index": json.load(open(out / "index.json", encoding="utf-8"))}


def abs_of(display):
    if display.startswith("lab/"):
        return os.path.join(P.lab_root, display[4:])
    return os.path.join(P.repo_root, display)


def line_at(source):
    with open(abs_of(source["file"]), encoding="utf-8") as fh:
        return fh.read().split("\n")[source["line"] - 1]


def edges(t, frm=None, to=None, rel=None):
    return [e for e in t["edges"] if (frm is None or e["from"] == frm) and (to is None or e["to"] == to) and (rel is None or e["rel"] == rel)]


def test_every_edge_has_source_that_exists(built):
    for h, t in built["traces"].items():
        for e in t["edges"]:
            s = e["source"]
            assert s["file"] and isinstance(s["line"], int) and s["line"] >= 1, (h, e)
            assert os.path.exists(abs_of(s["file"])), (h, e)
            if s["basis"] != "filename-convention":
                # the declared line must exist in the file
                n = len(open(abs_of(s["file"]), encoding="utf-8").read().split("\n"))
                assert s["line"] <= n, (h, e)


def test_no_edge_to_missing_file(built):
    for t in built["traces"].values():
        for n in t["nodes"]:
            if n["path"] and n["kind"] != "RUN":
                if "[열람 보류]" in n["path"]:  # NB4: withheld name redacted at generation; list-only blind EXP only
                    assert n.get("list_only") and n.get("blind"), n
                    continue
                assert os.path.exists(abs_of(n["path"])), n


def test_golden_h199(built):
    t = built["traces"]["H-199"]
    assert t["status"]["word"] == "falsified-as-framed"
    assert t["status"]["authority"] == "sidecar"
    assert t["status"]["source"]["file"].endswith("H-199.status.md") and t["status"]["source"]["line"] == 6
    assert line_at(t["status"]["source"]).startswith("status: falsified-as-framed")
    # card exp_id line
    e = edges(t, "H-199", "EXP-015", "pre-registered-in")
    assert e and e[0]["source"]["file"].endswith("H-199.md") and line_at(e[0]["source"]).startswith("exp_id:")
    # FREEZE, VERIFY r1..r3 for RESULTS, RETROFIT
    assert edges(t, "EXP-015", "EXP-015:FREEZE", "frozen-by")
    for r in (1, 2, 3):
        vn = f"EXP-015:VERIFY-results-r{r}"
        assert edges(t, "EXP-015:RESULTS", vn, "verified-by"), vn
    pre = {n["id"] for n in t["nodes"] if n["kind"] == "VERIFY" and n.get("target") == "prereg"}
    assert {"EXP-015:VERIFY-prereg-r1", "EXP-015:VERIFY-prereg-r2"} <= pre
    # ACJ
    acj = edges(t, "H-199", "ACJ:acj_H-199_20261003", "adjudicated-by")
    assert acj and any("H_id: H-199" in line_at(x["source"]) for x in acj)
    assert edges(t, "H-199.status", "ACJ:acj_H-199_20261003", "adjudicated-by")
    # scout
    sc = edges(t, "H-199", rel="scouted-in")
    assert sc and line_at(sc[0]["source"]).startswith("literature_link:")
    assert [n for n in t["nodes"] if n["id"] == sc[0]["to"]][0]["kind"] == "SCOUT"
    # sidecar basis -> RESULTS
    assert edges(t, "H-199.status", "EXP-015:RESULTS", "basis")
    # claim cells from the sidecar table, verbatim
    ids = {c["id"] for c in t["claims"] if c["kind"] == "card-claim"}
    assert ids == {"C-LTF-1", "C-LTF-2", "C-LTF-3"}
    for c in t["claims"]:
        assert c["status_cell"] in line_at(c["source"])
    # EXP-100 is mentioned but has no doc -> gap, not a file node
    assert any(g["code"] == "EXP_DOC_MISSING" and g["where"] == "EXP-100" for g in t["gaps"])
    # verifier token copied from the VERIFY file
    assert [n for n in t["nodes"] if n["id"] == "EXP-015:VERIFY-results-r3"][0]["verifier_token"] == "PASS"


def test_golden_h023_legacy(built):
    t = built["traces"]["H-023"]
    rel = {e["to"] for e in edges(t, "H-023", rel="related-to")}
    assert {"H-022", "H-014"} <= rel
    for e in edges(t, "H-023", rel="related-to"):
        assert line_at(e["source"]).startswith("related_hypotheses:")
    dead = [g for g in t["gaps"] if g["code"] == "DEAD_PATH" and "lab/audits/2026-06-23_inversion_acj.md" in g["what"]]
    assert dead and dead[0]["where"].endswith("HYPOTHESES.md:%d" % t["registry_row"]["line"])
    assert not any("lab/audits/2026-06-23_inversion_acj.md" in (n["path"] or "") for n in t["nodes"])
    # legacy latest_acj session has no directory -> gap, not edge
    assert any(g["code"] == "ACJ_DIR_MISSING" for g in t["gaps"])
    # vault-relative pending_evidence artifacts are gaps with their own code
    assert any(g["code"] == "UNRESOLVED_ROOT" and "20_Investigations" in g["what"] for g in t["gaps"])
    # canon-doc cell edge for the path that does exist
    assert edges(t, "H-023", rel="canon-doc")


def test_golden_h108_no_registry_row(built):
    t = built["traces"]["H-108"]
    assert t["registry_row"] is None and t["registry_note"] == "레지스트리 행 없음"
    assert any(g["code"] == "REGISTRY_ROW_MISSING" for g in t["gaps"])
    for fg in ("FG-0001", "FG-0002"):
        e = edges(t, fg, "H-108", "about-hypothesis")
        assert e and line_at(e[0]["source"]).startswith("hypothesis_ids:"), fg
        assert any(f["atom"] == fg for f in t["figures"])
    assert sum(1 for g in t["gaps"] if g["code"] == "FG_NO_RENDER") == 2
    idx = [x for x in built["index"]["hypotheses"] if x["h_id"] == "H-108"][0]
    assert idx["registry_row"] is False


def test_hypothesis_set_covers_registry_and_atoms(built):
    reg = {ln.split("|")[1].strip().strip("*") for ln in open(P.abs("registry.hypotheses"), encoding="utf-8") if re.match(r"\| \*\*", ln)}
    assert {h for h in reg if h not in ("H-NNN",)} <= set(built["traces"])
    assert "H-108" in built["traces"]


def test_status_cells_byte_equal(built):
    reg_lines = open(P.abs("registry.hypotheses"), encoding="utf-8").read().split("\n")
    n_checked = 0
    for h, t in built["traces"].items():
        rr = t["registry_row"]
        if rr:
            line = reg_lines[rr["line"] - 1]
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            assert cells[2] == rr["status_cell"], h  # byte for byte
            assert rr["status_cell"] in line
            n_checked += 1
        sc = t["status"] and t["status"].get("sidecar")
        if sc:
            assert sc["cell"] in line_at(sc["source"])
            # a bare controlled word is copied as the word; a qualified cell keeps word null and raises the gap
            assert sc["word"] == sc["cell"] or (sc["word"] is None and any(
                g["code"] == "STATUS_WORD_UNCONTROLLED" and g["what"].startswith("sidecar") for g in t["gaps"])), h
    assert n_checked >= 16
    t = built["traces"]["H-022"]  # registry cell that is not a controlled word stays verbatim, word null
    assert t["status"]["registry"]["word"] is None
    assert t["registry_row"]["status_cell"].startswith("**[split")
    # sidecar claim cells equal the table cell on the source line
    side = open(os.path.join(P.abs("theory.cards"), "H-199.status.md"), encoding="utf-8").read().split("\n")
    for c in built["traces"]["H-199"]["claims"]:
        cells = [x.strip() for x in side[c["source"]["line"] - 1].strip().split("|")[1:-1]]
        assert cells[1] == c["status_cell"]


def _walk(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, path + (i,))
    else:
        yield path, o


def _doc_names():
    wd = build.World(DEFAULT_PATHS_FILE)
    return wd, set(wd.W.doc_names)


_WD, DOC_NAMES = _doc_names()

# Matching-logic tests run against this fixed registry (the 2026-10-04 family set, all four run families
# withheld), so they keep testing the guard after the live registry releases a family (2026-10-20: h108n and
# pp-seedext released, user decisions A2/A3). Live-output scans below use the live registry.
SYN_REGISTRY = r"""
version: 1
families:
  - id: exp013-seed-ge2
    status: withheld
    exp_ids: [EXP-013]
    patterns:
      - '(?<![A-Za-z0-9])fig1_(?:mlp|tf)_(?:z|noz)_(?:iter|noiter)_k\d+_s(?:[2-9]|\d{2,})(?![0-9])'
      - '(?<![A-Za-z0-9])(?<!pp_)abl_[A-Za-z0-9_]+?_k\d+_s(?:[2-9]|\d{2,})(?![0-9])'
  - id: h108n
    status: withheld
    exp_ids: []
    patterns: ['(?<![A-Za-z0-9])H[-_ ]?108(?![0-9])\w*']
  - id: pp-seedext
    status: withheld
    exp_ids: []
    patterns: ['(?<![A-Za-z0-9])(?:pp[-_ ]?)?seed[-_ ]?ext\w*']
  - id: exp012-cleanr1
    status: withheld
    exp_ids: [EXP-012]
    patterns: ['(?<![A-Za-z0-9])clean[-_ ]?r1(?![0-9])\w*']
  - id: exp015-pp-ltf
    status: released
    exp_ids: [EXP-015]
    patterns: ['(?<![A-Za-z0-9])pp_ltf_\w*']
  - id: verdict-prose
    status: withheld
    exp_ids: []
    patterns: ['falsified|refuted|C-LTF|verdict']
"""


@pytest.fixture(scope="module")
def syn_w(tmp_path_factory):
    f = tmp_path_factory.mktemp("syn") / "WITHHELD_RUNS.yaml"
    f.write_text(SYN_REGISTRY, encoding="utf-8")
    w = Withheld(str(f))
    w.set_doc_names(DOC_NAMES)
    return w


SYNTHETIC_BLOCKED = [
    "lab/21_experiments/x/EXP-012_cleanr1_d0_k10_sm2.csv",
    "lab/x/EXP-013_fig1_tf_z_iter_k10_s2.csv",
    "lab/23_reports/x/EXP-013_results_pp_seedext_k3.csv",
    "checkpoints/power_permutation/H-108N5/step_1",
    "lab/x/H-108N5.csv",
    "lab/x/H-seedext.png",
    "lab/x/H-cleanr1.csv",
]


def test_match_path_identity_exemption(syn_w):
    for ref in SYNTHETIC_BLOCKED:
        assert syn_w.match_path(ref), ref
    W = _WD.W
    # real document names still pass
    doc12 = os.path.basename(_WD.expfiles["EXP-012"]["doc"])
    assert doc12 in DOC_NAMES and W.match_path(f"lab/21_experiments/{doc12}") is None
    assert W.match_path("lab/21_experiments/" + doc12 + "/cleanr1_d0_k10_sm2/step_1")
    # a fresh matcher with no document list exempts nothing by shape
    assert Withheld(P.abs("governance.withheld_runs")).match_path(f"lab/{doc12}")


def test_withheld_runs_not_emitted(built, syn_w):
    # synthetic ids from every withheld family of the fixed registry are detected
    samples = ["fig1_tf_z_iter_k5_s2", "H108N3_x", "pp_seedext_a", "cleanr1_d0_k5_sm2", "abl_a_k3_s4"]
    assert all(syn_w.match(s) for s in samples)
    assert syn_w.match("fig1_tf_z_iter_k5_s1") is None
    assert syn_w.match("pp_ltf_tf_k5_s1") is None  # released family never redacts
    assert syn_w.match("falsified-as-framed") is None  # prose family is not a run-id family
    W = Withheld(P.abs("governance.withheld_runs"))  # live registry for the output scan
    assert W.is_blind_exp("EXP-012") and W.is_blind_exp("EXP-013") and not W.is_blind_exp("EXP-015")
    for h, t in built["traces"].items():
        for n in t["nodes"]:
            if n["kind"] == "RUN":
                assert W.match(n["run_id"]) is None and (n["run_name"] is None or W.match(n["run_name"]) is None), n
        # whole-output scan of every string leaf except file-path/identifier-of-document fields
        # (a document filename such as EXP-012_clean-r1-... is not a run id); excludes family h108n,
        # which also matches the hypothesis id "H-108" and is only meaningful on run ids.
        pathy = {"file", "where", "what", "dir", "node", "annotation", "claim_id"}
        for p_, leaf in _walk(t):
            if not isinstance(leaf, str) or (p_ and p_[-1] in pathy):
                continue
            # exact allow-list of real document basenames (identity, not shape); everything else is scanned
            leaf_s = build_norm(leaf)
            for dn in sorted(DOC_NAMES, key=len, reverse=True):
                leaf_s = leaf_s.replace(dn, "")
            for fid, regs in W.families:
                if fid == "h108n":
                    continue
                for r in regs:
                    m = r.search(leaf_s)
                    assert m is None, (h, fid, p_, m.group(0))
    # a withheld run id in a synthetic atom/CSV is replaced by a count
    tr = build.Trace(build.World(DEFAULT_PATHS_FILE), "H-199")
    assert tr.add_run("H-199", "power_permutation", "zzz", "cleanr1_d0_k5_sm2", "evidence", {"file": "x", "line": 1}) is None
    assert tr.withheld_count == 1 and not tr.nodes.get("RUN:power_permutation/zzz")


def build_norm(s):
    from withheld import normalise
    return normalise(s)


def test_locked_and_withheld_exps_are_list_only(built):
    bad = [p for p in built["opened"] if re.search(r"EXP-01[2368]_", os.path.basename(p)) or "EXP-016_" in p or "EXP-018_" in p]
    assert not bad, bad
    for h in ("H-200", "H-202"):
        t = built["traces"][h]
        locked = [n for n in t["nodes"] if re.match(r"EXP-01[68]", n["id"])]
        assert locked
        assert any(g["code"] == "LOCKED_LIST_ONLY" for g in t["gaps"])
        for n in locked:
            assert "status_cell" not in n
            if n["kind"] != "EXP":
                assert n["sha256"] and n["blind"] in (False, True)
    # no data file under a withheld/locked EXP was opened
    assert not [p for p in built["opened"] if p.endswith((".csv", ".jsonl")) and "EXP-01" in p and "EXP-015" not in p]


NUM = re.compile(r"\b\d+\.\d+\b")


def test_no_numeric_values_in_results_and_run_nodes(built):
    skip = {"id", "path", "sha256", "mtime", "prereg_sha256", "judge_sha256", "freeze_manifest_sha256", "per_run_csv_sha256", "generated_at"}
    n_checked = 0
    for h, t in built["traces"].items():
        for n in t["nodes"]:
            if n["kind"] in ("RESULTS", "RUN", "VERIFY", "FREEZE"):
                for k, v in n.items():
                    if k in skip:
                        continue
                    for _, leaf in _walk({k: v}):
                        if isinstance(leaf, float):
                            pytest.fail(f"float in {n['id']}.{k}")
                        if isinstance(leaf, str) and k != "status_cell":
                            assert not NUM.search(leaf), (h, n["id"], k, leaf)
                n_checked += 1
        for e in t["edges"]:
            assert set(e) == {"from", "to", "rel", "source"}
    assert n_checked > 50
    # RUN nodes carry only identifiers
    runs = [n for t in built["traces"].values() for n in t["nodes"] if n["kind"] == "RUN"]
    assert runs and all(set(n) <= {"id", "kind", "path", "sha256", "mtime", "blind", "project", "run_id", "run_name", "run_verified"} for n in runs)


def test_index_and_outputs(built):
    out = built["out"]
    assert os.path.exists(out / "README.md") and os.path.exists(out / "index.json")
    idx = built["index"]
    assert idx["totals"]["hypotheses"] == len(built["traces"])
    readme = open(out / "README.md", encoding="utf-8").read()
    for code in build.GAP_CODES:
        assert code in readme
    for t in built["traces"].values():
        assert t["schema"] == "trm.hypothesis_trace/0.1"
        for g in t["gaps"]:
            assert g["code"] in build.GAP_CODES


def test_parse_failure_exits_nonzero(tmp_path):
    lab = tmp_path / "lab"
    paths = open(DEFAULT_PATHS_FILE, encoding="utf-8").read()
    import yaml
    d = yaml.safe_load(paths)
    d["lab_root"] = str(lab)
    for sub in (d["paths"]["registry.dir"], d["paths"]["atoms.claims"], d["paths"]["atoms.figures"], d["paths"]["theory.cards"],
                d["paths"]["experiments.dir"], d["paths"]["audits.dir"], d["paths"]["governance.root"]):
        (lab / sub).mkdir(parents=True, exist_ok=True)
    real = P.lab_root
    for name in ("registry.hypotheses", "registry.experiments", "governance.withheld_runs"):
        (lab / d["paths"][name]).write_text(open(os.path.join(real, P.map[name]), encoding="utf-8").read(), encoding="utf-8")
    bad = lab / d["paths"]["theory.cards"] / "H-999.md"
    bad.write_text("---\nid: H-999\nstatus: [unclosed\n---\n", encoding="utf-8")
    pf = tmp_path / "PATHS.yaml"
    pf.write_text(yaml.safe_dump(d), encoding="utf-8")
    r = subprocess.run([sys.executable, os.path.join(HERE, "build.py"), "--out", str(tmp_path / "o"), "--paths-file", str(pf)],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert "H-999.md" in r.stderr and "PARSE FAILURE" in r.stderr


def test_r3r8_r1_fixes(built):
    T = built["traces"]
    for h in ("H-200", "H-201", "H-202"):  # dict-form literature_link -> SCOUT edge, no dict dump
        assert [e for e in T[h]["edges"] if e["rel"] == "scouted-in"], h
        assert not any("{" in g["what"] for g in T[h]["gaps"]), h
    for h, frag in (("H-202", "north-star-claims-figures-plan"), ("H-022", "b3ii-lean-formalization")):
        assert not any(g["code"] == "DEAD_PATH" and frag in g["what"] for g in T[h]["gaps"]), h
        assert any(frag in e["to"] for e in T[h]["edges"]), h
    assert not any("cleanr1" in json.dumps(t).lower() for t in T.values())
    assert T["H-108"]["withheld_ref_count"] >= 1  # mixes runs and documents; after the 2026-10-20 h108n release the remaining hits are exp013-pattern run names (fig1_*_s2/s3 = H108N2/N7)
    assert not any("withheld_run_count" in t for t in T.values())
    assert Withheld(P.abs("governance.withheld_runs")).match_path("checkpoints/Sigma_k_new/cleanr1_d0_k10_sm2/step_1")
    rels = {(h, e["to"]): e["rel"] for h in ("H-022", "H-023", "H-025") for e in T[h]["edges"] if e["source"]["file"].endswith("EXPERIMENTS.md")}
    assert rels[("H-023", "EXP-006")] == "feeds" and rels[("H-023", "EXP-012")] == "context"


# ----------------------------------------------------------------------------- R6 S7 backfill
import hashlib  # noqa: E402

import backfill_s7  # noqa: E402

EVENTS_PATH = P.abs("monitoring.events")


def _sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


EVENTS_SHA_AT_IMPORT = _sha(EVENTS_PATH)


@pytest.fixture(scope="module")
def backfill(tmp_path_factory):
    d = tmp_path_factory.mktemp("bf")
    sc = backfill_s7.Scanner(DEFAULT_PATHS_FILE)
    rows = sc.run()
    a, b = str(d / "a.jsonl"), str(d / "b.jsonl")
    backfill_s7.write_rows(rows, a)
    sc2 = backfill_s7.Scanner(DEFAULT_PATHS_FILE)
    backfill_s7.write_rows(sc2.run(), b)
    return {"rows": rows, "a": a, "b": b, "sc": sc}


def test_backfill_rows_have_source_line_and_verbatim_note(backfill):
    assert backfill["rows"], "no backfill rows"
    for r in backfill["rows"]:
        assert r["stage"] == ("S7" if r["event"] else None) and r["code"] in {"STATUS", "RESULTS", "ACJ", "ATOM", "HYPSTATUS", "REGLOG", "EXPSTATUS"}
        assert r["ts_basis"] in ("stated", "mtime")
        s = r["source"]
        path = abs_of(s["file"])
        assert os.path.isfile(path), r
        lines = open(path, encoding="utf-8").read().split("\n")
        assert 1 <= s["line"] <= len(lines) and 1 <= s["ts_line"] <= len(lines), r
        assert r["note"] and r["note"] in lines[s["line"] - 1], r  # status word quoted verbatim, never normalised
        if r["ts_basis"] == "stated":
            assert r["ts"] in lines[s["ts_line"] - 1], r
        else:
            assert r["ts"] == s["mtime"] and r["ts_precision"] == "mtime", r
        assert r["path"] == (P.lab_rel(path) or r["path"])


def test_backfill_has_no_withheld_or_locked(backfill):
    W = Withheld(P.abs("governance.withheld_runs"))
    W.set_doc_names(backfill["sc"].W.doc_names)
    for r in backfill["rows"]:
        blob = json.dumps(r, ensure_ascii=False)
        assert not W.match_path(r["note"]) and not W.match_path(r["path"]) and not W.match_path(r["source"]["file"]), r
        assert not (set(r["exp_ids"]) & (backfill_s7.LOCKED_EXPS | W.exp_ids)), r
        assert "EXP-016_RESULTS" not in blob


def test_backfill_locked_exps_in_sync():
    assert backfill_s7.LOCKED_EXPS == build.LOCKED_EXPS


def test_backfill_deterministic(backfill):
    assert open(backfill["a"], "rb").read() == open(backfill["b"], "rb").read()
    keys = [(r["ts"], r["source"]["file"], r["source"]["line"], r["code"]) for r in backfill["rows"]]
    assert keys == sorted(keys)


def test_events_jsonl_unchanged_and_backfill_goes_elsewhere(backfill, tmp_path):
    assert backfill_s7.main(["--out", str(tmp_path / "x.jsonl")]) == 0
    build.run(str(tmp_path / "t"), "H-199", DEFAULT_PATHS_FILE)
    assert _sha(EVENTS_PATH) == EVENTS_SHA_AT_IMPORT
    assert P.abs("monitoring.backfill_events") != EVENTS_PATH


def test_timeline_origin_labels(built):
    seen = set()
    for t in built["traces"].values():
        for x in t["timeline"]:
            assert x["origin"] in ("events", "backfill", "mtime"), x
            assert x["ts_basis"] in ("event", "stated", "mtime"), x
            seen.add(x["origin"])
            if x["origin"] == "backfill":
                assert x["ts_basis"] in ("stated", "mtime") and x["backfill_source"]["line"] >= 1
    if os.path.exists(P.abs("monitoring.backfill_events")):
        assert "backfill" in seen


def test_h199_s7_event_gap_cleared_by_backfill(built):
    if not os.path.exists(P.abs("monitoring.backfill_events")):
        pytest.skip("backfill_events.jsonl not generated")
    t = built["traces"]["H-199"]
    assert any(x["stage"] == "S7" and x["origin"] == "backfill" for x in t["timeline"])
    assert not [g for g in t["gaps"] if g["code"] == "NO_S7_EVENT"]


def test_atom_ts_kind_review_vs_created(backfill):
    atoms = {r["source"]["file"].rsplit("/", 1)[-1]: r for r in backfill["rows"] if r["code"] == "ATOM"}
    fg3 = next(r for n, r in atoms.items() if n.startswith("FG-0003"))
    assert fg3["ts"] == "2026-10-20" and fg3["ts_kind"] == "review"
    assert "review:" in open(abs_of(fg3["source"]["file"]), encoding="utf-8").read().split("\n")[fg3["source"]["ts_line"] - 1]
    for r in atoms.values():
        assert r["ts_kind"] in ("created", "review")


def test_acj_covers_h023_donor_basin_verdict(backfill):
    rows = [r for r in backfill["rows"] if r["code"] == "ACJ" and "h023-donor-basin-acj" in r["source"]["file"]]
    assert rows and rows[0]["h_ids"] == ["H-023"] and rows[0]["ts"] == "2026-07-19" and rows[0]["event"]


def test_event_classification_and_last_ts(backfill, built):
    by = {}
    for r in backfill["rows"]:
        if r["code"] == "REGLOG" and ("UNRUN" in r["note"] or "enqueued" in r["note"]):
            assert not r["event"] and r["stage"] is None and r["ts_kind"] == "mention", r
        if r["code"] == "ATOM" and r["ts_kind"] == "created":
            assert not r["event"] and r["stage"] is None and r["event_kind"] == "created", r
        if r["code"] == "HYPSTATUS" and r["note"].startswith("**under-test**"):
            assert not r["event"], r
        if r["code"] == "EXPSTATUS" and r["note"].startswith("**in-progress"):
            assert not r["event"], r
        assert r["event"] == (r["stage"] == "S7")
    # index last_ts never comes from a non-event backfill row
    non_event_ts = {}
    for r in backfill["rows"]:
        if not r["event"]:
            for h in r["h_ids"]:
                non_event_ts.setdefault(h, set()).add(r["ts"])
    for row in built["index"]["hypotheses"]:
        t = built["traces"][row["h_id"]]
        ev_ts = {x["ts"] for x in t["timeline"] if x.get("event", True)}
        assert row["last_ts"] is None or row["last_ts"] in ev_ts, row["h_id"]


def test_nested_paper_claim_verdicts_covered(backfill):
    ks = [r for r in backfill["rows"] if r["code"] == "ACJ" and "/2026-09-30_paper-claim-agora/K" in r["source"]["file"]]
    assert len(ks) == 6, len(ks)
    assert all(r["h_ids"] == [] and r["event_kind"] == "paper-claim-acj" and r["event"] for r in ks)


def test_index_last_ts_verdict_only(backfill, built):
    verdict_ts = {}
    for r in backfill["rows"]:
        if r["event_kind"] in build.VERDICT_KINDS:
            for h in r["h_ids"]:
                verdict_ts.setdefault(h, set()).add(r["ts"])
    for row in built["index"]["hypotheses"]:
        assert "last_status_change_ts" in row
        if row["last_ts_basis"] == "stated":
            t = built["traces"][row["h_id"]]
            kinds = {x.get("event_kind") for x in t["timeline"] if x["ts"] == row["last_ts"] and x.get("event", True)}
            assert kinds & (build.VERDICT_KINDS | {None}), (row["h_id"], kinds)


def test_nb4_no_withheld_string_in_written_json(built, syn_w):
    """R7-r1 NB4: every string in every written JSON is free of withheld-pattern matches (bare H-108 id excepted)."""
    from withheld import REDACTED, redact_text
    W = Withheld(P.abs("governance.withheld_runs"))
    out = built["out"]
    n_red = 0
    for fn in os.listdir(out):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(out / fn, encoding="utf-8"))
        n_red += len(d.get("redacted_strings", []))

        def walk(o, ptr):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{ptr}/{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{ptr}/{i}")
            elif isinstance(o, str):
                assert not redact_text(W, o)[1], (fn, ptr, o)
        walk(d, "")
        for r in d.get("redacted_strings", []):
            node = d
            for part in r["pointer"].split("/")[1:]:
                node = node[int(part)] if isinstance(node, list) else node[part]
            assert REDACTED in node, (fn, r)
    assert n_red >= 1  # the NB4 cells exist in the current lab state
    assert "withheld_families_applied" not in built["index"]
    S = syn_w
    assert redact_text(S, "H-108")[1] == [] and redact_text(S, "H108N2 run")[1] == ["h108n"]
    # NB-2: text outside the token keeps its original characters (em dash survives)
    assert redact_text(S, "a — H108N2 — b")[0] == "a — [열람 보류] — b"
    # VERIFY-NB4 NB-1: the generated README carries no withheld match either
    for ln in open(out / "README.md", encoding="utf-8"):
        assert not redact_text(W, ln)[1], ln


def test_evidence_anchor(built):
    """VERIFY-NB4 NB-9/NB-4: the cited suffix survives as `anchor` for both ':range' and '#section' forms."""
    anchors = {(os.path.basename(e.get("path") or ""), e.get("anchor"))
               for t in built["traces"].values() for c in t["claims"] for e in c.get("evidence") or [] if e.get("anchor")}
    assert ("trm.py", "196-222") in anchors
    assert any(a.startswith("§") for _, a in anchors), anchors
