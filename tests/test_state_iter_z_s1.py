"""Unit tests for tools/state_iter_z_synthesis/s1.py.  SYNTHETIC fixtures only: no real wandb, checkpoint, lab, or NAS data.

Run: CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_state_iter_z_s1.py
"""
import builtins
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "s1", Path(__file__).resolve().parents[1] / "tools" / "state_iter_z_synthesis" / "s1.py")
s1 = importlib.util.module_from_spec(_SPEC)
sys.modules["s1"] = s1
_SPEC.loader.exec_module(s1)

PREREG_TEXT = """# synthetic prereg
#### 7.0.3 H 소스 행 (9)

| s | 토큰 출처 | 연결 EXP 행 | LH | 질문 | 예상 클래스 |
|---|---|---|---|---|---|
| H-023 | HYPOTHESES.md | EXP-001, EXP-005 | 아니오 | Q-Z1 | A-VERDICT |
| H-022 | HYPOTHESES.md | EXP-001, EXP-005 | 아니오 | Q-I2 | A-VERDICT |
| H-026 | HYPOTHESES.md | EXP-005 | 아니오 | Q-Z4 | A-DESC |
| H-202 | HYPOTHESES.md | EXP-018 | 예 | Q-Z6 | X-ABSENT |
| H-199 | status | EXP-025 | 아니오 | Q-I3 | A-VERDICT |
| H-201 | status | EXP-017 | 예 | Q-Z5 | D-LOCKED |

#### 7.0.4 A 소스 행 (12)
"""

INV_FIELDS = ["source_row", "kind", "data_present", "witness_present", "result_present",
              "tier3_eligible_(*_RESULTS_w/_last_verifier_PASS)", "queue_jobs", "queue_processing", "queue_done", "queue_failed"]
TOK_FIELDS = ["source_row", "kind", "tier", "tier_label", "file", "locator", "V_tok", "N_tok", "negation_flag", "qualifier_flag", "note"]
KEY_FIELDS = ["row_id", "q", "s", "F_q", "K_q", "N_named", "key_status", "evidence_file_line_key", "absence_or_count_checks", "prereg_class"]
CELL_FIELDS = ["source", "as_of", "q", "s", "kind", "expected_class", "expected_q_token"]

# (source, kind, data, wit, result, tier3)
INV = [
    ("CL-0003", "A", "", "", "", ""), ("CL-0006", "A", "", "", "", ""),
    ("EXP-010", "E", "False", "", "", ""), ("EXP-012/sealed", "E", "False", "", "", ""),
    ("EXP-013", "E", "False", "False", "False", "False"),
    ("EXP-005", "E", "True", "False", "False", "False"), ("EXP-001", "E", "True", "False", "False", "False"),
    ("EXP-025", "E", "True", "True", "True", "True"), ("EXP-023", "E", "True", "True", "False", "False"),
    ("EXP-016", "E", "True", "True", "True", "False"), ("EXP-018", "E", "False", "True", "False", "False"),
    ("EXP-017", "E", "True", "True", "True", "True"),
    ("EXP-006", "E", "True", "False", "False", "False"), ("EXP-022", "E", "True", "True", "True", "False"),
    ("EXP-002", "E", "True", "False", "False", "False"), ("H-022", "H", "", "", "", ""),
    ("H-023", "H", "", "", "", ""), ("H-026", "H", "", "", "", ""), ("H-202", "H", "", "", "", ""),
    ("H-199", "H", "", "", "", ""), ("H-201", "H", "", "", "", ""),
]
# (q, s, kind, expected, key_status, N_named)
CELLS = [
    ("Q-I1", "CL-0003", "A", "A-VERDICT", None, None), ("Q-Z7", "CL-0006", "A", "LISTED-ONLY", None, None),
    ("Q-I2", "EXP-010", "E", "X-PROV", None, None), ("Q-I2", "EXP-012/sealed", "E", "D-SEALED", None, None),
    ("Q-I2", "EXP-013", "E", "X-ABSENT", "UNKNOWN", "∅"),
    ("Q-I2", "EXP-005", "E", "A-DESC", "NAMED-ADMITTED", "{probe, all, recomp}"),
    ("Q-I2", "EXP-001", "E", "A-DESC", "UNKNOWN", "∅"),
    ("Q-I3", "EXP-025", "E", "A-DESC", "NAMED-ADMITTED", "{probe}"),
    ("Q-I3", "EXP-023", "E", "D-PREJUDGE", "NAMED-ADMITTED", "{probe}"),
    ("Q-I4", "EXP-016", "E", "D-LOCKED", "NAMED-ADMITTED", "{probe, all}"),
    ("Q-Z6", "EXP-018", "E", "X-ABSENT", "UNKNOWN", "—"),
    ("Q-Z5", "EXP-017", "E", "D-LOCKED", "UNKNOWN", "∅"),
    ("Q-Z4", "EXP-006", "E", "A-DESC", "NAMED-ADMITTED", "{pin}"), ("Q-I3", "EXP-022", "E", "A-DESC", "NAMED-ADMITTED", "{probe}"),
    ("Q-Z3", "EXP-002", "E", "A-DESC", "NAMED-ADMITTED", "{effr}"), ("Q-I2", "H-022", "H", "A-VERDICT", None, None),
    ("Q-Z3", "H-023", "H", "A-VERDICT", None, None),
    ("Q-Z1", "H-023", "H", "A-VERDICT", None, None), ("Q-Z4", "H-026", "H", "A-DESC", None, None),
    ("Q-Z6", "H-202", "H", "X-ABSENT", None, None), ("Q-I3", "H-199", "H", "UNCOVERED", None, None),
    ("Q-Z5", "H-201", "H", "D-LOCKED", None, None),
]
# (source, tier, label, file, locator, V, N, note)
TOK = [
    ("CL-0003", "1", "atom-verdict", "S0r2_snapshot/INDEX.jsonl", "verdict field", "supported", "", ""),
    ("CL-0006", "1", "atom-verdict", "S0r2_snapshot/INDEX.jsonl", "verdict field", "partial", "", ""),
    ("EXP-005", "5", "EXPERIMENTS.md Status cell", "S0r2_snapshot/EXPERIMENTS.md", "## EXP-005 Status row", "", "DRAFT", ""),
    ("EXP-001", "5", "EXPERIMENTS.md Status cell", "S0r2_snapshot/EXPERIMENTS.md", "## EXP-001 Status row", "", "completed", ""),
    ("EXP-025", "3", "RESULTS frontmatter", "EXP-015_RESULTS.md", "status: line 25", "", "", "tier3_eligible=True"),
    ("EXP-016", "3", "RESULTS frontmatter", "EXP-016_RESULTS.md", "status: line 8", "", "", ""),
    ("H-022", "4", "HYPOTHESES.md status cell", "S0r2_snapshot/HYPOTHESES.md", "Status column (col 3)", "conditional|supported", "under-test", ""),
    ("EXP-016", "5", "EXPERIMENTS.md Status cell", "S0r2_snapshot/EXPERIMENTS.md", "## EXP-016 Status row", "", "PASS|in-progress", ""),
    ("EXP-017", "5", "EXPERIMENTS.md Status cell", "S0r2_snapshot/EXPERIMENTS.md", "## EXP-017 Status row", "", "VERIFIED-rN|completed", ""),
    ("H-023", "4", "HYPOTHESES.md status cell", "S0r2_snapshot/HYPOTHESES.md", "Status column (col 3)", "falsified-as-framed", "", ""),
    ("H-026", "4", "HYPOTHESES.md status cell", "S0r2_snapshot/HYPOTHESES.md", "Status column (col 3)", "", "under-test", ""),
    ("H-199", "4", "HYPOTHESES.md status cell", "S0r2_snapshot/HYPOTHESES.md", "Status column (col 3)", "falsified-as-framed|supported", "", ""),
    ("H-199", "2", "H-NNN.status.md frontmatter status", "S0r2_snapshot/H-199.status.md", "status: line 6", "falsified-as-framed", "", ""),
    ("H-199", "2", "LH-check (name-string count)", "S0r2_snapshot/H-199.status.md", "whole-file literal count", "", "", "L5 result names cited: 2"),
    ("H-201", "2", "H-NNN.status.md frontmatter status", "S0r2_snapshot/H-201.status.md", "status: line 6", "supported", "", ""),
]
RUNS = [  # injected run index
    dict(run_name="fig1_mlp_z_iter_k3_s1", project="Sigma_k_fig12", wandb_dir="/w/a", wandb_run_id="a1", history_path="/w/a/run-a1.wandb", started_at="2026-07-01T00:00:00Z"),
    dict(run_name="fig1_mlp_z_iter_k3_s2", project="Sigma_k_fig12", wandb_dir="/w/b", wandb_run_id="b1", history_path="/w/b/run-b1.wandb", started_at="2026-07-01T00:00:00Z"),
    dict(run_name="pp_ltf_g7_lay2_k3_s1", project="power_permutation", wandb_dir="/w/c", wandb_run_id="c1", history_path="/w/c/run-c1.wandb", started_at="2026-10-17T07:53:05.581645Z"),
    dict(run_name="pp_ltf_g7_lay2_k3_s4", project="power_permutation", wandb_dir="/w/d", wandb_run_id="d1", history_path="/w/d/run-d1.wandb", started_at="2026-10-17T07:53:05.581645Z"),
    dict(run_name="e016_act_noz_iter_h16_k3_s1", project="power_permutation", wandb_dir="/w/e", wandb_run_id="e1", history_path="/w/e/run-e1.wandb", started_at="2026-10-17T07:53:05.581645Z"),
    dict(run_name="e020_s1_k3_s1", project="power_permutation", wandb_dir="/w/f", wandb_run_id="f1", history_path="/w/f/run-f1.wandb", started_at="2026-10-17T07:53:05.581645Z"),
    dict(run_name="fig1_tf_z_iter_k3_s1", project="Sigma_k_fig12", wandb_dir="/w/g", wandb_run_id="g1", history_path="/w/g/run-g1.wandb", started_at="2026-10-17T07:53:05.581645Z"),
]


def _w(path, fields, rows, comment=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if comment:
            fh.write(comment + "\n")
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerows(rows)


def _cfg(root, proj, run, data_path, arch="trm", mlp_t=False, seed=1, extra=None):
    d = root / "checkpoints" / proj / run
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"arch": {"name": f"recursive_reasoning.{arch}@M", "H_cycles": 3, "L_cycles": 6, "mlp_t": mlp_t},
           "data_paths": [data_path], "project_name": proj, "seed": seed, "z_probe_size": 512}
    cfg.update(extra or {})
    import yaml
    (d / "all_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture()
def world(tmp_path):
    root = tmp_path / "repo"
    out = root / "out"
    prereg = root / "prereg.md"
    prereg.parent.mkdir(parents=True)
    prereg.write_text(PREREG_TEXT, encoding="utf-8")
    sha = hashlib.sha256(PREREG_TEXT.encode()).hexdigest()
    _w(out / "S0r2_cells_expected.csv", CELL_FIELDS, [["x", "2026-10-30T00:00:00Z", q, s, k, e, "Q-INHERIT"] for q, s, k, e, _, _ in CELLS],
       "# synthetic")
    _w(out / "S0r2_class_repro_check.csv", ["q", "s", "expected_class", "as_of_class", "rule", "match"],
       [[q, s, e, e, "r", "True"] for q, s, k, e, _, _ in CELLS], "# as_of: x")
    _w(out / "S0r2_metric_keys.csv", KEY_FIELDS,
       [[f"R{i}", q, s, "ACC", "K", n, ks, "e", "c", e] for i, (q, s, k, e, ks, n) in enumerate(CELLS) if ks])
    _w(out / "S0r2_inventory.csv", INV_FIELDS,
       [[s, k, d, w, r, t, "0", "0", "27" if s == "EXP-023" else "0", "0"] for s, k, d, w, r, t in INV], "# as_of: x")
    _w(out / "S0r2_tokens.csv", TOK_FIELDS, [[s, "?", t, lab, f, loc, v, n, "False", "False", note] for s, t, lab, f, loc, v, n, note in TOK],
       "# as_of: x")
    (out / "S0r2_snapshot").mkdir()
    (out / "S0r2_snapshot" / "as_of.txt").write_text("as_of: 2026-10-30T13:51:29Z\n", encoding="utf-8")
    atoms = [{"atom_id": "CL-0003", "verdict": "supported", "cohort": "clean-D0", "verdict_scope": "SCOPE-3", "prohibited_upgrade": "PROHIB-3"},
             {"atom_id": "CL-0006", "verdict": "partial", "cohort": "clean-D0", "verdict_scope": "SCOPE-6", "prohibited_upgrade": "PROHIB-6"}]
    (out / "S0r2_snapshot" / "INDEX.jsonl").write_text("\n".join(json.dumps(a) for a in atoms) + "\n", encoding="utf-8")
    (out / "S0r2_snapshot" / "H-201.status.md").write_text("---\nid: H-201\nkind: sidecar\nsupersedes: x\nupdated: d\nstatus: supported\n---\n", encoding="utf-8")
    res = root / "EXP-016_RESULTS.md"
    res.write_text("---\nid: EXP-016\nkind: results\nsupersedes: x\nupdated: d\nowner: o\nrun: r\nstatus: PASS-pending-verifier\n---\nBODY-MUST-NOT-BE-RESTATED\n", encoding="utf-8")
    (out / "S0r2b_evidence_files.sha256").write_text(f"{hashlib.sha256(res.read_bytes()).hexdigest()}  EXP-016_RESULTS.md\n", encoding="utf-8")
    _cfg(root, "Sigma_k_fig12", "fig1_mlp_z_iter_k3_s1", "data/sigma_k_10/3", arch="trm_singlez", mlp_t=True, seed=1)
    _cfg(root, "Sigma_k_fig12", "fig1_mlp_z_iter_k3_s2", "data/sigma_k_10/3", arch="trm_singlez", mlp_t=True, seed=2)
    _cfg(root, "power_permutation", "pp_ltf_g7_lay2_k3_s1", "data/power_permutation/d1/3", arch="transformers_baseline", seed=1)
    _cfg(root, "power_permutation", "e016_act_noz_iter_h16_k3_s1", "data/power_permutation/d1/3", arch="transformers_baseline", seed=1)
    _cfg(root, "Sigma_k_fig12", "fig1_tf_z_iter_k3_s1", "data/sigma_k_10/3", arch="trm", seed=1)
    ds1 = root / "data" / "power_permutation" / "d1" / "3" / "test"
    ds1.mkdir(parents=True)
    (ds1 / "dataset.json").write_text(json.dumps({"total_puzzles": 888}), encoding="utf-8")
    ds = root / "data" / "sigma_k_10" / "3" / "test"
    ds.mkdir(parents=True)
    (ds / "dataset.json").write_text(json.dumps({"total_puzzles": 777}), encoding="utf-8")
    return dict(root=root, out=out, sha=sha, prereg_rel="prereg.md")


class Scanner:
    def __init__(self):
        self.calls = []

    def __call__(self, path, keys):
        self.calls.append((path, tuple(keys)))
        base = 1.0 if "/w/b/" in path else 0.5
        rows = [{"_step": 10, **{k: base for k in keys}}, {"_step": 30, **{k: base + 0.25 for k in keys}},
                {"_step": 20, **{k: base + 0.1 for k in keys}}]
        return rows


def stub_owner(project, run_name):
    return ({"fig1_mlp": "EXP-005", "pp_ltf_": "EXP-025", "e016_": "EXP-016", "fig1_tf": "EXP-001"}.get(next((k for k in ("fig1_mlp", "pp_ltf_", "e016_", "fig1_tf") if run_name.startswith(k)), "")) or "UNKNOWN", "stub")


def run(world, scanner=None, dry=False, index=RUNS, **kw):
    buf = io.StringIO()
    sc = scanner or Scanner()
    tick = iter(range(10 ** 6))
    rc = s1.run_s1(world["root"], world["out"], dry_run=dry, prereg_rel=world["prereg_rel"], prereg_sha=world["sha"],
                   scanner=sc, run_index=index, owner_resolver=stub_owner, now_fn=lambda: f"2026-10-30T13:00:{next(tick) % 60:02d}Z", stdout=buf,
                   expect_counts=None, **{"expect_class_counts": None, **kw})
    return rc, sc, buf.getvalue()


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ------------------------------------------------------------------ gate / guard tests
def test_denied_paths():
    for bad in ("data/x/sealed/a.npy", "checkpoints/p/r/step_5_all_preds.0", "a/z_snapshots/b", "x/preds_foo/y", "d/sealed-dirs/x"):
        assert s1.is_denied(bad), bad
    for ok in ("data/power_permutation/d1/5/test/dataset.json", "checkpoints/p/r/all_config.yaml", "wandb/run-1/files/wandb-metadata.json"):
        assert not s1.is_denied(ok), ok


def test_access_log_denies_and_logs(tmp_path):
    log = s1.AccessLog()
    with pytest.raises(s1.DeniedPath):
        log.access(str(tmp_path / "sealed" / "x"), "t", "c", "a")
    assert log.rows[-1]["event"] == "denied"
    dlog = s1.AccessLog(dry_run=True)
    with pytest.raises(s1.DeniedPath):
        dlog.access(str(tmp_path / "preds" / "x"), "t", "c", "a")
    f = tmp_path / "ok.txt"
    f.write_text("hi")
    assert dlog.access(str(f), "t", "c", "a") is False and dlog.rows[-1]["event"] == "would_open"
    assert log.access(str(f), "t", "c", "a") is True
    assert log.rows[-1]["sha256"] == hashlib.sha256(b"hi").hexdigest()


def test_prereg_sha_gate(world):
    world["sha"] = "0" * 64
    rc, sc, _ = run(world)
    assert rc == 2 and not sc.calls and not list(world["out"].glob("S1_*"))


def test_refuse_overwrite(world):
    assert run(world)[0] == 0
    rc, _, text = run(world)
    assert rc == 3 and "refuse" in text


def test_count_gate(world):
    buf = io.StringIO()
    rc = s1.run_s1(world["root"], world["out"], prereg_rel=world["prereg_rel"], prereg_sha=world["sha"], scanner=Scanner(),
                   run_index=RUNS, stdout=buf)  # default expect_counts=(58,27,40) vs synthetic 17 cells
    assert rc == 2 and "GATE" in buf.getvalue()


# ------------------------------------------------------------------ classification
def test_classification_branches(world):
    assert run(world)[0] == 0
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    exp = {("Q-I1", "CL-0003"): "A-VERDICT", ("Q-Z7", "CL-0006"): "LISTED-ONLY", ("Q-I2", "EXP-010"): "X-PROV",
           ("Q-I2", "EXP-012/sealed"): "D-SEALED", ("Q-I2", "EXP-013"): "X-ABSENT", ("Q-I2", "EXP-005"): "A-DESC",
           ("Q-I2", "EXP-001"): "A-DESC", ("Q-I3", "EXP-025"): "A-DESC", ("Q-I3", "EXP-023"): "D-PREJUDGE",
           ("Q-I4", "EXP-016"): "D-LOCKED", ("Q-Z6", "EXP-018"): "X-ABSENT", ("Q-Z5", "EXP-017"): "D-LOCKED",
           ("Q-Z1", "H-023"): "A-VERDICT", ("Q-Z4", "H-026"): "A-DESC", ("Q-Z6", "H-202"): "X-ABSENT",
           ("Q-I3", "H-199"): "UNCOVERED", ("Q-Z5", "H-201"): "D-LOCKED",
           ("Q-Z4", "EXP-006"): "A-DESC", ("Q-I3", "EXP-022"): "A-DESC", ("Q-Z3", "EXP-002"): "A-DESC",
           ("Q-I2", "H-022"): "A-VERDICT", ("Q-Z3", "H-023"): "A-VERDICT"}
    assert {k: v["s1_class"] for k, v in c.items()} == exp
    assert all(r["class_match"] == "True" for r in c.values())


def test_branch_order_exp018_absent_before_prejudge_and_locked():
    facts = {"EXP-018": dict(kind="E", data=False, wit=True, has_result=False, tier3=False, claim_V=set(), V1=set(), V2=set(), N2=set(),
                             V4=set(), N4=set(), in_index=False, status_file=False, lh_violation=0, N5=set(), tokens=[], qc={})}
    assert s1.classify("Q-Z6", "EXP-018", "E", facts, {}, {})[0] == "X-ABSENT"
    facts["EXP-018"]["data"] = True
    assert s1.classify("Q-Z6", "EXP-018", "E", facts, {}, {})[0] == "D-PREJUDGE"
    facts["EXP-018"]["has_result"] = True
    assert s1.classify("Q-Z6", "EXP-018", "E", facts, {}, {})[0] == "D-LOCKED"


def test_named_other_is_xout_and_unknown_is_not():
    f = dict(kind="E", data=True, wit=False, has_result=False, tier3=False, claim_V=set(), V1=set(), V2=set(), N2=set(), V4=set(),
             N4=set(), in_index=False, status_file=False, lh_violation=0, N5=set(), tokens=[], qc={})
    assert s1.classify("Q-I2", "EXP-X", "E", {"EXP-X": f}, {("Q-I2", "EXP-X"): {"key_status": "NAMED-OTHER"}}, {})[0] == "X-OUT"
    assert s1.classify("Q-I2", "EXP-X", "E", {"EXP-X": f}, {("Q-I2", "EXP-X"): {"key_status": "UNKNOWN"}}, {})[0] == "A-DESC"


def test_unknown_mismatch_vs_prereg_or_s0_confirmed_aborts_exit2(world):
    # EXP-005 loses its data in the inventory: S1 derives X-ABSENT, S0r2 repro says A-DESC and PREREG expects A-DESC -> exit 2, nothing written
    inv = world["out"] / "S0r2_inventory.csv"
    inv.write_text(inv.read_text(encoding="utf-8").replace("EXP-005,E,True", "EXP-005,E,False"), encoding="utf-8")
    rc, sc, out = run(world)
    assert rc == 2 and "GATE" in out and "S1 re-derived X-ABSENT != S0 confirmed A-DESC" in out
    assert sc.calls == [] and not list(world["out"].glob("S1_*"))
    # S0 and S1 agree on X-ABSENT but PREREG expects A-DESC and it is not a known mismatch -> still exit 2
    rp = world["out"] / "S0r2_class_repro_check.csv"
    rp.write_text(rp.read_text(encoding="utf-8").replace("Q-I2,EXP-005,A-DESC,A-DESC", "Q-I2,EXP-005,A-DESC,X-ABSENT"), encoding="utf-8")
    rc, _, out = run(world)
    assert rc == 2 and "not a known mismatch" in out


def _with_exp011(world, data="False"):
    """append the two EXP-011 cells (PREREG expects A-DESC) with S0r2-like facts: data absent -> X-ABSENT"""
    o = world["out"]
    (o / "S0r2_inventory.csv").write_text((o / "S0r2_inventory.csv").read_text(encoding="utf-8") + f"EXP-011,E,{data},False,False,False,0,0,0,10\n", encoding="utf-8")
    cl = "X-ABSENT" if data == "False" else "A-DESC"
    for q in ("Q-I2", "Q-I4"):
        (o / "S0r2_cells_expected.csv").write_text((o / "S0r2_cells_expected.csv").read_text(encoding="utf-8") + f"x,t,{q},EXP-011,E,A-DESC,Q-INHERIT\n", encoding="utf-8")
        (o / "S0r2_class_repro_check.csv").write_text((o / "S0r2_class_repro_check.csv").read_text(encoding="utf-8") + f"{q},EXP-011,A-DESC,{cl},r,{cl == 'A-DESC'}\n", encoding="utf-8")
        (o / "S0r2_metric_keys.csv").write_text((o / "S0r2_metric_keys.csv").read_text(encoding="utf-8") + f"R9{q[-1]},{q},EXP-011,ACC,K,{{probe}},NAMED-ADMITTED,e,c,A-DESC\n", encoding="utf-8")


def test_known_exp011_mismatch_is_allowed_and_recorded(world):
    _with_exp011(world)
    rc, sc, out = run(world)
    assert rc == 0
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    for q in ("Q-I2", "Q-I4"):
        r = c[(q, "EXP-011")]
        assert (r["s1_class"], r["s0_confirmed_class"], r["expected_class"], r["class_match"]) == ("X-ABSENT", "X-ABSENT", "A-DESC", "False")
        assert "§7.0.5 예상 A-DESC ≠ S0r2 X-ABSENT" in r["reason"] and r["n_table_rows"] == "0" and r["values_policy"].startswith("none: class X-ABSENT")
    assert sum(1 for r in c.values() if r["class_match"] == "False") == 2  # exactly the two known cells
    assert "DISCREPANCY" in out


def test_known_mismatch_only_valid_for_that_exact_pair(world):
    _with_exp011(world, data="True")  # S1 now derives A-DESC for EXP-011: the 'known' X-ABSENT expectation is violated
    rc, _, out = run(world)
    assert rc == 2 and "known mismatch (Q-I2,EXP-011) expected to be X-ABSENT" in out


def test_expected_class_counts_gate(world):
    rc, _, out = run(world, expect_class_counts={"A-DESC": 99})
    assert rc == 2 and "class counts" in out


def test_default_prefix_is_s0r2_and_old_s0_not_read(world):
    import shutil
    assert s1.S0_PREFIX_DEFAULT == "S0r2" and s1.s0_names("S0r2")["cells"] == "S0r2_cells_expected.csv"
    # old-generation files present but poisoned: default run must not touch them
    for n in ("S0_cells_expected.csv", "S0_metric_keys.csv", "S0_inventory.csv", "S0_tokens.csv"):
        (world["out"] / n).write_text("garbage\n")
    (world["out"] / "S0_snapshot").mkdir()
    rc, _, _ = run(world)
    assert rc == 0
    al = rd(world["out"] / "S1_access_log.csv")
    assert not any(Path(a["path"]).name.startswith("S0_") or "/S0_snapshot" in a["path"] for a in al)


def test_old_prefix_selectable_only_via_argument(world, monkeypatch, tmp_path, capsys):
    # fixture has only S0r2 files: asking for the old prefix must fail (exit 2), never silently fall back to S0r2
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    rc = s1.run_s1(world["root"], world["out"], prereg_rel=world["prereg_rel"], prereg_sha=world["sha"], scanner=Scanner(), run_index=RUNS,
                   owner_resolver=stub_owner, stdout=buf, expect_counts=None, s0_prefix="S0")
    assert rc == 2 and "missing S0 snapshot file" in buf.getvalue()
    with pytest.raises(SystemExit):
        s1.main(["--help"])
    assert "--s0-prefix" in capsys.readouterr().out


def test_manifest_records_all_s0r2_input_shas(world):
    run(world)
    man = (world["out"] / "S1_manifest.sha256").read_text(encoding="utf-8")
    for n in ("S0r2_cells_expected.csv", "S0r2_class_repro_check.csv", "S0r2_metric_keys.csv", "S0r2_inventory.csv", "S0r2_tokens.csv"):
        line = next(l for l in man.splitlines() if l.endswith(n))
        assert line.split()[0] == hashlib.sha256((world["out"] / n).read_bytes()).hexdigest()
    assert any(l.endswith("S0r2_snapshot.sha256") and l.startswith("MISSING") for l in man.splitlines())  # fixture has none
    assert "S0r2  s0_prefix" in man and "as_of(S0r2_snapshot/as_of.txt)" in man
    assert "2026-10-30T13:51:29Z" in man


def test_dry_run_with_s0r2_opens_no_run_data_and_writes_nothing(world):
    rc, sc, text = run(world, dry=True)
    assert rc == 0 and sc.calls == [] and not list(world["out"].glob("S1_*"))
    assert "DRY-RUN" in text


def test_q_tokens():
    t = s1.q_tokens({"a": ["X-PROV", "X-ABSENT"], "b": ["LISTED-ONLY"], "c": ["A-VERDICT", "A-DESC"], "d": ["D-LOCKED", "A-DESC"],
                     "e": ["UNCOVERED"], "f": ["X-ABSENT", "LISTED-ONLY"]})
    assert t == {"a": "Q-NONE", "b": "Q-LISTED", "c": "Q-INHERIT", "d": "Q-DESC-ONLY", "e": "Q-UNCOVERED", "f": "Q-LISTED"}


# ------------------------------------------------------------------ key resolution
def test_resolve_keys():
    row = {"key_status": "NAMED-ADMITTED", "N_named": "{probe, all, recomp}"}
    assert s1.resolve_keys("Q-I2", row)[0] == ["probe"]  # K(Q-I2) = {probe}: all/recomp not read
    assert s1.resolve_keys("Q-I4", row)[0] == ["probe", "all"] and s1.resolve_keys("Q-I4", row)[1][0][0] == "recomp"
    assert s1.resolve_keys("Q-Z4", {"key_status": "NAMED-ADMITTED", "N_named": "{pin}"}) == (["pin"], [], "probe/test_exact")  # CR-4: pin key read, never a tau
    assert s1.resolve_keys("Q-Z3", {"key_status": "NAMED-ADMITTED", "N_named": "{effr}"})[0] == ["effr"]
    assert s1.resolve_keys("Q-I2", {"key_status": "UNKNOWN", "N_named": "∅"}) == ([], [], "UNKNOWN")
    assert s1.resolve_keys("Q-Z6", None) == ([], [], "UNKNOWN")


# ------------------------------------------------------------------ value guards (the core of PREREG O-2 / sec.9 S1)
def test_no_adapter_for_nonvalue_classes_and_unknown_keys(world):
    rc, sc, _ = run(world)
    paths = [p for p, _ in sc.calls]
    # EXP-001 (UNKNOWN), EXP-013 (X-ABSENT), EXP-023 (D-PREJUDGE), EXP-017 (UNKNOWN D-LOCKED): none of their runs scanned
    al0 = rd(world["out"] / "S1_access_log.csv")
    assert {a["cell"] for a in al0 if a["path"].startswith("/w/g/") and a["event"] == "open"} == {"(Q-Z3,EXP-002)", "(Q-Z4,EXP-006)"}  # never EXP-001's
    assert not any("/w/d/" in p for p in paths)  # pp_ltf_*_s4 = EXP-023
    assert not any("/w/f/" in p for p in paths)  # e020_*
    cells = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    assert cells[("Q-I2", "EXP-001")]["metric_keys_read"] == "UNKNOWN" and cells[("Q-I2", "EXP-001")]["n_table_rows"] == "0"
    assert "키 이름 미고정" in cells[("Q-I2", "EXP-001")]["values_policy"]
    assert cells[("Q-I3", "EXP-023")]["token_V"] == "" and s1.PREJUDGE_NOTE in cells[("Q-I3", "EXP-023")]["values_policy"]
    assert cells[("Q-I3", "EXP-023")]["n_table_rows"] == "0"
    assert cells[("Q-I2", "EXP-012/sealed")]["token_V"] == "" and s1.SEALED_NOTE in cells[("Q-I2", "EXP-012/sealed")]["values_policy"]
    tab = rd(world["out"] / "S1_table.csv")
    assert {t["s"] for t in tab} == {"EXP-005", "EXP-025", "EXP-016", "EXP-002", "EXP-006"}  # EXP-006: CR-4 pin rows
    assert not any(t["run_name"] in ("pp_ltf_g7_lay2_k3_s4", "e020_s1_k3_s1") for t in tab)


def test_prejudge_logs_names_only_and_exclusions_logged(world, monkeypatch):
    monkeypatch.setitem(s1.RUN_SELECT, "EXP-025", ("power_permutation", r"^pp_ltf_"))  # broad selector: the O-2 guard must still drop _s4
    run(world)
    al = rd(world["out"] / "S1_access_log.csv")
    pj = [a for a in al if a["cell"] == "(Q-I3,EXP-023)"]
    assert len(pj) == 1 and pj[0]["event"] == "name_only" and "done=27" in pj[0]["note"]
    assert any(a["event"] == "name_only" and "pp_ltf_g7_lay2_k3_s4" in a["path"] and "O-2" in a["note"] for a in al)
    assert not any(a["event"] == "open" and a["path"].startswith("/w/d/") for a in al)


def test_no_sealed_path_ever_opened(world, monkeypatch):
    seen = []
    real = builtins.open

    def spy(file, *a, **k):
        seen.append(str(file))
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    run(world)
    assert not any("sealed" in p.split("/")[-1] or "/sealed/" in p or "z_snapshots" in p or "preds" in p for p in seen)


def test_rfinal_seedwise_no_aggregation_and_provenance(world):
    run(world)
    tab = rd(world["out"] / "S1_table.csv")
    assert list(tab[0].keys()) == s1.TABLE_FIELDS
    assert not any(("mean" in f.lower() or "avg" in f.lower() or "pool" in f.lower()) for f in s1.TABLE_FIELDS + s1.CELL_FIELDS)
    q_i2 = [t for t in tab if (t["q"], t["s"]) == ("Q-I2", "EXP-005")]
    assert len(q_i2) == 2 and {t["seed"] for t in q_i2} == {"1", "2"}  # one row per seed
    assert {t["metric_key"] for t in q_i2} == {"probe/test_exact"}  # all/recomp not in K(Q-I2)
    a = next(t for t in q_i2 if t["seed"] == "1")
    assert (a["value"], a["step"], a["reduction"], a["frame"], a["n_logged_points"]) == ("0.75", "30", "R-FINAL", "full", "3")
    assert a["label_GC"] == "NONE" and a["protocol"] == "P-PROBE" and a["eval_set_size"] == "UNKNOWN" and a["probe_size"] == "512"
    assert a["arch"] == "trm_singlez" and a["block"] == "mlp" and a["HL_train"] == "(3,6)" and a["k"] == "3"
    assert a["budget"] == "UNKNOWN" and a["weights"] == "UNKNOWN" and a["Tk_class"] == "UNKNOWN" and a["HL_eval"] == "UNKNOWN"
    assert a["provenance_sha256"] != "" and a["row_id"].startswith("T")
    ids = [t["row_id"] for t in tab]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_dlocked_two_keys_separate_protocol_rows(world):
    run(world)
    tab = [t for t in rd(world["out"] / "S1_table.csv") if t["s"] == "EXP-016"]
    assert {(t["metric_key"], t["protocol"]) for t in tab} == {("probe/test_exact", "P-PROBE"), ("all/exact_accuracy", "P-ALL")}
    p = next(t for t in tab if t["protocol"] == "P-ALL")
    assert p["weights"] == "online-raw" and p["probe_size"] == "n/a" and p["cohort"] == "clean-D1" and p["cohort_note"] == ""
    al = rd(world["out"] / "S1_access_log.csv")
    assert any(a["cell"] == "(Q-I4,EXP-016)" and a["event"] == "open" and a["path"].endswith("run-e1.wandb") and len(a["sha256"]) >= 7 for a in al)


def test_cohort_note_exact_string_on_legacy(world):
    run(world)
    tab = rd(world["out"] / "S1_table.csv")
    for t in tab:
        assert t["cohort_note"] == (s1.LEGACY_NOTE if t["cohort"] == "legacy" else "")
    assert s1.LEGACY_NOTE == "legacy, 신규 clean 코호트와 비교 불가"
    cells = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    assert cells[("Q-I2", "EXP-005")]["cohort_note"] == s1.LEGACY_NOTE
    assert cells[("Q-I2", "EXP-001")]["cohort_note"] == s1.LEGACY_NOTE  # legacy source even when no value row
    assert cells[("Q-I4", "EXP-016")]["cohort_note"] == ""
    assert "legacy:2" == cells[("Q-I2", "EXP-005")]["counts_by_cohort"]


def test_non_n10_data_path_excluded(world):
    _cfg(world["root"], "Sigma_k_fig12", "fig1_mlp_z_iter_k3_s2", "data/sigma_k/3", arch="trm_singlez", mlp_t=True, seed=2)
    run(world)
    tab = rd(world["out"] / "S1_table.csv")
    assert {t["seed"] for t in tab if t["s"] == "EXP-005"} == {"1"}
    al = rd(world["out"] / "S1_access_log.csv")
    assert any("not n=10 prefix" in a["note"] for a in al)


def test_atom_columns_verbatim_and_listed_only(world):
    run(world)
    cells = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    a = cells[("Q-I1", "CL-0003")]
    assert (a["atom_verdict_scope"], a["atom_prohibited_upgrade"], a["atom_cohort"], a["token_V"]) == ("SCOPE-3", "PROHIB-3", "clean-D0", "supported")
    b = cells[("Q-Z7", "CL-0006")]
    assert b["token_V"] == "partial" and b["atom_verdict_scope"] == "SCOPE-6" and b["atom_prohibited_upgrade"] == ""
    assert "LISTED-ONLY" in b["values_policy"]


def test_token_discrepancy_and_multi_v_flag(world):
    run(world)
    cells = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    h = cells[("Q-I3", "H-199")]  # UNCOVERED by LH rule: no token claim made, class says UNCOVERED
    assert h["s1_class"] == "UNCOVERED"
    d = cells[("Q-Z5", "H-201")]
    assert d["token_tier"] == "2" and d["token_V"] == "supported"


# ------------------------------------------------------------------ dry-run
def test_dry_run_writes_nothing_reads_no_runs_lists_keys(world):
    rc, sc, text = run(world, dry=True)
    assert rc == 0 and sc.calls == [] and not list(world["out"].glob("S1_*"))
    assert "DRY-RUN" in text and "WOULD_OPEN" in text and "probe/test_exact" in text
    assert "all/exact_accuracy" in text  # D-LOCKED EXP-016 keys listed
    assert "NAME_ONLY" in text  # D-PREJUDGE name-only lookup listed


def test_dry_run_with_injected_index_opens_no_history_or_config(world):
    rc, sc, text = run(world, dry=True)
    assert sc.calls == [] and "would_open".upper() in text
    assert "/w/e/run-e1.wandb" in text  # history files of selected runs are listed, not opened


# ------------------------------------------------------------------ adapters
def test_markdown_table_adapter(tmp_path):
    md = tmp_path / "R.md"
    md.write_text("text\n\n| arm | k | acc |\n|---|---|---|\n| a | 3 | 0.5 |\n| b | 5 | 0.6 |\n\nafter\n", encoding="utf-8")
    log = s1.AccessLog()
    t = s1.markdown_table_adapter(str(md), r"arm \| k", log, "c")
    assert t["header"] == ["arm", "k", "acc"] and t["rows"][1] == {"arm": "b", "k": "5", "acc": "0.6"} and t["line"] == 3
    assert log.rows[-1]["event"] == "open"
    assert s1.markdown_table_adapter(str(md), r"nomatch", log, "c") is None


def test_atom_adapter_missing_atom(world):
    log = s1.AccessLog()
    assert s1.atom_adapter(str(world["out"] / "S0r2_snapshot" / "INDEX.jsonl"), "CL-9999", log, "c") is None


def test_r_final_picks_last_step_not_last_row():
    rows = [{"_step": 5, "k": 1.0}, {"_step": 50, "k": 3.0}, {"_step": 20, "k": 2.0}, {"_step": 60}]
    assert s1.r_final(rows, "k") == (3.0, 50, 3)
    assert s1.r_final(rows, "z") is None


def test_run_selection_anchored_regexes_by_source_and_q(monkeypatch):
    sel, exc = s1.select_runs("EXP-025", "Q-I3", RUNS)
    assert [r["run_name"] for r in sel] == ["pp_ltf_g7_lay2_k3_s1"] and exc == []  # anchored s0r2b regex: _s4 never a candidate (CR-2)
    monkeypatch.setitem(s1.RUN_SELECT, "EXP-025", ("power_permutation", r"^pp_ltf_"))  # broad selector -> O-2 guard still drops D-PREJUDGE names
    sel, exc = s1.select_runs("EXP-025", "Q-I3", RUNS)
    assert [r["run_name"] for r in sel] == ["pp_ltf_g7_lay2_k3_s1"] and exc[0][0] == "pp_ltf_g7_lay2_k3_s4" and "O-2" in exc[0][1]
    base = dict(RUNS[2], run_name="pp_base_tf_z_iter_k3_s1")
    sel, _ = s1.select_runs("EXP-025", "Q-Z1", RUNS + [base])  # CR-6: (Q-Z1, EXP-025) selects the reference cohort only
    assert [r["run_name"] for r in sel] == ["pp_base_tf_z_iter_k3_s1"]
    sel, exc = s1.select_runs("EXP-022", "Q-I3", RUNS)
    assert sel == [] and "no run selector" in exc[0][1]
    sel, _ = s1.select_runs("EXP-016", "Q-I4", RUNS)
    assert [r["run_name"] for r in sel] == ["e016_act_noz_iter_h16_k3_s1"]
    assert s1.select_runs("EXP-016", "Q-I4", [dict(RUNS[4], run_name="e016_act_k3_s1")])[0] == []  # un-anchored r1 name no longer matches


def test_manifest_written_before_values_and_complete(world):
    run(world)
    lines = (world["out"] / "S1_manifest.sha256").read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("tools/state_iter_z_synthesis/s1.py") and "PREREG.md" not in lines[0]
    assert any("S0r2_metric_keys.csv" in l for l in lines) and any(l.endswith("S1_table.csv") for l in lines)
    for l in lines:
        if l.endswith("S1_cells.csv"):
            assert l.split()[0] == hashlib.sha256((world["out"] / "S1_cells.csv").read_bytes()).hexdigest()


def test_dry_run_unresolved_run_list_still_lists_planned_keys_and_selector(world):
    rc, sc, text = run(world, dry=True, index=[])
    assert rc == 0 and sc.calls == []
    assert "history keys ['probe/test_exact'] (run list not resolved in dry-run)" in text
    assert "selector=power_permutation:^e016_" in text


def test_missing_all_config_is_distinct_exclusion_not_n10_label(world):
    import shutil
    shutil.rmtree(world["root"] / "checkpoints" / "Sigma_k_fig12" / "fig1_mlp_z_iter_k3_s2")
    run(world)
    assert {t["seed"] for t in rd(world["out"] / "S1_table.csv") if t["s"] == "EXP-005"} == {"1"}
    al = rd(world["out"] / "S1_access_log.csv")
    assert any("all_config.yaml MISSING" in a["note"] for a in al)
    assert not any("not n=10 prefix" in a["note"] and "fig1_mlp_z_iter_k3_s2" in a["path"] for a in al)


def test_build_run_index_args_then_config_yaml_fallback(tmp_path):
    import yaml
    w = tmp_path / "wandb"
    a = w / "run-20261017_000000-aaaa1111"
    (a / "files").mkdir(parents=True)
    (a / "files" / "wandb-metadata.json").write_text(json.dumps({"args": ["+project_name=power_permutation", "+run_name=e016_x"]}))
    (a / "run-aaaa1111.wandb").write_bytes(b"")
    b = w / "run-20260701_000000-bbbb2222"  # legacy: metadata has no +project_name override; config.yaml has value-wrapped keys
    (b / "files").mkdir(parents=True)
    (b / "files" / "wandb-metadata.json").write_text(json.dumps({"args": ["+run_name=fig1_tf_z_iter_k1_s1"]}))
    (b / "files" / "config.yaml").write_text(yaml.safe_dump({"project_name": {"value": "Sigma_k_fig12"}}))
    c = w / "run-20260702_000000-cccc3333"  # no metadata at all: override strings inside config.yaml
    (c / "files").mkdir(parents=True)
    (c / "files" / "config.yaml").write_text(yaml.safe_dump({"_wandb": {"value": {"args": ["+project_name=Sigma_k_new", "+run_name=abl_x_k2"]}}}))
    log = s1.AccessLog()
    idx = {r["run_name"]: r["project"] for r in s1.build_run_index(tmp_path, "wandb", log)}
    assert idx == {"e016_x": "power_permutation", "fig1_tf_z_iter_k1_s1": "Sigma_k_fig12", "abl_x_k2": "Sigma_k_new"}
    assert not any("wandb-summary" in r["path"] for r in log.rows)


# ================================================================== INTERP_S1.md code-difference fixes (CD-1 .. CD-12)
UTC = s1.datetime.timezone.utc


def test_cd5_cohort_requires_project_path_and_start_time():
    c = s1.cohort_for
    assert c("Sigma_k_fig12", "data/sigma_k_10/3", "2026-07-01T00:00:00Z") == "legacy"
    assert c("Sigma_k_fig12", "data/sigma_k_10/3", "2026-07-20T04:17:59Z") == "legacy"
    assert c("Sigma_k_fig12", "data/sigma_k_10/3", "2026-07-20T04:18:00Z") == "UNKNOWN"  # inside the time-zone window
    assert c("Sigma_k_fig12", "data/sigma_k_10/9", "2026-07-25T00:00:00Z") == "UNKNOWN"   # after t_build (U-4 fig1d-like)
    assert c("Sigma_k_new", "data/sigma_k_10/3", "2026-07-20T13:29:00Z") == "UNKNOWN"
    assert c("Sigma_k_new", "data/sigma_k_10/3", "2026-07-20T13:30:00Z") == "clean-D0"
    assert c("Sigma_k_new", "data/sigma_k_10/11", "2026-07-21T12:00:00Z") == "UNKNOWN"   # k=11 rebuilt 07-21
    assert c("Sigma_k_new", "data/sigma_k_10/11", "2026-07-23T00:00:00Z") == "clean-D0"
    assert c("Sigma_k_new", "data/sigma_k/3", "2026-09-01T00:00:00Z") == "UNKNOWN"        # wrong path prefix (n=20)
    assert c("power_permutation", "data/power_permutation/d1/3", "2026-10-17T07:53:05.581645Z") == "clean-D1"
    assert c("power_permutation", "data/power_permutation/d0/3", "2026-10-17T07:53:05Z") == "UNKNOWN"
    assert c("power_permutation", "data/power_permutation/d1/3", None) == "UNKNOWN"       # start time unreadable
    assert c("power_permutation", "data/power_permutation/d1/3", "garbage") == "UNKNOWN"
    assert c("other_project", "data/sigma_k_10/3", "2026-10-01T00:00:00Z") == "UNKNOWN"


def test_cd5_naive_timestamp_read_as_utc_and_both_spellings_in_index(tmp_path):
    assert s1.parse_started("2026-07-01T00:00:00") == s1.datetime.datetime(2026, 7, 1, tzinfo=UTC)
    w = tmp_path / "wandb"
    for i, key in enumerate(("startedAt", "started_at")):
        d = w / f"run-2026070{i + 1}_000000-id{i}"
        (d / "files").mkdir(parents=True)
        (d / "files" / "wandb-metadata.json").write_text(json.dumps({key: f"2026-07-0{i + 1}T00:00:00Z",
                                                                      "args": ["+project_name=Sigma_k_fig12", f"+run_name=r{i}"]}))
    idx = {r["run_name"]: r["started_at"] for r in s1.build_run_index(tmp_path, "wandb", s1.AccessLog())}
    assert idx == {"r0": "2026-07-01T00:00:00Z", "r1": "2026-07-02T00:00:00Z"}


def test_cd5_end_to_end_legacy_run_started_after_build_is_unknown_and_not_evidence(world):
    run(world)
    tab = rd(world["out"] / "S1_table.csv")
    r = next(t for t in tab if t["s"] == "EXP-002")  # fig1_tf_z_iter_k3_s1 in Sigma_k_fig12, started 2026-10-17
    assert r["cohort"] == "UNKNOWN" and r["cohort_note"] == "" and r["evidence_use"].startswith("no:") and "cohort UNKNOWN" in r["evidence_use"]
    assert r["eval_set_size"] == "UNKNOWN" and r["started_at"].startswith("2026-10-17")
    l = next(t for t in tab if t["s"] == "EXP-005" and t["seed"] == "1")
    assert l["cohort"] == "legacy" and l["cohort_note"] == s1.LEGACY_NOTE and l["eval_set_size"] == "UNKNOWN"
    d1 = next(t for t in tab if t["s"] == "EXP-016" and t["protocol"] == "P-PROBE")
    assert d1["cohort"] == "clean-D1" and d1["eval_set_size"] == "888"  # CD-12: only clean cohorts read dataset.json


def test_cd10_registry_status_not_in_token_columns(world):
    run(world)
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    e16 = c[("Q-I4", "EXP-016")]
    assert (e16["token_tier"], e16["token_V"], e16["token_N"]) == ("", "", "")
    assert e16["registry_status_N"] == "PASS|in-progress" and e16["marker"] == s1.D_LOCKED_MARKER
    assert e16["token_restatement"] == "status: PASS-pending-verifier" and e16["token_note"] == ""  # CR-5: sha pin matches -> one line verbatim
    assert "BODY-MUST-NOT-BE-RESTATED" not in e16["token_restatement"] and "U-19" in e16["token_discrepancy"]  # tier3_eligible False
    e17 = c[("Q-Z5", "EXP-017")]
    assert e17["token_N"] == "" and e17["registry_status_N"] == "VERIFIED-rN|completed"
    assert c[("Q-I2", "EXP-005")]["token_N"] == "" and c[("Q-I2", "EXP-005")]["registry_status_N"] == "DRAFT"
    h = c[("Q-Z5", "H-201")]  # H D-LOCKED: tier-2 word + verbatim snapshot status line + marker
    assert (h["token_tier"], h["token_V"], h["marker"]) == ("2", "supported", s1.D_LOCKED_MARKER)
    assert h["token_restatement"] == "status: supported" and h["registry_status_N"] == ""
    for r in c.values():  # no tier-5/6 word may ever be a token
        assert r["token_tier"] in ("", "1", "2", "3", "4")


def test_cr4_pin_rows_and_exp022_open_nothing(world):
    rc, sc, _ = run(world)
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    pin, e22 = c[("Q-Z4", "EXP-006")], c[("Q-I3", "EXP-022")]
    assert pin["values_policy"].startswith("extract R-FINAL") and pin["metric_keys_read"] == "probe/test_exact"
    rows = [t for t in rd(world["out"] / "S1_table.csv") if t["s"] == "EXP-006"]
    assert rows and {(t["metric_key"], t["protocol"], t["reduction"]) for t in rows} == {("probe/test_exact", "P-PROBE", "R-FINAL")}
    assert all("τ 아님" in t["note"] and s1.LABEL_GC_NOTE in t["note"] for t in rows)
    assert not any("tau" in t["metric_key"].lower() or "τ" in t["metric_key"] for t in rd(world["out"] / "S1_table.csv"))  # no tau row
    assert "extract" not in e22["values_policy"] and "자체 run 없음" in e22["values_policy"]  # R08: nothing read, nothing shown as read (CR-9)
    assert e22["metric_keys_read"] == "" and e22["n_table_rows"] == "0"
    assert [a for a in rd(world["out"] / "S1_access_log.csv") if a["cell"] == "(Q-I3,EXP-022)"] == []


def test_cr4_plan_cell_policy_strings():
    pin = {"key_status": "NAMED-ADMITTED", "N_named": "{pin}"}
    pl = s1.plan_cell("Q-Z4", "EXP-006", "E", "A-DESC", pin)
    assert pl["extract"] is True and pl["read"] == ["pin"] and pl["mk"] == "probe/test_exact"
    assert s1.plan_cell("Q-I3", "EXP-022", "E", "A-DESC", {"key_status": "NAMED-ADMITTED", "N_named": "{probe}"})["policy"].startswith("none: 자체 run 없음")
    assert s1.plan_cell("Q-I2", "EXP-005", "E", "A-DESC", {"key_status": "NAMED-ADMITTED", "N_named": "{probe}"})["extract"] is True
    assert s1.plan_cell("Q-I2", "EXP-005", "E", "D-PREJUDGE", {"key_status": "NAMED-ADMITTED", "N_named": "{probe}"})["extract"] is False
    assert s1.plan_cell("Q-I2", "EXP-001", "E", "A-DESC", {"key_status": "UNKNOWN", "N_named": "∅"})["policy"].endswith("값 없음)")


def test_cr5_restatement_changed_file_missing_pin_and_dry_run(world):
    res = world["root"] / "EXP-016_RESULTS.md"
    res.write_text(res.read_text(encoding="utf-8").replace("PASS-pending-verifier", "PASS"), encoding="utf-8")  # edited after as_of
    rc, _, out = run(world)
    e16 = next(r for r in rd(world["out"] / "S1_cells.csv") if (r["q"], r["s"]) == ("Q-I4", "EXP-016"))
    assert rc == 0 and e16["token_restatement"] == "" and e16["token_note"].startswith("DISCREPANCY: as_of 뒤 변경")
    assert "DISCREPANCY: Q-I4 EXP-016" in out
    # no pin file -> no restatement
    for f in world["out"].glob("S1_*"):
        f.unlink()
    (world["out"] / "S0r2b_evidence_files.sha256").unlink()
    rc, _, _ = run(world)
    e16 = next(r for r in rd(world["out"] / "S1_cells.csv") if (r["q"], r["s"]) == ("Q-I4", "EXP-016"))
    assert e16["token_restatement"] == "" and "no sha pin" in e16["token_note"]
    # E D-LOCKED with no tier-3 row (EXP-017): discrepancy, nothing restated
    e17 = next(r for r in rd(world["out"] / "S1_cells.csv") if (r["q"], r["s"]) == ("Q-Z5", "EXP-017"))
    assert e17["token_restatement"] == "" and "no tier-3" in e17["token_note"]


def test_cr5_dry_run_does_not_read_results_file(world, monkeypatch):
    seen = []
    real = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (seen.append(str(f)), real(f, *a, **k))[1])
    rc, sc, text = run(world, dry=True)
    assert rc == 0 and not any(p.endswith("EXP-016_RESULTS.md") for p in seen)
    assert "EXP-016_RESULTS.md" in text and "WOULD_OPEN" in text and not list(world["out"].glob("S1_*"))


def test_cr7_manifest_pins_s0_scripts_interp_and_run_family_map(world):
    sd = world["root"] / "tools" / "state_iter_z_synthesis"
    sd.mkdir(parents=True)
    for n in ("s0r2a.py", "s0r2b.py", "s0r2c.py"):
        (sd / n).write_text(f"# {n}\n", encoding="utf-8")
    (world["out"] / "INTERP_S1.md").write_text("interp\n", encoding="utf-8")
    mp = world["root"] / s1.RUN_FAMILY_MAP_REL
    mp.parent.mkdir(parents=True)
    mp.write_text("[]\n", encoding="utf-8")
    run(world)
    man = (world["out"] / "S1_manifest.sha256").read_text(encoding="utf-8").splitlines()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    want = {"tools/state_iter_z_synthesis/s0r2a.py": sha(b"# s0r2a.py\n"), "tools/state_iter_z_synthesis/s0r2b.py": sha(b"# s0r2b.py\n"),
            "tools/state_iter_z_synthesis/s0r2c.py": sha(b"# s0r2c.py\n"), "out/INTERP_S1.md": sha(b"interp\n"),
            s1.RUN_FAMILY_MAP_REL: sha(b"[]\n"), "S0r2b_evidence_files.sha256": sha((world["out"] / "S0r2b_evidence_files.sha256").read_bytes())}
    for name, h in want.items():
        assert any(l.split(None, 1)[0] == h and l.endswith(name) for l in man), name
    # manifest part 1 precedes the output shas
    assert max(i for i, l in enumerate(man) if "s0r2c.py" in l or "run-family" in l) < next(i for i, l in enumerate(man) if l.endswith("S1_table.csv"))


def test_cr7_missing_inputs_are_recorded_as_missing_not_skipped(world):
    run(world)
    man = (world["out"] / "S1_manifest.sha256").read_text(encoding="utf-8").splitlines()
    for name in ("s0r2a.py", "s0r2b.py", "s0r2c.py", "INTERP_S1.md", "run-family-exp-map.yaml"):
        assert any(l.startswith("MISSING") and l.endswith(name) for l in man), name


def test_cr8_label_gc_note_on_every_value_row(world):
    run(world)
    tab = rd(world["out"] / "S1_table.csv")
    assert tab and all(t["label_GC"] == "NONE" and s1.LABEL_GC_NOTE in t["note"] for t in tab)
    assert s1.LABEL_GC_NOTE == "S1은 소스 G/C 정의를 적용하지 않음(NONE = 원값만)"


def test_cr9_metric_keys_read_empty_when_not_extracting(world):
    run(world)
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    for k, r in c.items():
        if r["values_policy"].startswith("extract"):
            assert r["metric_keys_read"] not in ("", "UNKNOWN"), k
        elif r["key_status"] == "NAMED-ADMITTED":
            assert r["metric_keys_read"] == "", k


def test_cr10_run_counts_as_evidence_only_if_every_dir_is_evidence(world):
    bad = dict(RUNS[4], wandb_dir="/w/e2", wandb_run_id="e2", history_path="/w/e2/run-e2.wandb", started_at="2026-07-01T00:00:00Z")  # 2nd dir: cohort UNKNOWN
    run(world, index=RUNS + [bad])
    e16 = next(r for r in rd(world["out"] / "S1_cells.csv") if (r["q"], r["s"]) == ("Q-I4", "EXP-016"))
    assert e16["n_runs"] == "1" and e16["n_runs_evidence_use"] == "0"
    tab = [t for t in rd(world["out"] / "S1_table.csv") if t["s"] == "EXP-016"]
    assert {t["evidence_use"][:3] for t in tab} == {"yes", "no:"}  # row level keeps both


def test_cd6_cd7_protocol_vocabulary_and_probe_size(world):
    run(world)
    tab = rd(world["out"] / "S1_table.csv")
    pr = next(t for t in tab if t["metric_key"] == "z/eff_rank")
    assert pr["protocol"] == "UNKNOWN" and "vocabulary" in pr["protocol_note"]
    vocab = {"P-ALL", "P-RECOMP", "P-PROBE", "P-RELOAD", "P-DECODE", "P-JUDGE", "UNKNOWN"}
    assert {t["protocol"] for t in tab} <= vocab
    assert s1.probe_size_for("P-PROBE", {"z_probe_size": 512}) == "512"
    assert s1.probe_size_for("P-PROBE", {"z_probe_size": 256}) == "256"
    assert s1.probe_size_for("P-PROBE", {"z_probe_size": 100}) == "UNKNOWN"
    assert s1.probe_size_for("P-PROBE", {"z_probe_size": "UNKNOWN"}) == "UNKNOWN"


def test_cd8_counts_are_run_counts_not_row_counts(world):
    dup = dict(RUNS[4], wandb_dir="/w/e2", wandb_run_id="e2", history_path="/w/e2/run-e2.wandb")  # same run name, 2nd wandb dir
    run(world, index=RUNS + [dup])
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    e16 = c[("Q-I4", "EXP-016")]
    assert e16["n_table_rows"] == "4" and e16["n_runs"] == "1" and e16["n_runs_evidence_use"] == "1"  # 2 dirs x 2 keys, 1 run
    assert e16["counts_by_cohort"] == "clean-D1:1"
    assert {t["dup_wandb_dirs"] for t in rd(world["out"] / "S1_table.csv") if t["s"] == "EXP-016"} == {"2"}
    e5 = c[("Q-I2", "EXP-005")]
    assert e5["n_runs"] == "2" and e5["n_table_rows"] == "2" and e5["counts_by_cohort"] == "legacy:2"


def test_cd9_card_level_marker_only_on_component_cells(world):
    run(world)
    c = {(r["q"], r["s"]): r for r in rd(world["out"] / "S1_cells.csv")}
    assert c[("Q-I2", "H-022")]["token_scope"].startswith("card-level") and "O(log k)" in c[("Q-I2", "H-022")]["token_scope"]
    assert "MULTIPLE-V-WORDS-UNRESOLVED" in c[("Q-I2", "H-022")]["notes"]
    assert c[("Q-Z3", "H-023")]["token_scope"].startswith("card-level") and "PR-signature" in c[("Q-Z3", "H-023")]["token_scope"]
    assert c[("Q-Z1", "H-023")]["token_scope"] == "" and c[("Q-Z5", "H-201")]["token_scope"] == ""


def test_cd4_owner_resolver_chain(tmp_path):
    import yaml
    root = tmp_path / "r"
    snap = root / "snap"
    snap.mkdir(parents=True)
    (snap / "ID_ALIASES.yaml").write_text(yaml.safe_dump({"entries": [{"canonical": "EXP-025", "aliases": [{"string": "EXP-015"}]}]}))
    mp = root / s1.RUN_FAMILY_MAP_REL
    mp.parent.mkdir(parents=True)
    ev = lambda who, role, tier="registry", src="lab/13_reference/11_registries/EXPERIMENTS.md:26": dict(applies_to=who, role=role, tier=tier, source=src)
    mp.write_text(yaml.safe_dump([
        dict(project="power_permutation", family="ltf", name_regex=r"^pp_ltf_g7_k\d+_s\d+$", evidence=[ev("EXP-015", "owner"), ev("EXP-016", "comparator")]),
        dict(project="Sigma_k_fig12", family="fig1_tf", name_regex=r"^fig1_tf_k\d+$", evidence=[ev("EXP-001", "owner"), ev("EXP-004", "analysis-consumer")]),
        dict(project="Sigma_k_fig12", family="two", name_regex=r"^two_k\d+$", evidence=[ev("EXP-001", "owner"), ev("EXP-005", "owner")]),
        dict(project="Sigma_k_fig12", family="doc", name_regex=r"^doc_k\d+$", evidence=[ev("EXP-007", "owner", tier="doc")]),
        dict(project="Sigma_k_fig12", family="none", name_regex=r"^none_k\d+$", evidence=[dict(role="campaign has no owning EXP", tier="doc")]),
        dict(project="Sigma_k_new", family="extension", name_regex=r"^ext_k\d+$", evidence=[ev("EXP-012", "owner (extension arm)", src="x.md:1")]),
    ]))
    res = s1.make_owner_resolver(root, snap, s1.AccessLog())
    assert res("power_permutation", "pp_ltf_g7_k3_s1") == ("EXP-025", "EXPERIMENTS.md(quoted in map)")  # alias EXP-015 -> canonical
    assert res("Sigma_k_fig12", "fig1_tf_k3")[0] == "EXP-001"                # analysis-consumer is not an owner
    assert res("Sigma_k_fig12", "two_k3")[0] == "UNKNOWN"                    # two owner candidates: not guessed
    assert res("Sigma_k_fig12", "doc_k3")[0] == "UNKNOWN"                    # non-registry tier ignored
    assert res("Sigma_k_fig12", "none_k3")[0] == "UNKNOWN"
    assert res("Sigma_k_new", "ext_k3") == ("EXP-012", "map")                # role 'owner (extension arm)' counts as owner
    assert res("Sigma_k_new", "abl_x_k3")[0] == "UNKNOWN"                    # family absent from the 2026-10-04 map
    assert res("power_permutation", "fig1_tf_k3")[0] == "UNKNOWN"            # right regex, wrong project
    empty = s1.make_owner_resolver(tmp_path / "nomap", snap, s1.AccessLog())
    assert empty("x", "y") == ("UNKNOWN", "run-family map unavailable")


def test_cd4_owner_unknown_end_to_end(world):
    _cfg(world["root"], "Sigma_k_new", "abl_x_k3_s1", "data/sigma_k_10/3", seed=1)
    x = dict(run_name="abl_x_k3_s1", project="Sigma_k_new", wandb_dir="/w/i", wandb_run_id="i1", history_path="/w/i/run-i1.wandb",
             started_at="2026-09-01T00:00:00Z")
    inv = world["out"] / "S0r2_inventory.csv"
    inv.write_text(inv.read_text(encoding="utf-8") + "EXP-011,E,True,False,False,False,0,0,0,0\n", encoding="utf-8")
    cells = world["out"] / "S0r2_cells_expected.csv"
    cells.write_text(cells.read_text(encoding="utf-8") + "x,2026-10-30T00:00:00Z,Q-I2,EXP-011,E,A-DESC,Q-INHERIT\n", encoding="utf-8")
    rp = world["out"] / "S0r2_class_repro_check.csv"
    rp.write_text(rp.read_text(encoding="utf-8") + "Q-I2,EXP-011,A-DESC,A-DESC,r,True\n", encoding="utf-8")
    keys = world["out"] / "S0r2_metric_keys.csv"
    keys.write_text(keys.read_text(encoding="utf-8") + "R99,Q-I2,EXP-011,ACC,K,{probe},NAMED-ADMITTED,e,c,A-DESC\n", encoding="utf-8")
    rc, _, txt = run(world, index=RUNS + [x], known_mismatches={})
    assert rc == 0, txt
    t = next(t for t in rd(world["out"] / "S1_table.csv") if t["s"] == "EXP-011")
    assert t["cohort"] == "clean-D0" and t["exp_owner"] == "UNKNOWN" and t["evidence_use"] == "no: exp_owner UNKNOWN"
    c = next(r for r in rd(world["out"] / "S1_cells.csv") if r["s"] == "EXP-011")
    assert c["n_runs"] == "1" and c["n_runs_evidence_use"] == "0" and "s0_ckpt_dirs_matched=" in c["notes"] and "s1_candidate_runs=1" in c["notes"]


def test_cd1_only_own_outputs_block_a_run(world):
    (world["out"] / "INTERP_S1.md").write_text("note")
    (world["out"] / "S1_scratch.txt").write_text("x")
    assert run(world)[0] == 0
    assert run(world)[0] == 3  # now S1_table.csv etc. exist


# ================================================================== VERIFY-V1-r1 fixes (F1 alias scope, F2 sha on every open event, --s0-dir)
def test_f1_ambiguous_alias_resolves_by_evidence_file_scope_not_by_last_entry(tmp_path):
    import yaml
    root = tmp_path / "r"
    snap = root / "snap"
    snap.mkdir(parents=True)
    reg = "lab/13_reference/11_registries/EXPERIMENTS.md"
    ph = "lab/23_reports/x/PLACEHOLDERS.md"
    (snap / "ID_ALIASES.yaml").write_text(yaml.safe_dump({"entries": [
        {"canonical": "EXP-025", "source": f"사용자 결정. 대상 레지스트리: {reg} 의 EXP-025", "aliases": [{"string": "EXP-015", "scope": "looped-tf-direct-control 산문·표"}]},
        {"canonical": "EXP-100", "source": f"{ph}:49", "aliases": [{"string": "EXP-015", "scope": "paper-position 폐기 라벨"}]}]}))  # EXP-100 listed LAST
    mp = root / s1.RUN_FAMILY_MAP_REL
    mp.parent.mkdir(parents=True)
    ev = lambda src: dict(applies_to="EXP-015", role="owner", tier="registry", source=src)
    mp.write_text(yaml.safe_dump([
        dict(project="power_permutation", family="ltf", name_regex=r"^pp_ltf_g7_lay2_k\d+_s\d+$", evidence=[ev(f"{reg}:587")]),
        dict(project="power_permutation", family="nowhere", name_regex=r"^pp_other_k\d+$", evidence=[ev("lab/unrelated/file.md:3")]),
    ]))
    res = s1.make_owner_resolver(root, snap, s1.AccessLog())
    assert res("power_permutation", "pp_ltf_g7_lay2_k3_s1") == ("EXP-025", "EXPERIMENTS.md(quoted in map)")  # never EXP-100
    assert res("power_permutation", "pp_other_k3")[0] == "UNKNOWN"  # no candidate names the evidence file: not guessed, not first/last wins


def test_f1_canonical_ids_and_single_candidate_aliases_unchanged(tmp_path):
    import yaml
    root = tmp_path / "r"
    snap = root / "snap"
    snap.mkdir(parents=True)
    (snap / "ID_ALIASES.yaml").write_text(yaml.safe_dump({"entries": [
        {"canonical": "EXP-025", "source": "s", "aliases": [{"string": "EXP-015", "scope": "a"}, {"string": "EXP-0150", "scope": "b"}]}]}))
    mp = root / s1.RUN_FAMILY_MAP_REL
    mp.parent.mkdir(parents=True)
    ev = lambda who: dict(applies_to=who, role="owner", tier="registry", source="any.md:1")
    mp.write_text(yaml.safe_dump([
        dict(project="p", family="a", name_regex=r"^a$", evidence=[ev("EXP-015")]),
        dict(project="p", family="b", name_regex=r"^b$", evidence=[ev("EXP-025")]),
        dict(project="p", family="c", name_regex=r"^c$", evidence=[ev("EXP-01")]),  # prefix of an alias: must NOT resolve to anything but itself
    ]))
    res = s1.make_owner_resolver(root, snap, s1.AccessLog())
    assert res("p", "a")[0] == "EXP-025" and res("p", "b")[0] == "EXP-025" and res("p", "c")[0] == "EXP-01"


def test_f2_every_open_event_has_sha_and_unopened_events_say_so(world, tmp_path):
    rc, _, _ = run(world)
    al = rd(world["out"] / "S1_access_log.csv")
    opens = [a for a in al if a["event"] == "open"]
    assert opens and all(a["sha256"] not in ("", None) for a in opens)
    assert all(a["sha256"] == s1.NOT_OPENED for a in al if a["event"] in ("name_only", "would_open", "denied"))
    assert any(a["event"] == "name_only" for a in al)
    # wandb-metadata.json / config.yaml lookups (the 1257 empty cells of V1-r1) now carry the file hash
    w = tmp_path / "wandb" / "run-20261017_000000-aaaa1111"
    (w / "files").mkdir(parents=True)
    meta = w / "files" / "wandb-metadata.json"
    meta.write_text(json.dumps({"args": ["+project_name=power_permutation", "+run_name=e016_x"]}))
    (w / "run-aaaa1111.wandb").write_bytes(b"")
    cfgd = tmp_path / "wandb" / "run-20260702_000000-cccc3333"
    (cfgd / "files").mkdir(parents=True)
    import yaml
    (cfgd / "files" / "config.yaml").write_text(yaml.safe_dump({"project_name": {"value": "Sigma_k_fig12"}, "run_name": {"value": "fig1_x"}}))
    log = s1.AccessLog()
    s1.build_run_index(tmp_path, "wandb", log)
    ev = {Path(r["path"]).name: r for r in log.rows if r["event"] == "open"}
    assert ev["wandb-metadata.json"]["sha256"] == hashlib.sha256(meta.read_bytes()).hexdigest()
    assert ev["config.yaml"]["sha256"] == hashlib.sha256((cfgd / "files" / "config.yaml").read_bytes()).hexdigest()


def test_s0_dir_lets_out_point_to_a_fresh_folder_without_s1_collision(world, tmp_path):
    fresh = tmp_path / "fresh_out"
    run(world)  # S1_* now exist in the S0 folder (the old output)
    buf = io.StringIO()
    rc = s1.run_s1(world["root"], fresh, prereg_rel=world["prereg_rel"], prereg_sha=world["sha"], scanner=Scanner(), run_index=RUNS,
                   owner_resolver=stub_owner, stdout=buf, expect_counts=None, expect_class_counts=None, s0_dir=world["out"],
                   now_fn=lambda: "2026-10-30T13:00:00Z")
    assert rc == 0, buf.getvalue()
    assert {p.name for p in fresh.glob("S1_*")} == {"S1_table.csv", "S1_cells.csv", "S1_access_log.csv", "S1_manifest.sha256"}
    man = (fresh / "S1_manifest.sha256").read_text(encoding="utf-8")
    assert "fresh_out" in man or "/S1_table.csv" in man
    assert any(l.endswith("S0r2_inventory.csv") for l in man.splitlines()) and "out/S0r2_inventory.csv" in man  # inputs labelled with the S0 dir
    # without s0_dir a fresh folder has no inputs -> gate failure, nothing written
    buf = io.StringIO()
    rc = s1.run_s1(world["root"], tmp_path / "empty_out", prereg_rel=world["prereg_rel"], prereg_sha=world["sha"], scanner=Scanner(),
                   run_index=RUNS, stdout=buf, expect_counts=None)
    assert rc == 2 and not list((tmp_path / "empty_out").glob("S1_*")) if (tmp_path / "empty_out").exists() else rc == 2
    # old S0 folder untouched by the fresh run
    assert (world["out"] / "S1_table.csv").read_bytes() != b""
