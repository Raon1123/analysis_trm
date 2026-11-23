"""Unit tests for tools/timeline/build_timeline.py (fixture lab tree in tmp_path)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "timeline" / "build_timeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_timeline", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bt = load_module()

REGISTRY_TEXT = """# Experiment Ledger

## EXP-001 — Fig 1/2 ablation (fixture)

| Field | Value |
|---|---|
| **EXP-id** | EXP-001 |
| **Date registered** | 2026-06-07 |
| **Status** | **closed** — postmortem 2026-06-23 (`:64`); mlp block **operationalized as EXP-005** |
| **H-NNN tested** | H-022 (depth), H-023 (z-attribution) |

## EXP-006 — fixture tau grid

| Field | Value |
|---|---|
| **Date registered** | 2026-07-02 |
| **Status** | in-progress — Phase A extraction **completed 2026-07-02** (fixture) |
| **H-NNN tested** | H-026 (fixture) |

## EXP-024 — fixture N1 shape

| Field | Value |
|---|---|
| **Date registered** | 2026-09-05 (`registered_at: "2026-09-05T00:40"`) |
| **Status** | RESULTS-COMPLETE-13-OF-13 (doc frontmatter `status`, `21_2026-09-05_n1-hl-grid-prereg/00_PREREG.md:9`) |
| **Design doc** | `lab/23_reports/11_lab-reports/21_2026-09-05_n1-hl-grid-prereg/00_PREREG.md` |
| **H-NNN tested** | H-108 (fixture) |

## EXP-002 — PR analysis (fixture)

| Field | Value |
|---|---|
| **EXP-id** | EXP-002 |
| **Date** | 2026-06-21 |
| **Status** | completed (no date in this cell) |
| **H-NNN tested** | H-023 (PR) |

## EXP-016 — fixture ACT axis

| Field | Value |
|---|---|
| **Date registered** | 2026-09-23 |
| **Design doc** | `lab/21_experiments/11_lab-experiments/EXP-016_fixture.md` |
| **Notes** | cites `lab/23_reports/11_lab-reports/fixture-other.md` (not a design doc) |
| **Status** | pre-registered-unrun (fixture) |
| **H-NNN tested** | H-200 (fixture) |

## EXP-112 — fixture H-108 depth sweep

| Field | Value |
|---|---|
| **Date registered** | 2026-09-02 (`registered_at: "2026-09-02T11:10"`) |
| **Status** | RESULTS-COMPLETE-13-OF-13 (2026-09-05 closeout; doc frontmatter `status`) |
| **H-NNN tested** | H-108 (fixture) |

## EXP-019 — fixture N4 (registry canonical)

| Field | Value |
|---|---|
| **Date registered** | 2026-10-30 |
| **Status** | pre-registered-unrun (fixture) |
| **H-NNN tested** | H-200 (fixture) |

## EXP-026 — fixture multi-date status

| Field | Value |
|---|---|
| **Date registered** | 2026-09-30 |
| **Status** | in-progress (2026-10-01) — judged 2026-10-05 (fixture) |
| **H-NNN tested** | H-199 (fixture) |

## EXP-025 (looped-tf-direct-control) — fixture

| Field | Value |
|---|---|
| **Date registered** | 2026-09-23 |
| **Status** | **completed (2026-10-03)** — judged; see EXP-015_FREEZE.sha256 |
| **H-NNN tested** | H-199 (C-LTF-1) |
"""

ALIASES_TEXT = """entries:
  - canonical: EXP-025
    aliases:
      - string: EXP-015
        scope: "2026-09-23 ~ 2026-10-30 기간의 모든 기록"
        note: "legacy"
    physical_prefix: "EXP-015_"
    decided: "fixture"
    source: "fixture"
  - canonical: EXP-100
    aliases:
      - string: EXP-015
        scope: "paper-position 캠페인 PLACEHOLDERS.md 의 폐기된 임시 라벨"
        note: "other object"
    physical_prefix: null
    decided: "fixture"
    source: "fixture"
  - canonical: EXP-024
    aliases:
      - string: N1-2026-09-05
        scope: "atoms exp_ids"
        note: "non-EXP alias"
    physical_prefix: null
    decided: "fixture"
    source: "fixture"
  - canonical: EXP-020
    aliases:
      - string: EXP-019
        scope: "2026-10-22 ~ 2026-10-23 draft number (explanatory)"
        note: "explanatory only; EXP-019 is a registry canonical"
    physical_prefix: "EXP-020_"
    decided: "fixture"
    source: "fixture"
"""

WITHHELD_OK = """version: 1
updated: 2026-10-31
families:
  - id: fam-released
    status: released
    exp_ids: [EXP-013]
    patterns:
      - 'fig1_mlp_z_iter_k\\d+_s2'
  - id: fam-secret
    status: withheld
    exp_ids: [EXP-099]
    patterns:
      - 'zzsecretword'
"""

EXP016_TEXT = """---
id: EXP-016
status: pre-registered-unrun # FROZEN at registration
date_designed: 2026-09-23
date_closed: ~
---

# EXP-016 fixture body
"""

WITNESS_TEXT = """# witness fixture

## r1 freeze — 2026-09-23T13:11:38Z (UTC)

enqueue record (2026-09-30T13:01:06Z, fixture) for EXP-016 jobs

hash check at 2026-09-23T13:20Z (UTC) matched
"""

RESULTS_TEXT = """---
title: "EXP-017 RESULTS fixture"
date: 2026-10-03
status: VERIFIED-r1-PASS-2026-10-04 (verifier target sha256 abc)
judged_at: 2026-10-04T02:33Z
---

body
"""

PP_TEXT = """---
title: "paper-position fixture"
date: 2026-09-01
status: DRAFT
---

The provisional label EXP-015 stands for something else here.
"""

H199_STATUS_TEXT = """---
id: H-199
exp_id: EXP-015 (looped-tf-direct-control)
status: "falsified-as-framed (fixture sentence)"
updated: 2026-10-03
---
"""

ATOM_LINE = json.dumps({
    "atom_id": "CL-0001", "created": "2026-09-02", "exp_ids": ["N1-2026-09-05", "EXP-112"],
    "hypothesis_ids": ["H-108"], "kind": "claim", "verdict": "partial", "slug": "fixture-atom",
}, ensure_ascii=False)

HANDOFF_TEXT = """# index fixture

- `[ARG]` [2026-10-27 09:12] **in-progress** *(taken 09:17)* — fixture row
- `[PPC]` [2026-10-27 09:12] **in-progress** — not autoresearch
"""

AUTORESEARCH_TEXT = """# AUTORESEARCH fixture

> **Rewritten 2026-10-19** (fixture instruction).
"""

VERIFY_YAML = """model: fixture
round: 1
date: '2026-10-03'
verdict: PASS
verdict_summary: 'fixture'
"""


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def lab(tmp_path: Path) -> Path:
    root = tmp_path / "lab"
    write(root / "13_reference/11_registries/EXPERIMENTS.md", REGISTRY_TEXT)
    write(root / "13_reference/11_registries/ID_ALIASES.yaml", ALIASES_TEXT)
    write(root / "11_governance/11_lab-root/WITHHELD_RUNS.yaml", WITHHELD_OK)
    write(root / "11_governance/11_lab-root/AUTORESEARCH.md", AUTORESEARCH_TEXT)
    write(root / "13_reference/12_atoms/INDEX.jsonl", ATOM_LINE + "\n")
    write(root / "31_handoffs/11_lab-handoffs/INDEX.md", HANDOFF_TEXT)
    exp = root / "21_experiments/11_lab-experiments"
    write(exp / "EXP-016_fixture.md", EXP016_TEXT)
    write(exp / "EXP-016_FREEZE_WITNESS.md", WITNESS_TEXT)
    write(exp / "EXP-017_RESULTS.md", RESULTS_TEXT)
    write(exp / "EXP-015_RESULTS_VERIFY-r1.return.yaml", VERIFY_YAML)
    rep = root / "23_reports/11_lab-reports"
    write(rep / "2026-09-02_fixture-prereg.md", "# no frontmatter, dated file name\n")
    write(rep / "15_2026-08-30_paper-position-campaign/pp.md", PP_TEXT)
    write(root / "27_theory/11_lab-theory/H-199.status.md", H199_STATUS_TEXT)
    write(root / "27_theory/11_lab-theory/H-199.md", "---\nid: H-199\ncreated: 2026-09-20\n---\n")
    ar = root / "22_plans/11_lab-plans/20_2026-10-19_autoresearch-rebuild"
    write(ar / "VERIFY-fixture-r1.return.yaml", VERIFY_YAML)
    return root


def build(lab: Path, out: Path) -> dict:
    return bt.build(lab, out)


def read_events(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "timeline_events.jsonl").read_text(encoding="utf-8").splitlines()]


def source_text(lab: Path, lab_rel_path: str) -> str:
    return (lab / lab_rel_path).read_text(encoding="utf-8")


# ----------------------------------------------------------------- date parser

@pytest.mark.parametrize("text, expect", [
    ("2026-06-07", ("2026-06-07", "2026-06-07T00:00:00", "day")),
    ("2026-10-04T02:33Z", ("2026-10-04T02:33", "2026-10-04T02:33:00", "minute")),
    ("2026-09-23T13:11:38Z (UTC)", ("2026-09-23T13:11:38", "2026-09-23T13:11:38", "second")),
    ("2026-10-27 09:12", ("2026-10-27 09:12", "2026-10-27T09:12:00", "minute")),
    ("TBD", None),
    ("2026-13-45", None),
])
def test_date_parser(text, expect):
    got = bt.parse_first_date(text)
    if expect is None:
        assert got is None
    else:
        assert (got[0], got[1], got[2]) == expect


# ----------------------------------------------------------------- id aliases

def test_alias_conversion_and_paper_position_ambiguity(lab: Path):
    tl = bt.Timeline(lab)
    tl.load_aliases()
    tl.reg_exp = {"EXP-001", "EXP-025"}
    inside = tl.resolve("EXP-015 looped", "21_experiments/11_lab-experiments/x.md", "2026-10-01T00:00:00")
    assert inside["exp"] == {"EXP-025"}
    assert inside["aliases"] == {("EXP-015", "EXP-025")}
    assert inside["ambiguous"] == set()

    pp = tl.resolve("EXP-015 label", "23_reports/11_lab-reports/15_2026-08-30_paper-position-campaign/p.md",
                    "2026-10-01T00:00:00")
    assert pp["exp"] == set()
    assert pp["ambiguous"] == {"EXP-015"}

    outside_window = tl.resolve("EXP-015", "21_experiments/11_lab-experiments/x.md", "2026-08-01T00:00:00")
    assert outside_window["exp"] == set()
    assert outside_window["ambiguous"] == {"EXP-015"}

    nseries = tl.resolve("N1-2026-09-05", "13_reference/12_atoms/INDEX.jsonl", "2026-09-05T00:00:00")
    assert nseries["exp"] == {"EXP-024"}
    assert nseries["aliases"] == {("N1-2026-09-05", "EXP-024")}


def _status_token(events, title_prefix):
    hits = [e for e in events if e["source_path"].endswith("EXPERIMENTS.md")
            and e["type"] == "status-change" and e["title"].startswith(title_prefix)]
    assert len(hits) == 1, (title_prefix, hits)
    return hits[0]["status_token"]


def _doc_status(events, title_prefix):
    hits = [e for e in events if e["source_path"].endswith("EXPERIMENTS.md")
            and e["type"] == "status-change" and e["title"].startswith(title_prefix)]
    assert len(hits) == 1, (title_prefix, hits)
    return hits[0]["doc_status_at_read"]


def test_registry_status_and_date_variants(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    reg = [e for e in events if e["source_path"].endswith("EXPERIMENTS.md")]
    exp2 = [e for e in reg if e["title"].startswith("EXP-002")]
    assert [e["type"] for e in exp2] == ["register"], "EXP-002 '**Date**' variant must yield a register event"
    assert exp2[0]["date"] == "2026-06-21"
    # anchored tokens: the later bold / backtick spans must NOT be picked
    # r2 rule: a status_token needs exactly one ISO date in the cell AND the token beside it
    # (same clause). Otherwise the token is doc_status_at_read (read-time, not time-attributed).
    assert _status_token(events, "EXP-001") is None  # token and date in different clauses
    assert _doc_status(events, "EXP-001") == "closed"
    assert _status_token(events, "EXP-006") is None
    assert _doc_status(events, "EXP-006") == "in-progress"
    assert _status_token(events, "EXP-026") is None  # two dates in one cell
    assert _doc_status(events, "EXP-026") == "in-progress"
    assert _status_token(events, "EXP-112") == "RESULTS-COMPLETE-13-OF-13"
    assert _status_token(events, "EXP-025") == "completed (2026-10-03)"
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert any("EXP-002: Status cell has no ISO date" in u[1] for u in stats["unparsed"])
    # EXP-024: its only status date is inside a cited path -> no status event, unparsed
    assert not any(e["title"].startswith("EXP-024") and e["type"] == "status-change" for e in reg)
    assert any("EXP-024: Status cell has no ISO date outside cited paths" in u[1]
               for u in stats["unparsed"])


def test_status_token_is_anchored_not_searched():
    assert bt.status_cell_token("**closed** — x **operationalized as EXP-005**") == "closed"
    assert bt.status_cell_token("in-progress — y **completed 2026-07-02**") == "in-progress"
    assert bt.status_cell_token("enqueued — z `k2b_d0_k{1,2}`") == "enqueued"
    assert bt.status_cell_token("RESULTS-COMPLETE-13-OF-13 (doc frontmatter `status`)") == \
        "RESULTS-COMPLETE-13-OF-13"


def test_design_doc_join_only(lab: Path, tmp_path: Path):
    tl = bt.Timeline(lab)
    tl.load_withheld()
    tl.load_aliases()
    tl.scan_registry()
    design = "21_experiments/11_lab-experiments/EXP-016_fixture.md"
    notes_only = "23_reports/11_lab-reports/fixture-other.md"
    assert tl.join.get(design) == {"EXP-016"}
    assert notes_only not in tl.join


def test_pp_path_stays_raw_ambiguous_in_output(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    pp = [e for e in events if "paper-position" in e["source_path"]]
    assert pp, "paper-position fixture must produce an event"
    assert all(e["ids"]["exp"] == [] for e in pp)


# ----------------------------------------------------------------- withheld

def test_withheld_malformed_is_fail_closed(lab: Path, tmp_path: Path):
    write(lab / "11_governance/11_lab-root/WITHHELD_RUNS.yaml",
          "version: 1\nfamilies:\n  - id: bad\n    status: withheld\n    exp_ids: []\n    patterns: []\n")
    with pytest.raises(SystemExit) as exc:
        build(lab, tmp_path / "out")
    assert exc.value.code == 3


def test_withheld_missing_is_fail_closed(lab: Path, tmp_path: Path):
    (lab / "11_governance/11_lab-root/WITHHELD_RUNS.yaml").unlink()
    with pytest.raises(SystemExit) as exc:
        build(lab, tmp_path / "out")
    assert exc.value.code == 3


def test_withheld_hit_strips_title_and_quote(lab: Path, tmp_path: Path):
    write(lab / "23_reports/11_lab-reports/2026-09-03_secret-note.md",
          "---\ntitle: zzsecretword title\ndate: 2026-09-03\nstatus: DRAFT\n---\n")
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    hit = [e for e in events if e["source_path"].endswith("2026-09-03_secret-note.md")]
    assert hit and all(e["withheld"] and e["title"] is None and e["quote"] is None for e in hit)


# ----------------------------------------------------------------- mtime

def test_mtime_fallback_is_flagged(lab: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    undated = "22_plans/11_lab-plans/undated-autoresearch-fixture"
    write(lab / undated / "NOTE.md", "no dates anywhere in this note\n")
    monkeypatch.setattr(bt, "AR_DIRS", (undated,))
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    mt = [e for e in events if e["date_basis"] == "mtime"]
    assert len(mt) == 1
    assert mt[0]["source_path"].endswith("NOTE.md")
    assert "mtime fallback" in mt[0]["notes"]
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert stats["mtime_basis_events"] == 1


# ----------------------------------------------------------------- verbatim checks

def test_quotes_and_tokens_are_verbatim_from_source(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    assert events
    for e in events:
        if e["withheld"]:
            continue
        if e["quote"] is not None:
            assert len(e["quote"]) <= 200
            assert e["quote"] in source_text(lab, e["source_path"])
        if e["status_token"] is not None:
            assert e["quote"] is not None and e["status_token"] in e["quote"]
            assert e["status_token"] in source_text(lab, e["source_path"])
        if e["doc_status_at_read"] is not None:
            assert e["quote"] is not None and e["doc_status_at_read"] in e["quote"]
            assert e["doc_status_at_read"] in source_text(lab, e["source_path"])


def test_no_invented_tokens(lab: Path, tmp_path: Path):
    """Every status token emitted must come from a verdict/status field in the source."""
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    tokens = {e["status_token"] for e in events if e["status_token"]}
    doc_tokens = {e["doc_status_at_read"] for e in events if e["doc_status_at_read"]}
    corpus = "\n".join(
        p.read_text(encoding="utf-8") for p in lab.rglob("*") if p.is_file() and p.suffix in (".md", ".yaml", ".jsonl")
    )
    for tok in tokens | doc_tokens:
        assert tok in corpus, f"token not present in any fixture source: {tok!r}"
    # result-time judged_at: the judge event records the status as read (doc_status_at_read)
    assert "VERIFIED-r1-PASS-2026-10-04" in doc_tokens
    assert "VERIFIED-r1-PASS-2026-10-04" not in tokens
    assert "PASS" in tokens  # verify return: its own verdict line
    assert "fabricated-verdict" not in tokens | doc_tokens


def test_frontmatter_status_is_doc_status_not_prereg_token(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    ev = [e for e in events if e["source_path"].endswith("EXP-016_fixture.md") and e["type"] == "prereg"]
    assert ev, "date_designed must produce a prereg event"
    assert ev[0]["status_token"] is None  # prereg does not record a status change
    assert ev[0]["doc_status_at_read"] == "pre-registered-unrun"
    assert ev[0]["quote"] == "status: pre-registered-unrun # FROZEN at registration"


def test_witness_lines_and_types(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    w = [e for e in events if e["source_path"].endswith("EXP-016_FREEZE_WITNESS.md")]
    types = sorted(e["type"] for e in w)
    assert types == ["enqueue", "freeze", "witness"]
    assert all(e["ids"]["exp"] == ["EXP-016"] for e in w)
    assert all(e["status_token"] is None for e in w)


def test_handoff_and_atoms_and_ledger_rows(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    hand = [e for e in events if e["type"] == "handoff"]
    # r2 rule: handoff bold word is the row's status as read, not a time-attributed token
    assert len(hand) == 1 and hand[0]["status_token"] is None
    assert hand[0]["doc_status_at_read"] == "in-progress"
    assert hand[0]["date"] == "2026-10-27T09:12"
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert any("handoff row tag [PPC] outside ARG/ARB scope" in u[1] for u in stats["unparsed"])
    atom = [e for e in events if e["ids"]["atoms"] == ["CL-0001"]]
    assert atom and atom[0]["status_token"] is None  # register type: verdict is doc_status only
    assert atom[0]["doc_status_at_read"] == "partial"
    assert atom[0]["ids"]["exp"] == ["EXP-024", "EXP-112"]
    assert atom[0]["aliases_seen"] == [{"raw": "N1-2026-09-05", "canonical": "EXP-024"}]


# ----------------------------------------------------------------- determinism

def test_rebuild_is_byte_identical(lab: Path, tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    build(lab, a)
    build(lab, b)
    for name in ("timeline_events.jsonl", "timeline_stats.json", "timeline_copy_events.jsonl",
                 "timeline_no_event_files.txt", "timeline_copy_files.txt"):
        ha = hashlib.sha256((a / name).read_bytes()).hexdigest()
        hb = hashlib.sha256((b / name).read_bytes()).hexdigest()
        assert ha == hb, name


def test_event_ids_unique_and_sorted(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    events = read_events(out)
    ids = [e["event_id"] for e in events]
    assert len(ids) == len(set(ids))
    keys = [(bt._norm_key(e), e["event_id"]) for e in events]
    assert keys == sorted(keys)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------- copy separation

@pytest.mark.parametrize("rel,reason_part", [
    ("23_reports/11_lab-reports/x/benchmark/EXP-015_RESULTS.v_r1.md", "benchmark"),
    ("23_reports/11_lab-reports/x/transcript_extracts/a.md", "transcript"),
    ("23_reports/11_lab-reports/x/volatile_sources_2026-10-14/a.md", "volatile"),
    ("23_reports/11_lab-reports/x/scratchpad_w4fix/before/w6/a.md", "scratchpad"),
    ("23_reports/11_lab-reports/x/scratchpad_w4fix/after/a.md", "scratchpad"),
    ("23_reports/11_lab-reports/x/backup_pre_retrofit/EXP-001_x.md", "backup"),
    ("23_reports/11_lab-reports/x/foo_snapshot/a.md", "snapshot"),
    ("23_reports/11_lab-reports/x/recovered/a.md", "recovered"),
    ("23_reports/11_lab-reports/x/01_RESULTS.r2-pass-copy.md", "copy"),
    ("23_reports/11_lab-reports/x/x_copy.md", "copy"),
    ("23_reports/11_lab-reports/x/x.copy.md", "copy"),
])
def test_is_copy_path_patterns(rel: str, reason_part: str):
    reason = bt.is_copy_path(rel)
    assert reason is not None and reason_part in reason


@pytest.mark.parametrize("rel", [
    # false-positive guards: canonical-looking files outside copy trees
    "21_experiments/11_lab-experiments/EXP-017_RESULTS.md",
    "21_experiments/11_lab-experiments/EXP-015_FREEZE_WITNESS.md",
    "21_experiments/11_lab-experiments/EXP-016_act-deep-supervision-axis_VERIFY-r1.return.yaml",
    # an _impl directory without a VERIFY return is canonical
    "23_reports/11_lab-reports/x/pc1_impl/notes.md",
    # r2 S3(b): original verifier returns in *_impl are canonical (cited by FREEZE_WITNESS files)
    "23_reports/11_lab-reports/x/pc1_impl/VERIFY-EXP022-FREEZE-r1.return.yaml",
    "23_reports/11_lab-reports/x/pc1_impl/VERIFY-PC1R9-r1.return.yaml",
    # 'copy' must be a whole name token, not a substring; a directory named copy-* is not a copy
    "23_reports/11_lab-reports/x/copying-notes.md",
    "23_reports/11_lab-reports/40_fixture-copy-check/01_RESULTS.md",
])
def test_is_copy_path_false_positive_guards(rel: str):
    assert bt.is_copy_path(rel) is None


def test_copies_go_to_separate_file(lab: Path, tmp_path: Path):
    write(lab / "23_reports/11_lab-reports/2026-10-25_fixture-bench/benchmark/EXP-017_RESULTS.v_r1.md",
          "---\njudged_at: 2026-10-25T10:00Z\n---\nbody\n")
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    copies = read_jsonl(out / "timeline_copy_events.jsonl")
    assert copies and all(e["is_copy"] is True and e["copy_reason"] for e in copies)
    assert all("is_copy" not in e and "copy_reason" not in e for e in canon)
    assert not any("benchmark/" in e["source_path"] for e in canon)
    assert any("benchmark/" in e["source_path"] for e in copies)
    # canonical EXP-017_RESULTS.md is not a copy (false-positive guard in the real pipeline)
    assert any(e["source_path"].endswith("EXP-017_RESULTS.md") and "is_copy" not in e for e in canon)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert stats["copy_events_total"] == len(copies)
    assert stats["events_total"] == len(canon)
    assert "benchmark" in (out / "timeline_copy_files.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------- frontmatter ids, dates

def test_frontmatter_exp_id_feeds_filename_fallback(lab: Path, tmp_path: Path):
    rep = lab / "23_reports/11_lab-reports"
    write(rep / "2026-10-23_fixture-ltf_RESULTS.md", "---\nexp_id: EXP-016\n---\nbody\n")
    write(rep / "2026-10-24_fixture-verify_VERIFY-r1.return.yaml", "model: x\nexp_id: EXP-016\nhypothesis: H-200\n")
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    for name in ("2026-10-23_fixture-ltf_RESULTS.md", "2026-10-24_fixture-verify_VERIFY-r1.return.yaml"):
        ev = [e for e in canon if e["source_path"].endswith(name)]
        assert ev, name
        assert ev[0]["date_basis"] == "filename"
        assert ev[0]["ids"]["exp"] == ["EXP-016"], name


def test_result_time_key_takes_precedence_over_date(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    ev = [e for e in canon if e["source_path"].endswith("EXP-017_RESULTS.md")]
    assert len(ev) == 1, "date: 2026-10-03 is superseded by judged_at, not emitted"
    assert ev[0]["type"] == "judge"
    assert ev[0]["date"].startswith("2026-10-04")
    assert ev[0]["date_tz"] == "Z"
    assert "superseded" in ev[0]["notes"]
    # judge is not a self-recording status type: the read-time status stays in doc_status_at_read
    assert ev[0]["status_token"] is None
    assert ev[0]["doc_status_at_read"] == "VERIFIED-r1-PASS-2026-10-04"


def test_registry_own_section_is_not_ambiguous_for_its_own_canonical(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    own = [e for e in canon if e["source_path"].endswith("EXPERIMENTS.md") and "EXP-019" in e["ids"]["exp"]]
    assert own, "EXP-019 registry section must keep its own canonical"
    assert all("EXP-019" not in e["ids_ambiguous"] for e in own)


def test_ambiguous_registry_canonical_alias_token(lab: Path, tmp_path: Path):
    write(lab / "23_reports/11_lab-reports/2026-10-25_fixture-n4-note.md",
          "---\ndate: 2026-10-25\nexp_id: EXP-019\n---\nbody\n")
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    ev = [e for e in canon if e["source_path"].endswith("2026-10-25_fixture-n4-note.md")]
    assert ev
    assert ev[0]["ids_ambiguous"] == ["EXP-019"]
    assert ev[0]["ids"]["exp"] == []
    assert not any(a["raw"] == "EXP-019" for a in ev[0]["aliases_seen"])


@pytest.mark.parametrize("text,tz", [
    ("2026-10-04T02:33Z", "Z"),
    ("2026-10-04T02:33+09:00", "+09:00"),
    ("2026-10-19 10:00 KST", "KST"),
    ("2026-10-19 UTC", "UTC"),
    ("2026-10-19", None),
])
def test_date_tz_preserved(text: str, tz):
    assert bt.parse_first_date(text)[3] == tz


def test_nested_date_key_goes_to_unparsed_not_events(lab: Path, tmp_path: Path):
    rel = "21_experiments/11_lab-experiments/EXP-016_nested_VERIFY-r1.return.yaml"
    write(lab / rel, "identity:\n  date_utc: 2026-09-23\n  exp_id: EXP-016\n")
    out = tmp_path / "out"
    build(lab, out)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert any(u[0].startswith(rel + ":") and "nested date-like key 'date_utc'" in u[1]
               for u in stats["unparsed"])
    assert not any(e["source_path"] == rel for e in read_jsonl(out / "timeline_events.jsonl"))
    no_ev = (out / "timeline_no_event_files.txt").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith(rel + "\tunknown-date-key") for line in no_ev)


# ----------------------------------------------------------------- witness and no-event accounting

def test_no_event_accounting(lab: Path, tmp_path: Path):
    write(lab / "23_reports/11_lab-reports/notes-plain.md", "# plain, no frontmatter, no date\n")
    out = tmp_path / "out"
    build(lab, out)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    lines = (out / "timeline_no_event_files.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == stats["files_no_event_total"]
    assert stats["files_read_total"] == stats["files_with_event_total"] + stats["files_no_event_total"]
    assert any(line.startswith("23_reports/11_lab-reports/notes-plain.md\tno-frontmatter") for line in lines)
    assert stats["no_event_reason_counts"].get("no-frontmatter", 0) >= 1
    assert len(stats["source_tree_hash"]) == 64
    assert stats["per_group_files"]["reports"]["no_event"] >= 1


# ----------------------------------------------------------------- r2 regressions

def test_impl_verify_return_is_canonical(lab: Path, tmp_path: Path):
    """Original verifier returns inside *_impl are canonical (cited by FREEZE_WITNESS files)."""
    rel = "23_reports/11_lab-reports/2026-10-21_fixture-impl/pc1_impl/VERIFY-EXP022-FREEZE-r1.return.yaml"
    write(lab / rel, "model: x\ndate: 2026-10-23\nverdict: PASS\n")
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    copies = read_jsonl(out / "timeline_copy_events.jsonl")
    ev = [e for e in canon if e["source_path"] == rel]
    assert len(ev) == 1 and ev[0]["type"] == "verify" and ev[0]["status_token"] == "PASS"
    assert not any(e["source_path"] == rel for e in copies)


@pytest.mark.parametrize("rel,is_copy", [
    ("40_x/01_RESULTS.r2-pass-copy.md", True),
    ("40_x/01_RESULTS.md", False),
    ("40_x/copying-notes.md", False),
    ("40_x/EXP-017_RESULTS.md", False),
])
def test_copy_token_is_whole_file_name_token(rel: str, is_copy: bool):
    assert (bt.is_copy_path(rel) is not None) is is_copy


def test_copy_marked_file_separated_and_guard_names_stay(lab: Path, tmp_path: Path):
    d = "23_reports/11_lab-reports/40_fixture-copy-check"
    write(lab / d / "01_RESULTS.r2-pass-copy.md", "---\ndate: 2026-10-13\nstatus: DRAFT-PENDING-VERIFIER\n---\nb\n")
    write(lab / d / "01_RESULTS.md", "---\ndate: 2026-10-13\nstatus: VERIFIED\n---\nb\n")
    write(lab / d / "copying-notes.md", "---\ndate: 2026-10-13\n---\nb\n")
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    copies = read_jsonl(out / "timeline_copy_events.jsonl")
    assert any(e["source_path"].endswith("/01_RESULTS.md") for e in canon)
    assert any(e["source_path"].endswith("/copying-notes.md") for e in canon)
    assert not any(e["source_path"].endswith("r2-pass-copy.md") for e in canon)
    assert any(e["source_path"].endswith("r2-pass-copy.md") and "copy" in e["copy_reason"] for e in copies)


def test_registry_status_token_only_with_one_date_beside_it(lab: Path, tmp_path: Path):
    """Pure-function check of the r2 clause rule (same segment as the single date)."""
    assert bt.token_in_date_segment("completed (2026-10-03)", "**completed (2026-10-03)** — judged", "2026-10-03")
    assert not bt.token_in_date_segment("closed", "closed — postmortem 2026-06-23", "2026-06-23")
    # two distinct dates in one cell: the caller refuses the token; the helper alone can still
    # find the date segment, so the cell-level count is what excludes it (tested via EXP-026)


def test_results_run_date_status_is_doc_status(lab: Path, tmp_path: Path):
    rel = "23_reports/11_lab-reports/2026-10-27_fixture-run_RESULTS.md"
    write(lab / rel, "---\ntitle: fixture run\nrun_date: 2026-10-27\nstatus: DRAFT r2\n---\nbody\n")
    out = tmp_path / "out"
    build(lab, out)
    ev = [e for e in read_jsonl(out / "timeline_events.jsonl") if e["source_path"] == rel]
    assert len(ev) == 1 and ev[0]["type"] == "results" and ev[0]["date"].startswith("2026-10-27")
    assert ev[0]["status_token"] is None
    assert ev[0]["doc_status_at_read"] == "DRAFT r2"


def test_verify_result_time_keys_audit_date_date_utc_audited_at(lab: Path, tmp_path: Path):
    d = "23_reports/11_lab-reports/2026-10-28_fixture-keys"
    write(lab / d / "VERIFY-K1-r1.return.yaml", "model: x\naudit_date: 2026-10-23\nverdict: PASS\n")
    write(lab / d / "VERIFY-K2-r1.return.yaml", "model: x\ndate_utc: 2026-10-24\nverdict: FAIL\n")
    write(lab / d / "VERIFY-K3-r1.return.yaml", "model: x\naudited_at: 2026-10-25T10:00Z\nverdict: PASS\n")
    out = tmp_path / "out"
    build(lab, out)
    canon = read_jsonl(out / "timeline_events.jsonl")
    want = {"VERIFY-K1-r1.return.yaml": ("2026-10-23", "PASS"),
            "VERIFY-K2-r1.return.yaml": ("2026-10-24", "FAIL"),
            "VERIFY-K3-r1.return.yaml": ("2026-10-25T10:00", "PASS")}
    for name, (date, tok) in want.items():
        ev = [e for e in canon if e["source_path"].endswith(name)]
        assert len(ev) == 1, name
        assert ev[0]["type"] == "verify" and ev[0]["date"] == date and ev[0]["status_token"] == tok, name


def test_unknown_key_only_file_gets_unknown_reason(lab: Path, tmp_path: Path):
    rel = "21_experiments/11_lab-experiments/EXP-016_unknownkey_VERIFY-r1.return.yaml"
    write(lab / rel, "model: x\nobserved_at: 2026-10-31\nverdict: PASS\n")
    out = tmp_path / "out"
    build(lab, out)
    no_ev = (out / "timeline_no_event_files.txt").read_text(encoding="utf-8").splitlines()
    assert any(line == rel + "\tunknown-date-key (see unparsed)\texperiments" for line in no_ev)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert any(u[0].startswith(rel + ":") and "observed_at" in u[1] for u in stats["unparsed"])


def test_path_embedded_date_is_not_a_date_value(lab: Path, tmp_path: Path):
    rel = "23_reports/11_lab-reports/2026-10-28_fixture-pathdate_RESULTS.md"
    write(lab / rel, "---\njudged: EXP-016_artifacts/stepC_2026-10-19/judged.json (sha256 abc)\n"
                     "date: 2026-10-28\n---\nbody\n")
    out = tmp_path / "out"
    build(lab, out)
    ev = [e for e in read_jsonl(out / "timeline_events.jsonl") if e["source_path"] == rel]
    assert len(ev) == 1 and ev[0]["type"] == "results" and ev[0]["date"].startswith("2026-10-28")
    assert not any(e["type"] == "judge" and e["source_path"] == rel for e in ev)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert any(u[0].startswith(rel + ":") and u[1].startswith("judged: no ISO date") for u in stats["unparsed"])


def test_freeze_typing_negation_and_reference(lab: Path, tmp_path: Path):
    rel = "21_experiments/11_lab-experiments/EXP-016_TYPING_FREEZE_WITNESS.md"
    write(lab / rel, "# typing\n"
          "not frozen yet, witness 2026-09-23T10:00Z\n"
          "after the freeze time check 2026-09-23T10:05Z\n"
          "FREEZE manifest re-hash 2026-09-23T10:06Z\n"
          "see `EXP-016_FREEZE.sha256` note 2026-09-23T10:07Z\n"
          "freeze 2026-09-23T10:10Z\n")
    out = tmp_path / "out"
    build(lab, out)
    ev = [e for e in read_jsonl(out / "timeline_events.jsonl") if e["source_path"] == rel]
    by_line = {int(e["date_source"].rsplit(":", 1)[1]): e["type"] for e in ev}
    assert by_line == {2: "witness", 3: "witness", 4: "witness", 5: "witness", 6: "freeze"}


def test_source_tree_hash_changes_with_input(lab: Path, tmp_path: Path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    build(lab, a)
    build(lab, b)
    ha = json.loads((a / "timeline_stats.json").read_text(encoding="utf-8"))["source_tree_hash"]
    hb = json.loads((b / "timeline_stats.json").read_text(encoding="utf-8"))["source_tree_hash"]
    assert ha == hb  # same tree -> same hash
    write(lab / "21_experiments/11_lab-experiments/EXP-016_fixture.md", EXP016_TEXT + "\nextra line\n")
    build(lab, c)
    hc = json.loads((c / "timeline_stats.json").read_text(encoding="utf-8"))["source_tree_hash"]
    assert hc != ha  # one changed input -> different hash


def test_same_day_mixed_precision_stat_present(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert isinstance(stats["same_day_mixed_precision_groups"], int)
    assert stats["same_day_mixed_precision_groups"] >= 0


def test_handoff_non_arg_tag_still_unparsed(lab: Path, tmp_path: Path):
    out = tmp_path / "out"
    build(lab, out)
    stats = json.loads((out / "timeline_stats.json").read_text(encoding="utf-8"))
    assert any("handoff row tag [PPC] outside ARG/ARB scope" in u[1] for u in stats["unparsed"])
