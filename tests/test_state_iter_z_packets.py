"""Fixture tests for tools/state_iter_z_synthesis/packets.py and the S2/S3 template set (PREREG sha 87fa6e7a...3e1b).

No real S1 table is used (none exists; PREREG §12 step 3 witnesses these templates before any value extraction).
The S1 fixtures are synthetic: every token, scope text, run name and value is invented. No network, no API key.
Run:  CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_state_iter_z_packets.py -q
"""
from __future__ import annotations

import copy
import csv
import io
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.state_iter_z_synthesis import packets as P  # noqa: E402
from tools.state_iter_z_synthesis import s1 as S1  # noqa: E402  (the producer: column names are checked against it)

FIX = REPO / "tests/fixtures/state_iter_z_synthesis"
TABLE = (FIX / "S1_table.fixture.csv").read_text(encoding="utf-8")
CELLS = (FIX / "S1_cells.fixture.csv").read_text(encoding="utf-8")
TPL = P.TEMPLATES_DIR
HEADER = TPL / "shared/siz_rules_header.md"
MAP_SYS = TPL / "shared/siz_map_system.md"
RED_SYS = TPL / "shared/siz_reduce_system.md"
MAP_SCHEMA = json.loads((TPL / "schemas/siz_review_note.json").read_text(encoding="utf-8"))
RED_SCHEMA = json.loads((TPL / "schemas/siz_reduce_note.json").read_text(encoding="utf-8"))
S0_EXPECTED = P.LAB_ROOT / P.SYNTH_REL / "S0r2_cells_expected.csv"          # S0r2 supersedes S0 r1
S0_REPRO = P.LAB_ROOT / P.SYNTH_REL / "S0r2_class_repro_check.csv"
ALLOWED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum"}


def prereg_text():
    if not P.PREREG_PATH.is_file():
        pytest.skip("frozen PREREG not reachable (lab/ is the NAS symlink)")
    return P.PREREG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mp():
    return {p["cell_id"]: p for p in P.build_map_packets(TABLE, CELLS)}


@pytest.fixture(scope="module")
def rp():
    return {p["q"]: p for p in P.build_reduce_packets(CELLS)}


def _rows(text):
    comments, body = P.split_comments(text)
    rd = csv.DictReader(io.StringIO(body))
    return comments, list(rd.fieldnames), list(rd)


def _write(comments, cols, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return "".join(c + "\n" for c in comments) + buf.getvalue()


def _set(text, match, changes):
    """Change the first row matching `match` (column -> value); column order independent."""
    comments, cols, rows = _rows(text)
    r = next(r for r in rows if all(r[k] == v for k, v in match.items()))
    r.update(changes)
    return _write(comments, cols, rows)


def _rename(text, old, new):
    comments, cols, rows = _rows(text)
    return _write(comments, [new if c == old else c for c in cols], [{(new if k == old else k): v for k, v in r.items()}
                                                                    for r in rows])


FX_CELLS = {(r["q"], r["s"]): r for r in _rows(CELLS)[2]}
N_TARGETS = sum(r["s1_class"] in P.S2_CLASSES for r in FX_CELLS.values())       # computed from S1_cells, not fixed


def fixed_inputs(packet_text, system=MAP_SYS):
    return [HEADER.read_text(encoding="utf-8"), system.read_text(encoding="utf-8"), packet_text]


# ------------------------------------------------------------------------------------------ PREREG binding
def test_prereg_sha_and_fixed_strings_are_verbatim():
    t = prereg_text()
    assert P.sha256_text(t) == P.PREREG_SHA256
    assert P.r3_text(t) == P.R3_TEXT
    flat = P.ws(t)
    for s in (P.LEGACY_NOTE, P.LOCKED_MARKER, P.MARGINAL, P.PROTOCOL_MIX, P.UNKNOWN_KEY, P.R8_CAP, P.H_STATUS,
              P.H_STATUS_RULE, P.PREJUDGE_ONLY, P.EXP025_FIRST, P.R3_II_PREFIX.split("(")[0], P.R3_II_SUFFIX):
        assert P.ws(s) in flat, s


def test_header_fixed_strings_are_literals_and_s0_is_complete():
    """W1: no template placeholder in the FIXED STRINGS block; every value there is PREREG text. W2: every §0 row
    (symbols used by the kept excerpts: S0, S1, as_of, G3/G4, S6 ...) is reproduced verbatim."""
    t, h = prereg_text(), HEADER.read_text(encoding="utf-8")
    fixed = h.split("## PREREG EXCERPT")[0].split("## FIXED STRINGS")[1]
    assert "<" not in fixed and ">" not in fixed
    flat = P.ws(t)
    for ln in fixed.splitlines()[1:]:
        if ln.startswith("- "):
            assert P.ws(ln.split(": ", 1)[1]) in flat, ln
    assert P.H_STATUS_RULE in h
    s0 = P._block(P._lines(t), "## 0.", lambda l: l.startswith("## "))
    assert "\n".join(s0).strip("\n") in h                     # the whole §0 block, every row, in order
    for label in ("as_of", "G1–G4", "S6", "S0–S4, V0–V2, G6(단계)", "cap_batch / cap_day / cap_month"):
        assert any(ln.startswith(f"| **{label}**") for ln in h.split("\n")), label


def test_prereg_mismatch_refused():
    with pytest.raises(P.PacketError, match="sha256"):
        P.build_rules_header("not the prereg")


def test_rules_header_is_the_committed_file():
    assert P.build_rules_header(prereg_text()) == HEADER.read_text(encoding="utf-8")


def test_universe_matches_s0r2_and_fixture_classes():
    assert len(P.UNIVERSE_CELLS) == 58 and len(set(P.UNIVERSE_CELLS)) == 58
    if not (S0_EXPECTED.is_file() and S0_REPRO.is_file()):
        pytest.skip("S0r2 files not reachable")
    _c, rows, _m = P.parse_csv(S0_EXPECTED.read_text(encoding="utf-8"), "S0r2", ("q", "s", "expected_class"))
    assert [(r["q"], r["s"]) for r in rows] == list(P.UNIVERSE_CELLS)
    assert {k: r["expected_class"] for k, r in FX_CELLS.items()} == {(r["q"], r["s"]): r["expected_class"] for r in rows}
    _c, rep, _m = P.parse_csv(S0_REPRO.read_text(encoding="utf-8"), "S0r2", ("q", "s", "as_of_class"))
    conf = {(r["q"], r["s"]): r["as_of_class"] for r in rep}
    assert {k: r["s1_class"] for k, r in FX_CELLS.items()} == conf          # fixture = S0r2 confirmed classes
    counts = {}
    for c in conf.values():
        counts[c] = counts.get(c, 0) + 1
    assert counts == S1.EXPECTED_CLASS_COUNTS["S0r2"]
    assert N_TARGETS == sum(v for c, v in counts.items() if c in P.S2_CLASSES) == 48


def test_fixtures_and_contract_follow_the_s1_producer():
    """Consumer adopts the producer's names: fixtures are in s1.py column order and format, and every column
    packets.py requires or emits is an s1.py column (a renamed s1 column breaks this test)."""
    for text, fields in ((CELLS, S1.CELL_FIELDS), (TABLE, S1.TABLE_FIELDS)):
        comments, cols, rows = _rows(text)
        assert cols == list(fields)
        assert "".join(c + "\n" for c in comments) + S1.to_csv_text(fields, rows) == text
    assert set(P.S1_CELLS_REQUIRED) <= set(S1.CELL_FIELDS) and set(P.PACKET_CELL_FIELDS) <= set(S1.CELL_FIELDS)
    assert set(P.INHERITED_COLUMNS) <= set(P.S1_CELLS_REQUIRED)
    assert set(P.S1_TABLE_REQUIRED) <= set(S1.TABLE_FIELDS) and set(P.PACKET_TABLE_FIELDS) <= set(S1.TABLE_FIELDS)
    assert "n_table_rows" not in P.PACKET_CELL_FIELDS
    roles = [role for role, _c, _m in P.S1_CELLS_MAP]
    assert len(set(roles)) == len(roles) and not (set(roles) - {c for _r, c, _m in P.S1_CELLS_MAP}) & set(S1.CELL_FIELDS)
    assert {r["q"]: r["q_token"] for r in FX_CELLS.values()} == S1.q_tokens(
        {q: [r["s1_class"] for r in FX_CELLS.values() if r["q"] == q] for q in P.QUESTIONS})
    assert set(S1.NO_VALUE_CLASSES) == set(P.NO_VALUE_CLASSES) and set(S1.VALUE_CLASSES) == {"A-DESC", "D-LOCKED"}


# ------------------------------------------------------------------------------------------ map packets
def test_map_packet_set(mp):
    assert len(mp) == N_TARGETS                                             # computed from S1_cells (S0r2: 48)
    assert sum(p["mode"] == "LOCKED" for p in mp.values()) == sum(
        r["s1_class"] in P.LOCKED_CLASSES for r in FX_CELLS.values())        # S0r2: D-LOCKED 5 + D-PREJUDGE 3
    assert {p["cell_class"] for p in mp.values()} == set(P.S2_CLASSES)
    for cid in mp:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,60}", cid)
    assert "Q-Z7__CL-0006" not in mp and "Q-I2__EXP-012_sealed" not in mp and "Q-I2__EXP-010" not in mp
    assert "Q-I2__EXP-011" not in mp and "Q-I4__EXP-011" not in mp          # S0r2 known mismatches: X-ABSENT


def test_map_packet_obligations(mp):
    t = mp["Q-I2__EXP-001"]["text"]
    assert P.LEGACY_NOTE in t and "[R-3] required" in t and P.R3_TEXT in t
    assert "missing noz_noiter" in t and P.MARGINAL in t and P.UNKNOWN_KEY in t
    t = mp["Q-I2__EXP-012_probe"]["text"]
    assert "2x2 presence clean-D0 / block tf / k 5: present z_iter, z_noiter, noz_iter, noz_noiter; complete" in t
    assert "clean-D0/tf/trm/(3,6)/5: 1, 2" in t and "[R-12]" not in t
    assert "[R-12] protocols present: P-ALL, P-PROBE" in mp["Q-Z1__EXP-005"]["text"]
    t = mp["Q-Z1__EXP-012_probe"]["text"]
    assert "[R-5]" in t and "k 9: Tk_class Tk-fx-e" in t and "k 11: Tk_class Tk-fx-f" in t
    t = mp["Q-I3__EXP-025"]["text"]
    assert "[R-3/R-7]" in t and P.EXP025_FIRST in t
    t = mp["Q-I4__EXP-016"]["text"]
    assert "mode: LOCKED" in t and P.LOCKED_MARKER in t and "[§6.2] D-LOCKED" in t
    assert P.PREJUDGE_ONLY in mp["Q-I4__EXP-020"]["text"]
    assert "Q-DESC-ONLY" in mp["Q-Z4__EXP-006"]["text"] and P.LOCKED_MARKER in mp["Q-Z4__EXP-006"]["text"]
    t = mp["Q-I1__CL-0003"]["text"]
    assert "token_V: supported" in t and "atom_verdict_scope: FIXTURE scope text" in t and "token_N: (none)" in t
    assert P.H_STATUS in mp["Q-Z1__H-A"]["text"]
    assert "[§5.2] figure atom" in mp["Q-Z5__FG-0003"]["text"]
    assert "atom ceiling: §5.2 row for CL-0003" in mp["Q-I1__CL-0003"]["text"]


def test_map_packets_carry_only_their_own_rows(mp):
    _c, rows, _m = P.parse_csv(TABLE, "t", P.S1_TABLE_REQUIRED)
    for cid, p in mp.items():
        own = {r["row_id"] for r in rows if P.cell_id(r["q"], r["s"]) == cid and r["evidence_use"] == "yes"}
        assert set(p["citable_ids"]) == own | {f"CELL:{cid}"}
        for r in rows:
            if r["row_id"] not in own:
                assert f"\n{r['row_id']}," not in p["text"]


def test_build_is_deterministic(tmp_path):
    a = [p["text"] for p in P.build_map_packets(TABLE, CELLS)]
    b = [p["text"] for p in P.build_map_packets(TABLE, CELLS)]
    assert a == b
    for d in ("x", "y"):
        assert P.main(["map", "--s1-table", str(FIX / "S1_table.fixture.csv"), "--s1-cells",
                       str(FIX / "S1_cells.fixture.csv"), "--out", str(tmp_path / d)]) == 0
    assert (tmp_path / "x/packets.sha256").read_bytes() == (tmp_path / "y/packets.sha256").read_bytes()
    assert len((tmp_path / "x/index.csv").read_text().splitlines()) == N_TARGETS + 1


def test_r3_variant_ii():
    t = P.r3_text_variant("lab/x/EXP-011_module-ablation-sigma-k-new.md:46")
    assert t.startswith(P.R3_TEXT.split(". 이 합성이 쓰는 소스")[0]) and P.R3_II_SUFFIX in t and "md:46" in t


@pytest.mark.parametrize("which,match,changes,msg", [
    ("t", {"row_id": "F001"}, {"cohort_note": "legacy"}, "cohort_note"),
    ("t", {"row_id": "F010"}, {"cohort_note": "note"}, "non-legacy"),
    ("t", {"row_id": "F031"}, {"value": "0.5"}, "D-PREJUDGE"),
    ("t", {"row_id": "F001"}, {"value": "0.3"}, "UNKNOWN"),
    ("t", {"row_id": "F060"}, {"s": "EXP-012/sealed", "cell_class": "D-SEALED"}, "D-SEALED"),
    ("t", {"row_id": "F061"}, {"row_id": "F060"}, "duplicate"),
    ("t", {"row_id": "F050"}, {"s": "EXP-099"}, "not an S1_cells cell"),
    ("t", {"row_id": "F060"}, {"run_name": "fx/sealed/k9_s1"}, "forbidden-path"),
    ("t", {"row_id": "F001"}, {"cell_class": "A-VERDICT"}, "cell_class"),
    ("t", {"row_id": "F001"}, {"evidence_use": "maybe"}, "evidence_use"),
    ("t", {"row_id": "F001"}, {"evidence_use": ""}, "evidence_use"),
    ("c", {"q": "Q-Z7", "s": "CL-0006"}, {"s": "CL-0099"}, "universe"),
    ("c", {"q": "Q-I3", "s": "EXP-023"}, {"q_token": "Q-NONE"}, "q_token"),
    ("c", {"q": "Q-I1", "s": "CL-0003"}, {"s1_class": "VERDICT", "s0_confirmed_class": "VERDICT"}, "class"),
    ("c", {"q": "Q-I1", "s": "CL-0003"}, {"s0_confirmed_class": "A-DESC"}, "S1 gate"),
    # W4: non-value fields in no-value cells (PREREG §6.3 / §7.1 갈래 1-5)
    ("t", {"row_id": "F031"}, {"step": "122050"}, "a step in a D-PREJUDGE"),
    ("t", {"row_id": "F031"}, {"label_GC": "G-label"}, "label_GC 'G-label'"),
    ("c", {"q": "Q-I2", "s": "EXP-010"}, {"token_V": "supported"}, "verdict word"),
    ("c", {"q": "Q-I3", "s": "EXP-023"}, {"token_V": "partial"}, "verdict word"),
    ("c", {"q": "Q-I2", "s": "EXP-012/sealed"}, {"atom_verdict_scope": "scope"}, "scope text"),
    ("c", {"q": "Q-I2", "s": "EXP-010"}, {"token_scope": "scope"}, "scope text"),
])
def test_fail_closed(which, match, changes, msg):
    t, c = (_set(TABLE, match, changes), CELLS) if which == "t" else (TABLE, _set(CELLS, match, changes))
    with pytest.raises(P.PacketError, match=msg):
        P.build_map_packets(t, c)
    with pytest.raises(P.PacketError, match=msg):
        P.s4_slices(t, c)
    if which == "c":
        with pytest.raises(P.PacketError):           # the reduce path validates S1_cells rows the same way
            P.build_reduce_packets(c)


@pytest.mark.parametrize("which,old", [("t", "value"), ("t", "evidence_use"), ("t", "cell_class"), ("c", "s1_class"),
                                       ("c", "token_V"), ("c", "atom_verdict_scope"), ("c", "s0_confirmed_class")])
def test_missing_s1_column_is_refused_not_synthesised(which, old):
    """A column of the s1.py contract that is absent is refused; packets.py never derives it from another column."""
    t, c = (_rename(TABLE, old, old + "_x"), CELLS) if which == "t" else (TABLE, _rename(CELLS, old, old + "_x"))
    with pytest.raises(P.PacketError, match="missing required column"):
        P.build_map_packets(t, c)
    if which == "c":
        with pytest.raises(P.PacketError, match="missing required column"):
            P.build_reduce_packets(c)


def test_label_gc_none_and_value_classes_still_pass():
    t = _set(TABLE, {"row_id": "F031"}, {"label_GC": ""})        # empty label_GC is NONE-equivalent
    assert len(P.build_map_packets(t, CELLS)) == N_TARGETS


def test_packet_columns_are_allow_listed(mp):
    """W4: only PACKET_TABLE_FIELDS / PACKET_CELL_FIELDS reach a packet or an S4 slice; other S1 columns (run_name,
    wandb ids, provenance paths, n_table_rows ...) are dropped, and the whole S1 text is still scanned."""
    comments, cols, rows = _rows(TABLE)
    assert {"run_name", "wandb_run_id", "provenance_file"} <= set(cols) - set(P.PACKET_TABLE_FIELDS)
    assert {"note", "protocol_note"} <= set(P.PACKET_TABLE_FIELDS)
    for r in rows:
        r.update(wandb_run_id="SENTINEL-WR", provenance_file="SENTINEL-PF")
    extra = _write(comments, cols, rows)
    pk = P.build_map_packets(extra, CELLS)
    s4 = P.s4_slices(extra, CELLS)
    texts = [p["text"] for p in pk] + [v for d in s4.values() for v in d.values()]
    for r in rows:
        assert all(r["run_name"] not in x for x in texts), r["run_name"]
    assert not any("SENTINEL" in x for x in texts)
    for p in pk:
        for ln in p["text"].splitlines():
            if ln.startswith("row_id,"):
                assert set(ln.split(",")) <= set(P.PACKET_TABLE_FIELDS) - set(P.TABLE_IDENT)
            if ln.startswith("q,s,kind,"):
                assert set(ln.split(",")) <= set(P.PACKET_CELL_FIELDS) and "n_table_rows" not in ln
    with pytest.raises(P.PacketError, match="forbidden-path"):      # dropped columns are scanned all the same
        P.build_map_packets(_set(extra, {"row_id": "F001"}, {"provenance_file": "see preds"}), CELLS)


def test_row_note_reaches_packets_and_s4(mp):
    """VERIFY-SIZ-C W-C1: the S1_table `note` (s1.py disclosure text) is carried into map packets and S4 slices."""
    t = _set(TABLE, {"row_id": "F010"}, {"note": "pin 키 R-FINAL — τ 아님 (fixture)"})
    t = _set(t, {"row_id": "F040"}, {"note": "S1은 소스 G/C 정의를 적용하지 않음(NONE = 원값만)"})
    pk = {p["cell_id"]: p for p in P.build_map_packets(t, CELLS)}
    assert "τ 아님 (fixture)" in pk["Q-I2__EXP-012_probe"]["text"] and "원값만" in pk["Q-Z1__EXP-005"]["text"]
    s4 = P.s4_slices(t, CELLS)
    assert "τ 아님 (fixture)" in s4["Q-I2"]["S1_table"] and "원값만" in s4["Q-Z1"]["S1_table"]
    hdr = next(ln for ln in pk["Q-I2__EXP-012_probe"]["text"].splitlines() if ln.startswith("row_id,"))
    assert hdr.split(",")[-1] == "note"


def test_row_protocol_note_reaches_packets_and_s4(mp):
    """VERIFY-SIZ-C W-C1: the S1_table `protocol_note` (e.g. the PR-protocol UNKNOWN note) reaches map packets and S4."""
    t = _set(TABLE, {"row_id": "F050"}, {"protocol_note": "PR key — protocol vocabulary covers test-exact only (fixture)"})
    pk = {p["cell_id"]: p for p in P.build_map_packets(t, CELLS)}
    assert "covers test-exact only (fixture)" in pk["Q-Z3__EXP-002"]["text"]
    assert "covers test-exact only (fixture)" in P.s4_slices(t, CELLS)["Q-Z3"]["S1_table"]
    hdr = next(ln for ln in pk["Q-Z3__EXP-002"]["text"].splitlines() if ln.startswith("row_id,"))
    assert "protocol_note" in hdr.split(",")


@pytest.mark.parametrize("note,refused", [("0.4137", True), ("value 41.37%", True), ("41 %", True), ("1e-3", True),
                                          (".5 left", True), ("판정 전 — 상태만, run 2", False), ("INTERP_S1 (7)", False),
                                          ("EXP-020 queue", False), ("", False)])
@pytest.mark.parametrize("col", ["note", "protocol_note"])
def test_value_number_in_no_value_note_is_refused(note, refused, col):
    """The no-value guard covers `note` and `protocol_note`: a value-like number there (decimal / scientific /
    percent) is refused in a no-value cell row (F031 = D-PREJUDGE); descriptions with bare integers or IDs pass."""
    t = _set(TABLE, {"row_id": "F031"}, {col: note})
    if refused:
        with pytest.raises(P.PacketError, match=f"value-like number in the {col} of a D-PREJUDGE"):
            P.build_map_packets(t, CELLS)
        with pytest.raises(P.PacketError, match="value-like number"):
            P.s4_slices(t, CELLS)
    else:
        assert len(P.build_map_packets(t, CELLS)) == N_TARGETS


def test_not_citable_rows_are_excluded_and_linted(mp, rp):
    """evidence_use 'no: ...' rows (s1.py: owner UNKNOWN) never reach rows, obligations, citable IDs or S4 slices;
    citing one is a HIGH lint finding in the map and reduce post-checks."""
    p = mp["Q-I2__EXP-012_probe"]
    assert "F015" not in p["citable_ids"] and p["not_citable"] == ["F015"]
    assert "- F015: no: exp_owner UNKNOWN (synthetic fixture)" in p["text"]
    assert "\nF015," not in p["text"] and "0.6161" not in p["text"]
    assert "clean-D0/tf/trm/(3,6)/5: 1, 2" in p["text"] and "(3,6)/5: 1, 2, 3" not in p["text"]   # F015 = seed 3
    s4 = P.s4_slices(TABLE, CELLS)["Q-I2"]["S1_table"]
    assert "F015" not in s4 and "0.6161" not in s4
    n = good_map_note()
    n["candidate_sentences"][0]["row_ids"] = ["F010", "F012", "F015"]
    assert "evidence_use" in {f["rule"] for f in _high(P.check_map_note(n, p, MAP_SCHEMA, fixed_inputs(p["text"]), P.R3_TEXT))}
    q = rp["Q-I2"]
    memos = P.group_memos(q, _memos_for(q, mp, {"Q-I2__EXP-012_probe": good_map_note()}))
    rn = good_reduce_note(q, mp)
    rn["allowed_sentences"] = [{"text": GOOD_TEXT, "row_ids": ["F010", "F015"], "cell_ids": ["Q-I2__EXP-012_probe"],
                                "token": "none", "tier": "none", "cohort_tags": ["clean-D0"], "rules_applied": [],
                                "proposed_by": "both", "is_headline": False}]
    fs = P.check_reduce_note(rn, q, list(mp.values()), RED_SCHEMA, _reduce_inputs(q, memos), P.R3_TEXT, memos=memos)
    assert "evidence_use" in {f["rule"] for f in _high(fs)}


# ------------------------------------------------------------------------------------------ reduce packets
def test_reduce_packets(rp):
    built = {q: p for q, p in rp.items() if p["status"] == "built"}
    assert set(built) == set(P.QUESTIONS) - {"Q-Z7"}
    assert rp["Q-Z7"]["status"] == "skipped" and "Q-LISTED" in rp["Q-Z7"]["reason"]
    for q, p in built.items():
        assert len(p["targets"]) == sum(r["q"] == q and r["s1_class"] in P.S2_CLASSES for r in FX_CELLS.values())
    assert len(built["Q-Z1"]["targets"]) == 9 and len(built["Q-I4"]["targets"]) == 3       # S0r2: EXP-011 X-ABSENT
    _c, rows, _m = P.parse_csv(TABLE, "t", P.S1_TABLE_REQUIRED)
    for p in built.values():
        assert not any(r["row_id"] in p["text"] for r in rows)                # S1_cells rows only (§9 S3)
        assert "digest" not in p["text"].lower()
        other = [q for q in P.QUESTIONS if q != p["q"]]
        assert not any(f"\n{q}," in p["text"] for q in other)                  # no other question's rows


def _empty_note(cid, cls):
    return {"scope_id": cid, "cell_class": cls, "scope_mismatches": [], "candidate_sentences": [], "rule_findings": []}


def test_group_memos_strict(rp):
    p = rp["Q-I4"]
    memos = [{"cell_id": c, "model": m, "note": {}} for c in p["targets"] for m in P.MAP_MODELS]
    assert len(P.group_memos(p, memos)) == 2 * len(p["targets"])
    with pytest.raises(P.PacketError, match="not fully collected"):
        P.group_memos(p, memos[:-1])
    with pytest.raises(P.PacketError, match="two memos"):
        P.group_memos(p, memos + memos[:1])


# ------------------------------------------------------------------------------------------ post-checks
GOOD_TEXT = (f"clean-D0 코호트 EXP-012/probe 행에서 seed 1의 trm (3,6) 셀 probe/test_exact는 0.4137, trm (1,1) 셀은 0.8823이다"
             f"(P-PROBE). {P.R3_TEXT}")


def good_map_note():
    n = _empty_note("Q-I2__EXP-012_probe", "A-DESC")
    n["candidate_sentences"] = [{"text": GOOD_TEXT, "row_ids": ["F010", "F012"], "token": "none", "tier": "none",
                                 "cohort_tags": ["clean-D0"], "rules_applied": ["R-2", "R-3", "R-6"]}]
    n["scope_mismatches"] = [{"row_ids": ["F011"], "kind": "needs_check", "evidence": "F011 value 0.0291",
                              "severity": "low", "suggested_check": "S1_table.csv row F011 column value"}]
    return n


def _high(findings):
    return [f for f in findings if f["severity"] == "high"]


def test_check_map_note_good(mp):
    p = mp["Q-I2__EXP-012_probe"]
    assert P.validate_schema(good_map_note(), MAP_SCHEMA) == []
    assert _high(P.check_map_note(good_map_note(), p, MAP_SCHEMA, fixed_inputs(p["text"]), P.R3_TEXT)) == []
    lk = mp["Q-I4__EXP-016"]
    n = _empty_note(lk["cell_id"], "D-LOCKED")
    n["rule_findings"] = [{"row_ids": ["F030"], "rule": "D-LOCKED", "kind": "missing_obligation",
                           "evidence": P.LOCKED_MARKER, "severity": "medium", "suggested_check": "S1_cells row"}]
    assert _high(P.check_map_note(n, lk, MAP_SCHEMA, fixed_inputs(lk["text"]), P.R3_TEXT)) == []


def _cand(text, rows, token="none", tier="none", cohort=("clean-D0",)):
    return {"text": text, "row_ids": list(rows), "token": token, "tier": tier, "cohort_tags": list(cohort),
            "rules_applied": []}


@pytest.mark.parametrize("cid,cand,rule", [
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT.replace("0.4137", "0.41"), ["F010", "F012"]), "N-number"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " 58개 셀 중 하나다.", ["F010", "F012"]), "N-number"),   # header-only number
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT, ["F010", "F012"], cohort=("clean-D1",)), "R-2"),        # declared tag != rows
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT, ["F010", "F999"]), "row_id"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT, []), "row_id"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " This rules out capacity.", ["F010", "F012"]), "R-1/R-10"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " z carry 때문이다.", ["F010", "F012"]), "N-causal"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " seed 평균으로 보면 같다.", ["F010", "F012"]), "R-6"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT.replace(P.R3_TEXT, ""), ["F010", "F012"]), "R-3"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT, ["F010", "F012"], token="supported", tier="4"), "N-token"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " (supported)", ["F010", "F012"]), "N-token"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " EXP-015 참조.", ["F010", "F012"]), "R-11"),
    ("Q-I2__EXP-012_probe", _cand(GOOD_TEXT + " CL-0006 참조.", ["F010", "F012"]), "R-9"),
    ("Q-I2__EXP-001", _cand(f"legacy 코호트 F001 행은 {P.UNKNOWN_KEY}.", ["F001"], cohort=("legacy",)), "R-2"),
    ("Q-I2__EXP-001", _cand(f"{P.LEGACY_NOTE}: F001 행.", ["F001"], cohort=("legacy",)), "§7.1-6"),
    ("Q-Z1__EXP-005", _cand(f"{P.LEGACY_NOTE}: 0.3141 (P-PROBE), 0.2718 (P-ALL).", ["F040", "F041"],
                            cohort=("legacy",)), "R-12"),
    ("Q-I1__CL-0003", _cand("CL-0003 원자는 partial 이다.", ["CELL:Q-I1__CL-0003"], token="partial", tier="1",
                            cohort=()), "N-token"),
    ("Q-Z4__EXP-006", _cand("legacy EXP-006 행의 상태를 적는다.", ["CELL:Q-Z4__EXP-006"], cohort=()), "§7.2"),
    ("Q-I4__EXP-016", _cand(f"{P.LOCKED_MARKER} EXP-016.", ["F030"], cohort=("clean-D1",)), "§9-S2"),
])
def test_check_map_note_catches(mp, cid, cand, rule):
    p = mp[cid]
    n = _empty_note(cid, p["cell_class"])
    n["candidate_sentences"] = [cand]
    rules = {f["rule"] for f in _high(P.check_map_note(n, p, MAP_SCHEMA, fixed_inputs(p["text"]), P.R3_TEXT))}
    assert rule in rules, rules


def test_check_map_note_identity_and_schema(mp):
    p = mp["Q-I2__EXP-012_probe"]
    n = good_map_note()
    n["scope_id"] = "Q-I2__EXP-005"
    assert "identity" in {f["rule"] for f in P.check_map_note(n, p, MAP_SCHEMA, fixed_inputs(p["text"]), P.R3_TEXT)}
    n = good_map_note()
    n["cell_summary"] = "free prose"
    assert {f["rule"] for f in P.check_map_note(n, p, MAP_SCHEMA, fixed_inputs(p["text"]), P.R3_TEXT)} == {"schema"}


def _memos_for(p, mpk, extra=None):
    out = []
    for c in p["targets"]:
        for m in P.MAP_MODELS:
            note = _empty_note(c, mpk[c]["cell_class"])
            if extra and c in extra:
                note = extra[c]
            out.append({"cell_id": c, "model": m, "note": note})
    return out


def good_reduce_note(p, mpk):
    return {"q": p["q"], "q_token": p["question_token"],
            "cells_seen": [{"cell_id": c, "cell_class": mpk[c]["cell_class"], "sonnet_memo": True, "opus_memo": True}
                           for c in p["targets"]],
            "allowed_sentences": [], "status_only_cells": [], "disagreements": [], "open_findings": [], "uncovered": []}


def _reduce_inputs(p, memos):
    return [HEADER.read_text(encoding="utf-8"), RED_SYS.read_text(encoding="utf-8"), p["text"]] + \
        [json.dumps(m["note"], ensure_ascii=False) for m in memos]


def test_check_reduce_good_and_render(rp, mp):
    p = rp["Q-I2"]
    memos = P.group_memos(p, _memos_for(p, mp, {"Q-I2__EXP-012_probe": good_map_note()}))
    n = good_reduce_note(p, mp)
    n["allowed_sentences"] = [
        {"text": GOOD_TEXT, "row_ids": ["F010", "F012"], "cell_ids": ["Q-I2__EXP-012_probe"], "token": "none", "tier": "none",
         "cohort_tags": ["clean-D0"], "rules_applied": ["R-2", "R-3"], "proposed_by": "both", "is_headline": False},
        {"text": "H-023 행의 상속 토큰은 partial (tier 4)이다.", "row_ids": ["CELL:Q-I2__H-023"],
         "cell_ids": ["Q-I2__H-023"], "token": "partial", "tier": "4", "cohort_tags": [], "rules_applied": ["R-1"],
         "proposed_by": "opus", "is_headline": False}]
    n["disagreements"] = [{"cell_id": "Q-I2__EXP-012_probe", "row_ids": ["F011"], "topic": "a | b", "sonnet_text": "x",
                           "opus_text": "y"}]
    assert P.validate_schema(n, RED_SCHEMA) == []
    fs = P.check_reduce_note(n, p, list(mp.values()), RED_SCHEMA, _reduce_inputs(p, memos), P.R3_TEXT, memos=memos)
    assert _high(fs) == [], fs
    n2 = copy.deepcopy(n)
    n2["allowed_sentences"][0]["text"] = GOOD_TEXT.replace("0.8823", "0.0291")      # a number no candidate sentence has
    fs = P.check_reduce_note(n2, p, list(mp.values()), RED_SCHEMA, _reduce_inputs(p, memos), P.R3_TEXT, memos=memos)
    assert "N-number" in {f["rule"] for f in _high(fs)}
    md = P.render_reduce_md(n, p)
    for h in ("## Scope", "## Allowed sentences", "## Status-only cells", "## Sonnet/Opus disagreements",
              "## Open rule findings", "## UNCOVERED"):
        assert h in md
    assert "a \\| b" in md and "DRAFT" in md
    assert P.render_reduce_md(n, p) == md


@pytest.mark.parametrize("q,sentence,rule", [
    ("Q-I4", {"text": f"EXP-016 상태. {P.LOCKED_MARKER}", "row_ids": ["F030"], "cell_ids": ["Q-I4__EXP-016"],
              "token": "none", "tier": "none", "cohort_tags": ["clean-D1"], "rules_applied": [], "proposed_by": "both",
              "is_headline": False}, "§6.2"),
    ("Q-Z4", {"text": "EXP-006 행의 상태를 적는다.", "row_ids": ["CELL:Q-Z4__EXP-006"], "cell_ids": ["Q-Z4__EXP-006"],
              "token": "none", "tier": "none", "cohort_tags": [], "rules_applied": [], "proposed_by": "both",
              "is_headline": False}, "§7.2"),
    ("Q-Z5", {"text": "FG-0003 그림은 곡선을 보여준다.", "row_ids": ["CELL:Q-Z5__FG-0003"], "cell_ids": ["Q-Z5__FG-0003"],
              "token": "supported", "tier": "1", "cohort_tags": ["clean-D1"], "rules_applied": [], "proposed_by": "both",
              "is_headline": True}, "§7.2-3"),
    ("Q-I2", {"text": "EXP-012 행.", "row_ids": ["F040"], "cell_ids": ["Q-I2__EXP-012_probe"], "token": "none",
              "tier": "none", "cohort_tags": [], "rules_applied": [], "proposed_by": "both", "is_headline": False},
     "row_id"),
])
def test_check_reduce_catches(rp, mp, q, sentence, rule):
    p = rp[q]
    memos = P.group_memos(p, _memos_for(p, mp))
    n = good_reduce_note(p, mp)
    n["allowed_sentences"] = [sentence]
    fs = P.check_reduce_note(n, p, list(mp.values()), RED_SCHEMA, _reduce_inputs(p, memos), P.R3_TEXT)
    assert rule in {f["rule"] for f in _high(fs)}, fs


def test_check_reduce_coverage(rp, mp):
    p = rp["Q-I4"]
    n = good_reduce_note(p, mp)
    n["cells_seen"] = n["cells_seen"][:1]
    fs = P.check_reduce_note(n, p, list(mp.values()), RED_SCHEMA, _reduce_inputs(p, []), P.R3_TEXT)
    assert "coverage" in {f["rule"] for f in fs}


# ------------------------------------------------------------------------------------------ schemas
@pytest.mark.parametrize("schema", [MAP_SCHEMA, RED_SCHEMA])
def test_schema_subset(schema):
    assert P.schema_keywords(schema) <= ALLOWED_KEYWORDS

    def walk(s):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False
            assert sorted(s["required"]) == sorted(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    walk(schema)


def test_map_schema_rows_and_tokens():
    items = MAP_SCHEMA["properties"]
    for arr in ("scope_mismatches", "candidate_sentences", "rule_findings"):
        assert "row_ids" in items[arr]["items"]["required"]
    assert "cell_summary" not in items                    # no un-cited free prose (every sentence cites a row_id)
    tok = items["candidate_sentences"]["items"]["properties"]["token"]["enum"]
    assert tok == list(P.V_TOK) + ["none"]
    assert items["cell_class"]["enum"] == list(P.S2_CLASSES)
    assert RED_SCHEMA["properties"]["q_token"]["enum"] == ["Q-INHERIT", "Q-DESC-ONLY", "Q-UNCOVERED"]


# ------------------------------------------------------------------------------------------ prompts
def test_prompts_enforce_rules():
    m = MAP_SYS.read_text(encoding="utf-8")
    r = RED_SYS.read_text(encoding="utf-8")
    for t in (m, r):
        for s in (P.LEGACY_NOTE, P.LOCKED_MARKER, P.MARGINAL, P.PROTOCOL_MIX, P.UNKNOWN_KEY, P.R8_CAP, "R-3",
                  "causal", "necessary", "원인", "필요", "CL-0006", "EXP-015"):
            assert s in t, s
    assert "empty list" in m and "LOCKED" in m and "No vote" in r and "no global digest" in r.lower()
    for phrase in ("rules out", "excludes", "proves", "deductive", "capacity ceiling", "truncation helps",
                   "the retained region", "z/eff_rank is blind to collapse", "first to observe", "ACT cannot rescue"):
        assert phrase in m, phrase                       # R-10 list restated verbatim in the map prompt


# ------------------------------------------------------------------------------------------ specs
def _yaml(rel):
    return yaml.safe_load((TPL / rel).read_text(encoding="utf-8"))


def test_map_specs_are_twins():
    s, o = _yaml("batch/map_sonnet.spec.yaml"), _yaml("batch/map_opus.spec.yaml")
    assert (s["model"], o["model"]) == P.MAP_MODELS
    a, b = copy.deepcopy(s), copy.deepcopy(o)
    for d in (a, b):
        for k in ("name", "model"):
            d.pop(k)
        d["budget"].pop("cap_usd_hard_worst")
    assert a == b
    assert s["budget"]["cap_usd_hard_worst"] + o["budget"]["cap_usd_hard_worst"] <= 15.0      # PREREG §10 map
    strip = lambda t: re.sub(r"sonnet|opus|5\.0|10\.0", "", t)  # noqa: E731
    assert strip((TPL / "batch/map_sonnet.spec.yaml").read_text()) == strip((TPL / "batch/map_opus.spec.yaml").read_text())


def test_spec_fields_and_references():
    from tools.api_batch.prices import PRICES
    from tools.api_batch.spec import EFFORTS, TTLS
    for rel in ("batch/map_sonnet.spec.yaml", "batch/map_opus.spec.yaml", "batch/reduce_fable.spec.yaml"):
        sp = _yaml(rel)
        assert sp["model"] in PRICES and sp["effort"] in EFFORTS and sp["cache_ttl"] in TTLS
        assert isinstance(sp["max_tokens"], int) and sp["max_tokens"] > 0
        assert sp["header"] == "shared/siz_rules_header.md"
        assert (TPL / sp["header"]).is_file() and (TPL / sp["prompt"]["system_template"]).is_file()
        assert (REPO / sp["output_schema"]).is_file()
        assert sp["packets"]["generator"] == "tools/state_iter_z_synthesis/packets.py"
        assert callable(getattr(P, sp["packets"]["function"]))
    red = _yaml("batch/reduce_fable.spec.yaml")
    assert red["model"] == P.REDUCE_MODEL and red["digest"] is False and red["group_by"] == ["q"]
    assert red["exclude_q_tokens"] == list(P.REDUCE_EXCLUDED) and red["budget"]["cap_usd_hard_worst"] <= 10.0
    assert set(red["map_batches"]) == set(P.MAP_MODELS) and "s1_table" not in red["data"]


def test_batch_specs_load_with_the_siz_uses():
    """tools/api_batch now has the SIZ-map / SIZ-reduce uses (added 2026-10-30; PACKET_SPEC.md section 5), so the
    drafts pass the generic spec load instead of being refused."""
    from tools.api_batch.manifest import USES
    assert "SIZ-map" in USES and "SIZ-reduce" in USES
    from tools.api_batch.spec import load_spec
    for rel in ("batch/map_sonnet.spec.yaml", "batch/map_opus.spec.yaml", "batch/reduce_fable.spec.yaml"):
        load_spec(TPL / rel)


def test_draft_sync_spec_loads():
    from tools.api_batch.draft import load_draft_spec
    sp = load_draft_spec(TPL / "draftsync/spec.yaml")
    ids = [s["id"] for s in sp["sections"]]
    assert ids == [f"rep-{q.lower()}" for q in P.QUESTIONS if q != "Q-Z7"] + ["paper-en-iter", "paper-en-z"]
    assert (TPL / "draftsync" / sp["system"]).is_file()
    for s in sp["sections"]:
        assert (TPL / "draftsync" / s["prompt"]).is_file()
        assert s["data"][0] == f"{P.SYNTH_REL}/S23_templates/shared/siz_rules_header.md"
        assert all(d.startswith(P.SYNTH_REL + "/") and Path(d).suffix in (".md", ".csv") for d in s["data"])
    assert sp["budget_usd"] <= 10
    txt = (TPL / "draftsync/spec.yaml").read_text(encoding="utf-8")
    assert "OUTSIDE THE FROZEN PREREG" in txt


# ------------------------------------------------------------------------------------------ deny / api_batch compat
TEMPLATE_TEXT_FILES = ["shared/siz_rules_header.md", "shared/siz_map_system.md", "shared/siz_reduce_system.md",
                       "draftsync/shared/siz_s4_system.md", "draftsync/shared/siz_s4_report_q.md",
                       "draftsync/shared/siz_s4_paper_en.md", "batch/map_sonnet.spec.yaml",
                       "batch/map_opus.spec.yaml", "batch/reduce_fable.spec.yaml", "draftsync/spec.yaml"]


def test_templates_and_packets_pass_the_live_deny_matcher(mp, rp):
    from tools.api_batch.common import Halt
    try:
        from tools.api_batch.deny import default_deny_files, load_deny
        deny = load_deny(default_deny_files())
    except (Halt, OSError) as e:
        pytest.skip(f"live deny sources not loadable: {e}")
    texts = {rel: (TPL / rel).read_text(encoding="utf-8") for rel in TEMPLATE_TEXT_FILES}
    texts.update({f"map:{c}": p["text"] for c, p in mp.items()})
    texts.update({f"reduce:{q}": p["text"] for q, p in rp.items() if p["text"]})
    hits = {k: deny.matcher.hits(v) for k, v in texts.items() if deny.matcher.hits(v)}
    assert hits == {}
    assert not any(P.FORBIDDEN_TEXT.search(v) for v in texts.values())


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


def test_requests_build_through_api_batch_offline(tmp_path, mp, rp):
    """The packets, staged prompts and schema pass PackBuilder (L1-L4) and manifest.validate_request offline, with a
    deny mirror of the three active families. Also a hard-worst estimate with the fixture-sized packets."""
    from tools.api_batch import manifest as M
    from tools.api_batch import prices
    from tools.api_batch.common import Halt, canon_json, est_tokens, sha256_text
    from tools.api_batch.pack import PackBuilder
    (tmp_path / "W.yaml").write_text(RELEASED_ONLY, encoding="utf-8")
    (tmp_path / "D.yaml").write_text(ACTIVE_DENY_MIRROR, encoding="utf-8")
    try:
        lab = tmp_path / "lab"
        batch = lab / "38_api-batch/11_batches/2099-01-01_siz-map-fixture"
        b = PackBuilder.from_files([tmp_path / "W.yaml", tmp_path / "D.yaml"], lab_root=lab, batch_dir=batch)
    except Halt as e:
        pytest.skip(f"api_batch deny loader unavailable: {e}")
    s1dir = lab / P.SYNTH_REL
    s1dir.mkdir(parents=True)
    (s1dir / "S1_table.csv").write_text(TABLE, encoding="utf-8")
    (s1dir / "S1_cells.csv").write_text(CELLS, encoding="utf-8")
    hashes = tmp_path / "HASHES.sha256"
    rels = [f"lab/{P.SYNTH_REL}/S23_templates/{r}" for r in
            ("shared/siz_rules_header.md", "shared/siz_map_system.md", "shared/siz_reduce_system.md",
             "batch/map_sonnet.spec.yaml")]
    hashes.write_text("".join(f"{P.sha256_file(REPO / r)}  {r}\n" for r in rels), encoding="utf-8")
    P.stage(rels, batch, hashes)
    assert (batch / "spec.yaml").read_bytes() == (TPL / "batch/map_sonnet.spec.yaml").read_bytes()
    system = b.build_verify_only("spec_text", "system", (batch / "shared/siz_map_system.md").read_text(encoding="utf-8"),
                                 origin=batch / "shared/siz_map_system.md")
    header = b.build_verify_only("spec_text", "header", (batch / "shared/siz_rules_header.md").read_text(encoding="utf-8"),
                                 origin=batch / "shared/siz_rules_header.md")
    parents = [b.build_verify_only("u1_data", n, (s1dir / n).read_text(encoding="utf-8"), origin=s1dir / n)
               for n in ("S1_table.csv", "S1_cells.csv")]
    shas = frozenset({sha256_text(canon_json(MAP_SCHEMA))})
    mt = {m: _yaml(f"batch/map_{m.split('-')[1]}.spec.yaml")["max_tokens"] for m in P.MAP_MODELS}
    caps = {m: _yaml(f"batch/map_{m.split('-')[1]}.spec.yaml")["budget"]["cap_usd_hard_worst"] for m in P.MAP_MODELS}
    worst = {m: 0.0 for m in P.MAP_MODELS}
    for cid, p in mp.items():
        cp = b.build_u1_cell(f"cell-{cid}", p["text"], parents, {"cell": cid})
        for model in P.MAP_MODELS:
            req = M.build_request(M.make_custom_id("SIZ-map", cp.text, f"{cid}{model[7:9]}", 1), model, "medium",
                                  mt[model], system, [header], [cp], "5m", MAP_SCHEMA, meta={"cell": cid})
            M.validate_request(req, schema_shas=shas)
            worst[model] += prices.hard_worst_request(model, est_tokens(header.text),
                                                      est_tokens(system.text) + est_tokens(cp.text), mt[model], "5m")
    for m in P.MAP_MODELS:                                # fixture-sized packets; plan's figure is authoritative
        assert worst[m] <= 0.9 * caps[m], (m, worst[m])
    rsys = b.build_verify_only("spec_text", "rsystem", (batch / "shared/siz_reduce_system.md").read_text(encoding="utf-8"),
                               origin=batch / "shared/siz_reduce_system.md")
    rshas = frozenset({sha256_text(canon_json(RED_SCHEMA))})
    for q, p in rp.items():
        if p["text"]:
            cp = b.build_u1_cell(f"reduce-{q}", p["text"], parents[1:], {"q": q})
            req = M.build_request(M.make_custom_id("SIZ-reduce", cp.text, q, 1), P.REDUCE_MODEL, "high", 16000, rsys,
                                  [header, cp], [], "none", RED_SCHEMA, meta={"q": q})
            M.validate_request(req, schema_shas=rshas)
    with pytest.raises(Exception):                        # the same request is refused once a deny family matches
        b.build_u1_cell("bad", "x sealed/ y", parents, {})


def test_stage_refuses_changed_bytes(tmp_path):
    rel = f"lab/{P.SYNTH_REL}/S23_templates/shared/siz_map_system.md"
    h = tmp_path / "H.sha256"
    h.write_text(f"{'0' * 64}  {rel}\n", encoding="utf-8")
    with pytest.raises(P.PacketError, match="differs"):
        P.stage([rel], tmp_path / "b", h)
    assert not (tmp_path / "b").exists()
    with pytest.raises(P.PacketError, match="not listed"):
        P.stage([rel.replace("map", "reduce")], tmp_path / "b", h)


def test_cli_reduce_s4_and_check(tmp_path, mp):
    t, c = str(FIX / "S1_table.fixture.csv"), str(FIX / "S1_cells.fixture.csv")
    assert P.main(["reduce", "--s1-cells", c, "--out", str(tmp_path / "r")]) == 0
    idx = list(csv.DictReader(io.StringIO((tmp_path / "r/index.csv").read_text())))
    assert len(idx) == 12 and sum(r["status"] == "built" for r in idx) == 11
    assert P.main(["s4-inputs", "--s1-table", t, "--s1-cells", c, "--out", str(tmp_path / "s4")]) == 0
    assert len(list((tmp_path / "s4").glob("*.csv"))) == 24
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "Q-I2__EXP-012_probe.claude-sonnet-5-5.json").write_text(json.dumps(good_map_note(), ensure_ascii=False))
    rc = P.main(["check-map", "--s1-table", t, "--s1-cells", c, "--notes", str(notes), "--out", str(tmp_path / "f.csv")])
    rows = list(csv.DictReader(io.StringIO((tmp_path / "f.csv").read_text())))
    assert rc == 1 and all(r["rule"] == "coverage" for r in rows if r["severity"] == "high")
    assert not any(r["where"].startswith("Q-I2__EXP-012_probe/") and r["severity"] == "high" for r in rows)
    bad = tmp_path / "bad_cells.csv"
    bad.write_text(CELLS.replace("Q-Z7,CL-0006", "Q-Z7,CL-0099"), encoding="utf-8")
    assert P.main(["map", "--s1-table", t, "--s1-cells", str(bad), "--out", str(tmp_path / "m")]) == 2
    assert not (tmp_path / "m").exists()


# ------------------------------------------------------------------------------------------ witness list
def test_hashes_file_is_current():
    hp = TPL / "HASHES.sha256"
    if not hp.is_file():
        pytest.skip("HASHES.sha256 not written yet")
    table = P.read_hashes(hp)
    for rel, h in table.items():
        assert P.sha256_file(REPO / rel) == h, rel
    listed = set(table)
    for f in TPL.rglob("*"):
        if f.is_file() and f.name != "HASHES.sha256":
            assert f"lab/{P.SYNTH_REL}/S23_templates/{f.relative_to(TPL).as_posix()}" in listed, f
    for rel in ("tools/state_iter_z_synthesis/packets.py", "tests/test_state_iter_z_packets.py",
                "tests/fixtures/state_iter_z_synthesis/S1_table.fixture.csv",
                "tests/fixtures/state_iter_z_synthesis/S1_cells.fixture.csv"):
        assert rel in listed, rel
