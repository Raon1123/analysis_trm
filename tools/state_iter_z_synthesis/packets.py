"""S2/S3 packet generator for the state/iteration/z synthesis.

Frozen pre-registration: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md
(sha256 87fa6e7a...3e1b, r6). This module implements the PREREG §9 S2/S3 rows: one map packet per (q, s) cell of
class A-VERDICT / A-DESC / D-LOCKED / D-PREJUDGE, one reduce packet per question q (Q-LISTED and Q-NONE excluded,
no global digest), plus offline post-checks of the model outputs and the rules header shared by map and reduce.

Pure functions over text (so a future tools/api_batch branch can call them on PackBuilder-scanned text, as
tools/api_batch/u1.py does) and a CLI over paths. No network, no API key, no wandb, no value computation:
S1 rows are copied verbatim. The only derived lines are presence lists (which cohorts / arch x HL cells / seeds /
protocols / T_k labels appear in the rows) -- never a number computed from a value.

Map/reduce packets read only the two S1 files; the PREREG is read only to build the rules header (its §4 names the
forbidden path globs literally, so it could never pass api_batch verify-only scanning as a request input).

Input contract: the column names written by the producer tools/state_iter_z_synthesis/s1.py (S1_TABLE_REQUIRED,
S1_CELLS_MAP below; the tests check them against the s1 module). S1_table rows whose evidence_use starts with 'no'
are not citable: they never enter packet rows, obligations, citable IDs or S4 slices.

Fail-closed (PacketError, exit 2): PREREG sha mismatch (header build); S1_cells (q, s) set != the 58-cell universe of PREREG
§7.0.5; a missing required column (never synthesised); s1_class != s0_confirmed_class; a legacy S1_table row whose
cohort_note is not the fixed R-2 string (or a non-legacy row with a non-empty cohort_note); a value in an A-VERDICT /
D-PREJUDGE / X-* / D-SEALED / LISTED-ONLY / UNCOVERED cell row (= s1.py NO_VALUE_CLASSES; stricter than the PREREG
§9 S1 wording) or in a metric_key = UNKNOWN row; an S1_table (q, s) that is not an S1_cells cell; a duplicate row_id;
an evidence_use other than 'yes' / 'no...'; a PREREG §4 forbidden-path token in either S1 file or any packet text.

Run (repo root):  CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m tools.state_iter_z_synthesis.packets <cmd> ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = REPO_ROOT / "lab"
SYNTH_REL = "23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis"
PREREG_PATH = LAB_ROOT / "23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md"
PREREG_SHA256 = "87fa6e7acf2ba7278c4dea1835c1c27790d1bfe519b2e966374568b300c73e1b"
TEMPLATES_DIR = LAB_ROOT / SYNTH_REL / "S23_templates"
MAP_FORMAT = "siz-map-packet/v1"
REDUCE_FORMAT = "siz-reduce-packet/v1"
MAP_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5")     # PREREG §9 S2: same packet, both models, no vote
REDUCE_MODEL = "claude-fable-5-1"                         # PREREG §9 S3

# ------------------------------------------------------------------------------------------ vocabulary (PREREG)
QUESTIONS = ("Q-I1", "Q-I2", "Q-I3", "Q-I4", "Q-I5", "Q-Z1", "Q-Z2", "Q-Z3", "Q-Z4", "Q-Z5", "Q-Z6", "Q-Z7")
CLASSES = ("X-PROV", "LISTED-ONLY", "D-SEALED", "X-ABSENT", "D-PREJUDGE", "X-OUT", "D-LOCKED", "A-VERDICT",
           "A-DESC", "UNCOVERED")                                         # §7.1 order
S2_CLASSES = ("A-VERDICT", "A-DESC", "D-LOCKED", "D-PREJUDGE")             # §9 S2 targets
LOCKED_CLASSES = ("D-LOCKED", "D-PREJUDGE")                                # §9 S2: (a)+(c) only
NO_VALUE_CLASSES = ("X-PROV", "LISTED-ONLY", "D-SEALED", "X-ABSENT", "D-PREJUDGE", "X-OUT", "UNCOVERED",
                    "A-VERDICT")   # = s1.py NO_VALUE_CLASSES: values only for A-DESC / D-LOCKED (§9 S1)
Q_TOKENS = ("Q-NONE", "Q-LISTED", "Q-INHERIT", "Q-DESC-ONLY", "Q-UNCOVERED")  # §7.2 order
REDUCE_EXCLUDED = ("Q-LISTED", "Q-NONE")                                   # §9 S3
V_TOK = ("supported", "refuted", "falsified-as-framed", "conditional", "partial")   # §0
COHORTS = ("legacy", "clean-D0", "clean-D1")
PROTOCOLS = ("P-ALL", "P-RECOMP", "P-PROBE", "P-RELOAD", "P-DECODE", "P-JUDGE")
KINDS = ("E", "A", "H")

# Fixed strings: byte-identical to the frozen PREREG (tests assert whitespace-normalized membership).
LEGACY_NOTE = "legacy, 신규 clean 코호트와 비교 불가"                        # R-2, §3.2 cohort_note
LOCKED_MARKER = "현황 기술(확증 아님)"                                      # §6.2, §7.2 Q-DESC-ONLY
MARGINAL = "marginal, not mechanistic"                                     # R-4
PROTOCOL_MIX = "프로토콜이 달라 직접 비교 불가"                              # §3.2 / R-12
UNKNOWN_KEY = "키 이름 미고정 — 값 없음"                                     # §7.1 갈래 6
R8_CAP = "consistent with optimization-not-capacity"                       # R-8
H_STATUS = "상태(판정 아님)"                                                # §7.1 갈래 9 (H)
H_STATUS_RULE = 'H: 상태 낱말을 "상태(판정 아님)"로 재진술'                     # §7.1 갈래 9 cell, verbatim (PREREG :575)
PREJUDGE_ONLY = "상태·run 수만"                                             # §7.1 갈래 5
EXP025_FIRST = "EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015)"    # R-11
R3_TEXT = ("iter((H,L)_train = (3,6))와 noiter((1,1))는 재귀 계산량이 약 6× 다르다. 이 차이는 정확도 수준의 marginal 관측이며, "
           "가중치 공유 재귀 구조의 효과와 단순 깊이/FLOPs 증가를 분리하지 않는다. 이 합성이 쓰는 소스 안에서 iter 셀과 계산량을 "
           "맞춘(depth-matched) non-recurrent 대조군으로 정의된 arm은 없다.")         # R-3 (PREREG :358-360, whitespace-joined)
R3_ANCHOR_START = '다음 문장을 붙인다:\n  "'
R3_ANCHOR_END = '" looped TF'
R3_II_PREFIX = "EXP-011 `abl_tfb_*` arm이 원문 정의("                        # R-3 범위 주 (ii)
R3_II_SUFFIX = ")상 비재귀 대조군이며, 이 합성에서는 A-DESC 서술만 한다"

# PREREG §4 forbidden paths (sealed dirs, prediction files, z snapshots). Same three families as the active
# API_BATCH_DENY.yaml sealed-artifact entries; here it is the PREREG rule, applied to every packet we emit.
FORBIDDEN_TEXT = re.compile(r"(?<![A-Za-z0-9])sealed/|(?<![A-Za-z0-9])preds(?![A-Za-z0-9])|z_snapshots", re.I)

# §7.0.5 universe (58 cells), in PREREG order. S1_cells must contain exactly these (q, s) pairs.
UNIVERSE = (
    ("Q-I1", ("H-108 sweep", "CL-0003", "CL-0005", "FG-0001")),
    ("Q-I2", ("EXP-001", "EXP-005", "EXP-010", "EXP-011", "EXP-012/probe", "EXP-012/sealed", "EXP-013", "H-022",
              "H-023", "H-025")),
    ("Q-I3", ("EXP-025", "EXP-022", "EXP-023", "FG-0006", "H-199")),
    ("Q-I4", ("EXP-011", "EXP-016", "EXP-020", "H-200")),
    ("Q-I5", ("CL-0004", "FG-0002")),
    ("Q-Z1", ("EXP-001", "EXP-004", "EXP-005", "EXP-007", "EXP-010", "EXP-012/probe", "EXP-012/sealed", "EXP-025",
              "H-023", "H-025", "H-A")),
    ("Q-Z2", ("EXP-001", "EXP-004", "EXP-007", "CL-0001", "CL-0002", "H-A", "H-023")),
    ("Q-Z3", ("EXP-002", "EXP-004", "H-023")),
    ("Q-Z4", ("EXP-006", "H-026")),
    ("Q-Z5", ("EXP-017", "FG-0003", "FG-0004", "FG-0005", "H-201")),
    ("Q-Z6", ("EXP-018", "EXP-020", "H-202", "H-200")),
    ("Q-Z7", ("CL-0006",)),
)
UNIVERSE_CELLS = tuple((q, s) for q, ss in UNIVERSE for s in ss)

# §2.2/§2.3 obligation columns, made operational. "always": the question row names the rule unconditionally;
# "contrast": only when the cell's rows span iter (3,6) and noiter (1,1) (the R-3 general rule applies to every q).
R3_ALWAYS = ("Q-I2", "Q-Z1")
R4_QUESTIONS = ("Q-I2", "Q-Z1")
R7_QUESTIONS = ("Q-I3",)
ITER, NOITER = (3, 6), (1, 1)

# S1 input contract = the column names written by the producer tools/state_iter_z_synthesis/s1.py (TABLE_FIELDS,
# CELL_FIELDS). packets.py is the consumer and adopts those names (orchestrator decision 2026-10-30); the tests compare
# every name below with the s1 module, so a renamed s1 column breaks them. A missing column is refused, never made up.
S1_TABLE_REQUIRED = ("row_id", "q", "s", "cell_class", "k", "arch", "block", "HL_train", "seed", "cohort",
                     "cohort_note", "protocol", "metric_key", "Tk_class", "value", "evidence_use")
# The single S1_cells map: (role used inside packets.py, S1_cells column written by s1.py, meaning).
S1_CELLS_MAP = (
    ("kind", "kind", "E / A / H"),
    ("class", "s1_class", "§7.1 class re-derived by S1 from the S0 snapshot"),
    ("class_s0", "s0_confirmed_class", "S0-confirmed class; must equal s1_class (else refused)"),
    ("expected_class", "expected_class", "PREREG §7.0.5 expectation (differs only for the S0r2 known mismatches)"),
    ("class_match", "class_match", "s1_class == expected_class, as written by s1.py (informational)"),
    ("q_token", "q_token", "§7.2 question token"),
    ("token", "token_V", "inherited verdict token: V_tok word set, '|'-separated (A-VERDICT, LISTED-ONLY, D-LOCKED)"),
    ("status_word", "token_N", "N_tok status word set; never a verdict (A-DESC H restatement, §7.1 branch 9)"),
    ("token_tier", "token_tier", "PREREG §5.1 tier of the token"),
    ("token_file", "token_file", "token source file"),
    ("token_locator", "token_locator", "token source locator"),
    ("scope", "atom_verdict_scope", "atom verdict scope text, verbatim (A rows)"),
    ("token_scope", "token_scope", "card-level scope note (H component cells)"),
    ("prohibited", "atom_prohibited_upgrade", "atom prohibited-upgrade text, verbatim"),
    ("restatement", "token_restatement", "D-LOCKED H status line, verbatim"),
    ("marker", "marker", "D-LOCKED marker"),
    ("registry_status_N", "registry_status_N", "tier-5 registry status; never a cell token"),
    ("values_policy", "values_policy", "S1 value policy for the cell"),
    ("n_runs", "n_runs", "run count (D-PREJUDGE: status and run counts only)"),
)
S1_CELLS_REQUIRED = ("q", "s") + tuple(col for _role, col, _m in S1_CELLS_MAP)
TABLE_IDENT = ("q", "s")          # repeated cell identity, dropped from the packet rows (as u1.py drops IDENT)
# S1_table columns that may enter packet text (allow-list): the required contract (without the bookkeeping columns
# cell_class and evidence_use) plus the PREREG §3.2 provenance columns and the row `note` / `protocol_note` (s1.py's
# disclosure text, e.g. 'τ 아님', 'NONE = 원값만', the PR-protocol UNKNOWN note; VERIFY-SIZ-C W-C1). Any other
# S1_table column (run_name, wandb ids, provenance paths ...) is dropped from every packet and S4 slice, not refused;
# the whole S1 text is still scanned (forbid + api_batch verify-only).
PACKET_TABLE_FIELDS = tuple(c for c in S1_TABLE_REQUIRED if c not in ("cell_class", "evidence_use")) + (
    "HL_eval", "weights", "eval_set_size", "probe_size", "reduction", "label_GC", "step", "budget", "frame", "exp_owner",
    "protocol_note", "note")
NOTE_COLUMNS = ("note", "protocol_note")         # description columns: no value-like number in a no-value cell row
# A value-like number (decimal, scientific or percent; a bare integer such as a count or an ID is not one). Refused
# in a NOTE_COLUMNS field of a no-value cell row: those are descriptions, never a place for a value.
VALUE_NUM_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\.\d+)(?:[eE][-+]?\d+)?|(?<![\w.])\d+(?:[eE][-+]?\d+|\s?%)")
# S1_cells columns that may enter packet text: every s1.py CELL_FIELDS column except n_table_rows (directory x key
# rows, "NOT a run count" per s1.py; it would invite pooled counts). Unknown future columns are dropped.
PACKET_CELL_FIELDS = S1_CELLS_REQUIRED + ("branch", "reason", "key_status", "metric_keys_read", "n_runs_evidence_use",
                                          "counts_by_cohort", "cohort_note", "token_discrepancy", "token_note",
                                          "atom_cohort", "notes")
INHERITED_COLUMNS = ("token_V", "token_N", "token_tier", "token_file", "token_locator", "atom_verdict_scope",
                     "token_scope", "atom_prohibited_upgrade", "token_restatement", "marker", "registry_status_N")
NO_TOKEN_CLASSES = ("X-PROV", "D-SEALED", "X-ABSENT", "X-OUT", "D-PREJUDGE", "UNCOVERED")   # §7.1 갈래 1-5, 10
NO_SCOPE_CLASSES = ("X-PROV", "D-SEALED")                                  # §7.1 갈래 1 / 3: no token, no scope
VTOK_WORD = re.compile(r"(?<![A-Za-z-])(supported|refuted|falsified-as-framed|conditional|partial)(?![A-Za-z-])", re.I)


class PacketError(Exception):
    """Fail-closed refusal: nothing is written."""


# ------------------------------------------------------------------------------------------ small helpers
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cell_id(q: str, s: str) -> str:
    """Filesystem- and custom_id-safe: 'Q-I2' + 'EXP-012/probe' -> 'Q-I2__EXP-012_probe'."""
    return f"{q}__{re.sub(r'[^A-Za-z0-9-]+', '_', s).strip('_')}"


def cell_ref(q: str, s: str) -> str:
    """Citable ID of the cell-level S1_cells row (for statements that rest on the token, not on a run row)."""
    return f"CELL:{cell_id(q, s)}"


def split_comments(text: str) -> tuple[list[str], str]:
    """Leading '#' lines (S0/S1 CSVs start with '# as_of: ...') and the CSV body."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return [ln.rstrip("\n") for ln in lines[:i]], "".join(lines[i:])


def parse_csv(text: str, label: str, required: tuple) -> tuple[list[str], list[dict], list[str]]:
    comments, body = split_comments(text)
    rd = csv.DictReader(io.StringIO(body))
    cols = list(rd.fieldnames or [])
    missing = [c for c in required if c not in cols]
    if missing:
        raise PacketError(f"{label}: missing required column(s) {missing} (contract: PACKET_SPEC.md)")
    rows = [dict(r) for r in rd]
    for r in rows:
        if None in r:
            raise PacketError(f"{label}: a row has more fields than the header")
    return cols, rows, comments


def csv_block(cols: list[str], rows: list[dict], drop: tuple = ()) -> str:
    keep = [c for c in cols if c not in drop]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(keep)
    for r in rows:
        w.writerow([r.get(c, "") for c in keep])
    return buf.getvalue().rstrip("\n")


def hl(value: str) -> tuple | None:
    nums = tuple(int(x) for x in re.findall(r"\d+", value or ""))
    return nums if len(nums) == 2 else None


def check_prereg(prereg_text: str) -> None:
    got = sha256_text(prereg_text)
    if got != PREREG_SHA256:
        raise PacketError(f"PREREG sha256 {got[:12]}... != frozen {PREREG_SHA256[:12]}... (immutable; refuse)")


def forbid(label: str, text: str) -> None:
    m = FORBIDDEN_TEXT.search(text)
    if m:
        raise PacketError(f"{label}: PREREG §4 forbidden-path token {m.group(0)!r} in packet text")


# ------------------------------------------------------------------------------------------ PREREG extraction
def _lines(prereg_text: str) -> list[str]:
    return prereg_text.split("\n")


def _block(lines: list[str], start: str, stop) -> list[str]:
    """Lines from the first line that starts with `start` up to (excluding) the first later line for which stop(l)."""
    for i, ln in enumerate(lines):
        if ln.startswith(start):
            for j in range(i + 1, len(lines)):
                if stop(lines[j]):
                    return lines[i:j]
            return lines[i:]
    raise PacketError(f"PREREG anchor {start!r} not found")


def r3_text(prereg_text: str) -> str:
    i = prereg_text.find(R3_ANCHOR_START)
    j = prereg_text.find(R3_ANCHOR_END, i)
    if i < 0 or j < 0:
        raise PacketError("PREREG R-3 fixed sentence anchors not found")
    return ws(prereg_text[i + len(R3_ANCHOR_START):j])


def r3_text_variant(r3_ii_cite: str | None) -> str:
    """R-3 범위 주: (i) the fixed text; (ii) if S0 found EXP-011's matching definition, its last sentence is replaced."""
    if not r3_ii_cite:
        return R3_TEXT
    head, sep, _last = R3_TEXT.rpartition(". 이 합성이 쓰는 소스 안에서")
    if not sep:
        raise PacketError("R-3 text has no last sentence to replace")
    return f"{head}. {R3_II_PREFIX}{r3_ii_cite}{R3_II_SUFFIX}."



HEADER_PREFACE = """# RULES HEADER - state/iteration/z synthesis (S2 map and S3 reduce)

This header is shared, byte for byte, by every S2 map request (claude-sonnet-5-5 and claude-opus-5-5) and every
S3 reduce request (claude-fable-5-1). Below the FIXED STRINGS block it consists only of verbatim excerpts of the
frozen pre-registration `2026-10-30_state-iter-z-synthesis_PREREG.md` (sha256 {sha}), generated by
`tools/state_iter_z_synthesis/packets.py build_rules_header`. "§" references point to that document. Parts that
are not reproduced are not needed for these tasks; if a rule seems to depend on an unreproduced part, report a
provenance gap instead of guessing.

## FIXED STRINGS (verbatim PREREG text; copy byte for byte; checks are exact string matches)

- R-2 legacy label (every legacy row, panel, caption, sentence): {legacy}
- R-3 iter/noiter text (every sentence, table or caption that contrasts iter and noiter): {r3}
- §6.2 / §7.2 status-only marker (D-LOCKED restatements; every Q-DESC-ONLY sentence): {locked}
- §7.1 branch 9 A-DESC hypothesis row (the PREREG cell; its fixed string is {hstatus}): {hstatus_rule}
- §7.1 branch 5 D-PREJUDGE: {prejudge}
- §7.1 branch 6 metric_key UNKNOWN: {unknown}
- R-4 downgrade when the 2x2 is incomplete: {marginal}
- R-8 ceiling for any sentence about the cause of collapse: {r8}
- R-12 / §3.2 protocol mix (same sentence): {mix}
- R-11 first mention of EXP-025: {exp025}
"""


def build_rules_header(prereg_text: str) -> str:
    """The shared rules header: preface + fixed strings + verbatim PREREG excerpts (§0 rows, §2.2-§2.4, §3.2, §5,
    §6.2, §7.1 decision table + duplicate-claim rule, §7.2). Deterministic; tests compare it to the committed file.
    The PREREG itself never enters a request (its §4 names the forbidden paths, so verify-only would refuse it);
    only this header, staged into <batch>/shared/, carries its rules."""
    check_prereg(prereg_text)
    if r3_text(prereg_text) != R3_TEXT:
        raise PacketError("R3_TEXT constant differs from the PREREG R-3 sentence")
    lines = _lines(prereg_text)
    s0 = _block(lines, "## 0.", lambda l: l.startswith("## "))          # every §0 row (define-before-use)
    s22 = _block(lines, "### 2.2", lambda l: l.startswith("### "))
    s23 = _block(lines, "### 2.3", lambda l: l.startswith("### "))
    s24 = _block(lines, "### 2.4", lambda l: l.startswith("## "))
    s32 = _block(lines, "### 3.2 provenance", lambda l: l.startswith("### "))
    s5 = _block(lines, "## 5.", lambda l: l.startswith("## "))
    s62 = _block(lines, "### 6.2", lambda l: l.startswith("### "))
    s71 = _block(lines, "### 7.1", lambda l: l.startswith("**경계 사례"))
    s72 = _block(lines, "### 7.2", lambda l: l.startswith("### "))
    pre = HEADER_PREFACE.format(sha=PREREG_SHA256, legacy=LEGACY_NOTE, r3=r3_text(prereg_text), locked=LOCKED_MARKER,
                                hstatus=H_STATUS, hstatus_rule=H_STATUS_RULE, prejudge=PREJUDGE_ONLY, unknown=UNKNOWN_KEY, marginal=MARGINAL,
                                r8=R8_CAP, mix=PROTOCOL_MIX, exp025=EXP025_FIRST)
    parts = [pre.rstrip("\n")]
    for title, blk in (("§0 symbols", s0), ("§2.2 iteration questions", s22),
                       ("§2.3 z questions", s23), ("§2.4 claim -> source-row mapping", s24),
                       ("§3.2 provenance columns", s32),
                       ("§5 claim-strength ceiling", s5), ("§6.2 locked sources", s62),
                       ("§7.1 cell classes (decision table)", s71), ("§7.2 question tokens", s72)):
        parts.append(f"## PREREG EXCERPT: {title}\n\n" + "\n".join(blk).strip("\n"))
    text = "\n\n".join(parts) + "\n"
    forbid("rules header", text)
    return text


# ------------------------------------------------------------------------------------------ S1 loading
def cell_view(r: dict) -> dict:
    """The raw s1.py row plus the S1_CELLS_MAP role names (role -> its s1 column). Verbatim CSV output always uses
    the raw s1 columns (PACKET_CELL_FIELDS); the roles are only read by this module's logic."""
    v = dict(r)
    for role, col, _m in S1_CELLS_MAP:
        if role != col:
            if role in r:
                raise PacketError(f"S1_cells: column {role!r} collides with a packets.py role name (S1_CELLS_MAP)")
            v[role] = r[col]
    return v


def check_cell_row(r: dict) -> None:
    """One S1_cells row (a cell_view): vocabulary, the S1 class gate, and no token / scope text where §7.1 allows
    none (갈래 1-5, 10)."""
    key = (r["q"], r["s"])
    if r["class"] not in CLASSES:
        raise PacketError(f"S1_cells {key}: class (s1_class) {r['class']!r} not in §7.1")
    if r["class_s0"] != r["class"]:
        raise PacketError(f"S1_cells {key}: s1_class {r['class']!r} != s0_confirmed_class {r['class_s0']!r} (S1 gate)")
    if r["q_token"] not in Q_TOKENS:
        raise PacketError(f"S1_cells {key}: q_token {r['q_token']!r} not in §7.2")
    if r["kind"] not in KINDS:
        raise PacketError(f"S1_cells {key}: kind {r['kind']!r} not in {KINDS}")
    if r["class"] in NO_TOKEN_CLASSES and VTOK_WORD.search(r["token"] or ""):
        raise PacketError(f"S1_cells {key}: a verdict word in the token (token_V) of a {r['class']} cell (§7.1: no token)")
    if r["class"] in NO_SCOPE_CLASSES and ((r["scope"] or "").strip() or (r["token_scope"] or "").strip()):
        raise PacketError(f"S1_cells {key}: scope text in a {r['class']} cell (§7.1: no value or token)")


def load_cells(cells_text: str) -> tuple[list[str], dict, list[str]]:
    """S1_cells -> (columns, {(q, s): cell_view}, comments). Shared by the map, reduce and S4 paths."""
    forbid("S1_cells", cells_text)
    ccols, crows, ccom = parse_csv(cells_text, "S1_cells", S1_CELLS_REQUIRED)
    cells: dict[tuple, dict] = {}
    for raw in crows:
        r = cell_view(raw)
        key = (r["q"], r["s"])
        if key in cells:
            raise PacketError(f"S1_cells: duplicate cell {key}")
        check_cell_row(r)
        cells[key] = r
    if set(cells) != set(UNIVERSE_CELLS):
        extra = sorted(set(cells) - set(UNIVERSE_CELLS))
        miss = sorted(set(UNIVERSE_CELLS) - set(cells))
        raise PacketError(f"S1_cells != §7.0.5 universe (58 cells): extra {extra[:4]} missing {miss[:4]}")
    return ccols, cells, ccom


def cells_cols_out(cols: list[str]) -> list[str]:
    return [c for c in cols if c in PACKET_CELL_FIELDS]


def not_citable(r: dict) -> bool:
    """S1_table evidence_use: 'yes' = citable; 'no...' (s1.py writes 'no: <why>') = never citable; else refused."""
    ev = (r["evidence_use"] or "").strip()
    if ev == "yes":
        return False
    if ev.lower().startswith("no"):
        return True
    raise PacketError(f"S1_table row {r['row_id']}: evidence_use {ev!r} is neither 'yes' nor 'no: ...'")


def table_cols_out(cols: list[str], drop: tuple = ()) -> list[str]:
    return [c for c in cols if c in PACKET_TABLE_FIELDS and c not in drop]


def load_s1(table_text: str, cells_text: str) -> dict:
    forbid("S1_table", table_text)                 # whole file, including columns that never reach a packet
    tcols, trows, tcom = parse_csv(table_text, "S1_table", S1_TABLE_REQUIRED)
    ccols, cells, ccom = load_cells(cells_text)
    qtok = {}
    for (q, _s), r in cells.items():
        if qtok.setdefault(q, r["q_token"]) != r["q_token"]:
            raise PacketError(f"S1_cells: question {q} carries more than one q_token")
    ids = set()
    by_cell: dict[tuple, list[dict]] = {k: [] for k in cells}
    excluded: dict[tuple, list[tuple]] = {k: [] for k in cells}
    for r in trows:
        rid = r["row_id"]
        if not rid or rid in ids or rid.startswith("CELL:"):
            raise PacketError(f"S1_table: empty, duplicate or reserved row_id {rid!r}")
        ids.add(rid)
        key = (r["q"], r["s"])
        if key not in cells:
            raise PacketError(f"S1_table row {rid}: (q, s) {key} is not an S1_cells cell")
        cls = cells[key]["class"]
        if r["cell_class"] != cls:
            raise PacketError(f"S1_table row {rid}: cell_class {r['cell_class']!r} != the cell's s1_class {cls!r}")
        if r["cohort"] not in COHORTS + ("UNKNOWN",):
            raise PacketError(f"S1_table row {rid}: cohort {r['cohort']!r} not in {COHORTS} or UNKNOWN")
        if r["cohort"] == "legacy" and r["cohort_note"] != LEGACY_NOTE:
            raise PacketError(f"S1_table row {rid}: legacy row without the exact R-2 cohort_note (§3.2: V1 FAIL)")
        if r["cohort"] != "legacy" and r["cohort_note"]:
            raise PacketError(f"S1_table row {rid}: non-legacy row with a non-empty cohort_note (§3.2)")
        if r["protocol"] not in PROTOCOLS + ("UNKNOWN",):
            raise PacketError(f"S1_table row {rid}: protocol {r['protocol']!r} not in §3.2 vocabulary")
        if r["value"].strip() and cls in NO_VALUE_CLASSES:
            raise PacketError(f"S1_table row {rid}: a value in a {cls} cell (§9 S1: no value extraction)")
        if cls in NO_VALUE_CLASSES and (r.get("step") or "").strip():
            raise PacketError(f"S1_table row {rid}: a step in a {cls} cell (§6.3 / §9 S1: status and run counts only)")
        if cls in NO_VALUE_CLASSES and (r.get("label_GC") or "").strip() not in ("", "NONE"):
            raise PacketError(f"S1_table row {rid}: label_GC {r['label_GC']!r} in a {cls} cell (an outcome label)")
        for col in NOTE_COLUMNS:
            if cls in NO_VALUE_CLASSES and VALUE_NUM_RE.search(r.get(col) or ""):
                raise PacketError(f"S1_table row {rid}: a value-like number in the {col} of a {cls} cell "
                                  "(no value extraction)")
        if r["value"].strip() and r["metric_key"] == "UNKNOWN":
            raise PacketError(f"S1_table row {rid}: a value with metric_key UNKNOWN (§7.1 branch 6: 값 없음)")
        if not_citable(r):                         # every check above still applies to the row
            excluded[key].append((rid, r["evidence_use"].strip()))
        else:
            by_cell[key].append(r)
    return {"table_cols": tcols, "cells_cols": ccols, "cells": cells, "rows": by_cell, "not_citable": excluded,
            "q_token": qtok,
            "comments": {"S1_table": tcom, "S1_cells": ccom},
            "sha": {"S1_table": sha256_text(table_text), "S1_cells": sha256_text(cells_text)}}


def citable_ids(q: str, s: str, rows: list[dict]) -> list[str]:
    return [cell_ref(q, s)] + [r["row_id"] for r in rows]


# ------------------------------------------------------------------------------------------ obligations
def obligations(q: str, s: str, cell: dict, rows: list[dict], q_token: str, r3: str) -> list[str]:
    """Presence-only obligation lines for one cell (no values, no counts)."""
    out = []
    cls, kind = cell["class"], cell["kind"]
    legacy = [r["row_id"] for r in rows if r["cohort"] == "legacy"]
    cohorts = sorted({r["cohort"] for r in rows})
    if rows:
        out.append(f"[R-2] cohorts in the rows: {', '.join(cohorts)}. Every empirical sentence carries its cohort tag.")
    if legacy:
        out.append(f"[R-2] legacy rows {', '.join(legacy)}: every sentence on them contains \"{LEGACY_NOTE}\"; "
                   "legacy and clean rows never share a sentence, a table cell or a figure axis.")
    if "UNKNOWN" in cohorts:
        out.append("[R-2] rows with cohort UNKNOWN: no empirical sentence (no cohort tag is available).")
    hls = {hl(r["HL_train"]) for r in rows}
    contrast = ITER in hls and NOITER in hls
    if q in R3_ALWAYS or contrast or (q == "Q-I1" and NOITER in hls):
        why = "the question row requires the full R-3 text" if q in R3_ALWAYS else \
            "rows span (H,L)_train (3,6) and (1,1)" if contrast else "a noiter-trained cell is reloaded"
        out.append(f"[R-3] required ({why}). Copy verbatim into every iter/noiter sentence: \"{r3}\"")
    else:
        out.append(f"[R-3] if any sentence contrasts iter and noiter, it carries verbatim: \"{r3}\"")
    if q in R7_QUESTIONS:
        out.append("[R-3/R-7] looped TF: quote the compute-matching definition of C only if this packet contains it "
                   "(otherwise report a provenance gap); no \"truncation helps\", no looped-TF vs TRM direction or "
                   "mechanism claim, acquisition (R-ANY) never extended to final value or retention, nothing "
                   "interpreted before the EXP-023 replication is judged.")
        out.append(f"[R-11] first mention of EXP-025 is \"{EXP025_FIRST}\"; never a bare EXP-015 elsewhere.")
    if q in R4_QUESTIONS and rows:
        groups: dict[tuple, set] = {}
        for r in rows:
            arm = {"trm": "z", "trm_singlez": "noz"}.get(r["arch"])
            h = hl(r["HL_train"])
            it = "iter" if h == ITER else "noiter" if h == NOITER else None
            g = groups.setdefault((r["cohort"], r["block"], r["k"]), set())
            if arm and it:
                g.add(f"{arm}_{it}")
        four = ("z_iter", "z_noiter", "noz_iter", "noz_noiter")
        for (co, bl, k), have in sorted(groups.items()):
            miss = [c for c in four if c not in have]
            state = "complete" if not miss else f"missing {', '.join(miss)} -> \"{MARGINAL}\" and name the missing cells"
            out.append(f"[R-4] 2x2 presence {co} / block {bl or 'UNKNOWN'} / k {k or 'UNKNOWN'}: "
                       f"present {', '.join(c for c in four if c in have) or 'none'}; {state}.")
    ks = []
    for r in rows:
        if r["k"] not in [k for k, _ in ks]:
            ks.append((r["k"], sorted({x["Tk_class"] for x in rows if x["k"] == r["k"]})))
    if len(ks) > 1:
        out.append("[R-5] rows span several k; every cross-k sentence names each k's T_k label from the rows: "
                   + "; ".join(f"k {k}: Tk_class {'/'.join(t) or 'UNKNOWN'}" for k, t in ks)
                   + ". Never explain by depth alone; cycle-type is predictor-only; no best / governing / first to observe.")
    if rows:
        seeds: dict[tuple, list] = {}
        for r in rows:
            g = seeds.setdefault((r["cohort"], r["block"], r["arch"], r["HL_train"], r["k"]), [])
            if r["seed"] not in g:
                g.append(r["seed"])
        out.append("[R-6] seeds are reported one by one, never averaged or pooled; seeds present by "
                   "(cohort, block, arch, HL_train, k): "
                   + "; ".join(f"{'/'.join(x or 'UNKNOWN' for x in key)}: {', '.join(v)}" for key, v in seeds.items()))
    protos = sorted({r["protocol"] for r in rows})
    if len(protos) > 1:
        out.append(f"[R-12] protocols present: {', '.join(protos)}. Never compare them in one sentence; if two must "
                   f"appear together, label each value and add \"{PROTOCOL_MIX}\" in the same sentence.")
    unk = [r["row_id"] for r in rows if r["metric_key"] == "UNKNOWN"]
    if unk:
        out.append(f"[§7.1 branch 6] rows {', '.join(unk)} have metric_key UNKNOWN: no value; write \"{UNKNOWN_KEY}\".")
    if cls == "D-LOCKED":
        out.append(f"[§6.2] D-LOCKED: tasks (a) and (c) only, candidate_sentences = []; any text describing this "
                   f"cell carries \"{LOCKED_MARKER}\"; never a basis for any atom, hypothesis or synthesis sentence.")
    if cls == "D-PREJUDGE":
        out.append(f"[§6.3 / §7.1 branch 5] D-PREJUDGE: tasks (a) and (c) only, candidate_sentences = []; "
                   f"{PREJUDGE_ONLY}; any value in this cell's rows is a high-severity mismatch.")
    if cls == "A-DESC":
        if kind == "H":
            out.append(f"[§7.1 branch 9] A-DESC hypothesis row, PREREG verbatim: {H_STATUS_RULE}; no verdict word.")
        else:
            out.append("[§7.1 branch 9] A-DESC: observation only, token none, no verdict word.")
    if cls == "A-VERDICT":
        out.append("[§7.1 branch 8] A-VERDICT: sentences stay inside the INHERITED TOKEN's scope text; the "
                   "prohibited text is binding; tier = token_tier.")
        if s.startswith("FG-"):
            out.append("[§5.2] figure atom: figure description only; never the basis of a synthesis sentence.")
    if q_token == "Q-DESC-ONLY":
        out.append(f"[§7.2] question token Q-DESC-ONLY: no verdict word in any sentence; every sentence carries "
                   f"\"{LOCKED_MARKER}\".")
    out.append(f"[R-8] any sentence about the cause of collapse stops at \"{R8_CAP}\"; no causal or necessity "
               "statement of any kind.")
    out.append("[R-9] never cite CL-0006; no Ω(log k) lower bound as a theorem; no H-023 original framing.")
    return out


# ------------------------------------------------------------------------------------------ map packets
def build_map_packets(table_text: str, cells_text: str, r3_ii_cite: str | None = None) -> list[dict]:
    """One packet per S2 target cell, in §7.0.5 order, from the two S1 files only (the PREREG rules reach the
    request through the staged rules header). Returns [{cell_id, q, s, kind, cell_class, mode, question_token,
    text, citable_ids, not_citable, token, token_tier, row_meta}]; row_meta = {row_id: {cohort, protocol, HL_train,
    metric_key}}. S1_table rows whose evidence_use starts with 'no' are never in the rows, the obligations or the
    citable IDs; only their row_id and evidence_use text are listed, so a citation of them is a lint violation."""
    s1 = load_s1(table_text, cells_text)
    r3 = r3_text_variant(r3_ii_cite)
    out = []
    for q, s in UNIVERSE_CELLS:
        cell = s1["cells"][(q, s)]
        if cell["class"] not in S2_CLASSES:
            continue
        rows = s1["rows"][(q, s)]
        excl = s1["not_citable"][(q, s)]
        cid = cell_id(q, s)
        mode = "LOCKED" if cell["class"] in LOCKED_CLASSES else "FULL"
        cites = citable_ids(q, s, rows)
        qt = s1["q_token"][q]
        parts = [
            f"# CELL PACKET {cid}",
            f"packet_format: {MAP_FORMAT}",
            f"prereg_sha256: {PREREG_SHA256}",
            f"s1_table_sha256: {s1['sha']['S1_table']}",
            f"s1_cells_sha256: {s1['sha']['S1_cells']}",
            *[f"s1_comment: {c.lstrip('# ').rstrip()}" for c in s1["comments"]["S1_cells"]],
            f"cell_id: {cid}",
            f"q: {q}",
            f"s: {s}",
            f"kind: {cell['kind']}",
            f"cell_class: {cell['class']}",
            f"question_token: {qt}",
            f"mode: {mode}" + (" (tasks (a) and (c) only; candidate_sentences must be [])" if mode == "LOCKED" else ""),
            "",
            "## CITABLE IDS",
            ", ".join(cites),
            "",
            "## RULE POINTERS (all in the RULES HEADER)",
            f"- question row: §2.2/§2.3 table row **{q}** (its ceiling column is the 2026-10-30 expectation; the "
            "INHERITED TOKEN below wins)",
            f"- claim -> source mapping: §2.4 rows that start with \"| {q} |\"",
            *([f"- atom ceiling: §5.2 row for {s} (the INHERITED TOKEN below wins)"] if cell["kind"] == "A" else []),
            f"- cell class {cell['class']}: §7.1; question token {qt}: §7.2",
            "",
            "## INHERITED TOKEN (S1_cells, verbatim; s1.py column names)",
            *[f"{col}: {cell[col] or '(none)'}" for col in INHERITED_COLUMNS],
            "",
            "## OBLIGATIONS FOR THIS CELL (computed from the rows below: presence lists only, no values)",
            *[f"- {x}" for x in obligations(q, s, cell, rows, qt, r3)],
            "",
            "## S1_cells ROW (verbatim CSV; allow-listed s1.py columns)",
            csv_block(cells_cols_out(s1["cells_cols"]), [cell]),
            "",
            "## S1_table ROWS (verbatim CSV; allow-listed columns only, identity columns q, s dropped)",
            csv_block(table_cols_out(s1["table_cols"], TABLE_IDENT), rows) if rows else "(no S1_table rows for this cell)",
            "",
            "## NOT CITABLE S1_table ROWS (evidence_use starts with \"no\"; never cite or use them; values not shown)",
            *([f"- {rid}: {ev}" for rid, ev in excl] or ["(none)"]),
        ]
        text = "\n".join(parts) + "\n"
        forbid(cid, text)
        out.append({"cell_id": cid, "q": q, "s": s, "kind": cell["kind"], "cell_class": cell["class"], "mode": mode,
                    "question_token": qt, "text": text, "citable_ids": cites, "not_citable": [rid for rid, _ in excl],
                    "token": cell["token"], "token_tier": cell["token_tier"],
                    "row_meta": {r["row_id"]: {k: r[k] for k in ("cohort", "protocol", "HL_train", "metric_key")}
                                 for r in rows}})
    return out


# ------------------------------------------------------------------------------------------ reduce packets
def build_reduce_packets(cells_text: str) -> list[dict]:
    """One packet per question (§9 S3): pointers into the rules header, all S1_cells rows of q (every class) and
    the list of expected map memos. No S1_table rows (the memos already carry row_ids); no other question."""
    ccols, cells, ccom = load_cells(cells_text)
    out = []
    for q, ss in UNIVERSE:
        rows = [cells[(q, s)] for s in ss]
        toks = {r["q_token"] for r in rows}
        if len(toks) != 1 or next(iter(toks)) not in Q_TOKENS:
            raise PacketError(f"S1_cells: question {q} has q_token {sorted(toks)}")
        qt = next(iter(toks))
        targets = [r for r in rows if r["class"] in S2_CLASSES]
        if qt in REDUCE_EXCLUDED:
            out.append({"q": q, "question_token": qt, "status": "skipped",
                        "reason": f"{qt}: no reduce request (PREREG §9 S3)", "text": None, "targets": []})
            continue
        exp = [f"{cell_id(q, r['s'])} | {r['class']} | " +
               (f"expected memos: {', '.join(MAP_MODELS)}" if r["class"] in S2_CLASSES else "no memo (not an S2 target)")
               for r in rows]
        parts = [
            f"# REDUCE PACKET {q}",
            f"packet_format: {REDUCE_FORMAT}",
            f"prereg_sha256: {PREREG_SHA256}",
            f"s1_cells_sha256: {sha256_text(cells_text)}",
            *[f"s1_comment: {c.lstrip('# ').rstrip()}" for c in ccom],
            f"q: {q}",
            f"question_token: {qt}",
            "",
            "## RULE POINTERS (all in the RULES HEADER)",
            f"- question row: §2.2/§2.3 table row **{q}**; claim -> source mapping: §2.4 rows that start with \"| {q} |\"",
            f"- question token {qt}: §7.2 row **{qt}**; cell classes: §7.1",
            "",
            "## CELLS OF THIS QUESTION AND EXPECTED MAP MEMOS (cell_id | class | memos)",
            *exp,
            "",
            "## CELL CITABLE REFERENCES",
            ", ".join(cell_ref(q, r["s"]) for r in rows),
            "",
            "## S1_cells ROWS FOR THIS QUESTION (verbatim CSV, every class)",
            csv_block(cells_cols_out(ccols), rows),
        ]
        text = "\n".join(parts) + "\n"
        forbid(f"reduce {q}", text)
        out.append({"q": q, "question_token": qt, "status": "built", "reason": "", "text": text,
                    "targets": [cell_id(q, r["s"]) for r in targets],
                    "classes": {cell_id(q, r["s"]): r["class"] for r in rows},
                    "tokens": {cell_id(q, r["s"]): (r["token"], r["token_tier"]) for r in rows}})
    return out


def group_memos(reduce_packet: dict, memos: list[dict]) -> list[dict]:
    """Strict: every S2 target cell of q has exactly one memo per map model (like u1.read_map_batch).
    memos: [{cell_id, model, note}] from the collected map batches. Returns the q's memos in packet order."""
    want = {(c, m) for c in reduce_packet["targets"] for m in MAP_MODELS}
    have: dict[tuple, dict] = {}
    for m in memos:
        key = (m["cell_id"], m["model"])
        if key in want:
            if key in have:
                raise PacketError(f"two memos for {key}")
            have[key] = m
    missing = sorted(want - set(have))
    if missing:
        raise PacketError(f"{reduce_packet['q']}: map not fully collected, missing {missing[:4]}; plan reduce later")
    return [have[(c, m)] for c in reduce_packet["targets"] for m in MAP_MODELS]


# ------------------------------------------------------------------------------------------ schema subset
def validate_schema(obj, schema: dict, path: str = "$") -> list[str]:
    """The structured-output subset the precedent review_note.json uses: type, properties, required,
    additionalProperties false, items, enum. Returns error strings (empty = valid)."""
    errs = []
    t = schema.get("type")
    py = {"object": dict, "array": list, "string": str, "boolean": bool}
    if t in py and not isinstance(obj, py[t]):
        return [f"{path}: expected {t}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in enum")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing {k}")
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected key {k}" for k in obj if k not in props]
        for k, v in obj.items():
            if k in props:
                errs += validate_schema(v, props[k], f"{path}.{k}")
    if t == "array":
        for i, v in enumerate(obj):
            errs += validate_schema(v, schema.get("items", {}), f"{path}[{i}]")
    return errs


def schema_keywords(schema, acc=None) -> set:
    acc = set() if acc is None else acc
    if isinstance(schema, dict):
        for k, v in schema.items():
            acc.add(k)
            if k == "properties":
                for sub in v.values():
                    schema_keywords(sub, acc)
            elif k == "items":
                schema_keywords(v, acc)
    return acc


# ------------------------------------------------------------------------------------------ post-checks
NUM_RE = re.compile(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?(?![A-Za-z_\d])")
VTOK_RE = re.compile(r"(?<![A-Za-z-])(supported|refuted|falsified-as-framed|conditional|partial)(?![A-Za-z])", re.I)
HIGH_PHRASES = ("rules out", "excludes", "proves", "deductive", "establishes", "iteration is the lever",
                "iteration is the mechanism", "truncation helps", "z/eff_rank is blind to collapse", "first to observe",
                "act cannot rescue", "배제한다", "증명한다", "증명했다", "연역")
MEDIUM_PHRASES = ("capacity ceiling", "retained region", "collapse ⇒ low pr", "collapse => low pr", "best", "governing",
                  "established", "demonstrates", "confirms", "shows that", "용량 한계")
CAUSAL_RE = re.compile(r"\bcaus\w*|\bdue to\b|\bbecause\b|\bdriv(?:e|es|en|ing)\b|\bresponsible for\b|\bleads? to\b|"
                       r"\bexplains?\b|\bnecessar\w*|\bsufficient\b|\brequire[sd]?\b|\bneeded\b|\bthe lever\b|"
                       r"원인|때문|야기|초래|기인|필요|필수|인과|덕분", re.I)
SEED_POOL_RE = re.compile(r"seed[^.;]{0,20}(?:평균|mean|averag|pool)|(?:평균|mean|averag|pool)[^.;]{0,20}seed|"
                          r"seed\s*평균|풀링", re.I)


def numbers(text: str) -> set:
    return set(NUM_RE.findall(text or ""))


def fixed_numbers(r3: str) -> set:
    """Numbers a proposed sentence may take from the FIXED STRINGS (R-3 text, R-11 form ...) and nowhere else in
    the header: header-wide numbers (rule IDs, § numbers, PREREG counts, dates) are not admissible in a sentence."""
    return numbers(" ".join([LEGACY_NOTE, r3, LOCKED_MARKER, H_STATUS, PREJUDGE_ONLY, UNKNOWN_KEY, MARGINAL, R8_CAP,
                             PROTOCOL_MIX, EXP025_FIRST]))


def cohort_tag_findings(where: str, declared: list[str], row_ids: list[str], row_meta: dict) -> list[dict]:
    cited = {row_meta[r]["cohort"] for r in row_ids if r in row_meta}
    if cited and set(declared) != cited:
        return [_finding(where, row_ids, "R-2", "high", f"cohort_tags {sorted(declared)} != cited rows' cohorts {sorted(cited)}")]
    return []


def _finding(where, row_ids, rule, severity, msg) -> dict:
    return {"where": where, "row_ids": ";".join(row_ids or []), "rule": rule, "severity": severity, "message": msg}


def sentence_findings(where: str, text: str, row_ids: list[str], row_meta: dict, allowed_numbers: set,
                      r3: str) -> list[dict]:
    """Rule checks for one proposed sentence (map candidate or reduce allowed sentence)."""
    f = []
    low = text.lower()
    new = sorted(numbers(text) - allowed_numbers)
    if new:
        f.append(_finding(where, row_ids, "N-number", "high", f"numbers not in the inputs: {new[:6]}"))
    for p in HIGH_PHRASES:
        if p in low:
            f.append(_finding(where, row_ids, "R-1/R-10", "high", f"banned phrase {p!r}"))
    for p in MEDIUM_PHRASES:
        if re.search(rf"(?<![A-Za-z]){re.escape(p)}(?![A-Za-z])", low):
            f.append(_finding(where, row_ids, "R-1/R-5/R-10", "medium", f"phrase {p!r} needs a human check"))
    m = CAUSAL_RE.search(text)
    if m:
        f.append(_finding(where, row_ids, "N-causal", "high", f"causal/necessity wording {m.group(0)!r}"))
    if SEED_POOL_RE.search(text):
        f.append(_finding(where, row_ids, "R-6", "high", "seed averaging / pooling wording"))
    if re.search(r"optimi[sz]ation|capacity|최적화|용량", text, re.I) and R8_CAP not in text:
        f.append(_finding(where, row_ids, "R-8", "medium", f"cause-of-collapse wording without exactly {R8_CAP!r}"))
    if "CL-0006" in text:
        f.append(_finding(where, row_ids, "R-9", "high", "CL-0006 cited"))
    if re.search(r"EXP-015", text.replace(EXP025_FIRST, "").replace("legacy alias EXP-015", "")):
        f.append(_finding(where, row_ids, "R-11", "high", "bare EXP-015"))
    meta = [row_meta[r] for r in row_ids if r in row_meta]
    cohorts = {m["cohort"] for m in meta}
    if "legacy" in cohorts and LEGACY_NOTE not in text:
        f.append(_finding(where, row_ids, "R-2", "high", "rests on a legacy row without the fixed legacy string"))
    if "legacy" in cohorts and cohorts - {"legacy"}:
        f.append(_finding(where, row_ids, "R-2", "high", "legacy and clean rows in one sentence"))
    hls = {hl(m["HL_train"]) for m in meta}
    if (ITER in hls and NOITER in hls) or (re.search(r"\bnoiter\b", low) and re.search(r"(?<!no)iter\b", low)):
        if ws(r3) not in ws(text):
            f.append(_finding(where, row_ids, "R-3", "high", "iter/noiter contrast without the verbatim R-3 text"))
    if len({m["protocol"] for m in meta}) > 1 and PROTOCOL_MIX not in text:
        f.append(_finding(where, row_ids, "R-12", "high", "rows of different protocols without the R-12 sentence"))
    if any(m["metric_key"] == "UNKNOWN" for m in meta) and UNKNOWN_KEY not in text:
        f.append(_finding(where, row_ids, "§7.1-6", "high", f"metric_key UNKNOWN row without {UNKNOWN_KEY!r}"))
    return f


def check_map_note(note: dict, packet: dict, schema: dict, input_texts: list[str], r3: str) -> list[dict]:
    """Deterministic post-check of one map memo against its packet (advisory lint; PREREG §9 V2 is the gate)."""
    cid = packet["cell_id"]
    f = [_finding(cid, [], "schema", "high", e) for e in validate_schema(note, schema)]
    if f:
        return f
    allowed = set().union(*(numbers(t) for t in input_texts))         # evidence / suggested_check
    strict = numbers(packet["text"]) | fixed_numbers(r3)              # proposed sentences: packet + fixed strings
    cites = set(packet["citable_ids"])
    if note["scope_id"] != cid:
        f.append(_finding(cid, [], "identity", "high", f"scope_id {note['scope_id']!r} != {cid}"))
    if note["cell_class"] != packet["cell_class"]:
        f.append(_finding(cid, [], "identity", "high", f"cell_class {note['cell_class']} != {packet['cell_class']}"))
    if packet["mode"] == "LOCKED" and note["candidate_sentences"]:
        f.append(_finding(cid, [], "§9-S2", "high", "candidate sentences for a locked (D-LOCKED/D-PREJUDGE) cell"))
    for arr in ("scope_mismatches", "candidate_sentences", "rule_findings"):
        for i, it in enumerate(note[arr]):
            where = f"{cid}/{arr}[{i}]"
            rid = it["row_ids"]
            if not rid:
                f.append(_finding(where, [], "row_id", "high", "item without row_ids"))
            nc = [r for r in rid if r in packet["not_citable"]]
            if nc:
                f.append(_finding(where, rid, "evidence_use", "high", f"cites rows whose evidence_use is 'no': {nc[:4]}"))
            bad = [r for r in rid if r not in cites]
            if bad:
                f.append(_finding(where, rid, "row_id", "high", f"row_ids not citable in this packet: {bad[:4]}"))
            for fld in ("evidence", "suggested_check"):
                if fld in it:
                    new = sorted(numbers(it[fld]) - allowed)
                    if new:
                        f.append(_finding(where, rid, "N-number", "high", f"{fld}: numbers not in the inputs {new[:6]}"))
    toks = {t for t in re.split(r"[|;, ]+", packet["token"] or "") if t in V_TOK}
    for i, c in enumerate(note["candidate_sentences"]):
        where = f"{cid}/candidate_sentences[{i}]"
        f += sentence_findings(where, c["text"], c["row_ids"], packet["row_meta"], strict, r3)
        f += cohort_tag_findings(where, c["cohort_tags"], c["row_ids"], packet["row_meta"])
        words = {w.lower() for w in VTOK_RE.findall(c["text"])}
        if packet["cell_class"] == "A-VERDICT":
            if c["token"] not in toks:
                f.append(_finding(where, c["row_ids"], "N-token", "high", f"token {c['token']!r} not inherited {sorted(toks)}"))
            if c["tier"] != (packet["token_tier"] or "none"):
                f.append(_finding(where, c["row_ids"], "N-token", "high", f"tier {c['tier']!r} != {packet['token_tier']!r}"))
            if words - toks:
                f.append(_finding(where, c["row_ids"], "N-token", "high", f"verdict words beyond the token: {sorted(words - toks)}"))
        else:
            if c["token"] != "none" or c["tier"] != "none":
                f.append(_finding(where, c["row_ids"], "N-token", "high", "token/tier must be none outside A-VERDICT"))
            if words:
                f.append(_finding(where, c["row_ids"], "N-token", "high", f"verdict word in a {packet['cell_class']} sentence"))
        if packet["question_token"] == "Q-DESC-ONLY" and LOCKED_MARKER not in c["text"]:
            f.append(_finding(where, c["row_ids"], "§7.2", "high", f"Q-DESC-ONLY sentence without {LOCKED_MARKER!r}"))
    return f


def check_reduce_note(note: dict, reduce_packet: dict, map_packets: list[dict], schema: dict,
                      input_texts: list[str], r3: str, memos: list[dict] | None = None) -> list[dict]:
    """Deterministic post-check of one reduce output for question q. With `memos` (the q's map memos), numbers in
    allowed sentences must come from the reduce packet, the memos' candidate sentences or the fixed strings."""
    q = reduce_packet["q"]
    f = [_finding(q, [], "schema", "high", e) for e in validate_schema(note, schema)]
    if f:
        return f
    allowed = set().union(*(numbers(t) for t in input_texts))
    if memos is not None:
        allowed = numbers(reduce_packet["text"]) | fixed_numbers(r3) | set().union(
            *(numbers(c["text"]) for m in memos for c in (m.get("note") or {}).get("candidate_sentences", [])))
    mp = {p["cell_id"]: p for p in map_packets if p["q"] == q}
    classes = reduce_packet["classes"]
    cites, excluded = set(), set()
    row_meta: dict = {}
    for c in classes:
        cites.add(f"CELL:{c}")
    for p in mp.values():
        cites |= set(p["citable_ids"])
        excluded |= set(p["not_citable"])
        row_meta.update(p["row_meta"])
    if note["q"] != q or note["q_token"] != reduce_packet["question_token"]:
        f.append(_finding(q, [], "identity", "high", "q / q_token do not match the reduce packet"))
    seen = {c["cell_id"] for c in note["cells_seen"]}
    if set(reduce_packet["targets"]) - seen:
        f.append(_finding(q, [], "coverage", "high", f"target cells not in cells_seen: {sorted(set(reduce_packet['targets']) - seen)[:4]}"))
    qt = reduce_packet["question_token"]
    if qt == "Q-UNCOVERED" and note["allowed_sentences"]:
        f.append(_finding(q, [], "§7.2", "high", "Q-UNCOVERED question with allowed sentences"))
    for i, a in enumerate(note["allowed_sentences"]):
        where = f"{q}/allowed_sentences[{i}]"
        rid = a["row_ids"]
        if not rid:
            f.append(_finding(where, [], "row_id", "high", "sentence without row_ids"))
        nc = [r for r in rid if r in excluded]
        if nc:
            f.append(_finding(where, rid, "evidence_use", "high", f"cites rows whose evidence_use is 'no': {nc[:4]}"))
        bad = [r for r in rid if r not in cites]
        if bad:
            f.append(_finding(where, rid, "row_id", "high", f"row_ids not citable for {q}: {bad[:4]}"))
        unknown = [c for c in a["cell_ids"] if c not in classes]
        if unknown or not a["cell_ids"]:
            f.append(_finding(where, rid, "cell_id", "high", f"cell_ids empty or not of {q}: {unknown[:4]}"))
        locked = [c for c in a["cell_ids"] if classes.get(c) in LOCKED_CLASSES]
        if locked:
            f.append(_finding(where, rid, "§6.2", "high", f"locked cells used as a sentence basis: {locked}"))
        f += sentence_findings(where, a["text"], rid, row_meta, allowed, r3)
        f += cohort_tag_findings(where, a["cohort_tags"], rid, row_meta)
        words = {w.lower() for w in VTOK_RE.findall(a["text"])}
        verdict_cells = [c for c in a["cell_ids"] if classes.get(c) == "A-VERDICT"]
        toks = set()
        for c in verdict_cells:
            toks |= {t for t in re.split(r"[|;, ]+", reduce_packet["tokens"][c][0] or "") if t in V_TOK}
        if qt == "Q-DESC-ONLY":
            if words or a["token"] != "none":
                f.append(_finding(where, rid, "§7.2", "high", "verdict word/token in a Q-DESC-ONLY question"))
            if LOCKED_MARKER not in a["text"]:
                f.append(_finding(where, rid, "§7.2", "high", f"Q-DESC-ONLY sentence without {LOCKED_MARKER!r}"))
        elif a["token"] != "none" and a["token"] not in toks:
            f.append(_finding(where, rid, "N-token", "high", f"token {a['token']!r} not inherited by its cells"))
        if words - toks:
            f.append(_finding(where, rid, "N-token", "high", f"verdict words not inherited: {sorted(words - toks)}"))
        if a["is_headline"]:
            ok = verdict_cells and len(verdict_cells) == len(a["cell_ids"]) and \
                all("supported" == (reduce_packet["tokens"][c][0] or "").strip() for c in verdict_cells) and \
                not any(c.split("__", 1)[1].startswith("FG-") for c in verdict_cells) and len(set(a["cohort_tags"])) == 1
            if not ok:
                f.append(_finding(where, rid, "§7.2-3", "high", "combined headline without its §7.2 conditions"))
    for d in note["disagreements"]:
        if d["cell_id"] not in classes:
            f.append(_finding(q, d["row_ids"], "cell_id", "high", f"disagreement on a cell outside {q}"))
    return f


def render_reduce_md(note: dict, reduce_packet: dict) -> str:
    """Deterministic Markdown rendering of a reduce output (PREREG §9 S3: one Markdown per question)."""
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    L = [f"# {note['q']} - S3 reduce (DRAFT, api-batch output - not evidence)", "",
         f"question_token: {note['q_token']} | prereg_sha256: {PREREG_SHA256[:12]}... | rendered by packets.py", "",
         "## Scope", "", "| cell_id | class | sonnet memo | opus memo |", "|---|---|---|---|"]
    L += [f"| {cell(c['cell_id'])} | {c['cell_class']} | {c['sonnet_memo']} | {c['opus_memo']} |" for c in note["cells_seen"]]
    L += ["", "## Allowed sentences", "",
          "| # | sentence | row_ids | cells | token | tier | cohort | rules | proposed_by | headline |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(note["allowed_sentences"], 1):
        L.append(f"| {i} | {cell(a['text'])} | {cell(', '.join(a['row_ids']))} | {cell(', '.join(a['cell_ids']))} | "
                 f"{a['token']} | {a['tier']} | {cell(', '.join(a['cohort_tags']))} | {cell(', '.join(a['rules_applied']))} | "
                 f"{a['proposed_by']} | {a['is_headline']} |")
    L += ["", f"## Status-only cells ({LOCKED_MARKER})", ""]
    L += [f"- {cell(s['cell_id'])} ({s['cell_class']}; {cell(', '.join(s['row_ids']))}): {cell(s['marker'])}"
          for s in note["status_only_cells"]] or ["- none"]
    L += ["", "## Sonnet/Opus disagreements (not resolved by vote)", "",
          "| cell_id | row_ids | topic | claude-sonnet-5-5 | claude-opus-5-5 |", "|---|---|---|---|---|"]
    L += [f"| {cell(d['cell_id'])} | {cell(', '.join(d['row_ids']))} | {cell(d['topic'])} | {cell(d['sonnet_text'])} | "
          f"{cell(d['opus_text'])} |" for d in note["disagreements"]] or ["| none | | | | |"]
    L += ["", "## Open rule findings", "", "| cell_id | row_ids | rule | kind | severity | from | evidence |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {cell(o['cell_id'])} | {cell(', '.join(o['row_ids']))} | {o['rule']} | {o['kind']} | {o['severity']} | "
          f"{o['from_models']} | {cell(o['evidence'])} |" for o in note["open_findings"]] or ["| none | | | | | | |"]
    L += ["", "## UNCOVERED", ""]
    L += [f"- {cell(u['cell_id'])}: {cell(u['reason'])}" for u in note["uncovered"]] or ["- none"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ S4 inputs, staging
def s4_slices(table_text: str, cells_text: str) -> dict[str, dict[str, str]]:
    """Per-question S1 slices for the S4 draft-sync spec: {q: {S1_cells: csv, S1_table: csv}} (verbatim rows,
    allow-listed columns; S1_table rows whose evidence_use starts with 'no' are left out: not citable)."""
    s1 = load_s1(table_text, cells_text)
    out = {}
    for q, ss in UNIVERSE:
        crow = [s1["cells"][(q, s)] for s in ss]
        trow = [r for s in ss for r in s1["rows"][(q, s)]]
        out[q] = {"S1_cells": csv_block(cells_cols_out(s1["cells_cols"]), crow) + "\n",
                  "S1_table": csv_block(table_cols_out(s1["table_cols"]), trow) + "\n"}
    return out


def read_hashes(path: Path) -> dict[str, str]:
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            h, rel = ln.split(None, 1)
            out[rel.strip().lstrip("*")] = h
    return out


def stage(files: list[str], dest: Path, hashes: Path) -> list[str]:
    """Copy template files (repo-relative paths listed in HASHES.sha256) into a batch / plan directory: `*.spec.yaml`
    and `spec.yaml` become <dest>/spec.yaml, everything else goes to <dest>/shared/ (tools/api_batch L1 admits
    prompts only from <batch>/shared/*.md). Each file's sha256 must equal its HASHES entry; any mismatch or an
    existing target with other bytes refuses before anything is written."""
    table = read_hashes(hashes)
    plan = []
    for rel in files:
        if rel not in table:
            raise PacketError(f"stage: {rel} is not listed in {hashes}")
        src = REPO_ROOT / rel
        if sha256_file(src) != table[rel]:
            raise PacketError(f"stage: {rel} sha256 differs from HASHES.sha256 (edited after witness)")
        tgt = Path(dest) / "spec.yaml" if src.name.endswith("spec.yaml") else Path(dest) / "shared" / src.name
        if tgt.exists() and sha256_file(tgt) != table[rel]:
            raise PacketError(f"stage: {tgt} exists with different bytes")
        plan.append((src, tgt))
    if len({t for _s, t in plan}) != len(plan):
        raise PacketError("stage: two files map to the same target")
    done = []
    for src, tgt in plan:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, tgt)
        done.append(str(tgt))
    return done


def _load_notes(d: Path) -> list[tuple[str, str, dict]]:
    """<cell_id or q>.<model>.json files -> [(stem, model, note)]; the model must be a known seat model."""
    out = []
    for f in sorted(Path(d).glob("*.json")):
        stem, _, model = f.stem.rpartition(".")
        if model not in MAP_MODELS + (REDUCE_MODEL,):
            raise PacketError(f"{f.name}: expected <id>.<model>.json with model in {MAP_MODELS + (REDUCE_MODEL,)}")
        out.append((stem, model, json.loads(f.read_text(encoding="utf-8"))))
    return out


def _write_findings(rows: list[dict], out: str | None) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["model", "where", "row_ids", "rule", "severity", "message"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    if out:
        Path(out).write_text(buf.getvalue(), encoding="utf-8")
    else:
        sys.stdout.write(buf.getvalue())


# ------------------------------------------------------------------------------------------ CLI
def _read(p: str) -> str:
    return Path(p).read_text(encoding="utf-8")


def _write_set(out: Path, items: list[tuple[str, str]], index_cols: list[str], index_rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in items:
        (out / name).write_text(text, encoding="utf-8")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=index_cols, lineterminator="\n")
    w.writeheader()
    w.writerows(index_rows)
    (out / "index.csv").write_text(buf.getvalue(), encoding="utf-8")
    sums = "".join(f"{sha256_text(t)}  {n}\n" for n, t in items)
    (out / "packets.sha256").write_text(sums, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="packets", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("header", help="print / write the shared rules header")
    h.add_argument("--prereg", default=str(PREREG_PATH))
    h.add_argument("--out")
    m = sub.add_parser("map", help="S2 map packets, one per target cell")
    m.add_argument("--s1-table", required=True)
    m.add_argument("--s1-cells", required=True)
    m.add_argument("--r3-ii-cite", help="file:line of EXP-011's compute-matching definition, if S0 found one (R-3 (ii))")
    m.add_argument("--out", required=True)
    r = sub.add_parser("reduce", help="S3 reduce packets, one per question")
    r.add_argument("--s1-cells", required=True)
    r.add_argument("--out", required=True)
    s4 = sub.add_parser("s4-inputs", help="per-question S1 slices for the S4 draft-sync spec")
    s4.add_argument("--s1-table", required=True)
    s4.add_argument("--s1-cells", required=True)
    s4.add_argument("--out", required=True)
    cm = sub.add_parser("check-map", help="post-check collected map memos (<cell_id>.<model>.json); exit 1 on a high finding")
    cm.add_argument("--s1-table", required=True)
    cm.add_argument("--s1-cells", required=True)
    cm.add_argument("--notes", required=True)
    cm.add_argument("--header", default=str(TEMPLATES_DIR / "shared/siz_rules_header.md"))
    cm.add_argument("--system", default=str(TEMPLATES_DIR / "shared/siz_map_system.md"))
    cm.add_argument("--schema", default=str(TEMPLATES_DIR / "schemas/siz_review_note.json"))
    cm.add_argument("--r3-ii-cite")
    cm.add_argument("--out")
    cr = sub.add_parser("check-reduce", help="post-check reduce outputs (<q>.claude-fable-5-1.json) and render Markdown")
    cr.add_argument("--s1-table", required=True)
    cr.add_argument("--s1-cells", required=True)
    cr.add_argument("--notes", required=True)
    cr.add_argument("--memos", required=True, help="the map memo directory checked by check-map")
    cr.add_argument("--header", default=str(TEMPLATES_DIR / "shared/siz_rules_header.md"))
    cr.add_argument("--system", default=str(TEMPLATES_DIR / "shared/siz_reduce_system.md"))
    cr.add_argument("--schema", default=str(TEMPLATES_DIR / "schemas/siz_reduce_note.json"))
    cr.add_argument("--r3-ii-cite")
    cr.add_argument("--render-to", help="write <q>.reduce.md here (S4_inputs) for notes without a high finding")
    cr.add_argument("--out")
    st = sub.add_parser("stage", help="copy sha-verified templates into a batch / plan dir (spec.yaml + shared/)")
    st.add_argument("--to", required=True)
    st.add_argument("--hashes", default=str(TEMPLATES_DIR / "HASHES.sha256"))
    st.add_argument("files", nargs="+", help="repo-relative paths as listed in HASHES.sha256")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "header":
            text = build_rules_header(_read(a.prereg))
            if a.out:
                Path(a.out).write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
        elif a.cmd == "map":
            pk = build_map_packets(_read(a.s1_table), _read(a.s1_cells), a.r3_ii_cite)
            _write_set(Path(a.out), [(f"{p['cell_id']}.md", p["text"]) for p in pk],
                       ["cell_id", "q", "s", "kind", "cell_class", "question_token", "mode", "packet_sha256"],
                       [{k: p[k] for k in ("cell_id", "q", "s", "kind", "cell_class", "question_token", "mode")}
                        | {"packet_sha256": sha256_text(p["text"])} for p in pk])
            print(json.dumps({"map_packets": len(pk), "locked": sum(p["mode"] == "LOCKED" for p in pk), "out": a.out}))
        elif a.cmd == "reduce":
            pk = build_reduce_packets(_read(a.s1_cells))
            built = [p for p in pk if p["status"] == "built"]
            _write_set(Path(a.out), [(f"{p['q']}.md", p["text"]) for p in built],
                       ["q", "question_token", "status", "reason", "targets", "packet_sha256"],
                       [{"q": p["q"], "question_token": p["question_token"], "status": p["status"],
                         "reason": p["reason"], "targets": ";".join(p["targets"]),
                         "packet_sha256": sha256_text(p["text"]) if p["text"] else ""} for p in pk])
            print(json.dumps({"reduce_packets": len(built), "skipped": [p["q"] for p in pk if p["status"] != "built"]}))
        elif a.cmd == "s4-inputs":
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            for q, d in s4_slices(_read(a.s1_table), _read(a.s1_cells)).items():
                for nm, text in d.items():
                    (out / f"{q}.{nm}.csv").write_text(text, encoding="utf-8")
            print(json.dumps({"s4_inputs": str(out)}))
        elif a.cmd == "check-map":
            pk = {p["cell_id"]: p for p in build_map_packets(_read(a.s1_table), _read(a.s1_cells), a.r3_ii_cite)}
            schema = json.loads(_read(a.schema))
            fixed = [_read(a.header), _read(a.system)]
            rows, seen = [], set()
            for stem, model, note in _load_notes(Path(a.notes)):
                if stem not in pk:
                    raise PacketError(f"note {stem}.{model}.json has no map packet")
                seen.add((stem, model))
                for x in check_map_note(note, pk[stem], schema, fixed + [pk[stem]["text"]], r3_text_variant(a.r3_ii_cite)):
                    rows.append({"model": model, **x})
            for c in pk:
                for mdl in MAP_MODELS:
                    if (c, mdl) not in seen:
                        rows.append({"model": mdl, **_finding(c, [], "coverage", "high", "memo missing")})
            _write_findings(rows, a.out)
            return 1 if any(r["severity"] == "high" for r in rows) else 0
        elif a.cmd == "check-reduce":
            mp = build_map_packets(_read(a.s1_table), _read(a.s1_cells), a.r3_ii_cite)
            rp = {p["q"]: p for p in build_reduce_packets(_read(a.s1_cells)) if p["status"] == "built"}
            memos = [{"cell_id": st_, "model": m_, "note": n_} for st_, m_, n_ in _load_notes(Path(a.memos))]
            schema = json.loads(_read(a.schema))
            fixed = [_read(a.header), _read(a.system)]
            rows = []
            for q, model, note in _load_notes(Path(a.notes)):
                if q not in rp or model != REDUCE_MODEL:
                    raise PacketError(f"reduce note {q}.{model}.json: no reduce packet or wrong model")
                mine = group_memos(rp[q], memos)
                texts = fixed + [rp[q]["text"]] + [json.dumps(m["note"], ensure_ascii=False) for m in mine]
                fs = check_reduce_note(note, rp[q], mp, schema, texts, r3_text_variant(a.r3_ii_cite), memos=mine)
                rows += [{"model": model, **x} for x in fs]
                if a.render_to and not any(x["severity"] == "high" for x in fs):
                    Path(a.render_to).mkdir(parents=True, exist_ok=True)
                    (Path(a.render_to) / f"{q}.reduce.md").write_text(render_reduce_md(note, rp[q]), encoding="utf-8")
            _write_findings(rows, a.out)
            return 1 if any(r["severity"] == "high" for r in rows) else 0
        elif a.cmd == "stage":
            print(json.dumps({"staged": stage(a.files, Path(a.to), Path(a.hashes))}))
    except PacketError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
