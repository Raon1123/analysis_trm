#!/usr/bin/env python3
"""S1 (deterministic table) for 2026-10-30_state-iter-z-synthesis (PREREG sha256 87fa6e7a...3e1b, immutable).

STATUS: WRITTEN, NEVER EXECUTED AGAINST REAL DATA.  PREREG sec.12 step 3: the S2/S3 template hashes must be
witnessed before any metric value is extracted.  Do not run without --dry-run until that witness exists.

Inputs (all four S0 outputs + the S0 snapshot; S1 reproduces sec.7 from the snapshot, never from live state files):
  S0r2_cells_expected.csv  S0r2_metric_keys.csv  S0r2_inventory.csv  S0r2_tokens.csv  S0r2_class_repro_check.csv  S0r2_snapshot/{as_of.txt,INDEX.jsonl,...}
  (default prefix S0r2; the superseded r1 files S0_* are read only with --s0-prefix S0)
Live reads (value-bearing, only for cells whose S1 class is A-DESC or D-LOCKED and whose key_status is NAMED-ADMITTED):
  wandb/run-*/files/wandb-metadata.json (run name/project only), wandb/run-*/run-*.wandb (history of the admitted
  keys only), checkpoints/<project>/<run>/all_config.yaml (arch/HL/seed/data_paths), <data_path>/test/dataset.json.
  wandb-summary.json is never opened (it holds metric values).

Outputs (PREREG sec.9 S1 / sec.11), written only into --out and only if no S1_* file exists (else exit 3):
  S1_table.csv        one row per (cell, run, metric_key); deterministic row_id; seed-wise only (no mean/pool, R-6)
  S1_cells.csv        58 rows: class (sec.7.1 re-derived from the snapshot), q-token (sec.7.2), reason when != expected
  S1_access_log.csv   every file opened / would-be-opened / denied / name-only, with sha256 and UTC time
  S1_manifest.sha256  script/input shas written BEFORE any value is read, output shas appended at the end
Naming note: the orchestrator task text said S1_access_log.tsv; PREREG sec.9/sec.11 say S1_access_log.csv.
PREREG wins (override with --access-log-name).

Guards (PREREG sec.4, sec.6.3 O-2, sec.7.1, sec.9 S1):
  * D-PREJUDGE / D-SEALED / X-PROV / X-ABSENT / X-OUT / LISTED-ONLY / UNCOVERED cells: zero value adapters are called.
    D-PREJUDGE gets one name_only access-log line (queue counts from S0r2_inventory.csv).
  * Runs that belong to a D-PREJUDGE source (EXP-020 e020_*, EXP-023 pp_ltf_*_s{4,5,6}) are excluded from EVERY
    other source's run selection (O-2 guard; kept as a superset guard even though the anchored s0r2b regexes of CR-2 already
    do not reach them).
  * Any path with a component equal to 'sealed' / 'z_snapshots' or containing 'preds' is refused (DeniedPath).
  * UNKNOWN key_status -> metric_key=UNKNOWN, no adapter call (PREREG sec.14.2 U-22).

Interpretation: INTERP_S1.md r2 (same dir) fixes one reading per ambiguous clause; priority PREREG > note.  Code follows it
(CR-1..CR-10 applied):
  CR-1 inputs = S0r2 files (default prefix S0r2)
  | class gate: S1 re-derives the sec.7.1 class from the S0r2 CSVs; it must equal S0r2_class_repro_check.csv and the PREREG sec.7.0.5
    expectation, except the two known cells (Q-I2,EXP-011) and (Q-I4,EXP-011) (expected A-DESC, S0r2 X-ABSENT: r1 queue regex
    `_abl_` false positive on pp_abl_*); any other difference, or a wrong class count (19/21/5/3/5/2/2/1), exits 2 before any write
  | CR-2 run selectors = anchored s0r2b.py PROBE ckpt regexes; CR-6 (Q-Z1,EXP-025) selects ^pp_base_tf_z_iter
  | CR-3 reason text on the two EXP-011 cells
  | CR-4 (Q-Z4,EXP-006): pin key probe/test_exact read as R-FINAL, protocol P-PROBE, note "τ 아님"; no tau row is ever created
    (EXP-022 still opens nothing; CR-9 metric_keys_read is empty on every cell that does not extract)
  | CR-5 D-LOCKED E cells: the RESULTS frontmatter status line is restated verbatim only if the live file sha256 equals the pin in
    S0r2b_evidence_files.sha256; otherwise a DISCREPANCY is printed and nothing is restated
  | CR-7 manifest part 1 pins the S0r2 files, the pins file, s0r2a/b/c.py, INTERP_S1.md and the run-family map (MISSING if absent)
  | CR-8 every table row's note carries the label_GC note (S1 does not apply the source G/C definitions; NONE = raw value only)
  | CR-10 a run counts as evidence use only if all its wandb dir rows are "yes"
  cohort needs project + data path + wandb startedAt/started_at (time-zone window -> UNKNOWN) | exp_owner via ID_ALIASES-normalised
  run-family map (UNKNOWN rows stay in S1_table with evidence_use="no: ...") | token_* only from PREREG 5.1 tiers 1/2/4/3 (tier 5
  -> registry_status_N) | no placeholder in token_V | counts are run counts | eval_set_size only for clean cohorts |
  PR protocol = UNKNOWN + note.
STILL UNRESOLVED (left UNKNOWN / flagged, never interpreted): budget (no total_steps key) | weights for P-PROBE (U-2) | HL_eval |
  Tk_class | label_GC always NONE (with note) | run->source attribution = s0r2b regex (provisional) |
  run-family map is a DRAFT dated 2026-10-04 and not in S0r2_snapshot (EXP-011 abl_* absent from it -> owner UNKNOWN) |
  EXPERIMENTS.md owner level not machine-readable beyond the map's quoted evidence | access log name csv (PREREG) vs tsv (task text).

Run:      CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s1.py --dry-run
Tests:    CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_state_iter_z_s1.py
Exit: 0 ok; 2 gate/consistency failure (PREREG sha, S0 input counts, missing snapshot); 3 refuse to overwrite.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

PREREG_REL = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md"
PREREG_SHA = "87fa6e7acf2ba7278c4dea1835c1c27790d1bfe519b2e966374568b300c73e1b"
OUT_REL = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis"
LEGACY_NOTE = "legacy, 신규 clean 코호트와 비교 불가"  # PREREG sec.3.2 cohort_note / R-2 fixed string
UNKNOWN_KEY_NOTE = "키 이름 미고정 — 값 없음"  # PREREG sec.7.1 branch 6 / U-22
SEALED_NOTE = "sealed 종점 — 이 합성에서 사용 안 함"
PREJUDGE_NOTE = "판정 전 — 상태만"

# ------------------------------------------------------------------ fixed vocab (PREREG sec.0, 3.2.1, 7.0)
L5 = {"EXP-016", "EXP-017", "EXP-018", "EXP-020", "EXP-023"}
LH = {"H-200", "H-201", "H-202"}
P_PROV = {"EXP-010"}
P_LIST = {"CL-0006"}
P_SEALED = {"EXP-012/sealed"}
LEGACY_SOURCES = {"EXP-001", "EXP-002", "EXP-004", "EXP-005", "EXP-006", "EXP-007"}  # PREREG sec.7.4 note / R-2
XABS_WORDS = {"planned", "pre-registered-unrun", "draft-prereg"}
VALUE_CLASSES = {"A-DESC", "D-LOCKED"}  # the only classes that may reach a value adapter (sec.9 S1)
NO_VALUE_CLASSES = {"X-PROV", "LISTED-ONLY", "D-SEALED", "X-ABSENT", "D-PREJUDGE", "X-OUT", "UNCOVERED", "A-VERDICT"}

# short key name used in S0r2_metric_keys.csv N_named  ->  (wandb key string, protocol label or None)
KEYMAP = {
    "probe": ("probe/test_exact", "P-PROBE"),
    "all": ("all/exact_accuracy", "P-ALL"),
    "recomp": ("recompute_test_acc", "P-RECOMP"),
    "effr": ("z/eff_rank", None),  # PR family: no P-* label exists in the sec.3.2 vocabulary (reported ambiguity)
    "pin": ("probe/test_exact", "P-PROBE"),  # R24: EXP-006 pin key (S0r2_metric_keys R24 evidence = probe/test_exact); INTERP_S1 (2)/CR-4
}
LABEL_GC_NOTE = "S1은 소스 G/C 정의를 적용하지 않음(NONE = 원값만)"  # INTERP_S1 CR-8
KEY_NOTES = {"pin": "Q-Z4 pin 키 R-FINAL — τ 아님"}  # INTERP_S1 (2): no tau row is ever created
# adapters actually wired.  recomp lives in analysis/data CSVs (pr_recompute.py output): no adapter, never reached by
# the 27-row constant table (K(q) never contains it).  pin = the raw R-FINAL of the pin key, never a tau.
WIRED = {"probe", "all", "effr", "pin"}
# K(q) from PREREG sec.3.2.1 (3); None = "no K(q)" (Q-I4, Q-Z6: all of N is read).  P-RELOAD / P-DECODE have no fixed string.
K_SHORT = {
    "Q-I1": set(), "Q-I2": {"probe"}, "Q-I3": {"probe"}, "Q-I4": None, "Q-Z1": {"probe"}, "Q-Z2": {"probe"},
    "Q-Z3": {"effr"}, "Q-Z4": {"pin"}, "Q-Z5": set(), "Q-Z6": None,
}
# run selection = s0r2b.py PROBE ckpt regexes (sha dd749905...; anchored, source = snapshot EXPERIMENTS.md run-name schema rows),
# INTERP_S1 (5)/CR-2.  (project, regex).  Only (Q-Z1, EXP-025) differs: PREREG 3.1/7.0.2 names the reference cohort pp_base_tf_z_iter (CR-6).
_Z, _IT = r"(?:z|noz)", r"(?:iter|noiter)"
RUN_SELECT = {
    "EXP-001": ("Sigma_k_fig12", rf"^fig1_tf_{_Z}_{_IT}_k\d+_s\d+$"),
    "EXP-002": ("Sigma_k_fig12", rf"^fig1_tf_{_Z}_{_IT}_k\d+_s\d+$"),
    "EXP-004": ("Sigma_k_fig12", r"."),
    "EXP-005": ("Sigma_k_fig12", rf"^fig1_mlp_{_Z}_{_IT}_k\d+_s\d+$"),
    "EXP-006": ("Sigma_k_fig12", rf"^fig1_tf_{_Z}_{_IT}_k\d+_s\d+$"),
    "EXP-007": ("Sigma_k_fig12", r"^fig1d_(?:tf|mlp)_(?:noz|z)_iter_k(?:9|11)_s[123]$"),
    "EXP-011": ("Sigma_k_new", r"^abl_"),
    "EXP-012/probe": ("Sigma_k_new", r"^cleanr1_d[01]_k\d+_sm[123]$"),
    "EXP-016": ("power_permutation", r"^e016_(?:fig1_tf_noz_iter|act_noz_iter_h16|act_ltf_fullgrad_h16|act_ltf_fullgrad_fixed_h16)_k\d+_s\d+$"),
    "EXP-025": ("power_permutation", r"^pp_ltf_(?:fullgrad|g7|noinj)_lay2_k\d+_s[123]$"),
}
RUN_SELECT_BY_Q = {("EXP-025", "Q-Z1"): ("power_permutation", r"^pp_base_tf_z_iter")}
RUN_SELECT_BASIS = "PROVISIONAL(s0r2b.py PROBE ckpt regex; PREREG defines no run->source rule)"
# run-name patterns that belong to D-PREJUDGE sources; excluded from every other source (O-2 back-door guard)
PREJUDGE_RUN_PATTERNS = {"EXP-020": (r"^e020_", r"e020_s[12]"), "EXP-023": (r"^pp_ltf_.*_s[456]$",)}  # superset of s0r2b EXP-020/023 regexes
N10_DATA_PREFIXES = ("data/sigma_k_10/", "data/power_permutation/")  # PREREG sec.0: n=10 only; data/sigma_k/ (n=20) excluded
COHORT_BY_PROJECT = {"Sigma_k_fig12": ("legacy", None), "Sigma_k_new": ("clean-D0", "data/sigma_k_10/"),
                     "power_permutation": ("clean-D1", "data/power_permutation/d1/")}
# INTERP_S1.md (6)/TN-6: PREREG gives t_build "2026-07-20 13:18" without a time zone -> take the unfavourable reading both ways.
T_BUILD_KST = datetime.datetime(2026, 7, 20, 4, 18, tzinfo=datetime.timezone.utc)   # 13:18 KST
T_BUILD_UTC = datetime.datetime(2026, 7, 20, 13, 18, tzinfo=datetime.timezone.utc)  # 13:18 UTC
T_CLEAN_START = datetime.datetime(2026, 7, 20, 13, 29, tzinfo=datetime.timezone.utc)  # INTERP_S1 (6): memory-derived, narrows only
T_K11_START = datetime.datetime(2026, 7, 22, 0, 0, tzinfo=datetime.timezone.utc)      # k=11 path rebuilt 07-21 (memory-derived)
# INTERP_S1.md (10)/CD-9: cells whose PREREG 2.4 row names a component of the hypothesis card; S1 has only the card token
COMPONENT_CELLS = {("Q-I2", "H-022"): "O(log k) upper-bound component", ("Q-Z3", "H-023"): "PR-signature sub-token"}
D_LOCKED_MARKER = "현황 기술(확증 아님)"
RUN_FAMILY_MAP_REL = "lab/24_audits/11_lab-audits/2026-10-04_run-family-exp-map.yaml"  # PREREG 3.2 exp_owner priority 3 (DRAFT)
S1_OUTPUT_NAMES = ("S1_table.csv", "S1_cells.csv", "S1_access_log.csv", "S1_manifest.sha256")

CELL_FIELDS = ["q", "s", "kind", "expected_class", "s1_class", "s0_confirmed_class", "branch", "class_match", "reason", "q_token",
               "key_status", "metric_keys_read", "values_policy", "n_table_rows", "n_runs", "n_runs_evidence_use",
               "counts_by_cohort", "cohort_note",
               "token_tier", "token_V", "token_N", "token_file", "token_locator", "token_scope", "token_discrepancy",
               "token_note", "token_restatement", "marker", "registry_status_N",
               "atom_cohort", "atom_verdict_scope", "atom_prohibited_upgrade", "notes"]
TABLE_FIELDS = ["row_id", "q", "s", "cell_class", "metric_key", "protocol", "value", "step", "reduction", "frame",
                "n_logged_points", "label_GC", "run_name", "wandb_run_id", "wandb_project", "provenance_file",
                "provenance_sha256", "weights", "eval_set_size", "probe_size", "budget", "cohort", "cohort_note", "started_at",
                "arch", "HL_train", "HL_eval", "block", "seed", "k", "Tk_class", "exp_owner", "exp_owner_basis",
                "evidence_use", "run_select_basis", "dup_wandb_dirs", "protocol_note", "note"]
ACCESS_FIELDS = ["ts_utc", "cell", "event", "adapter", "path", "sha256", "purpose", "note"]


class DeniedPath(Exception):
    pass


class GateFailure(Exception):
    pass


# ------------------------------------------------------------------ small utils
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def is_denied(path) -> bool:
    """PREREG sec.4: **/sealed/**, **/preds*, **/z_snapshots/**.  Stricter on purpose: any component containing 'preds'."""
    for comp in Path(str(path)).parts:
        c = comp.lower()
        if c == "sealed" or c.startswith("sealed-") or c == "z_snapshots" or "preds" in c:
            return True
    return False


def split_set(txt: str) -> set:
    return {t for t in (txt or "").split("|") if t}


def as_bool(v) -> bool:
    return str(v).strip() == "True"


def read_csv_skip_comments(path):
    with open(path, encoding="utf-8") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines))


def to_csv_text(fields, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    return buf.getvalue()


# ------------------------------------------------------------------ access log: the single gate for every open
NOT_OPENED = "not-opened"


class AccessLog:
    """access() is the only way adapters may touch the filesystem.  Events: open | would_open | denied | name_only.
    INPUT purposes (PREREG/S0 outputs/snapshot) are opened even in dry-run (needed to plan); DATA purposes are not."""

    def __init__(self, dry_run=False, now_fn=None, allow_run_index=False):
        self.dry_run = dry_run
        self.allow_run_index = allow_run_index
        self.now_fn = now_fn or (lambda: datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
        self.rows = []
        self._sha_cache = {}

    def _row(self, cell, event, adapter, path, sha, purpose, note=""):
        # F2 (VERIFY-V1-r1): every open event carries the sha256 of the file read; events that opened nothing say so explicitly
        if not sha and event != "open":
            sha = NOT_OPENED
        self.rows.append({"ts_utc": self.now_fn(), "cell": cell, "event": event, "adapter": adapter,
                          "path": str(path), "sha256": sha, "purpose": purpose, "note": note})

    def name_only(self, cell, adapter, source_label, note):
        self._row(cell, "name_only", adapter, source_label, "", "name/count lookup (no value file opened)", note)

    def access(self, path, purpose, cell, adapter, kind="DATA", with_sha=True) -> bool:  # with_sha=False is no longer used by any caller
        """-> True if the caller may now read `path`.  Raises DeniedPath for forbidden paths (dry-run too)."""
        if is_denied(path):
            self._row(cell, "denied", adapter, path, "", purpose, "forbidden path component (PREREG sec.4)")
            raise DeniedPath(str(path))
        if self.dry_run and not (kind == "INPUT" or (kind == "RUN_INDEX" and self.allow_run_index)):
            self._row(cell, "would_open", adapter, path, "", purpose, "dry-run: not opened")
            return False
        sha = ""
        if with_sha:
            if path not in self._sha_cache:
                self._sha_cache[path] = sha256_file(path) if os.path.isfile(path) else "MISSING"
            sha = self._sha_cache[path]
        self._row(cell, "open", adapter, path, sha, purpose)
        return True


# ------------------------------------------------------------------ S0 input loading
S0_PREFIX_DEFAULT = "S0r2"  # S0 r1 ("S0") is superseded (EXP-011 queue-regex false positive); selectable only via --s0-prefix
# cells whose S0-confirmed class legitimately differs from the PREREG 7.0.5 expectation, per S0 generation
KNOWN_REASON = {("Q-I2", "EXP-011"): "§7.0.5 예상 A-DESC ≠ S0r2 X-ABSENT: r1 큐 정규식 `_abl_`이 `pp_abl_*`에 걸린 거짓 양성, S0r2 앵커 정규식으로 데이터 0",
                ("Q-I4", "EXP-011"): "§7.0.5 예상 A-DESC ≠ S0r2 X-ABSENT: r1 큐 정규식 `_abl_`이 `pp_abl_*`에 걸린 거짓 양성, S0r2 앵커 정규식으로 데이터 0"}
KNOWN_MISMATCHES = {"S0r2": {("Q-I2", "EXP-011"): ("A-DESC", "X-ABSENT"), ("Q-I4", "EXP-011"): ("A-DESC", "X-ABSENT")}}
# class counts of the S0r2 confirmed classification (coordinator message, VERIFY-SIZ); checked unless disabled
EXPECTED_CLASS_COUNTS = {"S0r2": {"A-VERDICT": 19, "A-DESC": 21, "D-LOCKED": 5, "D-PREJUDGE": 3, "X-ABSENT": 5, "X-PROV": 2,
                                  "D-SEALED": 2, "LISTED-ONLY": 1}}


def s0_names(prefix):
    return dict(cells=f"{prefix}_cells_expected.csv", keys=f"{prefix}_metric_keys.csv", inv=f"{prefix}_inventory.csv",
                toks=f"{prefix}_tokens.csv", repro=f"{prefix}_class_repro_check.csv", snapdir=f"{prefix}_snapshot",
                snapsha=f"{prefix}_snapshot.sha256", diff=f"{prefix}_diff_vs_r1.csv", pins=f"{prefix}b_evidence_files.sha256")


def load_s0(out_dir: Path, log: AccessLog, prefix=S0_PREFIX_DEFAULT):
    nm = s0_names(prefix)
    d = {}
    for k in ("cells", "keys", "inv", "toks", "repro"):
        p = out_dir / nm[k]
        log.access(str(p), "S0 input", "-", "csv", kind="INPUT")
        d[k] = read_csv_skip_comments(p)
    cells = [(r["q"], r["s"], r["kind"], r["expected_class"]) for r in d["cells"]]
    keys = {(r["q"], r["s"]): r for r in d["keys"]}
    inv = {r["source_row"]: r for r in d["inv"]}
    toks = {}
    for r in d["toks"]:
        toks.setdefault(r["source_row"], []).append(r)
    repro = {(r["q"], r["s"]): r for r in d["repro"]}  # S0 confirmed classification
    return cells, keys, inv, toks, repro


def check_class_agreement(cls, repro, prefix, expect_class_counts, known=None):
    """S1 re-derived class must equal the S0-confirmed class for every cell, and differ from the PREREG 7.0.5 expectation only
    for the known mismatches of that S0 generation.  -> list of problems (non-empty => exit 2)."""
    problems = []
    known = KNOWN_MISMATCHES.get(prefix, {}) if known is None else known
    if set(repro) != set(cls):
        problems.append(f"{prefix}_class_repro_check.csv cell set differs from cells_expected ({len(repro)} vs {len(cls)})")
    for (q, s), (c, br, kind, exp) in cls.items():
        r = repro.get((q, s))
        if r and r["as_of_class"] != c:
            problems.append(f"({q},{s}): S1 re-derived {c} != S0 confirmed {r['as_of_class']}")
        if c != exp and known.get((q, s)) != (exp, c):
            problems.append(f"({q},{s}): class {c} != PREREG expected {exp} and not a known mismatch")
    for (q, s), (e, c) in known.items():
        if (q, s) in cls and cls[(q, s)][0] != c:
            problems.append(f"known mismatch ({q},{s}) expected to be {c} but S1 derived {cls[(q, s)][0]}")
    if expect_class_counts:
        got = {}
        for c, *_ in cls.values():
            got[c] = got.get(c, 0) + 1
        if got != expect_class_counts:
            problems.append(f"class counts {dict(sorted(got.items()))} != expected {dict(sorted(expect_class_counts.items()))}")
    return problems


def parse_h_links(prereg_text: str) -> dict:
    """PREREG sec.7.0.3 table: H row -> linked EXP rows (same parse as s0b.py)."""
    lines = prereg_text.split("\n")
    s = next(i for i, l in enumerate(lines) if l.startswith("#### 7.0.3"))
    e = next(i for i in range(s + 1, len(lines)) if lines[i].startswith("#### 7.0.4"))
    links = {}
    for l in lines[s:e]:
        if re.match(r"\| H-", l):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            links[c[0]] = [x.strip() for x in c[2].split(",")]
    return links


def check_s0_consistency(cells, keys, inv, expect=(58, 27, 40)):
    """expect=(cells, key rows, source rows) per PREREG sec.7.0.5 / 3.2 / 7.0; None disables (synthetic-fixture tests only)."""
    problems = []
    if expect:
        if len(cells) != expect[0] or len({(q, s) for q, s, _, _ in cells}) != expect[0]:
            problems.append(f"S0_cells_expected.csv: {len(cells)} rows / {len({(q, s) for q, s, _, _ in cells})} unique (expected {expect[0]})")
        if len(keys) != expect[1]:
            problems.append(f"S0_metric_keys.csv: {len(keys)} rows (expected {expect[1]})")
        if len(inv) != expect[2]:
            problems.append(f"S0_inventory.csv: {len(inv)} source rows (expected {expect[2]})")
    for q, s in keys:
        if (q, s) not in {(a, b) for a, b, _, _ in cells}:
            problems.append(f"metric_keys row {(q, s)} not in cells")
    return problems


# ------------------------------------------------------------------ facts + sec.7.1 re-derivation (snapshot-only)
def build_facts(inv: dict, toks: dict) -> dict:
    facts = {}
    for s, r in inv.items():
        t3col = next((c for c in r if c.startswith("tier3_eligible")), None)
        f = dict(kind=r["kind"], data=as_bool(r.get("data_present")), wit=as_bool(r.get("witness_present")),
                 has_result=as_bool(r.get("result_present")), tier3=as_bool(r.get(t3col, "")) if t3col else False,
                 ckpt=r.get("ckpt_dirs_matched", ""),
                 qc={k: r.get(f"queue_{k}", "") for k in ("jobs", "processing", "done", "failed")})
        rows = toks.get(s, [])
        f["tokens"] = rows
        f["claim_V"], f["V1"], f["V2"], f["N2"], f["V4"], f["N4"] = set(), set(), set(), set(), set(), set()
        f["in_index"], f["status_file"], f["lh_violation"], f["N5"] = False, False, 0, set()
        for t in rows:
            tier, lab = t["tier"], t["tier_label"]
            V, N = split_set(t["V_tok"]), split_set(t["N_tok"])
            if tier == "1":
                f["V1"], f["in_index"] = V, t["locator"] != "atom absent"
            elif tier == "2" and lab.startswith("H-NNN.status.md frontmatter status"):
                f["status_file"], f["V2"], f["N2"] = True, V, N
            elif tier == "2" and lab.startswith("LH-check"):
                m = re.search(r"L5 result names cited:\s*(\d+)", t["note"])
                f["lh_violation"] = int(m.group(1)) if m else 0
            elif tier == "3" and lab == "RESULTS frontmatter":
                f["claim_V"] = V
            elif tier == "4":
                f["V4"], f["N4"] = V, N
            elif tier == "5":
                f["N5"] = N
        facts[s] = f
    return facts


def e_absent(f) -> bool:
    return not f["data"]


def classify(q, s, kind, facts, keys, h_links):
    """sec.7.1 ten branches, first match wins.  -> (class, branch label)."""
    f = facts.get(s)
    if s in P_PROV:
        return "X-PROV", "branch1"
    if s in P_LIST:
        return "LISTED-ONLY", "branch2"
    if s in P_SEALED:
        return "D-SEALED", "branch3"
    if f is None:
        return "UNCOVERED", "branch10 (source row absent from S0_inventory)"
    if kind == "E" and e_absent(f):
        return "X-ABSENT", "branch4 (data absent)"
    if kind == "H":
        links = h_links.get(s, [])
        if links and all(x in facts and e_absent(facts[x]) for x in links):
            return "X-ABSENT", "branch4 (all linked EXP rows absent)"
    if kind == "E" and f["wit"] and not f["has_result"]:
        return "D-PREJUDGE", "branch5"
    if keys.get((q, s), {}).get("key_status") == "NAMED-OTHER":
        return "X-OUT", "branch6"
    if (kind == "E" and s in L5) or (kind == "H" and s in LH):
        return "D-LOCKED", "branch7"
    if kind == "A":
        if f["in_index"] and f["V1"]:
            return "A-VERDICT", "branch8 atom tier1 verdict in V_tok"
    elif kind == "H":
        if s not in LH and f["lh_violation"] > 0:
            return "UNCOVERED", "branch10 (LH rule: tier-2 file cites an L5 result name)"
        V = f["V2"] if f["status_file"] else f["V4"]
        N = f["N2"] if f["status_file"] else f["N4"]
        tier = "2" if f["status_file"] else "4"
        if V:
            return "A-VERDICT", f"branch8 H token tier{tier}"
        if N:
            return "A-DESC", f"branch9 H N_tok tier{tier}"
        return "UNCOVERED", "branch10 (no V/N word)"
    else:
        if f["tier3"] and f["claim_V"]:
            return "A-VERDICT", "branch8 E tier3 frontmatter claim verdict"
        if f["data"]:
            return "A-DESC", "branch9 E data present, branch8 false"
    return "UNCOVERED", "branch10"


def q_tokens(cell_classes: dict) -> dict:
    """sec.7.2.  cell_classes: q -> list of classes."""
    X = {"X-PROV", "X-ABSENT", "X-OUT"}
    D = {"D-SEALED", "D-PREJUDGE", "D-LOCKED"}
    out = {}
    for q, cl in cell_classes.items():
        if all(c in X for c in cl):
            out[q] = "Q-NONE"
        elif all(c in X or c == "LISTED-ONLY" for c in cl) and "LISTED-ONLY" in cl:
            out[q] = "Q-LISTED"
        elif "A-VERDICT" in cl:
            out[q] = "Q-INHERIT"
        elif any(c in D or c == "A-DESC" for c in cl):
            out[q] = "Q-DESC-ONLY"
        else:
            out[q] = "Q-UNCOVERED"
    return out


# ------------------------------------------------------------------ key resolution (S0_metric_keys.csv drives adapters)
def parse_n_named(txt: str) -> list:
    return re.findall(r"[A-Za-z_]+", txt or "") if txt and "∅" not in txt and "—" not in txt else []


def resolve_keys(q, keyrow):
    """-> (keys_to_read [short], skipped [(short, reason)], metric_key_field)."""
    if keyrow is None:
        return [], [], "UNKNOWN"
    if keyrow["key_status"] != "NAMED-ADMITTED":
        return [], [], "UNKNOWN"
    N = parse_n_named(keyrow["N_named"])
    K = K_SHORT.get(q)
    use = [n for n in N if (K is None or n in K)]
    read, skipped = [], []
    for n in use:
        if n in WIRED:
            read.append(n)
        else:
            skipped.append((n, "no adapter wired for this key (analysis CSV source)"))
    return read, skipped, "|".join(KEYMAP[n][0] for n in read)  # only keys actually read (CR-9)


# ------------------------------------------------------------------ adapters
# (1) atom INDEX.jsonl (snapshot copy)
def atom_adapter(index_path, atom_id, log: AccessLog, cell):
    if not log.access(index_path, "atom verdict/scope verbatim", cell, "atom-index", kind="INPUT"):
        return None
    with open(index_path, encoding="utf-8") as fh:
        for l in fh:
            d = json.loads(l)
            if d.get("atom_id") == atom_id:
                return {"verdict": str(d.get("verdict", "")), "cohort": str(d.get("cohort", "")),
                        "verdict_scope": str(d.get("verdict_scope", "")),
                        "prohibited_upgrade": str(d.get("prohibited_upgrade", ""))}
    return None


# (2) EXP RESULTS markdown table.  Generic parser, deliberately NOT wired to any cell: PREREG names no table/row/column
# selector for any cell (reported ambiguity).  Tested on synthetic markdown only.
def markdown_table_adapter(path, header_regex, log: AccessLog, cell):
    if not log.access(path, f"markdown table /{header_regex}/", cell, "results-table"):
        return None
    lines = open(path, encoding="utf-8").read().split("\n")
    for i, l in enumerate(lines):
        if l.startswith("|") and re.search(header_regex, l) and i + 1 < len(lines) and re.match(r"\|[\s:|-]+\|?\s*$", lines[i + 1]):
            hdr = [c.strip() for c in l.strip().strip("|").split("|")]
            rows = []
            for r in lines[i + 2:]:
                if not r.startswith("|"):
                    break
                cs = [c.strip() for c in r.strip().strip("|").split("|")]
                rows.append(dict(zip(hdr, cs)))
            return {"header": hdr, "rows": rows, "line": i + 1}
    return None


# (3) checkpoint metadata: ONLY checkpoints/<project>/<run>/all_config.yaml (PREREG sec.4; nothing else under a run dir)
CONFIG_MISSING = "ALL_CONFIG_MISSING"  # sentinel: file absent on disk (distinct from None = dry-run, not opened)


def config_adapter(root: Path, project, run_name, log: AccessLog, cell):
    p = str(root / "checkpoints" / project / run_name / "all_config.yaml")
    if not log.access(p, "run config (arch/HL/seed/data_paths)", cell, "all_config"):
        return None
    if not os.path.isfile(p):
        return CONFIG_MISSING
    import yaml
    return yaml.safe_load(open(p, encoding="utf-8")) or {}


def eval_set_size(root: Path, data_path, log: AccessLog, cell):
    """<data_path>/test/dataset.json total_puzzles (sealed split never touched)."""
    p = str(root / data_path / "test" / "dataset.json")
    if not log.access(p, "eval set size", cell, "dataset.json"):
        return "UNKNOWN"
    try:
        return str(json.load(open(p, encoding="utf-8"))["total_puzzles"])
    except Exception:
        return "UNKNOWN"


# (4) wandb local datastore.  scan_history vendored from analysis/pr_extract.py (analysis/ is gitignored).
def _json_num(v):
    try:
        return json.loads(v)
    except Exception:
        return v


def scan_wandb_history(wandb_file, keys):
    """Return [{'_step': int, key: value}] merged by step, only steps carrying >=1 requested key, sorted by step."""
    from wandb.proto import wandb_internal_pb2 as pb  # lazy: tests stub the scanner
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    by_step, kset = {}, set(keys)
    while True:
        try:
            rec = ds.scan_data()
        except Exception:
            break
        if rec is None:
            break
        r = pb.Record()
        r.ParseFromString(rec)
        if r.WhichOneof("record_type") != "history":
            continue
        d = {}
        for item in r.history.item:
            k = item.key if item.key else "/".join(item.nested_key)
            if k == "_step" or k in kset:
                d[k] = item.value_json
        if "_step" not in d:
            continue
        st = _json_num(d["_step"])
        slot = by_step.setdefault(st, {"_step": st})
        for k, v in d.items():
            if k in kset:
                slot[k] = _json_num(v)
    rows = [s for s in by_step.values() if any(k in s for k in kset)]
    rows.sort(key=lambda s: s["_step"] if isinstance(s["_step"], (int, float)) else 0)
    return rows


def r_final(history_rows, key):
    """R-FINAL: value at the run's last observed step carrying `key`.  -> (value, step, n_points) or None."""
    pts = [(r["_step"], r[key]) for r in history_rows if key in r and r[key] is not None]
    if not pts:
        return None
    step, val = max(pts, key=lambda p: p[0])
    return val, step, len(pts)


def _name_project_from_config_yaml(d, log, cell):
    """Fallback (analysis/pr_extract.py precedent): files/config.yaml top-level run_name/project_name (.value) or the
    +run_name=/+project_name= override strings anywhere in it.  -> (name|None, project|None)."""
    cfg_path = os.path.join(d, "files", "config.yaml")
    if not os.path.isfile(cfg_path) or not log.access(cfg_path, "run name/project lookup (fallback)", cell, "wandb-config",
                                                      kind="RUN_INDEX"):
        return None, None
    import yaml
    try:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    except Exception:
        return None, None
    found = {}
    if isinstance(cfg, dict):
        for key, flag in (("run_name", "+run_name"), ("project_name", "+project_name")):
            v = cfg.get(key)
            v = v.get("value") if isinstance(v, dict) else v
            if v:
                found[key] = str(v)
                continue
            m = re.search(re.escape(flag) + r"=([A-Za-z0-9_.\-]+)", json.dumps(cfg))
            if m:
                found[key] = m.group(1)
    return found.get("run_name"), found.get("project_name")


def build_run_index(root: Path, wandb_root: str, log: AccessLog, cell="run-index"):
    """name/project of every wandb run dir: files/wandb-metadata.json args first, files/config.yaml as fallback.  No values."""
    out = []
    for d in sorted(glob.glob(str(root / wandb_root / "*run-*"))):
        if not os.path.isdir(d):
            continue
        name = proj = started = None
        meta = os.path.join(d, "files", "wandb-metadata.json")
        if os.path.isfile(meta) and log.access(meta, "run name/project lookup", cell, "wandb-metadata", kind="RUN_INDEX"):
            try:
                md = json.load(open(meta, encoding="utf-8"))
                started = md.get("startedAt") or md.get("started_at")  # both spellings (INTERP_S1 (6)); metadata, not a value
                args = md.get("args", [])
                name = next((a.split("=", 1)[1] for a in args if a.startswith("+run_name=")), None)
                proj = next((a.split("=", 1)[1] for a in args if a.startswith("+project_name=")), None)
            except Exception:
                pass
        if not name or not proj:
            n2, p2 = _name_project_from_config_yaml(d, log, cell)
            name, proj = name or n2, proj or p2
        if not name:
            continue
        hist = sorted(glob.glob(os.path.join(d, "run-*.wandb")))
        out.append({"run_name": name, "project": proj or "UNKNOWN", "wandb_dir": d,
                    "wandb_run_id": os.path.basename(d).split("-")[-1], "history_path": hist[0] if hist else "",
                    "started_at": started})
    return out


# ------------------------------------------------------------------ run selection
def matches_prejudge(run_name: str) -> str:
    for src, pats in PREJUDGE_RUN_PATTERNS.items():
        if any(re.search(p, run_name) for p in pats):
            return src
    return ""


def select_runs(s, q, run_index):
    """-> (selected runs, excluded [(run_name, reason)]).  Selection is provisional (see RUN_SELECT_BASIS)."""
    spec = RUN_SELECT_BY_Q.get((s, q)) or RUN_SELECT.get(s)
    if spec is None:
        return [], [("-", f"no run selector defined for {s} (no own runs / undefined)")]
    proj, rx = spec
    sel, exc = [], []
    for r in run_index:
        if r["project"] != proj or not re.search(rx, r["run_name"]):
            continue
        owner = matches_prejudge(r["run_name"])
        if owner and owner != s:
            exc.append((r["run_name"], f"belongs to D-PREJUDGE source {owner} (O-2 guard)"))
            continue
        sel.append(r)
    return sel, exc


def arch_label(cfg):
    name = str((cfg.get("arch") or {}).get("name", ""))
    m = re.match(r"recursive_reasoning\.([A-Za-z0-9_]+)@", name)
    return m.group(1) if m and m.group(1) in ("trm", "trm_singlez", "transformers_baseline", "hrm") else "UNKNOWN"


def parse_started(v):
    """wandb startedAt / started_at -> aware UTC datetime or None (naive stamps are read as UTC: wandb writes UTC; unverified)."""
    if not v:
        return None
    try:
        d = datetime.datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def cohort_for(project, data_path, started):
    """INTERP_S1.md (6) / PREREG 0, 3.2: label only when project, data path AND start time all agree; else UNKNOWN.
    legacy: Sigma_k_fig12 and t_start < 13:18 KST (the earlier of the two readings of the time-zone-less t_build).
    clean-*: right project, right path prefix, and t_start after the later reading + 11 min margin (k=11: after 07-22)."""
    t = parse_started(started)
    if t is None or project not in COHORT_BY_PROJECT:
        return "UNKNOWN"
    label, req = COHORT_BY_PROJECT[project]
    if label == "legacy":
        return "legacy" if t < T_BUILD_KST else "UNKNOWN"
    if not data_path.startswith(req) or t <= T_CLEAN_START:
        return "UNKNOWN"
    if re.search(r"/11$", data_path) and t <= T_K11_START:
        return "UNKNOWN"
    return label


def run_provenance(cfg, idx_project, root, log, cell, started_at=None):
    """sec.3.2 labels that come from the run config; anything not on disk stays UNKNOWN (no inference)."""
    a = (cfg or {}).get("arch") or {}
    H, L = a.get("H_cycles"), a.get("L_cycles")
    dps = (cfg or {}).get("data_paths") or []
    dp = dps[0].rstrip("/") if dps else ""
    project = (cfg or {}).get("project_name") or idx_project
    k = (cfg or {}).get("k")
    if k is None and dp and re.search(r"/(\d+)$", dp):
        k = int(re.search(r"/(\d+)$", dp).group(1))
    block = {True: "mlp", False: "tf"}.get(a.get("mlp_t"), "UNKNOWN")
    return {
        "arch": arch_label(cfg or {}) if cfg else "UNKNOWN",
        "HL_train": f"({H},{L})" if H is not None and L is not None else "UNKNOWN",
        "block": block, "seed": (cfg or {}).get("seed", "UNKNOWN"), "k": k if k is not None else "UNKNOWN",
        "cohort": cohort_for(project, dp, started_at), "data_path": dp, "project": project,
        # total_steps is not a key in all_config.yaml (epochs/eval_interval only): not derived (reported ambiguity)
        "budget": (cfg or {}).get("total_steps", "UNKNOWN"),
        "z_probe_size": (cfg or {}).get("z_probe_size", "UNKNOWN"),
        "n10": bool(dp) and dp.startswith(N10_DATA_PREFIXES),
    }


def weights_for(protocol):
    return {"P-ALL": "online-raw", "P-RECOMP": "EMA-disk"}.get(protocol, "UNKNOWN")  # sec.0 definitions; P-PROBE = U-2 open


def probe_size_for(protocol, prov):
    if protocol == "P-ALL":
        return "n/a"
    if protocol == "P-RECOMP":
        return "256"
    if protocol == "P-PROBE":
        # INTERP_S1 (7): config z_probe_size only if it is in the sec.3.2 vocabulary {256, 512}; else UNKNOWN
        return str(prov["z_probe_size"]) if str(prov["z_probe_size"]) in ("256", "512") else "UNKNOWN"
    return "UNKNOWN"


def make_owner_resolver(root: Path, snap: Path, log: AccessLog):
    """PREREG 3.2 exp_owner: EXPERIMENTS.md > ID_ALIASES.yaml > run-family map.  Only the third is machine-readable; the
    map's evidence items quote EXPERIMENTS.md lines (tier registry, role owner*), so those count as priority 1.
    -> resolve(project, run_name) = (owner|'UNKNOWN', basis).  Exactly one canonical owner or UNKNOWN; never guessed.
    The map is DRAFT, dated 2026-10-04, and not in S0_snapshot (live INPUT read, sha256 in the access log)."""
    import yaml
    mp = str(root / RUN_FAMILY_MAP_REL)
    rows = []
    if os.path.isfile(mp) and log.access(mp, "exp_owner chain level 3 (run-family map)", "exp_owner", "run-family-map", kind="INPUT"):
        rows = yaml.safe_load(open(mp, encoding="utf-8")) or []
    canon = {}   # alias string / canonical id -> [(canonical, scope_text)]  (an alias may be listed under several canonicals)
    ap = snap / "ID_ALIASES.yaml"
    if ap.is_file() and log.access(str(ap), "exp_owner alias normalisation", "exp_owner", "id-aliases", kind="INPUT"):
        for e in (yaml.safe_load(open(ap, encoding="utf-8")) or {}).get("entries", []):
            canon.setdefault(e["canonical"], []).append((e["canonical"], None))
            for a in e.get("aliases") or []:
                canon.setdefault(a["string"], []).append((e["canonical"], f"{e.get('source') or ''} {a.get('scope') or ''}"))

    def normalise(applies_to, ev_source):
        """F1 (VERIFY-V1-r1): scope-aware alias lookup.  A canonical id maps to itself; an alias with one canonical maps to it; an alias
        listed under several canonicals (EXP-015 -> EXP-025 | EXP-100) maps only to the candidate whose entry source/scope text names the
        FILE of the quoted evidence (path before ':line'); not exactly one such candidate -> None (owner UNKNOWN).  No prefix matching."""
        cands = canon.get(applies_to)
        if not cands:
            return applies_to
        if any(sc is None for _, sc in cands):  # applies_to is itself a canonical id
            return applies_to
        names = {c for c, _ in cands}
        if len(names) == 1:
            return names.pop()
        path = str(ev_source or "").rsplit(":", 1)[0]
        hit = {c for c, sc in cands if path and path in sc}
        return hit.pop() if len(hit) == 1 else None

    def resolve(project, run_name):
        if not rows:
            return "UNKNOWN", "run-family map unavailable"
        fam = [r for r in rows if r.get("project") == project and re.match(r.get("name_regex") or "$^", run_name)]
        if len(fam) != 1:
            return "UNKNOWN", f"run-family map: {len(fam)} family rows match"
        owners, basis = set(), set()
        for ev in fam[0].get("evidence") or []:
            if ev.get("tier") == "registry" and str(ev.get("role") or "").startswith("owner") and ev.get("applies_to"):
                owners.add(normalise(ev["applies_to"], ev.get("source")) or f"AMBIGUOUS-ALIAS:{ev['applies_to']}")
                basis.add("EXPERIMENTS.md(quoted in map)" if "EXPERIMENTS.md" in str(ev.get("source")) else "map")
        if len(owners) == 1 and not next(iter(owners)).startswith("AMBIGUOUS-ALIAS:"):
            return owners.pop(), "|".join(sorted(basis))
        return "UNKNOWN", f"run-family map: {len(owners)} owner candidates ({fam[0].get('family')})"
    return resolve


def e_restatement(root: Path, s0_dir: Path, nm, f, log: AccessLog, cell):
    """INTERP_S1 (11)/CR-5: restate the status line of an E-row D-LOCKED cell's own RESULTS frontmatter, verbatim, ONLY if the
    file's live sha256 equals the pin in <prefix>b_evidence_files.sha256 (= the as_of content).  One line, never the body.
    -> (restatement, note, discrepancy, file, locator)."""
    row3 = next((t for t in f.get("tokens", []) if t["tier"] == "3" and t["tier_label"] == "RESULTS frontmatter"), None)
    if row3 is None:
        return "", "DISCREPANCY: no tier-3 RESULTS row in S0 tokens — 재진술 미충족", "no tier-3 row", "", ""
    path, loc = row3["file"], row3["locator"]
    m = re.search(r"status: line (\d+)", loc)
    pins = {}
    pf = s0_dir / nm["pins"]
    if log.access(str(pf), "sha pins for RESULTS restatement", cell, "evidence-pins", kind="INPUT") and pf.is_file():
        for l in open(pf, encoding="utf-8"):
            parts = l.split(None, 1)
            if len(parts) == 2:
                pins[parts[1].strip()] = parts[0]
    if not m or path not in pins:
        return "", "DISCREPANCY: no sha pin or no status line locator — 재진술 미충족", "no pin/locator", path, loc
    full = root / path
    if not log.access(str(full), "RESULTS frontmatter status line (sha-gated restatement)", cell, "results-status-line"):
        return "", "dry-run: not read", "", path, loc  # dry-run: would_open logged only
    live = log._sha_cache.get(str(full), "MISSING")
    if live != pins[path]:
        return ("", f"DISCREPANCY: as_of 뒤 변경 (pin {pins[path][:12]}…, live {live[:12]}…) — 재진술 미충족",
                "changed after as_of", path, loc)
    lines = open(full, encoding="utf-8").read().split("\n")
    n = int(m.group(1))
    return (lines[n - 1] if 0 < n <= len(lines) else ""), "", "", path, loc


def plan_cell(q, s, kind, c, keyrow):
    """Decide BEFORE touching anything whether a cell may reach an adapter.  -> dict(read, skipped, mk, spec, policy, extract)."""
    if not (kind == "E" and c in VALUE_CLASSES):
        return dict(read=[], skipped=[], mk="", spec=None, policy="", extract=False)
    read, skipped, mk = resolve_keys(q, keyrow)
    kst = keyrow["key_status"] if keyrow else ""
    spec = RUN_SELECT_BY_Q.get((s, q)) or RUN_SELECT.get(s)
    if kst != "NAMED-ADMITTED":
        pol = f"none: key_status={kst or 'ABSENT'} ({UNKNOWN_KEY_NOTE})"
    elif not read and skipped:
        pol = "none: " + "; ".join(f"{n}: {why}" for n, why in skipped)
    elif not read:
        pol = "none: N ∩ K(q) = ∅"
    elif spec is None:
        pol = "none: 자체 run 없음 (run selector 없음 — INTERP_S1 (3))"
    else:
        pol = "extract R-FINAL per run, per key, seed-wise (no aggregation)"
    return dict(read=read, skipped=skipped, mk=mk, spec=spec, policy=pol, extract=pol.startswith("extract"))


# ------------------------------------------------------------------ main pipeline
def run_s1(root: Path, out_dir: Path, *, dry_run=False, wandb_root="wandb", prereg_rel=PREREG_REL, prereg_sha=PREREG_SHA,
           scanner=None, run_index=None, owner_resolver=None, now_fn=None, access_log_name="S1_access_log.csv", resolve_runs=False,
           script_path=None, stdout=None, expect_counts=(58, 27, 40), s0_prefix=S0_PREFIX_DEFAULT, expect_class_counts="default", known_mismatches=None,
           s0_dir=None):
    """Returns exit code.  `scanner(path, keys) -> history rows` and `run_index` (list of dicts) are injectable for tests."""
    stdout = stdout or sys.stdout
    scanner = scanner or scan_wandb_history
    log = AccessLog(dry_run=dry_run, now_fn=now_fn, allow_run_index=resolve_runs)

    # ---- gates before anything else
    prereg = root / prereg_rel
    if sha256_file(prereg) != prereg_sha:
        print("PREREG sha256 mismatch -> abort", file=stdout)
        return 2
    nm = s0_names(s0_prefix)
    s0_dir = Path(s0_dir) if s0_dir else out_dir  # S0 inputs location (default: same dir as the outputs); --out may point to a fresh dir
    snap = s0_dir / nm["snapdir"]
    if expect_class_counts == "default":
        expect_class_counts = EXPECTED_CLASS_COUNTS.get(s0_prefix)
    for need in ("as_of.txt", "INDEX.jsonl"):
        if not (snap / need).is_file():
            print(f"missing S0 snapshot file {need} -> abort", file=stdout)
            return 2
    as_of = next(l.split(": ", 1)[1].strip() for l in open(snap / "as_of.txt", encoding="utf-8") if l.startswith("as_of:"))
    own = set(S1_OUTPUT_NAMES) | {access_log_name}  # INTERP_S1 CD-1: only S1's own outputs block a run (not any S1_* name)
    existing = [n for n in os.listdir(out_dir) if n in own] if out_dir.is_dir() else []
    if existing and not dry_run:
        print(f"refuse to overwrite {existing}", file=stdout)
        return 3
    h_links = parse_h_links(prereg.read_text(encoding="utf-8"))
    cells, keys, inv, toks, repro = load_s0(s0_dir, log, s0_prefix)
    problems = check_s0_consistency(cells, keys, inv, expect_counts)
    if problems:
        for p in problems:
            print("GATE:", p, file=stdout)
        return 2
    facts = build_facts(inv, toks)

    # ---- classification
    cls = {}
    for q, s, kind, exp in cells:
        c, br = classify(q, s, kind, facts, keys, h_links)
        cls[(q, s)] = (c, br, kind, exp)
    qtok = q_tokens({q: [cls[(qq, s)][0] for qq, s in cls if qq == q] for q in {q for q, _ in cls}})

    cls_problems = check_class_agreement(cls, repro, s0_prefix, expect_class_counts, known_mismatches)
    if cls_problems:
        for pr in cls_problems:
            print("GATE:", pr, file=stdout)
        return 2

    # ---- manifest part 1 (script + inputs), BEFORE any value is read (PREREG sec.9 S1 gate)
    script_path = script_path or os.path.abspath(__file__)
    manifest = [f"{sha256_file(script_path)}  tools/state_iter_z_synthesis/s1.py",
                f"{prereg_sha}  {prereg_rel}",
                f"{as_of}  as_of({nm['snapdir']}/as_of.txt)", f"{s0_prefix}  s0_prefix"]
    for n in (nm["cells"], nm["repro"], nm["keys"], nm["inv"], nm["toks"], nm["snapsha"], nm["diff"], nm["pins"]):
        pth = s0_dir / n
        manifest.append(f"{sha256_file(pth) if pth.is_file() else 'MISSING'}  {_rel(root, s0_dir)}/{n}")
    # CR-7: S0 generation scripts, INTERP_S1.md, and the run-family map (live exp_owner input) are pinned before any value is read
    sdir = root / "tools" / "state_iter_z_synthesis"
    for sc in (f"{s0_prefix.lower()}a.py", f"{s0_prefix.lower()}b.py", f"{s0_prefix.lower()}c.py"):
        pth = sdir / sc
        manifest.append(f"{sha256_file(pth) if pth.is_file() else 'MISSING'}  tools/state_iter_z_synthesis/{sc}")
    for rel, pth in (("INTERP_S1.md", s0_dir / "INTERP_S1.md"), (RUN_FAMILY_MAP_REL, root / RUN_FAMILY_MAP_REL)):
        manifest.append(f"{sha256_file(pth) if pth.is_file() else 'MISSING'}  {rel if rel != 'INTERP_S1.md' else _rel(root, s0_dir) + '/INTERP_S1.md'}")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "S1_manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")


    # ---- per-cell plan (decided before any adapter is touched), then the run index only if some cell will extract
    plans = {(q, s): plan_cell(q, s, kind, c, keys.get((q, s))) for (q, s), (c, br, kind, exp) in cls.items()}
    needs_values = any(pl["extract"] for pl in plans.values())
    if run_index is None:
        run_index = build_run_index(root, wandb_root, log) if (needs_values and (not dry_run or resolve_runs)) else []
    resolve_owner = owner_resolver or (make_owner_resolver(root, snap, log) if needs_values else (lambda p_, r_: ("UNKNOWN", "not resolved")))
    dup = {}
    for r in run_index:
        dup[(r["project"], r["run_name"])] = dup.get((r["project"], r["run_name"]), 0) + 1

    table, cell_rows = [], []
    hist_cache = {}
    index_path = str(snap / "INDEX.jsonl")
    for (q, s), (c, br, kind, exp) in cls.items():
        cell = f"({q},{s})"
        keyrow = keys.get((q, s))
        kst = keyrow["key_status"] if keyrow else ""
        pl = plans[(q, s)]
        row = dict(q=q, s=s, kind=kind, expected_class=exp, s1_class=c, branch=br, class_match=(c == exp), q_token=qtok[q],
                   key_status=kst, metric_keys_read="", values_policy="", n_table_rows=0, counts_by_cohort="",
                   cohort_note="", notes="")
        row["s0_confirmed_class"] = repro[(q, s)]["as_of_class"]
        if c == exp:
            row["reason"] = ""
        elif s0_prefix == "S0r2" and (q, s) in KNOWN_REASON:
            row["reason"] = KNOWN_REASON[(q, s)]
        else:
            row["reason"] = f"{s0_prefix} confirmed class {c} ({br}) != PREREG 7.0.5 expected {exp} (known mismatch; as_of snapshot condition differs; not overridden)"
        f = facts.get(s, {})

        # ---- token columns.  token_* hold ONLY a V_tok/N_tok word set from the PREREG 5.1 tier that may be a cell token:
        # A -> tier 1; H -> tier 2 else 4; E -> tier 3.  Tier 5/6 (registry Status, EXP prose) go to registry_status_N only
        # (INTERP_S1 CD-10).  Placeholders never go into token_V (CD-11).
        if c == "X-ABSENT":  # INTERP_S1 UD-7 default: fixed PREREG phrase + scope of the claim
            row["values_policy"] = "none: class X-ABSENT — 데이터 없음(미실행); 이 호스트 as_of 로컬 데이터 기준; 레지스트리 상태는 Q-S1 표"
        elif c in ("X-PROV", "X-OUT", "UNCOVERED"):
            row["values_policy"] = f"none: class {c}"
        elif c == "D-SEALED":
            row["values_policy"] = f"none: D-SEALED ({SEALED_NOTE})"
        elif c == "D-PREJUDGE":
            row["values_policy"] = f"none: D-PREJUDGE (O-2: status/run count only; {PREJUDGE_NOTE})"
            qc = f.get("qc", {})
            log.name_only(cell, "inventory", nm["inv"],
                          "queue name counts at as_of: " + ",".join(f"{k}={v}" for k, v in qc.items()))
        else:
            allowed = {"A": ("1",), "H": ("2", "4"), "E": ("3",)}[kind]
            trs = sorted([t for t in f.get("tokens", []) if t["tier"] in allowed], key=lambda t: int(t["tier"]))
            use = [t for t in trs if split_set(t["V_tok"]) or split_set(t["N_tok"])]
            if use:
                top = use[0]
                row.update(token_tier=top["tier"], token_V=top["V_tok"], token_N=top["N_tok"], token_file=top["file"],
                           token_locator=top["locator"])
                disc = [f"tier{t['tier']}:{t['V_tok']}" for t in use[1:] if split_set(t["V_tok"]) and t["V_tok"] != top["V_tok"]]
                row["token_discrepancy"] = ";".join(disc)
                if len(split_set(top["V_tok"])) > 1:
                    row["notes"] = "MULTIPLE-V-WORDS-UNRESOLVED (which word applies to the cell's component is not defined by PREREG)"
                if kind == "H" and (q, s) in COMPONENT_CELLS:  # PREREG 7.1 branch 8: no component token -> card token + card-level
                    row["token_scope"] = f"card-level (성분 귀속 미확정: {COMPONENT_CELLS[(q, s)]})"
            if kind == "E":
                row["registry_status_N"] = "|".join(sorted(f.get("N5", set())))  # tier 5: Q-S1 / DISCREPANCY only, not a token
            if kind == "H" and not row["values_policy"]:
                row["values_policy"] = "none: H row (status word only)"
            if c == "D-LOCKED":
                row["marker"] = D_LOCKED_MARKER
                if kind == "E":
                    rs, note, disc, tfile, tloc = e_restatement(root, s0_dir, nm, f, log, cell)
                    row.update(token_restatement=rs, token_note=note, token_file=tfile, token_locator=tloc)
                    ds = [x for x in (disc,) if x]
                    if not f.get("tier3"):  # U-19: frontmatter status carries no verifier PASS; return-file verdict is cited only
                        ds.append("U-19: frontmatter status carries no verifier PASS (tier3_eligible=False); return-file verdict not inherited")
                    row["token_discrepancy"] = ";".join([x for x in (row.get("token_discrepancy"), *ds) if x])
                    if note.startswith("DISCREPANCY"):
                        print("DISCREPANCY:", q, s, note, file=stdout)
                if kind == "H" and row.get("token_tier") == "2":
                    m = re.search(r"line (\d+)", row.get("token_locator", ""))
                    tf = s0_dir / row["token_file"] if row.get("token_file") else None
                    if m and tf and tf.is_file() and log.access(str(tf), "D-LOCKED H restatement (snapshot status line, verbatim)", cell, "snapshot-status", kind="INPUT"):
                        ls = open(tf, encoding="utf-8").read().split("\n")
                        n = int(m.group(1))
                        row["token_restatement"] = ls[n - 1] if 0 < n <= len(ls) else ""
        if kind == "A" and c in ("A-VERDICT", "LISTED-ONLY"):
            a = atom_adapter(index_path, s, log, cell)
            if a:
                # LISTED-ONLY (CL-0006): verdict word + verdict_scope verbatim only (PREREG sec.7.1 branch 2 / sec.7.4)
                row.update(atom_cohort=a["cohort"], atom_verdict_scope=a["verdict_scope"],
                           atom_prohibited_upgrade="" if c == "LISTED-ONLY" else a["prohibited_upgrade"])
                row["token_V"] = a["verdict"] if c == "LISTED-ONLY" else row.get("token_V", "")
            row["values_policy"] = ("none: LISTED-ONLY (Q-S1 table only: verdict word + verdict_scope verbatim)"
                                    if c == "LISTED-ONLY" else "none: atom verdict token only (no metric value)")
        if s in LEGACY_SOURCES:
            row["cohort_note"] = LEGACY_NOTE

        # ---- values (only when plan_cell said extract; otherwise NO file of any run is opened for this cell)
        if kind == "E" and c in VALUE_CLASSES:
            read, spec = pl["read"], pl["spec"]
            row["metric_keys_read"] = pl["mk"] if (pl["extract"] or kst != "NAMED-ADMITTED") else ""  # CR-9: nothing is shown as read unless read
            row["values_policy"] = pl["policy"]
            if pl["skipped"]:
                row["notes"] = "; ".join(f"{n}: {why}" for n, why in pl["skipped"])
            if pl["extract"]:
                row["notes"] += ("; " if row["notes"] else "") + f"selector={spec[0]}:{spec[1]}"
                if dry_run and not run_index:
                    log.access(f"wandb/*run-*/run-*.wandb[{spec[0]}:{spec[1]}]",
                               f"history keys {[KEYMAP[kk][0] for kk in read]} (run list not resolved in dry-run)", cell, "wandb")
                sel, exc = select_runs(s, q, run_index)
                for rn, why in exc:
                    log.name_only(cell, "run-select", f"{s}:{rn}", why)
                row["notes"] += f"; s0_ckpt_dirs_matched={f.get('ckpt', '')}; s1_candidate_runs={len({(r['project'], r['run_name']) for r in sel})}"
                if not sel and not exc and not dry_run:
                    row["notes"] += "; no runs matched selector"
                n_excl_cfg = n_excl_n10 = 0
                for r in sel:
                    cfg = config_adapter(root, r["project"], r["run_name"], log, cell)
                    if cfg is None and dry_run:
                        for kk in read:
                            log.access(r["history_path"] or f"{r['wandb_dir']}/run-*.wandb", f"history {KEYMAP[kk][0]}", cell, "wandb")
                        continue
                    if cfg == CONFIG_MISSING:
                        n_excl_cfg += 1
                        log.name_only(cell, "run-select", f"{s}:{r['run_name']}",
                                      "excluded: all_config.yaml MISSING under checkpoints/<project>/<run>; provenance (data_paths/arch/HL/seed) unavailable, nothing inferred")
                        continue
                    prov = run_provenance(cfg, r["project"], root, log, cell, r.get("started_at"))
                    if not prov["n10"]:
                        n_excl_n10 += 1
                        log.name_only(cell, "run-select", f"{s}:{r['run_name']}", f"excluded: data_paths '{prov['data_path']}' not n=10 prefix")
                        continue
                    if not r["history_path"]:
                        row["notes"] += f"; no .wandb for {r['run_name']}"
                        continue
                    # INTERP_S1 CD-12: the current dataset.json is the run's eval set only for clean cohorts
                    es = (eval_set_size(root, prov["data_path"], log, cell)
                          if prov["data_path"] and prov["cohort"] in ("clean-D0", "clean-D1") else "UNKNOWN")
                    owner, obasis = resolve_owner(r["project"], r["run_name"])
                    use_ok = prov["cohort"] != "UNKNOWN" and owner != "UNKNOWN"
                    why_not = ";".join(w for w, bad in (("cohort UNKNOWN", prov["cohort"] == "UNKNOWN"), ("exp_owner UNKNOWN", owner == "UNKNOWN")) if bad)
                    ckey = r["history_path"]
                    need = [KEYMAP[kk][0] for kk in read]
                    if ckey not in hist_cache or not set(need) <= hist_cache[ckey][0]:
                        want = set(need) | (hist_cache[ckey][0] if ckey in hist_cache else set())
                        log.access(ckey, f"history keys {sorted(want)}", cell, "wandb")
                        hist_cache[ckey] = (want, scanner(ckey, sorted(want)))
                    else:
                        log.access(ckey, f"history keys {sorted(need)} (cached scan)", cell, "wandb")
                    for kk in read:
                        keystr, protocol = KEYMAP[kk]
                        fin = r_final(hist_cache[ckey][1], keystr)
                        table.append(dict(
                            q=q, s=s, cell_class=c, metric_key=keystr, protocol=protocol or "UNKNOWN",
                            protocol_note="" if protocol else "PR key — sec.3.2 protocol vocabulary covers test-exact protocols only (INTERP_S1 (7))",
                            value="" if fin is None else fin[0], step="" if fin is None else fin[1], reduction="R-FINAL", frame="full",
                            n_logged_points=0 if fin is None else fin[2], label_GC="NONE", run_name=r["run_name"],
                            wandb_run_id=r["wandb_run_id"], wandb_project=r["project"], provenance_file=ckey,
                            provenance_sha256=log._sha_cache.get(ckey, ""), weights=weights_for(protocol),
                            eval_set_size=es, probe_size=probe_size_for(protocol, prov), budget=prov["budget"],
                            cohort=prov["cohort"], cohort_note=LEGACY_NOTE if prov["cohort"] == "legacy" else "",
                            started_at=r.get("started_at") or "UNKNOWN",
                            arch=prov["arch"], HL_train=prov["HL_train"], HL_eval="UNKNOWN", block=prov["block"],
                            seed=prov["seed"], k=prov["k"], Tk_class="UNKNOWN", exp_owner=owner, exp_owner_basis=obasis,
                            evidence_use="yes" if use_ok else f"no: {why_not}",
                            run_select_basis=RUN_SELECT_BASIS, dup_wandb_dirs=dup.get((r["project"], r["run_name"]), 1),
                            note="; ".join(x for x in (LABEL_GC_NOTE, KEY_NOTES.get(kk, ""),
                                                       "" if fin is not None else "key never logged in this run") if x)))
                if n_excl_cfg or n_excl_n10:
                    row["notes"] += f"; excluded runs: all_config MISSING={n_excl_cfg}, not n=10={n_excl_n10}"
        row["notes"] = row["notes"].lstrip("; ")
        cell_rows.append(row)

    # ---- fill cell counts.  CD-8: counts are RUN counts (distinct (project, run_name)), never dir x key rows (R-6 raw counts)
    for row in cell_rows:
        mine = [t for t in table if (t["q"], t["s"]) == (row["q"], row["s"])]
        row["n_table_rows"] = len(mine)  # directory x key rows; NOT a run count
        runs = {(t["wandb_project"], t["run_name"]): t for t in mine}
        row["n_runs"] = len(runs)
        # CR-10: a run counts as evidence use only if EVERY wandb dir row of that (project, run_name) is "yes"
        row["n_runs_evidence_use"] = sum(1 for k in runs if all(t["evidence_use"] == "yes" for t in mine
                                                             if (t["wandb_project"], t["run_name"]) == k))
        cc = {}
        for t in runs.values():
            cc[t["cohort"]] = cc.get(t["cohort"], 0) + 1
        row["counts_by_cohort"] = ";".join(f"{k}:{v}" for k, v in sorted(cc.items()))
        if any(t["cohort"] == "legacy" for t in mine):
            row["cohort_note"] = LEGACY_NOTE

    # ---- deterministic row_id
    table.sort(key=lambda t: (t["q"], t["s"], t["run_name"], t["metric_key"], t["provenance_file"]))
    for i, t in enumerate(table, 1):
        t["row_id"] = f"T{i:05d}"

    if dry_run:
        print(f"# DRY-RUN as_of={as_of} (no S1_* file written, no run history opened)", file=stdout)
        for row in cell_rows:
            print("\t".join(["CELL", f"({row['q']},{row['s']})", row["s1_class"], row["q_token"], f"key_status={row['key_status']}",
                             f"keys={row['metric_keys_read'] or '-'}", row["values_policy"], row["notes"]]), file=stdout)
        seen = set()
        for r in log.rows:
            if r["event"] in ("would_open", "name_only") and (r["event"], r["path"], r["purpose"]) not in seen:
                seen.add((r["event"], r["path"], r["purpose"]))
                print("\t".join([r["event"].upper(), r["adapter"], r["path"], r["purpose"], r["note"]]), file=stdout)
        print(f"# NOTE: run history files are listed only with --resolve-runs (needs wandb-metadata.json reads, no values)", file=stdout)
        return 0

    # ---- write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"S1_table.csv": to_csv_text(TABLE_FIELDS, table), "S1_cells.csv": to_csv_text(CELL_FIELDS, cell_rows),
               access_log_name: None}
    log.rows.append({"ts_utc": log.now_fn(), "cell": "-", "event": "info", "adapter": "s1", "path": "-", "sha256": "",
                     "purpose": "end of run", "note": f"{len(table)} table rows, {len(cell_rows)} cells"})
    outputs[access_log_name] = to_csv_text(ACCESS_FIELDS, log.rows)
    for n, txt in outputs.items():
        (out_dir / n).write_text(txt, encoding="utf-8")
    man = [f"{sha256_text(txt)}  {_rel(root, out_dir)}/{n}" for n, txt in outputs.items()]
    with open(out_dir / "S1_manifest.sha256", "a", encoding="utf-8") as fh:
        fh.write("\n".join(man) + "\n")
    mism = [r for r in cell_rows if not r["class_match"]]
    print(f"S1 done: {len(cell_rows)} cells, {len(table)} table rows, {len(mism)} class != PREREG expected", file=stdout)
    for r in mism:
        print("DISCREPANCY:", r["q"], r["s"], r["reason"], file=stdout)
    return 0


def _rel(root: Path, d: Path) -> str:
    """manifest label of a directory: path relative to root when inside it (default case == OUT_REL), else the path as given."""
    try:
        return str(Path(d).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--root", default=str(root))
    ap.add_argument("--out", default=None, help=f"default <root>/{OUT_REL}")
    ap.add_argument("--s0-dir", default=None, help="dir holding the S0 inputs/snapshot/INTERP_S1.md (default: --out, i.e. unchanged behaviour)")
    ap.add_argument("--wandb-root", default="wandb")
    ap.add_argument("--dry-run", action="store_true", help="print planned files/keys only; no value read, nothing written")
    ap.add_argument("--resolve-runs", action="store_true", help="with --dry-run: also read wandb-metadata.json (names only)")
    ap.add_argument("--access-log-name", default="S1_access_log.csv")
    ap.add_argument("--s0-prefix", default=S0_PREFIX_DEFAULT,
                    help="S0 file-name prefix (default S0r2; the superseded r1 'S0' is selectable only here)")
    a = ap.parse_args(argv)
    rt = Path(a.root)
    out = Path(a.out) if a.out else rt / OUT_REL
    os.chdir(rt)
    return run_s1(rt, out, dry_run=a.dry_run, wandb_root=a.wandb_root, access_log_name=a.access_log_name,
                  resolve_runs=a.resolve_runs, s0_prefix=a.s0_prefix,
                  s0_dir=(Path(a.s0_dir) if a.s0_dir else (rt / OUT_REL if a.out else None)))


if __name__ == "__main__":
    sys.exit(main())
