"""P-C1 judge: rules loader + mechanical family tokens (spec §2.3, §3). No prose, no status words.

* FROZEN_RULES are parsed ONLY from the ```yaml fence of the pre-registration whose body starts with
  ``exp_id:`` (spec §2.3 "FROZEN_RULES 초안"). Every threshold used by gates / classes / family logic is read
  through ``rule()`` / ``num()`` from that dict (I-9); a missing key raises FrozenRuleMissing.
* At import time ``czz2`` of the parsed block is compared with ``analysis.z_dynamics.judge.FROZEN_RULES["czz2"]``
  (deep-equal); a mismatch (or an unreadable prereg) exits with status 2 (spec §2.3 last paragraph).
* Freeze constants (top of file): ``FROZEN_PREREG_SHA256`` (None = DRAFT mode), ``DEFAULT_PREREG``, ``EXP_ID``.
  Module load enforces the sha when it is set. In DRAFT mode a loud banner is printed to stderr and every token
  function / the aggregator raises JudgeAbort (``ALLOW_DRAFT_TOKENS`` is flipped only by tests). The env override
  ``PC1_PREREG`` is honoured in DRAFT mode only.
* ``aggregate`` / CLI (``python -m analysis.ltf_decode.judge --data <run dir> --manifest <M1_manifest.csv> --out
  <verdict.json>``): reads run_manifest.json, gates.json, cell_stats.csv; I-8: a record (analysed) or an exclusion
  record for each of the 30 cells and a complete run manifest, else exit 2. Exit 0 = tokens written, 3 = written with
  >= 1 excluded cell, 2 = abort, 4 = HALT (G5' failure, or the G1'/G-P3 failed-cell count above the max in
  ``FROZEN_RULES.halt_rules``; all cells are processed first, then tokens are withheld).
* ``family_token`` / ``key_xt`` / ``key_e2`` are the §2.3 / §3 decision functions. They return tokens and row ids
  only (``XT-a`` ...); the reading text lives in the prereg and is not reproduced here.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import statistics
import sys
from typing import Any

import yaml

# ---- single freeze-time constants (set these, and only these, at freeze; any edit changes the judge.py sha) ----
FROZEN_PREREG_SHA256: str | None = "84b20d127299e08a469e725e3d4d1c7a60ade12715adc965fa0a2aacfe1f4ee2"  # None = DRAFT mode (loud banner, no tokens). Set to the freeze sha.
DEFAULT_PREREG = ("/home/ayp/project/trm/lab/23_reports/11_lab-reports/"
                  "2026-10-21_pc1-ltf-per-loop-decode-prereg.md")      # path of the frozen document
EXP_ID = "EXP-022"                             # replaced by the allocated EXP id at freeze
DRAFT_MODE = FROZEN_PREREG_SHA256 is None
ALLOW_DRAFT_TOKENS = False                     # tests flip this; the CLIs never do

EXIT_OK, EXIT_ABORT = 0, 2


class FrozenRuleMissing(KeyError):
    """A rule/threshold is not in FROZEN_RULES (I-9)."""


class JudgeAbort(RuntimeError):
    """Inputs inconsistent (czz2 mismatch, sha mismatch, unparsable block)."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def parse_frozen_rules(prereg_path: str, expected_sha256: str | None = None) -> dict:
    if expected_sha256 is not None and sha256_file(prereg_path) != expected_sha256:
        raise JudgeAbort(f"prereg sha256 {sha256_file(prereg_path)} != expected {expected_sha256}")
    with open(prereg_path, encoding="utf-8") as f:
        text = f.read()
    blocks = [b for b in re.findall(r"```yaml\n(.*?)```", text, flags=re.S) if b.lstrip().startswith("exp_id:")]
    if len(blocks) != 1:
        raise JudgeAbort(f"expected exactly 1 FROZEN_RULES yaml fence starting 'exp_id:', found {len(blocks)}")
    rules = yaml.safe_load(blocks[0])
    if not isinstance(rules, dict) or "czz2" not in rules:
        raise JudgeAbort("FROZEN_RULES block has no czz2")
    return rules


CZZ1_USED_KEYS = ("PF_A1", "EPS_DROP", "V_PLUS", "V_MINUS")


def check_czz2(rules: dict) -> None:
    """czz2 deep-equal to EXP-017's (spec §2.3) AND czz1_used[k] == EXP-017 czz1[k] for the four keys the shared
    EXP-017 functions read on P-C1's behalf (PF_A1, EPS_DROP via cell_M_counts; V_PLUS, V_MINUS here). I-9."""
    from analysis.z_dynamics import judge as zj
    if rules["czz2"] != zj.FROZEN_RULES["czz2"]:
        raise JudgeAbort("P-C1 czz2 is not deep-equal to analysis.z_dynamics.judge.FROZEN_RULES['czz2']")
    used = rules.get("czz1_used")
    if not isinstance(used, dict) or any(k not in used for k in CZZ1_USED_KEYS):
        raise JudgeAbort("P-C1 czz1_used is missing a key of " + str(CZZ1_USED_KEYS))
    for k in CZZ1_USED_KEYS:
        ref = zj.FROZEN_RULES.get("czz1", {}).get(k, "<absent>")
        if used[k] != ref:
            raise JudgeAbort(f"P-C1 czz1_used[{k}]={used[k]} != EXP-017 czz1[{k}]={ref}")


def _default_path() -> str:
    # the env override exists for tests / drafts only; in frozen mode the constant is the only path
    return os.environ.get("PC1_PREREG", DEFAULT_PREREG) if DRAFT_MODE else DEFAULT_PREREG


def load_rules(prereg_path: str | None = None, expected_sha256: str | None = None) -> dict:
    rules = parse_frozen_rules(prereg_path or _default_path(), expected_sha256)
    check_czz2(rules)
    return rules


DRAFT_BANNER = ("*" * 78 + "\nP-C1 judge: DRAFT MODE (FROZEN_PREREG_SHA256 is None): rules are NOT sha-checked and "
                "this judge REFUSES to emit tokens.\n" + "*" * 78)
if DRAFT_MODE:
    print(DRAFT_BANNER, file=sys.stderr)


def require_tokens_allowed() -> None:
    if DRAFT_MODE and not ALLOW_DRAFT_TOKENS:
        raise JudgeAbort("DRAFT mode: FROZEN_PREREG_SHA256 is None; tokens are not emitted")


try:
    FROZEN_RULES: dict[str, Any] = load_rules(None, FROZEN_PREREG_SHA256)
except (JudgeAbort, OSError, yaml.YAMLError) as _e:  # spec §2.3: refuse (exit 2) at import time
    print(f"ltf_decode.judge: refusing: {_e}", file=sys.stderr)
    raise SystemExit(EXIT_ABORT)


def rule(path: str, rules: dict | None = None) -> Any:
    node: Any = FROZEN_RULES if rules is None else rules
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise FrozenRuleMissing(f"I-9: '{path}' is not in FROZEN_RULES (missing '{part}')")
        node = node[part]
    return node


def num(path: str, rules: dict | None = None) -> float:
    v = rule(path, rules)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FrozenRuleMissing(f"I-9: '{path}' is not numeric (got {v!r})")
    return v


def t_stat() -> list[int]:
    lo, hi = rule("readouts.t_stat")
    return list(range(int(lo), int(hi) + 1))


def t_all() -> list[int]:
    lo, hi = rule("readouts.t_all")
    return list(range(int(lo), int(hi) + 1))


def dgrid() -> list[int]:
    return [int(x) for x in rule("readouts.dgrid_descriptive")]


def extrapolation_range() -> list[int] | None:
    v = rule("readouts.extrapolation_descriptive")
    return None if v is None else list(range(int(v[0]), int(v[1]) + 1))


def need_fraction(path: str) -> tuple[int, int]:
    m = re.search(r"ceil\((\d+)n'?/(\d+)\)", str(rule(path)))
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse ceil(a n/b) from {path}")
    return int(m.group(1)), int(m.group(2))


def ceil_frac(n: int, frac: tuple[int, int]) -> int:
    a, b = frac
    return -((-a * n) // b)


def exclusion_divisor() -> int:
    m = re.search(r"floor\(n_frozen/(\d+)\)", str(rule("gate_failure_policy.family")))
    if not m:
        raise FrozenRuleMissing("I-9: cannot parse floor(n_frozen/d)")
    return int(m.group(1))


# ------------------------------------------------------------------ class rules (G / C / P / U)

_CMP = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def _clause(c: str) -> tuple[str, str, float]:
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(>=|<=|<|>)\s*([0-9.]+)\s*", c)
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse class clause {c!r}")
    return m.group(1), m.group(2), float(m.group(3))


def outcome_class(final_test: float | None, train_proxy: float | None) -> str:
    """G / C / P / U (U = C-candidate with train_final_proxy below its C threshold); 'UNKNOWN' if test missing.
    Keys: final_probe_test_exact, train_final_proxy (class_rules, spec §2.3)."""
    if final_test is None or (isinstance(final_test, float) and math.isnan(final_test)):
        return "UNKNOWN"
    vals = {"final_probe_test_exact": final_test, "train_final_proxy": train_proxy}

    def ok(cl):
        f, op, thr = cl
        v = vals[f]
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else _CMP[op](v, thr)

    g = [_clause(c) for c in str(rule("class_rules.G")).split(" and ")]
    if all(ok(c) for c in g):
        return "G"
    c_parts = [_clause(c) for c in str(rule("class_rules.C")).split(" and ")]
    c_test = [c for c in c_parts if c[0] == "final_probe_test_exact"]
    c_train = [c for c in c_parts if c[0] != "final_probe_test_exact"]
    if all(ok(c) for c in c_test):
        tr = [ok(c) for c in c_train]
        if any(t is None for t in tr):
            return "UNKNOWN"
        if all(tr):
            return "C"
        mu = re.fullmatch(r"\s*C-candidate with (.+)", str(rule("class_rules.U")))
        if not mu:
            raise FrozenRuleMissing("I-9: cannot parse class_rules.U")
        return "U" if ok(_clause(mu.group(1))) else "UNKNOWN"
    ps = str(rule("class_rules.P"))
    m = re.fullmatch(r"\s*([0-9.]+)\s*<\s*(\w+)\s*<\s*([0-9.]+)\s*", ps)
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse class_rules.P = {ps!r}")
    return "P" if float(m.group(1)) < vals[m.group(2)] < float(m.group(3)) else "UNKNOWN"


# ------------------------------------------------------------------ family token (spec §2.3, order)

def family_token(cells: list[dict], n_frozen: int) -> dict:
    """cells: one dict per family cell that was frozen into the family, keys:
        run_name, excluded (bool), Z ('PF'|'stepwise'|'jump'|'unordered'; ignored if excluded), V (float).
    family_order = [INVALID-EMPTY, INVALID-GATE, S2-0, V-FAIL, S2-1, S2-2, S2-3].
    Returns {"token", "annotations": [...], n, n_excl, n_PF, n_prime, need, counts, slack, at_threshold, V_median, Vfam}.
    """
    require_tokens_allowed()
    elig = [c for c in cells if not c["excluded"]]
    n_excl = len(cells) - len(elig)
    n = len(elig)
    out: dict[str, Any] = {"n_frozen": n_frozen, "n": n, "n_excluded": n_excl, "annotations": []}
    if n == 0:
        return dict(out, token="INVALID-EMPTY")
    if n_excl > n_frozen // exclusion_divisor():
        return dict(out, token="INVALID-GATE")
    counts = {z: sum(1 for c in elig if c["Z"] == z) for z in ("PF", "stepwise", "jump", "unordered")}
    n_pf = counts["PF"]
    out.update(counts=counts, n_PF=n_pf)
    need_pf = ceil_frac(n, need_fraction("czz2.family_PF_precedence"))
    if n_pf >= need_pf:
        return dict(out, token="S2-0", need_PF=need_pf, slack=n_pf - need_pf, at_threshold=(n_pf == need_pf))
    n1 = n - n_pf
    need = ceil_frac(n1, need_fraction("czz2.family_need"))
    out.update(n_prime=n1, need=need)
    vv = [c["V"] for c in elig if c["Z"] != "PF"]
    if any(math.isnan(v) for v in vv):
        raise JudgeAbort("V(c) is NaN for an eligible non-PF cell")
    v_med = statistics.median(vv)
    vplus, vminus = num("czz1_used.V_PLUS"), num("czz1_used.V_MINUS")
    vfam = "V+" if v_med >= vplus else ("V~" if v_med >= vminus else "V-")
    out.update(V_median=v_med, Vfam=vfam)
    if vfam == "V-":
        return dict(out, token="V-FAIL")
    if vfam == "V~":
        out["annotations"].append("(validity < 0.50)")
    if counts["stepwise"] >= need:
        tok, win = "S2-1", counts["stepwise"]
    elif counts["jump"] >= need:
        tok, win = "S2-2", counts["jump"]
    else:
        return dict(out, token="S2-3", slack=None, at_threshold=False)
    slack = win - need
    if slack == 0:
        out["annotations"].append("(at threshold: one cell flips the branch)")
    return dict(out, token=tok, slack=slack, at_threshold=(slack == 0))


_NO_READING = {"INVALID-EMPTY", "INVALID-GATE", "S2-0", "V-FAIL"}


def key_xt(g7_token: str) -> str:
    """Row id of key X-T (g7_G token vs the quoted T token); first match wins (spec §3)."""
    require_tokens_allowed()
    if g7_token in _NO_READING:
        return "XT-0"
    return {"S2-2": "XT-a", "S2-1": "XT-b", "S2-3": "XT-c"}[g7_token]


def key_e2(g7_kboth_token: str, full_token: str) -> str:
    """Row id of key E2 (g7_G_Kboth vs full_G); first match wins (spec §3)."""
    require_tokens_allowed()
    if g7_kboth_token in _NO_READING or full_token in _NO_READING:
        return "E2-0"
    pair = {g7_kboth_token, full_token}
    if g7_kboth_token == full_token and g7_kboth_token in ("S2-1", "S2-2"):
        return "E2-same"
    if pair == {"S2-1", "S2-2"}:
        return "E2-diff"
    return "E2-U"


# ------------------------------------------------------------------ aggregator (spec §6 I-8, §2.3, §3)

EXIT_EXCLUSIONS, EXIT_HALT = 3, 4


def pf_from_row(row: dict) -> bool:
    """PF label recomputed with P-C1's own czz1_used (PF_A1, EPS_DROP), not EXP-017's (I-9)."""
    return float(row["a1"]) >= num("czz1_used.PF_A1") and float(row["maxdrop"]) <= num("czz1_used.EPS_DROP")


def cell_Z(row: dict) -> str:
    z = "PF" if pf_from_row(row) else row["Z_raw"]
    if row.get("Z") not in (None, "") and row["Z"] != z:
        raise JudgeAbort(f"{row.get('run_name')}: cell_stats Z={row['Z']} != recomputed {z}")
    return z


def halt_rules() -> tuple[int, str]:
    """(max union count of cells failing G1' or G-P3, G5' policy) from FROZEN_RULES['halt_rules']; fail closed."""
    try:
        hr = rule("halt_rules")
        mx = hr["G1p_or_GP3_failed_cells_max"]
        cnt, g5 = str(hr["count"]), hr["G5p_fail"]
    except (FrozenRuleMissing, KeyError, TypeError) as e:
        raise JudgeAbort(f"FROZEN_RULES halt_rules missing or malformed: {e}")
    if isinstance(mx, bool) or not isinstance(mx, int) or mx < 0 or "union" not in cnt or g5 != "halt":
        raise JudgeAbort(f"unsupported halt_rules: max={mx!r} count={cnt!r} G5p_fail={g5!r}")
    return mx, g5


def aggregate(data_dir: str, manifest_rows: list[dict]) -> dict:
    """Family tokens + key row ids. Raises JudgeAbort on I-8 violations; returns {'halt': True, ...} on HALT."""
    import csv
    import json
    require_tokens_allowed()
    max_fail, _g5_policy = halt_rules()
    try:
        with open(os.path.join(data_dir, "run_manifest.json")) as f:
            man = json.load(f)
        with open(os.path.join(data_dir, "gates.json")) as f:
            gj = json.load(f)
        with open(os.path.join(data_dir, "cell_stats.csv"), newline="") as f:
            cs = list(csv.DictReader(f))
    except OSError as e:
        raise JudgeAbort(f"I-8: incomplete run directory: {e}")
    if man.get("status") != "complete" or man.get("partial"):
        raise JudgeAbort(f"I-8: run_manifest status={man.get('status')!r} partial={man.get('partial')!r}")
    names = [r["run_name"] for r in manifest_rows]
    if len(names) != 30 or len(set(names)) != 30:
        raise JudgeAbort(f"I-8: manifest has {len(names)} cells ({len(set(names))} distinct), need 30")
    recs = gj.get("cells", {})
    analysed, excluded, fail_union = {}, {}, set()
    for r in manifest_rows:
        rec = recs.get(r["run_name"])
        if rec is None:
            raise JudgeAbort(f"I-8: no record and no exclusion record for {r['run_name']}")
        st = rec.get("status")
        if st == "analysed" and r["terminal_ok"]:
            rows = [c for c in cs if c["run_name"] == r["run_name"] and c["split"] == "test" and c["readout_kind"] == "ltf"]
            if len(rows) != 1:
                raise JudgeAbort(f"I-8: {r['run_name']} analysed but {len(rows)} test/ltf cell_stats rows")
            analysed[r["run_name"]] = rows[0]
        elif st == "excluded" and rec.get("exclusion_reason"):
            excluded[r["run_name"]] = rec["exclusion_reason"]
        else:
            raise JudgeAbort(f"I-8: {r['run_name']} has status {st!r} (terminal_ok={r['terminal_ok']}) without a valid record")
        if rec.get("G1") is False or (isinstance(rec.get("G_P3"), dict) and rec["G_P3"].get("pass") is False):
            fail_union.add(r["run_name"])
    g5 = gj.get("G5", {})
    out: dict[str, Any] = {"n_cells": 30, "n_analysed": len(analysed), "excluded": excluded,
                           "G5": {k: g5.get(k) for k in ("unit", "ran", "pass")}, "n_fail_G1_or_GP3": len(fail_union)}
    if not (g5.get("ran") and g5.get("pass")) or len(fail_union) > max_fail:
        return dict(out, halt=True, halt_reason=("G5'" if not (g5.get("ran") and g5.get("pass")) else "I-11"))
    fams: dict[str, list] = {"g7_G": [], "full_G": [], "g7_G_Kboth": []}
    for r in manifest_rows:
        for t in str(r["tags"]).split(";"):
            if t in fams:
                fams[t].append(r["run_name"])
    toks = {}
    for fam, members in fams.items():
        cl = []
        for n in members:
            if n in analysed:
                row = analysed[n]
                cl.append({"run_name": n, "excluded": False, "Z": cell_Z(row), "V": float(row["V"])})
            else:
                cl.append({"run_name": n, "excluded": True, "Z": None, "V": float("nan")})
        toks[fam] = family_token(cl, len(members))
    out["families"] = toks
    out["key_XT"] = key_xt(toks["g7_G"]["token"])
    out["key_E2"] = key_e2(toks["g7_G_Kboth"]["token"], toks["full_G"]["token"])
    return out


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from analysis.ltf_decode import cells
    try:
        res = aggregate(a.data, cells.read_manifest(a.manifest))
    except JudgeAbort as e:
        print(f"judge: abort: {e}", file=sys.stderr)
        return EXIT_ABORT
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    if res.get("halt"):
        return EXIT_HALT
    return EXIT_EXCLUSIONS if res["excluded"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
