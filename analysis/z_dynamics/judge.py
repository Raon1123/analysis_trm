"""EXP-017 judge: apply the FROZEN rules of the pre-registration to run_zdyn outputs.

This file is hash-witnessed before any EXP-017 value exists (EXP-017 §4.6 step 4), so every
threshold used by any decision lives here, in the single ``FROZEN_RULES`` dict:

* ``FROZEN_RULES`` minus the ``gates`` key is ``yaml.safe_load`` of the verbatim YAML block
  between ``<!-- FROZEN_RULES_BEGIN -->`` and ``<!-- FROZEN_RULES_END -->`` of EXP-017 §2.5
  (embedded below as ``FROZEN_RULES_YAML``). At run time the block is re-parsed from ``--prereg``
  and must be deep-equal to it, and the prereg file must have the frozen sha256 (I-7).
* ``FROZEN_RULES["gates"]`` holds the gate tolerances that §6 states only in prose (G3 <= 0.02,
  G4 <= 0.01, I-11 "> 3 cells", I-4 terminal steps, G5 unit, G7 keys). VERIFY-r1 N2 asks that they come
  from one frozen structure instead of being typed loosely where they are used. Every entry carries an
  anchor substring, and the judge checks that each anchor occurs verbatim in the prereg text.

Numbers inside rule strings (``final_probe_test_exact >= 0.9``, ``ceil(2n'/3)``,
``floor(n_frozen/3)``, ``'6->7'``) are parsed with regexes from those strings, not retyped (I-9).
All reads go through ``rule()`` / ``num()``. A missing key raises ``FrozenRuleMissing``, and then
the judge refuses (exit 2). Nothing in the decision path uses ``dict.get(key, default)`` on rules.

Decision order (verdict.json):
  1. I-7 prereg sha / YAML equality / §6 anchors; I-8 a record for every one of the 41 cells and a
     complete run manifest.                                                   -> abort, exit 2
  2. G5 determinism failure or I-11 (> 3 cells failing G3 or G4).             -> HALT, exit 4
  3. per family (C-Z-1, C-Z-2): INVALID-EMPTY (n = 0) > INVALID-GATE
     (> floor(n_frozen/3) excluded) > S1-0 / S2-0 (PF precedence) > branch.  (N2 order)
  4. C-Z-3: S3-0 (any stratum cell excluded or U) > S3-1 > S3-2 > S3-3.
  5. Anything the frozen branches do not cover (I-10) -> token UNCOVERED with a reason.
Exit codes: 0 = verdict written, no cell excluded; 3 = verdict written, >= 1 cell excluded (gate
failure, I-3, I-4 or U); 2 = refused or aborted (no verdict written); 4 = HALT (tokens withheld).

CLI (EXP-017 §4.5):
  uv run python analysis/z_dynamics/judge.py --data <dir> --prereg <EXP-017 md> --out <dir>/verdict.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from typing import Any

import yaml

EXP_ID = "EXP-017"
# sha256 of the frozen pre-registration (EXP-017_FREEZE.sha256, VERIFY-r1 audited_artifacts.prereg).
EXPECTED_PREREG_SHA256 = "567e6a40e04bcbffeb3327de76730a1a1dae6206217cf8049b58fb0f4d8cd107"
DEFAULT_PREREG = "/mnt/ayp/agents/trm/21_experiments/11_lab-experiments/EXP-017_z-dynamics-latent-formation.md"

# ---------------------------------------------------------------------------------------------
# Verbatim copy of the EXP-017 §2.5 FROZEN_RULES YAML block (do not edit; I-7 / I-9).
# ---------------------------------------------------------------------------------------------
FROZEN_RULES_YAML = r'''exp_id: EXP-017
h_id: H-201
readouts: {primary: preliminary_answer_state, j_all: [1, 18], j_stat: [1, 17], native_j: [6, 12, 18], trained_j: 18}
splits: {decode: test_all_1000, pr: train_first_512, reproduction_gate: test_first_512}
class_rules:
  G: "final_probe_test_exact >= 0.9"          # closeout final_class=success
  C: "final_probe_test_exact <= 0.05 and train_probe_exact_G4 >= 0.90"
  P: "0.05 < final_probe_test_exact < 0.9"
  U: "C-candidate with train_probe_exact_G4 < 0.90"
czz1:
  cell_class_order: [PF, M+, M-, M0]
  PF_A1: 0.90                # PF: a_1 >= PF_A1 and maxdrop <= EPS_DROP
  R_MIN: 0.20
  EPS_DROP: 0.10
  V_PLUS: 0.50
  V_MINUS: 0.05
  T_TRIVIAL: 0.50
  T_all_nan_default: 1.0
  family_PF_precedence: "n_PF >= ceil(2n/3) -> S1-0; else votes/medians over the n' = n - n_PF non-PF cells"
  family_need: "ceil(2n'/3)"
  subclass: {jump_frac: 0.8, staircase_boundary_frac: 0.6, boundary_diffs: ["6->7", "12->13"]}
czz2:
  HAM_MAX: 3
  m_range: "0..ord(sigma)-1 per sample; unique argmin required"
  on_path: "0..k"                       # used for gamma (no mixture qualifier)
  proper_partial: "m* in A_i (identifiable: exists cycle length L with m mod L not in {0, 1 mod L, k mod L}) and Ham(y, sigma^m*) < Ham_mix(y) (best per-cycle mixture of sigma^0, sigma^1, sigma^k)"
  P_int_denominator: "entries (i, j<=17) over informative samples (A_i nonempty); 0 if none"
  GAMMA_MIN: 0.50
  P_MAX: 0.01
  N_PERM: 2000
  PERM_SEED: 20260923
  PINT_FLOOR: 0.10
  PINT_MULT: 10.0
  chance: "uniform S_10, same proper conditions, 2000 draws per distinct cycle type, mean over informative samples"
  family_PF_precedence: "PF cells from czz1 -> n_PF >= ceil(2n/3) -> S2-0; else votes over non-PF cells"
  family_need: "ceil(2n'/3)"
  vis_support: 0.10                     # of informative samples, mixture-qualified
czz3:
  statistic: "E_L = sum_{j=1..17}|ln PR_{j+1} - ln PR_j| - |ln PR_18 - ln PR_1| on z_L^{(j)}, train_first_512"
  descriptive_only: ["Theta_L (tortuosity, floor 0.05)", "Lambda_L"]
  direction: "C > G"
  rule_S3_1: "all strata: min_C > max_G"
  rule_S3_2: "all strata: max_C < min_G"
  precedence: [S3-0, S3-1, S3-2, S3-3]
  exact_null: {S3_1: 0.0556, S3_2: 0.0556, union: 0.1111}
families:
  F1_trm_D1: [pp_base_tf_z_iter_k3_s1, pp_base_tf_z_iter_k3_s2, pp_base_tf_z_iter_k3_s3, pp_base_tf_z_iter_k5_s1, pp_base_tf_z_iter_k6_s1, pp_base_tf_z_iter_k7_s1, pp_base_tf_z_iter_k7_s2, pp_base_tf_z_iter_k7_s3, pp_base_tf_z_iter_k10_s1, pp_base_tf_z_iter_k10_s2, pp_base_tf_z_iter_k10_s3, pp_fig1_tf_z_iter_k3_s1, pp_fig1_tf_z_iter_k6_s1, pp_fig1_tf_z_iter_k7_s1, pp_fig1_tf_z_iter_k10_s1]
  F2_singlez_D1: [pp_fig1_tf_noz_iter_k3_s1, pp_fig1_tf_noz_iter_k10_s1, pp_fig1_tf_noz_iter_k10_s2]
  F3_singlez_D0: [pp_seedext_tf_noz_iter_k3_s2, pp_seedext_tf_noz_iter_k3_s3, pp_seedext_tf_noz_iter_k4_s2, pp_seedext_tf_noz_iter_k4_s3, pp_seedext_tf_noz_iter_k7_s2, pp_seedext_tf_noz_iter_k8_s2, pp_seedext_tf_noz_iter_k10_s2, pp_seedext_tf_noz_iter_k10_s3]
strata:
  S_A_trm_D1_k5: {G: [pp_base_tf_z_iter_k5_s1], C: [pp_base_tf_z_iter_k5_s2, pp_fig1_tf_z_iter_k5_s1]}
  S_B_trm_D1_k6: {G: [pp_base_tf_z_iter_k6_s1, pp_fig1_tf_z_iter_k6_s1], C: [pp_base_tf_z_iter_k6_s2]}
  S_C_singlez_D0_k7: {G: [pp_seedext_tf_noz_iter_k7_s2], C: [pp_seedext_tf_noz_iter_k7_s3]}
descriptive_only_C: [pp_fig1_tf_noz_iter_k5_s1, pp_fig1_tf_noz_iter_k6_s1, pp_fig1_tf_noz_iter_k7_s1, pp_seedext_tf_noz_iter_k5_s2, pp_seedext_tf_noz_iter_k5_s3, pp_seedext_tf_noz_iter_k6_s2, pp_seedext_tf_noz_iter_k6_s3]
descriptive_only_P: [pp_base_tf_z_iter_k5_s3, pp_base_tf_z_iter_k6_s3, pp_fig1_tf_noz_iter_k10_s3, pp_seedext_tf_noz_iter_k8_s3]
gate_failure_policy:
  family: "exclude failed cells; n := remaining; if more than floor(n_frozen/3) excluded -> family verdict INVALID-GATE; if n = 0 -> INVALID-EMPTY"
  stratum: "any stratum cell excluded or U -> S3-0"
boundaries: "all thresholds inclusive as written (>=, <=); ties in E_L (incl. both 0) -> S3-3; gamma undefined -> 0; constant a_j -> M0 unless a_1 >= 0.90 (then PF)"
'''

# Gate tolerances stated in EXP-017 §6 prose only (VERIFY-r1 N2). Each "anchor" must occur verbatim
# in the prereg; the judge refuses otherwise.
FROZEN_GATES: dict[str, Any] = {
    "source": "EXP-017 §6 gate table, I-4 and I-11 prose (outside the YAML block; VERIFY-r1 N2)",
    "G3_REL_TOL": {"value": 0.02, "anchor": "relative difference ≤ 0.02"},
    "G4_ABS_TOL": {"value": 0.01, "anchor": "\\|Δ\\| ≤ 0.01 each"},
    "I11_MAX_FAILING_CELLS": {"value": 3, "anchor": "CPU numerics fail G3 or G4 on more than 3 cells"},
    "G5_UNIT": {"value": "pp_base_tf_z_iter_k5_s1", "anchor": "re-run `pp_base_tf_z_iter_k5_s1` end-to-end"},
    "G5_FAILURE": {"value": "HALT", "anchor": "every per_call / pr_traj / cell_stats value is identical",
                   "note": "a determinism failure is pipeline-level: stop-and-fix, not cell exclusion (N2)"},
    "TERMINAL_STEP": {"value": {"D1": 122050, "D0": 244100}, "anchor": "step ≠ 122050 for D1, ≠ 244100 for D0"},
    "G7_TEXT": {
        "value": [
            "arch.H_cycles = 3, arch.L_cycles = 6, arch.L_layers = 2, arch.hidden_size = 512, arch.expansion = 4, "
            "arch.num_heads = 8, arch.pos_encodings = rope, arch.puzzle_emb_len = 16, arch.puzzle_emb_ndim = 512, "
            "arch.halt_max_steps = 1, arch.forward_dtype = bfloat16, arch.mlp_t = false, arch.no_ACT_continue = true",
            "lr = 1e-4, puzzle_emb_lr = 1e-4, weight_decay = 1.0, puzzle_emb_weight_decay = 1.0, "
            "global_batch_size = 2048, ema = true, ema_rate = 0.999, eval_interval = 2000, lr_warmup_steps = 2000, "
            "lr_min_ratio = 1.0, beta1 = 0.9, beta2 = 0.95, z_probe_size = 512",
        ],
        "anchor": None,  # each value line is itself the anchor
    },
    "G7_WITHIN_GROUP": {"value": ["arch.name", "epochs", "data_root"],
                        "anchor": "**Within a family or stratum**, arch.name, epochs and the data root must also be identical"},
}

FROZEN_RULES: dict[str, Any] = dict(yaml.safe_load(FROZEN_RULES_YAML))
FROZEN_RULES["gates"] = FROZEN_GATES

TOKENS_CZ1 = ("S1-0", "S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-6")
TOKENS_CZ2 = ("S2-0", "S2-1", "S2-2", "S2-3")
TOKENS_CZ3 = ("S3-0", "S3-1", "S3-2", "S3-3")
TOKENS_SPECIAL = ("INVALID-GATE", "INVALID-EMPTY", "UNCOVERED", "HALT")

EXIT_OK, EXIT_ABORT, EXIT_EXCLUSIONS, EXIT_HALT = 0, 2, 3, 4


class FrozenRuleMissing(KeyError):
    """A rule/threshold was requested that is not in FROZEN_RULES (I-9). The judge refuses."""


class JudgeAbort(RuntimeError):
    """Inputs incomplete or inconsistent (I-7, I-8, data inconsistency). No verdict is written."""


class Uncovered(Exception):
    """An outcome the frozen branches do not cover (I-10)."""


# ---------------------------------------------------------------------------------------------
# Rule access (the only way any threshold is read)
# ---------------------------------------------------------------------------------------------

def rule(path: str, rules: dict | None = None) -> Any:
    """Return FROZEN_RULES[a][b]... for path 'a.b...'; raise FrozenRuleMissing if absent."""
    node: Any = FROZEN_RULES if rules is None else rules
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise FrozenRuleMissing(f"I-9: '{path}' is not in FROZEN_RULES (missing '{part}')")
        node = node[part]
    if isinstance(node, dict) and set(node) >= {"value", "anchor"}:
        return node["value"]
    return node


def num(path: str, rules: dict | None = None) -> float:
    v = rule(path, rules)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FrozenRuleMissing(f"I-9: '{path}' is not a numeric threshold in FROZEN_RULES (got {v!r})")
    return v


def need_fraction(path: str, rules: dict | None = None) -> tuple[int, int]:
    """Parse 'ceil(2n'/3)' (or 'n_PF >= ceil(2n/3) ...') -> (2, 3)."""
    s = rule(path, rules)
    m = re.search(r"ceil\((\d+)n'?/(\d+)\)", str(s))
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse ceil(a n/b) from '{path}' = {s!r}")
    return int(m.group(1)), int(m.group(2))


def ceil_frac(n: int, frac: tuple[int, int]) -> int:
    a, b = frac
    return -((-a * n) // b)  # exact integer ceil(a*n/b)


def exclusion_divisor(rules: dict | None = None) -> int:
    s = rule("gate_failure_policy.family", rules)
    m = re.search(r"floor\(n_frozen/(\d+)\)", str(s))
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse floor(n_frozen/d) from gate_failure_policy.family = {s!r}")
    return int(m.group(1))


def boundary_diff_indices(rules: dict | None = None) -> list[int]:
    """'6->7' -> index 5 of np.diff(a_1..a_17) (a_7 - a_6)."""
    out = []
    for s in rule("czz1.subclass.boundary_diffs", rules):
        m = re.fullmatch(r"(\d+)->(\d+)", str(s))
        if not m or int(m.group(2)) != int(m.group(1)) + 1:
            raise FrozenRuleMissing(f"I-9: cannot parse boundary diff {s!r}")
        out.append(int(m.group(1)) - 1)
    return out


def j_stat(rules: dict | None = None) -> int:
    lo, hi = rule("readouts.j_stat", rules)
    if lo != 1:
        raise FrozenRuleMissing("I-9: readouts.j_stat must start at 1")
    return int(hi)


_CMP = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def _parse_clause(clause: str) -> tuple[str, str, float]:
    m = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*(>=|<=|<|>)\s*([0-9.]+)\s*", clause)
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse class clause {clause!r}")
    return m.group(1), m.group(2), float(m.group(3))


def class_rule_parts(rules: dict | None = None) -> dict:
    """Parse class_rules strings into comparisons.

    G: 'final_probe_test_exact >= 0.9'
    C: 'final_probe_test_exact <= 0.05 and train_probe_exact_G4 >= 0.90'
    P: '0.05 < final_probe_test_exact < 0.9'
    U: 'C-candidate with train_probe_exact_G4 < 0.90'
    """
    g = [_parse_clause(c) for c in str(rule("class_rules.G", rules)).split(" and ")]
    c = [_parse_clause(x) for x in str(rule("class_rules.C", rules)).split(" and ")]
    ps = str(rule("class_rules.P", rules))
    m = re.fullmatch(r"\s*([0-9.]+)\s*<\s*(\w+)\s*<\s*([0-9.]+)\s*", ps)
    if not m:
        raise FrozenRuleMissing(f"I-9: cannot parse class_rules.P = {ps!r}")
    p = (float(m.group(1)), m.group(2), float(m.group(3)))
    us = str(rule("class_rules.U", rules))
    mu = re.fullmatch(r"\s*C-candidate with (.+)", us)
    if not mu:
        raise FrozenRuleMissing(f"I-9: cannot parse class_rules.U = {us!r}")
    u = _parse_clause(mu.group(1))
    return {"G": g, "C": c, "P": p, "U": u}


def outcome_class(final_test: float | None, train_g4: float | None, rules: dict | None = None) -> str:
    """G / C / P / U from the closeout test value and the G4-remeasured train-probe exact."""
    parts = class_rule_parts(rules)
    vals = {"final_probe_test_exact": final_test, "train_probe_exact_G4": train_g4}

    def ok(cl):
        field, op, thr = cl
        v = vals[field]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return _CMP[op](v, thr)

    if final_test is None or math.isnan(final_test):
        return "UNKNOWN"
    if all(ok(cl) for cl in parts["G"]):
        return "G"
    c_test = [cl for cl in parts["C"] if cl[0] == "final_probe_test_exact"]
    c_train = [cl for cl in parts["C"] if cl[0] != "final_probe_test_exact"]
    if all(ok(cl) for cl in c_test):  # C-candidate
        tr = [ok(cl) for cl in c_train]
        if any(t is None for t in tr):
            return "UNKNOWN"
        if all(tr):
            return "C"
        if ok(parts["U"]):
            return "U"
        return "UNKNOWN"
    lo, field, hi = parts["P"]
    if lo < vals[field] < hi:
        return "P"
    return "UNKNOWN"


# ---------------------------------------------------------------------------------------------
# Gate predicates (tolerances from FROZEN_RULES["gates"])
# ---------------------------------------------------------------------------------------------

def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def g3_pass(g3: dict | None, arch: str, rules: dict | None = None) -> bool:
    """G3: PR(zHn[3]) / PR(S_L[18]) on the train probe vs snapshot PR; rel diff <= G3_REL_TOL. trm only."""
    if arch != "trm":
        return True  # not applicable: trm_singlez snapshots hold None latents
    tol = num("gates.G3_REL_TOL", rules)
    if not g3:
        return False
    for a, b in (("pr_zH", "pr_zH_snapshot"), ("pr_zL", "pr_zL_snapshot")):
        x, y = g3.get(a), g3.get(b)  # data fields, not rules
        if not (_finite(x) and _finite(y)) or y == 0:
            return False
        if abs(x - y) / abs(y) > tol:
            return False
    return True


def g4_pass(g4: dict | None, final_test: float | None, rules: dict | None = None) -> bool:
    tol = num("gates.G4_ABS_TOL", rules)
    if not g4 or not _finite(final_test):
        return False
    t, tr, snap = g4.get("test512_exact"), g4.get("train512_exact"), g4.get("snapshot_train_exact")
    if not (_finite(t) and _finite(tr) and _finite(snap)):
        return False
    return abs(t - final_test) <= tol and abs(tr - snap) <= tol


def parse_g7_expected(rules: dict | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in rule("gates.G7_TEXT", rules):
        for item in line.split(", "):
            key, val = item.split(" = ")
            out[key.strip()] = _coerce(val.strip())
    return out


def _coerce(s: str) -> Any:
    if s in ("true", "false"):
        return s == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------------------------
# Cell lists derived from FROZEN_RULES
# ---------------------------------------------------------------------------------------------

def frozen_cells(rules: dict | None = None) -> dict[str, str]:
    """run_name -> frozen class (G/C/P) from families / strata / descriptive lists (41 cells)."""
    out: dict[str, str] = {}

    def put(name, cls):
        if name in out and out[name] != cls:
            raise JudgeAbort(f"FROZEN_RULES lists {name} as both {out[name]} and {cls}")
        out[name] = cls

    for names in rule("families", rules).values():
        for n in names:
            put(n, "G")
    for st in rule("strata", rules).values():
        for n in st["G"]:
            put(n, "G")
        for n in st["C"]:
            put(n, "C")
    for n in rule("descriptive_only_C", rules):
        put(n, "C")
    for n in rule("descriptive_only_P", rules):
        put(n, "P")
    return out


def arch_of(run_name: str) -> str:
    return "trm_singlez" if "_noz_" in run_name else "trm"


# ---------------------------------------------------------------------------------------------
# Per-cell class recomputation (independent of stats.py labels)
# ---------------------------------------------------------------------------------------------

def _f(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None or v == "" or v == "None":
        return float("nan")
    return float(v)


def recompute_M(row: dict, rules: dict | None = None) -> str:
    a1, R, md, mr = _f(row, "a1"), _f(row, "R"), _f(row, "maxdrop"), _f(row, "maxrise")
    if any(math.isnan(x) for x in (a1, R, md, mr)):
        raise Uncovered("C-Z-1 cell quantities missing/NaN")
    order = rule("czz1.cell_class_order", rules)
    if list(order) != ["PF", "M+", "M-", "M0"]:
        raise FrozenRuleMissing("I-9: czz1.cell_class_order is not [PF, M+, M-, M0]")
    eps, rmin, pfa = num("czz1.EPS_DROP", rules), num("czz1.R_MIN", rules), num("czz1.PF_A1", rules)
    if a1 >= pfa and md <= eps:
        return "PF"
    if R >= rmin and md <= eps:
        return "M+"
    if R <= -rmin and mr <= eps:
        return "M-"
    return "M0"


def recompute_thr(row: dict, rules: dict | None = None) -> float:
    p0 = _f(row, "P_int0")
    if math.isnan(p0):
        p0 = 0.0  # no informative sample: P_int^0 undefined; the floor applies
    return max(num("czz2.PINT_FLOOR", rules), num("czz2.PINT_MULT", rules) * p0)


def recompute_Z(row: dict, M: str, rules: dict | None = None) -> str:
    if M == "PF":
        return "PF"
    g = _f(row, "gamma")
    if math.isnan(g):
        g = 0.0  # boundaries: gamma undefined -> 0
    p = _f(row, "p")
    pint = _f(row, "P_int")
    if math.isnan(pint):
        pint = 0.0  # P_int := 0 if the cell has no informative sample
    thr = recompute_thr(row, rules)
    if g >= num("czz2.GAMMA_MIN", rules):
        if math.isnan(p):
            raise Uncovered("gamma >= GAMMA_MIN but no permutation p recorded")
        if p <= num("czz2.P_MAX", rules):
            return "stepwise" if pint >= thr else "jump"
    return "unordered"


# ---------------------------------------------------------------------------------------------
# Core judge (pure; tests call this directly)
# ---------------------------------------------------------------------------------------------

def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def cell_eligibility(rec: dict, rules: dict | None = None) -> tuple[bool, list[str], str]:
    """Return (gate_ok, failed_gates, class) for one gates.json cell record."""
    name = rec["run_name"]
    arch = rec.get("arch")
    if arch != arch_of(name):
        raise JudgeAbort(f"{name}: arch {arch!r} disagrees with the run name")
    final = rec.get("final_probe_test_exact")
    if rec.get("status") != "analysed":
        return False, [f"excluded:{rec.get('exclusion_reason')}"], "excluded"
    failed = []
    for g in ("G0", "G1", "G2", "G6", "G7", "I3_ok", "I4_ok"):
        if rec.get(g) is not True:
            failed.append(g)
    if not g3_pass(rec.get("G3"), arch, rules):
        failed.append("G3")
    if not g4_pass(rec.get("G4"), final, rules):
        failed.append("G4")
    g4 = rec.get("G4") or {}
    cls = outcome_class(final, g4.get("train512_exact"), rules)
    return (not failed), failed, cls


def judge_records(cells: dict[str, dict], stats_rows: list[dict], g5: dict | None,
                  rules: dict | None = None) -> tuple[dict, int]:
    """Apply §2 to in-memory records. Returns (verdict dict, exit code). Raises JudgeAbort / FrozenRuleMissing."""
    R = FROZEN_RULES if rules is None else rules
    frozen = frozen_cells(R)
    missing = sorted(set(frozen) - set(cells))
    if missing:
        raise JudgeAbort(f"I-8: no record (or exclusion record) for {len(missing)} cell(s): {missing}")
    extra = sorted(set(cells) - set(frozen))
    if extra:
        raise JudgeAbort(f"unexpected cell records not in FROZEN_RULES: {extra}")

    stats: dict[tuple[str, str, str], dict] = {}
    for r in stats_rows:
        stats[(r["run_name"], r["split"], r["readout_kind"])] = r

    # ---- per-cell eligibility and class
    per_cell: dict[str, dict] = {}
    n_g34_fail = 0
    for name, fcls in frozen.items():
        ok, failed, cls = cell_eligibility(cells[name], R)
        if cls not in ("excluded", "UNKNOWN") and cls != fcls and not (fcls == "C" and cls == "U"):
            raise JudgeAbort(f"{name}: computed class {cls} disagrees with frozen class {fcls}")
        if "G3" in failed or "G4" in failed:
            n_g34_fail += 1
        if ok and (name, "test", "prelim") not in stats and fcls != "P":
            raise JudgeAbort(f"I-8: {name} passed gates but has no test/prelim cell_stats row")
        per_cell[name] = {"frozen_class": fcls, "class": cls, "gate_ok": ok, "failed": failed}

    verdict: dict[str, Any] = {"exp_id": EXP_ID, "h_id": rule("h_id", R), "cells": per_cell,
                               "decision_order": ["I-7/I-8 abort", "G5/I-11 HALT", "INVALID-EMPTY",
                                                  "INVALID-GATE", "S1-0/S2-0 (PF)", "branch"]}

    # ---- halts
    halts = []
    g5_ok = bool(g5) and g5.get("ran") is True and g5.get("identical") is True \
        and g5.get("unit") == rule("gates.G5_UNIT", R)
    if not g5_ok:
        halts.append({"gate": "G5", "detail": g5})
    if n_g34_fail > num("gates.I11_MAX_FAILING_CELLS", R):
        halts.append({"gate": "I-11", "n_cells_failing_G3_or_G4": n_g34_fail})
    if halts:
        verdict["HALT"] = halts
        verdict["C-Z-1"] = {f: {"token": "HALT"} for f in rule("families", R)}
        verdict["C-Z-2"] = {f: {"token": "HALT"} for f in rule("families", R)}
        verdict["C-Z-3"] = {"token": "HALT"}
        return verdict, EXIT_HALT

    eligible = {n for n, c in per_cell.items() if c["gate_ok"] and c["class"] not in ("U", "excluded", "UNKNOWN")}

    # ---- G7 within-family / within-stratum parity (arch.name, epochs, data root)
    def group_parity(names):
        keys = rule("gates.G7_WITHIN_GROUP", R)
        vals = {}
        for n in names:
            gv = cells[n].get("G7_values") or {}
            if cells[n].get("status") == "analysed":
                vals[n] = tuple(gv.get(k) for k in keys)
        return len(set(vals.values())) <= 1

    fam_need = need_fraction("czz1.family_need", R)
    fam_pf1 = need_fraction("czz1.family_PF_precedence", R)
    fam_need2 = need_fraction("czz2.family_need", R)
    fam_pf2 = need_fraction("czz2.family_PF_precedence", R)
    div = exclusion_divisor(R)

    cz1, cz2 = {}, {}
    for fam, names in rule("families", R).items():
        n_frozen = len(names)
        elig = [n for n in names if n in eligible]
        n = len(elig)
        excl = [n_ for n_ in names if n_ not in eligible]
        base = {"n_frozen": n_frozen, "n": n, "excluded": excl, "max_exclusions": n_frozen // div}
        if not group_parity(names):
            cz1[fam] = dict(base, token="INVALID-GATE", reason="G7 within-family parity (arch.name/epochs/data root)")
            cz2[fam] = dict(cz1[fam])
            continue
        if n == 0:
            cz1[fam] = dict(base, token="INVALID-EMPTY")
            cz2[fam] = dict(base, token="INVALID-EMPTY")
            continue
        if len(excl) > n_frozen // div:
            cz1[fam] = dict(base, token="INVALID-GATE")
            cz2[fam] = dict(base, token="INVALID-GATE")
            continue
        try:
            cz1[fam] = dict(base, **_family_cz1(elig, stats, R, fam_need, fam_pf1))
        except Uncovered as e:
            cz1[fam] = dict(base, token="UNCOVERED", reason=str(e))
        try:
            cz2[fam] = dict(base, **_family_cz2(elig, stats, R, fam_need2, fam_pf2))
        except Uncovered as e:
            cz2[fam] = dict(base, token="UNCOVERED", reason=str(e))
    verdict["C-Z-1"] = cz1
    verdict["C-Z-2"] = cz2

    # ---- C-Z-3
    try:
        verdict["C-Z-3"] = _cz3(per_cell, eligible, stats, R, group_parity)
    except Uncovered as e:
        verdict["C-Z-3"] = {"token": "UNCOVERED", "reason": str(e)}

    any_excluded = any(not c["gate_ok"] or c["class"] in ("U", "excluded", "UNKNOWN") for c in per_cell.values())
    return verdict, (EXIT_EXCLUSIONS if any_excluded else EXIT_OK)


def _cell_row(stats, name, split="test", kind="prelim") -> dict:
    row = stats.get((name, split, kind))
    if row is None:
        raise Uncovered(f"{name}: missing {split}/{kind} cell_stats row")
    return row


def _check_label(row: dict, col: str, recomputed: str, name: str):
    lab = row.get(col)
    if lab not in (None, "") and lab != recomputed:
        raise JudgeAbort(f"{name}: cell_stats {col}={lab!r} but the frozen rule gives {recomputed!r}")


def _family_cz1(elig, stats, R, fam_need, fam_pf) -> dict:
    Ms, Vs, Ts = {}, {}, {}
    for name in elig:
        row = _cell_row(stats, name)
        M = recompute_M(row, R)
        _check_label(row, "M", M, name)
        Ms[name] = M
        Vs[name] = _f(row, "V")
        Ts[name] = _f(row, "T")
    n = len(elig)
    n_pf = sum(1 for m in Ms.values() if m == "PF")
    counts = {k: sum(1 for m in Ms.values() if m == k) for k in ("PF", "M+", "M-", "M0")}
    out = {"n_PF": n_pf, "counts": counts, "cell_M": Ms}
    if n_pf >= ceil_frac(n, fam_pf):
        return dict(out, token="S1-0", need_PF=ceil_frac(n, fam_pf))
    non_pf = [c for c in elig if Ms[c] != "PF"]
    n1 = len(non_pf)
    need = ceil_frac(n1, fam_need)
    if counts["M+"] >= need:
        mfam = "M+"
    elif counts["M-"] >= need:
        mfam = "M-"
    else:
        mfam = "M0"
    vv = [Vs[c] for c in non_pf]
    if any(math.isnan(v) for v in vv):
        raise Uncovered("V(c) NaN for an eligible non-PF cell")
    v_med = _median(vv)
    if v_med >= num("czz1.V_PLUS", R):
        vfam = "V+"
    elif v_med >= num("czz1.V_MINUS", R):
        vfam = "V~"
    else:
        vfam = "V-"
    tt = [Ts[c] for c in non_pf if not math.isnan(Ts[c])]
    t_med = _median(tt) if tt else num("czz1.T_all_nan_default", R)
    trivial = t_med >= num("czz1.T_TRIVIAL", R)
    out.update({"n_prime": n1, "need": need, "Mfam": mfam, "V_median": v_med, "Vfam": vfam,
                "T_median": t_med, "Tfam": "trivial" if trivial else "non-trivial"})
    if mfam == "M0":
        tok = "S1-5"
    elif mfam == "M-":
        tok = "S1-6"
        out["uncovered_mechanism"] = True  # S1-6 is pre-labelled UNCOVERED mechanism (§2.1)
    elif vfam == "V+":
        tok = "S1-2" if trivial else "S1-1"
    elif vfam == "V~":
        tok = "S1-3"
    elif vfam == "V-":
        tok = "S1-4"
    else:
        raise Uncovered(f"unmapped C-Z-1 combination {mfam}/{vfam}")
    out["token"] = tok
    return out


def _family_cz2(elig, stats, R, fam_need, fam_pf) -> dict:
    Zs = {}
    for name in elig:
        row = _cell_row(stats, name)
        M = recompute_M(row, R)
        Z = recompute_Z(row, M, R)
        _check_label(row, "Z", Z, name)
        thr = recompute_thr(row, R)
        if row.get("thr") not in (None, "") and not math.isclose(float(row["thr"]), thr, rel_tol=0, abs_tol=0):
            raise JudgeAbort(f"{name}: cell_stats thr={row['thr']} but max(PINT_FLOOR, PINT_MULT*P_int0)={thr}")
        Zs[name] = Z
    n = len(elig)
    counts = {k: sum(1 for z in Zs.values() if z == k) for k in ("PF", "stepwise", "jump", "unordered")}
    n_pf = counts["PF"]
    out = {"n_PF": n_pf, "counts": counts, "cell_Z": Zs}
    if n_pf >= ceil_frac(n, fam_pf):
        return dict(out, token="S2-0", need_PF=ceil_frac(n, fam_pf))
    n1 = n - n_pf
    need = ceil_frac(n1, fam_need)
    out.update({"n_prime": n1, "need": need})
    if counts["stepwise"] >= need:
        tok = "S2-1"
    elif counts["jump"] >= need:
        tok = "S2-2"
    else:
        tok = "S2-3"
    out["token"] = tok
    return out


def _cz3(per_cell, eligible, stats, R, group_parity) -> dict:
    strata = rule("strata", R)
    if list(rule("czz3.precedence", R)) != ["S3-0", "S3-1", "S3-2", "S3-3"]:
        raise FrozenRuleMissing("I-9: czz3.precedence is not [S3-0, S3-1, S3-2, S3-3]")
    if rule("czz3.direction", R) != "C > G":
        raise FrozenRuleMissing("I-9: czz3.direction is not 'C > G'")
    lost = {}
    for s, grp in strata.items():
        names = list(grp["G"]) + list(grp["C"])
        bad = [n for n in names if n not in eligible]
        if not group_parity(names):
            bad.append("G7-within-stratum")
        if bad:
            lost[s] = bad
    if lost:
        return {"token": "S3-0", "lost": lost}
    EL = {}
    per = {}
    for s, grp in strata.items():
        g = []
        c = []
        for n in grp["G"]:
            g.append(_f(_cell_row(stats, n, "train", "prelim"), "E_L"))
        for n in grp["C"]:
            c.append(_f(_cell_row(stats, n, "train", "prelim"), "E_L"))
        if any(math.isnan(x) for x in g + c):
            raise Uncovered(f"{s}: E_L NaN")
        per[s] = {"G": dict(zip(grp["G"], g)), "C": dict(zip(grp["C"], c)),
                  "min_C_gt_max_G": min(c) > max(g), "max_C_lt_min_G": max(c) < min(g)}
        EL[s] = (g, c)
    if all(v["min_C_gt_max_G"] for v in per.values()):
        tok = "S3-1"
    elif all(v["max_C_lt_min_G"] for v in per.values()):
        tok = "S3-2"
    else:
        tok = "S3-3"
    return {"token": tok, "strata": per, "exact_null": rule("czz3.exact_null", R)}


# ---------------------------------------------------------------------------------------------
# Prereg checks and file IO
# ---------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_frozen_block(text: str) -> str:
    m = re.search(r"<!-- FROZEN_RULES_BEGIN -->\s*```yaml\n(.*?)```\s*<!-- FROZEN_RULES_END -->", text, re.S)
    if not m:
        raise JudgeAbort("I-7: FROZEN_RULES block not found in prereg")
    return m.group(1)


def check_prereg(path: str, expected_sha: str = EXPECTED_PREREG_SHA256) -> dict:
    sha = sha256_file(path)
    if sha != expected_sha:
        raise JudgeAbort(f"I-7: prereg sha256 {sha} != frozen {expected_sha}")
    text = open(path, encoding="utf-8").read()
    parsed = yaml.safe_load(extract_frozen_block(text))
    embedded = {k: v for k, v in FROZEN_RULES.items() if k != "gates"}
    if parsed != embedded:
        raise JudgeAbort("I-9: FROZEN_RULES embedded in judge.py differ from the prereg YAML block")
    if extract_frozen_block(text) != FROZEN_RULES_YAML:
        raise JudgeAbort("I-9: embedded FROZEN_RULES_YAML text is not byte-identical to the prereg block")
    anchors = []
    for key, ent in FROZEN_GATES.items():
        if not isinstance(ent, dict):
            continue
        if key == "G7_TEXT":
            anchors += list(ent["value"])
        elif ent.get("anchor"):
            anchors.append(ent["anchor"])
    miss = [a for a in anchors if a not in text]
    if miss:
        raise JudgeAbort(f"I-9: §6 gate anchors not found verbatim in prereg: {miss}")
    return {"path": path, "sha256": sha, "yaml_equal": True, "anchors_checked": len(anchors)}


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_data(data_dir: str) -> tuple[dict, list[dict], dict | None, dict]:
    man_p = os.path.join(data_dir, "run_manifest.json")
    gates_p = os.path.join(data_dir, "gates.json")
    stats_p = os.path.join(data_dir, "cell_stats.csv")
    for p in (man_p, gates_p, stats_p):
        if not os.path.isfile(p):
            raise JudgeAbort(f"I-8: missing {p}")
    manifest = json.load(open(man_p))
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise JudgeAbort(f"I-8: run_manifest status={manifest.get('status')!r} smoke={manifest.get('smoke')!r}; "
                         "partial or smoke outputs are never judged")
    gates = json.load(open(gates_p))
    return gates["cells"], read_csv(stats_p), gates.get("G5"), manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        pre = check_prereg(a.prereg)
        cells, stats_rows, g5, manifest = load_data(a.data)
        verdict, code = judge_records(cells, stats_rows, g5)
    except FrozenRuleMissing as e:
        print(f"REFUSED (I-9): {e}", file=sys.stderr)
        return EXIT_ABORT
    except JudgeAbort as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return EXIT_ABORT
    verdict["prereg"] = pre
    verdict["judge_sha256"] = sha256_file(os.path.abspath(__file__))
    verdict["run_manifest"] = {k: manifest.get(k) for k in ("git_sha", "sha256", "torch", "device", "threads")}
    verdict["exit_code"] = code
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(verdict, f, indent=1, default=str)
    summary = {"C-Z-1": {k: v["token"] for k, v in verdict["C-Z-1"].items()},
               "C-Z-2": {k: v["token"] for k, v in verdict["C-Z-2"].items()},
               "C-Z-3": verdict["C-Z-3"]["token"]}
    print(json.dumps(summary))
    print(f"exit={code} -> {a.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
