"""EXP-017 §4.1 cell manifest (frozen) + export to EXP017_cells.csv.

MANIFEST is transcribed programmatically from the §4.1 table of the frozen prereg (41 rows; values from
the closeout CSV: col 1 run_id, col 15 final_probe_test_exact, col 21 traj_class). ``check_manifest``
asserts agreement with FROZEN_RULES families / strata / descriptive lists (judge.frozen_cells).

  uv run python analysis/z_dynamics/cells.py --out analysis/z_dynamics/EXP017_cells.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.z_dynamics import judge  # noqa: E402

FIELDS = ["idx", "run_name", "run_id", "arch", "protocol", "k", "seed", "step", "final_probe_test_exact",
          "traj_class", "class", "family_stratum"]

# (idx, run_name, run_id, arch, protocol, k, seed, step, final_probe_test_exact, traj_class, class, family/stratum)
MANIFEST = [
    (1, "pp_base_tf_z_iter_k3_s1", "i6aflih7", "trm", "D1", 3, 1, 122050, 1.0, "stable_success", "G", "F1"),
    (2, "pp_base_tf_z_iter_k3_s2", "am4x82su", "trm", "D1", 3, 2, 122050, 1.0, "instrument_flagged", "G", "F1"),
    (3, "pp_base_tf_z_iter_k3_s3", "ih6tt3y7", "trm", "D1", 3, 3, 122050, 1.0, "stable_success", "G", "F1"),
    (4, "pp_base_tf_z_iter_k5_s1", "wo6g7f0p", "trm", "D1", 5, 1, 122050, 1.0, "stable_success", "G", "F1, S_A-G"),
    (5, "pp_base_tf_z_iter_k5_s2", "yemeqgka", "trm", "D1", 5, 2, 122050, 0.02734, "never_reached", "C", "S_A-C"),
    (6, "pp_base_tf_z_iter_k5_s3", "7939mdns", "trm", "D1", 5, 3, 122050, 0.06445, "never_reached", "P", "—"),
    (7, "pp_base_tf_z_iter_k6_s1", "yflnarn2", "trm", "D1", 6, 1, 122050, 1.0, "instrument_flagged", "G", "F1, S_B-G"),
    (8, "pp_base_tf_z_iter_k6_s2", "tfzembsc", "trm", "D1", 6, 2, 122050, 0.01563, "real_collapse", "C", "S_B-C"),
    (9, "pp_base_tf_z_iter_k6_s3", "202xh8qz", "trm", "D1", 6, 3, 122050, 0.11133, "never_reached", "P", "—"),
    (10, "pp_base_tf_z_iter_k7_s1", "1okzun6g", "trm", "D1", 7, 1, 122050, 1.0, "instrument_flagged", "G", "F1"),
    (11, "pp_base_tf_z_iter_k7_s2", "v286cpf8", "trm", "D1", 7, 2, 122050, 0.9375, "stable_success", "G", "F1"),
    (12, "pp_base_tf_z_iter_k7_s3", "vywudlm2", "trm", "D1", 7, 3, 122050, 0.99219, "stable_success", "G", "F1"),
    (13, "pp_base_tf_z_iter_k10_s1", "pqcixzlt", "trm", "D1", 10, 1, 122050, 1.0, "real_collapse", "G", "F1"),
    (14, "pp_base_tf_z_iter_k10_s2", "3sud1ncr", "trm", "D1", 10, 2, 122050, 1.0, "stable_success", "G", "F1"),
    (15, "pp_base_tf_z_iter_k10_s3", "9jyglfui", "trm", "D1", 10, 3, 122050, 1.0, "real_collapse", "G", "F1"),
    (16, "pp_fig1_tf_z_iter_k3_s1", "22qnow61", "trm", "D1", 3, 1, 122050, 1.0, "real_collapse", "G", "F1"),
    (17, "pp_fig1_tf_z_iter_k5_s1", "a5wvgbt5", "trm", "D1", 5, 1, 122050, 0.02344, "real_collapse", "C", "S_A-C"),
    (18, "pp_fig1_tf_z_iter_k6_s1", "25jh8zoh", "trm", "D1", 6, 1, 122050, 1.0, "stable_success", "G", "F1, S_B-G"),
    (19, "pp_fig1_tf_z_iter_k7_s1", "oq3pf211", "trm", "D1", 7, 1, 122050, 1.0, "instrument_flagged", "G", "F1"),
    (20, "pp_fig1_tf_z_iter_k10_s1", "ot5riyf3", "trm", "D1", 10, 1, 122050, 1.0, "instrument_flagged", "G", "F1"),
    (21, "pp_fig1_tf_noz_iter_k3_s1", "euk0ausa", "trm_singlez", "D1", 3, 1, 122050, 1.0, "stable_success", "G", "F2"),
    (22, "pp_fig1_tf_noz_iter_k5_s1", "h1zuu3v4", "trm_singlez", "D1", 5, 1, 122050, 0.00586, "never_reached", "C", "D-3/D-4"),
    (23, "pp_fig1_tf_noz_iter_k6_s1", "zvyjaj5z", "trm_singlez", "D1", 6, 1, 122050, 0.02344, "never_reached", "C", "D-4"),
    (24, "pp_fig1_tf_noz_iter_k7_s1", "t2ur05ay", "trm_singlez", "D1", 7, 1, 122050, 0.01953, "never_reached", "C", "D-3/D-4"),
    (25, "pp_fig1_tf_noz_iter_k10_s1", "2qcusl5z", "trm_singlez", "D1", 10, 1, 122050, 1.0, "stable_success", "G", "F2"),
    (26, "pp_fig1_tf_noz_iter_k10_s2", "xbrcoanm", "trm_singlez", "D1", 10, 2, 122050, 1.0, "stable_success", "G", "F2"),
    (27, "pp_fig1_tf_noz_iter_k10_s3", "e0nuqbwi", "trm_singlez", "D1", 10, 3, 122050, 0.81836, "stable_success", "P", "—"),
    (28, "pp_seedext_tf_noz_iter_k3_s2", "8y8jyodj", "trm_singlez", "D0", 3, 2, 244100, 1.0, "stable_success", "G", "F3"),
    (29, "pp_seedext_tf_noz_iter_k3_s3", "untj40n5", "trm_singlez", "D0", 3, 3, 244100, 1.0, "stable_success", "G", "F3"),
    (30, "pp_seedext_tf_noz_iter_k4_s2", "jixldlb8", "trm_singlez", "D0", 4, 2, 244100, 1.0, "stable_success", "G", "F3"),
    (31, "pp_seedext_tf_noz_iter_k4_s3", "h7yeuunf", "trm_singlez", "D0", 4, 3, 244100, 1.0, "instrument_flagged", "G", "F3"),
    (32, "pp_seedext_tf_noz_iter_k5_s2", "wb7z2era", "trm_singlez", "D0", 5, 2, 244100, 0.01953, "never_reached", "C", "D-3/D-4"),
    (33, "pp_seedext_tf_noz_iter_k5_s3", "032mv2y1", "trm_singlez", "D0", 5, 3, 244100, 0.01758, "never_reached", "C", "D-3/D-4"),
    (34, "pp_seedext_tf_noz_iter_k6_s2", "jbmrvylv", "trm_singlez", "D0", 6, 2, 244100, 0.02344, "never_reached", "C", "D-4"),
    (35, "pp_seedext_tf_noz_iter_k6_s3", "i2bc5dop", "trm_singlez", "D0", 6, 3, 244100, 0.01367, "never_reached", "C", "D-4"),
    (36, "pp_seedext_tf_noz_iter_k7_s2", "dycv7aw0", "trm_singlez", "D0", 7, 2, 244100, 1.0, "stable_success", "G", "F3, S_C-G"),
    (37, "pp_seedext_tf_noz_iter_k7_s3", "q0law9h8", "trm_singlez", "D0", 7, 3, 244100, 0.02344, "never_reached", "C", "S_C-C"),
    (38, "pp_seedext_tf_noz_iter_k8_s2", "76wbuqwh", "trm_singlez", "D0", 8, 2, 244100, 1.0, "instrument_flagged", "G", "F3"),
    (39, "pp_seedext_tf_noz_iter_k8_s3", "4fq2o8vd", "trm_singlez", "D0", 8, 3, 244100, 0.86719, "stable_success", "P", "—"),
    (40, "pp_seedext_tf_noz_iter_k10_s2", "s7j8m6od", "trm_singlez", "D0", 10, 2, 244100, 1.0, "stable_success", "G", "F3"),
    (41, "pp_seedext_tf_noz_iter_k10_s3", "4wafgti1", "trm_singlez", "D0", 10, 3, 244100, 1.0, "stable_success", "G", "F3"),
]

DATA_ROOT = {"D1": "data/power_permutation/d1", "D0": "data/sigma_k_10"}


def unit_name(run_name: str, step: int) -> str:
    """Analysis-unit schema EXP017__<source_run_name>__step_<N> (§4)."""
    return f"EXP017__{run_name}__step_{step}"


def rows() -> list[dict]:
    return [dict(zip(FIELDS, r)) for r in MANIFEST]


def check_manifest() -> None:
    fz = judge.frozen_cells()
    got = {r["run_name"]: r["class"] for r in rows()}
    if len(got) != 41 or got != fz:
        raise AssertionError(f"manifest disagrees with FROZEN_RULES: "
                             f"{sorted(set(got.items()) ^ set(fz.items()))}")
    term = judge.rule("gates.TERMINAL_STEP")
    for r in rows():
        if r["step"] != term[r["protocol"]]:
            raise AssertionError(f"{r['run_name']}: step {r['step']} is not terminal for {r['protocol']}")
        if r["arch"] != judge.arch_of(r["run_name"]):
            raise AssertionError(f"{r['run_name']}: arch mismatch")


def write_csv(path: str) -> None:
    check_manifest()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows():
            w.writerow(r)


def read_csv(path: str) -> list[dict]:
    """Read EXP017_cells.csv and assert it equals MANIFEST (the CSV is an export, never an input of record)."""
    with open(path, newline="") as f:
        got = list(csv.DictReader(f))
    ref = [{k: str(v) for k, v in r.items()} for r in rows()]
    if got != ref:
        raise AssertionError(f"{path} differs from cells.MANIFEST")
    return rows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "EXP017_cells.csv"))
    a = ap.parse_args()
    write_csv(a.out)
    print(f"wrote {len(MANIFEST)} cells -> {a.out}")
