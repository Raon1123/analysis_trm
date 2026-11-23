"""EXP-027 P3 audit of one built cell directory (read-only).

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/exp027_data_audit.py \
      data/statebench_e027/G_N16 --json lab/.../p3_audit_G_N16.json

Exit codes: 0 all checks pass; 1 at least one check failed; 2 unreadable / malformed cell.

The label recomputation uses an interpreter written HERE from the token layout of
EXP-027 v2 prereg s3.3 (it imports neither sb_generator nor build_dataset):
  0 PAD | 1..5 REG_0..REG_4 | 6 TMP | 7..133 VAL_0..VAL_126 | 134..260 INC_0..INC_126
  one assignment = [REG_target, ARG]; ARG in VAL = constant, in REG/TMP = copy, in INC = add.
  Input = 2*(m+1+N) op tokens, then the m answer slots REG_0..REG_{m-1};
  label = 0 on op positions, 7 + final value on the m answer positions.
Checks: shapes/dtypes/vocab/ignore positions, label recomputation by execution, in-split
uniqueness, train-test disjointness, tier token ranges, initial-assignment structure, sibling
row alignment (A/S/G of the same N, when present), manifest sha256; plus tier statistics
(G: swap counts, overlap, distinct final permutations; S: final-value distribution).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

M, Q = 5, 127
R = M + 1
VOCAB = 261
PAD, REG0, TMP, VAL0, INC0 = 0, 1, 6, 7, 134
NPY = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")
TIERS = ("A", "S", "G")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def execute(inp: np.ndarray):
    """Run all rows at once.  Returns (final (rows, R) int64, bad_read (rows,) bool).

    ``bad_read`` marks rows that copy/increment from a register never written.
    """
    rows, L = inp.shape
    n_op = L - M
    assert n_op % 2 == 0
    ar = np.arange(rows)
    state = np.full((rows, R), -1, dtype=np.int64)
    bad = np.zeros(rows, dtype=bool)
    for j in range(0, n_op, 2):
        t = inp[:, j].astype(np.int64) - REG0
        a = inp[:, j + 1].astype(np.int64)
        t = np.clip(t, 0, R - 1)  # range violations are reported by the structure checks
        is_const = (a >= VAL0) & (a < INC0)
        is_copy = (a >= REG0) & (a <= TMP)
        is_inc = a >= INC0
        src = np.clip(a - REG0, 0, R - 1)
        srcval = state[ar, src]
        cur = state[ar, t]
        bad |= (is_copy & (srcval < 0)) | (is_inc & (cur < 0))
        new = np.where(is_const, a - VAL0,
                       np.where(is_copy, srcval, np.where(is_inc, (cur + (a - INC0)) % Q, cur)))
        state[ar, t] = new
    return state, bad


def parse_swaps(inp_row: np.ndarray, n_init: int = R):
    """G tier: ops after the initial assignments -> (swaps [(a,b)], n_scratch, n_other).

    A swap is the triple  TMP<-a ; a<-b ; b<-TMP  (copies); a lone copy TMP<-r is a scratch load.
    """
    ops = [(int(inp_row[j]) - REG0, int(inp_row[j + 1])) for j in range(0, len(inp_row) - M, 2)][n_init:]
    swaps, scratch, other, i = [], 0, 0, 0
    while i < len(ops):
        t, a = ops[i]
        if t == M and REG0 <= a <= TMP - 1 and i + 2 < len(ops):
            (t2, a2), (t3, a3) = ops[i + 1], ops[i + 2]
            if t2 == a - REG0 and REG0 <= a2 <= TMP - 1 and t3 == a2 - REG0 and a3 == TMP:
                swaps.append((a - REG0, a2 - REG0))
                i += 3
                continue
        if t == M and REG0 <= a <= TMP - 1:
            scratch += 1
        else:
            other += 1
        i += 1
    return swaps, scratch, other


def net_perm(swaps):
    perm = list(range(M))  # perm[i] = original register whose content now sits in i
    for a, b in swaps:
        perm[a], perm[b] = perm[b], perm[a]
    return tuple(perm)


def load_split(d: Path, split: str):
    out = {}
    for n in NPY:
        out[n] = np.load(d / split / f"all__{n}.npy")
    out["meta"] = json.loads((d / split / "dataset.json").read_text())
    return out


def row_keys(a: np.ndarray):
    return {r.tobytes() for r in np.ascontiguousarray(a)}


def check_split(tier, split, S, rows_expected):
    """Structure + execution checks of one split.  Returns (checks dict name->bool, info)."""
    ck, info = {}, {}
    inp, lab = S["inputs"], S["labels"]
    meta = S["meta"]
    seq = inp.shape[1] if inp.ndim == 2 else -1
    ck["dtypes_int32"] = all(S[n].dtype == np.int32 for n in NPY)
    ck["shapes"] = (inp.ndim == 2 and lab.shape == inp.shape and seq >= 2 * R + M and (seq - M) % 2 == 0
                    and S["puzzle_identifiers"].shape == (inp.shape[0],)
                    and S["puzzle_indices"].shape == (inp.shape[0] + 1,)
                    and S["group_indices"].shape == (inp.shape[0] + 1,))
    if not ck["shapes"]:
        return ck, info
    rows = inp.shape[0]
    n_op = seq - M
    ck["row_count"] = rows_expected is None or rows == rows_expected
    ck["metadata"] = (meta == {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0,
                               "vocab_size": VOCAB, "seq_len": seq, "num_puzzle_identifiers": 1,
                               "total_groups": rows, "mean_puzzle_examples": 1.0,
                               "total_puzzles": rows, "sets": ["all"]})
    ck["index_arrays"] = (np.array_equal(S["puzzle_indices"], np.arange(rows + 1))
                          and np.array_equal(S["group_indices"], np.arange(rows + 1))
                          and not S["puzzle_identifiers"].any())
    ck["input_vocab_range"] = bool(inp.min() >= 1 and inp.max() < VOCAB)  # no PAD in inputs
    ck["ignore_positions"] = bool((lab[:, :n_op] == 0).all())
    ck["answer_slots"] = bool((inp[:, n_op:] == np.arange(REG0, REG0 + M)).all())
    ck["label_range"] = bool((lab[:, n_op:] >= VAL0).all() and (lab[:, n_op:] < INC0).all())
    tg, ag = inp[:, 0:n_op:2], inp[:, 1:n_op:2]
    ck["targets_are_registers"] = bool((tg >= REG0).all() and (tg <= TMP).all())
    # first R assignments: every register (incl. TMP) initialised exactly once, with a constant
    ck["initial_assignments"] = bool(
        (np.sort(tg[:, :R], axis=1) == np.arange(REG0, TMP + 1)).all()
        and ((ag[:, :R] >= VAL0) & (ag[:, :R] < INC0)).all())
    rest = ag[:, R:]
    if tier == "G":
        ck["tier_token_range"] = bool((rest < INC0).all())  # const or copy, never INC
        ck["tier_has_no_inc"] = ck["tier_token_range"]
    elif tier == "S":
        ck["tier_token_range"] = bool(((rest >= INC0) | ((rest >= VAL0) & (rest < INC0))).all())  # VAL or INC
        ck["tier_has_no_copy"] = bool(not ((rest >= REG0) & (rest <= TMP)).any())
    else:
        ck["tier_token_range"] = bool(((rest >= VAL0) & (rest < INC0)).all())  # constants only
    final, bad = execute(inp)
    ck["no_read_before_write"] = bool(not bad.any())
    ck["labels_match_execution"] = bool((lab[:, n_op:] == final[:, :M] + VAL0).all())
    info["label_mismatch_rows"] = int((lab[:, n_op:] != final[:, :M] + VAL0).any(axis=1).sum())
    keys = row_keys(inp)
    ck["rows_unique"] = len(keys) == rows
    info["duplicate_rows"] = rows - len(keys)
    info["_final"] = final
    return ck, info


def tier_stats(tier, tr_inp, tr_final):
    st = {}
    if tier == "G":
        sw_hist, perms, overlap, nontriv, other_tot, scratch_tot = Counter(), Counter(), 0, 0, 0, 0
        per_row_perm = []
        for r in tr_inp:
            sw, sc, ot = parse_swaps(r)
            sw_hist[len(sw)] += 1
            p = net_perm(sw)
            perms[p] += 1
            overlap += int(any(set(sw[i]) & set(sw[j]) for i in range(len(sw)) for j in range(i + 1, len(sw))))
            nontriv += int(p != tuple(range(M)))
            other_tot += ot
            scratch_tot += sc
        n = len(tr_inp)
        st = {"swap_count_hist": dict(sorted(sw_hist.items())),
              "frac_rows_with_overlapping_swaps": overlap / n,
              "frac_rows_net_permutation_nontrivial": nontriv / n,
              "distinct_final_permutations": len(perms),
              "modal_permutation": list(perms.most_common(1)[0][0]),
              "modal_permutation_freq": perms.most_common(1)[0][1] / n,
              "total_scratch_loads": scratch_tot, "total_non_swap_non_scratch_ops": other_tot,
              "note": "overlap = some pair of swaps in the trace shares a register (then they do not commute)"}
    elif tier == "S":
        vals = tr_final[:, :M]
        per_reg = []
        for k in range(M):
            c = np.bincount(vals[:, k], minlength=Q) / len(vals)
            ent = float(-(c[c > 0] * np.log2(c[c > 0])).sum())
            per_reg.append({"reg": k, "distinct": int((c > 0).sum()), "max_freq": float(c.max()),
                            "entropy_bits": ent})
        allc = np.bincount(vals.ravel(), minlength=Q) / vals.size
        st = {"per_register": per_reg, "pooled_distinct_values": int((allc > 0).sum()),
              "pooled_max_freq": float(allc.max()), "max_entropy_bits": math.log2(Q),
              "distinct_final_tuples": len({r.tobytes() for r in np.ascontiguousarray(vals)})}
    else:
        vals = tr_final[:, :M]
        st = {"distinct_final_tuples": len({r.tobytes() for r in np.ascontiguousarray(vals)}),
              "pooled_max_freq": float((np.bincount(vals.ravel(), minlength=Q) / vals.size).max())}
    return st


def audit(cell: Path, siblings: bool = True) -> dict:
    cell = Path(cell)
    name = cell.name
    tier, _, nstr = name.partition("_N")
    if tier not in TIERS or not nstr.isdigit():
        raise ValueError(f"cell name must look like G_N16, got {name!r}")
    N = int(nstr)
    man = None
    if (cell / "build_manifest.json").exists():
        man = json.loads((cell / "build_manifest.json").read_text())
    exp_rows = {s: (man["rows"][s] if man else None) for s in ("train", "test")}
    splits = {s: load_split(cell, s) for s in ("train", "test")}
    checks, info = {}, {}
    finals = {}
    for s in ("train", "test"):
        ck, inf = check_split(tier, s, splits[s], exp_rows[s])
        for k, v in ck.items():
            checks[f"{s}.{k}"] = bool(v)
        finals[s] = inf.pop("_final", None)
        info[s] = inf
        info[s]["rows"] = int(splits[s]["inputs"].shape[0])
        info[s]["seq_len"] = int(splits[s]["inputs"].shape[1]) if splits[s]["inputs"].ndim == 2 else None
    checks["seq_len_matches_N"] = all(info[s]["seq_len"] == 2 * (R + N) + M for s in ("train", "test"))
    checks["train_test_same_seq_len"] = info["train"]["seq_len"] == info["test"]["seq_len"]
    if checks["train_test_same_seq_len"] and splits["train"]["inputs"].ndim == 2:
        tr = row_keys(splits["train"]["inputs"])
        te = row_keys(splits["test"]["inputs"])
        info["train_test_overlap_rows"] = len(tr & te)
        checks["train_test_disjoint"] = len(tr & te) == 0
    else:
        checks["train_test_disjoint"] = False
    if man:
        bad = [rel for rel, sha in man["files_sha256"].items()
               if not (cell / rel).exists() or sha256_file(cell / rel) != sha]
        info["manifest_sha_mismatch"] = bad
        checks["manifest_sha256"] = not bad
    if siblings:
        al = {}
        for t2 in TIERS:
            sib = cell.parent / f"{t2}_N{N}"
            if t2 == tier or not (sib / "build_manifest.json").exists():
                continue
            ok = True
            for s in ("train", "test"):
                a, b = splits[s], load_split(sib, s)
                n_op = a["inputs"].shape[1] - M
                ok &= (a["inputs"].shape == b["inputs"].shape
                       and np.array_equal(a["inputs"][:, 0:n_op:2], b["inputs"][:, 0:n_op:2])
                       and np.array_equal(a["labels"], b["labels"]))
            al[sib.name] = bool(ok)
            checks[f"sibling_alignment.{sib.name}"] = bool(ok)
        info["siblings_checked"] = sorted(al)
    stats = {}
    if checks.get("train.labels_match_execution") is not None and finals["train"] is not None:
        stats = tier_stats(tier, splits["train"]["inputs"], finals["train"])
        if tier == "G":
            # every post-init op must be a swap triple or scratch load when the cell has no resets
            stats["note_c0_parse_complete"] = stats["total_non_swap_non_scratch_ops"] == 0
            ts = tier_stats("G", splits["test"]["inputs"], finals["test"])
            stats["test_swap_count_hist"] = ts["swap_count_hist"]
            stats["test_distinct_final_permutations"] = ts["distinct_final_permutations"]
            stats["test_frac_rows_with_overlapping_swaps"] = ts["frac_rows_with_overlapping_swaps"]
    return {"cell": name, "tier": tier, "N": N, "passed": all(checks.values()),
            "failed_checks": sorted(k for k, v in checks.items() if not v),
            "checks": checks, "info": info, "tier_stats": stats}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cell", help="cell directory, e.g. data/statebench_e027/G_N16")
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--no-siblings", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = audit(Path(a.cell), siblings=not a.no_siblings)
    except (OSError, ValueError, KeyError) as e:
        print(f"AUDIT ERROR: {e}", file=sys.stderr)
        return 2
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"cell": rep["cell"], "passed": rep["passed"], "failed_checks": rep["failed_checks"]}))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
