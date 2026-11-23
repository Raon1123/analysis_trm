"""EXP-027 dataset builder: StateBench-style register traces -> pretrain.py npy format.

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/build_dataset.py \
      --tier G --N 16 --out data/statebench_e027/G_N16 --date 2026-11-11

Encoding (EXP-027 v2 prereg s1.3, s3.3; vocab 261):
  0 PAD | 1..5 REG_0..REG_4 | 6 TMP | 7..133 VAL_0..VAL_126 | 134..260 INC_0..INC_126
  One assignment = 2 tokens [REG_target, ARG]; ARG = VAL_c (const c), REG_s (copy from s),
  INC_d (increment by d).  Input = 2(m+1+N) op tokens + m answer-slot tokens REG_0..REG_{m-1}.
  Labels: 0 (= ignore_label_id) on op positions; 7+final_value on the last m positions only
  (TMP is excluded).  seq_len = 2(m+1+N) + m = 17 + 2N for m = 5.

Tiers: G = general finite (``periodic`` source ops), S = soluble (``solvable``),
A = aperiodic.  The three tiers derive from the same sources; for equal (N, seeds, m, q, c)
row i of G_N, S_N, A_N is the same source.  Dedup/disjointness is decided on the token
sequences of ALL three tiers jointly, so alignment holds and each tier's train/test input
sets are disjoint.

Determinism: all randomness is random.Random seeded from (seed, N, m, q, c); the date comes
from --date (never the wall clock); the manifest holds no absolute paths.  Two builds with
equal arguments are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sb_generator as G  # noqa: E402

TIER_TO_FAMILY = {"G": "periodic", "S": "solvable", "A": "aperiodic"}
PAD, REG0, TMP_TOKEN, VAL0, INC0 = 0, 1, 6, 7, 134
VOCAB = 261
SEED_TRAIN_DEFAULT = 270001
SEED_TEST_DEFAULT = 270002
BUILDER_VERSION = "1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(trace: G.Trace, m: int) -> tuple[list[int], list[int]]:
    """Trace -> (inputs, labels), each of length 2*len(operations) + m."""
    inp: list[int] = []
    for kind, t, a in trace.operations:
        inp.append(REG0 + t)  # REG_t; t == m gives token m+1 (= 6 for m = 5, the TMP token)
        if kind == "const":
            inp.append(VAL0 + a)
        elif kind == "copy":
            inp.append(REG0 + a)
        elif kind == "inc":
            inp.append(INC0 + a)
        else:
            raise ValueError(kind)
    n_op_tokens = len(inp)
    inp.extend(REG0 + r for r in range(m))
    lab = [0] * n_op_tokens + [VAL0 + trace.final_state[r] for r in range(m)]
    return inp, lab


def n_swaps(source: G.Trace) -> int:
    # Operations after the m+1 initial constants are scratch loads (copy TMP<-reg), constant
    # resets, and 3-copy swaps; with c = 0 there are no resets, so (copies - scratch) / 3.
    ops = source.operations[source.num_variables + 1:]
    return sum(1 for o in ops if o[0] == "copy") // 3


def build_split(*, tier, N, m, q, c_min, c_max, rows, seed, forbid):
    """Draw sources until ``rows`` unique rows are collected.

    Returns (inputs, labels, swaps_hist, stats, keyset).  A source is dropped when the
    concatenated input tokens of any tier (G|S|A) are already in ``seen`` (own split) or in
    ``forbid`` (the other split).  ``stats`` counts drops by reason and by own-tier collision.
    """
    rng = random.Random(f"{seed}:{N}:{m}:{q}:{c_min}:{c_max}")
    fam = TIER_TO_FAMILY[tier]
    # per-tier sets of input-token tuples; a collision in any single tier drops the source
    seen_tier = {k: set() for k in ("periodic", "solvable", "aperiodic")}
    ins, labs = [], []
    swaps = Counter()
    stats = {"drawn": 0, "dropped_dup_within_split": 0, "dropped_dup_vs_other_split": 0,
             "dropped_own_tier_dup_within_split": 0, "dropped_own_tier_dup_vs_other_split": 0}
    other = forbid or {k: set() for k in seen_tier}
    while len(ins) < rows:
        src = G.periodic_source(m=m, q=q, n_extra_min=N, n_extra_max=N,
                                ap_const_min=c_min, ap_const_max=c_max, rng=rng)
        fams = G.derive(src)
        enc = {k: encode(fams[k], m) for k in seen_tier}
        keys = {k: tuple(enc[k][0]) for k in seen_tier}
        stats["drawn"] += 1
        in_other = [k for k in seen_tier if keys[k] in other[k]]
        in_self = [k for k in seen_tier if keys[k] in seen_tier[k]]
        if in_other:
            stats["dropped_dup_vs_other_split"] += 1
            stats["dropped_own_tier_dup_vs_other_split"] += int(fam in in_other)
            continue
        if in_self:
            stats["dropped_dup_within_split"] += 1
            stats["dropped_own_tier_dup_within_split"] += int(fam in in_self)
            continue
        for k in seen_tier:
            seen_tier[k].add(keys[k])
        ins.append(enc[fam][0])
        labs.append(enc[fam][1])
        swaps[n_swaps(src)] += 1
    return (np.asarray(ins, dtype=np.int32), np.asarray(labs, dtype=np.int32),
            dict(sorted(swaps.items())), stats, seen_tier)


def write_split(d: Path, inputs: np.ndarray, labels: np.ndarray, vocab_size: int):
    d.mkdir(parents=True, exist_ok=True)
    rows, seq_len = inputs.shape
    meta = {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": vocab_size,
            "seq_len": int(seq_len), "num_puzzle_identifiers": 1, "total_groups": int(rows),
            "mean_puzzle_examples": 1.0, "total_puzzles": int(rows), "sets": ["all"]}
    with open(d / "dataset.json", "w") as f:
        json.dump(meta, f)
    np.save(d / "all__inputs.npy", inputs)
    np.save(d / "all__labels.npy", labels)
    np.save(d / "all__puzzle_identifiers.npy", np.zeros(rows, dtype=np.int32))
    np.save(d / "all__puzzle_indices.npy", np.arange(rows + 1, dtype=np.int32))
    np.save(d / "all__group_indices.npy", np.arange(rows + 1, dtype=np.int32))


def build(*, tier, N, out, seed_train, seed_test, m, q, train, test, c_min, c_max, date) -> dict:
    out = Path(out)
    if m != 5 or q != 127:
        # The 261-token layout (5 registers + TMP, 127 values/increments) is fixed by v2 s3.3.
        raise ValueError("encoding is fixed to m=5, q=127 (vocab 261)")
    kw = dict(tier=tier, N=N, m=m, q=q, c_min=c_min, c_max=c_max)
    tr_i, tr_l, tr_sw, tr_st, tr_seen = build_split(rows=train, seed=seed_train, forbid=None, **kw)
    te_i, te_l, te_sw, te_st, _ = build_split(rows=test, seed=seed_test, forbid=tr_seen, **kw)
    assert tr_i.shape[1] == te_i.shape[1] == 2 * (m + 1 + N) + m
    assert int(tr_i.max()) < VOCAB and int(te_i.max()) < VOCAB
    write_split(out / "train", tr_i, tr_l, VOCAB)
    write_split(out / "test", te_i, te_l, VOCAB)
    files = {}
    for split in ("train", "test"):
        for p in sorted((out / split).iterdir()):
            files[f"{split}/{p.name}"] = sha256_file(p)
    manifest = {
        "builder": "tools/statebench_tiers/build_dataset.py", "builder_version": BUILDER_VERSION,
        "date": date,
        "args": {"tier": tier, "N": N, "m": m, "q": q, "c_min": c_min, "c_max": c_max,
                 "train": train, "test": test, "seed_train": seed_train, "seed_test": seed_test},
        "generator_sha256": sha256_file(HERE / "sb_generator.py"),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "seq_len": int(tr_i.shape[1]), "vocab_size": VOCAB,
        "rows": {"train": int(tr_i.shape[0]), "test": int(te_i.shape[0])},
        "dedupe": {"rule": "drop a source if the input token sequence of ANY tier (G, S, A) "
                           "already occurs in the same split or in the train split (for test)",
                   "train": tr_st, "test": te_st},
        "swap_count_distribution": {"train": tr_sw, "test": te_sw},
        "swap_count_note": "(copy ops after the m+1 initial consts) // 3 on the G source; "
                           "identical across tiers for the same source",
        "files_sha256": files,
    }
    with open(out / "build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", required=True, choices=sorted(TIER_TO_FAMILY))
    ap.add_argument("--N", type=int, required=True, help="extra operations after the m+1 initial consts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-train", type=int, default=SEED_TRAIN_DEFAULT)
    ap.add_argument("--seed-test", type=int, default=SEED_TEST_DEFAULT)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--q", type=int, default=127)
    ap.add_argument("--train", type=int, default=100000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--c", type=int, default=0, help="aperiodic_constants min (prereg: c = 0)")
    ap.add_argument("--c-max", type=int, default=None, help="aperiodic_constants max (default = --c)")
    ap.add_argument("--date", required=True, help="date string recorded in the manifest (no wall clock)")
    a = ap.parse_args(argv)
    man = build(tier=a.tier, N=a.N, out=a.out, seed_train=a.seed_train, seed_test=a.seed_test,
                m=a.m, q=a.q, train=a.train, test=a.test, c_min=a.c,
                c_max=a.c if a.c_max is None else a.c_max, date=a.date)
    print(json.dumps({"tier": a.tier, "N": a.N, "seq_len": man["seq_len"], "vocab": man["vocab_size"],
                      "rows": man["rows"], "dedupe": man["dedupe"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
