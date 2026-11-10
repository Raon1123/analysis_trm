"""Read-only integrity / shortcut / overlap audit of sigma^k Format-D datasets.

Adapted from the methodology of allenai/state-bench (manifest overlap report,
``disjoint_from``, ``audit_periodic_shortcuts.py``); see
lab/28_literature/11_lab-literature/2026-11-05_state-bench-applicability.md.

Layout audited: ``<root>/<k>/{train,test}/all__{inputs,labels}.npy`` where
positions 0..n-1 hold sigma (1-indexed) and sigma^k (1-indexed), PAD = 0.

Per k:
  * within-split duplicates and train ∩ test (must be 0)
  * labels == sigma^k recomputed from inputs (must hold)
  * min / max ord(sigma) per split (min must be > k when the order filter is on)
  * trivial-predictor floors on test: identity and copy-sigma, token acc + exact
  * cycle-type histogram of test sigmas
Across k:
  * |train(k_a) ∩ test(k_b)| for every ordered pair; |test ∩ test| and
    |train ∩ train| for every unordered pair

Exit code: 0 clean; 1 integrity failure (within-k overlap, duplicates, label
mismatch, order-filter violation); 2 cross-k overlap present AND
``--fail-on-cross-k`` was given; 3 usage / missing data.

Usage:
  uv run python tools/sigmak_data_audit.py data/sigma_k_10 [--ks 3 7 10]
      [--no-order-filter] [--fail-on-cross-k] [--json OUT] [-v]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_split(root: Path, k: int, split: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (sigma 0-indexed [N,n], labels 0-indexed [N,n], n)."""
    d = root / str(k) / split
    inp = np.load(d / "all__inputs.npy")
    lbl = np.load(d / "all__labels.npy")
    # n = number of non-PAD positions; Format D pads the tail with 0.
    nonpad = (inp != 0).sum(axis=1)
    if nonpad.min() != nonpad.max():
        raise ValueError(f"{d}: rows have different n ({nonpad.min()}..{nonpad.max()})")
    n = int(nonpad[0])
    return inp[:, :n].astype(np.int64) - 1, lbl[:, :n].astype(np.int64) - 1, n


def cycle_type(sigma: np.ndarray) -> tuple[int, ...]:
    n = len(sigma)
    seen = np.zeros(n, dtype=bool)
    lens = []
    for i in range(n):
        if not seen[i]:
            c, j = 0, i
            while not seen[j]:
                seen[j] = True
                j = int(sigma[j])
                c += 1
            lens.append(c)
    return tuple(sorted(lens, reverse=True))


def order_of(ct: tuple[int, ...]) -> int:
    o = 1
    for c in ct:
        o = o * c // math.gcd(o, c)
    return o


def power(sigmas: np.ndarray, k: int) -> np.ndarray:
    n = sigmas.shape[1]
    res = np.broadcast_to(np.arange(n), sigmas.shape).copy()
    for _ in range(k):
        res = np.take_along_axis(sigmas, res, axis=1)
    return res


def keyset(sigmas: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in np.ascontiguousarray(sigmas, dtype=np.int64)}


def audit_k(root: Path, k: int, order_filter: bool) -> tuple[dict, list[str], dict[str, set[bytes]]]:
    errors: list[str] = []
    rec: dict = {"k": k}
    sets = {}
    for split in ("train", "test"):
        sig, lbl, n = load_split(root, k, split)
        rec["n"] = n
        keys = keyset(sig)
        sets[split] = keys
        dup = len(sig) - len(keys)
        if not np.array_equal(np.sort(sig, axis=1), np.broadcast_to(np.arange(n), sig.shape)):
            errors.append(f"k={k} {split}: inputs are not all permutations")
        mismatch = int((power(sig, k) != lbl).any(axis=1).sum())
        cts = [cycle_type(s) for s in sig]
        orders = [order_of(ct) for ct in cts]
        rec[split] = {
            "size": len(sig),
            "duplicates": dup,
            "label_mismatch_rows": mismatch,
            "min_order": min(orders),
            "max_order": max(orders),
        }
        if dup:
            errors.append(f"k={k} {split}: {dup} duplicate sigmas")
        if mismatch:
            errors.append(f"k={k} {split}: {mismatch} rows with labels != sigma^k")
        if order_filter and min(orders) <= k:
            errors.append(f"k={k} {split}: min ord(sigma)={min(orders)} <= k (order filter violated)")
        if split == "test":
            ident = np.broadcast_to(np.arange(n), sig.shape)
            rec["test"]["floors"] = {
                "identity_token_acc": float((lbl == ident).mean()),
                "identity_exact": float((lbl == ident).all(axis=1).mean()),
                "copy_sigma_token_acc": float((lbl == sig).mean()),
                "copy_sigma_exact": float((lbl == sig).all(axis=1).mean()),
            }
            hist = Counter("+".join(map(str, ct)) for ct in cts)
            rec["test"]["cycle_type_hist"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    inter = len(sets["train"] & sets["test"])
    rec["train_test_overlap"] = inter
    if inter:
        errors.append(f"k={k}: train ∩ test = {inter}")
    return rec, errors, sets


def audit(root: Path, ks: list[int], order_filter: bool = True) -> dict:
    per_k, errors, all_sets = [], [], {}
    for k in ks:
        rec, errs, sets = audit_k(root, k, order_filter)
        per_k.append(rec)
        errors.extend(errs)
        all_sets[k] = sets
    cross = []
    for ka in ks:
        for kb in ks:
            if ka == kb:
                continue
            cross.append({
                "train_k": ka,
                "test_k": kb,
                "train_test_overlap": len(all_sets[ka]["train"] & all_sets[kb]["test"]),
                "test_size": len(all_sets[kb]["test"]),
            })
    test_test, train_train = [], []
    for i, ka in enumerate(ks):
        for kb in ks[i + 1:]:
            test_test.append({
                "k_a": ka, "k_b": kb,
                "test_test_overlap": len(all_sets[ka]["test"] & all_sets[kb]["test"]),
            })
            train_train.append({
                "k_a": ka, "k_b": kb,
                "train_train_overlap": len(all_sets[ka]["train"] & all_sets[kb]["train"]),
            })
    return {
        "root": str(root),
        "ks": ks,
        "order_filter": order_filter,
        "per_k": per_k,
        "cross_k": cross,
        "cross_k_test_test": test_test,
        "cross_k_train_train": train_train,
        "cross_k_overlap_total": sum(c["train_test_overlap"] for c in cross)
        + sum(c["test_test_overlap"] for c in test_test)
        + sum(c["train_train_overlap"] for c in train_train),
        "integrity_errors": errors,
    }


def discover_ks(root: Path) -> list[int]:
    return sorted(
        int(p.name) for p in root.iterdir()
        if p.is_dir() and p.name.isdigit()
        and (p / "train" / "all__inputs.npy").exists() and (p / "test" / "all__inputs.npy").exists()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--no-order-filter", action="store_true",
                    help="dataset was built with order_filter=False; skip the min-order check")
    ap.add_argument("--fail-on-cross-k", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every nonzero cross-k pair")
    a = ap.parse_args(argv)

    if not a.root.is_dir():
        print(f"error: {a.root} is not a directory", file=sys.stderr)
        return 3
    ks = a.ks or discover_ks(a.root)
    if not ks:
        print(f"error: no <k>/{{train,test}} dirs under {a.root}", file=sys.stderr)
        return 3
    try:
        report = audit(a.root, ks, order_filter=not a.no_order_filter)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    print(f"root={report['root']} ks={ks}")
    print(f"{'k':>3} {'train':>6} {'test':>5} {'tr∩te':>6} {'minord':>6} "
          f"{'id_tok':>7} {'cp_tok':>7} {'id_ex':>6} {'cp_ex':>6}")
    for r in report["per_k"]:
        f = r["test"]["floors"]
        print(f"{r['k']:>3} {r['train']['size']:>6} {r['test']['size']:>5} {r['train_test_overlap']:>6} "
              f"{min(r['train']['min_order'], r['test']['min_order']):>6} "
              f"{f['identity_token_acc']:>7.3f} {f['copy_sigma_token_acc']:>7.3f} "
              f"{f['identity_exact']:>6.3f} {f['copy_sigma_exact']:>6.3f}")
    nonzero = [c for c in report["cross_k"] if c["train_test_overlap"]]
    worst = max(nonzero, key=lambda c: c["train_test_overlap"] / c["test_size"], default=None)
    print(f"cross-k train∩test nonzero pairs: {len(nonzero)}/{len(report['cross_k'])}"
          + (f" (max: train k={worst['train_k']} ∩ test k={worst['test_k']} = "
             f"{worst['train_test_overlap']}/{worst['test_size']})" if worst else ""))
    for c in (nonzero if a.verbose else []):
        print(f"  train k={c['train_k']} ∩ test k={c['test_k']}: {c['train_test_overlap']}/{c['test_size']}")
    tt = [c for c in report["cross_k_test_test"] if c["test_test_overlap"]]
    print(f"cross-k test∩test nonzero pairs: {len(tt)}/{len(report['cross_k_test_test'])}")
    rr = [c for c in report["cross_k_train_train"] if c["train_train_overlap"]]
    print(f"cross-k train∩train nonzero pairs: {len(rr)}/{len(report['cross_k_train_train'])}")

    if a.json:
        try:
            a.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"error: cannot write --json {a.json}: {e}", file=sys.stderr)
            return 3

    if report["integrity_errors"]:
        print("INTEGRITY ERRORS:", file=sys.stderr)
        for e in report["integrity_errors"]:
            print(f"  {e}", file=sys.stderr)
        return 1
    if a.fail_on_cross_k and report["cross_k_overlap_total"]:
        print("cross-k overlap present (--fail-on-cross-k)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
