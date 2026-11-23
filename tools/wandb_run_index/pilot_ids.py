#!/usr/bin/env python3
"""Seeded, stratified selection of the run-card pilot batch ids (offline; reads cards only).

  uv run --no-sync --with pyyaml python tools/wandb_run_index/pilot_ids.py --seed 20261110 --out <batch_dir>/ids.txt

Strata (each filled first, from cards with a non-empty stats table and not history_empty; a card may satisfy several):
  cohort clean >=6, legacy >=6, other >=3 | history_truncated >=3 | snapshot >=1 | multi_attribution >=4 (>=2 of them
  pp_ltf_*_lay2_s{4,5,6}) | status no-summary with stats >=2 | link tier != api-confirmed >=2 | z/* keys >=2 |
  the card with the most stats rows (max_tokens probe) | then seeded random fill to --n (default 24), balanced by cohort.
Writes the ids with a header recording the seed, the n, the per-id strata; the file is then the frozen pilot list
(plan reads the ids file, it never re-runs this script).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prose_check import Card  # noqa: E402

PPLTF = re.compile(r"pp_ltf_.*_lay2_.*s[456]\Z|pp_ltf_.*_lay2_k\d+_s[456]\Z")


def features(cards_root: Path) -> list[dict]:
    out = []
    for p in sorted(cards_root.glob("*/*.md")):
        if p.parent.name == "_meta":
            continue
        c = Card(p.read_text(encoding="utf-8"))
        fm = c.fm
        if not c.stats or fm.get("history_empty") is True:
            continue
        out.append({"id": str(fm["id"]), "name": str(fm.get("name")), "cohort": fm["cohort"], "status": fm.get("status"),
                    "tier": c.tier, "trunc": fm.get("history_truncated") is True, "snap": fm.get("snapshot") is True,
                    "multi": fm.get("multi_attribution") is True, "pp": bool(PPLTF.search(str(fm.get("name")))),
                    "z": any(k.startswith("z/") for k in c.stats), "rows": len(c.stats), "rel": p.name})
    return out


REQS = [  # (label, predicate, minimum)
    ("clean", lambda f: f["cohort"] == "clean", 6), ("legacy", lambda f: f["cohort"] == "legacy", 6),
    ("other", lambda f: f["cohort"] == "other", 3), ("truncated", lambda f: f["trunc"], 3),
    ("snapshot", lambda f: f["snap"], 1), ("multi", lambda f: f["multi"], 4),
    ("multi_pp_ltf_s456", lambda f: f["multi"] and f["pp"], 2),
    ("no-summary+stats", lambda f: f["status"] == "no-summary", 2),
    ("link!=api-confirmed", lambda f: f["tier"] != "api-confirmed", 2), ("z-keys", lambda f: f["z"], 2)]


def select(feats: list[dict], seed: int, n: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    chosen: dict[str, set] = {}

    def add(f, why):
        chosen.setdefault(f["id"], set()).add(why)

    big = max(feats, key=lambda f: (f["rows"], f["id"]))
    add(big, f"largest-table({big['rows']} rows)")
    for label, pred, need in sorted(REQS, key=lambda r: sum(map(r[1], feats))):          # rarest stratum first
        pool = [f for f in feats if pred(f)]
        if len(pool) < need:
            raise SystemExit(f"stratum {label!r}: only {len(pool)} candidates (< {need})")
        have = [f for f in pool if f["id"] in chosen]
        rng.shuffle(pool)
        for f in have:
            add(f, label)
        for f in [f for f in pool if f["id"] not in chosen][:max(0, need - len(have))]:
            add(f, label)
    by = {f["id"]: f for f in feats}
    pool = [f for f in feats if f["id"] not in chosen]
    rng.shuffle(pool)
    while len(chosen) < n and pool:           # balanced random fill: take from the cohort with the fewest chosen
        cnt = {c: sum(1 for i in chosen if by[i]["cohort"] == c) for c in ("clean", "legacy", "other")}
        want = min(("clean", "legacy"), key=lambda c: cnt[c])
        f = next((x for x in pool if x["cohort"] == want), pool[0])
        pool.remove(f)
        add(f, "random-fill")
    if len(chosen) > n:
        raise SystemExit(f"strata need {len(chosen)} cards > --n {n}")
    return [by[i] for i in sorted(chosen)], chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cards-root", type=Path, default=REPO / "lab/21_experiments/12_run-cards")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sel, why = select(features(a.cards_root), a.seed, a.n)
    lines = [f"# run-card pilot ids; seed={a.seed} n={len(sel)} (pilot_ids.py); frozen list — plan reads this file",
             f"# cohorts: " + ", ".join(f"{c}={sum(1 for f in sel if f['cohort'] == c)}" for c in ("clean", "legacy", "other"))]
    lines += [f"{f['id']}  # {f['cohort']} {f['name']} rows={f['rows']} [{', '.join(sorted(why[f['id']]))}]" for f in sel]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pilot_ids: {len(sel)} ids -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
