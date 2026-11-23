#!/usr/bin/env python3
"""Exit-code gate for the run-card pilot (decision without human arithmetic).

  uv run --no-sync --with pyyaml python tools/wandb_run_index/pilot_gate.py <batch_dir> [--lab-root lab]

Reads the collected batch (plan.json, manifest.jsonl, items/*.md) and re-runs prose_check on it. Exit 0 only if ALL hold:
  G1 every manifest request has a collected item (none missing)
  G2 every item has stop_reason == end_turn (0 max_tokens, 0 refusals, 0 pause_turn) and result_status == succeeded
  G3 model_returned == model_requested == the manifest model, for every item
  G4 (quarantined + missing) / N_req <= 0.20
  G5 max output_tokens <= 0.6 * max_tokens
  G6 sum(cost_est) <= plan hard-worst, and no item's cost_est exceeds its own hard-worst
Exit 1 = a rule failed (all numbers and the per-rule quarantine histogram are printed); 2 = error (could not evaluate).
Not decided here (human): reading >= 6 injected samples, and judging whether the quarantine reasons were checker
over-strictness (fix the rule, re-run prose_check on the same items for free, re-run this gate).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prose_check import check_batch, item_batch_ids, parse_item, resolve_sent  # noqa: E402

MAX_QUARANTINE_RATE, MAX_TOKEN_FRAC = 0.20, 0.6


def histogram(quarantined: list[dict]) -> collections.Counter:
    h = collections.Counter()
    for q in quarantined:
        for r in q["reasons"]:
            h[re.sub(r"\[\d+\]", "[]", re.sub(r"'[^']*'", "'…'", r))[:110]] += 1
    return h


def gate(batch_dir: Path, lab_root: Path) -> tuple[int, list[str]]:
    from tools.api_batch import manifest as M
    bd = Path(batch_dir)
    rows, plan, msha, src = resolve_sent(bd, lab_root, item_batch_ids(bd))
    n = len(rows)
    worst_by = {x["custom_id"]: x["hard_worst_usd"] for x in plan["requests"]}
    lines, ok = [f"pilot_gate: {bd.name}  N_req={n}  manifest {msha[:12]} ({src})  plan hard-worst=${plan['est_usd_hard_worst']:.6f}"], True

    def rule(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")

    items, stop, model_bad, out_tok, cost, over, cache_read = {}, collections.Counter(), [], [], 0.0, [], 0
    no_usage, no_cost = [], []
    for r in rows:
        ip = bd / "items" / f"{r['custom_id']}.md"
        if not ip.is_file():
            continue
        fm, _ = parse_item(ip)
        items[r["custom_id"]] = fm
        stop[f"{fm.get('stop_reason')}/{fm.get('result_status')}"] += 1
        if not (fm.get("model_returned") == fm.get("model_requested") == r["params"]["model"]):
            model_bad.append(r["custom_id"])
        try:
            u = json.loads(fm.get("usage") or "{}")
            out_tok.append(int(u["output_tokens"]))
            cache_read += int(u.get("cache_read_input_tokens") or 0)
        except (ValueError, KeyError, TypeError):
            no_usage.append(r["custom_id"])        # missing / unreadable usage must FAIL G5, never read as 0 tokens
        try:
            c = float(fm["cost_est_usd"])
        except (ValueError, KeyError, TypeError):
            no_cost.append(r["custom_id"])
            c = 0.0
        cost += c
        if c > worst_by[r["custom_id"]] + 1e-9:
            over.append(r["custom_id"])
    rule("G1 all requests collected", len(items) == n, f"{len(items)}/{n} items")
    bad_stop = {k: v for k, v in stop.items() if k != "end_turn/succeeded"}
    rule("G2 stop_reason end_turn / succeeded for all", not bad_stop and len(items) > 0, f"counts {dict(stop)}")
    rule("G3 returned model == requested == manifest", not model_bad and len(items) > 0, f"{len(model_bad)} mismatches")
    chk = check_batch(bd, lab_root)
    nq, nm = len(chk["quarantined"]), len(chk["missing"])
    rate = (nq + nm) / n
    rule(f"G4 quarantine rate <= {MAX_QUARANTINE_RATE}", rate <= MAX_QUARANTINE_RATE, f"({nq} quarantined + {nm} missing)/{n} = {rate:.3f}")
    mt = max(r["params"]["max_tokens"] for r in rows)
    mo = max(out_tok) if out_tok else 0
    rule(f"G5 max output_tokens <= {MAX_TOKEN_FRAC} * max_tokens", out_tok != [] and not no_usage and mo <= MAX_TOKEN_FRAC * mt,
         f"max output_tokens {mo} vs {MAX_TOKEN_FRAC * mt:.0f} (max_tokens {mt}); {len(no_usage)} items without usage")
    rule("G6 cost_est <= hard-worst", cost <= plan["est_usd_hard_worst"] and not over and not no_cost and len(items) > 0,
         f"sum cost_est ${cost:.6f} vs ${plan['est_usd_hard_worst']:.6f}; plan expected ${plan['est_usd_expected']:.6f}; {len(over)} items over their own hard-worst; {len(no_cost)} items without cost_est_usd")
    lines.append(f"  info  cache_read_input_tokens total {cache_read}; accepted {len(chk['accepted'])}")
    lines.append("  quarantine histogram (per rule, count of reasons):")
    for k, v in histogram(chk["quarantined"]).most_common():
        lines.append(f"    {v:4d}  {k}")
    if chk["missing"]:
        lines.append(f"    {nm:4d}  missing item (not collected / L5-quarantined)")
    lines.append(f"pilot_gate: {'PASS (exit 0)' if ok else 'FAIL (exit 1)'}  — human gate still required: read >= 6 injected samples")
    return (0 if ok else 1), lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--lab-root", type=Path, default=REPO / "lab")
    ap.add_argument("--write", action="store_true", help="also write the printed gate report to <batch_dir>/pilot_gate.txt (default: stdout only)")
    ap.add_argument("--report", type=Path, default=None, metavar="PATH", help="write the printed gate report to PATH")
    a = ap.parse_args(argv)
    try:
        rc, lines = gate(a.batch_dir, a.lab_root)
    except Exception as e:  # noqa: BLE001
        print(f"pilot_gate: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    for dest in ([a.report] if a.report else []) + ([a.batch_dir / "pilot_gate.txt"] if a.write else []):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
