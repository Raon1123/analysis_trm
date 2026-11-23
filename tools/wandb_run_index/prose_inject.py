#!/usr/bin/env python3
"""Inject accepted run-card prose into the prose:auto region of the cards (idempotent, region-only).

  uv run --no-sync --with pyyaml python tools/wandb_run_index/prose_inject.py <batch_dir> [--lab-root lab]
                                                   [--dry-run] [--overwrite] [--no-fixed-fill]

Re-runs prose_check on the batch itself (a stale postcheck.json is never trusted), then for every ACCEPTED output
rewrites only the text between `<!-- prose:auto start -->` and `<!-- prose:auto end -->`; quarantined / missing
outputs are never injected. Cards that got no request because their stats table is empty (plan.json runcard.skipped)
get a fixed script sentence in the same region (--no-fixed-fill to skip).
Guards: the card's blanked sha256 must equal the planned one (else refused, re-plan); a non-empty region that
differs from the new text is a conflict (refused unless --overwrite); after writing, the bytes before the start
marker and from the end marker on are asserted identical to the original. Writes <batch_dir>/inject_report.json (not with --dry-run, which writes nothing into the batch dir unless --report PATH is given).
Exit: 0 all done; 3 some outputs quarantined/missing/conflicting/drifted (accepted ones are still injected); 2 error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prose_check import check_batch, norm, resolve_sent  # noqa: E402
from tools.api_batch import runcard as RC  # noqa: E402

DRAFT = "DRAFT (api-batch output — not evidence)"


def render_attribution(card_text: str) -> list[str]:
    """Script-rendered attribution lines (ids / source / confidence exactly as the card frontmatter prints them)."""
    fm, _ = RC.split_frontmatter(card_text)
    at = fm.get("attribution") or {}
    L = []
    for kind, lst, det in (("exps", at.get("exps") or [], at.get("exp_attribution") or []),
                           ("hyps", at.get("hyps") or [], at.get("hyp_attribution") or [])):
        if not lst:
            continue
        info = {str(d.get("id")): d for d in det if isinstance(d, dict)}
        L.append(f"- {kind}: " + "; ".join(
            f"{i} (source {info[str(i)].get('source')}, confidence {info[str(i)].get('confidence')})" if str(i) in info else str(i)
            for i in lst))
    return L


def render_model(out: dict, model: str, batch_id: str, cid: str, card_text: str | None = None) -> str:
    attr = render_attribution(card_text) if card_text else []
    L = [f"{DRAFT} · model {model} · batch {batch_id} · {cid}", "",
         f"**What this run is.** {norm(out['what_this_run_is'])}", "",
         f"**Attribution (DRAFT map).** {norm(out['attribution'])}", *attr, "",
         f"**What the numbers show.** {norm(out['numbers_summary'])}", "",
         "| key | stat | value | step |", "|---|---|---|---|"]
    L += [f"| `{e['key']}` | {e['stat']} | {e['value']} | {str(e['step']).lstrip('@')} |" for e in out["what_the_numbers_show"]]
    L += ["", "**Caveats.**"]
    L += [f"- ({c['kind']}) {norm(c['text'])}" for c in out["caveats"]] or ["- none noted"]
    if out["see_also"]:
        L += ["", "**See also.** " + "; ".join(norm(s) for s in out["see_also"])]
    return "\n".join(L)


def render_fixed(status: str, cohort: str, reason: str = "no_stats_rows") -> str:
    why = ("The local .wandb file contains no history records (history_empty)." if reason == "history_empty"
           else "No metric rows were recorded for this run (empty stats table).")
    return (f"DRAFT (script-filled, no model — not evidence) · {why} No model prose was requested. "
            f"Status: {status}; cohort: {cohort}.")


def region_bounds(text: str) -> tuple[int, int]:
    """(end of start marker, start of end marker); RC.blanked_text validates there is exactly one ordered pair."""
    RC.blanked_text(text)
    return text.index(RC.MARK_S) + len(RC.MARK_S), text.index(RC.MARK_E)


def apply_region(text: str, body: str, overwrite: bool) -> tuple[str, str]:
    """(new_text, state) with state in injected|unchanged|conflict. Only bytes strictly between the markers change."""
    a, b = region_bounds(text)
    cur = text[a:b]
    new = f"\n{body}\n"
    if cur == new:
        return text, "unchanged"
    if cur.strip() and not overwrite:
        return text, "conflict"
    out = text[:a] + new + text[b:]
    na, nb = region_bounds(out)
    assert out[:a] == text[:a] and out[nb:] == text[b:] and na == a, "bytes outside the prose region changed"
    return out, "injected"


def inject(batch_dir: Path, lab_root: Path, *, dry_run=False, overwrite=False, fixed_fill=True, report: Path | None = None) -> dict:
    bd = Path(batch_dir)
    chk = check_batch(bd, lab_root)
    srows, plan, _msha, _src = resolve_sent(bd, lab_root, {a.get("batch_id") for a in chk["accepted"]})
    rep = {"batch_dir": str(bd), "manifest_sha256": chk["manifest_sha256"], "dry_run": dry_run, "injected": [], "unchanged": [],
           "conflict": [], "drift": [], "fixed_fill": [], "quarantined": chk["quarantined"], "missing": chk["missing"]}
    jobs = []
    for a in chk["accepted"]:
        jobs.append((a["card_rel"], a["run_id"], None,
                     lambda t, a=a: render_model(a["output"], a["model"], a["batch_id"], a["custom_id"], t)))
    if fixed_fill:
        for s in plan["runcard"]["skipped"]:
            jobs.append((s["rel"], s["id"], s["card_sha256"], render_fixed(s["status"], s["cohort"], s.get("reason", "no_stats_rows"))))
    planned = {}
    for r in srows:
        planned[r["meta"]["card_rel"]] = r["meta"]["card_sha256"]
    for rel, rid, sha, body in jobs:
        sha = sha or planned[rel]
        p = Path(lab_root) / rel
        text = p.read_text(encoding="utf-8")
        if RC.card_sha(text) != sha:
            rep["drift"].append({"card_rel": rel, "run_id": rid, "reason": "card changed since plan; re-plan"})
            continue
        new, state = apply_region(text, body(text) if callable(body) else body, overwrite)
        if state == "injected" and not dry_run:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(new, encoding="utf-8")
            tmp.replace(p)
        key = "fixed_fill" if (state == "injected" and sha != planned.get(rel)) else state
        (rep[key] if key in rep else rep["injected"]).append({"card_rel": rel, "run_id": rid})
    if report or not dry_run:                      # --dry-run never writes into the batch dir (only to a caller-given path)
        rp = Path(report) if report else bd / "inject_report.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--lab-root", type=Path, default=REPO / "lab")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-fixed-fill", action="store_true")
    ap.add_argument("--report", type=Path, default=None, help="where to write the report (default: <batch_dir>/inject_report.json, "
                    "and none at all with --dry-run, which then only prints counts)")
    a = ap.parse_args(argv)
    try:
        rep = inject(a.batch_dir, a.lab_root, dry_run=a.dry_run, overwrite=a.overwrite, fixed_fill=not a.no_fixed_fill,
                    report=a.report)
    except (ValueError, OSError, KeyError, AssertionError, RC.RunCardError) as e:
        print(f"prose_inject: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    n = {k: len(rep[k]) for k in ("injected", "unchanged", "conflict", "drift", "fixed_fill", "quarantined", "missing")}
    print(f"prose_inject{' (dry-run)' if a.dry_run else ''}: {n}")
    if a.dry_run and not a.report:
        print(json.dumps(rep, indent=1, ensure_ascii=False, sort_keys=True))
    return 0 if not (n["conflict"] or n["drift"] or n["quarantined"] or n["missing"]) else 3


if __name__ == "__main__":
    sys.exit(main())
