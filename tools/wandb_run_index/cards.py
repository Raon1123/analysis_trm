#!/usr/bin/env python3
"""Run-card skeletons: one markdown file per wandb run (Stage A, deterministic, no LLM prose).

Run:  uv run python tools/wandb_run_index/cards.py [--dry-run | --apply] [--exclude-cohort legacy ...]
                                                     [--only-project P ...] [--root .]

Inputs (all local, read-only):
  lab/21_experiments/11_lab-experiments/_wandb_runs_index.jsonl      (build.py)
  lab/21_experiments/12_run-cards/_meta/scan_cache.jsonl            (scan.py: sha256, stats)
  lab/21_experiments/12_run-cards/_meta/scan_snapshot.json          (generated_at = snapshot time)
  lab/21_experiments/12_run-cards/_meta/links.jsonl                 (links.py: tiers)
  checkpoints/<project>/<run_name>/step_*                           (existence only)
Output: lab/21_experiments/12_run-cards/<project>/<run_name>__<id>.md and _meta/cards_manifest.jsonl.

Idempotent: output is a pure function of the inputs (generated_at is the scan snapshot time, not now());
a file is rewritten only when its content changes; text between the prose:auto markers is preserved.
Cards are never deleted; cards present on disk but not in the current set are reported as stale.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402
from common import (CARDS_REL, COHORT_RULE_TEXT, END_STATE_RULE_TEXT, end_state_fields_differ, META_REL, card_relpath, collisions,  # noqa: E402
                    filter_cohort, render_card)

INDEX = "lab/21_experiments/11_lab-experiments/_wandb_runs_index.jsonl"
MAP = "lab/24_audits/11_lab-audits/2026-10-04_run-family-exp-map.yaml"
EXPD = "lab/21_experiments/11_lab-experiments"
THD = "lab/27_theory/11_lab-theory"
GENERATOR = "tools/wandb_run_index/cards.py v2"
MAX_SIBLINGS = 40


def sha_file(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in open(p)] if p.exists() else []


def checkpoint_info(root: Path, project: str, name: str) -> dict | None:
    if not name or project in ("", "?"):
        return None
    d = root / "checkpoints" / project / name
    if not d.is_dir():
        return None
    steps = sorted((x.name for x in d.iterdir() if re.fullmatch(r"step_\d+", x.name)),
                   key=lambda s: int(s.split("_")[1]))
    return {"dir": f"checkpoints/{project}/{name}", "real_dir": os.path.realpath(d), "steps": steps,
            "all_config_yaml": (d / "all_config.yaml").exists()}


def related_docs(root: Path, card_abs: Path, exps, hyps) -> list[tuple[str, str]]:
    out = []
    for e in sorted(exps):
        for p in sorted(glob.glob(str(root / EXPD / f"{e}_*.md"))):
            b = os.path.basename(p)
            if re.search(r"_(FREEZE|RETROFIT|VERIFY|PROPOSAL)", b):
                continue
            out.append((f"{e}: {b}", os.path.relpath(p, card_abs.parent)))
    for h in sorted(hyps):
        for b in (f"{h}.md", f"{h}.status.md"):
            p = root / THD / b
            if p.exists():
                out.append((f"{h}: {b}", os.path.relpath(p, card_abs.parent)))
    side = root / THD / "H-025_H-026_H-A_WANDB_RUNS.md"
    if any(h in ("H-025", "H-026", "H-A") for h in hyps) and side.exists():
        out.append(("H-025/H-026/H-A sidecar", os.path.relpath(side, card_abs.parent)))
    return out


def link_block(t: dict) -> dict:
    tier = t["tier"]
    b = {"tier": tier, "url_status": t["url_status"]}
    if tier == "api-confirmed":
        b["url"] = t["url"]
        b["api_name"], b["api_state"] = t.get("api_name"), t.get("api_state")
        b["name_mismatch_vs_local"] = bool(t.get("name_mismatch"))
    elif tier == "queue-log":
        b["url_printed_in_queue_log_not_api_confirmed"] = t["url"]
        b["finding"] = t.get("finding")
    elif tier == "constructed-unverified":
        b["url_constructed_unverified"] = t["url"]
    b["evidence"] = t.get("evidence", [])
    return b


def table_last_step(stats: dict | None) -> int | None:
    ks = ((stats or {}).get("keys") or {}) if (stats or {}).get("status") == "ok" else {}
    steps = [v["last_step"] for v in ks.values() if isinstance(v.get("last_step"), (int, float))]
    return max(steps) if steps else None


def attribution_lines(exp_pairs: list[dict]) -> list[str]:
    """Body lines for runs the (DRAFT) map assigns to more than one EXP.  Factual; no resolution."""
    ids = sorted({p["id"] for p in exp_pairs})
    if len(ids) < 2:
        return []
    lines = [f"The attribution map (DRAFT) assigns this run to several EXPs ({', '.join(ids)}); this is unresolved "
             "in this card."]
    by_src = {}
    for p in exp_pairs:
        by_src.setdefault(p["source"], set()).add(p["id"])
    if "prefix-rule" in by_src and "map" in by_src and "EXP-023" in by_src["prefix-rule"]:
        lines.append(f"{'/'.join(sorted(by_src['map']))} match(es) come from the map `name_regex` "
                     f"(`{build.MAP}`); EXP-023 comes from the generator prefix rule "
                     "(`tools/wandb_run_index/build.py` PREFIX_RULES, `pp_ltf_*_lay2_s{4,5,6}`), not from the map.")
    else:
        lines.append("Source per EXP: " + "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sorted(by_src.items()))
                     + ".")
    return lines


def build_cards(root: Path, rows: list[dict], scan: dict, links: dict, snapshot: str, inputs: dict,
                written_ids: set, rules: list | None = None) -> dict[str, tuple[dict, str]]:
    if rules is None:
        rules = build.load_rules(root) if (root / MAP).exists() else []
    by_sib, by_dup = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by_sib[r.get("sibling_key_sha")].append(r)
        by_dup[(r["project"], r["name"])].append(r)
    cards = {}
    paths = {r["id"]: card_relpath(r["project"], r["name"], r["id"]) for r in rows}
    for r in rows:
        if r["id"] not in written_ids:
            continue
        rel = paths[r["id"]]
        card_abs = root / CARDS_REL / rel
        t = links.get(r["id"])
        sc = scan.get(r["id"]) or {}
        notes = []
        if not r.get("has_metadata"):
            notes.append("no files/wandb-metadata.json: identity recovered from the .wandb RunRecord header "
                         f"(name source: {r.get('name_source')})")
        if r.get("status") == "no-summary":
            notes.append("files/wandb-summary.json is absent in the local run directory; the local files do not record an end state")
        if r.get("status") == "running?":
            notes.append(".wandb modified within 15 min of the scan snapshot: run may still be training; "
                         "stats are a partial snapshot")
        if r.get("cohort") == "unknown":
            notes.append(f"cohort unknown: {r.get('cohort_reason')}")
        if sc.get("wandb_sha256") != r.get("wandb_sha256"):
            notes.append("scan cache sha256 differs from index sha256 -- stats may not match the indexed file")
        dups = [x for x in by_dup[(r["project"], r["name"])] if x["id"] != r["id"]]
        ck = checkpoint_info(root, r["project"], r["name"])
        if ck and dups:
            ck["shared_by_duplicate_run_names"] = True
            notes.append(f"{len(dups)} other run(s) share this project+run_name; the checkpoint dir is keyed by "
                         "name and cannot be attributed to one run id")
        if t is None:
            raise SystemExit(f"ERROR: no link tier for {r['id']}")
        if t.get("name_mismatch"):
            notes.append(f"API run name {t.get('api_name')!r} differs from local name {r['name']!r}")
        pairs = build.attribution_pairs(r["project"], r["name"], rules)
        exp_pairs = [{k: p[k] for k in ("id", "source", "confidence")} for p in pairs if p["kind"] == "EXP"]
        hyp_pairs = [{k: p[k] for k in ("id", "source", "confidence")} for p in pairs if p["kind"] == "H"]
        multi = len({p["id"] for p in exp_pairs}) > 1
        summ_step, tbl_step = r.get("last_step"), table_last_step(sc.get("stats"))
        trunc = (r.get("last_step_source") == "wandb-summary.json:_step" and isinstance(summ_step, (int, float))
                 and tbl_step is not None and summ_step > tbl_step)
        running = r.get("status") == "running?"
        hist_empty = (sc.get("stats") or {}).get("status") == "ok" and (sc.get("stats") or {}).get("history_records") == 0
        sibs = sorted((x for x in by_sib[r.get("sibling_key_sha")] if x["id"] != r["id"]),
                      key=lambda x: (x["name"], x["id"]))
        fm = {
            "id": r["id"], "name": r["name"], "project": r["project"],
            "wandb_project": r.get("wandb_project"), "group": r.get("group"),
            "started": r.get("started_at"), "status": r.get("status"), "mode": r.get("mode"),
            "has_metadata": r.get("has_metadata"), "last_step": r.get("last_step"),
            "last_step_source": r.get("last_step_source"),
            "history_truncated": trunc,
            **({"summary_last_step": summ_step, "history_last_step": sc.get("last_history_step"),
                "stats_table_last_step": tbl_step} if trunc else {}),
            **({"history_empty": True} if hist_empty else {}),
            "snapshot": running, **({"snapshot_utc": snapshot} if running else {}),
            "exit_code": r.get("exit_code"),
            "end_state_fields_differ": end_state_fields_differ(r.get("status"), r.get("exit_code"), link_block(t).get("api_state")),
            "end_state_rule": "tools/wandb_run_index/common.py:END_STATE_RULE_TEXT (see _meta/README.md)",
            "cohort": r.get("cohort"), "cohort_reason": r.get("cohort_reason"),
            "cohort_rule": "tools/wandb_run_index/common.py:COHORT_RULE_TEXT (see _meta/README.md)",
            "link": link_block(t),
            "local": {"wandb_file": r.get("wandb_file"), "wandb_size": r.get("wandb_size"),
                      "wandb_sha256": r.get("wandb_sha256"), "run_dir": r.get("run_dir"),
                      "synced_marker": r.get("synced_marker")},
            "config": {"arch_preset": r.get("arch_preset"), "arch_name": r.get("arch_name"),
                       "H_cycles": r.get("H_cycles"), "L_cycles": r.get("L_cycles"), "k": r.get("k"),
                       "seed": r.get("seed"), "data_paths": r.get("data_paths"), "ema": r.get("ema"),
                       "log_z_dynamics": r.get("log_z_dynamics"), "epochs": r.get("epochs") or None,
                       "source": r.get("config_source")},
            "git_commit": r.get("git_commit"),
            "checkpoints": ck,
            "multi_attribution": multi,
            "attribution": {"status": "DRAFT (run-family map + generator prefix rules; not evidence)",
                            "exps": r.get("exps", []), "hyps": r.get("hyps", []),
                            "exp_attribution": exp_pairs, "hyp_attribution": hyp_pairs},
            "siblings": {"rule": "same project + identical config except seed/run_name/run_group/cell_id/"
                                 "checkpoint/output paths (common.SIBLING_IGNORE)",
                         "n": len(sibs),
                         "runs": [{"id": x["id"], "name": x["name"], "seed": x.get("seed"),
                                   "card": os.path.relpath(root / CARDS_REL / paths[x["id"]], card_abs.parent)}
                                  for x in sibs[:MAX_SIBLINGS]]},
            "duplicates": [{"id": x["id"], "started": x.get("started_at"), "status": x.get("status"),
                            "card": os.path.relpath(root / CARDS_REL / paths[x["id"]], card_abs.parent)}
                           for x in sorted(dups, key=lambda x: (x.get("started_at") or "", x["id"]))],
            "queue_logs": t.get("queue_logs", []),
            "generator": GENERATOR, "generated_at": snapshot, "inputs_sha256": inputs,
        }
        stats = sc.get("stats")
        rel_docs = related_docs(root, card_abs, r.get("exps", []), r.get("hyps", []))
        old = card_abs.read_text() if card_abs.exists() else None
        cards[rel] = (r, render_card(fm, stats, rel_docs, old, notes,
                                                 attribution_lines(exp_pairs) if multi else None))
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ayp/project/trm")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--exclude-cohort", action="append", default=[],
                    choices=["clean", "legacy", "other", "unknown"])
    ap.add_argument("--only-project", action="append", default=[])
    a = ap.parse_args()
    root = Path(a.root).resolve()
    meta = root / META_REL
    rows = load_jsonl(root / INDEX)
    scan = {r["id"]: r for r in load_jsonl(meta / "scan_cache.jsonl")}
    links = {r["id"]: r for r in load_jsonl(meta / "links.jsonl")}
    if not rows or not scan or not links:
        print(f"ERROR: empty input (index {len(rows)}, scan {len(scan)}, links {len(links)})", file=sys.stderr)
        return 2
    snapshot = json.loads((meta / "scan_snapshot.json").read_text())["snapshot_utc"]
    inputs = {"index": sha_file(root / INDEX), "scan_cache": sha_file(meta / "scan_cache.jsonl"),
              "links": sha_file(meta / "links.jsonl"), "attribution_map": sha_file(root / MAP)}
    sel = filter_cohort(rows, a.exclude_cohort)
    if a.only_project:
        sel = [r for r in sel if r["project"] in a.only_project]
    excluded = len(rows) - len(sel)
    ids = {r["id"] for r in sel}
    all_paths = [card_relpath(r["project"], r["name"], r["id"]) for r in rows]
    col = collisions(all_paths)
    if col:
        print(f"ERROR: {len(col)} filename collisions (case-insensitive): {col[:5]}", file=sys.stderr)
        return 3
    cards = build_cards(root, rows, scan, links, snapshot, inputs, ids)
    changed = new = same = 0
    changed_running, changed_unexpected = [], []
    for rel, (r, txt) in sorted(cards.items()):
        p = root / CARDS_REL / rel
        if not p.exists():
            new += 1
        elif p.read_text() != txt:
            changed += 1
            (changed_running if r.get("status") == "running?" else changed_unexpected).append(r["id"])
        else:
            same += 1
            continue
        if a.apply:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
    on_disk = {os.path.relpath(p, root / CARDS_REL) for p in glob.glob(str(root / CARDS_REL / "*" / "*.md"))
               if "/_meta/" not in p}
    stale = sorted(on_disk - set(all_paths))
    print(f"rows {len(rows)}  selected {len(sel)}  excluded {excluded} "
          f"(cohort {a.exclude_cohort or '-'}, project {a.only_project or '-'})")
    print(f"cards: new {new}  changed {changed} (running {len(changed_running)}, "
          f"unexpected {len(changed_unexpected)})  unchanged {same}  collisions 0  stale-on-disk {len(stale)}"
          f"  [{'APPLIED' if a.apply else 'dry-run'}]")
    print(f"unexpected changes: {len(changed_unexpected)} {changed_unexpected[:20] or ''}")
    if changed_running:
        print(f"changed (running, snapshot cards): {len(changed_running)} {changed_running[:20]}")
    print("cohort of selected:", dict(collections.Counter(r.get("cohort") for r in sel)))
    if a.apply:
        meta.mkdir(parents=True, exist_ok=True)
        man = meta / "cards_manifest.jsonl"
        lines = [json.dumps({"id": r["id"], "project": r["project"], "name": r["name"], "cohort": r.get("cohort"),
                             "status": r.get("status"), "link_tier": links[r["id"]]["tier"], "card": rel,
                             "sha256": hashlib.sha256(txt.encode()).hexdigest()}, sort_keys=True)
                 for rel, (r, txt) in sorted(cards.items())]
        if a.exclude_cohort or a.only_project:
            man = meta / "cards_manifest.filtered.jsonl"
        new_txt = "\n".join(lines) + "\n"
        if not man.exists() or man.read_text() != new_txt:
            man.write_text(new_txt)
        rd = meta / "README.md"
        rd_txt = ("# run-card _meta (script-generated; do not edit)\n\n"
                  "Files: `scan_cache.jsonl` (scan.py), `scan_snapshot.json`, `api_listing.json` + `links.jsonl` + "
                  "`links_counts.json` (links.py), `cards_manifest.jsonl` (cards.py).\n\n"
                  "## Cohort rule (common.COHORT_RULE_TEXT)\n\n" + COHORT_RULE_TEXT + "\n\n"
                  "## End-state fields rule (common.END_STATE_RULE_TEXT)\n\n" + END_STATE_RULE_TEXT + "\n")
        if not rd.exists() or rd.read_text() != rd_txt:
            rd.write_text(rd_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
