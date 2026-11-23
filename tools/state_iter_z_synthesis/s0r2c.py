#!/usr/bin/env python3
"""S0r2_diff_vs_r1.csv: cell-level r1 -> r2 as_of class comparison + probe-count changes + snapshot-file hash changes."""
import csv, os, sys
os.chdir("/home/ayp/project/trm")
OUT = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis"
rd = lambda p: list(csv.DictReader([l for l in open(f"{OUT}/{p}", encoding="utf-8") if not l.startswith("#")]))
r1 = {(r["q"], r["s"]): r for r in rd("S0_class_repro_check.csv")}
r2 = {(r["q"], r["s"]): r for r in rd("S0r2_class_repro_check.csv")}
inv1 = {r["source_row"]: r for r in rd("S0_inventory.csv")}
inv2 = {r["source_row"]: r for r in rd("S0r2_inventory.csv")}
dst = f"{OUT}/S0r2_diff_vs_r1.csv"
if os.path.exists(dst):
    sys.exit("refuse to overwrite")
def shalist(p):
    return {l.split("  ", 1)[1].strip(): l.split("  ", 1)[0] for l in open(f"{OUT}/{p}") if l.strip()}
s1, s2 = shalist("S0_snapshot.sha256"), shalist("S0r2_snapshot.sha256")
with open(dst, "w", newline="", encoding="utf-8") as f:
    f.write("# S0r2 vs S0 r1 (as_of 2026-10-30T12:35:46Z -> r2 as_of in S0r2_snapshot/as_of.txt). section=cell|probe|snapshot_file\n")
    w = csv.writer(f)
    w.writerow(["section", "key", "r1", "r2", "changed", "r1_rule_or_note", "r2_rule_or_note"])
    for k in r1:
        a, b = r1[k], r2[k]
        w.writerow(["cell", f"{k[0]}|{k[1]}", a["as_of_class"], b["as_of_class"], a["as_of_class"] != b["as_of_class"], a["rule"], b["rule"]])
    cols = ["ckpt_dirs_matched", "queue_jobs", "queue_processing", "queue_done", "queue_failed", "path_probe_hits", "data_present"]
    for s in inv1:
        if inv1[s]["kind"] != "E": continue
        v1 = "/".join(inv1[s][c] for c in cols); v2 = "/".join(inv2[s][c] for c in cols)
        w.writerow(["probe", s, v1, v2, v1 != v2, "ckpt/jobs/proc/done/failed/path/data", ""])
    for n in sorted(set(s1) | set(s2)):
        if n in ("as_of.txt", "evidence_files.sha256"): 
            continue
        w.writerow(["snapshot_file", n, s1.get(n, "-")[:12], s2.get(n, "-")[:12], s1.get(n) != s2.get(n), "", ""])
