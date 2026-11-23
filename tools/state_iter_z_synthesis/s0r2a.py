#!/usr/bin/env python3
"""S0 for 2026-10-30_state-iter-z-synthesis (PREREG sha256 87fa6e7a...). No metric values are read.

Produces (PREREG sec.9 S0, sec.11):
  S0r2_snapshot/            copies of state files + name/existence listings + as_of.txt
  S0r2_snapshot.sha256      sha256 of every file under S0r2_snapshot/
  S0r2_metric_keys.csv      27 rows (PREREG sec.7.1 branch-6 constant table, evidence strings checked)
  S0r2_cells_expected.csv   58 cells (PREREG sec.7.0.5 expected classes + as_of existence checks)

Evidence checks read design-doc / script files only as Python strings and emit LINE NUMBERS and
COUNTS of fixed-key substrings; no line content is ever written out.
Exit code: 0 ok; 2 any evidence/constant/count discrepancy (PREREG is immutable -> stop and report).
Run: CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s0r2a.py
"""
import csv, datetime, hashlib, os, re, shutil, sys
import yaml

AS_OF = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
ROOT = "/home/ayp/project/trm"
os.chdir(ROOT)
PREREG = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md"
PREREG_SHA = "87fa6e7acf2ba7278c4dea1835c1c27790d1bfe519b2e966374568b300c73e1b"
OUT = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis"
SNAP = f"{OUT}/S0r2_snapshot"
EXPD = "lab/21_experiments/11_lab-experiments"
REPD = "lab/23_reports/11_lab-reports"
REG = "lab/13_reference/11_registries"
GOV = "lab/11_governance/11_lab-root"
problems = []


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flag(msg):
    problems.append(msg)
    print("DISCREPANCY:", msg)


# ---------------------------------------------------------------- 0. immutability gate
if sha(PREREG) != PREREG_SHA:
    print("PREREG sha mismatch -> abort")
    sys.exit(2)
SCRIPT_SHA = sha(__file__)
if os.path.exists(SNAP):
    print("S0r2_snapshot exists -> refuse to overwrite"); sys.exit(3)
SUPER = (f"# S0r2 (as_of {AS_OF}) supersedes S0 r1 (as_of 2026-10-30T12:35:46Z): r1 EXP-011 queue probe `_abl_` matched pp_abl_* "
         f"(false positive, VERIFY-SIZ-B C1); PREREG unchanged\n")
os.makedirs(SNAP, exist_ok=True)
with open(f"{SNAP}/as_of.txt", "w") as f:
    f.write(f"as_of: {AS_OF}\nprereg: {PREREG}\nprereg_sha256: {PREREG_SHA}\n"
            f"script: tools/state_iter_z_synthesis/s0r2a.py\nscript_sha256_at_run: {SCRIPT_SHA}\n")

# ---------------------------------------------------------------- 1. snapshot
copies = [
    f"{REG}/EXPERIMENTS.md", f"{REG}/HYPOTHESES.md", f"{REG}/ID_ALIASES.yaml",
    "lab/13_reference/12_atoms/INDEX.jsonl",
    "lab/27_theory/11_lab-theory/H-199.status.md", "lab/27_theory/11_lab-theory/H-200.status.md",
    "lab/27_theory/11_lab-theory/H-201.status.md",
    f"{GOV}/WITHHELD_RUNS.yaml", f"{GOV}/API_BATCH_DENY.yaml",
]
for p in copies:
    if not os.path.exists(p):
        flag(f"snapshot source missing: {p}")
        continue
    dst = f"{SNAP}/{os.path.basename(p)}"
    shutil.copy2(p, dst)

# queue names (jobs/processing/done/failed); the sealed-* subdir is NOT touched (PREREG: no existence check)
with open(f"{SNAP}/queue_names.txt", "w") as f:
    for d in ["jobs", "processing", "done", "failed"]:
        for n in sorted(os.listdir(f"scripts/queue/{d}")):
            if n.startswith("sealed"):
                continue
            if os.path.isfile(f"scripts/queue/{d}/{n}"):
                f.write(f"{d}\t{n}\n")
# experiments dir name list
with open(f"{SNAP}/exp_dir_names.txt", "w") as f:
    for n in sorted(os.listdir(EXPD)):
        f.write(n + "\n")
# 23_reports direct-child *_RESULTS.md : only the frontmatter exp_id line
top_results = {}  # filename -> first token
for n in sorted(os.listdir(REPD)):
    p = f"{REPD}/{n}"
    if n.endswith("_RESULTS.md") and os.path.isfile(p):
        tok = "NO_EXP_ID"
        with open(p, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        if lines and lines[0].strip() == "---":
            for ln in lines[1:80]:
                if ln.strip() == "---":
                    break
                m = re.match(r"exp_id:\s*(.+)", ln)
                if m:
                    tok = re.split(r"[\s(#]", m.group(1).strip().strip("\"'"))[0]
                    break
        top_results[n] = tok
with open(f"{SNAP}/results_top_exp_ids.tsv", "w") as f:
    for n, t in top_results.items():
        f.write(f"{n}\t{t}\n")

# ID aliases
aliases = {}
ya = yaml.safe_load(open(f"{REG}/ID_ALIASES.yaml", encoding="utf-8"))
for e in ya["entries"]:
    aliases.setdefault(e["canonical"], {e["canonical"]})
    for a in e.get("aliases", []) or []:
        aliases[e["canonical"]].add(a["string"])

# ---------------------------------------------------------------- 2. parse PREREG tables
text = open(PREREG, encoding="utf-8").read().split("\n")


def section(start_pat, end_pat):
    s = next(i for i, l in enumerate(text) if re.match(start_pat, l))
    e = next(i for i in range(s + 1, len(text)) if re.match(end_pat, text[i]))
    return text[s:e]


def cols(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


# 7.0.5 universe
sec = section(r"#### 7\.0\.5", r"### 7\.1")
cells = []  # (q, s, class, qtoken)
for l in sec:
    if not l.startswith("| Q-"):
        continue
    c = cols(l)
    q = c[0]
    qtok = re.match(r"Q-[A-Z-]+", c[3]).group(0)
    for part in c[1].split(" · "):
        s, cl = [x.strip() for x in part.split("→")]
        cells.append((q, s, cl, qtok))


def kind_of(s):
    if s.startswith("EXP-") or s == "H-108 sweep":
        return "E"
    if s.startswith(("CL-", "FG-")):
        return "A"
    return "H"


# 7.1 branch-6 constant table
sec6 = section(r"\*\*갈래 6 상수표", r"### 7\.2")
const = {}  # rid -> dict
for l in sec6:
    m = re.match(r"\| (R\d\d) \|", l)
    if not m:
        continue
    c = cols(l)
    q, s = re.match(r"\((Q-[A-Z0-9]+), (.+)\)$", c[1]).groups()
    if s == "H-108":
        s = "H-108 sweep"
    const[c[0]] = dict(q=q, s=s, cls=c[2], ev=c[3], N=c[4], status=c[5])
if len(const) != 27:
    flag(f"constant table rows parsed = {len(const)} (expected 27)")

# ---------------------------------------------------------------- 3. evidence specs
D = lambda n: f"{EXPD}/{n}"
F = dict(
    e001=D("EXP-001_fig12-z-iter-ablation.md"), e002=D("EXP-002_participation-ratio-analysis.md"),
    e004=D("EXP-004_h023-noz-collapse-reframe.md"), e005=D("EXP-005_fig1-mlp-block-ablation.md"),
    e006=D("EXP-006_sigma-k-tau-grid.md"), e007=D("EXP-007_k9-k11-cycletype-discriminator.md"),
    e011=D("EXP-011_module-ablation-sigma-k-new.md"), e012=D("EXP-012_clean-r1-mechanism-scaling-gates.md"),
    e016=D("EXP-016_act-deep-supervision-axis.md"), e017=D("EXP-017_z-dynamics-latent-formation.md"),
    e018=D("EXP-018_separation-signal-gates.md"), e025=D("EXP-015_looped-tf-direct-control.md"),
    h108=f"{REPD}/2026-09-02_h108-depth-sweep-prereg.md",
    e022=f"{REPD}/2026-10-21_pc1-ltf-per-loop-decode-prereg.md",
    e023=f"{REPD}/2026-10-21_pg1-ltf-seed-replication-prereg.md",
    e020=f"{REPD}/2026-10-22_exp020-act-budget-extension-prereg.md",
    pr="analysis/pr_recompute.py", dse="analysis/depth_sweep_eval.py", h108s="analysis/h108_summarize.py",
)
P, A, RC, EF = "probe/test_exact", "all/exact_accuracy", "recompute_test_acc", "z/eff_rank"
FIXED = [P, A, RC, EF]
# presence items: (file, key, stated_lines). count items: (file, key, expected_count). regex-count items.
EV = {
    "R01": dict(pres=[("h108", "test_exact", [27, 74]), ("h108", "all.exact_accuracy", [166]), ("h108", P, [167])],
                zero=[("dse", k) for k in FIXED] + [("h108s", k) for k in FIXED]),
    "R02": dict(pres=[("e001", "test_exact", [30, 67, 145]), ("e001", EF, [68])], count=[("pr", EF, 4)],
                zero=[("pr", P), ("pr", A), ("pr", RC)]),
    "R03": dict(pres=[("e005", P, [141, 168, 184, 241]), ("e005", A, [140, 141, 153, 154, 169, 241]),
                      ("e005", RC, [141])]),
    "R04": dict(pres=[("e011", P, [122, 150, 153, 161])]),
    "R05": dict(pres=[("e012", P, [101])]),
    "R06": dict(absent_file=f"{EXPD}/EXP-013_fig1-ablation-multiseed-replication.md"),
    "R07": dict(pres=[("e025", P, [66, 67, 79, 165])]),
    "R08": dict(pres=[("e022", P, [61, 62, 73, 127, 205, 276, 319, 520]), ("e022", "train/exact_accuracy", [205, 522])]),
    "R09": dict(pres=[("e023", P, [44, 204, 206])]),
    "R10": dict(pres=[("e011", P, [122, 150, 153, 161])]),
    "R11": dict(pres=[("e016", P, [78, 85, 107, 139]), ("e016", A, [505])]),
    "R12": dict(pres=[("e020", P, [63, 212])]),
    "R14": dict(pres=[("e004", P, [170])], count=[("e004", P, 1)]),
    "R16": dict(pres=[("e007", P, [41, 140])]),
    "R22": dict(pres=[("e002", EF, [43, 101])]),
    "R23": dict(pres=[("e004", "analysis/data/pr_recompute_test.csv", [68, 82]), ("e004", P, [170])],
                zero=[("e004", EF)], zero_re=[("e004", r"\.py\b")]),
    "R24": dict(pres=[("e006", P, [48, 106, 160])]),
    "R25": dict(pres=[("e017", P, [57, 517]), ("e017", EF, [59])]),
    "R26": dict(pres=[("e018", A, [59, 355]), ("e018", "all.exact_accuracy", [376])]),
}
SAME = {"R13": "R02", "R15": "R03", "R17": "R05", "R18": "R07", "R19": "R02", "R20": "R14", "R21": "R16", "R27": "R12"}
for k, v in SAME.items():
    EV[k] = EV[v]

FQ = {"Q-I1": ("ACC", "{P-RELOAD key; no fixed string => empty}"), "Q-I2": ("ACC", "{probe/test_exact}"),
      "Q-I3": ("ACC+DEC", "{probe/test_exact} U {P-DECODE key; no fixed string}"), "Q-I4": ("ACC", "none (source-judge protocol)"),
      "Q-Z1": ("ACC", "{probe/test_exact}"), "Q-Z2": ("ACC", "{probe/test_exact}"), "Q-Z3": ("PR", "{z/eff_rank}"),
      "Q-Z4": ("TAU", "{EXP-006 pin key}"), "Q-Z5": ("DEC", "{P-DECODE key; no fixed string}"),
      "Q-Z6": ("ACC(EXP-020) / - (EXP-018)", "none; EXP-018 '-' => UNKNOWN constant")}

# per-file occurrence cache: path -> text
_cache = {}


def lines_of(path):
    if path not in _cache:
        _cache[path] = open(path, encoding="utf-8").read().split("\n")
    return _cache[path]


def occ_lines(path, key):
    return [i + 1 for i, l in enumerate(lines_of(path)) if key in l]


def occ_count(path, key):
    return sum(l.count(key) for l in lines_of(path))


evidence_files = set()
key_rows = []
for rid in sorted(const):
    c = const[rid]
    ev = EV.get(rid)
    if ev is None:
        flag(f"{rid}: no evidence spec")
        continue
    # cross-check encoded stated lines against the PREREG row text (non-"same" rows)
    if rid not in SAME:
        row_nums = {int(x) for x in re.findall(r":(\d+)", c["ev"])}
        enc_nums = {n for (_, _, ls) in ev.get("pres", []) for n in ls}
        if row_nums != enc_nums:
            flag(f"{rid}: encoded stated lines {sorted(enc_nums)} != PREREG row text {sorted(row_nums)}")
    obs = []
    for (fk, key, stated) in ev.get("pres", []):
        path = F[fk]
        evidence_files.add(path)
        ls = occ_lines(path, key)
        if not ls:
            flag(f"{rid}: key '{key}' ABSENT in {path} (stated {stated})")
            obs.append(f"{path}:ABSENT:{key}")
            continue
        status = "line_match" if set(stated) <= set(ls) else f"line_shift(stated {stated})"
        obs.extend(f"{path}:{n}:{key}" for n in ls)
        obs.append(f"[{status}]")
    chk = []
    for (fk, key, n) in ev.get("count", []):
        evidence_files.add(F[fk])
        got = occ_count(F[fk], key)
        chk.append(f"count({F[fk]},{key})={got} expect {n}")
        if got != n:
            flag(f"{rid}: count of '{key}' in {F[fk]} = {got}, constant says {n}")
    for (fk, key) in ev.get("zero", []):
        evidence_files.add(F[fk])
        got = occ_count(F[fk], key)
        chk.append(f"count({F[fk]},{key})={got} expect 0")
        if got != 0:
            flag(f"{rid}: '{key}' occurs {got}x in {F[fk]}; constant asserts 0")
    for (fk, rx) in ev.get("zero_re", []):
        evidence_files.add(F[fk])
        got = sum(len(re.findall(rx, l)) for l in lines_of(F[fk]))
        chk.append(f"count({F[fk]},re:{rx})={got} expect 0")
        if got != 0:
            flag(f"{rid}: regex '{rx}' occurs {got}x in {F[fk]}; constant asserts 0")
    if "absent_file" in ev:
        ex = os.path.exists(ev["absent_file"])
        chk.append(f"exists({ev['absent_file']})={ex} expect False")
        if ex:
            flag(f"{rid}: {ev['absent_file']} exists; constant says no design doc")
    q = c["q"]
    key_rows.append(dict(
        row_id=rid, q=q, s=c["s"], F_q=FQ[q][0], K_q=FQ[q][1], N_named=c["N"], key_status=c["status"],
        evidence_file_line_key="; ".join(obs) if obs else "(none: absence-type row)",
        absence_or_count_checks="; ".join(chk) if chk else "",
        prereg_class=c["cls"]))

# ---------------------------------------------------------------- 4. assertions on counts
key_pairs = {(r["q"], r["s"]) for r in key_rows}
exp_pairs = {(q, s) for (q, s, cl, _) in cells if kind_of(s) == "E" and cl not in ("X-PROV", "D-SEALED")}
if key_pairs != exp_pairs:
    flag(f"metric-key (q,s) set != 7.0.5 E cells minus X-PROV/D-SEALED: only_keys={sorted(key_pairs - exp_pairs)} only_cells={sorted(exp_pairs - key_pairs)}")
if len(key_rows) != 27:
    flag(f"metric key rows = {len(key_rows)}")
from collections import Counter
st = Counter(r["key_status"] for r in key_rows)
if (st["UNKNOWN"], st["NAMED-ADMITTED"], st["NAMED-OTHER"]) != (8, 19, 0):
    flag(f"key_status counts {dict(st)} != UNKNOWN 8 / NAMED-ADMITTED 19 / NAMED-OTHER 0")
cc = Counter(cl for (_, _, cl, _) in cells)
want = {"A-VERDICT": 19, "A-DESC": 23, "D-LOCKED": 5, "D-PREJUDGE": 3, "X-ABSENT": 3, "X-PROV": 2, "D-SEALED": 2, "LISTED-ONLY": 1}
if len(cells) != 58 or dict(cc) != want:
    flag(f"class counts {dict(cc)} (n={len(cells)}) != {want}")
kc = Counter(kind_of(s) for (_, s, _, _) in cells)
if dict(kc) != {"E": 31, "A": 12, "H": 15}:
    flag(f"kind counts {dict(kc)} != E31/A12/H15")
if len({(q, s) for (q, s, _, _) in cells}) != 58:
    flag("duplicate (q,s) in universe")
# class of key rows agrees with universe class
cls_of = {(q, s): cl for (q, s, cl, _) in cells}
for r in key_rows:
    if cls_of.get((r["q"], r["s"])) != r["prereg_class"]:
        flag(f"{r['row_id']}: class in constant table {r['prereg_class']} != 7.0.5 {cls_of.get((r['q'], r['s']))}")

# ---------------------------------------------------------------- 5. existence checks per E source (as_of)
PFX = {"EXP-025": "EXP-015", "EXP-012/probe": "EXP-012", "EXP-012/sealed": "EXP-012"}
FIXED_C = {"EXP-022": f"{REPD}/2026-10-23_pc1-ltf-per-loop-decode_RESULTS.md",
           "H-108 sweep": f"{REPD}/16_2026-09-02_h108-depth-sweep/summary.md"}
canon = lambda s: PFX.get(s, s)
exist_rows = []
e_sources = sorted({s for (_, s, _, _) in cells if kind_of(s) == "E"})
for s in e_sources:
    if s in ("EXP-010", "EXP-012/sealed"):
        exist_rows.append(dict(s=s, note="not checked (branch 1 X-PROV / branch 3 D-SEALED precede; sealed path never touched)"))
        continue
    p = PFX.get(s, s)
    if s == "H-108 sweep":
        wit, a, b, c_ = False, False, [], os.path.exists(FIXED_C[s])
    else:
        wit = os.path.exists(f"{EXPD}/{p}_FREEZE_WITNESS.md") or os.path.exists(f"{EXPD}/{p}_FREEZE.sha256")
        a = os.path.exists(f"{EXPD}/{p}_RESULTS.md")
        idset = aliases.get(s, {s}) | {s}
        b = [n for n, t in top_results.items() if t in idset]
        c_ = os.path.exists(FIXED_C[s]) if s in FIXED_C else False
    has_result = bool(a or b or c_)
    exist_rows.append(dict(s=s, witness=wit, result_a=a, result_b=";".join(b), result_c=c_, has_result=has_result,
                           branch5_fires=bool(wit and not has_result)))
ex = {r["s"]: r for r in exist_rows}
for s in ("EXP-020", "EXP-023", "EXP-018"):
    if not ex[s].get("branch5_fires"):
        flag(f"{s}: witness/result existence does not give branch-5 (witness={ex[s].get('witness')}, has_result={ex[s].get('has_result')})")
for s, r in ex.items():
    if r.get("branch5_fires") and s not in ("EXP-020", "EXP-023", "EXP-018"):
        flag(f"{s}: branch-5 existence pattern fires but expected class is not D-PREJUDGE")
for s in ("EXP-016", "EXP-017", "EXP-022", "EXP-025", "H-108 sweep"):
    if not ex[s].get("has_result"):
        flag(f"{s}: expected a result file at as_of, none found")

# ---------------------------------------------------------------- 6. write outputs
with open(f"{OUT}/S0r2_metric_keys.csv", "w", newline="", encoding="utf-8") as f:
    f.write(SUPER)
    w = csv.DictWriter(f, fieldnames=["row_id", "q", "s", "F_q", "K_q", "N_named", "key_status",
                                      "evidence_file_line_key", "absence_or_count_checks", "prereg_class"])
    w.writeheader()
    for r in key_rows:
        w.writerow(r)

with open(f"{OUT}/S0r2_cells_expected.csv", "w", newline="", encoding="utf-8") as f:
    f.write(SUPER)
    w = csv.writer(f)
    w.writerow(["source", "as_of", "q", "s", "kind", "expected_class", "expected_q_token", "in_metric_keys",
                "witness_present", "result_file_a", "result_file_b", "result_file_c", "branch5_existence_pattern"])
    for (q, s, cl, qt) in cells:
        r = ex.get(s, {}) if kind_of(s) == "E" else {}
        w.writerow([f"PREREG sec.7.0.5 expected class (sha {PREREG_SHA[:8]}); NOT an S1 as_of classification", AS_OF, q, s,
                    kind_of(s), cl, qt, (q, s) in key_pairs,
                    r.get("witness", ""), r.get("result_a", ""), r.get("result_b", ""), r.get("result_c", ""),
                    r.get("branch5_fires", r.get("note", ""))])

with open(f"{SNAP}/evidence_files.sha256", "w") as f:
    for p in sorted(evidence_files):
        f.write(f"{sha(p)}  {p}\n")

# snapshot sha list (all files under S0r2_snapshot/)
snap_lines = []
for dp, _, fns in os.walk(SNAP):
    for fn in sorted(fns):
        p = os.path.join(dp, fn)
        snap_lines.append(f"{sha(p)}  {os.path.relpath(p, SNAP)}")
snap_lines.sort(key=lambda l: l.split("  ", 1)[1])
with open(f"{OUT}/S0r2_snapshot.sha256", "w") as f:
    f.write("\n".join(snap_lines) + "\n")

print("as_of", AS_OF, "script_sha", SCRIPT_SHA)
print("key rows", len(key_rows), dict(st), "| cells", len(cells), dict(cc), dict(kc))
print("problems:", len(problems))
sys.exit(2 if problems else 0)
