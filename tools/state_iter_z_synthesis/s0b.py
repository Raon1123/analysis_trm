#!/usr/bin/env python3
"""S0b for 2026-10-30_state-iter-z-synthesis: S0_inventory.csv, S0_tokens.csv, S0_class_repro_check.csv.

Reads (frontmatter / status-cell / verdict-field word sets ONLY; no body, no values, no wandb, no checkpoints):
  - S0_snapshot copies (EXPERIMENTS.md Status cells, HYPOTHESES.md status cells, INDEX.jsonl verdict field,
    H-NNN.status.md frontmatter, queue/exp-dir name lists)  -- as_of state of s0.py
  - RESULTS-file YAML frontmatter fences (EXP-015/016/017, EXP-022 (c) file); H-108 summary.md has no fence
  - checkpoints/<project>/ DIRECTORY NAMES only (data-presence), one os.listdir each
Emits word sets and booleans, never sentences. Does not touch s0.py outputs (new files only).
Run: CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s0b.py
"""
import csv, hashlib, os, re, sys
import yaml

os.chdir("/home/ayp/project/trm")
PREREG = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md"
PREREG_SHA = "87fa6e7acf2ba7278c4dea1835c1c27790d1bfe519b2e966374568b300c73e1b"
OUT = "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis"
SNAP = f"{OUT}/S0_snapshot"
EXPD = "lab/21_experiments/11_lab-experiments"
REPD = "lab/23_reports/11_lab-reports"
THEORY = "lab/27_theory/11_lab-theory"
problems = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flag(m):
    problems.append(m)
    print("DISCREPANCY:", m)


assert sha(PREREG) == PREREG_SHA, "PREREG sha mismatch"
AS_OF = [l.split(": ", 1)[1] for l in open(f"{SNAP}/as_of.txt").read().split("\n") if l.startswith("as_of:")][0]
S0_SHA = sha("tools/state_iter_z_synthesis/s0.py")

# ------------------------------------------------------------------ vocab
V_TOK = ["supported", "refuted", "falsified-as-framed", "conditional", "partial"]
N_TOK = [("under-test", r"under-test"), ("not-addressed", r"not-addressed"), ("superseded", r"superseded"),
         ("unproven", r"unproven"), ("planned", r"planned"), ("in-progress", r"in-progress"),
         ("completed", r"completed"), ("draft-prereg", r"draft-prereg"), ("pre-registered", r"pre-registered(?!-unrun)"),
         ("pre-registered-unrun", r"pre-registered-unrun"), ("S6 DRAFT", r"S6 DRAFT"), ("PI 판정 대기", r"PI 판정 대기"),
         ("DRAFT", r"(?<![\w-])DRAFT(?![\w-])"), ("VERIFIED-rN", r"VERIFIED-r\d+"), ("PASS", r"(?<![\w-])PASS(?![\w-])")]
REG_STATUS = [("closed", r"(?<![\w-])closed(?![\w-])"), ("abandoned", r"abandoned")]


def vwords(txt):
    """V_tok words in a string; (set, negated_flag, qualifier_flag)."""
    got, neg, qual = set(), False, False
    for w in V_TOK:
        for m in re.finditer(rf"(?<![\w]){re.escape(w)}(?![\w])", txt):
            # 'falsified-as-framed' must not be double-counted by bare words inside it
            got.add(w)
            if re.search(r"(not|non|un)[\s-]*$", txt[max(0, m.start() - 6):m.start()].lower()):
                neg = True
            if txt[m.end():m.end() + 4].startswith("-but") or txt[m.end():m.end() + 1] == "-":
                qual = True
    # "supported" etc. appearing only inside a longer hyphen compound of another V word is still reported; harmless
    return got, neg, qual


def nwords(txt, extra=()):
    out = set()
    for name, rx in list(N_TOK) + list(extra):
        if re.search(rx, txt):
            out.add(name)
    return out


# ------------------------------------------------------------------ parse PREREG tables (expected classes, links, claim IDs)
text = open(PREREG, encoding="utf-8").read().split("\n")


def sect(sp, ep):
    s = next(i for i, l in enumerate(text) if re.match(sp, l))
    e = next(i for i in range(s + 1, len(text)) if re.match(ep, text[i]))
    return text[s:e]


cols = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
cells = []
for l in sect(r"#### 7\.0\.5", r"### 7\.1"):
    if l.startswith("| Q-"):
        c = cols(l)
        for part in c[1].split(" · "):
            s, cl = [x.strip() for x in part.split("→")]
            cells.append((c[0], s, cl))
exp_cls = {(q, s): cl for q, s, cl in cells}
E_ROWS = [cols(l)[0] for l in sect(r"#### 7\.0\.2", r"#### 7\.0\.3") if re.match(r"\| (EXP-|H-108)", l)]
H_LINKS = {}
for l in sect(r"#### 7\.0\.3", r"#### 7\.0\.4"):
    if re.match(r"\| H-", l):
        c = cols(l)
        H_LINKS[c[0]] = [x.strip() for x in c[2].split(",")]
A_ROWS = ["CL-0001", "CL-0002", "CL-0003", "CL-0004", "CL-0005", "CL-0006", "FG-0001", "FG-0002", "FG-0003", "FG-0004", "FG-0005", "FG-0006"]
H_ROWS = list(H_LINKS)
SOURCES = [(s, "E") for s in E_ROWS] + [(s, "H") for s in H_ROWS] + [(s, "A") for s in A_ROWS]
if len(SOURCES) != 40:
    flag(f"source rows = {len(SOURCES)}")
# 5.2 expected atom verdicts
ATOM_EXP = {}
for l in sect(r"### 5\.2", r"### 5\.3"):
    m = re.match(r"\| ((?:CL|FG)-\d{4}(?:\.\.\d{4})?) \| (\w+) \|", l)
    if m:
        a, v = m.groups()
        if ".." in a:
            lo, hi = a.split("..")
            for n in range(int(lo[-4:]) if False else int(a[3:7]), int(hi) + 1):
                ATOM_EXP[f"{a[:2]}-{n:04d}"] = v
        else:
            ATOM_EXP[a] = v
key_status = {}
for r in csv.DictReader(open(f"{OUT}/S0_metric_keys.csv", encoding="utf-8")):
    key_status[(r["q"], r["s"])] = r["key_status"]

# ------------------------------------------------------------------ snapshot-derived inputs
queue = [l.split("\t") for l in open(f"{SNAP}/queue_names.txt").read().split("\n") if l]
expdir = set(open(f"{SNAP}/exp_dir_names.txt").read().split("\n"))
top_results = dict(l.split("\t") for l in open(f"{SNAP}/results_top_exp_ids.tsv").read().split("\n") if l)
aliases = {}
for e in yaml.safe_load(open(f"{SNAP}/ID_ALIASES.yaml", encoding="utf-8"))["entries"]:
    aliases.setdefault(e["canonical"], {e["canonical"]})
    for a in e.get("aliases", []) or []:
        aliases[e["canonical"]].add(a["string"])

# EXPERIMENTS.md Status cell per EXP block
exp_status_cell = {}
cur = None
for l in open(f"{SNAP}/EXPERIMENTS.md", encoding="utf-8").read().split("\n"):
    m = re.match(r"## (EXP-\d{3})\b", l)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"\| \*\*Status\*\* \| (.*) \|\s*$", l)
    if m and cur and cur not in exp_status_cell:
        exp_status_cell[cur] = m.group(1)
# HYPOTHESES.md status cell per H row
hyp_cell = {}
for l in open(f"{SNAP}/HYPOTHESES.md", encoding="utf-8").read().split("\n"):
    m = re.match(r"\| \*\*(H-[0-9A-Za-z]+)\*\* \|", l)
    if m:
        c = cols(l)
        if len(c) >= 7:
            hyp_cell[m.group(1)] = c[2]
        else:
            flag(f"HYPOTHESES row {m.group(1)} has {len(c)} cells")
# atoms
import json
atoms = {}
for l in open(f"{SNAP}/INDEX.jsonl", encoding="utf-8"):
    d = json.loads(l)
    atoms[d["atom_id"]] = d


def frontmatter(path):
    """-> (dict or None, line_of_status or None)"""
    t = open(path, encoding="utf-8").read().split("\n")
    if not t or t[0].strip() != "---":
        return None, None
    try:
        e = t[1:].index("---")
    except ValueError:
        return None, None
    fm = yaml.safe_load("\n".join(t[1:e + 1]))
    sl = next((i + 1 for i, l in enumerate(t[1:e + 1], start=1) if l.startswith("status:")), None)
    return fm, sl


def terminal_nodes(node):
    """yield flattened text of terminal mappings (dicts without dict/list-of-dict children) and bare strings in lists."""
    if isinstance(node, dict):
        kids = [v for v in node.values() if isinstance(v, dict) or (isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v))]
        if not kids:
            yield " ".join([str(k) for k in node] + [" ".join(map(str, v)) if isinstance(v, list) else str(v) for v in node.values()])
        for v in node.values():
            yield from terminal_nodes(v)
    elif isinstance(node, list):
        for x in node:
            if isinstance(x, (dict, list)):
                yield from terminal_nodes(x)
            else:
                yield str(x)
    elif isinstance(node, str):
        yield node


def tier3_eligible(fm):
    """last verifier verdict (largest r) in frontmatter 'status'/'verifier'/'revision' strings is PASS."""
    best = None
    for v in terminal_nodes(fm):
        for m in re.finditer(r"r(\d+)[-\s:=]*(PASS|FAIL)", v):
            n, vd = int(m.group(1)), m.group(2)
            if best is None or n > best[0] or (n == best[0] and vd == "FAIL"):
                best = (n, vd)
    return (best is not None and best[1] == "PASS"), best


# ------------------------------------------------------------------ per-source facts
FILES = {
    "EXP-001": "EXP-001_fig12-z-iter-ablation.md", "EXP-002": "EXP-002_participation-ratio-analysis.md",
    "EXP-004": "EXP-004_h023-noz-collapse-reframe.md", "EXP-005": "EXP-005_fig1-mlp-block-ablation.md",
    "EXP-006": "EXP-006_sigma-k-tau-grid.md", "EXP-007": "EXP-007_k9-k11-cycletype-discriminator.md",
    "EXP-010": "EXP-010_fig1-clean-rerun-sigma-k-new.md", "EXP-011": "EXP-011_module-ablation-sigma-k-new.md",
    "EXP-012/probe": "EXP-012_clean-r1-mechanism-scaling-gates.md", "EXP-012/sealed": "EXP-012_clean-r1-mechanism-scaling-gates.md",
    "EXP-013": "EXP-013_fig1-ablation-multiseed-replication.md", "EXP-025": "EXP-015_looped-tf-direct-control.md",
    "EXP-016": "EXP-016_act-deep-supervision-axis.md", "EXP-017": "EXP-017_z-dynamics-latent-formation.md",
    "EXP-018": "EXP-018_separation-signal-gates.md",
}
PATHS = {k: f"{EXPD}/{v}" for k, v in FILES.items()}
PATHS.update({"EXP-020": f"{REPD}/2026-10-22_exp020-act-budget-extension-prereg.md",
              "EXP-022": f"{REPD}/2026-10-21_pc1-ltf-per-loop-decode-prereg.md",
              "EXP-023": f"{REPD}/2026-10-21_pg1-ltf-seed-replication-prereg.md",
              "H-108 sweep": f"{REPD}/2026-09-02_h108-depth-sweep-prereg.md"})
PFX = {"EXP-025": "EXP-015", "EXP-012/probe": "EXP-012", "EXP-012/sealed": "EXP-012"}
REGID = {"EXP-012/probe": "EXP-012", "EXP-012/sealed": "EXP-012", "H-108 sweep": "EXP-112"}
FIXED_C = {"EXP-022": f"{REPD}/2026-10-23_pc1-ltf-per-loop-decode_RESULTS.md",
           "H-108 sweep": f"{REPD}/16_2026-09-02_h108-depth-sweep/summary.md"}
L5 = {"EXP-016", "EXP-017", "EXP-018", "EXP-020", "EXP-023"}
LH = {"H-200", "H-201", "H-202"}
CLAIM_IDS = {"EXP-025": r"C-LTF-\d", "EXP-016": r"C-ACT-\d", "EXP-017": r"C-Z-\d",
             "EXP-022": r"(?:XT-[abc]|E2-(?:0|same|diff|U)|S2-\d+)"}
CKPT = {}
for proj in ("Sigma_k_fig12", "Sigma_k_new", "power_permutation", "EXP018_sigma_k"):
    p = f"checkpoints/{proj}"
    CKPT[proj] = sorted(os.listdir(p)) if os.path.isdir(p) else []
# data probes: list of (kind, project|None, regex)
PROBE = {
    "EXP-001": [("ckpt", "Sigma_k_fig12", r"^fig1_tf_")], "EXP-002": [("ckpt", "Sigma_k_fig12", r"^fig1_tf_")],
    "EXP-004": [("ckpt", "Sigma_k_fig12", r".")], "EXP-005": [("ckpt", "Sigma_k_fig12", r"^fig1_mlp_")],
    "EXP-006": [("ckpt", "Sigma_k_fig12", r"^(?!fig1n20_)")], "EXP-007": [("ckpt", "Sigma_k_fig12", r"^fig1d_")],
    "EXP-011": [("ckpt", "Sigma_k_new", r"^abl_"), ("queue", None, r"_abl_")],
    "EXP-012/probe": [("ckpt", "Sigma_k_new", r"^cleanr1_")],
    "EXP-013": [("ckpt", "Sigma_k_new", r"multiseed|exp013"), ("queue", None, r"multiseed|exp013")],
    "EXP-025": [("ckpt", "power_permutation", r"^pp_ltf_"), ("ckpt", "power_permutation", r"^pp_base_tf_z_iter")],
    "EXP-016": [("ckpt", "power_permutation", r"^e016_")],
    "EXP-017": [("path", None, f"{EXPD}/EXP-017_RESULTS_artifacts")],
    "EXP-018": [("ckpt", "EXP018_sigma_k", r"."), ("queue", None, r"EXP018|exp018")],
    "EXP-020": [("queue", None, r"e020_s[12]"), ("ckpt", "power_permutation", r"^e020_")],
    "EXP-022": [("path", None, FIXED_C["EXP-022"])],
    "EXP-023": [("queue", None, r"pp_ltf_.*_s[456]"), ("ckpt", "power_permutation", r"^pp_ltf_.*_s[456]")],
    "H-108 sweep": [("path", None, f"{REPD}/16_2026-09-02_h108-depth-sweep/depth_sweep.jsonl")],
}


def probe(s):
    """-> (n_ckpt, qcounts dict, n_path, data_present)"""
    nck, nqq, npath = 0, {"jobs": 0, "processing": 0, "done": 0, "failed": 0}, 0
    for kind, proj, rx in PROBE.get(s, []):
        if kind == "ckpt":
            nck += sum(1 for n in CKPT[proj] if re.search(rx, n))
        elif kind == "queue":
            for d, n in queue:
                if re.search(rx, re.sub(r"^\d+_|\.job$", "", n)):
                    nqq[d] += 1
        else:
            npath += int(os.path.exists(rx))
    return nck, nqq, npath, bool(nck or sum(nqq.values()) or npath)


evid_files = set()
inv, tokens = [], []
facts = {}
for (s, kind) in SOURCES:
    f = dict(kind=kind)
    if kind == "E":
        p = PFX.get(s, s)
        path = PATHS.get(s)
        f["path"] = path
        f["path_exists"] = bool(path and os.path.exists(path))
        f["pfx"] = p
        if s in ("EXP-010", "EXP-012/sealed"):
            f.update(wit="", wit_files="", a="", b="", c="", has_result="", canon_result="", tier3="", note="not checked (branch 1/3 precede; sealed path never touched)")
        else:
            wfs = [n for n in (f"{p}_FREEZE_WITNESS.md", f"{p}_FREEZE.sha256") if n in expdir] if s != "H-108 sweep" else []
            a = f"{p}_RESULTS.md" in expdir if s != "H-108 sweep" else False
            idset = aliases.get(s, {s}) | {s}
            b = [n for n, t in top_results.items() if t in idset] if s != "H-108 sweep" else []
            c = FIXED_C.get(s) if s in FIXED_C and os.path.exists(FIXED_C[s]) else ""
            canon = f"{EXPD}/{p}_RESULTS.md" if a else (c or (f"{REPD}/{b[0]}" if b else ""))
            f.update(wit=bool(wfs), wit_files=";".join(wfs), a=a, b=";".join(b), c=c, has_result=bool(a or b or c), canon_result=canon, note="")
            if len(b) + bool(a) + bool(c) > 1:
                f["note"] = "multiple result-rule hits: DISCREPANCY disclosed, class unchanged"
        nck, nqq, npath, dp = probe(s)
        f.update(nck=nck, nq=nqq, npath=npath, data=dp, regstat=exp_status_cell.get(REGID.get(s, s), ""))
    facts[s] = f

# ------------------------------------------------------------------ tokens
tok_for = {}  # s -> dict(V=set, Nset=set, tier=int, src=str)


def add_tok(s, tier, label, file, locator, V, N, neg=False, qual=False, note=""):
    tokens.append(dict(source_row=s, kind=facts[s]["kind"], tier=tier, tier_label=label, file=file, locator=locator,
                       V_tok="|".join(sorted(V)), N_tok="|".join(sorted(N)), negation_flag=neg, qualifier_flag=qual, note=note))


for (s, kind) in SOURCES:
    f = facts[s]
    if kind == "A":
        a = atoms.get(s)
        if not a:
            add_tok(s, 1, "atom-verdict", "S0_snapshot/INDEX.jsonl", "atom absent", set(), set(), note="atom not in INDEX")
            f["tokV"], f["in_index"] = set(), False
            continue
        v = str(a.get("verdict", ""))
        V, neg, qual = vwords(v)
        f.update(tokV=V, in_index=True, path=a.get("path", ""), verdict_raw=(v if re.fullmatch(r"[A-Za-z-]{1,30}", v) else "<non-token>"))
        f["path_exists"] = os.path.exists(f["path"]) if f["path"] else False
        add_tok(s, 1, "atom-verdict", "S0_snapshot/INDEX.jsonl", "verdict field", V, set(), neg, qual,
                note=f"verdict_raw={f['verdict_raw']}; PREREG5.2_expected={ATOM_EXP.get(s,'?')}; match={f['verdict_raw']==ATOM_EXP.get(s)}")
        if f["verdict_raw"] != ATOM_EXP.get(s):
            flag(f"{s}: INDEX verdict '{f['verdict_raw']}' != PREREG 5.2 '{ATOM_EXP.get(s)}' (S0 value wins; disclosed)")
    elif kind == "H":
        sf = f"{THEORY}/{s}.status.md"
        f["status_file"] = sf if os.path.exists(sf) else ""
        f["path"] = sf if os.path.exists(sf) else "lab/13_reference/11_registries/HYPOTHESES.md"
        f["path_exists"] = True
        cell = hyp_cell.get(s)
        # tier 4 (always recorded)
        if cell is None:
            add_tok(s, 4, "HYPOTHESES.md status cell", "S0_snapshot/HYPOTHESES.md", "row absent", set(), set(), note="no row")
            V4, N4 = set(), set()
        else:
            V4, neg, qual = vwords(cell)
            N4 = nwords(cell)
            add_tok(s, 4, "HYPOTHESES.md status cell", "S0_snapshot/HYPOTHESES.md", "Status column (col 3)", V4, N4, neg, qual)
        f.update(V4=V4, N4=N4)
        if f["status_file"]:
            fm, sl = frontmatter(sf)
            evid_files.add(sf)
            st = str(fm.get("status", "")) if fm else ""
            V2, neg, qual = vwords(st)
            N2 = nwords(st)
            add_tok(s, 2, "H-NNN.status.md frontmatter status", f"S0_snapshot/{s}.status.md", f"frontmatter status: line {sl}", V2, N2, neg, qual)
            # claim-level words co-occurring with claim IDs (terminal nodes)
            cl = set()
            for node in terminal_nodes(fm):
                if re.search(r"C-(LTF|ACT|Z)-\d", node):
                    cl |= vwords(node)[0]
            if cl:
                add_tok(s, 2, "H-NNN.status.md frontmatter claim-level", f"S0_snapshot/{s}.status.md", "terminal nodes w/ C-* id", cl, set())
            f.update(V2=V2, N2=N2)
            # LH-outside check: L5 result names cited anywhere in the status file (count only)
            if s not in LH:
                body = open(sf, encoding="utf-8").read()
                n = sum(body.count(x) for x in ("EXP-016_RESULTS", "EXP-017_RESULTS", "EXP-020_RESULTS", "EXP-023_RESULTS"))
                f["lh_violation_count"] = n
                add_tok(s, 2, "LH-check (name-string count)", f"S0_snapshot/{s}.status.md", "whole-file literal count", set(), set(), note=f"L5 result names cited: {n}")
    else:  # E
        if s in ("EXP-010", "EXP-012/sealed"):
            continue
        # tier 3
        fm = None
        f["tier3_elig"], f["tier3_best"], f["claim_V"] = False, None, set()
        rf = f["canon_result"]
        if rf and rf.endswith("_RESULTS.md"):
            fm, sl = frontmatter(rf)
            evid_files.add(rf)
            if fm is not None:
                el, best = tier3_eligible(fm)
                f["tier3_elig"], f["tier3_best"] = el, best
                allv, anyN = set(), set()
                cidrx = CLAIM_IDS.get(s)
                cv = set()
                for node in terminal_nodes(fm):
                    allv |= vwords(node)[0]
                    if cidrx and re.search(cidrx, node):
                        cv |= vwords(node)[0]
                f["claim_V"] = cv
                add_tok(s, 3, "RESULTS frontmatter", rf, f"frontmatter fence; status: line {sl}", cv, set(),
                        note=f"tier3_eligible={el}; last_verifier={best}; V_words_anywhere_in_frontmatter={'|'.join(sorted(allv)) or '-'}; claim_id_cooccurrence_V={'|'.join(sorted(cv)) or '-'}")
            else:
                add_tok(s, 3, "RESULTS frontmatter", rf, "no YAML fence", set(), set(), note="no frontmatter")
        elif rf:
            add_tok(s, 3, "result file (not *_RESULTS.md)", rf, "no YAML fence read", set(), set(), note="tier-3 ineligible by name")
        # tier 5
        cell = f["regstat"]
        N5 = nwords(cell, extra=[("partial(progress)", r"(?<![\w-])partial(?![\w-])")] + REG_STATUS)
        add_tok(s, 5, "EXPERIMENTS.md Status cell", "S0_snapshot/EXPERIMENTS.md", f"## {REGID.get(s,s)} Status row", set(), N5,
                note="" if cell else "no Status row found")
        f["N5"] = N5

# ------------------------------------------------------------------ branch evaluation (S0 reproduction, not S1)
XABS_WORDS = {"planned", "pre-registered-unrun", "draft-prereg"}


def e_xabsent(s):
    f = facts[s]
    return (not f["data"]) or (bool(f.get("N5", set()) & XABS_WORDS) and not f["data"])


def classify(q, s):
    kind = facts[s]["kind"]
    f = facts[s]
    if s == "EXP-010":
        return "X-PROV", "branch1"
    if s == "CL-0006":
        return "LISTED-ONLY", "branch2"
    if s == "EXP-012/sealed":
        return "D-SEALED", "branch3"
    if kind == "E":
        if e_xabsent(s):
            return "X-ABSENT", "branch4 (data absent)"
    if kind == "H":
        links = [x for x in H_LINKS[s]]
        if links and all(e_xabsent(x) for x in links):
            return "X-ABSENT", "branch4 (all linked EXP rows absent)"
    if kind == "E" and f.get("wit") and not f.get("has_result"):
        return "D-PREJUDGE", "branch5"
    if key_status.get((q, s)) == "NAMED-OTHER":
        return "X-OUT", "branch6"
    if (kind == "E" and s in L5) or (kind == "H" and s in LH):
        return "D-LOCKED", "branch7"
    if kind == "A":
        if f.get("in_index") and (f["tokV"]):
            return "A-VERDICT", "branch8 atom verdict in V_tok"
    elif kind == "H":
        if s not in LH and f.get("lh_violation_count", 0) > 0:
            return "UNCOVERED", "branch10 (LH rule: status file cites L5 result name)"
        tokV = f["V2"] if f["status_file"] else f["V4"]
        tokN = f["N2"] if f["status_file"] else f["N4"]
        if tokV:
            return "A-VERDICT", "branch8 H token tier " + ("2" if f["status_file"] else "4")
        if tokN:
            return "A-DESC", "branch9 H N_tok"
        return "UNCOVERED", "branch10 (no V/N word)"
    else:
        if f.get("tier3_elig") and f.get("claim_V"):
            return "A-VERDICT", "branch8 E tier3"
        if f["data"]:
            return "A-DESC", "branch9 E data present, branch8 false"
    return "UNCOVERED", "branch10"


repro = []
for (q, s, cl) in cells:
    got, why = classify(q, s)
    repro.append(dict(q=q, s=s, expected_class=cl, as_of_class=got, rule=why, match=(got == cl)))
mism = [r for r in repro if not r["match"]]

# ------------------------------------------------------------------ write
os.makedirs(OUT, exist_ok=True)
for fn in ("S0_inventory.csv", "S0_tokens.csv", "S0_class_repro_check.csv"):
    if os.path.exists(f"{OUT}/{fn}"):
        print("refuse to overwrite", fn)
        sys.exit(3)

qs_of = {}
for q, s, cl in cells:
    qs_of.setdefault(s, []).append(f"{q}:{cl}")
with open(f"{OUT}/S0_inventory.csv", "w", newline="", encoding="utf-8") as fh:
    fh.write(f"# as_of: {AS_OF}\n")
    w = csv.writer(fh)
    w.writerow(["source_row", "kind", "canonical_path", "canonical_path_exists", "witness_present", "witness_files",
                "result_rule_a", "result_rule_b", "result_rule_c", "result_present", "canonical_result_file",
                "tier3_eligible_(*_RESULTS_w/_last_verifier_PASS)", "registry_exp_id", "registry_status_words_tier5",
                "ckpt_dirs_matched", "queue_jobs", "queue_processing", "queue_done", "queue_failed", "path_probe_hits",
                "data_present", "cells_(q:expected_class)", "note"])
    for (s, kind) in SOURCES:
        f = facts[s]
        if kind == "E":
            w.writerow([s, kind, f["path"], f["path_exists"], f["wit"], f["wit_files"], f["a"], f["b"], f["c"], f["has_result"],
                        f["canon_result"], f.get("tier3_elig", ""), REGID.get(s, s), "|".join(sorted(f.get("N5", set()))) if "N5" in f else "",
                        f["nck"], f["nq"]["jobs"], f["nq"]["processing"], f["nq"]["done"], f["nq"]["failed"], f["npath"], f["data"],
                        ";".join(qs_of[s]), f["note"] + ("" if f["path_exists"] else "; design doc absent on disk")])
        else:
            w.writerow([s, kind, f.get("path", ""), f.get("path_exists", ""), "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                        ";".join(qs_of[s]), ("tier2 file present" if f.get("status_file") else "tier4 (HYPOTHESES.md)") if kind == "H" else ""])
with open(f"{OUT}/S0_tokens.csv", "w", newline="", encoding="utf-8") as fh:
    fh.write(f"# as_of: {AS_OF}\n")
    w = csv.DictWriter(fh, fieldnames=list(tokens[0].keys()))
    w.writeheader()
    w.writerows(tokens)
with open(f"{OUT}/S0_class_repro_check.csv", "w", newline="", encoding="utf-8") as fh:
    fh.write(f"# as_of: {AS_OF}; S0 reproduction of PREREG 7.1 from names/words only (NOT S1_cells.csv); mismatches are reported, never fixed\n")
    w = csv.DictWriter(fh, fieldnames=["q", "s", "expected_class", "as_of_class", "rule", "match"])
    w.writeheader()
    w.writerows(repro)
ev = sorted(evid_files)
with open(f"{OUT}/S0b_evidence_files.sha256", "w") as fh:
    for p in ev:
        fh.write(f"{sha(p)}  {p}\n")

from collections import Counter
print("inventory rows", len(SOURCES), "token rows", len(tokens), "cells", len(repro))
print("as_of class counts", dict(Counter(r["as_of_class"] for r in repro)))
print("expected counts   ", dict(Counter(r["expected_class"] for r in repro)))
print("mismatch cells:", [(r["q"], r["s"], r["expected_class"], r["as_of_class"], r["rule"]) for r in mism])
print("s0.py sha", S0_SHA)
sys.exit(0)
