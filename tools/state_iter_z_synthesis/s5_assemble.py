#!/usr/bin/env python3
"""S5 assembly for the state-iter-z synthesis (PREREG r6, sha 87fa6e7a...).

Deterministic string assembly. No model call, no GPU, no new number or verdict word.
Reads only the inputs listed in INPUTS below (+ ledger read-only) and writes three files:
  1. <D>/S5_assembly/QS1_table.md
  2. <D>/S5_assembly/paper/state_iter_z_paper_EN.tex
  3. lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_RESULTS.md
Every embedded source block is bracketed by S5:BEGIN/S5:END markers and re-checked by sha256
after the file is written (re-read from disk).

Run:  CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s5_assemble.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "lab/23_reports/11_lab-reports"
D = REPORTS / "2026-10-30_state-iter-z-synthesis"
A = D / "S5_assembly"
PREREG = REPORTS / "2026-10-30_state-iter-z-synthesis_PREREG.md"
LEDGER_PATH = ROOT / "lab/38_api-batch/ledger.jsonl"
OUT_QS1 = A / "QS1_table.md"
OUT_TEX = A / "paper/state_iter_z_paper_EN.tex"
OUT_RESULTS = REPORTS / "2026-10-30_state-iter-z-synthesis_RESULTS.md"

PREREG_SHA_EXPECTED = "87fa6e7acf2ba7278c4dea1835c1c27790d1bfe519b2e966374568b300c73e1b"
AS_OF_EXPECTED = "2026-10-30T13:51:29Z"
REP_IDS = ["i2", "i3", "i4", "i5", "z1", "z2", "z3", "z4", "z5", "z6"]
PAPER_IDS = ["paper-en-iter", "paper-en-z-a", "paper-en-z-b"]
LEGACY_NOTE = "legacy, 신규 clean 코호트와 비교 불가"
NO_SRC = "출처 없음"
NO_QUOTE = "어떤 문장의 근거로도 인용 금지"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shas(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rd(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def read_csv_skip_comments(p: Path):
    lines = [ln for ln in rd(p).splitlines(keepends=True) if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def die(msg: str):
    print("S5 ABORT:", msg, file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------- inputs
INPUTS: dict[str, Path] = {}
INPUTS["PREREG"] = PREREG
for f in ["S0r2_inventory.csv", "S0r2_tokens.csv", "S0r2_cells_expected.csv", "S0r2_snapshot/as_of.txt",
          "S1r2/S1_cells.csv", "S1r2/S1_table.csv", "S1r2/S1_access_log.csv", "S1r2/S1_manifest.sha256",
          "S2_check_map.csv", "S3_check_reduce.csv", "S4_draft/DESC_iteration.md", "S4_draft/DESC_z.md",
          "S4_draft/VERIFY-V2-desc-r7.return.yaml", "VERIFY-V1-r2-V0.return.yaml", "VERIFY-V0-r2.return.yaml",
          "S5_assembly/VERIFY-V2-r3.return.yaml"]:
    INPUTS[f] = D / f
for i in REP_IDS:
    INPUTS[f"S5_assembly/reports/rep-q-{i}.md"] = A / "reports" / f"rep-q-{i}.md"
for pid in PAPER_IDS:
    INPUTS[f"S5_assembly/paper/{pid}.tex"] = A / "paper" / f"{pid}.tex"
for k, p in INPUTS.items():
    if not p.is_file():
        die(f"missing input {k}: {p}")
if sha(PREREG) != PREREG_SHA_EXPECTED:
    die("PREREG sha differs from r6 sha")
ledger_bytes = LEDGER_PATH.read_bytes()
ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
ledger_rows = [json.loads(l) for l in ledger_bytes.decode("utf-8").splitlines() if l.strip()]
SCRIPT_SHA = sha(Path(__file__).resolve())

as_of_line = rd(D / "S0r2_snapshot/as_of.txt").splitlines()[0]
if as_of_line.split(":", 1)[1].strip() != AS_OF_EXPECTED:
    die(f"S0r2 as_of is {as_of_line!r}")

# S1r2 manifest -> the four S1r2 hashes, re-verified against disk
manifest = {}
for ln in rd(D / "S1r2/S1_manifest.sha256").splitlines():
    m = re.match(r"^([0-9a-f]{64})\s+(\S+)$", ln)
    if m:
        manifest[m.group(2)] = m.group(1)
S1_HASH_KEYS = {
    "s1.py": "tools/state_iter_z_synthesis/s1.py",
    "S1_table.csv": "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S1r2/S1_table.csv",
    "S1_cells.csv": "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S1r2/S1_cells.csv",
    "S1_access_log.csv": "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S1r2/S1_access_log.csv",
}
S1_HASHES = {}
for short, key in S1_HASH_KEYS.items():
    if key not in manifest:
        die(f"manifest lacks {key}")
    disk = sha(ROOT / key)
    if disk != manifest[key]:
        die(f"S1r2 manifest sha != disk for {key}")
    S1_HASHES[short] = disk

inv = read_csv_skip_comments(D / "S0r2_inventory.csv")
tok = read_csv_skip_comments(D / "S0r2_tokens.csv")
cells = list(csv.DictReader(open(D / "S1r2/S1_cells.csv", newline="", encoding="utf-8")))
map_check = list(csv.DictReader(open(D / "S2_check_map.csv", newline="", encoding="utf-8")))
red_check = list(csv.DictReader(open(D / "S3_check_reduce.csv", newline="", encoding="utf-8")))

# ----------------------------------------------------------------------------- Q-S1 table
if len(inv) != 40:
    ROWS_REASON = f"{len(inv)} 행 (40이 아님): S0r2_inventory.csv 행 수가 PREREG §7.0 소스 행 수와 다르다"
else:
    ROWS_REASON = "40 행 = S0r2_inventory.csv 데이터 행 수 = PREREG §7.0 소스 행 수(§7.4: X-PROV·X-ABSENT·D-SEALED·LISTED-ONLY 행 포함)"
src_in_cells = {c["s"] for c in cells}
inv_names = [r["source_row"] for r in inv]
if set(inv_names) != src_in_cells:
    die("inventory source_row set != S1_cells s set")

cells_by_s: dict[str, list] = defaultdict(list)
for c in cells:
    cells_by_s[c["s"]].append(c)
tok_by_s: dict[str, list] = defaultdict(list)
for t in tok:
    tok_by_s[t["source_row"]].append(t)

STR_XPROV = "provenance·의존 전용(증거 셀 아님)"
STR_SEALED = "sealed 종점 — 이 합성에서 사용 안 함"
STR_PREJ = "판정 전 — 상태만"
STR_XABS = "데이터 없음(미실행)"


def tok_cell(s: str) -> str:
    classes = {c["s1_class"] for c in cells_by_s[s]}
    rows = tok_by_s.get(s, [])

    def fmt(t):
        v, n = t["V_tok"], t["N_tok"]
        head = f"tier {t['tier']} ({t['tier_label']}) {t['file']} · {t['locator']}"
        if v == "" and n == "":
            note = t["note"]
            return f"{head}: (빈 칸)" + (f"; note={note}" if note else "")
        parts = []
        if v:
            parts.append(f"V_tok={v}")
        if n:
            parts.append(f"N_tok={n}")
        return f"{head}: " + "; ".join(parts) + (" [negation_flag]" if t["negation_flag"] == "True" else "") + \
            (" [qualifier_flag]" if t["qualifier_flag"] == "True" else "")

    if "D-PREJUDGE" in classes:
        extra = "; ".join(fmt(t) for t in rows if t["N_tok"] or t["V_tok"])
        return esc(f"{STR_PREJ}" + (f" (상태 낱말 원문: {extra})" if extra else ""))
    if "X-PROV" in classes:
        return esc(STR_XPROV)
    if "D-SEALED" in classes:
        return esc(STR_SEALED)
    if "LISTED-ONLY" in classes:
        c = cells_by_s[s][0]
        return esc(f"verdict={c['token_V']} (tier {c['token_tier']}, {c['token_file']} · {c['token_locator']}); "
                   f"verdict_scope(원문, 소스 원자 자체의 텍스트): {c['atom_verdict_scope']}; {NO_QUOTE}")
    if not rows:
        return (esc(STR_XABS) + f" (토큰 행 {NO_SRC})") if "X-ABSENT" in classes else NO_SRC
    out = "<br>".join(esc(fmt(t)) for t in rows)
    if "X-ABSENT" in classes:
        out = esc(STR_XABS) + "<br>" + out
    return out + f"<br>[{NO_QUOTE}]"


def result_cell(r) -> str:
    if r["result_present"] == "" and r["result_rule_a"] == "":
        return NO_SRC
    f = r["canonical_result_file"]
    rules = f"(a)={r['result_rule_a'] or '∅'} · (b)={r['result_rule_b'] or '∅'} · (c)={r['result_rule_c'] or '∅'}"
    return esc(f"result_present={r['result_present']}; {rules}" + (f"; file={f}" if f else ""))


def witness_cell(r) -> str:
    if r["witness_present"] == "":
        return NO_SRC
    return esc(f"witness_present={r['witness_present']}" + (f"; {r['witness_files']}" if r["witness_files"] else ""))


def queue_cell(r):
    q = [r["queue_jobs"], r["queue_processing"], r["queue_done"], r["queue_failed"]]
    if any(x == "" for x in q):
        return NO_SRC, NO_SRC, NO_SRC
    j, p, d, f = map(int, q)
    return f"{j}/{p}/{d}/{f}", f"{d}/{j + p + d + f}", (r["ckpt_dirs_matched"] or NO_SRC)


def class_cell(s: str) -> str:
    items = []
    for c in cells_by_s[s]:
        t = f"{c['q']}:{c['s0_confirmed_class']}"
        if c["marker"]:
            t += f"[{c['marker']}]"
        if c["class_match"] != "True":
            t += f" (§7.0.5 예상 {c['expected_class']} ≠ S0r2 {c['s0_confirmed_class']}; reason: {c['reason']})"
        items.append(t)
    return esc("; ".join(items))


def cohort_cell(s: str) -> str:
    notes = sorted({c["cohort_note"] for c in cells_by_s[s] if c["cohort_note"]})
    return esc("; ".join(notes)) if notes else "—"


def build_qs1() -> str:
    hdr = ["소스 행", "kind", "queue jobs/processing/done/failed", "queue done/(jobs+processing+done+failed)",
           "ckpt_dirs_matched", "결과 파일", "동결 witness", f"판정/상태 토큰 ({NO_QUOTE})",
           "§7.1 클래스 (셀별, S1r2 s0_confirmed_class)", "cohort_note", "열린 사용자 결정", "S0r2 note"]
    rows = []
    for r in inv:
        s = r["source_row"]
        qc, ratio, ck = queue_cell(r)
        rows.append([s, r["kind"], qc, ratio, ck, result_cell(r), witness_cell(r), tok_cell(s), class_cell(s),
                     cohort_cell(s), NO_SRC, esc(r["note"]) if r["note"] else "—"])
    out = []
    out.append(f"**as_of:** {AS_OF_EXPECTED} (S0r2_snapshot/as_of.txt: `{as_of_line}`). 이 표는 as_of 이후 디스크 변화를 반영하지 않는다.\n")
    out.append("**행 수:** " + ROWS_REASON + ".\n")
    out.append("**열별 출처 파일** (모든 경로는 `lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/` 하위):\n")
    out.append("- 소스 행·kind·queue 열·ckpt_dirs_matched·결과 파일·동결 witness·S0r2 note: `S0r2_inventory.csv` "
               "(열 이름: `source_row`, `kind`, `queue_jobs`/`queue_processing`/`queue_done`/`queue_failed`, "
               "`ckpt_dirs_matched`, `result_present`+`result_rule_a/b/c`+`canonical_result_file`, "
               "`witness_present`+`witness_files`, `note`). 인벤토리 칸이 공란이면 `" + NO_SRC + "`.")
    out.append("- 토큰: `S0r2_tokens.csv` (행마다 `tier (tier_label) file · locator`와 `V_tok`/`N_tok` 원문; "
               "`|`는 소스 원문의 다중 낱말 구분자이며 표 이스케이프는 `\\|`). 토큰 행 자체가 없으면 `" + NO_SRC + "`, "
               "행은 있으나 두 칸이 비면 `(빈 칸)`과 `note` 열. D-PREJUDGE·X-PROV·D-SEALED 행의 칸은 PREREG §7.1/§7.4가 정한 고정 문구. "
               "CL-0006 칸의 verdict와 verdict_scope 원문은 `S1r2/S1_cells.csv` (`token_V`, `atom_verdict_scope`)이며, "
               "scope 원문 안의 숫자는 소스 원자 자체의 텍스트이지 S1 값이 아니다.")
    out.append("- §7.1 클래스·마커·cohort_note: `S1r2/S1_cells.csv` (`s0_confirmed_class`, `marker`, `cohort_note`, `class_match`). "
               "cohort_note는 PREREG §7.4 비차단 주석/U-21이 지적한 칸을 S1_cells의 `cohort_note` 열로 채운 것.")
    out.append(f"- 열린 사용자 결정: 스냅샷·인벤토리·토큰·S1_cells 어디에도 소스 행별 열린-결정 필드가 없다. 전 행 `{NO_SRC}` "
               "(키워드 검색으로 합성하지 않음).")
    out.append("- 수치: run 수(queue 열·ckpt_dirs_matched)와 날짜(as_of), 그리고 CL-0006 verdict_scope 원문에 들어 있는 source-atom 숫자뿐(S1 값 아님). 메트릭 값 칸 없음. "
               "`queue done/(…)` 열은 `queue_done / (queue_jobs + queue_processing + queue_done + queue_failed)` 의 파생 표기다.\n")
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    out.append("클래스별 셀 수 (S1_cells 셀 기준, 개수만; 소스 행 수 아님): " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(c["s0_confirmed_class"] for c in cells).items())) + f" (셀 합 {len(cells)}).")
    return "\n".join(out) + "\n"


QS1_BODY = build_qs1()
QS1_FRONT = (
    "---\n"
    "exp_id: UNASSIGNED   # 합성(synthesis) 문서; 레지스트리 번호 미배정 (PREREG §13 O-1)\n"
    "slug: state-iter-z-synthesis\n"
    "doc_type: s5-qs1-status-table\n"
    "status: DRAFT\n"
    "parent_results: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_RESULTS.md\n"
    f"parent_prereg_sha256: {PREREG_SHA_EXPECTED}\n"
    f"as_of: {AS_OF_EXPECTED}\n"
    f"generator: tools/state_iter_z_synthesis/s5_assemble.py   # sha256_at_run {SCRIPT_SHA}\n"
    "---\n\n"
)

# ----------------------------------------------------------------------------- helpers for embedding
FENCE = re.compile(r"^\s*```")
HEAD = re.compile(r"^(#{1,5}) ")


def split_keep(text: str):
    return text.splitlines(keepends=True)


def demote(text: str) -> str:
    out, infence = [], False
    for ln in split_keep(text):
        if FENCE.match(ln):
            infence = not infence
        elif not infence and HEAD.match(ln):
            ln = "#" + ln
        out.append(ln)
    return "".join(out)


def undemote(text: str) -> str:
    out, infence = [], False
    for ln in split_keep(text):
        if FENCE.match(ln):
            infence = not infence
        elif not infence and re.match(r"^#{2,6} ", ln):
            ln = ln[1:]
        out.append(ln)
    return "".join(out)


def has_h6(text: str) -> bool:
    infence = False
    for ln in split_keep(text):
        if FENCE.match(ln):
            infence = not infence
        elif not infence and re.match(r"^#{6} ", ln):
            return True
    return False


def slice_between(text: str, start_pat: str, stop_pat: str):
    """lines from the line matching start_pat (inclusive) to the next line matching stop_pat (exclusive)"""
    ls = split_keep(text)
    a = next(i for i, l in enumerate(ls) if re.match(start_pat, l))
    b = next(i for i in range(a + 1, len(ls)) if re.match(stop_pat, ls[i]))
    return "".join(ls[a:b]), a + 1, b  # 1-based inclusive line range a+1..b


def frontmatter_split(text: str):
    ls = split_keep(text)
    if not ls or ls[0].rstrip("\n") != "---":
        die("no frontmatter")
    end = next(i for i in range(1, len(ls)) if ls[i].rstrip("\n") == "---")
    return "".join(ls[: end + 1]), "".join(ls[end + 1:])


def wrap_full_desc(src: str) -> str:
    fm, body = frontmatter_split(src)
    if has_h6(body):
        die("H6 heading in DESC body")
    return "```yaml\n" + fm + "```\n" + demote(body)


def unwrap_full_desc(emb: str) -> str:
    assert emb.startswith("```yaml\n")
    rest = emb[len("```yaml\n"):]
    i = rest.index("---\n```\n", 4)  # closing frontmatter + fence
    fm = rest[: i + len("---\n")]
    body = rest[i + len("---\n```\n"):]
    return fm + undemote(body)


def block(bid: str, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    return f"<!-- S5:BEGIN {bid} -->\n{text}<!-- S5:END {bid} -->\n"


def extract_block(doc: str, bid: str) -> str:
    m = re.search(rf"<!-- S5:BEGIN {re.escape(bid)} -->\n(.*?)<!-- S5:END {re.escape(bid)} -->\n", doc, re.S)
    if not m:
        die(f"block {bid} not found")
    return m.group(1)


# ----------------------------------------------------------------------------- source texts
desc_iter = rd(D / "S4_draft/DESC_iteration.md")
desc_z = rd(D / "S4_draft/DESC_z.md")
rep_txt = {i: rd(A / "reports" / f"rep-q-{i}.md") for i in REP_IDS}
paper_txt = {p: rd(A / "paper" / f"{p}.tex") for p in PAPER_IDS}
for p, t in paper_txt.items():
    if t.startswith("---"):
        die(f"{p} begins with a frontmatter fence")
    if "\\subsection" in t.split("\n", 1)[0] and p == "paper-en-z-b":
        die("z-b must have no \\subsection")

sym_iter_src, sym_iter_a, sym_iter_b = slice_between(desc_iter, r"## 0\. 기호", r"## ")
sym_z_src, sym_z_a, sym_z_b = slice_between(desc_z, r"## 0\. 기호", r"## ")
qi1_src, qi1_a, qi1_b = slice_between(desc_iter, r"## Q-I1 ", r"## Q-I2 ")
for nm, t in [("sym_iter", sym_iter_src), ("sym_z", sym_z_src), ("qi1", qi1_src)]:
    if any(FENCE.match(l) for l in split_keep(t)):
        die(f"fence inside slice {nm}")
qz7_head = next(l for l in split_keep(desc_z) if l.startswith("## Q-Z7 "))
qz7_title = qz7_head[len("## Q-Z7 — "):].strip()
qz7_cell = [c for c in cells if c["q"] == "Q-Z7"][0]
if qz7_cell["s1_class"] != "LISTED-ONLY":
    die("Q-Z7 cell not LISTED-ONLY")


def ledger_summary():
    plan_by_hash, planned_rows = {}, []
    for r in ledger_rows:
        if r["event"] == "planned":
            planned_rows.append(r)
            plan_by_hash[r["manifest_sha256"]] = r
            if r.get("plan_hash"):
                plan_by_hash[r["plan_hash"]] = r
    per = defaultdict(lambda: dict(cost=0.0, n_cost=0, batch_ids=[], approvals=[], models=set(), plan_hashes=set(),
                                   manifests=set(), planned=None, n_planned=0, errors=[]))
    for r in ledger_rows:
        m = r.get("manifest_sha256")
        pl = plan_by_hash.get(m)
        bd = pl["batch_dir"] if pl else r.get("batch_dir")
        if bd is None:
            continue
        e = per[bd]
        if r["event"] == "planned":
            e["planned"] = r
            e["n_planned"] += 1
            e["manifests"].add(r["manifest_sha256"])
            if r.get("plan_hash"):
                e["plan_hashes"].add(r["plan_hash"])
        elif r["event"] in ("collected", "sync_done"):
            e["cost"] += r["cost_est_usd"]
            e["n_cost"] += 1
            if r["event"] == "sync_done":
                e["models"].add(r["model_returned"])
        elif r["event"] == "submitted":
            e["batch_ids"].append(r["batch_id"])
        elif r["event"] == "approved":
            e["approvals"].append((r.get("approval_id") or ("approval.ts=" + r.get("approval", {}).get("ts", r["ts"])), r.get("lane") or "?"))
        elif r["event"] == "error":
            e["errors"].append((r.get("custom_id"), r.get("error"), r["ts"]))
    return per


LEDGER = ledger_summary()
siz_dirs = sorted(d for d in LEDGER if "2026-10-30_siz-" in d)
other_dirs = sorted(d for d in LEDGER if d not in siz_dirs)
siz_total = sum(LEDGER[d]["cost"] for d in siz_dirs)
all_total = sum(LEDGER[d]["cost"] for d in LEDGER)
dangling = [r for r in ledger_rows if r["event"] in ("collected", "sync_done")
            and r["manifest_sha256"] not in {k for k in (set(x["manifest_sha256"] for x in ledger_rows if x["event"] == "planned")
                                                         | set(x.get("plan_hash") for x in ledger_rows if x["event"] == "planned"
                                                               and x.get("plan_hash")))}]
if dangling:
    die("cost rows with no planned row")
ledger_cost_rows = [r for r in ledger_rows if r["event"] in ("collected", "sync_done")]
if abs(sum(r["cost_est_usd"] for r in ledger_cost_rows) - all_total) > 1e-9:
    die("ledger total mismatch")
unlisted_errors = [r for r in ledger_rows if r["event"] == "error"]


def pstr(p: Path) -> str:
    return f"`{rel(p)}`"


# ----------------------------------------------------------------------------- paper tex (2)
TEX_HDR = r"""% =====================================================================================
% DRAFT -- fragment concatenation, NOT a compiled or reviewed paper.
% This file is the verbatim concatenation of three English paper pieces (V2 r3 reviewed; sha256 differences
% between the reviewed and the concatenated files are listed in the parent report, section 5.4):
%   (1) paper-en-iter.tex   (iteration axis, one \subsection)
%   (2) paper-en-z-a.tex    (z axis, Q-Z1..Q-Z3, one \subsection)
%   (3) paper-en-z-b.tex    (z axis, Q-Z4..Q-Z6, NO \subsection of its own)
% No Korean counterpart (ko sub-text) exists yet.
% Every sentence in these pieces is API-model output assembled by string operations; none of it
% is evidence. Numbers trace to S1r2/S1_table.csv rows; verdict words are inherited tokens only.
% exp_id: UNASSIGNED (synthesis document; see PREREG section 13 O-1)
% Parent report: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_RESULTS.md
% Regenerate:  CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s5_assemble.py
% Piece boundaries are the "BEGIN piece" / "END piece" comment lines below (sha256 of each piece
% is on its BEGIN line).
% =====================================================================================
"""


def build_tex() -> str:
    parts = [TEX_HDR]
    for i, pid in enumerate(PAPER_IDS, 1):
        t = paper_txt[pid]
        if not t.endswith("\n"):
            t += "\n"
        parts.append(f"\n% ==== BEGIN piece {i}/3: {pid}.tex  sha256={sha(A / 'paper' / (pid + '.tex'))} ====\n")
        parts.append(t)
        parts.append(f"% ==== END piece {i}/3: {pid}.tex ====\n")
    parts.append("\n% ==== assembly footer (inputs; script sha256 at run: " + SCRIPT_SHA + ") ====\n")
    for pid in PAPER_IDS:
        parts.append(f"% input {rel(A / 'paper' / (pid + '.tex'))}  {sha(A / 'paper' / (pid + '.tex'))}\n")
    return "".join(parts)


def nsub(t: str) -> int:
    return len(re.findall(r"^\\subsection\{", t, re.M))


TEX = build_tex()
if nsub(TEX) != 2:
    die(f"expected 2 \\subsection, found {nsub(TEX)}")


def check_tex(doc: str):
    res = []
    for i, pid in enumerate(PAPER_IDS, 1):
        src = rd(A / "paper" / f"{pid}.tex")
        if not src.endswith("\n"):
            src += "\n"
        m = re.search(rf"% ==== BEGIN piece {i}/3: {re.escape(pid)}\.tex[^\n]*====\n(.*?)% ==== END piece {i}/3: {re.escape(pid)}\.tex ====\n", doc, re.S)
        got = m.group(1) if m else None
        res.append((pid, sha(A / "paper" / f"{pid}.tex"), shas(got) if got is not None else None, got == src))
    return res


# ----------------------------------------------------------------------------- V2 r3 targets vs disk
v2r3 = rd(D / "S5_assembly/VERIFY-V2-r3.return.yaml")
tblock = re.search(r"^targets_sha256:\n((?:  .+\n)+)", v2r3, re.M)
targets = dict(re.findall(r"^  ([\w.\-]+): ([0-9a-f]{64})$", tblock.group(1), re.M)) if tblock else {}
sha_cmp = []
for i in REP_IDS:
    n = f"rep-q-{i}"
    sha_cmp.append((n, targets.get(n), sha(A / "reports" / f"{n}.md")))
for pid in PAPER_IDS:
    sha_cmp.append((pid, targets.get(pid), sha(A / "paper" / f"{pid}.tex")))
differs = [n for n, t, d in sha_cmp if t != d]

# ----------------------------------------------------------------------------- RESULTS assembly
SEC = {}  # block id -> (source description, expected embedded text, kind)
EXPECT: dict[str, tuple[str, str, str]] = {}  # id -> (source path/range, expected text, description)


def reg(bid: str, src_desc: str, text: str, how: str):
    EXPECT[bid] = (src_desc, text if text.endswith("\n") else text + "\n", how)
    return block(bid, text)


def keys_unknown():
    return [(c["q"], c["s"], c["s1_class"]) for c in cells if c["key_status"] == "UNKNOWN"]


def cell_list(cls: str):
    return [f"{c['q']}:{c['s']}" for c in cells if c["s0_confirmed_class"] == cls]


def build_results() -> str:
    L: list[str] = []
    w = L.append
    fm_cmp = "; ".join(f"{n}" for n in differs) if differs else "없음"
    w("---")
    w("exp_id: UNASSIGNED          # 합성(synthesis) 문서; 레지스트리 번호 미배정(PREREG §13 O-1, 2026-10-30 사용자 결정). 자기 배정 금지")
    w("slug: state-iter-z-synthesis")
    w("doc_type: results-draft-assembled")
    w("status: DRAFT")
    w("language: ko (부록 B 일부·논문 조각은 영문 원문)")
    w("prereg: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md")
    w(f"prereg_sha256: {PREREG_SHA_EXPECTED}   # r6")
    w(f"s0_of_record: \"S0r2, as_of {AS_OF_EXPECTED}\"")
    w("s1r2_hashes:                 # S1r2/S1_manifest.sha256, 디스크와 재대조함")
    for k, v in S1_HASHES.items():
        w(f"  {k}: {v}")
    w("verification_tokens:")
    w("  V1: V1-PASS-S1r2            # lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/VERIFY-V1-r2-V0.return.yaml (verdicts.G-V1)")
    w("  V0: V0-PASS-r2              # lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/VERIFY-V0-r2.return.yaml (verdict)")
    w("  V2_desc: V2-PASS-desc (r7)  # lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S4_draft/VERIFY-V2-desc-r7.return.yaml (overall)")
    w("  V2_r3: \"9/10 섹션 PASS + 2건 결정론적 수정 후 조립\"   # S5_assembly/VERIFY-V2-r3.return.yaml: overall FAIL(blocking B-r3-1 rep-q-i3, B-r3-2 paper-en-iter); 나머지 PASS; 수정본에 대한 별도 verifier return은 이 디렉터리에 없음")
    w(f"v2_r3_target_sha_differs_on_disk: [{fm_cmp}]   # 아래 §5.4")
    w("verifier_verdict_on_this_assembly: ~   # 이 조립물은 아직 verifier를 거치지 않았다. 판정 토큰은 orchestrator가 기입")
    w("evidence_status: \"DRAFT. 모든 API 출력은 증거가 아니다\"")
    w(f"generator: tools/state_iter_z_synthesis/s5_assemble.py   # sha256_at_run {SCRIPT_SHA}")
    w("---")
    w("")
    w("# 상태·iteration·z 종합 — 결과 보고서 (조립본, DRAFT)")
    w("")
    w("## 읽는 법")
    w("")
    w("- 이 문서는 **조립본(DRAFT)** 이다. 새 주장·새 수치·새 판정어를 만들지 않았고, 아래 각 절은 이미 V2를 거친 원문 파일을 문자 그대로 옮긴 블록이다. "
      "블록은 `<!-- S5:BEGIN id -->` … `<!-- S5:END id -->` 마커 사이에 있고, 마커 밖의 문장·표는 조립 스크립트가 쓴 편집 주 또는 S0r2/S1r2/ledger에서 결정론적으로 뽑은 표다. "
      "블록 ↔ 원문의 sha256 일치는 §5.5에 있다.")
    w("- 수치 문장은 문단 끝 〔row_id: …〕로 `S1r2/S1_table.csv`를 가리킨다. 이 문서에는 메트릭 값을 새로 쓰지 않았다(Q-S1 표의 수치는 run 수·날짜와 CL-0006 verdict_scope 원문 속 source-atom 숫자뿐이며 S1 값이 아니다).")
    w("- 판정 낱말은 원문 토큰의 재진술(출처 tier 병기)으로만 나온다. Q-S1 표의 토큰 열은 현황 표기이며 " + NO_QUOTE + ".")
    w("- 규칙: PREREG r6의 §7.4(Q-S1은 분류 밖)와 §7.2 질문 토큰 규칙을 따른다. 질문 토큰·UNCOVERED 목록은 각 질문 절 원문에 있는 그대로이며 "
      "이 조립은 새 분류를 하지 않았다.")
    w("- 모든 API 출력(map Sonnet 5.5 + Opus 5.5, reduce Fable 5.1, 초안 Sonnet 5.5, 퇴고 Opus 5.5)은 **증거가 아니라 DRAFT**이다.")
    w("")
    w("## 0. 기호 사전")
    w("")
    w("아래 두 표는 각각 `DESC_iteration.md`와 `DESC_z.md`의 `## 0. 기호 (사용 전 정의)` 절 본문을 원문 그대로 옮긴 것이다. 두 표는 겹치는 행이 있으나 합치지 않았다(중복은 그대로 둠).")
    w("")
    w(f"### 0-A. DESC_iteration.md §0 (출처: {pstr(D / 'S4_draft/DESC_iteration.md')} L{sym_iter_a+1}–L{sym_iter_b}, 헤딩 줄 제외)")
    w(reg("sym-iter", f"S4_draft/DESC_iteration.md L{sym_iter_a + 1}-L{sym_iter_b}", sym_iter_src.split("\n", 1)[1], "verbatim lines after the '## 0.' heading").rstrip("\n"))
    w("")
    w(f"### 0-B. DESC_z.md §0 (출처: {pstr(D / 'S4_draft/DESC_z.md')} L{sym_z_a+1}–L{sym_z_b}, 헤딩 줄 제외)")
    w(reg("sym-z", f"S4_draft/DESC_z.md L{sym_z_a + 1}-L{sym_z_b}", sym_z_src.split("\n", 1)[1], "verbatim lines after the '## 0.' heading").rstrip("\n"))
    w("")
    w("## 1. Q-S1 현황 표")
    w("")
    w(f"출처 파일·열 정의·40행 근거는 표 위에 적혀 있다. 이 블록은 `{rel(OUT_QS1)}`의 본문(프런트매터 제외)과 같다.")
    w("")
    w(reg("qs1", rel(OUT_QS1) + " (body after frontmatter)", QS1_BODY, "QS1_table.md body").rstrip("\n"))
    w("")
    w("## 2. iteration 축")
    w("")
    w("**편집 주(조립 단계 문장, 원문 아님):** 이 질문은 S3 reduce 가 두 번 시도에서 허용 문장을 내지 못해(coverage high) reduce 기반 서술이 없다. "
      "아래는 S1 값 서술만이다. (참고: Q-I1 셀에는 S1_table 값 행이 없고, 아래 블록은 DESC_iteration.md의 Q-I1 절 — S1_cells 토큰 재진술 — 이다. "
      "근거 파일: `S3_check_reduce.csv`의 Q-I1 coverage 행, §4.)")
    w("")
    w(reg("q-i1", f"S4_draft/DESC_iteration.md L{qi1_a}-L{qi1_b} (heading demoted ## -> ###)", demote(qi1_src), "Q-I1 section of DESC_iteration; heading level demoted by one"))
    for i in ["i2", "i3", "i4", "i5"]:
        w("")
        w(reg(f"rep-q-{i}", f"S5_assembly/reports/rep-q-{i}.md (whole file)", rep_txt[i], "whole file"))
    w("")
    w("## 3. z 축")
    for i in ["z1", "z2", "z3", "z4", "z5", "z6"]:
        w("")
        w(reg(f"rep-q-{i}", f"S5_assembly/reports/rep-q-{i}.md (whole file)", rep_txt[i], "whole file"))
    w("")
    w(f"### Q-Z7 {qz7_title}")
    w("")
    w(f"- Q-Z7 (CL-0006): LISTED-ONLY — Q-S1 표에서만. (`S1r2/S1_cells.csv`: q=Q-Z7, s=CL-0006, s1_class={qz7_cell['s1_class']}, q_token={qz7_cell['q_token']}. 본문 절 없음.)")
    w("")
    # ---------------------------------------------------------------- section 4
    w("## 4. 한계와 열린 항목 (사실 진술만)")
    w("")
    rc = red_check[0]
    w(f"1. **Q-I1 reduce 부재.** `S3_check_reduce.csv` {len(red_check)}행: model={rc['model']}, where={rc['where']}, rule={rc['rule']}, severity={rc['severity']}, message={rc['message']}. "
      "§2의 Q-I1 블록은 reduce 산출이 아니라 DESC_iteration.md의 S1_cells 토큰 재진술이다.")
    e11 = next(r for r in inv if r["source_row"] == "EXP-011")
    w(f"2. **EXP-011 로컬 부재.** `S0r2_inventory.csv` EXP-011: data_present={e11['data_present']}, ckpt_dirs_matched={e11['ckpt_dirs_matched']}, "
      f"queue_jobs/processing/done/failed={e11['queue_jobs']}/{e11['queue_processing']}/{e11['queue_done']}/{e11['queue_failed']}; S1_cells 셀 "
      + ", ".join(f"{c['q']}:{c['s1_class']}" for c in cells_by_s['EXP-011']) +
      f" (S0r2 앵커 정규식 기준; §7.0.5 예상 A-DESC와 다름 — S1_cells `reason`: r1 큐 정규식 `_abl_`이 `pp_abl_*`에 걸린 거짓 양성).")
    leg = sorted({c["s"] for c in cells if c["cohort_note"]})
    w(f"3. **legacy 코호트 비교 불가.** S1_cells `cohort_note` = \"{LEGACY_NOTE}\" 인 소스 행: {', '.join(leg)}. 이 행들의 값은 clean 코호트 값과 같은 문장에서 비교되지 않는다(원문 블록 각 절의 고정 문구).")
    def cl(name, cls):
        return f"{name}={len(cell_list(cls))} ({', '.join(cell_list(cls))})"
    w("4. **D-LOCKED 등 클래스 현황 (기술만, 셀 단위; S1_cells `s0_confirmed_class`).** "
      + "; ".join([cl("D-LOCKED", "D-LOCKED"), cl("D-PREJUDGE", "D-PREJUDGE"), cl("D-SEALED", "D-SEALED"), cl("X-PROV", "X-PROV"),
                   cl("X-ABSENT", "X-ABSENT"), cl("LISTED-ONLY", "LISTED-ONLY")])
      + ". D-LOCKED 셀은 `marker`=현황 기술(확증 아님)이고 어떤 종합 문장의 근거도 아니다(PREREG §6.2).")
    unk = keys_unknown()
    w(f"5. **값 없는 셀 {len(unk)}개** (S1_cells `key_status`=UNKNOWN → metric key 미고정, S1_table 값 행 없음): "
      + ", ".join(f"{q}:{s}({k})" for q, s, k in unk) + ".")
    w(f"6. **map 단계 lint {len(map_check)}건(severity high), reduce open_findings로 이월.** `S2_check_map.csv`: "
      + "; ".join(f"[{m['model']}] {m['where']} / rule={m['rule']} / {m['message']}" for m in map_check)
      + ". 해당 셀 id의 보고서 절 내 등장 횟수(문자열 검색): "
      + ", ".join(f"{cid}: z2={rep_txt['z2'].count(cid)}, z5={rep_txt['z5'].count(cid)}" for cid in ["Q-Z2__EXP-001", "Q-Z2__EXP-007", "Q-Z5__FG-0004"]) + ".")
    w("7. **reduce Q-I5 허용 문장 0.** `rep-q-i5.md`(§2의 Q-I5 블록): reduce의 \"Allowed sentences\" 표가 비어 있고 Q-I5 S1_table 행도 0건이라 명제 단락이 없다. "
      "Scope 표의 두 셀(Q-I5__CL-0004, Q-I5__FG-0002)은 A-VERDICT(Q-INHERIT)이다. 허용 문장 표가 빈 사유는 reduce 입력에 없다.")
    w("8. **사전등록 이탈 (사실만).** (a) S4 서술 초안·퇴고에 쓰기 API(draft-sync lane)를 사용했다. (b) 그 지출은 PREREG 하위 상한 배분(map ≤ $15, reduce ≤ $10, G6 탐침 ≤ $2, 재제출 예비 ≤ $8 → 합계 ≤ $35; PREREG 해당 표 행)에 배정된 항목이 아니다. "
      "두 가지 모두 사용자 결정(2026-10-30; orchestrator가 이 조립 지시에 전달한 내용, 결정 기록 문서는 이 조립에서 열람하지 않음)에 따른다. 실제 ledger 합산은 §5.3.")
    w(f"9. **V2 r3 이후 파일 상태.** `VERIFY-V2-r3.return.yaml`의 overall은 FAIL(blocking 2건: B-r3-1 rep-q-i3, B-r3-2 paper-en-iter)이다. 조립 입력 13개(보고서 10 + 논문 3) 중 `targets_sha256`와 디스크 sha가 다른 파일은 {len(differs)}개({', '.join(differs) if differs else '없음'})이다(V2 r3 이후 바뀐 것으로 추정되나 변경 내용은 확인하지 않음). 이 파일들에 대한 post-fix verifier return은 `S5_assembly/`에 없다(§5.4).")
    w("10. 이 조립본 자체는 verifier를 거치지 않았다. LaTeX는 컴파일하지 않았다.")
    w("")
    # ---------------------------------------------------------------- section 5
    w("## 5. 재현·출처")
    w("")
    w("### 5.1 모델·좌석")
    w("")
    w("map = Sonnet 5.5 + Opus 5.5 (독립 교차), reduce = Fable 5.1, 초안 = Sonnet 5.5, 퇴고 = Opus 5.5 (orchestrator 지시문 기준). ledger가 기록한 모델은 §5.3 표의 `planned.model`과 `sync_done.model_returned`.")
    w("")
    w("### 5.2 재현 명령")
    w("")
    w("```")
    w('CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s5_assemble.py')
    w("```")
    w("")
    w(f"스크립트 sha256 at run: `{SCRIPT_SHA}`. 입력 목록과 sha256은 조립 푸터에 있다. 스크립트는 S0r2/S1r2/S4/S5 입력과 ledger(읽기 전용)만 읽고, "
      "마지막에 쓴 세 파일을 디스크에서 다시 읽어 블록 sha를 재대조하며 불일치하면 비정상 종료한다.")
    w("")
    w("### 5.3 API 지출 (ledger 합산)")
    w("")
    w(f"방법: `lab/38_api-batch/ledger.jsonl`({len(ledger_rows)}행, sha256 `{ledger_sha}`; 읽기만)에서 `event` ∈ {{`collected`(batch lane), `sync_done`(sync lane)}} 행의 `cost_est_usd`를 "
      "`planned` 행(`manifest_sha256` 또는 `plan_hash`가 같은 행)의 `batch_dir`로 묶어 합산했다. 값은 추정치(`cost_est_usd`)이며 청구서가 아니다. "
      f"`2026-10-30_siz-` 접두 batch_dir만 이 합성의 부분합이고, 나머지 batch_dir은 제외 목록으로 따로 적었다. 비용 행 {len(ledger_cost_rows)}개.")
    w("")
    w("| batch_dir (`lab/38_api-batch/…`) | lane(approved) | planned.model | n_requests(planned) | batch_id (batch lane) | plan_hash (sync lane, 앞 12) | approval_id | sync_done.model_returned | 비용 행 수 | Σ cost_est_usd (USD) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for d in siz_dirs:
        e = LEDGER[d]
        pl = e["planned"]
        w("| " + " | ".join([
            esc(d.replace("lab/38_api-batch/", "")),
            esc(",".join(sorted({a[1] for a in e["approvals"]})) or NO_SRC),
            esc(str(pl.get("model")) if pl else NO_SRC),
            esc(str(pl.get("n_requests")) if pl else NO_SRC),
            esc(", ".join(e["batch_ids"]) or "—"),
            esc(", ".join(sorted(h[:12] for h in e["plan_hashes"])) or "—"),
            esc(", ".join(f"{a[0]}" for a in e["approvals"]) or NO_SRC),
            esc(", ".join(sorted(e["models"])) or "—"),
            str(e["n_cost"]),
            f"{e['cost']:.4f}",
        ]) + " |")
    w(f"| **siz-* 부분합 ({len(siz_dirs)} batch_dir)** | | | | | | | | {sum(LEDGER[d]['n_cost'] for d in siz_dirs)} | **{siz_total:.4f}** |")
    w("")
    w(f"전체 ledger 합(모든 batch_dir): {all_total:.4f} USD (비용 행 {len(ledger_cost_rows)}). 제외한 batch_dir({len(other_dirs)}개, siz-* 아님):")
    w("")
    for d in other_dirs:
        w(f"- `{d.replace('lab/38_api-batch/', '')}`: {LEDGER[d]['cost']:.4f} USD ({LEDGER[d]['n_cost']} 비용 행)")
    w("")
    if unlisted_errors:
        w("ledger `error` 행: " + "; ".join(f"ts={r['ts']}, where={r.get('where')}, custom_id={r.get('custom_id')}, error={r.get('error')}" for r in unlisted_errors)
          + " (이 행은 전송 전 중단이며 비용 행이 아니다; 합산에 들어가지 않음).")
        w("")
    w("### 5.4 입력 파일 sha256 vs `VERIFY-V2-r3.return.yaml` `targets_sha256`")
    w("")
    w("| 입력 | VERIFY-V2-r3 target sha256 | 조립 시 디스크 sha256 | 일치 |")
    w("|---|---|---|---|")
    for n, t, dsk in sha_cmp:
        w(f"| {n} | {t or NO_SRC} | {dsk} | {'예' if t == dsk else '아니오'} |")
    w("")
    w("‘아니오’ 행은 V2 r3가 검증한 sha와 다른 파일이다(V2 r3 이후 바뀐 것으로 추정). 이 조립은 지시에 따라 입력을 수정하지 않았고 무엇이 바뀌었는지는 확인하지 않았다(검증 시점 원본은 입력 목록 밖). 따라서 이 파일들이 “검증된 파일과 바이트 동일”하다고 주장하지 않는다.")
    w("")
    w("### 5.5 블록 ↔ 원문 sha256 일치 (조립 후 디스크 재읽기 결과)")
    w("")
    w("@@CHECK_TABLE@@")
    w("")
    w("### 5.6 검증 이력 (return 파일 경로)")
    w("")
    for p in ["VERIFY-V0-r2.return.yaml", "VERIFY-V1-r2-V0.return.yaml", "VERIFY-SIZ-A.return.yaml", "VERIFY-SIZ-B.return.yaml",
              "VERIFY-SIZ-C.return.yaml", "VERIFY-V1-r1.return.yaml"]:
        w(f"- `lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/{p}`")
    for r in range(1, 8):
        w(f"- `lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S4_draft/VERIFY-V2-desc-r{r}.return.yaml`")
    w("- `lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S5_assembly/VERIFY-V2-r3.return.yaml`")
    w("- PREREG 검증: `lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG_VERIFY-r1.return.yaml` … `_PREREG_VERIFY-r6.return.yaml` (PREREG frontmatter `verify_history`).")
    w("")
    w("이 목록에서 `VERIFY-V1-r2-V0.return.yaml`의 `verdicts.G-V0`은 `V0-FAIL`(blocking V7·V4)이고, 그 수정 후 `VERIFY-V0-r2.return.yaml`이 `V0-PASS-r2`를 낸 것이 frontmatter의 V0 토큰이다. "
      "`VERIFY-SIZ-A/B`의 overall은 FAIL, `VERIFY-SIZ-C`는 FIRST-RUN-READY로 각 파일에 적혀 있다(이 조립은 그 내용을 재해석하지 않음).")
    w("")
    w("### 5.7 지위")
    w("")
    w("이 문서와 그 안의 모든 API 출력은 증거가 아니라 DRAFT이다. 판정·status 변경·동결은 이 조립의 권한 밖이다.")
    w("")
    w("## 부록 A. DESC_iteration.md 전문")
    w("")
    w("원문 전체를 옮겼다. 변경은 두 가지뿐이다: (i) 원문 YAML frontmatter를 ```yaml 펜스로 감쌌다(문서 중간의 `---`가 setext 헤딩으로 읽히는 것을 막기 위함), "
      "(ii) 코드 펜스 밖의 헤딩 줄을 한 단계 내렸다(`#`→`##` 등). 역변환(펜스 해제·헤딩 복원)이 원문 바이트와 일치함을 §5.5에서 검사한다.")
    w("")
    w(reg("app-a", "S4_draft/DESC_iteration.md (whole file)", wrap_full_desc(desc_iter), "whole file; frontmatter fenced, headings demoted one level"))
    w("")
    w("## 부록 B. DESC_z.md 전문")
    w("")
    w("변경은 부록 A와 같다(frontmatter 펜스, 헤딩 한 단계 내림).")
    w("")
    w(reg("app-b", "S4_draft/DESC_z.md (whole file)", wrap_full_desc(desc_z), "whole file; frontmatter fenced, headings demoted one level"))
    w("")
    w("## 조립 푸터 (입력 sha256)")
    w("")
    w("```")
    w(f"script  tools/state_iter_z_synthesis/s5_assemble.py  {SCRIPT_SHA}")
    w(f"ledger  {rel(LEDGER_PATH)}  {ledger_sha}  ({len(ledger_rows)} rows, read-only)")
    for k in INPUTS:
        w(f"input   {rel(INPUTS[k])}  {sha(INPUTS[k])}")
    w(f"output  {rel(OUT_QS1)}  {sha(OUT_QS1) if OUT_QS1.exists() else '@@QS1SHA@@'}")
    w(f"output  {rel(OUT_TEX)}  {sha(OUT_TEX) if OUT_TEX.exists() else '@@TEXSHA@@'}")
    w("```")
    w("")
    return "\n".join(L) + "\n"


def check_results(doc: str):
    res = []
    for bid, (src_desc, exp, how) in EXPECT.items():
        got = extract_block(doc, bid)
        ok = got == exp
        res.append((bid, src_desc, shas(exp), shas(got), ok, how))
    return res


# ----------------------------------------------------------------------------- independent source re-slices (post-write)
def independent_source_check(doc: str):
    """Re-derive each block's expected text from the source files on disk (not from the in-memory EXPECT)."""
    out = []
    for i in REP_IDS:
        src = rd(A / "reports" / f"rep-q-{i}.md")
        got = extract_block(doc, f"rep-q-{i}")
        out.append((f"rep-q-{i}", f"S5_assembly/reports/rep-q-{i}.md", sha(A / "reports" / f"rep-q-{i}.md"), shas(got), got == src))
    for bid, p, tr in [("app-a", D / "S4_draft/DESC_iteration.md", None), ("app-b", D / "S4_draft/DESC_z.md", None)]:
        src = rd(p)
        got = extract_block(doc, bid)
        inv_ok = unwrap_full_desc(got) == src
        out.append((bid, rel(p) + " (inverse transform)", sha(p), shas(unwrap_full_desc(got)), inv_ok))
    for bid, p, pat in [("sym-iter", D / "S4_draft/DESC_iteration.md", r"## 0\. 기호"), ("sym-z", D / "S4_draft/DESC_z.md", r"## 0\. 기호")]:
        sl, a, b = slice_between(rd(p), pat, r"## ")
        body = sl.split("\n", 1)[1]
        got = extract_block(doc, bid)
        out.append((bid, f"{rel(p)} L{a + 1}-L{b}", shas(body if body.endswith('\n') else body + '\n'), shas(got), got == (body if body.endswith('\n') else body + '\n')))
    sl, a, b = slice_between(rd(D / "S4_draft/DESC_iteration.md"), r"## Q-I1 ", r"## Q-I2 ")
    got = extract_block(doc, "q-i1")
    out.append(("q-i1", f"{rel(D / 'S4_draft/DESC_iteration.md')} L{a}-L{b} (inverse heading demotion)", shas(sl), shas(undemote(got)), undemote(got) == sl))
    return out


def main():
    # 1. Q-S1
    OUT_QS1.write_text(QS1_FRONT + QS1_BODY, encoding="utf-8")
    # 2. paper
    OUT_TEX.write_text(TEX, encoding="utf-8")
    # 3. RESULTS (two-pass: the check table is filled after a first render, using the same blocks)
    doc = build_results()
    chk = check_results(doc)
    ind = independent_source_check(doc)
    tbl = ["| 블록 id | 원문 (파일/줄) | 원문 sha256 (또는 변환 기대값) | 문서에 들어간 블록 sha256 | 일치 |", "|---|---|---|---|---|"]
    for bid, src_desc, s_exp, s_got, ok, how in chk:
        tbl.append(f"| {bid} | {src_desc} ({how}) | `{s_exp}` | `{s_got}` | {'일치' if ok else '불일치'} |")
    tbl.append("")
    tbl.append("원문 파일에서 독립적으로 다시 계산한 대조(전체 파일은 원문 파일 sha256, 변환 블록은 역변환 결과의 sha256):")
    tbl.append("")
    tbl.append("| 블록 id | 원문 | 원문 sha256 | 역변환/추출 결과 sha256 | 일치 |")
    tbl.append("|---|---|---|---|---|")
    for bid, src_desc, s_src, s_got, ok in ind:
        tbl.append(f"| {bid} | {src_desc} | `{s_src}` | `{s_got}` | {'일치' if ok else '불일치'} |")
    tex_chk = check_tex(TEX)
    tbl.append("")
    tbl.append(f"논문 조각 → `{rel(OUT_TEX)}`: `\\subsection` 개수 {nsub(TEX)} (iter 1, z-a 1, z-b 0 의 합).")
    tbl.append("")
    tbl.append("| 논문 조각 | 원문 sha256 | 연결본 안의 조각 sha256 | 일치 |")
    tbl.append("|---|---|---|---|")
    for pid, s_src, s_got, ok in tex_chk:
        tbl.append(f"| {pid}.tex | `{s_src}` | `{s_got}` | {'일치' if ok else '불일치'} |")
    qs1_ok = rd(OUT_QS1).endswith(QS1_BODY) and rd(OUT_QS1).startswith(QS1_FRONT)
    tbl.append("")
    tbl.append(f"Q-S1 표: `{rel(OUT_QS1)}` = 프런트매터 + 본문, 본문 sha256 `{shas(QS1_BODY)}` ({'RESULTS §1 블록과 일치' if extract_block(doc, 'qs1') == QS1_BODY else '불일치'}).")
    doc = doc.replace("@@CHECK_TABLE@@", "\n".join(tbl))
    doc = doc.replace("@@QS1SHA@@", sha(OUT_QS1)).replace("@@TEXSHA@@", sha(OUT_TEX))
    OUT_RESULTS.write_text(doc, encoding="utf-8")

    # ---- post-write verification from disk
    disk = rd(OUT_RESULTS)
    bad = []
    for bid, (src_desc, exp, how) in EXPECT.items():
        if extract_block(disk, bid) != exp:
            bad.append(bid)
    for r in independent_source_check(disk):
        if not r[4]:
            bad.append("independent:" + r[0])
    for r in check_tex(rd(OUT_TEX)):
        if not r[3]:
            bad.append("tex:" + r[0])
    if not qs1_ok:
        bad.append("qs1-file")
    if "@@" in disk:
        bad.append("unfilled placeholder")
    print("S5 assembled.")
    print("  QS1_table.md :", sha(OUT_QS1), f"({len(inv)} source rows, {len(cells)} cells)")
    print("  paper tex    :", sha(OUT_TEX), f"subsections={nsub(TEX)}")
    print("  RESULTS.md   :", sha(OUT_RESULTS))
    print("  siz-* est. USD:", f"{siz_total:.4f}", "| whole ledger est. USD:", f"{all_total:.4f}")
    print("  V2-r3 sha differs on disk:", differs)
    for bid, src_desc, s_exp, s_got, ok, how in chk:
        print(f"  block {bid:10s} {s_got[:16]} {'OK' if ok else 'MISMATCH'}")
    for pid, s_src, s_got, ok in check_tex(rd(OUT_TEX)):
        print(f"  tex   {pid:14s} {s_src[:16]} {'OK' if ok else 'MISMATCH'}")
    if bad:
        die("post-write mismatch: " + ", ".join(bad))
    print("  all byte checks OK")


if __name__ == "__main__":
    main()
