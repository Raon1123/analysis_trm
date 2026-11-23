#!/usr/bin/env python
"""S4 description-only draft generator (DESC_iteration.md, DESC_z.md).

Inputs (only): S1r2/S1_table.csv, S1r2/S1_cells.csv.  INTERP_S1.md r2 / PREREG r6 are cited, not parsed.
No GPU, no wandb, no sealed/preds/z_snapshots.  Run:

    CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s4_desc_gen.py            # (re)generate both files
    CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s4_desc_gen.py --check    # mechanical self-check
    CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s4_desc_gen.py --rows T00001-T00096,T00110
    CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s4_desc_gen.py --cells "Q-I1,CL-0003"

Every number in the prose is computed here from S1_table.csv (no hand-typed data numbers).
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ayp/project/trm")
SYN = ROOT / "lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis"
S1D = SYN / "S1r2"
OUT = SYN / "S4_draft"
SELF = Path(__file__).resolve()
GEN_CMD = 'CUDA_VISIBLE_DEVICES="" uv run python tools/state_iter_z_synthesis/s4_desc_gen.py'

LEG = "legacy, 신규 clean 코호트와 비교 불가"
R3 = ("iter((H,L)_train = (3,6))와 noiter((1,1))는 재귀 계산량이 약 6× 다르다. 이 차이는 정확도 수준의 marginal 관측이며, "
      "가중치 공유 재귀 구조의 효과와 단순 깊이/FLOPs 증가를 분리하지 않는다. 이 합성이 쓰는 소스 안에서 iter 셀과 계산량을 맞춘"
      "(depth-matched) non-recurrent 대조군으로 정의된 arm은 없다.")
EXP011_1 = ('iter 축(Q-I2)의 비재귀 대조군(EXP-011 `abl_tfb_*` arm)에 대한 직접 근거(run·체크포인트)는 이 호스트의 as_of 로컬 데이터에 없다 '
            '((Q-I2, EXP-011)·(Q-I4, EXP-011) X-ABSENT).')
EXP011_2 = ("EXP-011 문서·레지스트리 기록은 Q-S1 현황 표의 상태 서술로만 인용한다. CL-0005는 Q-I1 절에서 자기 verdict·scope로 진술되며, "
            "이 문단에서는 포인터('Q-I1 절 참조')로만 나온다.")
# R-5 verbatim from PREREG §5.3 (check() re-parses the PREREG line and asserts equality, incl. the quoted prohibition).
R5 = ("k를 가로지르는 문장은 각 k의 T_k 등급을 병기하고, ⌊log₂k⌋ 깊이로만 설명하지 않는다. cycle-type 축은 predictor-only이며 기제가 아니고, "
      'n=10에서 parity와 collinear다. "best / governing / first to observe" 금지.')
PREREG = SYN.parent / "2026-10-30_state-iter-z-synthesis_PREREG.md"
EXP25_DOC = ROOT / "lab/21_experiments/11_lab-experiments/EXP-015_looped-tf-direct-control.md"   # = EXP-025 design doc (legacy file name)
EXP25_LINES = (15, 62, 351, 399, 401, 403, 404, 405, 406)
H199_DOC = ROOT / "lab/27_theory/11_lab-theory/H-199.md"     # states the TRM-side C formula (L46)
H199_LINES = (46,)
R11 = "EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015)"
NOVAL = "값 없음"
NOLOG = "값 없음(키 미로깅)"
MARK = "현황 기술(확증 아님)"


# ----------------------------------------------------------------------------------------------- data
def load():
    t = pd.read_csv(S1D / "S1_table.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(S1D / "S1_cells.csv", dtype=str, keep_default_na=False)
    t["rn"] = t.row_id.str[1:].astype(int)
    return t, c


T, C = load()
# cells that S1_cells itself marks as legacy-source (cohort_note); includes row-less UNKNOWN cells
LEGSRC = [(a, b) for a, b, n in zip(C.q, C.s, C.cohort_note) if n]
assert all(n == LEG for n in C.cohort_note if n), "S1_cells cohort_note differs from the R-2 fixed string"
_SSRC = {b for _, b in LEGSRC}
assert not [1 for b, n in zip(C.s, C.cohort_note) if b in _SSRC and not n], "a source token is shared with a cell lacking cohort_note"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rid_str(ids) -> str:
    """Compress sorted T-ids into ranges; ranges are only formed over numerically consecutive ids."""
    ns = sorted({int(i[1:]) for i in ids})
    if not ns:
        return "—"
    out, a, b = [], ns[0], ns[0]
    for n in ns[1:]:
        if n == b + 1:
            b = n
        else:
            out.append((a, b))
            a = b = n
    out.append((a, b))
    f = lambda n: f"T{n:05d}"
    return ", ".join(f(a) if a == b else f"{f(a)}–{f(b)}" for a, b in out)


def cell_rows(q, s, key=None, only_evidence=False):
    d = T[(T.q == q) & (T.s == s)]
    if key:
        d = d[d.metric_key == key]
    if only_evidence:
        d = d[d.evidence_use == "yes"]
    return d


def cell(q, s):
    d = C[(C.q == q) & (C.s == s)]
    assert len(d) == 1, (q, s)
    return d.iloc[0]


def variant(rn: str) -> str:
    m = re.match(r"fig1[a-z0-9]*_(tf|mlp)_(z|noz)_(iter|noiter)_", rn)
    return f"{m.group(2)}_{m.group(3)}" if m else ""


def val(r) -> str:
    return r.value if r.value != "" else NOLOG


def stepf(r) -> str:
    return str(int(float(r.step))) if r.step != "" else "—"


# ----------------------------------------------------------------------------------------------- doc builder
class Doc:
    def __init__(self, name):
        self.name = name
        self.lines: list[str] = []
        self.ledger: list[tuple[str, str, str, str]] = []
        self.np = 0

    def raw(self, s=""):
        self.lines.append(s)

    def h(self, level, text):
        self.lines += ["", "#" * level + " " + text, ""]

    def quote(self, text):
        self.lines += ["", "> " + text, ""]

    def para(self, text, ids=None, cells=None, ev=None, cmd=None):
        """Paragraph with the closing row_id list. ids = S1_table row_ids backing every data number in `text`."""
        nc = named_cells(text + ' ' + (cells or ''))
        legcell = bool(nc)
        if legcell and LEG not in text:         # name exactly the cohort_note cells this paragraph cites (B-r4-1)
            text = text.rstrip() + f" {'·'.join(f'({q}, {s_})' for q, s_ in nc)} 셀의 소스는 S1_cells에 legacy 코호트로 표시돼 있다 — {LEG}."
        self.np += 1
        pid = f"{self.name}-P{self.np:03d}"
        ev_o, cmd_o = ev, cmd
        if ids is not None and len(list(ids)) > 0:
            ids = list(ids)
            tail = f"〔row_id: {rid_str(ids)}〕"
            ev = f"S1r2/S1_table.csv {rid_str(ids)}"
            cmd = f"{GEN_CMD} --rows {rid_str(ids).replace('–', '-').replace(', ', ',')}"
        else:
            cref = cells or "S1_cells (q,s) 표 참조"
            tail = f"〔row_id: 해당 없음 (S1_table 수치 없음); 출처 S1_cells {cref}〕"
            ev = f"S1r2/S1_cells.csv {cref}"
            cmd = f"{GEN_CMD} --cells \"{cells}\"" if cells else f"{GEN_CMD} --cells \"<q>,<s>\""
        first = cut90(re.split(r"(?<=[.。])\s", text.strip(), maxsplit=1)[0])
        first = leg_claim(first, ev_o or ev, force=legcell)
        self.ledger.append((pid, first, ev_o or ev, cmd_o or cmd))
        self.lines += ["", text.strip(), tail, ""]

    def table(self, caption, header, rows, ids, flt=None):
        """Markdown table; each data row carries its own row_id column (already part of `rows`)."""
        ids = list(ids)
        self.np += 1
        pid = f"{self.name}-P{self.np:03d}"
        self.lines += ["", caption, "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        for r in rows:
            self.lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
        self.lines += ["", f"〔row_id: {rid_str(ids)}〕", ""]
        evs = f"S1r2/S1_table.csv filter [{flt}] ({len(ids)} rows)" if flt else f"S1r2/S1_table.csv {rid_str(ids)}"
        if flt and not ev_has_legacy(evs):                        # filter strings carry no T-ids; use the rows' own cohort
            evs += " " + rid_str(ids)
        claim = leg_claim(cut90(caption.replace("**", "")), evs)
        if flt:
            self.ledger.append((pid, claim, f"S1r2/S1_table.csv filter [{flt}] ({len(ids)} rows)", f"{GEN_CMD} --cell \"{flt}\""))
        else:
            self.ledger.append((pid, claim, f"S1r2/S1_table.csv {rid_str(ids)}",
                                f"{GEN_CMD} --rows {rid_str(ids).replace('–', '-').replace(', ', ',')}"))

    def text(self):
        return "\n".join(self.lines).rstrip() + "\n"


def mentions_cell(text: str, q: str, s_: str) -> bool:
    """True if `text` names the (q, s) cell in any form: '(q, s)', 'q,s', or the source token s alone ('EXP-004 행', 'Q-Z2 (EXP-004 / EXP-007 ...)').
    Safe because every source token of a cohort_note cell belongs only to cohort_note cells (asserted below)."""
    return f"({q}, {s_})" in text or f"{q},{s_}" in text or re.search(rf"(?<![\w\-]){re.escape(s_)}(?![\w\-/])", text) is not None


def named_cells(text: str):
    """The cohort_note cells that `text` actually names: exact '(q, s)' / 'q,s', or the source token s together with its question token q
    (or, if no q of that s appears at all, every cohort_note cell carrying that s)."""
    out = []
    for q, s_ in LEGSRC:
        tok = re.search(rf"(?<![\w\-]){re.escape(s_)}(?![\w\-/])", text) is not None
        if f"({q}, {s_})" in text or f"{q},{s_}" in text:
            out.append((q, s_))
        elif tok and q in text:
            out.append((q, s_))
        elif tok and not any(q2 in text for q2, s2 in LEGSRC if s2 == s_):
            out.append((q, s_))
    return out


def ev_has_legacy(ev: str) -> bool:
    """True if a ledger evidence string points at a legacy referent: a cohort_note cell ('Q-Z1,EXP-001 / EXP-010', 'Q-Z1|EXP-004|key')
    or S1_table row_ids whose S1 cohort is legacy."""
    ev = ev.replace("\\|", "|")
    for q, s_ in LEGSRC:
        if f"{q}|{s_}" in ev:
            return True
        for m in re.finditer(re.escape(q) + r",([^;`]*)", ev):
            if re.search(rf"(?<![\w\-]){re.escape(s_)}(?![\w\-/])", m.group(1)):
                return True
    ids = []
    for a, b in re.findall(r"T(\d{5})(?:–T(\d{5}))?", ev):
        ids += [f"T{n:05d}" for n in range(int(a), int(b or a) + 1)]
    return bool(ids) and bool((T[T.row_id.isin(ids)].cohort == "legacy").any())


def cut90(t: str, n: int = 240) -> str:
    """Cut a ledger claim at a word boundary without leaving an unclosed '(' (keeps qualifiers such as '(확증 아님)' whole)."""
    if len(t) <= n:
        return t
    c = t[:n]
    if c.count("(") > c.count(")"):
        k = t.find(")", n)
        c = t[:k + 1] if 0 <= k <= n + 40 else c[:c.rfind("(")]
    elif t[n] not in " .,;)":
        c = c[:c.rfind(" ")] if " " in c else c
    return c.rstrip() + " …"


def leg_claim(claim: str, ev: str, force: bool = False) -> str:
    """R-2: a ledger claim that mentions legacy / a cohort_note cell carries the fixed string (the em-dash split / 90-char cut would drop it),
    but ONLY if its evidence has a legacy referent (cohort_note cell or legacy S1_table rows) -- never as a label on a clean-cohort row (B-r4-2)."""
    # two-way rule (B-r5-2): the claim carries the string iff the evidence holds a legacy referent (a cohort_note cell, or any S1_table row with cohort = legacy)
    return claim if (LEG in claim or not ev_has_legacy(ev)) else f"{claim} [{LEG}]"


def prot_desc(d: pd.DataFrame) -> str:
    u = lambda c: "/".join(sorted(set(d[c])))
    return f"probe_size={u('probe_size')}, weights={u('weights')}, eval_set_size={u('eval_set_size')}"


def run_rows(d: pd.DataFrame, legacy: bool):
    """Per-directory rows, sorted deterministically."""
    d = d.copy()
    d["_k"] = d.k.astype(int)
    d["_s"] = d.seed.astype(int)
    d = d.sort_values(["block", "arch", "HL_train", "_k", "_s", "rn"])
    rows = []
    for r in d.itertuples():
        coh = LEG if legacy else r.cohort
        rows.append([r.run_name, r.wandb_run_id, r.k, r.seed, r.arch, r.HL_train, val(r), stepf(r), r.dup_wandb_dirs, coh, r.row_id])
    return rows


RUN_HDR = ["run_name", "wandb_run_id", "k", "seed", "arch", "(H,L)_train", "value (R-FINAL)", "step", "dup", "코호트 / cohort_note", "row_id"]


def front(title_slug, cover, extra=""):
    s1t, s1c, ip = S1D / "S1_table.csv", S1D / "S1_cells.csv", SYN / "INTERP_S1.md"
    return f"""---
exp_id: UNASSIGNED            # 부모 PREREG와 같음(PREREG §13 O-1). 자기 배정 금지
slug: state-iter-z-synthesis
doc_type: s4-draft-description-only
covers: "{cover}"
parent_prereg: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis_PREREG.md
parent_prereg_sha256: {sha(SYN.parent / '2026-10-30_state-iter-z-synthesis_PREREG.md')}
s0_of_record: "S0r2, as_of 2026-10-30T13:51:29Z"
s1_table: {{path: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S1r2/S1_table.csv, sha256: {sha(s1t)}}}
s1_cells: {{path: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/S1r2/S1_cells.csv, sha256: {sha(s1c)}}}
interp: {{path: lab/23_reports/11_lab-reports/2026-10-30_state-iter-z-synthesis/INTERP_S1.md, revision: r2, sha256: {sha(ip)}}}
generator: tools/state_iter_z_synthesis/s4_desc_gen.py   # sha256는 본문 §재생성 참조
status: DRAFT                 # 판정·동결·status 변경 권한 없음. verifier 판정은 orchestrator 몫
values_opened_beyond_S1: 0    # S1r2 표 외 값 열람 없음; sealed / preds / z_snapshots 미열람; GPU 미사용
{extra}---
"""


SYMBOLS = [
    ("n", "순열의 크기 n = 10 고정(PREREG §0). S1_table에는 data_path 칸이 없어 이 문서의 run이 모두 n=10 경로인지는 S1 표만으로 검증하지 않았다(부록 B)."),
    ("k", "합성 깊이 k (σ ↦ σ^k의 지수). 데이터셋 매개변수. S1_table `k` 칸."),
    ("seed", "run 이름의 `_s<seed>` / `_sm<seed>` 접미. S1_table `seed` 칸. 값은 seed별로만 나열하고 seed를 합산·풀링하지 않는다(R-6)."),
    ("z_* / noz_*", "z_* = `arch=trm`(z_H·z_L 둘 다 carry, z-carry), noz_* = `arch=trm_singlez`(z_L만 carry, single-z). run 이름 접두 `z`/`noz`."),
    ("(H,L)_train", "훈련 시 (H_cycles, L_cycles). iter = (3,6), noiter = (1,1). S1_table `HL_train` 칸. `UNKNOWN` = 칸에서 읽지 못함(looped TF 등)."),
    ("(H,L)_eval", "평가 시 순환 수. S1_table 전 행 `UNKNOWN` — 이 문서는 (H,L)_eval 값을 쓰지 않는다."),
    ("P-PROBE", "wandb `probe/test_exact` (probe_size 512). `probe/test_exact` 행의 `weights`는 S1의 모든 코호트에서 UNKNOWN(PREREG U-2 미해결)."),
    ("P-ALL", "wandb `all/exact_accuracy` (online raw 가중치, S1 `weights`=online-raw). P-PROBE와 다른 프로토콜 — 같은 표·문장에서 직접 비교하지 않는다(R-12)."),
    ("R-FINAL", "run 디렉터리별 마지막 logged step의 값. 소스 판정이 쓴 환원(예: looped TF 계열의 R-ANY)과 다를 수 있다."),
    ("step", "그 디렉터리의 마지막 logged step(`step` 칸). 공식 `budget`(total_steps)은 S1 전 행 UNKNOWN이라 쓰지 않는다."),
    ("dup", "같은 (project, run_name)의 wandb 디렉터리 수. 2이면 두 행을 나란히 적고 대표값을 고르지 않는다(INTERP_S1 (9))."),
    ("코호트 태그", f"legacy = `Sigma_k_fig12`(오염 가능 빌드; 항상 \"{LEG}\"를 붙인다) · clean-D0 = `Sigma_k_new` · clean-D1 = `power_permutation` · UNKNOWN = 조건 미확인(증거 사용 제외)."),
    ("evidence_use", "S1_table 칸. `yes`인 행만 이 문서의 서술 근거. `no: …` 행은 사유와 row_id만 적고 값을 쓰지 않는다."),
    ("row_id", "S1_table.csv의 `T#####`. `T00001–T00096`은 연속 구간(양끝 포함)."),
    ("값 없음", "그 (q, s) 셀에 증거 사용 가능한 S1_table 값 행이 없음. 사유는 S1_cells와 부록에 적는다."),
    ("현황 기술(확증 아님)", "잠겼던 소스(D-LOCKED)나 자기 토큰의 원문 재진술 표지. 종합 문장의 근거가 아니다(PREREG §6.2)."),
    ("Tk_class", "k별 cycle-type 보존 비율 등급. S1_table 전 행 UNKNOWN."),
    ("S0r2", "S0 스냅샷의 개정 2(as_of 2026-10-30T13:51:29Z). (q, s) 셀의 데이터 클래스(`S0r2 분류`)는 이 스냅샷 기준이다(frontmatter `s0_of_record`)."),
    ("PREREG r6", "부모 사전등록(`2026-10-30_state-iter-z-synthesis_PREREG.md`). 이 문서의 약어·번호 — R-1..R-12(§5.3 문장 규칙), U-n(§14 미해결 항목), 데이터 클래스·질문 토큰(§7.1·§7.2) — 는 그 문서의 정의를 따르며 이 표는 쓰이는 것만 요약한다."),
    ("데이터 클래스", "S0r2가 (q, s) 셀마다 붙인 분류: X-PROV · LISTED-ONLY · X-ABSENT · X-OUT · D-SEALED · D-PREJUDGE · D-LOCKED · A-VERDICT · A-DESC · UNCOVERED(PREREG §7.1). 이 문서는 분류 이름만 쓰고 정의를 다시 쓰지 않는다. `s1_class`/`S0r2 분류`는 S1_cells 칸."),
    ("질문 토큰", "질문 q 단위 토큰: Q-NONE · Q-LISTED · Q-INHERIT · Q-DESC-ONLY · Q-UNCOVERED(PREREG §7.2). 이 문서는 질문 절 제목에 그 토큰을 병기한다."),
    ("P-RELOAD / P-DECODE", "P-RELOAD = 체크포인트를 (H,L)_eval로 바꿔 reload 후 오프라인 재평가 (H-108 sweep, N-series), P-DECODE = forward 중간 상태의 per-call / per-loop decode readout(EXP-017 `a_j`, EXP-022). 둘 다 P-ALL·P-PROBE와 섞지 않으며 이 문서에는 값이 없다(PREREG §0)."),
    ("kind / tier / key_status", "kind = 소스 행 종류 E(실험·분석)·A(원자)·H(가설). tier = 토큰 출처 tier(S1_cells `token_tier` 칸; 우선순위는 PREREG §5.1). key_status = S1_cells 칸으로, UNKNOWN이면 metric key 이름이 미고정이라 S1_table 값 행이 없다(부록 A). NAMED-ADMITTED는 S1_cells에 적힌 key_status 값이다."),
    ("판정 낱말 / 상태 낱말", "판정 낱말(V_tok)은 PREREG §0이 열거하는 닫힌 집합이며 이 문서는 S1_cells 토큰 원문(출처 tier 병기)으로만 재진술한다. 상태 낱말(N_tok)은 상태·진행을 뜻하는 낱말(under-test, not-addressed, superseded, unproven 및 진행 낱말)이며 판정 낱말이 아니다(PREREG §0)."),
    ("arm fullgrad / g7 / noinj", "EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015) looped TF arm 이름: `loop_grad_cycles`·`loop_input_injection` 설정이 다른 변형(EXP-025 설계 문서 §4.2). 이 문서는 run 이름 접두로만 센다."),
    ("기타 약어", "T_k, ceiling, hazard, right-censored, a_j, τ(steps-to-grok), H-collapse, C-LTF, C-ACT, ACT, card-level은 소스 원문 용어이며 토큰 재진술 안에서만 나온다. 정의는 각 소스와 PREREG §0·§5.2를 따르고 이 문서가 새로 정의하지 않는다."),
]


def symbols_table(doc: Doc, extra=()):
    doc.h(2, "0. 기호 (사용 전 정의)")
    doc.raw("| 기호 | 정의 |")
    doc.raw("|---|---|")
    for a, b in list(SYMBOLS) + list(extra):
        doc.raw(f"| {a} | {b} |")
    doc.raw("")


def assert_constants():
    assert_prose_facts()
    assert set(T.budget) == {"UNKNOWN"} and set(T.HL_eval) == {"UNKNOWN"} and set(T.Tk_class) == {"UNKNOWN"}
    assert set(T.label_GC) == {"NONE"} and set(T.reduction) == {"R-FINAL"} and set(T.frame) == {"full"}


def assert_prose_facts():
    """Hard-coded prose claims (cohort of a cell, run-name families, constants in captions) re-derived from S1_table at build time (B-r6-1 lesson)."""
    ev_ = lambda q, s_, key="probe/test_exact": T[(T.q == q) & (T.s == s_) & (T.metric_key == key) & (T.evidence_use == "yes")]
    for q, s_, coh in [("Q-I2", "EXP-005", "legacy"), ("Q-I2", "EXP-012/probe", "clean-D0"), ("Q-I3", "EXP-025", "clean-D1"), ("Q-I4", "EXP-016", "clean-D1"),
                       ("Q-Z1", "EXP-012/probe", "clean-D0"), ("Q-Z1", "EXP-004", "legacy"), ("Q-Z3", "EXP-002", "legacy"), ("Q-Z4", "EXP-006", "legacy")]:
        assert set(T[(T.q == q) & (T.s == s_) & (T.evidence_use == "yes")].cohort) == {coh}, (q, s_, coh)
    z1 = ev_("Q-Z1", "EXP-004")
    assert set(z1.probe_size) == {"512"} and set(z1.weights) == {"UNKNOWN"} and set(z1.eval_set_size) == {"UNKNOWN"}          # Z-1 caption constants
    assert z1[z1.exp_owner == "EXP-001"].run_name.str.startswith("fig1_tf_").all()                                              # 'owner EXP-001 = fig1_tf_*'
    assert ev_("Q-I2", "EXP-005").run_name.str.startswith("fig1_mlp_").all()
    assert ev_("Q-Z3", "EXP-002", "z/eff_rank").run_name.str.startswith("fig1_tf_").all() and ev_("Q-Z4", "EXP-006").run_name.str.startswith("fig1_tf_").all()
    assert set(ev_("Q-I3", "EXP-025").arch) == {"transformers_baseline"} and set(ev_("Q-I3", "EXP-025").seed) == {"1", "2", "3"}
    assert ev_("Q-I4", "EXP-016").step.nunique() > 1                                                                           # 'step values differ'
    assert set(T[T.metric_key == "probe/test_exact"].weights) == {"UNKNOWN"}                                                   # P-PROBE weights UNKNOWN in every cohort
    leg = T[T.metric_key == "probe/test_exact"]
    assert (leg.groupby("wandb_run_id").evidence_use.nunique() == 1).all()                                                     # a run is in Z-1 xor Z-2 in every cell


# ----------------------------------------------------------------------------------------------- token tables
def tok_text(r, special=None):
    if special:
        return special
    v, n = r.token_V, r.token_N
    return v if v else (n if n else "—")


def cell_status_table(doc, items, notes=None):
    """One row per (q,s). Value-row counts carry the row_id range in the same row."""
    notes = notes or {}
    hdr = ["(q, s)", "kind", "S0r2 분류", "key_status", "S1 값 행", "run (증거 사용)", "코호트별 run", "비고", "row_id"]
    rows, allids = [], []
    for q, s in items:
        r = cell(q, s)
        d = T[(T.q == q) & (T.s == s)]
        ids = list(d.row_id)
        allids += ids
        if len(d):
            vals = f"{len(d)}"
            ru = f"{r.n_runs_evidence_use}"
            co = r.counts_by_cohort
        else:
            vals, ru, co = NOVAL, "—", "—"
        note = notes.get((q, s), "")
        if ("legacy" in co or r.cohort_note) and LEG not in note:      # R-2: legacy row carries the fixed string
            note = f"{note}; {LEG}" if note else LEG
        rows.append([f"({q}, {s})", r.kind, r.s1_class, r.key_status or "—", vals, ru, co, note,
                     rid_str(ids) if ids else "—"])
    return hdr, rows, allids


# ----------------------------------------------------------------------------------------------- shared token paragraph lines
ATOM_SCOPE = {
    "CL-0003": "state_dict 수준의 기계적 reload 가능성만. 다른 순환 수에서 정확도가 보존된다는 주장은 범위 밖.",
    "CL-0005": "비재귀로 훈련된 셀에서 추가 재귀를 줄 때의 열화 관측(관측만). 기제·상전이·'isolated point' 서사, 구(舊) 원문 인용은 금지.",
    "FG-0001": "패널 미렌더·review 미통과 — 결론 금지. 결정 규칙을 '실행됨'으로 서술하는 것 금지.",
    "CL-0004": "총 재귀량을 고정했을 때 H/L 분할이 중요하다는 것만. 어느 쪽이 이기는지·이유는 범위 밖(방향 주장 금지).",
    "FG-0002": "그림 서술만(소스 N1이 허용한 문장의 그림 서술). 'the retention rule'로 캡션하는 것 금지.",
    "FG-0006": "EXP-025 per-run 서술만. H-199·C-LTF 판정이 아니며, 방향·기제 주장은 금지.",
    "CL-0001": "clean-D1 ceiling 도달 run의 post-ceiling hazard의 모양과 하한만. D1 관측 창 밖 외삽·기제·legacy로의 이전은 금지(legacy, 신규 clean 코호트와 비교 불가). 우측 절단(right-censored).",
    "CL-0002": "데이터 빌더 필터가 코호트마다 다른 순열을 남긴다는 코드 사실만. 학습 결과 판정이 아니다.",
    "FG-0003": "그려진 곡선의 서술·가독성만. H-201·C-Z 판정이 아니며, 'a_j가 상승 = 단계적 합성' 해석과 k 간 시점 순서 해석은 금지.",
    "FG-0004": "그려진 곡선의 서술·가독성만(위와 같은 금지). 원자 본문에 사용 보류 표기(WITHHELD NOTE)가 있다 — 사용자 결정 전까지 closeout 자료 사용 보류.",
    "FG-0005": "그려진 곡선의 서술·가독성만(위와 같은 금지).",
}


def token_rows(items, special=None):
    special = special or {}
    hdr = ["(q, s)", "kind", "S0r2 분류", "토큰 (원문)", "tier", "출처", "원자 코호트", "범위(수치 없는 요약)"]
    rows = []
    for q, s in items:
        r = cell(q, s)
        tk = tok_text(r, special.get((q, s)))
        row = [f"({q}, {s})", r.kind, r.s1_class, tk, r.token_tier or "—", f"{r.token_file} · {r.token_locator}".strip(" ·") if r.token_file else "—",
               r.atom_cohort or "—", ATOM_SCOPE.get(s, "")]
        rows.append([c.replace("|", "\\|") for c in row])      # '|' inside a token (e.g. 'pre-registered|under-test') must not add a cell
    return hdr, rows


def exp25_lines() -> dict:
    """Verbatim lines of the EXP-025 design doc that R-3 requires to be quoted (asserted so a silent drift fails the build)."""
    ls = EXP25_DOC.read_text().splitlines()
    out = {n: ls[n - 1] for n in EXP25_LINES}
    assert out[15].startswith("# Looped-transformer direct control (matched params, matched C = 42)")
    assert out[62].startswith("| `C`, `G` | block applications per forward")
    assert "C = loops·H_layers = 42 at loops 21" in out[351]
    assert out[406].startswith("| ref `T` = `pp_base_tf_z_iter`")
    assert out[399].startswith("### 4.2 Cohort") and out[401].startswith("| cohort (tag) | loops")
    assert out[403].startswith("| `fullgrad`") and out[404].startswith("| `g7`") and out[405].startswith("| `noinj`")
    return out


def h199_lines() -> dict:
    ls = H199_DOC.read_text().splitlines()
    out = {n: ls[n - 1] for n in H199_LINES}
    assert "TRM: H_cycles·(L_cycles+1)·L_layers = 42" in out[46] and "looped TF: loops·H_layers" in out[46]
    return out


def exp25_quote(D: Doc):
    """PREREG R-3 last sentence / §2.2 Q-I3 / §0 row C: verbatim EXP-025 text on C and its compute matching, with file:line."""
    q = exp25_lines()
    rel = EXP25_DOC.relative_to(ROOT)
    D.raw("")
    D.raw(f"**EXP-025 원문 인용 (R-3, PREREG §2.2·§0).** `{rel}` (sha256 `{sha(EXP25_DOC)}`), 줄 번호는 그 파일의 줄이다. 아래 코드 블록은 해당 줄을 그대로 옮긴 것이다.")
    D.raw("")
    D.raw("```text")
    for n in EXP25_LINES:
        D.raw(f"L{n}: {q[n]}")
    D.raw("```")
    h = h199_lines()
    relh = H199_DOC.relative_to(ROOT)
    D.raw("")
    D.raw(f"**TRM 쪽 C 산식 원문 인용.** `{relh}` (sha256 `{sha(H199_DOC)}`) L46. EXP-025 설계 문서에는 TRM 쪽 C 산식이 없고(`H_cycles`는 L63의 기호 표에만 나온다) 이 줄에 있다.")
    D.raw("")
    D.raw("```text")
    for n in H199_LINES:
        D.raw(f"L{n}: {h[n]}")
    D.raw("```")
    D.para("위 인용에서 C는 EXP-025 L62에서 forward당 블록 적용 수로 정의되고, looped TF의 C는 L351에서 loops와 H_layers의 곱으로 주어지며, §4.2 표(L401-L406)의 참조 코호트 `pp_base_tf_z_iter` 행과 "
           "L15의 제목이 arm들과 참조 코호트의 C가 같다는 것을 보인다. TRM 쪽 C의 산식은 `H-199.md` L46에 있다. "
           "위 인용에 없는 해석을 덧붙이지 않는다.",
           cells="Q-I3,EXP-025 (설계 문서·H-199 인용; S1_table 수치 없음)",
           ev=f"{rel} {', '.join(f'L{n}' for n in EXP25_LINES)} (sha256 {sha(EXP25_DOC)}); {relh} L46 (sha256 {sha(H199_DOC)})",
           cmd=f"sed -n '{';'.join(f'{n}p' for n in EXP25_LINES)}' {rel}; sed -n '46p' {relh}")


# =========================================================================================== DESC_iteration
def build_iteration() -> Doc:
    D = Doc("DI")
    D.raw(front("iteration", "Q-I1..Q-I5 (iteration 축)",
                extra=f"design_doc_quoted: {{path: lab/21_experiments/11_lab-experiments/EXP-015_looped-tf-direct-control.md, sha256: {sha(EXP25_DOC)}, lines: [15, 62, 351, 399, 401, 403, 404, 405, 406], h199: {{path: lab/27_theory/11_lab-theory/H-199.md, sha256: {sha(H199_DOC)}, quoted: [46], read: \"L41-47 및 키워드 grep(설계 상수·정의·판정 분기 행; run·metric 값 없음)\"}}, note: \"EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015) 설계 문서와 H-199.md L46; R-3 인용 전용(인용은 H-199.md L46뿐); RESULTS 미열람; run·metric 값 열람 없음\"}}\n"))
    D.raw("# 상태·iteration·z 종합 — 데이터 서술 초안 (iteration 축, Q-I1..Q-I5)")
    D.quote("지위: S4 서술 초안(DRAFT), description-only. 새 통계·새 임계·새 판정 낱말·인과 또는 일반화 문장을 만들지 않는다. 판정 낱말은 S1_cells에 적힌 "
            "원문 토큰을 출처 tier와 함께 재진술할 때만 나온다. D-LOCKED 셀(EXP-016, H-200)은 현황 기술(확증 아님)이며 어떤 종합 문장의 근거도 아니다.")
    D.h(2, "입력과 읽는 법")
    D.raw("- 입력: `S1r2/S1_table.csv`, `S1r2/S1_cells.csv`(S0r2 분류 포함), `INTERP_S1.md` r2. 해시는 frontmatter. 그 밖에는 Q-I3 절의 R-3 의무 인용을 위해 EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015) 설계 문서(`EXP-015_looped-tf-direct-control.md`)의 인용 줄(L15, L62, L351, L399–406)과 그 주변 줄·키워드 검색 결과, 그리고 `H-199.md`의 L41–47과 키워드 grep 결과(설계 상수·정의·판정 분기 행)를 읽었다. 인용한 것은 `H-199.md` L46뿐이다(RESULTS 미열람, run·metric 값 열람 없음). 이 문서 기준으로 다른 파일은 열지 않았다(`DESC_z.md`는 `H-199.md`를 읽지 않았고 EXP-025 설계 문서는 §4.2만 읽었다).")
    D.raw("- 수치 문장은 문단 끝 〔row_id: …〕에 S1_table row_id를 단다. 수치가 없는 문단은 \"해당 없음\"과 S1_cells 출처를 단다.")
    D.raw("- 값은 run(디렉터리)별 R-FINAL을 seed별로 나열한다. seed별 값을 합산·풀링한 값은 어디에도 없다(R-6).")
    D.raw("- S1_table의 `label_GC`는 전 행 NONE이다. 이 문서는 G/C 라벨 개수를 만들지 않는다.")
    D.raw("- `Tk_class`가 전 행 UNKNOWN이므로 k를 가로지르는 문장은 쓰지 않는다(R-5, INTERP_S1 §6). 표가 (k, seed)별 값을 나열할 뿐이다.")
    D.raw("- 여러 변형이 한 표에 있어도 표는 비교 주장을 하지 않는다. 문장은 표의 행을 가리키기만 한다.")
    assert_constants()
    symbols_table(D)

    # ------------------------------------------------------------------ Q-I1
    D.h(2, "Q-I1 — 훈련 depth와 추론 depth가 다를 때 정확도는 유지되는가 (질문 토큰: Q-INHERIT)")
    items = [("Q-I1", "H-108 sweep"), ("Q-I1", "CL-0003"), ("Q-I1", "CL-0005"), ("Q-I1", "FG-0001")]
    D.para("(Q-I1, H-108 sweep)은 S0r2 분류 A-DESC이고 key_status가 UNKNOWN(키 이름 미고정)이다. 따라서 이 셀에는 S1_table 값 행이 없다. "
           "**값 없음.** P-RELOAD 값(재평가 정확도)은 S1 입력 허용 목록 밖이라 이 문서에 나오지 않는다(INTERP_S1 (13)(14)).", cells="Q-I1,H-108 sweep")
    hdr, rows = token_rows(items[1:])
    D.raw("**표 I-0.** Q-I1 원자 토큰 (원문 재진술, 서술 아님; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para("세 원자는 서로 다른 명제이므로 각자의 토큰·범위로만 읽는다. 결합 헤드라인은 만들지 않는다(구성 명제 중 partial이 있다). "
           "CL-0003의 코호트 칸은 code, CL-0005·FG-0001은 clean-D0이다. 원자 범위가 셀 수준 관측이므로 일반화 문장을 만들지 않는다(R-6).",
           cells="Q-I1,CL-0003 / CL-0005 / FG-0001")
    D.raw("CL-0005는 noiter 훈련 셀을 다루므로 R-3 의무 문장을 붙인다.")
    D.quote(R3)

    # ------------------------------------------------------------------ Q-I2
    D.h(2, "Q-I2 — iter on/off와 z×iter 2×2 셀 (질문 토큰: Q-INHERIT)")
    items = [("Q-I2", "EXP-001"), ("Q-I2", "EXP-005"), ("Q-I2", "EXP-010"), ("Q-I2", "EXP-011"), ("Q-I2", "EXP-012/probe"),
             ("Q-I2", "EXP-012/sealed"), ("Q-I2", "EXP-013")]
    notes = {("Q-I2", "EXP-001"): "키 이름 미고정 — 값 없음", ("Q-I2", "EXP-010"): "provenance 전용(증거 셀 아님)",
             ("Q-I2", "EXP-011"): "로컬 데이터 없음(§7.0.5 예상 A-DESC ≠ S0r2 X-ABSENT, 아래 한계 문장)",
             ("Q-I2", "EXP-012/sealed"): "sealed 종점 — 이 합성에서 사용 안 함",
             ("Q-I2", "EXP-013"): "데이터 없음(미실행; 이 호스트 as_of 로컬 데이터 기준; 레지스트리 상태는 Q-S1 표)"}
    hdr, rows, ids = cell_status_table(D, items, notes)
    D.table("**표 I-1.** Q-I2 E 셀의 분류와 값 행 (S1_cells; legacy 셀은 \"" + LEG + "\"). 개수만 나열하며 코호트 간 값 비교는 하지 않는다.", hdr, rows, ids)

    # legacy EXP-005
    e5 = cell_rows("Q-I2", "EXP-005", "probe/test_exact", True)
    assert len(e5) == len(cell_rows("Q-I2", "EXP-005"))
    nv = e5.groupby(e5.run_name.map(variant)).size().to_dict()
    ks = sorted(set(e5.k.astype(int)))
    ss = sorted(set(e5.seed.astype(int)))
    assert set(e5.step.astype(float)) == {244100.0}
    D.para(f"(Q-I2, EXP-005) 셀은 코호트 legacy이다 — {LEG}. P-PROBE(`probe/test_exact`, R-FINAL) 행이 {len(e5)}개이고 서로 다른 run이 {e5.run_name.nunique()}개이며 "
           f"모두 `fig1_mlp_*`이다. 변형별 run 수는 z_iter {nv['z_iter']}, z_noiter {nv['z_noiter']}, noz_iter {nv['noz_iter']}, noz_noiter {nv['noz_noiter']}이다. "
           f"k는 {ks[0]}부터 {ks[-1]}까지 {len(ks)}개 값이고 seed는 {','.join(map(str, ss))}이다. 모든 행의 step은 {stepf(e5.iloc[0])}이다. "
           f"S1_table 칸상 {prot_desc(e5)}이다(weights는 PREREG U-2 미해결). 네 변형이 모두 있지만 이 문서는 z 또는 iter로의 귀속 문장을 쓰지 않는다(R-4: 서술만).",
           ids=e5.row_id)
    D.quote(R3)
    D.table(f"**표 I-2.** (Q-I2, EXP-005) P-PROBE R-FINAL, run별. 코호트 legacy — {LEG}. {prot_desc(e5)}.",
            RUN_HDR, run_rows(e5, True), e5.row_id)

    # (EXP-001) + guard
    D.para(f"(Q-I2, EXP-001) 셀은 key_status UNKNOWN이라 값이 없다. **값 없음.** (Q-Z4, EXP-006)·(Q-Z3, EXP-002)가 같은 `fig1_tf_*` run을 다른 키로 읽은 행은 "
           f"있으나, 이 행들로 (Q-I2, EXP-001)의 서술을 채우지 않는다(INTERP_S1 (2) 가드 ii). legacy tf 계열의 값은 이 문서의 Q-I2 절에 없다 — {LEG}.",
           cells="Q-I2,EXP-001")

    # EXP-012/probe
    e12 = cell_rows("Q-I2", "EXP-012/probe", "probe/test_exact", True)
    assert len(e12) == len(cell_rows("Q-I2", "EXP-012/probe"))
    nan12 = e12[e12.value == ""]
    dupruns = e12[e12.dup_wandb_dirs == "2"].run_name.nunique()
    ks12 = sorted(set(e12.k.astype(int)))
    ss12 = sorted(set(e12.seed.astype(int)))
    var12 = sorted(set(zip(e12.arch, e12.HL_train)))
    assert var12 == [("trm_singlez", "(3,6)")], var12
    D.para(f"(Q-I2, EXP-012/probe) 셀은 코호트 clean-D0이다. P-PROBE R-FINAL 디렉터리 행이 {len(e12)}개이고 서로 다른 run이 {e12.run_name.nunique()}개이다. "
           f"그중 {dupruns}개 run은 wandb 디렉터리가 둘이라 두 행으로 나란히 적었다. 모든 행이 `arch=trm_singlez`, (H,L)_train = (3,6)이므로 "
           f"2×2 네 셀(z_iter, z_noiter, noz_iter, noz_noiter) 중 noz_iter만 있고 z_iter, z_noiter, noz_noiter는 없다. 이 코호트에서 z×iter 귀속 문장은 쓰지 않는다(R-4: marginal, not mechanistic). "
           f"seed는 {','.join(map(str, ss12))}만, k는 {','.join(map(str, ks12))}이다. 키가 로깅되지 않아 값이 없는 행이 {len(nan12)}개 있다({', '.join(nan12.row_id)}).",
           ids=e12.row_id)
    D.table(f"**표 I-3.** (Q-I2, EXP-012/probe) P-PROBE R-FINAL, 디렉터리별(dup=2이면 같은 run의 두 디렉터리를 모두 표시). 코호트 clean-D0. {prot_desc(e12)}.",
            RUN_HDR, run_rows(e12, False), e12.row_id)
    D.para("(Q-I2, EXP-013)은 X-ABSENT이고 key_status가 UNKNOWN이다. 로컬 데이터가 없어 서술할 관측이 없다. **값 없음.** "
           "(Q-I2, EXP-010)은 X-PROV(provenance 전용, 증거 셀 아님)이며 이 문서에 값·토큰을 쓰지 않는다. "
           "(Q-I2, EXP-012/sealed)는 D-SEALED이며 \"sealed 종점 — 이 합성에서 사용 안 함\"만 쓴다. sealed 경로는 열지 않았다.",
           cells="Q-I2,EXP-013 / EXP-010 / EXP-012/sealed")
    D.para("(Q-I2, EXP-011)은 S0r2에서 X-ABSENT로 분류됐다. §7.0.5 예상 A-DESC와 다른 이유는 r1 큐 정규식 `_abl_`이 `pp_abl_*`에 걸린 거짓 양성이었고, "
           "S0r2의 앵커 정규식으로는 로컬 데이터가 없기 때문이다(S1_cells `reason` 칸). 이 셀에 대해 EXP-011이 무엇을 관측했는지는 쓰지 않는다. 한계 문장 두 개를 그대로 적는다.",
           cells="Q-I2,EXP-011")
    D.quote(EXP011_1)
    D.quote(EXP011_2)
    items = [("Q-I2", "H-022"), ("Q-I2", "H-023"), ("Q-I2", "H-025")]
    hdr, rows = token_rows(items, special={("Q-I2", "H-022"): "(card-level, 성분 귀속 미확정 — 판정 낱말은 Q-S1 표에서만 원문 인용; 이 문서에서는 쓰지 않음)"})
    D.raw("**표 I-4.** Q-I2 가설 토큰 (원문 재진술, 서술 아님; 출처 S1_cells; 코호트 칸 없음).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para("H-023·H-025의 토큰은 HYPOTHESES.md 상태 칸(tier 4)이다. 가설 토큰은 가설 행에만 귀속되고 위 EXP 셀의 값으로 옮겨 붙지 않는다. "
           "원인 문장은 'consistent with optimization-not-capacity'를 넘지 않는다(R-8); 이 문서는 원인 문장을 쓰지 않는다.", cells="Q-I2,H-022 / H-023 / H-025")

    # ------------------------------------------------------------------ Q-I3
    D.h(2, "Q-I3 — weight-tied looped TF는 TRM 대비 어떤가 — EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015) (질문 토큰: Q-INHERIT)")
    items = [("Q-I3", "EXP-025"), ("Q-I3", "EXP-022"), ("Q-I3", "EXP-023")]
    notes = {("Q-I3", "EXP-022"): "자체 run 없음(값 공백 범주 ii, INTERP_S1 (3)) — 값 없음",
             ("Q-I3", "EXP-023"): "D-PREJUDGE: 판정 전 — 상태만, 값 열람 없음"}
    hdr, rows, ids = cell_status_table(D, items, notes)
    D.table("**표 I-5.** Q-I3 E 셀의 분류와 값 행 (S1_cells).", hdr, rows, ids)
    e25 = cell_rows("Q-I3", "EXP-025", "probe/test_exact", True)
    assert len(e25) == len(cell_rows("Q-I3", "EXP-025")) and set(e25.step) == {"122050"}
    arms = e25.run_name.str.extract(r"pp_ltf_(\w+?)_lay2")[0]
    an = arms.value_counts().to_dict()
    ks25 = sorted(set(e25.k.astype(int)))
    D.para(f"EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015)의 (Q-I3, EXP-025) 셀은 코호트 clean-D1이다. P-PROBE R-FINAL 행이 {len(e25)}개이고 서로 다른 run이 "
           f"{e25.run_name.nunique()}개이며 모두 `arch=transformers_baseline`이다. run 이름의 arm별 run 수는 fullgrad {an['fullgrad']}, g7 {an['g7']}, noinj {an['noinj']}이다. "
           f"k는 {','.join(map(str, ks25))}, seed는 1,2,3이다. 모든 행의 step은 {stepf(e25.iloc[0])}(마지막 logged step)이다. "
           f"이 표의 값은 R-FINAL이며 소스가 판정에 쓴 환원(R-ANY, 획득)의 값이 아니다. S1_table 칸상 {prot_desc(e25)}이다. "
           "arm 사이의 우열·방향·기제 문장은 쓰지 않는다(R-7).", ids=e25.row_id)
    exp25_quote(D)
    D.table(f"**표 I-6.** (Q-I3, EXP-025) P-PROBE R-FINAL, run별. 코호트 clean-D1. {prot_desc(e25)}.",
            RUN_HDR, run_rows(e25, False), e25.row_id)
    D.para("(Q-I3, EXP-022)는 자체 run이 없어 값 없음이며, EXP-025 run을 빌려 채우지 않는다. (Q-I3, EXP-023)은 상태만(D-PREJUDGE)이며 run 수와 값은 이 문서에 쓰지 않는다(Q-S1 표 참조).",
           cells="Q-I3,EXP-022 / EXP-023")
    items = [("Q-I3", "FG-0006"), ("Q-I3", "H-199")]
    hdr, rows = token_rows(items)
    D.raw("**표 I-7.** Q-I3 토큰 (원문 재진술; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    h199 = cell("Q-I3", "H-199")
    D.quote(f"H-199 DISCREPANCY (S1_cells 토큰 불일치 칸 원문): {h199.token_discrepancy}")
    D.para(f"H-199의 토큰은 tier 2(status 파일)이며 위 인용처럼 tier 4 칸에는 낱말이 서로 다르게 있다. 이 문서는 tier 2 토큰을 따르고 "
           "C-LTF 하위 claim의 방향·기제 문장을 쓰지 않는다. FG-0006의 토큰은 그림 서술의 범위이며 H-199 또는 C-LTF의 판정이 아니다.", cells="Q-I3,H-199 / FG-0006")

    # ------------------------------------------------------------------ Q-I4
    D.h(2, "Q-I4 — 적응적 iteration(ACT·deep supervision 예산)은 무엇을 바꾸는가 (질문 토큰: Q-DESC-ONLY)")
    D.quote(f"절 전체 표지: {MARK}. 이 절에서 verdict 낱말은 D-LOCKED 셀 자기 토큰의 원문 재진술(표지 동반)뿐이다.")
    items = [("Q-I4", "EXP-011"), ("Q-I4", "EXP-016"), ("Q-I4", "EXP-020"), ("Q-I4", "H-200")]
    notes = {("Q-I4", "EXP-011"): "로컬 데이터 없음(§7.0.5 예상 A-DESC ≠ S0r2 X-ABSENT)", ("Q-I4", "EXP-016"): "D-LOCKED: 현황 기술(확증 아님)",
             ("Q-I4", "EXP-020"): "D-PREJUDGE: 상태만", ("Q-I4", "H-200"): "D-LOCKED"}
    hdr, rows, ids = cell_status_table(D, items, notes)
    D.table("**표 I-8.** Q-I4 셀의 분류와 값 행 (S1_cells). EXP-016은 두 프로토콜 키 × 디렉터리 행이다.", hdr, rows, ids)
    D.para("(Q-I4, EXP-011)은 X-ABSENT이다(한계 문장은 Q-I2 절의 두 문장과 같다). (Q-I4, EXP-020)은 D-PREJUDGE이며 소유 트랙의 judge 전이라 상태만 쓰고 값은 읽지 않았다(run 수는 Q-S1 표).",
           cells="Q-I4,EXP-011 / EXP-020")
    D.quote(EXP011_1)
    D.quote(EXP011_2)
    p16 = cell_rows("Q-I4", "EXP-016", "probe/test_exact", True)
    a16 = cell_rows("Q-I4", "EXP-016", "all/exact_accuracy", True)
    assert len(p16) == len(cell_rows("Q-I4", "EXP-016", "probe/test_exact")) and len(a16) == len(cell_rows("Q-I4", "EXP-016", "all/exact_accuracy"))
    assert (p16.run_name.nunique() == a16.run_name.nunique()) and (set(p16.run_name) == set(a16.run_name))
    arm16 = p16.run_name.str.extract(r"e016_(.+?)_k\d")[0]
    an16 = arm16.groupby(arm16).apply(lambda x: p16.loc[x.index, "run_name"].nunique()).to_dict()
    dup16 = p16[p16.dup_wandb_dirs == "2"].run_name.nunique()
    D.para(f"(Q-I4, EXP-016) 셀은 코호트 clean-D1이고 D-LOCKED이다({MARK}). 서로 다른 run은 {p16.run_name.nunique()}개이며 키마다 디렉터리 행이 {len(p16)}개씩 있다 — "
           f"이 중 {dup16}개 run은 wandb 디렉터리가 둘이라 두 행으로 나란히 적었다. run 이름의 arm별 run 수는 "
           + ", ".join(f"`e016_{k}` {v}" for k, v in sorted(an16.items())) + "이다. "
           f"두 키는 프로토콜이 달라(P-ALL = `all/exact_accuracy`, online raw; P-PROBE = `probe/test_exact`) 같은 표에 병치하지 않고 표 I-9·I-10으로 나눴다. 프로토콜이 달라 직접 비교 불가하다(R-12). "
           "step은 디렉터리별 마지막 logged step이며 값이 서로 다르다(표의 step 칸). 이 값들은 어떤 원자·가설의 강도를 올리거나 내리는 근거로 쓰지 않는다(PREREG §6.2).",
           ids=list(p16.row_id) + list(a16.row_id))
    D.table(f"**표 I-9.** (Q-I4, EXP-016) P-PROBE `probe/test_exact` R-FINAL, 디렉터리별. 코호트 clean-D1. {MARK}. {prot_desc(p16)}.",
            RUN_HDR, run_rows(p16, False), p16.row_id, flt="Q-I4|EXP-016|probe/test_exact")
    D.table(f"**표 I-10.** (Q-I4, EXP-016) P-ALL `all/exact_accuracy` R-FINAL, 디렉터리별. 코호트 clean-D1. {MARK}. {prot_desc(a16)}. 표 I-9와 프로토콜이 달라 직접 비교 불가.",
            RUN_HDR, run_rows(a16, False), a16.row_id, flt="Q-I4|EXP-016|all/exact_accuracy")
    r16, rh = cell("Q-I4", "EXP-016"), cell("Q-I4", "H-200")
    D.raw(f"EXP-016 자기 토큰 원문 재진술 (출처: {r16.token_file} · {r16.token_locator}; {r16.token_discrepancy}):")
    D.quote(f"{r16.token_restatement} — {MARK}")
    D.raw(f"H-200 토큰 원문 재진술 (출처: {rh.token_file} · {rh.token_locator}):")
    D.quote(f"{rh.token_restatement} — {MARK}")
    D.para("위 두 인용은 S1_cells의 재진술 칸을 그대로 옮긴 것이다. 이 문서는 이 토큰을 새 증거로 쓰지 않고, 다른 셀의 판정 강도를 바꾸는 데 쓰지 않는다.",
           cells="Q-I4,EXP-016 / H-200")

    # ------------------------------------------------------------------ Q-I5
    D.h(2, "Q-I5 — 총 재귀량 H·L 하나로 요약되는가, H/L 분할이 중요한가 (질문 토큰: Q-INHERIT)")
    hdr, rows = token_rows([("Q-I5", "CL-0004"), ("Q-I5", "FG-0002")])
    D.raw("**표 I-11.** Q-I5 원자 토큰 (원문 재진술; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para("Q-I5에는 E 셀이 없고 S1_table 행도 없다. 따라서 이 절은 두 원자의 토큰·범위만 재진술한다. CL-0004는 분할이 중요하다는 것만을 말하며 방향은 말하지 않는다. "
           "원자 본문이 두 원자 모두 seed당 셀 수준 관측임을 밝히고 있어 일반화 문장을 만들지 않는다(R-6). 코호트는 clean-D0이다.", cells="Q-I5,CL-0004 / FG-0002")

    # ------------------------------------------------------------------ appendix
    D.h(2, "부록 A. 값 없는 셀과 값을 쓰지 않은 셀 (iteration 축)")
    uk = C[(C.key_status == "UNKNOWN") & C.q.str.startswith("Q-I")]
    D.para("S1_cells에서 key_status가 UNKNOWN인 셀 가운데 이 문서(iteration 축)에 속하는 것은 아래 표와 같다. 나머지 UNKNOWN 셀은 DESC_z.md에 있다. 이 셀들은 키 이름이 미고정이라 값 행이 없다. **값 없음.** "
           "(Q-I2, EXP-013)은 UNKNOWN이면서 X-ABSENT(데이터 없음)이다.", cells="key_status == UNKNOWN")
    D.raw("**표 I-A0.** key_status UNKNOWN 셀 (iteration 축; S1_cells).")
    D.raw("")
    D.raw("| (q, s) | S0r2 분류 | key_status | S1 값 행 | 비고 |")
    D.raw("|---|---|---|---|---|")
    for a, b, cl in zip(uk.q, uk.s, uk.s1_class):
        D.raw(f"| ({a}, {b}) | {cl} | UNKNOWN | {NOVAL} | {LEG if (a, b) in LEGSRC else '—'} |")
    D.raw("")
    rows = [
        ["(Q-I3, EXP-022)", "A-DESC", "NAMED-ADMITTED", "자체 run 없음 — 값 공백 범주 (ii)(INTERP_S1 (3))"],
        ["(Q-I2, EXP-011)·(Q-I4, EXP-011)", "X-ABSENT", "NAMED-ADMITTED", "로컬 run 없음"],
        ["(Q-I3, EXP-023)·(Q-I4, EXP-020)", "D-PREJUDGE", "NAMED-ADMITTED", "판정 전: 값 열람 없음(상태만)"],
        ["(Q-I2, EXP-012/sealed)", "D-SEALED", "—", "sealed 종점 — 사용 안 함"],
        ["(Q-I2, EXP-010)", "X-PROV", "—", "provenance 전용"],
    ]
    D.raw("**표 I-A.** key_status UNKNOWN이 아니지만 값 행이 없거나 값을 쓰지 않은 셀 (S1_cells; 수치 없음).")
    D.raw("")
    D.raw("| (q, s) | S0r2 분류 | key_status | 사유 |")
    D.raw("|---|---|---|---|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.h(2, "부록 B. 이 초안이 확인하지 않은 전제")
    for s in [
        "(Q-I3) C 계산량 매칭: "+R11+" 설계 문서에서 C의 정의(L62)·looped TF의 C 산식(L351)·§4.2 표 헤더·arm 행·참조 코호트 T 행(L399–406)·제목의 \"matched C\"(L15)를 원문 인용했고, "
        "TRM 쪽 C 산식은 `H-199.md` L46에서 원문 인용했다(EXP-025 설계 문서 자체에는 TRM 쪽 산식이 없다 — `H_cycles`는 L63 기호 표에만 있음, grep 확인). "
        "\"TRM 쪽 C가 왜 그 값인가\"를 유도하는 별도 문장은 이 문서에서 찾지 못했다(L46의 산식이 그 값을 준다). 읽기 범위: 위 인용 줄과 그 주변 줄·키워드 검색 결과, `H-199.md` L41–47과 키워드 grep 결과(인용은 L46뿐). "
        "EXP-025 RESULTS·다른 sealed/preds/z_snapshots 경로는 열지 않았다.",
        "run이 n=10 경로(`data/sigma_k_10/<k>` 등)에서 선택됐다는 전제는 S1_table에 data_path 칸이 없어 검증하지 않았다. 코호트는 S1 `cohort` 칸(project + 시작 시각 규칙)을 따른다.",
        f"legacy 행의 코호트는 S1_table `cohort` 칸을 그대로 따른 것이며 t_build 시간대(INTERP_S1 TN-6)는 미해결이다 — {LEG}.",
        "P-PROBE의 `weights`(EMA 여부)는 S1의 모든 코호트에서 `probe/test_exact` 전 행이 UNKNOWN이다(PREREG U-2). eval_set_size는 S1 칸 값이며 메모리 기재(500 대 1000)와의 불일치(U-1)는 해소하지 않았다.",
        "EXP-016의 frontmatter에 verifier PASS 기록이 없다(S1_cells 토큰 칸의 DISCREPANCY). 이 문서는 그 판정을 물려받지 않는다.",
        "S1_table의 값을 wandb 로컬 datastore에서 다시 읽어 대조하지 않았다(V1 몫).",
    ]:
        D.raw("- " + s)
    D.raw("")
    ledger(D)
    return D


# =========================================================================================== DESC_z
def build_z() -> Doc:
    D = Doc("DZ")
    D.raw(front("z", "Q-Z1..Q-Z7 (z 축)",
                extra=f"design_doc_read: {{path: lab/21_experiments/11_lab-experiments/EXP-015_looped-tf-direct-control.md, sha256: {sha(EXP25_DOC)}, note: \"{R11} 설계 문서 §4.2 표(L399-406)를 SYMBOLS의 arm 행 설명을 위해 읽음; 인용 없음; RESULTS 미열람; run·metric 값 열람 없음\"}}\n"))
    D.raw("# 상태·iteration·z 종합 — 데이터 서술 초안 (z 축, Q-Z1..Q-Z7)")
    D.quote("지위: S4 서술 초안(DRAFT), description-only. 새 통계·새 임계·새 판정 낱말·인과 또는 일반화 문장을 만들지 않는다. 판정 낱말은 S1_cells에 적힌 "
            "원문 토큰을 출처 tier와 함께 재진술할 때만 나온다. D-LOCKED 셀(EXP-017, H-200, H-201)은 현황 기술(확증 아님)이며 어떤 종합 문장의 근거도 아니다.")
    D.h(2, "입력과 읽는 법")
    D.raw("- 입력: `S1r2/S1_table.csv`, `S1r2/S1_cells.csv`(S0r2 분류 포함), `INTERP_S1.md` r2. 해시는 frontmatter. 그 밖에는 SYMBOLS의 arm 행 설명을 위해 "+R11+" 설계 문서(`EXP-015_looped-tf-direct-control.md`) §4.2 표(L399–406)만 읽었다(인용 없음, RESULTS 미열람, run·metric 값 열람 없음). 이 문서 기준으로 다른 파일은 열지 않았다(`DESC_iteration.md`는 별도로 `H-199.md` L41–47과 키워드 grep 결과를 읽었다).")
    D.raw("- 수치 문장은 문단 끝 〔row_id: …〕에 S1_table row_id를 단다. 수치가 없는 문단은 \"해당 없음\"과 S1_cells 출처를 단다.")
    D.raw("- 값은 run(디렉터리)별 R-FINAL을 seed별로 나열한다. seed별 값을 합산·풀링한 값은 어디에도 없다(R-6).")
    D.raw("- S1_table의 `label_GC`는 전 행 NONE이다. 이 문서는 G/C 라벨 개수를 만들지 않는다.")
    D.raw("- `Tk_class`가 전 행 UNKNOWN이므로 k를 가로지르는 문장은 쓰지 않는다(R-5, INTERP_S1 §6).")
    D.raw("- 같은 run이 여러 (q, s) 셀에 나오면 표 Z-1에서 run 한 줄로 합치고 셀별 row_id를 모두 적는다. run은 질문마다 한 번만 센다.")
    assert_constants()
    symbols_table(D, extra=[("PR(z_H)", "wandb `z/eff_rank`(훈련 중 train probe). `trm`에만 존재. S1_table에서 이 키의 행은 (Q-Z3, EXP-002) 셀(legacy)뿐이다 — " + LEG + ". PR(z_L)은 S1_table에 행이 없다(forward 재계산 전용)."),
                            ("owner", f"S1_table `exp_owner`: run을 EXP에 귀속한 결과(EXPERIMENTS.md 우선). (Q-Z1, EXP-004) 셀의 owner는 EXP-001/005/007/008일 수 있다 — {LEG}.")])

    legacy_cells = [("Q-Z1", "EXP-004"), ("Q-Z1", "EXP-005"), ("Q-Z1", "EXP-007"), ("Q-Z2", "EXP-004"), ("Q-Z2", "EXP-007"), ("Q-Z4", "EXP-006")]
    short = {("Q-Z1", "EXP-004"): "Z1·004", ("Q-Z1", "EXP-005"): "Z1·005", ("Q-Z1", "EXP-007"): "Z1·007",
             ("Q-Z2", "EXP-004"): "Z2·004", ("Q-Z2", "EXP-007"): "Z2·007", ("Q-Z4", "EXP-006"): "Z4·006"}
    L = pd.concat([cell_rows(q, s, "probe/test_exact") for q, s in legacy_cells])
    L = L.assign(cellkey=[short[(q, s)] for q, s in zip(L.q, L.s)])
    groups = []
    for wid, g in L.groupby("wandb_run_id", sort=False):
        for col in ["run_name", "value", "step", "evidence_use", "k", "seed", "arch", "HL_train", "cohort", "dup_wandb_dirs"]:
            assert g[col].nunique() == 1, (wid, col)
        groups.append(g)
    # (each legacy run appears once per cell; EXP-004 covers all)
    ev = [g for g in groups if g.evidence_use.iloc[0] == "yes"]
    ex = [g for g in groups if g.evidence_use.iloc[0] != "yes"]
    assert all(len(g) == len(set(g.cellkey)) for g in groups)

    # ------------------------------------------------------------------ Q-Z1
    D.h(2, "Q-Z1 — z-carry(trm) vs single-z(trm_singlez)의 k별 G/C 패턴 (질문 토큰: Q-INHERIT)")
    D.para("EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015)는 z 축에서 참조 코호트 `pp_base_tf_z_iter`의 후보로만 나온다.", cells="Q-Z1,EXP-025")
    items = [("Q-Z1", "EXP-001"), ("Q-Z1", "EXP-004"), ("Q-Z1", "EXP-005"), ("Q-Z1", "EXP-007"), ("Q-Z1", "EXP-010"), ("Q-Z1", "EXP-012/probe"),
             ("Q-Z1", "EXP-012/sealed"), ("Q-Z1", "EXP-025")]
    notes = {("Q-Z1", "EXP-001"): "키 이름 미고정 — 값 없음",
             ("Q-Z1", "EXP-004"): "후보 = Sigma_k_fig12 전체(INTERP_S1 (4)); owner는 행마다 다름; legacy, 신규 clean 코호트와 비교 불가",
             ("Q-Z1", "EXP-025"): "참조 코호트 pp_base_tf_z_iter: 후보 행은 있으나 owner UNKNOWN이라 증거 사용 0 — 값 없음",
             ("Q-Z1", "EXP-010"): "provenance 전용(증거 셀 아님)", ("Q-Z1", "EXP-012/sealed"): "sealed 종점 — 이 합성에서 사용 안 함"}
    hdr, rows, ids = cell_status_table(D, items, notes)
    D.table("**표 Z-0.** Q-Z1 E 셀의 분류와 값 행 (S1_cells; legacy 행은 \"" + LEG + "\"). 개수만 나열하며 코호트 간 값 비교는 하지 않는다.", hdr, rows, ids)

    e4 = cell_rows("Q-Z1", "EXP-004", "probe/test_exact")
    e4y = e4[e4.evidence_use == "yes"]
    own = e4y.exp_owner.value_counts().to_dict()
    blocks = e4y.groupby(["block", "arch", "HL_train"]).size()
    nlog_missing = [g for g in ev if g.value.iloc[0] == ""]
    pfx = lambda g: re.match(r"^([A-Za-z0-9]+?)_", g.run_name.iloc[0]).group(1)
    z1p, z2p = Counter(pfx(g) for g in ev), Counter(pfx(g) for g in ex)         # run counts per name prefix in Table Z-1 / Table Z-2
    fmt = lambda ps, cnt: ", ".join(f"`{x}_*` {cnt[x]}개" for x in ps)
    assert not ({g.run_name.iloc[0] for g in ev} & {g.run_name.iloc[0] for g in ex}), "a run is in both Table Z-1 and Table Z-2"
    assert "fig1" not in z2p
    parts = []
    for x in sorted((set(z1p) | set(z2p)) - {"fig1"}):          # the prefix FAMILY is the subject; both counts are stated (B-r6-1)
        a_, b_ = z1p.get(x, 0), z2p.get(x, 0)
        if a_ and b_:
            parts.append(f"`{x}_*` 이름의 run은 표 Z-1에 {a_}개(증거 사용)가 있고, 이와 별개로 {b_}개가 증거 사용에서 제외돼 표 Z-2에 있다(이 {b_}개는 표 Z-1에 없다).")
        elif a_:
            parts.append(f"`{x}_*` 이름의 run은 표 Z-1에 {a_}개(증거 사용)가 있고 표 Z-2에는 없다.")
        else:
            parts.append(f"`{x}_*` 이름의 run은 표 Z-1에 없고, {b_}개 전부 증거 사용에서 제외돼 표 Z-2에만 있다.")
    prefix_sentence = "그 밖의 이름 계열은 다음과 같다. " + " ".join(parts)
    D.para(f"Q-Z1의 legacy 증거는 하나의 run 표(표 Z-1)로 합쳤다. (Q-Z1, EXP-004) 셀은 코호트 legacy이다 — {LEG}. 후보 행 {len(e4)}개는 서로 다른 run {e4.run_name.nunique()}개이며 "
           f"증거 사용 `yes`는 {len(e4y)}개, 제외는 {len(e4) - len(e4y)}개이다. 증거 사용 행의 owner별 run 수는 "
           + ", ".join(f"{k} {v}" for k, v in sorted(own.items())) + f"이다. owner EXP-005의 {own.get('EXP-005', 0)}개 run은 (Q-Z1, EXP-005) 셀과, "
           f"owner EXP-007의 {own.get('EXP-007', 0)}개 run은 (Q-Z1, EXP-007) 셀의 run과 같은 run이므로 한 번만 센다. "
           f"owner EXP-001의 {own.get('EXP-001', 0)}개 run은 `fig1_tf_*`이며 (Q-Z1, EXP-004) 셀의 행으로만 서술한다(EXP-001 셀의 값이 아니다). {prefix_sentence} "
           + (f"owner EXP-008은 PREREG source_exp_ids 밖이며 INTERP_S1 (4)(iii)에 따라 (Q-Z1, EXP-004) 셀 행과 그 거울인 (Q-Z2, EXP-004) 셀 행(데이터 근거 Sigma_k_fig12 전체 범위의 run, owner는 해소된 EXP)으로만 서술한다. " if "EXP-008" in own else "")
           + (f"증거 사용 run 중 키가 로깅되지 않아 값이 없는 run이 {len(nlog_missing)}개 있다. " if nlog_missing else "증거 사용 run은 모두 값이 기록돼 있다. ") + f"아래 표는 모든 변형을 한 표에 나열하지만 변형 간 비교는 하지 않는다.",
           ids=e4.row_id)
    D.quote(R3)
    D.quote(R5)

    def z1_rows(groups_):
        rows_ = []
        gs = sorted(groups_, key=lambda g: (g.block.iloc[0], g.arch.iloc[0], g.HL_train.iloc[0], int(g.k.iloc[0]), int(g.seed.iloc[0]), g.run_name.iloc[0], g.wandb_run_id.iloc[0]))
        for g in gs:
            r = g.iloc[0]
            ids_ = "; ".join(f"{ck}:{rid}" for ck, rid in sorted(zip(g.cellkey, g.row_id), key=lambda z: list(short.values()).index(z[0])))
            rows_.append([r.run_name, r.wandb_run_id, r.k, r.seed, r.arch, r.HL_train, val(r), stepf(r), r.exp_owner, LEG, ids_])
        return rows_

    hdr = ["run_name", "wandb_run_id", "k", "seed", "arch", "(H,L)_train", "value (P-PROBE R-FINAL)", "step", "owner", "코호트 / cohort_note", "row_id (셀별: Z1·004 = Q-Z1/EXP-004, Z1·005, Z1·007, Z2·004, Z2·007, Z4·006)"]
    ev_legacy = [g for g in ev if g.cohort.iloc[0] == "legacy"]
    assert len(ev_legacy) == len(ev)
    all_ids = [rid for g in ev for rid in g.row_id]
    D.table(f"**표 Z-1.** legacy 증거 사용 run(합친 표), P-PROBE `probe/test_exact` R-FINAL, run별. 코호트 legacy — {LEG}. 모든 셀의 같은 run은 같은 값·step을 가진다(스크립트 검증). "
            f"probe_size=512, weights=UNKNOWN, eval_set_size=UNKNOWN. Z4·006 항목은 Q-Z4 pin 키 R-FINAL이며 τ가 아니다.",
            hdr, z1_rows(ev), all_ids,
            flt="Q-Z1|EXP-004|probe/test_exact;Q-Z1|EXP-005|probe/test_exact;Q-Z1|EXP-007|probe/test_exact;Q-Z2|EXP-004|probe/test_exact;Q-Z2|EXP-007|probe/test_exact;Q-Z4|EXP-006|probe/test_exact (evidence_use=yes만)")

    exrows = []
    for g in sorted(ex, key=lambda g: g.row_id.min()):
        r = g.iloc[0]
        ids_ = "; ".join(f"{ck}:{rid}" for ck, rid in sorted(zip(g.cellkey, g.row_id), key=lambda z: list(short.values()).index(z[0])))
        exrows.append([r.run_name, r.wandb_run_id, r.cohort, LEG if r.cohort == "legacy" else "—", r.exp_owner, r.evidence_use, ids_])
    exids = [rid for g in ex for rid in g.row_id]
    D.table(f"**표 Z-2.** legacy 후보 중 증거 사용에서 제외된 run (값을 쓰지 않음; 사유와 row_id만). {LEG}.", ["run_name", "wandb_run_id", "cohort", "cohort_note", "owner", "evidence_use", "row_id (셀:row_id)"],
            exrows, exids)
    D.para(f"제외 run은 {len(ex)}개이며 사유는 cohort UNKNOWN(INTERP_S1 (6) 규칙상 코호트 조건 미확인) 또는 owner UNKNOWN(귀속 지도에 후보 없음)이다. 이 run들의 값은 이 문서의 어떤 문장에도 쓰지 않았다. "
           f"코호트 UNKNOWN 행은 \"{LEG}\" 문자열을 붙일 legacy 확정 행이 아니므로 별도 표에 두었다. 제외 run 중 cohort가 legacy인 run은 {sum(g.cohort.iloc[0] == 'legacy' for g in ex)}개이다 — {LEG}.", ids=exids)

    e5 = cell_rows("Q-Z1", "EXP-005", "probe/test_exact", True)
    D.para(f"(Q-Z1, EXP-005) 셀은 코호트 legacy이다 — {LEG}. 증거 사용 행 {len(e5)}개(`fig1_mlp_*` run {e5.run_name.nunique()}개, 네 변형 모두 존재)이고 값은 표 Z-1에 있다. "
           f"(Q-Z1, EXP-007) 셀은 행 {len(cell_rows('Q-Z1', 'EXP-007'))}개 중 증거 사용 {len(cell_rows('Q-Z1', 'EXP-007', None, True))}개(`fig1d_*`, k={','.join(map(str, sorted(set(cell_rows('Q-Z1', 'EXP-007', None, True).k.astype(int)))))})이고 "
           f"나머지는 cohort UNKNOWN으로 제외됐다. 두 셀의 run은 모두 (Q-Z1, EXP-004) 셀에도 있다(표 Z-1의 row_id 칸). `fig1_tf_*`·`fig1_mlp_*` 모두 네 변형(z/noz × iter/noiter)이 있으나 이 문서는 z 또는 iter로의 귀속 문장을 쓰지 않는다(R-4: 서술만).",
           ids=list(e5.row_id) + list(cell_rows("Q-Z1", "EXP-007").row_id))

    # clean-D0
    e12 = cell_rows("Q-Z1", "EXP-012/probe", "probe/test_exact", True)
    assert len(e12) == len(cell_rows("Q-Z1", "EXP-012/probe"))
    assert sorted(set(zip(e12.arch, e12.HL_train))) == [("trm_singlez", "(3,6)")]
    nan12 = e12[e12.value == ""]
    D.para(f"(Q-Z1, EXP-012/probe) 셀은 코호트 clean-D0이며 legacy 표(표 Z-1; {LEG})와 별도 표로 둔다. P-PROBE R-FINAL 디렉터리 행 {len(e12)}개, 서로 다른 run {e12.run_name.nunique()}개이다. "
           f"모든 행이 `arch=trm_singlez`, (H,L)_train = (3,6)이므로 z_iter, z_noiter, noz_noiter 셀이 이 코호트에 없다 — z-carry 대 single-z 대조는 서술할 수 없다(R-4). "
           f"k는 {','.join(map(str, sorted(set(e12.k.astype(int)))))}, seed는 {','.join(map(str, sorted(set(e12.seed.astype(int)))))}이다. 키 미로깅으로 값이 없는 행이 {len(nan12)}개({', '.join(nan12.row_id)})이다.",
           ids=e12.row_id)
    D.table(f"**표 Z-3.** (Q-Z1, EXP-012/probe) P-PROBE R-FINAL, 디렉터리별. 코호트 clean-D0. {prot_desc(e12)}.", RUN_HDR, run_rows(e12, False), e12.row_id)

    e25 = cell_rows("Q-Z1", "EXP-025", "probe/test_exact")
    assert set(e25.evidence_use) == {"no: exp_owner UNKNOWN"}
    D.para(f"EXP-025 (looped-tf-direct-control; legacy 별칭 EXP-015)의 (Q-Z1, EXP-025) 셀은 참조 코호트 `pp_base_tf_z_iter`(코호트 clean-D1)의 후보 행 {len(e25)}개를 가지나 "
           f"전부 owner UNKNOWN(귀속 지도에 후보 없음)이라 증거 사용이 0개이다. **값 없음.** 이 문서는 이 행들의 값을 쓰지 않는다.", ids=e25.row_id)
    D.para("(Q-Z1, EXP-001)은 키 이름 미고정(UNKNOWN)이라 값 행이 없다. **값 없음.** (Q-Z1, EXP-010)은 X-PROV(provenance 전용)이고, (Q-Z1, EXP-012/sealed)는 D-SEALED(sealed 종점 — 이 합성에서 사용 안 함)이다. sealed 경로는 열지 않았다.",
           cells="Q-Z1,EXP-001 / EXP-010 / EXP-012/sealed")
    items = [("Q-Z1", "H-023"), ("Q-Z1", "H-025"), ("Q-Z1", "H-A")]
    hdr2, rows2 = token_rows(items)
    D.raw("**표 Z-4.** Q-Z1 가설 토큰 (원문 재진술; H-A는 상태 낱말이며 판정이 아님; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr2) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr2)) + "|")
    for r in rows2:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para("각 가설 토큰은 가설 행에만 귀속되며 위 EXP 셀의 값으로 옮겨 붙지 않는다. k를 가로지르는 문장은 쓰지 않으므로(R-5, Tk_class UNKNOWN) \"z-carry와 single-z의 k별 패턴\"에 대한 문장은 표가 대신한다. "
           "원인 문장은 'consistent with optimization-not-capacity'를 넘지 않는다(R-8); 이 문서는 원인 문장을 쓰지 않는다.", cells="Q-Z1,H-023 / H-025 / H-A")

    # ------------------------------------------------------------------ Q-Z2
    D.h(2, "Q-Z2 — collapse 현상학과 ceiling 이후 유지 (질문 토큰: Q-INHERIT)")
    z2_004 = cell_rows("Q-Z2", "EXP-004", "probe/test_exact")
    z2_007 = cell_rows("Q-Z2", "EXP-007", "probe/test_exact")
    # equality check with Q-Z1 mirrors
    for qs, a, b in [("004", cell_rows("Q-Z1", "EXP-004", "probe/test_exact"), z2_004), ("007", cell_rows("Q-Z1", "EXP-007", "probe/test_exact"), z2_007)]:
        m = a.merge(b, on=["wandb_run_id"], suffixes=("_1", "_2"))
        assert len(m) == len(a) == len(b)
        for col in ["run_name", "value", "step", "evidence_use", "cohort", "exp_owner"]:
            assert (m[col + "_1"] == m[col + "_2"]).all(), (qs, col)
    items = [("Q-Z2", "EXP-001"), ("Q-Z2", "EXP-004"), ("Q-Z2", "EXP-007")]
    notes = {("Q-Z2", "EXP-001"): "키 이름 미고정 — 값 없음", ("Q-Z2", "EXP-004"): "Q-Z1 EXP-004와 같은 run·같은 값(표 Z-1의 Z2·004 항목)",
             ("Q-Z2", "EXP-007"): "Q-Z1 EXP-007과 같은 run·같은 값(표 Z-1의 Z2·007 항목)"}
    hdr, rows, ids = cell_status_table(D, items, notes)
    D.table("**표 Z-5.** Q-Z2 E 셀의 분류와 값 행 (S1_cells; legacy 행은 \"" + LEG + "\").", hdr, rows, ids)
    D.para(f"(Q-Z2, EXP-004)와 (Q-Z2, EXP-007) 셀은 코호트 legacy이다 — {LEG}. 행은 각각 {len(z2_004)}개와 {len(z2_007)}개이고, 같은 run의 (Q-Z1) 셀 행과 run 이름·값·step·증거 사용이 모두 일치한다(스크립트 검증, 일치 쌍 수 "
           f"{len(z2_004)}와 {len(z2_007)}). 값은 표 Z-1에서 해당 run의 `Z2·004`/`Z2·007` row_id 항목으로 읽는다. 이 문서는 값을 다시 나열하지 않는다.",
           ids=list(z2_004.row_id) + list(z2_007.row_id) + list(cell_rows("Q-Z1", "EXP-004", "probe/test_exact").row_id) + list(cell_rows("Q-Z1", "EXP-007", "probe/test_exact").row_id))
    D.quote(R3)
    D.quote(R5)
    D.para("S1_table은 run별 최종 값(R-FINAL) 하나만 가지며 train_exact와 쌍을 이룬 값, 시간 경로, 소스 동결 G/C 라벨(`label_GC` 전 행 NONE)이 없다. 따라서 \"H-collapse가 어떤 (코호트, k)에서 나타나는가\"와 "
           "\"ceiling 이후 유지되는가\"는 이 문서에서 서술하지 않는다. 시간 경로에 관한 서술은 원자 CL-0001의 토큰·범위로만 한다.", cells="Q-Z2,EXP-004 / EXP-007 (행 없음 라벨)")
    hdr, rows = token_rows([("Q-Z2", "CL-0001"), ("Q-Z2", "CL-0002"), ("Q-Z2", "H-A"), ("Q-Z2", "H-023")])
    D.raw("**표 Z-6.** Q-Z2 토큰 (원문 재진술; H-A는 상태 낱말; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para("CL-0001의 hazard 서술은 D1 관측 창 안에서 우측 절단(right-censored)되어 있고 그 창 밖의 hazard에 대해서는 아무것도 말하지 않는다. 마지막 관측 시점에 ceiling에 있던 run은 생존이 아니라 절단으로 읽으며, 이 문서의 어떤 표도 시간 경로를 담고 있지 않다.", cells="Q-Z2,CL-0001")
    D.para(f"CL-0001은 clean-D1 코호트의 원자이며 legacy 또는 clean-D0 행으로 옮기지 않는다. CL-0002는 코드 사실이며, 코호트가 다른 값을 같은 결과로 합쳐 읽는 것을 막는 근거로만 쓴다(legacy 행에 \"{LEG}\"를 붙이는 R-2 규칙을 가리킨다). "
           "R-5에 따라 cycle-type 축은 predictor-only이며 기제가 아니고 n=10에서 parity와 collinear이다(PREREG §0); 이 문서는 \"best / governing\"류 문장을 쓰지 않는다.", cells="Q-Z2,CL-0001 / CL-0002")

    # ------------------------------------------------------------------ Q-Z3
    D.h(2, "Q-Z3 — PR(z_L)·PR(z_H)의 collapse 서명 (질문 토큰: Q-INHERIT)")
    z3 = cell_rows("Q-Z3", "EXP-002", "z/eff_rank", True)
    assert len(z3) == len(cell_rows("Q-Z3", "EXP-002")) and set(z3[z3.value != ""].step) == {"244100"} and set(z3[z3.value == ""].step) == {""}
    g3 = z3.groupby(["arch", "HL_train"])
    z3rows = []
    for (arch, hl), g in sorted(g3, key=lambda x: x[0]):
        has = (g.value != "").sum()
        z3rows.append([arch, hl, len(g), has, len(g) - has, rid_str(g.row_id), LEG])
    trm_ids = z3[z3.arch == "trm"].row_id
    sz_ids = z3[z3.arch == "trm_singlez"].row_id
    assert (z3[z3.arch == "trm"].value != "").all() and (z3[z3.arch == "trm_singlez"].value == "").all()
    D.para(f"(Q-Z3, EXP-002) 셀은 코호트 legacy이다 — {LEG}. `z/eff_rank`(= PR(z_H)) R-FINAL 행이 {len(z3)}개이며 모두 `fig1_tf_*` run이다. 이 중 `arch=trm` 행 {len(trm_ids)}개에는 값이 기록돼 있고, "
           f"`arch=trm_singlez` 행 {len(sz_ids)}개에서는 키가 기록되지 않았다(z_H가 없는 변형; 값 없음). 값이 기록된 행의 step은 모두 {stepf(z3[z3.value != ""].iloc[0])}이다. "
           "S1_table `protocol`은 이 PR 키에 대해 UNKNOWN이며, `protocol_note`는 \"PR key — sec.3.2 protocol vocabulary covers test-exact protocols only (INTERP_S1 (7))\"이다. PR(z_L)은 S1_table에 행이 없다.",
           ids=z3.row_id)
    D.table(f"**표 Z-7.** (Q-Z3, EXP-002) `z/eff_rank` 행 구성 (값 칸 없음 — 아래 문단 사유). 코호트 legacy — {LEG}.",
            ["arch", "(H,L)_train", "행 수", "값 기록 행", "값 없음(키 미로깅) 행", "row_id", "코호트 / cohort_note"], z3rows, z3.row_id)
    D.para(f"이 표는 PR 값을 나열하지 않는다. 이유: S1_table `label_GC`가 전 행 NONE이라 collapse run 구분 불가이므로 `trm` 행 전체의 값을 보류했다. "
           f"이 보류는 보수적 선택이다(PREREG §0이 절대값 제한을 두는 것은 PR(z_L)이며 `z/eff_rank`가 아니다). 이 보류는 이 초안의 선택이며 orchestrator 결정 사항이다. "
           f"값은 S1_table의 위 row_id에서 읽는다. 이 문서는 PR의 높고 낮음에 대한 문장과 어떤 run의 collapse 판정을 쓰지 않는다 — {LEG}. 표에는 iter와 noiter 행이 함께 있다.", ids=z3.row_id)
    D.quote(R3)
    D.para("(Q-Z3, EXP-004)는 key_status UNKNOWN이라 값 행이 없다. **값 없음.**", cells="Q-Z3,EXP-004")
    hdr, rows = token_rows([("Q-Z3", "H-023")])
    D.raw("**표 Z-8.** Q-Z3 가설 토큰 (원문 재진술; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para(f"H-023 토큰은 카드 수준이며 PR 서명 하위 성분 토큰은 따로 지정돼 있지 않다(S1_cells 토큰 범위 칸 원문: \"{cell('Q-Z3', 'H-023').token_scope}\"). 이 토큰은 EXP-002 행의 값으로 옮겨 붙지 않는다.", cells="Q-Z3,H-023 / EXP-002")

    # ------------------------------------------------------------------ Q-Z4
    D.h(2, "Q-Z4 — τ(steps-to-grok)의 분포 (질문 토큰: Q-DESC-ONLY)")
    D.quote(f"절 전체 표지: {MARK}. 이 절에는 verdict 낱말이 없다(D-LOCKED 셀 없음).")
    z4 = cell_rows("Q-Z4", "EXP-006", "probe/test_exact", True)
    assert len(z4) == len(cell_rows("Q-Z4", "EXP-006")) and set(z4.step) == {"244100"}
    z4n = z4.groupby(z4.run_name.map(variant)).size().to_dict()
    D.para(f"(Q-Z4, EXP-006) 셀은 코호트 legacy이다 — {LEG}. S1이 읽은 키는 EXP-006의 metric pin인 `probe/test_exact`이며 값은 run별 R-FINAL이다. **이 값들은 τ(steps-to-grok)가 아니다.** "
           f"τ 행은 만들지 않았고(τ = UNKNOWN) 이 문서는 grok 시점이나 τ 분포를 서술하지 않는다(INTERP_S1 (2) 가드). 행 {len(z4)}개, 서로 다른 run {z4.run_name.nunique()}개이며 모두 `fig1_tf_*`이다. "
           f"변형별 run 수는 z_iter {z4n['z_iter']}, z_noiter {z4n['z_noiter']}, noz_iter {z4n['noz_iter']}, noz_noiter {z4n['noz_noiter']}이고 step은 모두 {stepf(z4.iloc[0])}이다. "
           f"값은 표 Z-1에서 해당 run의 `Z4·006` row_id 항목으로 읽는다(같은 run·같은 값, 스크립트 검증). 이 값으로 (Q-Z1/Q-Z2/Q-I2, EXP-001)의 값 없음을 채우지 않는다. Q-Z4는 Q-DESC-ONLY로 남는다.",
           ids=list(z4.row_id) + list(T[(T.q == "Q-Z1") & (T.metric_key == "probe/test_exact") & T.wandb_run_id.isin(set(z4.wandb_run_id))].row_id))
    D.quote(R3)
    assert not T.run_name.str.startswith("fig1n20").any()
    D.para("EXP-006 Phase B 스키마 `fig1n20_*`는 n=20이라 n=10 합의와 충돌한다(PREREG §3 플래그). S1_table 전체에 `fig1n20_*` 이름의 run은 없고, 이 문서는 그 스키마의 run을 쓰지 않는다.", cells="Q-Z4,EXP-006 (S1_table run_name 전체에서 fig1n20 접두 없음)")
    h26 = cell("Q-Z4", "H-026")
    D.para(f"H-026(HYPOTHESES.md 상태 칸, tier {h26.token_tier})의 상태 낱말은 {h26.token_N}이며 상태 낱말이지 판정이 아니다. 이 문서는 H-026에 판정 낱말을 붙이지 않는다.", cells="Q-Z4,H-026")

    # ------------------------------------------------------------------ Q-Z5
    D.h(2, "Q-Z5 — forward 안에서 z 상태는 어떻게 형성되는가 (질문 토큰: Q-INHERIT)")
    items = [("Q-Z5", "EXP-017"), ("Q-Z5", "FG-0003"), ("Q-Z5", "FG-0004"), ("Q-Z5", "FG-0005"), ("Q-Z5", "H-201")]
    D.para("(Q-Z5, EXP-017)은 D-LOCKED이고 key_status가 UNKNOWN(P-DECODE 키 문자열 미고정)이라 S1_table 값 행이 없다. **값 없음.** 이 문서는 EXP-017의 값을 서술하지 않는다. "
           "자기 토큰 원문은 아래 재진술로만 적는다.", cells="Q-Z5,EXP-017")
    re17, rh201 = cell("Q-Z5", "EXP-017"), cell("Q-Z5", "H-201")
    D.raw(f"EXP-017 자기 토큰 원문 재진술 (출처: {re17.token_file} · {re17.token_locator}):")
    D.quote(f"{re17.token_restatement} — {MARK}")
    D.raw(f"H-201 토큰 원문 재진술 (출처: {rh201.token_file} · {rh201.token_locator}):")
    D.quote(f"{rh201.token_restatement} — {MARK}")
    hdr, rows = token_rows(items[1:4])
    D.raw("**표 Z-9.** Q-Z5 그림 원자 토큰 (원문 재진술; 그림 서술 범위뿐; 출처 S1_cells).")
    D.raw("")
    D.raw("| " + " | ".join(hdr) + " |")
    D.raw("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.para("FG-0003..0005의 토큰은 그려진 곡선의 서술과 가독성에 한정된다. A-VERDICT 셀이 그림 서술 원자뿐이므로 이 절은 결합 헤드라인을 만들지 않는다(PREREG §7.2). "
           "EXP-017·H-201은 현황 기술(확증 아님)이며 이 절의 어떤 문장의 근거도 아니다. 이 문서는 형성 순서, 단계적 합성, k 간 시점 비교를 서술하지 않는다.", cells="Q-Z5,FG-0003 / FG-0004 / FG-0005")

    # ------------------------------------------------------------------ Q-Z6
    D.h(2, "Q-Z6 — z 개입 플래그(EXP-018 게이트, EXP-020 singlez ACT arm) (질문 토큰: Q-DESC-ONLY)")
    D.quote(f"절 전체 표지: {MARK}. 이 절에서 verdict 낱말은 D-LOCKED 셀(H-200) 자기 토큰의 원문 재진술(표지 동반)뿐이다.")
    D.para("(Q-Z6, EXP-018)은 X-ABSENT(key_status UNKNOWN)이고 (Q-Z6, H-202)도 연결 EXP 행이 모두 X-ABSENT라 X-ABSENT이다. 로컬 데이터가 없어 서술할 관측이 없다. **값 없음.** "
           "(Q-Z6, EXP-020)은 D-PREJUDGE이며 소유 트랙의 judge 전이라 상태만 쓰고 값은 읽지 않았다(run 수는 Q-S1 표).", cells="Q-Z6,EXP-018 / H-202 / EXP-020")
    rh200 = cell("Q-Z6", "H-200")
    D.raw(f"H-200 토큰 원문 재진술 (출처: {rh200.token_file} · {rh200.token_locator}):")
    D.quote(f"{rh200.token_restatement} — {MARK}")
    D.para("H-200 재진술은 (Q-I4, H-200) 셀의 재진술과 같은 원문이다. 이 문서는 EXP-020 singlez ACT arm과 z 플래그의 연결에 대한 문장을 쓰지 않는다.", cells="Q-Z6,H-200")

    # ------------------------------------------------------------------ Q-Z7
    D.h(2, "Q-Z7 — single-z macro-cycle의 near-idempotence / z-carry의 z_H write")
    D.raw("Q-LISTED, 본문 절 없음, Q-S1 표 참조.")
    D.raw("")

    # ------------------------------------------------------------------ appendix
    D.h(2, "부록 A. 값 없는 셀과 값을 쓰지 않은 셀 (z 축)")
    uk = C[(C.key_status == "UNKNOWN") & C.q.str.startswith("Q-Z")]
    D.para("S1_cells에서 key_status가 UNKNOWN인 셀 가운데 이 문서(z 축)에 속하는 것은 아래 표와 같다. 나머지 UNKNOWN 셀은 DESC_iteration.md에 있다. 이 셀들은 키 이름이 미고정이라 값 행이 없다. **값 없음.** "
           "(Q-Z6, EXP-018)은 UNKNOWN이면서 X-ABSENT(데이터 없음), (Q-Z5, EXP-017)은 UNKNOWN이면서 D-LOCKED이다.", cells="key_status == UNKNOWN")
    D.raw("**표 Z-A0.** key_status UNKNOWN 셀 (z 축; S1_cells).")
    D.raw("")
    D.raw("| (q, s) | S0r2 분류 | key_status | S1 값 행 | 비고 |")
    D.raw("|---|---|---|---|---|")
    for a, b, cl in zip(uk.q, uk.s, uk.s1_class):
        D.raw(f"| ({a}, {b}) | {cl} | UNKNOWN | {NOVAL} | {LEG if (a, b) in LEGSRC else '—'} |")
    D.raw("")
    D.raw("**표 Z-A.** key_status UNKNOWN이 아니지만 값 행이 없거나 값을 쓰지 않은 셀 (S1_cells; 수치 없음).")
    D.raw("")
    D.raw("| (q, s) | S0r2 분류 | key_status | 사유 |")
    D.raw("|---|---|---|---|")
    for r in [["(Q-Z1, EXP-025)", "A-DESC", "NAMED-ADMITTED", "후보 행은 있으나 owner UNKNOWN — 증거 사용 없음"],
              ["(Q-Z6, EXP-020)", "D-PREJUDGE", "NAMED-ADMITTED", "판정 전: 값 열람 없음(상태만)"],
              ["(Q-Z1, EXP-012/sealed)", "D-SEALED", "—", "sealed 종점 — 사용 안 함"],
              ["(Q-Z1, EXP-010)", "X-PROV", "—", "provenance 전용"]]:
        D.raw("| " + " | ".join(r) + " |")
    D.raw("")
    D.h(2, "부록 B. 이 초안이 확인하지 않은 전제")
    for s in [
        "run이 n=10 경로(`data/sigma_k_10/<k>` 등)에서 선택됐다는 전제는 S1_table에 data_path 칸이 없어 검증하지 않았다. 코호트는 S1 `cohort` 칸(project + 시작 시각 규칙)을 따른다.",
        f"legacy 행의 코호트는 S1_table `cohort` 칸을 그대로 따른 것이며 t_build 시간대(INTERP_S1 TN-6)는 미해결이다 — {LEG}. 코호트 UNKNOWN run의 사유는 S1_table `cohort` 칸이 따른 규칙(INTERP_S1 (6))에 의한 것이며 이 문서가 다시 판정하지 않았다.",
        "P-PROBE의 `weights`(EMA 여부)는 S1의 모든 코호트에서 `probe/test_exact` 전 행이 UNKNOWN이다(PREREG U-2).",
        f"legacy run의 eval_set_size는 S1 칸 값 UNKNOWN이다 — {LEG}.",
        "owner 귀속(`exp_owner`)은 S1이 EXPERIMENTS.md와 run-family 지도로 해소한 결과이며 이 문서가 다시 검증하지 않았다.",
        "z/eff_rank가 trm에만 값이 있다는 점은 S1 표 구성(값 기록 행 대 키 미로깅 행; 해당 행은 legacy, 신규 clean 코호트와 비교 불가)으로 확인했으며 원인(코드 경로)은 확인하지 않았다.",
        "EXP-017의 status 줄 원문은 S1_cells 재진술 칸에서 옮긴 것이며 RESULTS 파일을 이 초안이 열지 않았다.",
        "S1_table의 값을 wandb 로컬 datastore에서 다시 읽어 대조하지 않았다(V1 몫).",
    ]:
        D.raw("- " + s)
    D.raw("")
    ledger(D)
    return D


def ledger(D: Doc):
    D.h(2, "claim ledger")
    D.raw("`verifier verdict` 칸은 비워 둔다(independent-verifier 몫). regeneration 명령은 이 저장소 루트에서 실행한다.")
    D.raw("")
    D.raw("| # | claim | evidence (run/step/file) | regeneration command | verifier verdict |")
    D.raw("|---|---|---|---|---|")
    for pid, claim, ev, cmd in D.ledger:
        e = lambda x: x.replace("|", "\\|")
        D.raw(f"| {pid} | {e(claim)} | {e(ev)} | `{e(cmd)}` | |")
    D.raw("")
    D.raw(f"전체 재생성: `{GEN_CMD}` 후 `{GEN_CMD} --check`. 생성기 sha256: {sha(SELF)}.")


# =========================================================================================== check
ID_WORD = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_/\-\.]*\d[A-Za-z0-9_/\-\.]*(?![A-Za-z0-9_])")
STRIPS = [re.compile(p) for p in [r"`[^`]*`", r"〔[^〕]*〕", r"§[\d.]+", r"\(\d+\)", r"\d{4}-\d{2}-\d{2}(T[\d:]+Z?)?", r"\(\d,\d\)", r"(?<![A-Za-z0-9])n ?= ?(10|20)(?![0-9])", r"tier \d", r"line \d+", r"col \d+", r"legacy 별칭 EXP-015"]]
# Rows of non-legacy cells that mention the fixed string only inside explanatory text (the cell's own cohort column says clean-*):
LEG_EXPLAIN = ["legacy로의 이전은 금지("]
FORBID = ["collapses", "proves", "establishes", "confirms", "rules out", "excludes", "capacity ceiling", "truncation helps", "평균", "양봉", "bimodal",
          "first to observe", "the retained region", "ACT cannot rescue", "blind to collapse", "deductive", "iteration is the lever", "supported", "falsified", "refuted"]


def prereg_items() -> dict:
    """Re-parse PREREG §5.3 R-2/R-3/R-5 (continuation lines joined, whitespace collapsed) for mechanical comparison."""
    t = PREREG.read_text().splitlines()

    def item(prefix):
        for i, l in enumerate(t):
            if l.startswith(prefix):
                out, j = [l], i + 1
                while j < len(t) and t[j].startswith("  "):
                    out.append(t[j])
                    j += 1
                return re.sub(r"\s+", " ", " ".join(x.strip() for x in out))
        raise AssertionError(prefix)
    r2, r3, r5 = item("- **R-2 "), item("- **R-3 "), item("- **R-5 ")
    m3 = re.search(r'붙인다: "(.*?)" looped TF', r3)
    m2 = re.search(r'\*\*"(legacy, [^"]*)"\*\*', r2)
    m5 = re.search(r"\)\.\*\* (.*)$", r5)
    assert m3 and m2 and m5
    return dict(R3=m3.group(1), LEG=m2.group(1), R5=m5.group(1))


def sentences(par: str):
    return [x for x in re.split(r"(?<=[.。])\s+|\n", par) if x.strip()]


def check_doc(path: Path) -> dict:
    txt = path.read_text()
    body = txt.split("---", 2)[2]
    res = dict(file=path.name, numeric_sentences=0, unattached=[], bad_ids=[], mismatches=[], forbid=[], legacy_missing=[], tables=0, table_rows=0,
               prereg_mismatch=[])
    # R-2 / R-3 / R-5 constants and the blockquoted R-3 / R-5 blocks must equal the PREREG text (W-9)
    pi = prereg_items()
    for nm, const in [("R3", R3), ("R5", R5), ("LEG", LEG)]:
        if const != pi[nm]:
            res["prereg_mismatch"].append((f"const {nm}", const[-60:], pi[nm][-60:]))
    for ln in body.splitlines():
        if ln.startswith(">"):
            if "cycle-type 축은 predictor-only" in ln and ln != "> " + pi["R5"]:
                res["prereg_mismatch"].append(("quote R5", ln[-60:], pi["R5"][-60:]))
            if ln.startswith("> iter((H,L)_train") and ln != "> " + pi["R3"]:
                res["prereg_mismatch"].append(("quote R3", ln[-60:], pi["R3"][-60:]))
    # sections
    blocks = re.split(r"\n\s*\n", body)
    for b in blocks:
        bs = b.strip()
        if not bs or bs.startswith("#") or bs.startswith("**표") or bs.startswith("전체 재생성") or bs.startswith(">") or bs.startswith("|") or bs.startswith("-") or bs.startswith("```"):
            continue
        core = re.sub(r"〔[^〕]*〕", "", bs)
        has_tail = "〔row_id:" in bs
        # legacy rule
        if "legacy" in re.sub(r"legacy 별칭 EXP-015", "", core) and LEG not in core:
            res["legacy_missing"].append(bs[:60])
        s2 = core
        for p in STRIPS:
            s2 = p.sub(" ", s2)
        s2 = ID_WORD.sub(" ", s2)
        for sent in sentences(s2):
            if re.search(r"\d", sent):
                res["numeric_sentences"] += 1
                if not has_tail or not re.search(r"T\d{5}", bs):
                    res["unattached"].append(sent[:80])
        if not has_tail and not bs.startswith("**표") and not bs.startswith("EXP-") and not bs.startswith("H-200") and not bs.startswith("Q-LISTED") and not bs.startswith("`verifier") and not bs.startswith("전체 재생성"):
            # paragraphs lacking any tail are flagged only if they contain digits (handled above)
            pass
    # R-2: every body line (table rows, captions, bullets, ledger included) that mentions legacy carries the fixed string
    def ledger_cols(ln):
        return re.split(r"(?<!\\)\|", ln) if re.match(r"\| D[IZ]-P\d{3} \|", ln) else None
    for ln in body.splitlines():
        core_ln = re.sub(r"EXP-015_[\w\-]+\.md", " ", re.sub(r"legacy 별칭 EXP-015", " ", ln))
        lc = ledger_cols(ln)
        if lc and not ev_has_legacy(lc[3]):       # ledger row on a non-legacy referent: the word 'legacy' there is explanation text, no label owed
            continue
        if "legacy" in core_ln and pi["LEG"] not in core_ln:
            res["legacy_missing"].append(ln[:70])
    # R-2 converse (B-r4-1/2): the fixed string must not be attached to a non-legacy referent
    res["label_misplaced"] = []
    LAB = "S1_cells에 legacy 코호트로 표시돼 있다"
    for blk in re.split(r"\n\s*\n", body):
        for m in re.finditer(r"((?:\(Q-[A-Z]\d+, [^)]+\)·?)*) ?(?:셀의 )?소스는 " + re.escape(LAB), blk):
            cl = re.findall(r"\((Q-[A-Z]\d+), ([^)]+)\)", m.group(1))
            if not cl or any(c not in LEGSRC for c in cl):
                res["label_misplaced"].append(("explicit sentence names non-legacy/no cell", blk[:60]))
        if LAB in blk and not re.search(r"\(Q-[A-Z]\d+, [^)]+\)(?:·\([^)]+\))* 셀의 소스는 " + re.escape(LAB), blk):
            res["label_misplaced"].append(("generic label sentence", blk[:60]))
    for ln in body.splitlines():
        lc = ledger_cols(ln)
        if lc and f"[{pi['LEG']}]" in ln and not ev_has_legacy(lc[3]):
            res["label_misplaced"].append(("[LEG] on ledger row without legacy evidence", ln[:40]))
        if lc and ev_has_legacy(lc[3]) and pi["LEG"] not in lc[2]:
            res["label_misplaced"].append(("ledger row with legacy evidence lacks the string (claim column)", ln[:40]))
        if ln.startswith("| (Q-"):
            m = re.match(r"\| \((Q-[A-Z]\d+), ([^)]+)\)", ln)
            if m and (m.group(1), m.group(2)) not in LEGSRC and pi["LEG"] in ln and not any(x in ln for x in LEG_EXPLAIN):
                res["label_misplaced"].append(("table row of a non-legacy cell carries the string", ln[:50]))
    # an orphan label: a non-table block that has the string but no legacy word, no cohort_note cell and no legacy row_id
    for blk in re.split(r"\n\s*\n", body):
        if blk.lstrip().startswith(("|", "#", ">", "```", "---")) or blk.startswith(("`verifier", "전체 재생성", "- ")):
            continue
        rest = blk.replace(pi["LEG"], "")
        if pi["LEG"] in blk and "legacy" not in re.sub(r"legacy 별칭 EXP-015", "", rest) and not named_cells(rest) and not ev_has_legacy(blk):
            res["label_misplaced"].append(("orphan label", blk[:60]))
    # R-2 (source-based, B3r): every line naming a cell that S1_cells marks legacy-source (cohort_note) carries the fixed string
    units = []                                 # table rows are checked per line, every other block as a whole (paragraph + its row_id tail)
    for blk in re.split(r"\n\s*\n", body):
        if blk.lstrip().startswith("|"):
            units += blk.splitlines()[2:]          # drop header + separator rows (column legends, not data rows)
        else:
            units.append(blk)
    for q_, s_ in LEGSRC:
        for u in units:
            if " | UNKNOWN | — | " in u:      # Z-2 rows: S1_table cohort = UNKNOWN (no cohort_note; INTERP_S1 'UNKNOWN rows are not labelled'); the EXP token is the owner column
                continue
            if mentions_cell(u, q_, s_) and pi["LEG"] not in u:
                res["legacy_missing"].append(f"[S1_cells cohort_note {q_},{s_}] " + u[:60])
    # R-11: the first mention of EXP-025 in the body is the full form
    k0 = body.find("EXP-025")
    if k0 >= 0 and not body.startswith(R11, k0):
        res["forbid"].append(("R-11 first mention", body[k0:k0 + 60]))
    # forbidden words: every line (tables, bullets, ledger, blockquotes, code fences) except token tables and S1_cells restatements
    in_tok = False
    for ln in body.splitlines():
        if ln.startswith("|"):
            if re.match(r"\| \(q, s\) \| kind \| S0r2 분류 \| 토큰", ln):
                in_tok = True
            if in_tok:
                continue
        else:
            in_tok = False
        if ln.startswith(">"):
            if ln in ("> " + pi["R5"], "> " + pi["R3"], "> " + EXP011_1, "> " + EXP011_2) or ln.endswith(f" — {MARK}") or ln.startswith("> H-199 DISCREPANCY"):
                continue
        for w in FORBID:
            if w in ln:
                res["forbid"].append((w, ln[:80]))
    if re.search(r"(?<!legacy 별칭 )EXP-015(?!_)", body):
        res["forbid"].append(("bare EXP-015", ""))
    # tables: verify each row's row_id/value/step/run_name against S1_table
    byid = T.set_index("row_id")
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("| run_name |"):
            hdr = [x.strip() for x in lines[i].strip("|").split("|")]
            res["tables"] += 1
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                cells = [x.strip() for x in lines[j].strip("|").split("|")]
                row = dict(zip(hdr, cells))
                res["table_rows"] += 1
                rid_cell = [v for k, v in row.items() if k.startswith("row_id")][0]
                ids = re.findall(r"T\d{5}", rid_cell)
                for rid in ids:
                    if rid not in byid.index:
                        res["bad_ids"].append(rid)
                        continue
                    s = byid.loc[rid]
                    vcol = [k for k in hdr if k.startswith("value")]
                    if vcol:
                        shown = row[vcol[0]]
                        exp = s.value if s.value != "" else NOLOG
                        if shown != exp:
                            res["mismatches"].append((rid, "value", shown, exp))
                    if "step" in row and row["step"] != ("—" if s.step == "" else str(int(float(s.step)))):
                        res["mismatches"].append((rid, "step", row["step"], s.step))
                    if row.get("run_name") != s.run_name:
                        res["mismatches"].append((rid, "run_name", row.get("run_name"), s.run_name))
                    if "wandb_run_id" in row and row["wandb_run_id"] != s.wandb_run_id:
                        res["mismatches"].append((rid, "wandb_run_id", row["wandb_run_id"], s.wandb_run_id))
                    if "k" in row and row["k"] != s.k:
                        res["mismatches"].append((rid, "k", row["k"], s.k))
                    if "seed" in row and row["seed"] != s.seed:
                        res["mismatches"].append((rid, "seed", row["seed"], s.seed))
                    cn = row.get("코호트 / cohort_note")
                    if cn and s.cohort == "legacy" and cn != LEG:
                        res["mismatches"].append((rid, "cohort_note", cn, LEG))
                    if cn and s.cohort != "legacy" and LEG in cn:
                        res["mismatches"].append((rid, "cohort_note(non-legacy shows LEG)", cn, s.cohort))
                    if cn and s.cohort != "legacy" and cn != s.cohort:
                        res["mismatches"].append((rid, "cohort", cn, s.cohort))
                j += 1
            i = j
        else:
            i += 1
    # every cited T-id exists
    for rid in set(re.findall(r"T\d{5}", body)):
        if rid not in byid.index:
            res["bad_ids"].append(rid)
    # verbatim strings
    res["R3_count"] = body.count(R3)
    res["EXP011_pair"] = (body.count(EXP011_1), body.count(EXP011_2))
    res["rows_cited"] = len(set(re.findall(r"T\d{5}", body)))
    return res


def check():
    ok = True
    for name in ["DESC_iteration.md", "DESC_z.md"]:
        r = check_doc(OUT / name)
        print(name, {k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()})
        for k in ["unattached", "bad_ids", "mismatches", "forbid", "legacy_missing", "prereg_mismatch", "label_misplaced"]:
            if r[k]:
                ok = False
                print("  FAIL", k, r[k][:8])
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =========================================================================================== CLI
def expand_rows(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += [f"T{n:05d}" for n in range(int(a[1:]), int(b[1:]) + 1)]
        elif part:
            out.append(part)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rows")
    ap.add_argument("--cells")
    ap.add_argument("--cell")
    a = ap.parse_args()
    if a.rows:
        d = T[T.row_id.isin(expand_rows(a.rows))]
        print(d[["row_id", "q", "s", "run_name", "wandb_run_id", "metric_key", "protocol", "value", "step", "cohort", "evidence_use", "dup_wandb_dirs"]].to_string(index=False))
        return 0
    if a.cell:
        specs = [x.strip().split(" (")[0] for x in a.cell.split(";")]
        d = pd.concat([T[(T.q == sp.split("|")[0]) & (T.s == sp.split("|")[1]) & (T.metric_key == sp.split("|")[2])] for sp in specs])
        print(d[["row_id", "q", "s", "run_name", "wandb_run_id", "metric_key", "protocol", "value", "step", "cohort", "evidence_use", "dup_wandb_dirs"]].to_string(index=False))
        return 0
    if a.cells:
        q, s = a.cells.split(",", 1)
        for ss in [x.strip() for x in s.split("/")]:
            d = C[(C.q == q) & (C.s == ss)]
            print(d[["q", "s", "kind", "s0_confirmed_class", "s1_class", "key_status", "n_table_rows", "token_tier", "token_V", "token_N", "token_locator"]].to_string(index=False))
        return 0
    if a.check:
        return check()
    OUT.mkdir(exist_ok=True)
    for name, fn in [("DESC_iteration.md", build_iteration), ("DESC_z.md", build_z)]:
        (OUT / name).write_text(fn().text())
        print("wrote", OUT / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
