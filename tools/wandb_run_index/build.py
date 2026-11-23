#!/usr/bin/env python3
"""wandb run index -> Hypothesis/Experiment markdown blocks (idempotent).

Run:  uv run python tools/wandb_run_index/build.py [--apply] [--root .]
Default is a dry run (prints a plan, writes only the scratch report).

Source of truth for the run list: wandb-metadata.json of every run dir under
  ./wandb and /mnt/ayp/trm-offload/wandb   (args hold +project_name / +run_name)
run_id = the dir suffix.  The markdown blocks use names, ids, start time and config flags only.
Since 2026-11-10 (Stage A run cards) the jsonl index also carries per-run status (from the presence of
files/wandb-summary.json and the .wandb mtime), last_step (summary `_step`), the .wandb path/size/sha256
and header identity from the scan cache written by scan.py, config essentials, and a data-epoch
`cohort` (rule: common.COHORT_RULE_TEXT).  Runs without wandb-metadata.json are kept (has_metadata=false)
with identity from the .wandb RunRecord header.  WITHHELD_RUNS.yaml: all families `released`.
Registries (HYPOTHESES.md / EXPERIMENTS.md) are only written with --registries; any target pinned by a
*_FREEZE.sha256 manifest is skipped.

Attribution (run family -> EXP/H) is the DRAFT audit map
  lab/24_audits/11_lab-audits/2026-10-04_run-family-exp-map.yaml
plus PREFIX_RULES below for families newer than the map.  `confidence` is
carried verbatim from the map; generator-added rules are `prefix-rule` and are
never promoted to `registry`.

Frozen documents (hash-pinned in EXP-NNN_FREEZE.sha256) are NOT edited: they
get a sidecar `EXP-NNN_WANDB_RUNS.md` (RETROFIT pattern).  H-199/200/201 use
their mutable `*.status.md`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cohort, norm_data_paths, parse_started, sha256_text, sibling_key  # noqa: E402

START, END = "<!-- wandb-runs:auto start -->", "<!-- wandb-runs:auto end -->"
WANDB_DIRS = ["wandb", "/mnt/ayp/trm-offload/wandb"]
MAP = "lab/24_audits/11_lab-audits/2026-10-04_run-family-exp-map.yaml"
EXPD = "lab/21_experiments/11_lab-experiments"
THD = "lab/27_theory/11_lab-theory"

# families newer than the 2026-10-04 map: (project, regex) -> (exps, hyps, note)
PREFIX_RULES = [
    ("power_permutation", r"^e020_", ["EXP-020"], ["H-200"],
     "prefix-rule: e020_ = EXP-020 (ACT budget extension); id from FREEZE_WITNESS"),
    # EXP-023 (P-G1 LTF seed replication): no `e023` prefix exists on disk; queue jobs 0651-0695 are
    # `pp_ltf_{fullgrad,g7,noinj}_lay2_k*_s{4,5,6}` (EXP-023_FREEZE_WITNESS.md:49 "S_rep means pp_ltf_*_s4|s5|s6");
    # hypotheses from the prereg frontmatter `hypotheses: [H-199]`.  NOTE: the 2026-10-04 map regex
    # ^pp_ltf_*_lay2_k\d+_s\d+$ also attributes these to EXP-015 (over-match, map not edited).
    ("power_permutation", r"^pp_ltf_(fullgrad|g7|noinj)_lay2_k\d+_s[456]$", ["EXP-023"], ["H-199"],
     "prefix-rule: pp_ltf_*_lay2_s{4,5,6} = EXP-023 S_rep (FREEZE_WITNESS:49); unverified"),
    # EXP-018: wandb project EXP018_sigma_k, names e018_*; hypotheses from EXP-018 doc frontmatter [H-202]
    ("EXP018_sigma_k", r"^e018_", ["EXP-018"], ["H-202"],
     "prefix-rule: EXP018_sigma_k/e018_ = EXP-018 (G1 jobs 0707-0763); unverified"),
]

# EXP-NNN -> where the block goes.  sidecar = pre-reg is hash-pinned (or lives in 23_reports).
SIDECAR_EXPS = {"EXP-015", "EXP-016", "EXP-017", "EXP-018", "EXP-020", "EXP-022", "EXP-023"}
# H-ID -> file (mutable surface).  Hypotheses without a standalone doc -> one sidecar.
H_DOCS = {
    "H-022": f"{THD}/H-022.md", "H-023": f"{THD}/H-023.md", "H-202": f"{THD}/H-202.md",
    "H-199": f"{THD}/H-199.status.md", "H-200": f"{THD}/H-200.status.md",
    "H-201": f"{THD}/H-201.status.md",
}
H_SIDECAR = f"{THD}/H-025_H-026_H-A_WANDB_RUNS.md"


SCAN_CACHE = "lab/21_experiments/12_run-cards/_meta/scan_cache.jsonl"
RUNNING_IDLE_S = 15 * 60


def _unwrap_config(path: str) -> dict:
    """wandb config.yaml stores every key as {value: ...} (older: {desc, value}); unwrap it."""
    try:
        raw = yaml.safe_load(open(path)) or {}
    except Exception:
        return {}
    return {k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
            for k, v in raw.items() if k != "_wandb"}


def _flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, kk + "."))
        else:
            out[kk] = v
    return out


def _kdir_mtimes(root: Path, paths) -> dict:
    out = {}
    for p in paths:
        fs = [f for f in glob.glob(str(root / p) + "/**/*", recursive=True) if os.path.isfile(f)]
        if fs:
            out[p] = dt.datetime.fromtimestamp(max(os.path.getmtime(f) for f in fs), dt.timezone.utc)
    return out


def scan_runs(root: Path):
    """One row per unique run id under WANDB_DIRS (first occurrence wins; offload dirs are symlinked
    into ./wandb).  Runs without files/wandb-metadata.json are kept with has_metadata=False and
    identity recovered from the .wandb RunRecord header (scan cache) -- never silently skipped."""
    scan = {}
    sp = root / SCAN_CACHE
    if sp.exists():
        for ln in open(sp):
            r = json.loads(ln)
            scan[r["id"]] = r
    snap_p = sp.parent / "scan_snapshot.json"
    snap = parse_started(json.load(open(snap_p))["snapshot_utc"]) if snap_p.exists() else None
    seen, runs = {}, []
    for base in WANDB_DIRS:
        base = base if os.path.isabs(base) else str(root / base)
        for d in sorted(glob.glob(base + "/run-*") + glob.glob(base + "/offline-run-*")):
            rid = d.rsplit("-", 1)[-1]
            if rid in seen:
                continue
            seen[rid] = 1
            sc = scan.get(rid) or {}
            hdr = sc.get("header") or {}
            try:
                m = json.load(open(d + "/files/wandb-metadata.json"))
            except Exception:
                m = None
            a = (m or {}).get("args", [])
            kv = {x.split("=", 1)[0].lstrip("+"): x.split("=", 1)[1] for x in a if "=" in x}
            cfg = _unwrap_config(d + "/files/config.yaml")
            cfg_src = "config.yaml" if cfg else ("wandb-header" if hdr.get("config") else "none")
            if not cfg:
                cfg = hdr.get("config") or {}
            name = kv.get("run_name") or cfg.get("run_name") or hdr.get("display_name")
            name_src = ("metadata.args" if kv.get("run_name") else "config" if cfg.get("run_name")
                        else "wandb-header" if hdr.get("display_name") else "none")
            proj = kv.get("project_name")
            if m is not None and kv.get("run_name"):
                proj = proj or "?"
            else:
                proj = proj or cfg.get("project_name") or hdr.get("project") or "?"
            started_full = ((m or {}).get("started_at") or (m or {}).get("startedAt") or hdr.get("start_time") or "")
            summ = None
            try:
                summ = json.load(open(d + "/files/wandb-summary.json"))
            except Exception:
                pass
            mode = "offline" if os.path.basename(d).startswith("offline-run-") else "online"
            mt = sc.get("wandb_mtime_ns")
            if snap is not None and mt is not None and snap.timestamp() - mt / 1e9 < RUNNING_IDLE_S:
                status = "running?"
            elif summ:
                status = "ended"
            else:
                status = "no-summary"
            last_step, ls_src = None, None
            if summ and isinstance(summ.get("_step"), (int, float)):
                last_step, ls_src = summ["_step"], "wandb-summary.json:_step"
            elif sc.get("last_history_step") is not None:
                last_step, ls_src = sc["last_history_step"], "scan:max history _step"
            dp = cfg.get("data_paths") if cfg.get("data_paths") is not None else kv.get("data_paths")
            arch = cfg.get("arch") if isinstance(cfg.get("arch"), dict) else {}
            zflag = kv.get("log_z_dynamics", cfg.get("log_z_dynamics", ""))
            flat = _flat(cfg)
            runs.append(dict(
                project=proj, name=str(name or ""), id=rid,
                started=started_full[:10],
                z=str(zflag).lower() == "true",
                arch=kv.get("arch", ""), epochs=kv.get("epochs", cfg.get("epochs", "")),
                x=dict(
                    has_metadata=m is not None, name_source=name_src,
                    wandb_project=hdr.get("project") or (proj if proj not in ("?", "") else None),
                    mode=mode, synced_marker=bool(sc.get("synced_marker")),
                    started_at=started_full or None,
                    status=status, last_step=last_step, last_step_source=ls_src,
                    exit_code=(sc.get("exit") or {}).get("exit_code"),
                    run_dir=os.path.realpath(d), run_dir_link=os.path.relpath(d, root) if d.startswith(str(root) + "/") else d,
                    wandb_file=sc.get("wandb_file"), wandb_size=sc.get("wandb_size"),
                    wandb_sha256=sc.get("wandb_sha256"),
                    group=cfg.get("run_group") or hdr.get("run_group"),
                    k=cfg.get("k"), seed=cfg.get("seed"),
                    arch_preset=kv.get("arch") or None, arch_name=arch.get("name"),
                    H_cycles=arch.get("H_cycles"), L_cycles=arch.get("L_cycles"),
                    data_paths=norm_data_paths(dp), ema=cfg.get("ema"),
                    log_z_dynamics=cfg.get("log_z_dynamics", kv.get("log_z_dynamics")),
                    git_commit=((m or {}).get("git") or {}).get("commit") or (sc.get("env") or {}).get("git_commit"),
                    config_source=cfg_src,
                    sibling_key_sha=sha256_text(sibling_key({"project": proj, "config_flat": flat}))[:16],
                )))
    allp = {p for r in runs for p in r["x"]["data_paths"]}
    km = _kdir_mtimes(root, allp)
    for r in runs:
        c, why = cohort(r["x"]["data_paths"], r["x"]["started_at"], km)
        r["x"]["cohort"], r["x"]["cohort_reason"] = c, why
    return runs


def family(name: str) -> str:
    n = re.sub(r"^\d+_", "", name)
    n = re.sub(r"_s\d+$", "", n)
    n = re.sub(r"_k\d+", "", n)
    return n


def load_rules(root: Path):
    rows = yaml.safe_load(open(root / MAP))
    rows = rows.get("families", rows.get("rows", rows)) if isinstance(rows, dict) else rows
    rules = []
    for r in rows:
        if not r.get("name_regex"):
            continue
        rules.append(dict(project=r["project"], rx=re.compile(r["name_regex"]),
                          exps=r.get("exp_ids") or [], hyps=r.get("hypothesis_ids") or [],
                          conf=r.get("confidence", "none"), note=r.get("note", ""),
                          doc_labels=r.get("doc_labels") or [], src="map"))
    for proj, rx, ex, hy, note in PREFIX_RULES:
        rules.append(dict(project=proj, rx=re.compile(rx), exps=ex, hyps=hy,
                          conf="prefix-rule", note=note, doc_labels=[], src="prefix-rule"))
    return rules


def attribution_pairs(project: str, name: str, rules) -> list[dict]:
    """Per-attribution provenance for one run: [{id, source: map|prefix-rule, confidence: <verbatim label>}].
    Same matching as attribute(); EXP ids sorted, an id matched by several rules appears once per distinct
    (source, confidence).  Hypothesis ids are returned with kind "H"."""
    n = re.sub(r"^\d+_", "", name)
    seen, out = set(), []
    for ru in rules:
        if ru["project"] == project and (ru["rx"].search(name) or ru["rx"].search(n)):
            ids = [(e, "EXP") for e in ru["exps"]] + [(x, "EXP") for x in ru["doc_labels"] if re.fullmatch(r"EXP-\d{3}", x)]
            ids += [(h, "H") for h in ru["hyps"]] + [(x, "H") for x in ru["doc_labels"] if re.fullmatch(r"H-\d{3}", x)]
            for i, kind in ids:
                key = (kind, i, ru["src"], ru["conf"])
                if key not in seen:
                    seen.add(key)
                    out.append(dict(id=i, kind=kind, source=ru["src"], confidence=ru["conf"]))
    return sorted(out, key=lambda d: (d["kind"], d["id"], d["source"], str(d["confidence"])))


def attribute(runs, rules):
    for r in runs:
        r["exps"], r["hyps"], r["conf"], r["notes"] = set(), set(), set(), []
        n = re.sub(r"^\d+_", "", r["name"])
        for ru in rules:
            if ru["project"] == r["project"] and (ru["rx"].search(r["name"]) or ru["rx"].search(n)):
                # doc_labels are free text (e.g. "EXP-015(provisional, unregistered; != registry EXP-015)");
                # only bare canonical IDs are used -- annotated labels must not collide with registry IDs.
                r["exps"] |= set(ru["exps"]) | {x for x in ru["doc_labels"] if re.fullmatch(r"EXP-\d{3}", x)}
                r["hyps"] |= set(ru["hyps"]) | {x for x in ru["doc_labels"] if re.fullmatch(r"H-\d{3}", x)}
                r["conf"].add(ru["conf"])
    return runs


def render_block(title_id: str, runs: list, gen: str, kind: str) -> str:
    fam = defaultdict(list)
    for r in runs:
        fam[(r["project"], family(r["name"]))].append(r)
    lines = [START,
             f"## wandb run 목록 — {title_id} (자동 생성, 수정 금지)",
             "",
             f"> 생성 {gen} · 도구 `tools/wandb_run_index/build.py` · 출처: wandb-metadata.json args "
             f"(`{', '.join(WANDB_DIRS)}`) + 귀속 초안 `{MAP}` (DRAFT, 정본 레지스트리 아님). "
             "**결과 값은 읽지 않았다** — 이름·id·시작일·config 플래그만. "
             "confidence는 map 값 그대로(`registry`/`doc`/`none`, `prefix-rule`은 이 도구가 새 계열에 붙인 미검증 규칙). "
             "귀속이 분석 전용 소비(analysis-consumer)인 경우 이 실험이 run을 *학습*시킨 것은 아니다.",
             "",
             "| project | family | n | z-dyn 플래그(`+log_z_dynamics=true`; 지표 실존은 미검증) | 시작일 범위 | confidence |",
             "|---|---|---:|---:|---|---|"]
    for (p, f), rs in sorted(fam.items()):
        ds = sorted(x["started"] for x in rs if x["started"])
        rng = f"{ds[0]}…{ds[-1]}" if ds else "?"
        conf = "/".join(sorted(set().union(*[x["conf"] for x in rs]))) or "none"
        lines.append(f"| {p} | `{f}` | {len(rs)} | {sum(x['z'] for x in rs)} | {rng} | {conf} |")
    lines += ["", f"합계 **{len(runs)}** run · z-dyn(`+log_z_dynamics=true`) {sum(x['z'] for x in runs)}",
              "", f"<details><summary>run 전체 ({len(runs)}) — name · wandb id</summary>", ""]
    for (p, f), rs in sorted(fam.items()):
        ids = ", ".join(f"`{x['name']}`:{x['id']}" for x in sorted(rs, key=lambda x: x["name"]))
        lines.append(f"- **{p}/{f}** ({len(rs)}): {ids}")
    lines += ["", "</details>", END, ""]
    return "\n".join(lines)


def inject(path: Path, block: str, header: str | None):
    if path.exists():
        txt = path.read_text()
        if START in txt and END in txt:
            pre, rest = txt.split(START, 1)
            inner, post = rest.split(END, 1)
            # Hand-written text that another session placed INSIDE the markers, before the generated
            # heading (e.g. the 2026-11-03 "§S" pointers), is moved out above START instead of being
            # overwritten (2026-11-10 incident: 9 EXP docs lost §S on regeneration; restored).
            head = inner.split("## wandb run 목록", 1)[0] if "## wandb run 목록" in inner else ""
            if head.strip() and not head.lstrip().startswith("<!-- H-"):
                pre = pre.rstrip("\n") + "\n\n" + head.strip("\n") + "\n\n"
            new = pre + block.rstrip("\n") + post
        else:
            new = txt.rstrip("\n") + "\n\n" + block
    else:
        new = (header or "") + block
    return new


def frozen_paths(root: Path) -> set:
    """Every file pinned by any *_FREEZE.sha256 under lab/ (resolved).  build.py never writes these."""
    out = set()
    for mf in glob.glob(str(root / "lab") + "/**/*_FREEZE.sha256", recursive=True):
        d = os.path.dirname(mf)
        for ln in open(mf):
            parts = ln.strip().split(None, 1)
            if len(parts) != 2 or ln.startswith("#"):
                continue
            p = parts[1].lstrip("*")
            for c in ([p] if os.path.isabs(p) else [os.path.join(d, p), str(root / "lab" / p), str(root / p)]):
                if os.path.exists(c):
                    out.add(os.path.realpath(c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--registries", action="store_true",
                    help="also rewrite the blocks in HYPOTHESES.md/EXPERIMENTS.md (off by default since 2026-11-10)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    runs = attribute(scan_runs(root), load_rules(root))
    import collections as _c
    print("has_metadata:", dict(_c.Counter(r["x"]["has_metadata"] for r in runs)),
          "status:", dict(_c.Counter(r["x"]["status"] for r in runs)),
          "mode:", dict(_c.Counter(r["x"]["mode"] for r in runs)),
          "cohort:", dict(_c.Counter(r["x"]["cohort"] for r in runs)))
    ov = [r for r in runs if "EXP-023" in r["exps"] and "EXP-015" in r["exps"]]
    print(f"EXP-023 runs also attributed to EXP-015 by the map regex: {len(ov)}")
    gen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    print(f"runs scanned: {len(runs)}  unattributed: {sum(1 for r in runs if not r['exps'] and not r['hyps'])}")

    by_exp, by_h = defaultdict(list), defaultdict(list)
    for r in runs:
        for e in r["exps"]:
            by_exp[e].append(r)
        for h in r["hyps"]:
            by_h[h].append(r)

    # EXP docs (all IDs that have a doc or runs)
    docs = {}
    for p in glob.glob(str(root / EXPD / "EXP-[0-9][0-9][0-9]_*.md")):
        b = os.path.basename(p)
        if re.search(r"RESULTS|FREEZE|RETROFIT|WITNESS|VERIFY|PROPOSAL|artifacts", b):
            continue
        docs[b[:7]] = Path(p)
    targets = []
    for e in sorted(x for x in set(docs) | set(by_exp) | SIDECAR_EXPS if re.fullmatch(r"EXP-\d{3}", x)):
        rs = by_exp.get(e, [])
        if e in SIDECAR_EXPS or e not in docs:
            path = root / EXPD / f"{e}_WANDB_RUNS.md"
            hdr = (f"# {e} wandb run 목록 sidecar\n본문: {docs[e] if e in docs else '(21_experiments에 사전등록 본문 없음 — 23_reports/ 또는 22_plans/ 참조)'} "
                   "(hash-frozen이거나 별도 위치; 수정하지 않음)\n\n")
        else:
            path, hdr = docs[e], None
        targets.append((e, path, rs, hdr))
    for h, f in H_DOCS.items():
        targets.append((h, root / f, by_h.get(h, []), None))
    others = [h for h in by_h if h not in H_DOCS]
    for h in sorted(others):
        pass
    side_runs = [r for r in runs if r["hyps"] & set(others)]
    # H-025/H-026/H-A (no standalone doc) -> one sidecar, per-H sections
    sect = []
    for h in sorted(others):
        sect.append(render_block(h, by_h[h], gen, "H").replace(START, f"<!-- {h} -->").replace(END, f"<!-- /{h} -->"))
    h_side = (START + "\n" + "\n".join(sect) + "\n" + END + "\n")
    targets.append(("H-025/H-026/H-A", root / H_SIDECAR, side_runs, "# H-025 · H-026 · H-A wandb run 목록 sidecar\n(독립 문서가 없는 가설; 레지스트리 HYPOTHESES.md는 수정하지 않음)\n\n"))

    # registries: one compact block (family tables per ID; full id lists live in the per-doc blocks)
    for fname, ids, label in (("HYPOTHESES.md", sorted(by_h), "H"), ("EXPERIMENTS.md", sorted(by_exp), "EXP")):
        parts = []
        for i in ids:
            b = render_block(i, by_h[i] if label == "H" else by_exp[i], gen, label)
            b = b.split("<details>")[0].rstrip() + "\n"
            parts.append(b.replace(START, "").replace(END, "").replace("## wandb run", "### wandb run"))
        reg = START + "\n## wandb run 목록 색인 (자동 생성; run name·id 전체는 각 문서의 블록/sidecar와 `_wandb_runs_index.jsonl`)\n\n" + "\n".join(parts) + END + "\n"
        targets.append((f"registry:{fname}", root / "lab/13_reference/11_registries" / fname, [], "REG:" + reg))
    frozen = frozen_paths(root)
    blocked = []
    for tid, path, rs, hdr in targets:
        if tid.startswith("registry:") and not a.registries:
            print(f"skip  {tid:18s} (registries are not written without --registries)")
            continue
        if os.path.realpath(path) in frozen:
            blocked.append(str(path))
            print(f"SKIP  {tid:18s} -> {path.relative_to(root)}  [hash-pinned by a *_FREEZE.sha256]")
            continue
        block = h_side if tid.startswith("H-025") else (hdr[4:] if hdr and hdr.startswith("REG:") else render_block(tid, rs, gen, "x"))
        if hdr and hdr.startswith("REG:"):
            hdr = None
        print(f"{'APPLY' if a.apply else 'plan '} {tid:18s} runs={len(rs):4d} -> {path.relative_to(root)}"
              f"{'' if path.exists() else '  [new]'}")
        if a.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inject(path, block, hdr))
    # machine-readable index
    out = root / EXPD / "_wandb_runs_index.jsonl"
    if a.apply:
        with open(out, "w") as f:
            for r in sorted(runs, key=lambda x: (x["project"], x["name"])):
                f.write(json.dumps(dict(project=r["project"], name=r["name"], id=r["id"], started=r["started"],
                                        z_dyn=r["z"], exps=sorted(r["exps"]), hyps=sorted(r["hyps"]),
                                        confidence=sorted(r["conf"]), **r["x"]), ensure_ascii=False) + "\n")
        print("wrote", out.relative_to(root))
    if blocked:
        print(f"frozen targets skipped: {len(blocked)}")


if __name__ == "__main__":
    sys.exit(main())
