"""Operations behind cli.py. Everything here is offline unless a client object is passed in; the
client is always injected (RealClient only from cli.py's metered commands, FakeClient in tests)."""
from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import manifest as M
from . import normalize as N
from . import prices
from .common import (FROZEN_DOC_LINT, canon_batch_dir, PKG_DIR, REPO_ROOT, Halt, canon_json, est_tokens, in_claude_code_session,
                     sha256_bytes, sha256_file, sha256_text, trm_mcp_module_files)
from .ledger import LIMIT_BATCH_USD, LIMIT_DAY_USD, LIMIT_MONTH_USD, Ledger, approval_id, now
from .pack import PackBuilder, image_tokens_b64
from .spec import load_spec


COST_FORMULA = 2       # see _rows_hard_worst; recorded in plan.json so old plans re-derive with their own formula


class Refused(Halt):
    pass


def _pack_py_sha() -> str:
    return sha256_file(PKG_DIR / "pack.py")


# ------------------------------------------------------------------------------------- plan
def _replan_guard(bd: Path, ledger: Ledger) -> None:
    """plan overwrites manifest.jsonl / plan.json / pack/ of the batch dir, so it must not run while the dir's batch is in
    flight (a `submitted` row with no later `collected` / `canceled`), nor once the dir holds collected raw/ results: the
    as-sent record would be replaced by one the batch was never given (incident 2026-11-10, run-cards-pilot). Re-plan into
    a NEW dir instead. The dir is compared as a canonical physical path (cwd-relative argument vs ledger form)."""
    here = Path(bd).resolve()                                   # argument: relative to the caller's cwd
    last_sub = last_end = -1
    for i, r in enumerate(ledger.rows()):
        try:
            same = bool(r.get("batch_dir")) and canon_batch_dir(r["batch_dir"]) == here
        except (OSError, ValueError):
            same = False
        if not same:
            continue
        if r["event"] == "submitted":
            last_sub = i
        elif r["event"] in ("collected", "canceled"):
            last_end = i
    if last_sub > last_end:
        raise Halt(f"plan refused: {bd} has a submitted batch that is not yet collected/canceled; re-planning would replace the "
                   "as-sent manifest/plan/pack. Wait for collect, or plan into a new directory.")
    raw = Path(bd) / "raw"
    if raw.is_dir() and any(raw.iterdir()):
        raise Halt(f"plan refused: {bd} already holds collected raw/ results; plan into a new directory so the as-sent record "
                   "of the collected batch is kept.")


def plan(batch_dir: Path, builder: PackBuilder, ledger: Ledger) -> dict:
    bd = Path(batch_dir)
    _replan_guard(bd, ledger)
    spec_path = bd / "spec.yaml"
    spec = load_spec(spec_path)
    builder.batch_dir = bd
    scanned: list = []
    reqs: list[M.Request] = []
    schema_shas: set = set()
    frozen: frozenset = frozenset()
    lint_inputs = None
    u1_info = None
    figqa_info = None
    runcard_info = None
    docaudit_info = None
    scanned.append(builder.build_verify_only("spec_text", "spec.yaml", spec_path.read_text(encoding="utf-8"), origin=spec_path))

    def schema_of(name):
        if not name:
            return None
        sch = json.loads((REPO_ROOT / name).read_text(encoding="utf-8")) if "/" in str(name) else \
            json.loads((PKG_DIR / "schemas" / f"{name}.json").read_text(encoding="utf-8"))
        schema_shas.add(sha256_text(canon_json(sch)))
        return sch

    if spec["use"] == "probe":
        for pr in spec["probe_requests"]:
            docs = [builder.build_probe(pr["prefix"])] if pr.get("prefix") else []
            var = builder.build_probe(pr["var"])
            scanned += docs + [var]
            extra = {"fallbacks": pr["fallbacks"]} if pr.get("fallbacks") else None
            reqs.append(M.build_request(pr["id"], pr["model"], pr["effort"], pr["max_tokens"], None, docs, [var],
                                        pr.get("cache_ttl", "none"), schema_of(pr.get("schema")), extra=extra,
                                        meta={"probe": True}))
    elif spec["use"] == "seat-doclint":
        # §8.8: both inputs verify-only (any hit refuses the whole plan, nothing is masked), then the frozen
        # doc_lint.build_params applied to the scanned Packs' bytes (M.build_doclint_request).
        from .doclint_bridge import doc_lint_module
        dl = doc_lint_module()
        packs = {}
        for nm in ("results", "prereg"):
            src = Path(spec["inputs"][nm])
            src = src if src.is_absolute() else builder.lab_root / src
            packs[nm] = builder.build_verify_only("seat_input", src.name, src.read_text(encoding="utf-8"), origin=src,
                                                  lane="seat-doclint")
        scanned += [packs["results"], packs["prereg"]]
        frozen = frozenset({dl.PROMPT, dl.INSTRUCTION})
        schema_shas.add(sha256_text(canon_json(dl.SCHEMA)))
        lint_inputs = {"results_sha256": packs["results"].pack_sha256, "prereg_sha256": packs["prereg"].pack_sha256,
                       "results": packs["results"].name, "prereg": packs["prereg"].name}
        for j in range(1, int(spec["n"]) + 1):
            cid = M.make_custom_id("seat-doclint", packs["results"].text, "o", j)
            reqs.append(M.build_doclint_request(cid, spec["model"], packs["results"], packs["prereg"]))
    elif spec["use"] == "U1":
        # §6.2 map: one request per cell. All data enters as u1_data verify-only (any L2/L3b hit refuses the run,
        # nothing is masked); per-cell slices are rescanned and sealed as Packs of their own (single-path invariant).
        from . import u1
        tpl = REPO_ROOT / spec["prompt"]["system_template"]
        system = builder.build_verify_only("spec_text", "system", tpl.read_text(encoding="utf-8"), origin=tpl)
        hdr_path = bd / spec["header"]
        header = builder.build_verify_only("spec_text", "u1_header", hdr_path.read_text(encoding="utf-8"), origin=hdr_path)
        parents, texts = {}, {}
        for nm, rel in spec["data"].items():
            src = _resolve_data(builder, rel)
            texts[nm] = src.read_text(encoding="utf-8")
            parents[nm] = builder.build_verify_only("u1_data", src.name, texts[nm], origin=src)
        sch = schema_of(spec["output_schema"])
        scanned += [system, header, *parents.values()]
        cells = u1.build_cells(texts)
        for c in cells:
            k = c["key"]
            cp = builder.build_u1_cell(f"cell-{k['run_name']}", c["text"], list(parents.values()),
                                       {"cell": k["run_name"], "rows": c["rows"]})
            scanned.append(cp)
            cid = M.make_custom_id("U1", cp.text, u1.cell_cond(k), 1)
            reqs.append(M.build_request(cid, spec["model"], spec["effort"], spec["max_tokens"], system, [header], [cp],
                                        spec["cache_ttl"], sch,
                                        meta={"prompt_sha256": sha256_text(system.text), "arm": k["arm"], "k": k["k"],
                                              "seed": k["seed"], "cell": k["run_name"], "header_sha256": header.pack_sha256}))
        u1_info = {"n_cells": len(cells), "cells": [{"cell": c["key"]["run_name"], "rows": c["rows"]} for c in cells]}
    elif spec["use"] == "U1-reduce":
        # §6.2 reduce: second batch over the collected map items (derived Packs: l5 pass + listed in INBOX).
        from . import u1
        tpl = REPO_ROOT / spec["prompt"]["system_template"]
        system = builder.build_verify_only("spec_text", "system", tpl.read_text(encoding="utf-8"), origin=tpl)
        hdr_path = bd / spec["header"]
        header = builder.build_verify_only("spec_text", "u1_header", hdr_path.read_text(encoding="utf-8"), origin=hdr_path)
        map_dir = Path(spec["map_batch"])
        map_dir = map_dir if map_dir.is_absolute() else builder.lab_root / map_dir
        inbox = bd.parent.parent / "INBOX.jsonl"
        items = u1.read_map_batch(map_dir, inbox)
        dpacks = {it["custom_id"]: builder.build_derived(it["item"], inbox) for it in items}
        scanned += [system, header, *dpacks.values()]
        groups = u1.group_items(items)
        sets = [(f"{a}k{k}", f"group arm={a} k={k}", its, spec["max_tokens"]) for (a, k), its in sorted(groups.items())]
        if spec.get("digest", True):
            sets.append(("digest", "cross-cell digest over ALL cells (every arm, k and seed)", items, spec["digest_max_tokens"]))
        if spec.get("scopes"):                              # re-run a subset (e.g. scopes truncated at max_tokens)
            unknown = sorted(set(spec["scopes"]) - {c for c, *_ in sets})
            if unknown:
                raise Halt(f"scopes not in this map's groups: {unknown}")
            sets = [s for s in sets if s[0] in spec["scopes"]]
        for cond, label, its, mt in sets:
            ids = ",".join(sorted(i["custom_id"] for i in its))
            task = builder.build_spec_text(f"task-{cond}", f"TASK: {label}. The {len(its)} map items above are the only material.\n")
            scanned.append(task)
            cid = M.make_custom_id("U1-reduce", ids, cond, 1)
            # breakpoint on the header (shared by every scope); the item documents differ per scope
            reqs.append(M.build_request(cid, spec["model"], spec["effort"], mt, system, [header, *[dpacks[i["custom_id"]] for i in its]],
                                        [task], spec["cache_ttl"], None, cache_after=[header],
                                        meta={"prompt_sha256": sha256_text(system.text), "scope": cond, "n_items": len(its),
                                              "header_sha256": header.pack_sha256}))
        u1_info = {"n_items": len(items), "scopes": [c for c, *_ in sets], "map_batch": str(map_dir)}
    elif spec["use"] == "seat-figqa":
        figqa_info = _plan_figqa(spec, bd, builder, scanned, reqs, schema_of)
    elif spec["use"] in ("SIZ-map", "SIZ-reduce"):
        u1_info = _plan_siz(spec, bd, builder, scanned, reqs, schema_of)
    elif spec["use"] == "runcard":
        from . import runcard
        runcard_info = runcard.plan_runcard(spec, bd, builder, scanned, reqs, schema_of)
    elif spec["use"] == "docaudit":
        from . import docaudit
        docaudit_info = docaudit.plan_docaudit(spec, bd, builder, scanned, reqs, schema_of)
    else:
        tpl = REPO_ROOT / spec["prompt"]["system_template"]
        system = builder.build_verify_only("spec_text", "system", tpl.read_text(encoding="utf-8"), origin=tpl)
        builder.cap_tokens = int(spec["context_pack"].get("cap_tokens", 60000))
        gen = builder.build_generation("context", spec["context_pack"]["sources"])
        persp = spec["prompt"].get("perspectives") or ["base"]
        repeats = int(spec["prompt"].get("repeats", 1))
        sch = schema_of(spec.get("output_schema"))
        scanned += [system, gen]
        for i, v in enumerate(persp, 1):
            vp = builder.build_spec_text(f"perspective-{i:02d}", str(v))
            scanned.append(vp)
            for j in range(1, repeats + 1):
                cid = M.make_custom_id(spec["use"], gen.text, f"v{i:02d}", j)
                reqs.append(M.build_request(cid, spec["model"], spec["effort"], spec["max_tokens"], system, [gen], [vp],
                                            spec["cache_ttl"], sch,
                                            meta={"prompt_sha256": sha256_text(system.text), "perspective": i, "sample": j}))
    rows_w, rows_e, tok_in = [], 0.0, 0
    per_req = []
    for r in reqs:
        M.validate_request(r, frozen_texts=frozen, schema_shas=frozenset(schema_shas))
        p = r.params
        docs_t = sum(est_tokens(b["source"]["data"]) for m in p["messages"] for b in m["content"] if b["type"] == "document")
        other_t = sum(est_tokens(b["text"]) for m in p["messages"] for b in m["content"] if b["type"] == "text") + \
            sum(est_tokens(b["text"]) for b in p.get("system", []))
        img_t = sum(int(pk.report["est_tokens"]) for pk in r.packs if pk.kind == "figqa_image")
        if img_t != _image_tokens(p):
            raise Refused(f"{r.custom_id}: image token term {img_t} != {_image_tokens(p)} derived from the image blocks")
        if img_t:
            r.meta["image_tokens"] = img_t
        other_t += img_t
        t_ctx, t_var, ttl = M.cache_split(p)          # cost formula 2: everything up to the last breakpoint is written
        w = prices.hard_worst_request(p["model"], t_ctx, t_var, p["max_tokens"], ttl)
        e = prices.expected_request(p["model"], t_ctx, t_var, p["max_tokens"], ttl)
        per_req.append({"custom_id": r.custom_id, "hard_worst_usd": round(w, 6)})
        rows_w.append(w); rows_e += e; tok_in += docs_t + other_t
    worst = sum(rows_w)
    cap = min(float(spec["budget"]["cap_usd_hard_worst"]), LIMIT_BATCH_USD)
    if worst > cap:
        raise Refused(f"plan refused: hard-worst ${worst:.4f} > cap ${cap:.2f}")
    # write artefacts
    (bd / "pack").mkdir(exist_ok=True)
    seen = {}
    for p in scanned:
        seen[p.pack_sha256] = p
        (bd / "pack" / f"{p.kind}-{p.pack_sha256[:8]}.txt").write_text(p.text, encoding="utf-8")
    (bd / "pack" / "pack_manifest.json").write_text(json.dumps([p.manifest_entry() for p in seen.values()], indent=1,
                                                               ensure_ascii=False, sort_keys=True), encoding="utf-8")
    msha = M.write_manifest(reqs, bd / "manifest.jsonl")
    plan_d = {"name": spec["name"], "use": spec["use"], "n_requests": len(reqs), "est_input_tokens": tok_in,
              "cost_formula": COST_FORMULA,
              "est_usd_expected": round(rows_e, 6), "est_usd_hard_worst": round(worst, 6),
              "expected_assumption": "r_hit=0.5, output=0.5*max_tokens (planning figure, not billing)",
              "cap_usd": cap, "manifest_sha256": msha, "spec_sha256": sha256_file(spec_path),
              "pack_sha256": sha256_text(",".join(sorted(seen))), "schema_sha256": sorted(schema_shas),
              "deny_sha256": builder.deny.sha256, "pack_py_sha256": _pack_py_sha(),
              "masked_blocks": sum(p.report.get("masked_blocks", 0) for p in seen.values()),
              "scan": [{"name": p.name, "kind": p.kind, "mode": p.mode, "pack_sha256": p.pack_sha256, **p.report} for p in seen.values()],
              "requests": per_req}
    if lint_inputs:
        plan_d["lint_inputs"] = lint_inputs
    if figqa_info:
        plan_d["figqa"] = figqa_info
    if runcard_info:
        plan_d["runcard"] = runcard_info
    if docaudit_info:
        plan_d["docaudit"] = docaudit_info
    if u1_info:
        plan_d["siz" if spec["use"].startswith("SIZ-") else "u1"] = u1_info
    allowed = [{"path": p.sources[0]["path"], "allowed_by": p.report["allowed_by"], "allowed_hits": p.report["allowed_hits"]}
               for p in seen.values() if p.report.get("allowed_by")]
    if allowed:
        plan_d["allowed_files"] = allowed
    (bd / "plan.json").write_text(json.dumps(plan_d, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    ledger.append("planned", batch_dir=str(bd), manifest_sha256=msha, spec_sha256=plan_d["spec_sha256"],
                  pack_sha256=plan_d["pack_sha256"], schema_sha256=plan_d["schema_sha256"], n_requests=len(reqs),
                  est_usd_expected=plan_d["est_usd_expected"], est_usd_hard_worst=plan_d["est_usd_hard_worst"],
                  model=spec.get("model"), effort=spec.get("effort"), cache_ttl=spec.get("cache_ttl"),
                  deny_sha256=plan_d["deny_sha256"], pack_py_sha256=plan_d["pack_py_sha256"],
                  allowed_files=[{"path": a["path"], "sha256": a["allowed_by"]["sha256"], "n_hits": len(a["allowed_hits"])} for a in allowed] or None)
    return plan_d


def _image_tokens(params: dict) -> int:
    """Image-token term of a request, derived from its image blocks (PNG IHDR); 0 for requests without images."""
    return sum(image_tokens_b64(b["source"]["data"]) for m in params["messages"] if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "image")


def _plan_figqa(spec: dict, bd: Path, builder: PackBuilder, scanned: list, reqs: list, schema_of) -> dict:
    """seat-figqa (fig-qa blind readers B1/B2/B3, design §4.2/§5.5): per figure one image Pack (L1/L2 on the origin and
    exp_id; pixels are not scannable), the verify-only canonical context and caption (B2/B3 only), three requests.
    Packs of documents are titled neutrally ('problem_context', 'caption') so no slug/path reaches the reader."""
    tpl = REPO_ROOT / spec["prompt"]["system_template"]
    system = builder.build_verify_only("spec_text", "system", tpl.read_text(encoding="utf-8"), origin=tpl)
    cpath = REPO_ROOT / spec["context"]
    ctx = builder.build_verify_only("spec_text", "problem_context", cpath.read_text(encoding="utf-8"), origin=cpath)
    sch = schema_of(spec["output_schema"])
    scanned += [system, ctx]
    tasks = {"b1": "Condition B1: no caption is provided.", "b2": "Condition B2.", "b3": "Condition B2."}
    tpacks = {}
    for c, t in tasks.items():
        tpacks[c] = builder.build_spec_text(f"task-{c}", t)
        scanned.append(tpacks[c])
    figs = []
    for f in spec["figures"]:
        raw = (bd / f["png"]).read_bytes()
        img = builder.build_figqa_image(Path(f["png"]).stem, bd / f["png"], f["origin"], f["exp_id"])
        cap_path = builder.lab_root / f["caption"]
        cap = builder.build_verify_only("seat_input", "caption", cap_path.read_text(encoding="utf-8"), origin=cap_path)
        scanned += [img, cap]
        png_sha = sha256_bytes(raw)
        ids = []
        for c in ("b1", "b2", "b3"):
            cid = M.make_custom_id("seat-figqa", raw, c, 1)
            ids.append(cid)
            meta = {"slug": f["slug"], "exp_id": f["exp_id"], "cond": c, "model": spec["models"][c], "png_sha256": png_sha,
                    "prompt_sha256": sha256_text(system.text)}
            if c != "b1":
                meta["caption_sha256"] = cap.pack_sha256
            if spec["cache_ttl"] == "none":          # legacy layout: image first, no breakpoint (byte-identical)
                kw = {}
            else:
                # ctx-first-v2 (2026-11-05 user decision): [system, ctx*, image*, caption, task]. ctx is shared by
                # every figure of one reader model, ctx+image by the b1/b2 readers of one figure on the same model.
                kw = {"images_after": ctx, "cache_after": [ctx, img]}
                meta["layout"] = "ctx-first-v2"
            reqs.append(M.build_request(cid, spec["models"][c], spec["effort"], spec["max_tokens"], system,
                                        [ctx] if c == "b1" else [ctx, cap], [tpacks[c]], spec["cache_ttl"], sch, meta=meta,
                                        image_packs=[img], **kw))
        figs.append({"slug": f["slug"], "exp_id": f["exp_id"], "png_sha256": png_sha, "width": img.report["width"],
                     "height": img.report["height"], "image_tokens": img.report["est_tokens"], "custom_ids": ids})
    return {"n_figures": len(figs), "figures": figs}


def _resolve_data(builder: PackBuilder, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    for root in (builder.lab_root, builder.repo_root):
        if (Path(root) / p).is_file():
            return Path(root) / p
    raise Refused(f"data file {rel!r} not found under lab/ or the repo")


def _plan_siz(spec: dict, bd: Path, builder: PackBuilder, scanned: list, reqs: list, schema_of) -> dict:
    """SIZ-map: one request per S2 target cell (packets.build_map_packets over the scanned S1 files), one model per
    spec. SIZ-reduce: one request per question q (Q-LISTED / Q-NONE skipped, no digest) over both collected map
    batches. All data is u1_data verify-only (any L2/L3b hit refuses, nothing is masked); packets are rescanned and
    sealed as Packs of their own; the staged header/system/schema are sha-bound to S23_templates/HASHES.sha256."""
    from . import siz
    P = siz.packets()
    shas = siz.check_templates(bd, spec, builder.lab_root, builder.repo_root)
    sp, hp = bd / spec["prompt"]["system_template"], bd / spec["header"]
    system = builder.build_verify_only("spec_text", sp.stem, sp.read_text(encoding="utf-8"), origin=sp)
    header = builder.build_verify_only("spec_text", hp.stem, hp.read_text(encoding="utf-8"), origin=hp)
    texts, parents = {}, {}
    for nm, rel in spec["data"].items():                      # scanned BEFORE packets.py sees a byte
        src = _resolve_data(builder, rel)
        texts[nm] = src.read_text(encoding="utf-8")
        parents[nm] = builder.build_verify_only("u1_data", src.name, texts[nm], origin=src)
    sch = schema_of(spec["output_schema"])
    scanned += [system, header, *parents.values()]
    meta0 = {"prompt_sha256": sha256_text(system.text), "header_sha256": header.pack_sha256,
             "header_witness_sha256": shas["header"], "model": spec["model"]}
    if spec["use"] == "SIZ-map":
        pk = siz.call(P.build_map_packets, texts["s1_table"], texts["s1_cells"], spec["packets"].get("r3_ii_cite") or None)
        tag = siz.MODEL_TAG[spec["model"]]
        for p in pk:
            cp = builder.build_u1_cell(f"cell-{p['cell_id']}", p["text"], [parents["s1_table"], parents["s1_cells"]],
                                       {"cell_id": p["cell_id"], "q": p["q"], "s": p["s"]})
            scanned.append(cp)
            cid = M.make_custom_id("SIZ-map", cp.text, f"{p['cell_id']}{tag}", 1)
            reqs.append(M.build_request(cid, spec["model"], spec["effort"], spec["max_tokens"], system, [header], [cp],
                                        spec["cache_ttl"], sch,
                                        meta={**meta0, "q": p["q"], "s": p["s"], "cell_id": p["cell_id"],
                                              "cell_class": p["cell_class"], "mode": p["mode"]}))
        return {"n_cells": len(pk), "n_locked": sum(p["mode"] == "LOCKED" for p in pk), "template_sha256": shas,
                "cells": [p["cell_id"] for p in pk]}
    # ---- SIZ-reduce
    inbox = bd.parent.parent / "INBOX.jsonl"
    rows_by, memos = {}, []
    for model in P.MAP_MODELS:
        md = Path(spec["map_batches"][model])
        md = md if md.is_absolute() else builder.lab_root / md
        rows_by[model], its = siz.read_map_batch(md, inbox, model)
        memos += [{**it, "note": it["item"]} for it in its]
    siz.check_twins(rows_by)
    rps = siz.call(P.build_reduce_packets, texts["s1_cells"])
    skipped = [{"q": r["q"], "question_token": r["question_token"], "reason": r["reason"]} for r in rps if r["status"] != "built"]
    built = [r for r in rps if r["status"] == "built"]
    if spec.get("scopes"):                                     # re-run a subset (e.g. questions truncated at max_tokens)
        bad = sorted(set(spec["scopes"]) - {r["q"] for r in built})
        if bad:
            raise Refused(f"scopes are not built reduce questions (unknown, or {list(P.REDUCE_EXCLUDED)}): {bad}")
        built = [r for r in built if r["q"] in spec["scopes"]]
    for rp in built:
        q = rp["q"]
        its = siz.call(P.group_memos, rp, memos)
        rpack = builder.build_u1_cell(f"reduce-{q}", rp["text"], [parents["s1_cells"]], {"q": q})
        dpacks = [builder.build_derived(it["item"], inbox) for it in its]
        index = builder.build_spec_text(f"memo-index-{q}", "".join(
            [f"MAP MEMOS FOR {q} (document title -> cell_id, model):\n"] +
            [f"- {d.name} -> {it['cell_id']}, {it['model']}\n" for d, it in zip(dpacks, its)] +
            [f"TASK: reduce question {q} only. The header, the reduce packet and the {len(its)} memos above are the only material.\n"]))
        scanned += [rpack, *dpacks, index]
        cid = M.make_custom_id("SIZ-reduce", ",".join(sorted(it["custom_id"] for it in its)) or rpack.text, q, 1)
        # breakpoint on the header (shared by every question); the reduce packet and memos differ per question
        reqs.append(M.build_request(cid, spec["model"], spec["effort"], spec["max_tokens"], system, [header, rpack, *dpacks],
                                    [index], spec["cache_ttl"], sch, cache_after=[header],
                                    meta={**meta0, "q": q, "question_token": rp["question_token"], "n_memos": len(its)}))
    return {"questions": [r["q"] for r in built], "skipped": skipped, "n_memos": len(memos), "template_sha256": shas,
            "map_batches": {m: str(spec["map_batches"][m]) for m in P.MAP_MODELS}}


def _load_plan(bd: Path) -> dict:
    d = json.loads((Path(bd) / "plan.json").read_text(encoding="utf-8"))
    _rows, msha = M.load_manifest(Path(bd) / "manifest.jsonl")
    if msha != d["manifest_sha256"]:
        raise Refused("manifest.jsonl no longer matches plan.json (edited after plan)")
    w = _rows_hard_worst(_rows, int(d.get("cost_formula", 1)))
    if abs(w - float(d["est_usd_hard_worst"])) > 1e-6:      # W2: same tolerance as draft.load_plan (plan() rounds to 6 decimals, max legit gap 5e-7)        # plan.json's dollar figure is what approve/submit/limits trust
        raise Refused(f"plan.json est_usd_hard_worst {d['est_usd_hard_worst']} != re-derived {w:.6f} (edited after plan)")
    return d


def _rows_hard_worst(rows: list[dict], formula: int = 1) -> float:
    """Same formula as plan(): sum of prices.hard_worst_request over the manifest rows. formula 1 = plans written
    before 2026-11-05 (documents at the write price, system/text/images at the input price; kept so those plans
    still re-derive exactly); formula 2 = M.cache_split (every block up to the last breakpoint at the write price)."""
    tot = 0.0
    for r in rows:
        p = r["params"]
        if formula == 2:
            t_ctx, t_var, ttl = M.cache_split(p)
            tot += prices.hard_worst_request(p["model"], t_ctx, t_var, p["max_tokens"], ttl)
            continue
        docs_t = sum(est_tokens(b["source"]["data"]) for m in p["messages"] for b in m["content"] if b["type"] == "document")
        other_t = sum(est_tokens(b["text"]) for m in p["messages"] for b in m["content"] if b["type"] == "text") + \
            sum(est_tokens(b["text"]) for b in p.get("system", []))
        other_t += _image_tokens(p)
        ttl = (r.get("meta") or {}).get("cache_ttl", "none")
        tot += prices.hard_worst_request(p["model"], docs_t, other_t, p["max_tokens"], ttl if docs_t else "none")
    return tot


def _limits_ok(ledger: Ledger, worst: float, ts: str, include_self: bool) -> None:
    if worst > LIMIT_BATCH_USD:
        raise Refused(f"batch hard-worst ${worst:.2f} > ${LIMIT_BATCH_USD:.0f}")
    for pre, lim, nm in ((ts[:10], LIMIT_DAY_USD, "day"), (ts[:7], LIMIT_MONTH_USD, "month")):
        tot = ledger.committed_usd(pre) + (0 if include_self else worst)
        if tot > lim + 1e-9:
            raise Refused(f"{nm} committed hard-worst ${tot:.2f} > ${lim:.0f}")


# ------------------------------------------------------------------------------------- approve
def approve(batch_dir: Path | None, usd: float, ledger: Ledger, *, sync_plan: dict | None = None, env=None,
            sync_factor: float = 1.0) -> dict:
    """User-only. Refuses inside a Claude Code session BEFORE reading anything."""
    marker = in_claude_code_session(env)
    if marker:
        raise Refused(f"approve refused: running inside a Claude Code session ({marker} is set). "
                      f"Run it yourself from your own terminal.")
    if sync_plan is not None:
        key, worst, label = sync_plan["plan_hash"], float(sync_plan["hard_worst_usd"]), sync_plan.get("label", "sync")
    else:
        d = _load_plan(batch_dir)
        key, worst, label = d["manifest_sha256"], float(d["est_usd_hard_worst"]), str(batch_dir)
    worst *= sync_factor                 # probe-sync is billed at standard price = 2x batch
    if sync_plan is not None and worst <= 0:
        raise Refused("approve refused: a sync plan needs its hard-worst USD (--sync-worst), not 0")      # NB-3
    if usd < worst:
        raise Refused(f"approve refused: --usd {usd} < hard-worst {worst:.4f}")
    ts = now()
    _limits_ok(ledger, worst, ts, include_self=False)
    # lane: a batch approval is consumed by the manifest-hash check at submit; a sync approval authorises ONE
    # send (doclint-sync / probe-sync record its approval_id on every sync_sent row), a second send needs a new one.
    lane = "sync" if (sync_plan is not None or sync_factor != 1.0) else "batch"
    return ledger.append("approved", batch_dir=label, manifest_sha256=key, est_usd_hard_worst=round(worst, 6),
                         approval={"by": "user-cli", "usd": usd, "manifest_sha256": key, "ts": ts}, ts=ts,
                         lane=lane, approval_id=secrets.token_hex(6))


def _approved(ledger: Ledger, key: str, worst: float, factor: float = 1.0, *, fresh: bool = False, lane: str | None = None) -> dict:
    """Latest matching user approval. fresh=True (sync sends): skip approvals already consumed by a sync_sent
    row for this key, so one approval authorises at most one send (NB-1/NB-2)."""
    used = {r.get("approval_id") for r in ledger.rows("sync_sent", manifest_sha256=key)} if fresh else set()
    seen = False
    for r in reversed(ledger.rows("approved", manifest_sha256=key)):
        a = r.get("approval") or {}
        if lane is not None and (r.get("lane") or "batch") != lane:      # a probe-sync approval must not authorise submit (and vice versa)
            continue
        if a.get("by") == "user-cli" and a.get("manifest_sha256") == key and a.get("usd", 0) >= worst * factor:
            if approval_id(r) in used:
                seen = True
                continue
            return r
    raise Refused("every matching approval is already consumed by a send; a second send needs a new approve" if seen
                  else "no matching user approval for this manifest hash / amount")


# ------------------------------------------------------------------------------------- submit
def submit(batch_dir: Path, ledger: Ledger, client, *, recover: bool = False) -> dict:
    d = _load_plan(batch_dir)
    key = d["manifest_sha256"]
    _approved(ledger, key, d["est_usd_hard_worst"], lane="batch")
    _limits_ok(ledger, d["est_usd_hard_worst"], now(), include_self=True)
    prior = [r for r in ledger.rows() if r.get("manifest_sha256") == key and r["event"] in ("submitting", "submitted")]
    if recover:
        sub = [r for r in prior if r["event"] == "submitting"]
        if not sub or any(r["event"] == "submitted" for r in prior):
            raise Refused("nothing to recover for this manifest")
        t0 = datetime.fromisoformat(sub[-1]["ts"].replace("Z", "+00:00"))
        cands = [b for b in client.list()
                 if datetime.fromisoformat(str(b["created_at"]).replace("Z", "+00:00")) >= t0
                 and sum((b.get("request_counts") or {}).values()) == d["n_requests"]]
        if len(cands) != 1:
            raise Refused(f"recover: {len(cands)} candidate batches (need exactly 1); ask the user")
        return ledger.append("submitted", batch_dir=str(batch_dir), manifest_sha256=key, batch_id=cands[0]["id"],
                             n_requests=d["n_requests"], est_usd_hard_worst=d["est_usd_hard_worst"], recovered=True)
    if prior:
        raise Refused("this manifest hash was already submitted (or is mid-submit); use --recover if a crash left it")
    rows, _ = M.load_manifest(Path(batch_dir) / "manifest.jsonl")
    ledger.append("submitting", batch_dir=str(batch_dir), manifest_sha256=key, n_requests=d["n_requests"],
                  est_usd_hard_worst=d["est_usd_hard_worst"])
    try:
        b = client.create([{"custom_id": r["custom_id"], "params": r["params"]} for r in rows])
    except Exception as e:  # noqa: BLE001
        ledger.append("error", batch_dir=str(batch_dir), manifest_sha256=key, where="create", error=f"{type(e).__name__}: {str(e)[:200]}")
        raise
    return ledger.append("submitted", batch_dir=str(batch_dir), manifest_sha256=key, batch_id=b["id"],
                         create_request_id=b.get("_request_id"), n_requests=d["n_requests"],
                         est_usd_hard_worst=d["est_usd_hard_worst"], sdk_version=getattr(client, "sdk_version", None))


def _batch_id(ledger: Ledger, bd: Path) -> str:
    r = ledger.last("submitted", batch_dir=str(bd))
    if not r:
        raise Refused("no submitted row for this batch directory")
    return r["batch_id"]


def status(batch_dir: Path, ledger: Ledger, client) -> dict:
    bid = _batch_id(ledger, batch_dir)
    b = client.retrieve(bid)
    ledger.append("status", batch_dir=str(batch_dir), batch_id=bid, processing_status=b["processing_status"],
                  request_counts=b.get("request_counts"))
    if b["processing_status"] == "ended" and not ledger.rows("ended", batch_id=bid):
        ledger.append("ended", batch_dir=str(batch_dir), batch_id=bid, request_counts=b.get("request_counts"))
    return b


def cancel(batch_dir: Path, ledger: Ledger, client) -> dict:
    bid = _batch_id(ledger, batch_dir)
    b = client.cancel(bid)
    ledger.append("canceled", batch_dir=str(batch_dir), batch_id=bid)
    return b


# ------------------------------------------------------------------------------------- collect
def _lint_quote_summary(bd: Path, plan_d: dict, text: str) -> str | None:
    """seat-doclint items: the frozen doc_lint.quote_checks over the sealed packs' bytes (§8.8)."""
    li = plan_d.get("lint_inputs")
    if not li or not text:
        return None
    try:
        findings = json.loads(text)["findings"]
    except Exception:  # noqa: BLE001
        return None
    src = {}
    for nm in ("results", "prereg"):
        t = (bd / "pack" / f"seat_input-{li[nm + '_sha256'][:8]}.txt").read_text(encoding="utf-8")
        if sha256_text(t) != li[nm + "_sha256"]:
            raise Refused(f"collect: pack/{nm} bytes no longer match plan.json lint_inputs")
        src[nm] = t
    from .doclint_bridge import doc_lint_module
    qc = doc_lint_module().quote_checks(findings, src["results"], src["prereg"])
    return f"findings={len(qc)} exact={sum(c['exact'] for c in qc)} ws_normalized={sum(c['ws_normalized'] for c in qc)}"


def _figqa_extra(use: str, row: dict, st: str) -> dict:
    if use != "seat-figqa":
        return {}
    return {"figqa_condition": row["meta"].get("cond"), "figqa_reader": "REFUSED" if st == "refusal" else None}


def collect(batch_dir: Path, ledger: Ledger, client, batch_root: Path, builder: PackBuilder, *,
            offline: bool = False, delete: bool = True) -> dict:
    bd = Path(batch_dir)
    plan_d = _load_plan(bd)
    rows, _ = M.load_manifest(bd / "manifest.jsonl")
    by_id = {r["custom_id"]: r for r in rows}
    bid = _batch_id(ledger, bd)
    raw_path = bd / "raw" / "results.jsonl"
    prior = ledger.last("collected", batch_id=bid)
    use = plan_d["use"]
    if raw_path.exists() and prior and sha256_file(raw_path) == prior["results_sha256"]:
        recs = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        fetched = False
    else:
        if offline:
            raise Refused("collect --offline: no verified raw/results.jsonl to renormalize")
        b = client.retrieve(bid)
        if b["processing_status"] != "ended":
            raise Refused(f"batch {bid} is {b['processing_status']}, not ended")
        recs = list(client.results(bid))
        ids = [r["custom_id"] for r in recs]
        if len(ids) != len(set(ids)) or set(ids) != set(by_id):
            (bd / "raw").mkdir(exist_ok=True)
            (bd / "raw" / "results.invalid.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8")
            ledger.append("error", batch_dir=str(bd), batch_id=bid, where="collect",
                          error=f"custom_id mismatch: dup={len(ids) - len(set(ids))} missing={sorted(set(by_id) - set(ids))[:5]} "
                                f"unexpected={sorted(set(ids) - set(by_id))[:5]}")
            raise Refused("collect: result custom_ids differ from the manifest; inbox not updated")
        (bd / "raw").mkdir(exist_ok=True)
        raw_path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        fetched = True
    rsha = sha256_file(raw_path)
    # normalise from the raw file, keyed by custom_id (order-independent)
    items_dir, quar_dir = bd / "items", bd / "quarantine"
    items_dir.mkdir(exist_ok=True)
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    counts: dict = {}
    cost_total = released = 0.0
    inbox_new, refusals = [], {}
    worst_by = {x["custom_id"]: x["hard_worst_usd"] for x in plan_d["requests"]}
    ended = now()
    for rec in sorted(recs, key=lambda r: r["custom_id"]):
        cid, row = rec["custom_id"], by_id[rec["custom_id"]]
        expect_json = bool((row["params"].get("output_config") or {}).get("format"))
        st = N.classify(rec, expect_json)
        counts[st] = counts.get(st, 0) + 1
        model = row["params"]["model"]
        cost = N.cost_est(model, rec)
        cost_total += cost
        if st in N.UNBILLED:
            released += worst_by.get(cid, 0.0)
        msg = (rec["result"] or {}).get("message") or {}
        for k in tot:
            tot[k] += int((msg.get("usage") or {}).get(k) or 0)
        if st == "refusal":
            cat = (msg.get("stop_details") or {}).get("category", "unknown")
            refusals[cat] = refusals.get(cat, 0) + 1
        text = N.message_text(msg)
        hit = bool(text) and builder.deny.matcher.matches(text)          # L5
        l5 = "quarantined" if hit else "pass"
        body = N.render_item(use=use, batch_id=bid, custom_id=cid, row=row, rec=rec, status=st, cost=cost,
                             ended_at=ended, l5=l5, extra={"quote_checks": _lint_quote_summary(bd, plan_d, text),
                                                          **_figqa_extra(use, row, st)})
        if hit:
            quar_dir.mkdir(exist_ok=True)
            (quar_dir / f"{cid}.md").write_text(body, encoding="utf-8")
            continue
        ip = items_dir / f"{cid}.md"
        if ip.exists():
            # items are immutable after collect: keep the first rendering, only compare the body
            if ip.read_text(encoding="utf-8").split("\n---\n", 1)[-1] != body.split("\n---\n", 1)[-1]:
                ledger.append("error", batch_dir=str(bd), batch_id=bid, where="collect", error=f"item {cid} would change")
                raise Refused(f"item {cid} would change on re-collect")
        else:
            ip.write_text(body, encoding="utf-8")
        inbox_new.append({"custom_id": cid, "batch_id": bid, "use": use, "result_status": st, "item": f"items/{cid}.md"})
    inbox = Path(batch_root) / "INBOX.jsonl"
    old = [json.loads(ln) for ln in inbox.read_text(encoding="utf-8").splitlines() if ln.strip()] if inbox.exists() else []
    allrows = [r for r in old if r.get("batch_id") != bid] + inbox_new
    inbox.parent.mkdir(parents=True, exist_ok=True)
    inbox.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in allrows), encoding="utf-8")
    if not (prior and prior["results_sha256"] == rsha):
        hits = tot["cache_read_input_tokens"]
        den = hits + tot["cache_creation_input_tokens"]
        all_in = den + tot["input_tokens"]
        # r_hit_measured = read/(read+write) is kept for continuity; it ignores uncached input. cache_coverage =
        # read/(read+write+input) is the share of all input tokens served from cache (CACHE_AUDIT_2026-11-05.md).
        ledger.append("collected", batch_dir=str(bd), batch_id=bid, manifest_sha256=plan_d["manifest_sha256"],
                      results_sha256=rsha, usage_totals=tot, r_hit_measured=(hits / den if den else None),
                      cache_coverage=(round(hits / all_in, 6) if all_in else None),
                      cost_est_usd=round(cost_total, 6), released_usd=round(released, 6), refusals=refusals,
                      counts=counts, sdk_version=getattr(client, "sdk_version", None))
    if delete and not offline and not ledger.rows("deleted", batch_id=bid):
        try:
            client.delete(bid)
            ledger.append("deleted", batch_dir=str(bd), batch_id=bid, deleted_at=now())
        except Exception as e:  # noqa: BLE001
            ledger.append("error", batch_dir=str(bd), batch_id=bid, where="delete", error=f"{type(e).__name__}: {str(e)[:200]}")
    return {"counts": counts, "fetched": fetched, "quarantined": len(recs) - len(inbox_new), "results_sha256": rsha}


# ------------------------------------------------------------------------------------- scan
def scan_verify_only(paths: list[Path], builder: PackBuilder, ledger: Ledger, kind: str = "seat_input",
                     lane: str | None = None) -> int:
    files, rc, err = [], 0, None
    shas = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        files.append({"path": str(p), "text_sha256": sha256_text(text)})
        shas.append(sha256_text(text))
        try:
            pk = builder.build_verify_only(kind, Path(p).name, text, origin=Path(p), lane=lane)
            if pk.report.get("allowed_by"):          # an allow_files exception was used: never silent
                files[-1]["allowed_by"] = pk.report["allowed_by"]
                files[-1]["allowed_hits"] = pk.report["allowed_hits"]
        except Halt as e:
            rc, err = 1, f"{type(e).__name__}: {str(e)[:200]}"
            break
    ledger.append("scanned", paths=[str(p) for p in paths], files=files, pack_sha256=sha256_text(",".join(shas)),
                  deny_sha256=builder.deny.sha256, pack_py_sha256=_pack_py_sha(), exit_code=rc, error=err, lane=lane)
    return rc


# ------------------------------------------------------------------------------------- selftest
def _test_files() -> list[Path]:
    return sorted((PKG_DIR / "tests").glob("test_*.py")) + sorted((PKG_DIR / "tests").glob("conftest.py"))


def integrity_snapshot() -> dict:
    return {"pack_py_sha256": _pack_py_sha(), "deny_py_sha256": sha256_file(PKG_DIR / "deny.py"),
            "test_files": {p.name: sha256_file(p) for p in _test_files()},
            "trm_mcp": {p.name: sha256_file(p) for p in trm_mcp_module_files()},
            "frozen_doc_lint": {p.name: sha256_file(p) for p in FROZEN_DOC_LINT}}


def selftest(ledger: Ledger, runner=None) -> int:
    def default_runner() -> int:
        import os
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        return subprocess.run([sys.executable, "-m", "pytest", str(PKG_DIR / "tests"), "-q"], cwd=str(REPO_ROOT), env=env).returncode
    rc = (runner or default_runner)()
    if rc == 0:
        ledger.append("tests_passed", **integrity_snapshot(), note="deny source not bound (user-managed, may be absent)")
    return rc


def tests_passed_current(ledger: Ledger) -> tuple[bool, str]:
    r = ledger.last("tests_passed")
    if not r:
        return False, "no tests_passed row"
    now_s = integrity_snapshot()
    for k in ("pack_py_sha256", "deny_py_sha256", "trm_mcp", "frozen_doc_lint", "test_files"):
        if r.get(k) != now_s[k]:
            return False, f"{k} differs from the tests_passed row"
    return True, "ok"


# ------------------------------------------------------------------------------------- sync paths
def _default_doclint_runner(plan_path: Path, out: Path, budget: float) -> int:
    return subprocess.run([sys.executable, str(REPO_ROOT / "tools/api_seats/doc_lint.py"), "--plan", str(plan_path),
                           "--out", str(out), "--models", "claude-opus-5-5", "--n", "3", "--budget-usd", str(budget)],
                          cwd=str(REPO_ROOT)).returncode


def doclint_sync(results: Path, prereg: Path, builder: PackBuilder, ledger: Ledger, *, p0a_verifier_ref: str,
                 plan_only: bool = False, runner=None, workdir: Path | None = None) -> dict:
    """Sole executor of the synchronous doc-lint shadow. Lapses (raises Refused, sends nothing) unless
    (1) tests_passed is current, (2) the P0a verifier PASS reference is given (NB-R3-5), (3) an independent
    verify-only scan exits 0 and its `scanned` row is written BEFORE the send. Sends copies of the scanned bytes
    (NB-R3-4), approval keyed by the plan hash."""
    ok, why = tests_passed_current(ledger)
    if not ok:
        raise Refused(f"doclint-sync lapses: {why}")
    if not (p0a_verifier_ref or "").strip():
        raise Refused("doclint-sync lapses: the first sync send needs the P0a verifier PASS reference")
    if scan_verify_only([results, prereg], builder, ledger, lane="seat-doclint") != 0:
        raise Refused("doclint-sync lapses: verify-only scan found a denied target (doc_lint.py not called)")
    srow = ledger.last("scanned")
    wd = Path(workdir or tempfile.mkdtemp(prefix="doclint_sync_"))
    cp = {}
    for nm, src in (("results", results), ("prereg", prereg)):
        dst = wd / f"{nm}{Path(src).suffix or '.md'}"
        dst.write_bytes(Path(src).read_text(encoding="utf-8").encode("utf-8"))      # the scanned text bytes
        cp[nm] = dst
    rs, ps = (f["text_sha256"] for f in srow["files"])
    plan_json = [{"label": f"sync-{rs[:8]}", "results": str(cp["results"]), "prereg": str(cp["prereg"])}]
    plan_path = wd / "plan.json"
    plan_path.write_text(json.dumps(plan_json, sort_keys=True), encoding="utf-8")
    plan_hash = sha256_text(canon_json({"results": rs, "prereg": ps, "models": ["claude-opus-5-5"], "n": 3}))
    tr, tp = est_tokens(Path(results).read_text(encoding="utf-8")), est_tokens(Path(prereg).read_text(encoding="utf-8"))
    worst = prices.standard_hard_worst("claude-opus-5-5", tr + tp, 100, 32000, "5m", n=3)
    out = {"plan_hash": plan_hash, "hard_worst_usd": round(worst, 4), "plan_path": str(plan_path)}
    if plan_only:
        return out
    appr = _approved(ledger, plan_hash, worst, fresh=True, lane="sync")      # W1: a batch-lane approval must not authorise a sync send
    # sync_sent is written BEFORE the send and consumes the approval even if the runner then fails (spend unknown)
    ledger.append("sync_sent", plan_hash=plan_hash, manifest_sha256=plan_hash, scanned_ts=srow["ts"], p0a_verifier_ref=p0a_verifier_ref,
                  est_usd_hard_worst=round(worst, 4), approval_id=approval_id(appr))
    rc = (runner or _default_doclint_runner)(plan_path, wd / "run", float(appr["approval"]["usd"]))
    mismatch = None
    dl_ledger = wd / "run" / "ledger.jsonl"
    if dl_ledger.exists():
        for ln in dl_ledger.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            if r.get("event") == "sent" and (r.get("results_sha256") != rs or r.get("prereg_sha256") != ps):
                mismatch = "doc_lint.py ledger input sha differs from the scanned row"
    ledger.append("sync_done", plan_hash=plan_hash, rc=rc, exposure_incident=mismatch)
    if mismatch:
        ledger.append("error", where="doclint-sync", error="EXPOSURE INCIDENT: " + mismatch)
    return {**out, "rc": rc, "exposure_incident": mismatch}


def probe_sync(batch_dir: Path, ledger: Ledger, client, ids: list[str] | None = None) -> list[dict]:
    d = _load_plan(batch_dir)
    key = d["manifest_sha256"]
    appr = _approved(ledger, key, d["est_usd_hard_worst"], factor=2.0, fresh=True, lane="sync")      # standard price = 2x batch
    rows, _ = M.load_manifest(Path(batch_dir) / "manifest.jsonl")
    out = []
    for r in rows:
        if ids and r["custom_id"] not in ids:
            continue
        ledger.append("sync_sent", batch_dir=str(batch_dir), manifest_sha256=key, custom_id=r["custom_id"],
                      params_sha256=r["params_sha256"], approval_id=approval_id(appr))
        msg = client.send_sync({"custom_id": r["custom_id"], "params": r["params"]})
        cost = 2.0 * prices.cost_from_usage(r["params"]["model"], msg.get("usage") or {})
        ledger.append("sync_done", batch_dir=str(batch_dir), manifest_sha256=key, custom_id=r["custom_id"],
                      message_id=msg.get("id"), model_returned=msg.get("model"), cost_est_usd=round(cost, 6))
        out.append(msg)
    return out
