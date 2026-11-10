"""draft-sync: synchronous (standard-price) drafting of report / paper sections through the same
safety rails as the batch lane (2026-10-30 user: "batch가 아니더라도 작성에 API 키를 활용").

  plan <spec>      offline. spec = {name, model (prices.py only), effort, max_tokens, system: <file>,
                   sections: [{id, prompt: <file>, data: [<path>...]}], budget_usd?,
                   cache_ttl? (5m default | 1h | none: breakpoints on system + shared data runs)}. system/prompt files are
                   `spec_text` (only <plan_dir>/shared/*.md or tools/api_batch/prompts/*.md, L1); data files are
                   verify-only Packs (lab .md/.txt -> seat_input, lab .csv/.json + reports curves -> u1_data):
                   any deny hit refuses the whole plan, nothing is masked. Writes manifest.jsonl, plan.json
                   (plan_hash, hard-worst at STANDARD price = SYNC_PRICE_FACTOR x batch), ledger `planned`.
  approve          user-only, unchanged (`approve <plan_dir> --usd X` reads plan.json after the session guard).
                   One approval covers the whole plan (N sections) and is consumed by one run.
  run <plan_dir>   metered (with-anthropic-key). Re-derives manifest sha, hard-worst and plan_hash from the
                   files (plan.json is not trusted), needs current tests_passed + a fresh matching approval,
                   rescans every text segment (content) and every recorded source path (deny path globs /
                   name patterns, as at plan) against the current deny source, then per section: `sync_sent`
                   BEFORE the send, raw/<id>.json + out/<id>.md (DRAFT header) or quarantine/<id>.md (L5 hit),
                   `sync_done` with rc (0 ok, 1 send raised, 2 model fallback, 3 L5 quarantine, 4 refusal /
                   max_tokens / other stop). Sends stream (client.send_stream) so long max_tokens is safe.
                   A re-run (new approval) skips sections already rc=0; with none pending it returns
                   `nothing_to_do` before any gate/approval lookup, and `approve` refuses such a plan. Stops before a send whose hard-worst would push the running cost past
                   the approved amount. No `fallbacks` param is ever built; a returned model that differs from
                   the requested one is recorded as rc=2 and stops the run.
This module never reads or prints an API key; the client is injected (RealClient only from cli.py).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

from . import manifest as M
from . import normalize as N
from . import prices
from .common import Halt, canon_json, est_tokens, sha256_file, sha256_text
from .ledger import LIMIT_BATCH_USD, Ledger, approval_id, now
from .ops import Refused, _approved, _limits_ok, _pack_py_sha, tests_passed_current
from .pack import PackBuilder
from .spec import EFFORTS, SpecError

USE = "draft-sync"
# Synchronous Messages API is billed at the standard price = 2x the batch table in prices.py (F3: batch is 50%).
SYNC_PRICE_FACTOR = 2.0
SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
DATA_KIND = {".md": "seat_input", ".txt": "seat_input", ".csv": "u1_data", ".json": "u1_data"}
DRAFT_KEYS = {"name", "model", "effort", "max_tokens", "system", "sections", "budget_usd", "cache_ttl"}
DEFAULT_CACHE_TTL = "5m"     # sections are sent one after another, so a 5m entry written by one is read by the next
COST_FORMULA = 2             # 1 = plans written before 2026-11-05 (no breakpoints, all input at p_in)


# ------------------------------------------------------------------------------------- spec
def load_draft_spec(path: Path) -> dict:
    try:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SpecError(f"draft spec unreadable: {type(e).__name__}")
    if not isinstance(spec, dict):
        raise SpecError("draft spec is not a mapping")
    for k in ("name", "model", "effort", "max_tokens", "system", "sections"):
        if k not in spec:
            raise SpecError(f"draft spec: missing key {k!r}")
    extra = set(spec) - DRAFT_KEYS
    if extra:
        raise SpecError(f"draft spec: unknown keys {sorted(extra)}")
    if spec["model"] not in prices.PRICES:
        raise SpecError(f"draft spec: model {spec['model']!r} not in prices.py")
    if spec["effort"] not in EFFORTS:
        raise SpecError("draft spec: effort must be explicit (low|medium|high|xhigh|max)")
    if not isinstance(spec["max_tokens"], int) or isinstance(spec["max_tokens"], bool) or spec["max_tokens"] <= 0:
        raise SpecError("draft spec: max_tokens must be a positive integer")
    if not isinstance(spec["system"], str) or not spec["system"]:
        raise SpecError("draft spec: system must be a prompt file path")
    if spec.get("cache_ttl", DEFAULT_CACHE_TTL) not in ("5m", "1h", "none"):
        raise SpecError("draft spec: cache_ttl (optional, default 5m) must be 5m | 1h | none")
    b = spec.get("budget_usd")
    if b is not None and (not isinstance(b, (int, float)) or b <= 0):
        raise SpecError("draft spec: budget_usd (optional) must be a positive number")
    secs = spec["sections"]
    if not isinstance(secs, list) or not secs:
        raise SpecError("draft spec: sections must be a non-empty list")
    ids = []
    for s in secs:
        if not isinstance(s, dict) or set(s) - {"id", "prompt", "data"} or "id" not in s or "prompt" not in s:
            raise SpecError("draft spec: each section is {id, prompt, data?}")
        if not isinstance(s["id"], str) or not SECTION_ID_RE.match(s["id"]):
            raise SpecError(f"draft spec: section id {s.get('id')!r} must match {SECTION_ID_RE.pattern}")
        d = s.get("data", [])
        if not isinstance(d, list) or not all(isinstance(x, str) and x for x in d):
            raise SpecError(f"draft spec: section {s['id']} data must be a list of paths")
        ids.append(s["id"])
    if len(set(ids)) != len(ids):
        raise SpecError("draft spec: duplicate section id")
    return spec


def _resolve(builder: PackBuilder, base: Path, rel: str, *, data: bool) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    roots = (builder.lab_root, builder.repo_root) if data else (base, builder.repo_root)
    for root in roots:
        if (Path(root) / p).is_file():
            return Path(root) / p
    raise Refused(f"draft-sync: file {rel!r} not found")


# ------------------------------------------------------------------------------------- cost
def _req_tokens(params: dict) -> tuple[int, int]:
    docs_t = sum(est_tokens(b["source"]["data"]) for m in params["messages"] for b in m["content"] if b["type"] == "document")
    other_t = sum(est_tokens(b["text"]) for m in params["messages"] for b in m["content"] if b["type"] == "text") + \
        sum(est_tokens(b["text"]) for b in params.get("system", []))
    return docs_t, other_t


def request_hard_worst(params: dict, formula: int = COST_FORMULA) -> float:
    """Standard-price hard-worst of one sync request, all of max_tokens. formula 2: every block up to the last
    breakpoint at the cache-write price (M.cache_split), the rest at p_in; formula 1 (old plans): no cache, every
    input token at p_in. Both agree on a request without breakpoints."""
    if formula == 2:
        t_ctx, t_var, ttl = M.cache_split(params)
        return SYNC_PRICE_FACTOR * prices.hard_worst_request(params["model"], t_ctx, t_var, params["max_tokens"], ttl)
    docs_t, other_t = _req_tokens(params)
    return SYNC_PRICE_FACTOR * prices.hard_worst_request(params["model"], docs_t, other_t, params["max_tokens"], "none")


def _shared_depths(doc_lists: list[list]) -> tuple[int, list[int]]:
    """(g, d): g = number of leading data documents (by pack sha) shared by EVERY section; d[i] = deepest leading
    run section i shares with at least one other section. Both index the documents that follow the system block."""
    def lcp(a, b):
        n = 0
        while n < len(a) and n < len(b) and a[n].pack_sha256 == b[n].pack_sha256:
            n += 1
        return n
    n = len(doc_lists)
    d = [max((lcp(doc_lists[i], doc_lists[j]) for j in range(n) if j != i), default=0) for i in range(n)]
    g = min((lcp(doc_lists[0], x) for x in doc_lists[1:]), default=0) if n > 1 else 0
    return g, d


def compute_plan_hash(manifest_sha256: str, hard_worst_usd: float, formula: int = 1) -> str:
    """formula 1 hashes exactly as plans written before 2026-11-05 did; later formulas add the key."""
    h = {"use": USE, "manifest_sha256": manifest_sha256,
         "hard_worst_usd": round(hard_worst_usd, 6), "sync_price_factor": SYNC_PRICE_FACTOR}
    if formula != 1:
        h["cost_formula"] = formula
    return sha256_text(canon_json(h))


def _no_fallback(params: dict, cid: str) -> None:
    if "fallbacks" in params:
        raise Refused(f"draft-sync: {cid} carries a server-side model fallback (forbidden)")


# ------------------------------------------------------------------------------------- plan
def plan(spec_path: Path, builder: PackBuilder, ledger: Ledger) -> dict:
    spec_path = Path(spec_path)
    pd = spec_path.parent
    spec = load_draft_spec(spec_path)
    builder.batch_dir = pd
    scanned = [builder.build_spec_text("draft-spec", spec_path.read_text(encoding="utf-8"))]
    sp = _resolve(builder, pd, spec["system"], data=False)
    system = builder.build_verify_only("spec_text", "system", sp.read_text(encoding="utf-8"), origin=sp)
    scanned.append(system)
    reqs, built = [], []
    for s in spec["sections"]:
        pp = _resolve(builder, pd, s["prompt"], data=False)
        prompt = builder.build_verify_only("spec_text", f"prompt-{s['id']}", pp.read_text(encoding="utf-8"), origin=pp)
        docs = []
        for rel in s.get("data", []):
            src = _resolve(builder, pd, rel, data=True)
            kind = DATA_KIND.get(src.suffix.lower())
            if kind is None:
                raise Refused(f"draft-sync: data file {rel!r} has no allowed extension ({sorted(DATA_KIND)})")
            docs.append(builder.build_verify_only(kind, src.name, src.read_text(encoding="utf-8"), origin=src))
        scanned += [prompt, *docs]
        built.append((s, prompt, docs))
    ttl = spec.get("cache_ttl", DEFAULT_CACHE_TTL)
    g, depth = _shared_depths([docs for _, _, docs in built])
    for (s, prompt, docs), d in zip(built, depth):
        cid = f"draft-{s['id']}"
        # source paths (as L1/L2 saw them) go into the manifest meta, so they are covered by manifest sha / plan_hash
        # and `run` can re-apply the CURRENT deny path globs to them (NB-1: a glob added after plan must refuse).
        src_paths = sorted({x["path"] for p in (system, prompt, *docs) for x in p.sources})
        # breakpoints: the system prompt (shared by all sections), the end of the data run every section shares,
        # and the end of the deepest run this section shares with another one (deduplicated, at most 3)
        bps = [system] + [docs[i - 1] for i in sorted({g, d}) if i > 0]
        req = M.build_request(cid, spec["model"], spec["effort"], spec["max_tokens"], system, docs, [prompt], ttl, None,
                              meta={"section": s["id"], "prompt_sha256": sha256_text(system.text),
                                    "source_paths": src_paths}, cache_after=bps)
        M.validate_request(req)
        _no_fallback(req.params, cid)
        reqs.append(req)
    per_req = [{"custom_id": r.custom_id, "hard_worst_usd": round(request_hard_worst(r.params), 6)} for r in reqs]
    worst = sum(request_hard_worst(r.params) for r in reqs)
    cap = min(float(spec.get("budget_usd") or LIMIT_BATCH_USD), LIMIT_BATCH_USD)
    if worst > cap:
        raise Refused(f"draft-sync plan refused: standard-price hard-worst ${worst:.4f} > cap ${cap:.2f}")
    _limits_ok(ledger, worst, now(), include_self=False)       # day / month already committed + this plan
    (pd / "pack").mkdir(exist_ok=True)
    seen = {p.pack_sha256: p for p in scanned}
    for p in seen.values():
        (pd / "pack" / f"{p.kind}-{p.pack_sha256[:8]}.txt").write_text(p.text, encoding="utf-8")
    (pd / "pack" / "pack_manifest.json").write_text(json.dumps([p.manifest_entry() for p in seen.values()], indent=1,
                                                               ensure_ascii=False, sort_keys=True), encoding="utf-8")
    msha = M.write_manifest(reqs, pd / "manifest.jsonl")
    plan_hash = compute_plan_hash(msha, worst, COST_FORMULA)
    plan_d = {"name": spec["name"], "use": USE, "model": spec["model"], "n_requests": len(reqs),
              "sections": [s["id"] for s in spec["sections"]], "est_usd_hard_worst": round(worst, 6),
              "price_basis": f"standard = {SYNC_PRICE_FACTOR}x batch table (prices.py), every block up to the last "
                             f"breakpoint written ({ttl}), full max_tokens",
              "cost_formula": COST_FORMULA, "cache_ttl": ttl,
              "cap_usd": cap, "manifest_sha256": msha, "plan_hash": plan_hash, "spec_sha256": sha256_file(spec_path),
              "pack_sha256": sha256_text(",".join(sorted(seen))), "deny_sha256": builder.deny.sha256,
              "pack_py_sha256": _pack_py_sha(), "masked_blocks": 0,
              "scan": [{"name": p.name, "kind": p.kind, "mode": p.mode, "pack_sha256": p.pack_sha256, **p.report}
                       for p in seen.values()],
              "requests": per_req,
              "approve_cmd": f"approve {pd} --usd {math.ceil(worst * 1e4) / 1e4:.4f}"}   # ceil: never below hard-worst
    (pd / "plan.json").write_text(json.dumps(plan_d, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    ledger.append("planned", batch_dir=str(pd), lane=USE, plan_hash=plan_hash, manifest_sha256=msha,
                  spec_sha256=plan_d["spec_sha256"], pack_sha256=plan_d["pack_sha256"], n_requests=len(reqs),
                  est_usd_hard_worst=plan_d["est_usd_hard_worst"], model=spec["model"], effort=spec["effort"],
                  deny_sha256=plan_d["deny_sha256"], pack_py_sha256=plan_d["pack_py_sha256"])
    return plan_d


def is_draft_plan(pd: Path) -> bool:
    try:
        return json.loads((Path(pd) / "plan.json").read_text(encoding="utf-8")).get("use") == USE
    except Exception:  # noqa: BLE001
        return False


def load_plan(pd: Path) -> tuple[dict, list[dict], float]:
    """plan.json + manifest rows, with manifest sha, hard-worst and plan_hash re-derived from the files."""
    pd = Path(pd)
    d = json.loads((pd / "plan.json").read_text(encoding="utf-8"))
    if d.get("use") != USE:
        raise Refused(f"{pd} is not a draft-sync plan")
    rows, msha = M.load_manifest(pd / "manifest.jsonl")
    if msha != d.get("manifest_sha256"):
        raise Refused("draft-sync: manifest.jsonl no longer matches plan.json (edited after plan)")
    formula = int(d.get("cost_formula", 1))
    worst = sum(request_hard_worst(r["params"], formula) for r in rows)
    if compute_plan_hash(msha, worst, formula) != d.get("plan_hash") or abs(worst - float(d.get("est_usd_hard_worst", -1))) > 1e-6:
        raise Refused("draft-sync: plan_hash / hard-worst in plan.json do not match the manifest (tampered)")
    for r in rows:
        _no_fallback(r["params"], r["custom_id"])
    return d, rows, worst


def _done_ids(ledger: Ledger, key: str) -> set:
    return {x.get("custom_id") for x in ledger.rows("sync_done", manifest_sha256=key) if x.get("rc") == 0}


def pending_sections(pd: Path, ledger: Ledger, key: str, rows: list[dict]) -> list[dict]:
    """Rows still to send: not (sync_done rc=0 in the ledger AND out/<section>.md on disk)."""
    done = _done_ids(ledger, key)
    return [r for r in rows if not (r["custom_id"] in done
                                    and (Path(pd) / "out" / f"{r['meta'].get('section') or r['custom_id']}.md").exists())]


def is_fallback(requested: str, returned) -> bool:
    """True unless `returned` is the requested id itself or that id plus a dated snapshot suffix (-YYYYMMDD).
    A bare prefix match is NOT enough: claude-sonnet-5-50 / claude-sonnet-5-5-1 / ...-5-5-mini are other models."""
    return not (isinstance(returned, str) and re.fullmatch(re.escape(requested) + r"(?:-\d{8})?", returned) is not None)


def sync_plan_for_approve(pd: Path, ledger: Ledger | None = None) -> dict:
    """The `sync_plan` argument ops.approve takes, built from a verified plan (called only after the session guard).
    With a ledger, a plan whose every section is already done is refused: that approval could never be consumed
    by a send and would only inflate committed_usd (sync approvals count one by one)."""
    d, rows, worst = load_plan(pd)
    if ledger is not None and not pending_sections(pd, ledger, d["plan_hash"], rows):
        raise Refused(f"draft-sync: nothing to do -- every section of {pd} is already done; no approval needed")
    return {"plan_hash": d["plan_hash"], "hard_worst_usd": round(worst, 6), "label": f"{USE}:{pd}"}


# ------------------------------------------------------------------------------------- run
def run(pd: Path, ledger: Ledger, client, builder: PackBuilder) -> dict:
    pd = Path(pd)
    d, rows, worst = load_plan(pd)
    key = d["plan_hash"]
    if not pending_sections(pd, ledger, key, rows):
        # all sections already rc=0 with out/<id>.md: no gate, no approval lookup, no ledger row -> an unused
        # approval stays unconsumed and committed_usd is untouched.
        return {"plan_hash": key, "approval_id": None, "spent_usd_est": 0.0, "approved_usd": None,
                "sections": [{"section": r["meta"].get("section") or r["custom_id"], "rc": 0, "skipped": "already done"}
                             for r in rows], "halted": None, "nothing_to_do": True}
    ok, why = tests_passed_current(ledger)
    if not ok:
        raise Refused(f"draft-sync run lapses: {why}")
    appr = _approved(ledger, key, worst, fresh=True, lane="sync")           # W1: sync lane only; once, before the loop: one approval = one run
    if float(appr.get("est_usd_hard_worst") or 0) + 1e-6 < worst:
        raise Refused(f"draft-sync: approval records hard-worst ${appr.get('est_usd_hard_worst')} < plan ${worst:.4f}")
    _limits_ok(ledger, worst, now(), include_self=True)
    budget = float(appr["approval"]["usd"])
    builder.batch_dir = pd
    for r in rows:                                             # deny source may have grown since plan: rescan
        for i, seg in enumerate(M._segments(r["params"])):
            builder.build_spec_text(f"{r['custom_id']}-seg{i}", seg)
        paths = r["meta"].get("source_paths")                  # ... and the L2 path checks plan applied
        if not isinstance(paths, list):
            raise Refused(f"draft-sync: {r['custom_id']} has no recorded source paths (plan predates NB-1); re-plan")
        for p in paths:
            g = builder.deny.path_hit(p)
            if g:
                raise Refused(f"draft-sync run: source {p!r} now matches deny glob {g!r} (added after plan)")
            if builder.deny.matcher.matches(p):
                raise Refused(f"draft-sync run: source path/name {p!r} now matches the deny source")
    done =_done_ids(ledger, key)
    for sub in ("raw", "out"):
        (pd / sub).mkdir(exist_ok=True)
    spent, results, halted = 0.0, [], None
    aid = approval_id(appr)
    for r in rows:
        cid, params = r["custom_id"], r["params"]
        sid = r["meta"].get("section") or cid
        if cid in done and (pd / "out" / f"{sid}.md").exists():
            results.append({"section": sid, "rc": 0, "skipped": "already done"})
            continue
        w = request_hard_worst(params, int(d.get("cost_formula", 1)))
        if spent + w > budget + 1e-9:
            halted = f"budget: spent ${spent:.4f} + next hard-worst ${w:.4f} > approved ${budget:.4f}"
            ledger.append("error", where=USE, manifest_sha256=key, custom_id=cid, error=f"halted before send: {halted}")
            break
        ledger.append("sync_sent", lane=USE, batch_dir=str(pd), plan_hash=key, manifest_sha256=key, custom_id=cid,
                      params_sha256=r["params_sha256"], approval_id=aid, est_usd_hard_worst=round(w, 6))
        try:
            msg = client.send_stream({"custom_id": cid, "params": params})     # SSE: long max_tokens safe
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
            ledger.append("sync_done", lane=USE, plan_hash=key, manifest_sha256=key, custom_id=cid, rc=1, error=err,
                          cost_est_usd=None)
            halted = f"send failed for {cid}: {err}"
            results.append({"section": sid, "rc": 1})
            break
        (pd / "raw" / f"{sid}.json").write_text(json.dumps(msg, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        cost = SYNC_PRICE_FACTOR * prices.cost_from_usage(params["model"], msg.get("usage") or {})
        spent += cost
        rec = {"custom_id": cid, "result": {"type": "succeeded", "message": msg, "error": None}}
        st = N.classify(rec, False)
        text = N.message_text(msg)
        hit = bool(text) and builder.deny.matcher.matches(text)        # L5, same check as collect
        # a dated snapshot id echoed for an alias request is not a fallback; any other model is
        fallback = is_fallback(params["model"], msg.get("model"))
        body = N.render_item(use=USE, batch_id=f"sync-{key[:12]}", custom_id=cid, row=r, rec=rec, status=st, cost=cost,
                             ended_at=now(), l5="quarantined" if hit else "pass",
                             extra={"model": params["model"], "plan_hash": key, "section": sid,
                                    "price_basis": f"standard ({SYNC_PRICE_FACTOR}x batch)"})
        if hit or fallback:
            (pd / "quarantine").mkdir(exist_ok=True)
            (pd / "quarantine" / f"{sid}.md").write_text(body, encoding="utf-8")
        else:
            (pd / "out" / f"{sid}.md").write_text(body, encoding="utf-8")
        rc = 2 if fallback else 3 if hit else (0 if st == "succeeded" else 4)     # 3 = L5 quarantine, 4 = refusal/max_tokens/...
        ledger.append("sync_done", lane=USE, plan_hash=key, manifest_sha256=key, custom_id=cid, rc=rc,
                      message_id=msg.get("id"), model_returned=msg.get("model"), result_status=st,
                      l5="quarantined" if hit else "pass", usage=msg.get("usage"), cost_est_usd=round(cost, 6))
        results.append({"section": sid, "rc": rc, "result_status": st, "l5": "quarantined" if hit else "pass",
                        "cost_est_usd": round(cost, 6)})
        if fallback:
            halted = f"model fallback: requested {params['model']}, returned {msg.get('model')}"
            ledger.append("error", where=USE, manifest_sha256=key, custom_id=cid, error=halted)
            break
        if spent > budget + 1e-9:
            halted = f"budget: spent ${spent:.4f} > approved ${budget:.4f}"
            ledger.append("error", where=USE, manifest_sha256=key, custom_id=cid, error=halted)
            break
    return {"plan_hash": key, "approval_id": aid, "spent_usd_est": round(spent, 6), "approved_usd": budget,
            "sections": results, "halted": halted}
