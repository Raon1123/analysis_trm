"""jev: TypeSafe Jev (System One) lane -- typed judgments (Noul / Choice / Score) through the same safety rails as
the Anthropic lanes (plan api-joyful-island D; 2026-11-05 user: "JEV와 같은 다른 라이브러리와의 호환").

Jev returns typed answers with probabilities and no free text (docs.typesafe.ai: /v1/systemone; models jev-1.14.0,
jev-1.13.0; 64k tokens per request / 32k per question; input billed, output free; no batch API, no cross-request
cache, no documented seed). The cache analogue is fan-out: one request carries every question about one shared
`state`, which Jev ingests once. Fits triage / verdict-token classification / rubric-item checks; not memos,
drafts or free-text readings.

  plan <spec>      offline. spec = {name, model (JEV_MODELS), questions, items: [{id, state: [<path>...]}],
                   repeats?: int (default 1; >1 measures decision stability), budget_usd?}.
                   questions (inline in spec.yaml, so the L1 allow-list stays unchanged) =
                   {qid: {type: noul|choice|score, instructions: str, criteria?}}; every
                   instruction / criterion text becomes a spec_text Pack (deny-scanned, refused on any hit);
                   every state file a verify-only Pack (lab .md/.txt -> seat_input, .csv/.json -> u1_data). Choice
                   option names are code-controlled labels ([a-z0-9_-]{1,40}). Writes manifest.jsonl, plan.json
                   (plan_hash over manifest sha + hard-worst), ledger `planned` (lane jev).
  approve          user-only, unchanged (`approve <plan_dir> --usd X`, refused inside a Claude Code session).
  run <plan_dir>   metered, sends lab text to a third party (TypeSafe). Re-derives manifest sha / hard-worst /
                   plan_hash, needs current tests_passed + a fresh matching sync approval, rescans every text
                   segment against the current deny source, then per request: `sync_sent` BEFORE the send,
                   raw/<id>.json + out/results.jsonl (normalized), `sync_done` with rc (0 ok, 1 send raised,
                   2 model mismatch, 4 response failed validation). The API key is read only by the SDK from
                   TYPESAFE_API_KEY; this module never reads or prints it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

from .common import Halt, canon_json, est_tokens, sha256_file, sha256_text
from .ledger import LIMIT_BATCH_USD, Ledger, approval_id, now
from .pack import Pack, PackBuilder
from .spec import SpecError

USE = "jev"
# Pinned versions only (aliases jev-latest / jev-preview move on release; thresholds tuned on one version).
JEV_MODELS = ("jev-1.14.0", "jev-1.13.0")
# $ per input MTok (docs.typesafe.ai/models.md, read 2026-11-05: "$42 / $0.042" per Btok / Mtok); output is free.
# The docs figure is far below typical LLM prices -- confirm in the TypeSafe console before relying on it.
JEV_PRICE_IN_PER_MTOK = 0.042
JEV_PRICE_SAFETY = 10.0          # hard-worst multiplier on the unconfirmed docs price
MAX_REQUEST_TOKENS = 64_000      # state + all questions
MAX_QUESTION_TOKENS = 32_000     # state + one question
QTYPES = ("noul", "choice", "score")
QID_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
LABEL_RE = re.compile(r"^[a-z0-9_-]{1,40}$")
ITEM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
SPEC_KEYS = {"name", "model", "questions", "items", "repeats", "budget_usd"}
DATA_KIND = {".md": "seat_input", ".txt": "seat_input", ".csv": "u1_data", ".json": "u1_data"}


class Refused(Halt):
    pass


class JevError(Halt):
    pass


# ------------------------------------------------------------------------------------- capabilities
CAPABILITIES = {
    "anthropic": {"batch": True, "prefix_cache": True, "images": True, "output_kind": "free_text",
                  "max_context_tokens": 1_000_000},
    "jev": {"batch": False, "prefix_cache": False, "images": True, "output_kind": "typed_judgment",
            "max_context_tokens": MAX_REQUEST_TOKENS, "shared_state_fanout": True},
}


# ------------------------------------------------------------------------------------- spec
def load_spec(path: Path) -> dict:
    try:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SpecError(f"jev spec unreadable: {type(e).__name__}")
    if not isinstance(spec, dict):
        raise SpecError("jev spec is not a mapping")
    for k in ("name", "model", "questions", "items"):
        if k not in spec:
            raise SpecError(f"jev spec: missing key {k!r}")
    if set(spec) - SPEC_KEYS:
        raise SpecError(f"jev spec: unknown keys {sorted(set(spec) - SPEC_KEYS)}")
    if spec["model"] not in JEV_MODELS:
        raise SpecError(f"jev spec: model must be a pinned version {JEV_MODELS} (no aliases)")
    rep = spec.get("repeats", 1)
    if not isinstance(rep, int) or isinstance(rep, bool) or not 1 <= rep <= 5:
        raise SpecError("jev spec: repeats must be an integer 1..5")
    b = spec.get("budget_usd")
    if b is not None and (not isinstance(b, (int, float)) or b <= 0):
        raise SpecError("jev spec: budget_usd (optional) must be a positive number")
    items = spec["items"]
    if not isinstance(items, list) or not items:
        raise SpecError("jev spec: items must be a non-empty list")
    ids = []
    for it in items:
        if not isinstance(it, dict) or set(it) != {"id", "state"} or not isinstance(it["id"], str) or not ITEM_RE.fullmatch(it["id"]):
            raise SpecError("jev spec: each item is exactly {id: [a-z0-9_-], state: [<path>...]}")
        if not isinstance(it["state"], list) or not it["state"] or not all(isinstance(x, str) and x for x in it["state"]):
            raise SpecError(f"jev spec: item {it['id']} state must be a non-empty list of paths")
        ids.append(it["id"])
    if len(set(ids)) != len(ids):
        raise SpecError("jev spec: duplicate item id")
    return spec


def load_questions(q) -> dict:
    if not isinstance(q, dict) or not q:
        raise SpecError("jev questions: a non-empty mapping {qid: {type, instructions, criteria?}}")
    for qid, d in q.items():
        if not isinstance(qid, str) or not QID_RE.fullmatch(qid):
            raise SpecError(f"jev questions: id {qid!r} must match {QID_RE.pattern}")
        if not isinstance(d, dict) or set(d) - {"type", "instructions", "criteria"} or d.get("type") not in QTYPES:
            raise SpecError(f"jev questions: {qid} must be {{type: noul|choice|score, instructions, criteria?}}")
        if not isinstance(d.get("instructions"), str) or not d["instructions"].strip():
            raise SpecError(f"jev questions: {qid} instructions must be a non-empty string")
        c, t = d.get("criteria"), d["type"]
        if t == "noul" and c is not None and (not isinstance(c, dict) or set(c) - {"true", "false"} or not c
                                              or not all(isinstance(v, str) and v for v in c.values())):
            raise SpecError(f"jev questions: {qid} noul criteria must be {{true?: str, false?: str}}")
        if t == "choice" and (not isinstance(c, dict) or not 1 <= len(c) <= 255 or not all(isinstance(k, str) and LABEL_RE.fullmatch(k) for k in c)
                              or not all(v is None or (isinstance(v, str) and v) for v in c.values())):
            raise SpecError(f"jev questions: {qid} choice criteria must map 1..255 labels [a-z0-9_-] to a description or null")
        if t == "score" and (not isinstance(c, list) or not 1 <= len(c) <= 10 or not all(isinstance(v, str) and v for v in c)):
            raise SpecError(f"jev questions: {qid} score criteria must be a list of 1..10 level descriptions")
    return q


# ------------------------------------------------------------------------------------- request build / validate
def _q_texts(q: dict) -> list[tuple[str, str]]:
    """(name, text) of every free-text leaf of one question, in a fixed order."""
    out = [("instructions", q["instructions"])]
    c = q.get("criteria")
    if isinstance(c, dict):
        out += [(f"criteria.{k}", v) for k, v in c.items() if v is not None]
    elif isinstance(c, list):
        out += [(f"criteria.{i}", v) for i, v in enumerate(c)]
    return out


def build_params(model: str, state: dict[str, Pack], questions: dict[str, dict], qpacks: dict[str, Pack]) -> dict:
    """state documents -> {name: text}; questions with every text leaf replaced by its Pack's bytes (identical text)."""
    qs = {}
    for qid, q in questions.items():
        d = {"type": q["type"], "instructions": qpacks[f"{qid}.instructions"].text}
        c = q.get("criteria")
        if isinstance(c, dict):
            d["criteria"] = {k: (qpacks[f"{qid}.criteria.{k}"].text if v is not None else None) for k, v in c.items()}
        elif isinstance(c, list):
            d["criteria"] = [qpacks[f"{qid}.criteria.{i}"].text for i in range(len(c))]
        qs[qid] = d
    return {"model": model, "state": {n: p.text for n, p in state.items()}, "questions": qs}


def _q_leaves(questions: dict) -> list:
    """Every free-text leaf of a questions mapping (instructions, criteria descriptions / levels), in order."""
    out = []
    for q in questions.values():
        out.append(q.get("instructions"))
        c = q.get("criteria")
        if isinstance(c, dict):
            out += [v for v in c.values() if v is not None]
        elif isinstance(c, list):
            out += list(c)
    return out


def _text_leaves(params: dict) -> list:
    st = params.get("state")
    return (list(st.values()) if isinstance(st, dict) else [st]) + _q_leaves(params.get("questions") or {})


def validate_params(params: dict, packs: list[Pack]) -> None:
    """Single-path invariant for the Jev lane: every text the request carries is the exact bytes of a sealed Pack
    it references, and every referenced Pack appears. Structure is code-controlled (no assets, no extra keys)."""
    if set(params) != {"model", "state", "questions"}:
        raise JevError(f"jev params keys must be exactly model/state/questions (got {sorted(params)})")
    if params["model"] not in JEV_MODELS:
        raise JevError(f"jev model {params['model']!r} is not a pinned version")
    if not packs or any(not isinstance(p, Pack) or not p.sealed for p in packs):
        raise JevError("jev request references no Pack or an unsealed Pack")
    st = params["state"]
    if not isinstance(st, dict) or not st or not all(isinstance(k, str) and isinstance(v, str) for k, v in st.items()):
        raise JevError("jev state must be a non-empty {name: text} mapping")
    if list(st) != [f"doc{i}" for i in range(1, len(st) + 1)]:
        raise JevError("jev state keys must be the neutral doc1..docN (no file names reach the model)")
    qs = params["questions"]
    if not isinstance(qs, dict) or not qs:
        raise JevError("jev questions must be a non-empty mapping")
    for qid, q in qs.items():
        if not QID_RE.fullmatch(qid) or not isinstance(q, dict) or set(q) - {"type", "instructions", "criteria"} or q.get("type") not in QTYPES:
            raise JevError(f"jev question {qid!r} has an invalid shape")
        if q["type"] == "choice" and not all(LABEL_RE.fullmatch(k) for k in (q.get("criteria") or {})):
            raise JevError(f"jev question {qid}: choice labels must be code-controlled [a-z0-9_-] tokens")
    texts = {p.text for p in packs}
    used = set()
    for t in _text_leaves(params):
        if not isinstance(t, str) or t not in texts:
            raise JevError("jev request text is not the bytes of a referenced Pack")
        used.add(t)
    if used != texts:
        raise JevError("a referenced Pack's bytes do not appear in the jev request")
    s_tok = sum(est_tokens(v) for v in st.values())
    q_tok = [sum(est_tokens(t) for t in _q_leaves({k: v})) for k, v in qs.items()]
    if s_tok + max(q_tok) > MAX_QUESTION_TOKENS or s_tok + sum(q_tok) > MAX_REQUEST_TOKENS:
        raise JevError(f"jev request too large: state {s_tok} + questions {q_tok} est tok "
                       f"(limits {MAX_QUESTION_TOKENS}/question, {MAX_REQUEST_TOKENS}/request)")


def request_hard_worst(params: dict) -> float:
    """Input tokens only (output is free). Conservative: est_tokens over-estimates, state is counted once per
    question (billing granularity is not documented), and the unconfirmed docs price is multiplied by JEV_PRICE_SAFETY."""
    s_tok = sum(est_tokens(v) for v in params["state"].values())
    q_tok = sum(est_tokens(t) for t in _q_leaves(params["questions"]))
    n_q = len(params["questions"])
    return (s_tok * n_q + q_tok) * JEV_PRICE_IN_PER_MTOK * JEV_PRICE_SAFETY / 1e6


def _usage_ok(usage) -> bool:
    """usage must be a mapping whose input_tokens is a finite non-negative integer (output is free, not checked)."""
    if not isinstance(usage, dict):
        return False
    n = usage.get("input_tokens")
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def cost_from_usage(usage: dict) -> float:
    return int((usage or {}).get("input_tokens") or 0) * JEV_PRICE_IN_PER_MTOK / 1e6


# ------------------------------------------------------------------------------------- response
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _unit(x) -> bool:
    return _num(x) and 0 <= x <= 1


def normalize(raw: dict, params: dict) -> dict:
    """Validate a /v1/systemone response against its request and flatten it:
    {qid: {type, value, confidence, probabilities}} where value = P(yes) for noul, the chosen label for choice,
    the expected level for score. Raises JevError on any mismatch (missing/extra answer, wrong type, value outside
    the asked options)."""
    if not isinstance(raw, dict):
        raise JevError("response is not a JSON object")
    ans = raw.get("answers")
    qs = params["questions"]
    if not isinstance(ans, dict) or set(ans) != set(qs):
        raise JevError(f"answers {sorted(ans) if isinstance(ans, dict) else ans!r} != questions {sorted(qs)}")
    out = {}
    for qid, q in qs.items():
        a = ans[qid]
        if not isinstance(a, dict) or a.get("type") != q["type"]:
            raise JevError(f"{qid}: answer type {a.get('type') if isinstance(a, dict) else a!r} != {q['type']}")
        t = q["type"]
        if t == "noul":
            v = a.get("noul")
            if not _unit(v):
                raise JevError(f"{qid}: noul {v!r} not a number in [0, 1]")
            out[qid] = {"type": t, "value": float(v), "confidence": None, "probabilities": {"true": float(v), "false": 1 - float(v)}}
            continue
        allowed = set(q["criteria"]) if t == "choice" else {str(i) for i in range(len(q["criteria"]))}
        v, pr, conf = a.get(t), a.get("probabilities", {}), a.get("confidence")
        if not isinstance(pr, dict) or not set(pr) <= allowed or not all(_unit(x) for x in pr.values()):
            raise JevError(f"{qid}: probabilities must map the asked options to numbers in [0, 1]")
        if conf is not None and not _unit(conf):
            raise JevError(f"{qid}: confidence {conf!r} not a number in [0, 1]")
        if t == "choice":
            if not isinstance(v, str) or v not in allowed:
                raise JevError(f"{qid}: choice {v!r} outside the asked labels")
            out[qid] = {"type": t, "value": v, "confidence": conf, "probabilities": pr}
        else:
            if not _num(v) or not 0 <= v <= len(allowed) - 1:
                raise JevError(f"{qid}: score {v!r} outside 0..{len(allowed) - 1}")
            out[qid] = {"type": t, "value": float(v), "confidence": conf, "probabilities": pr}
    return out


# ------------------------------------------------------------------------------------- clients
class RealJevClient:
    """typesafe-sdk wrapper. Imported lazily; the SDK reads TYPESAFE_API_KEY itself. Returns the raw HTTP JSON
    (the documented OpenAPI shape) because the SDK's typed objects do not document usage / probabilities."""

    def __init__(self):
        from typesafe_sdk import TypeSafeClient         # lazy: only the metered `run` constructs this
        self._c = TypeSafeClient()
        try:
            from importlib.metadata import version
            self.sdk_version = version("typesafe-sdk")
        except Exception:  # noqa: BLE001
            self.sdk_version = None

    def send(self, params: dict) -> dict:
        res = self._c.system_one(params["state"], params["questions"], model=params["model"])
        raw = res.raw_http_response.json()
        return {"raw": raw, "request_id": getattr(res, "request_id", None)}


class FakeJevClient:
    """Deterministic offline stand-in: noul 0.8, first choice label, score at the top level."""

    def __init__(self, model_override: str | None = None, fail: bool = False, bad_answer: bool = False):
        self.calls, self.model_override, self.fail, self.bad_answer = [], model_override, fail, bad_answer
        self.sdk_version = "fake"

    def send(self, params: dict) -> dict:
        self.calls.append(params)
        if self.fail:
            raise RuntimeError("fake send failure")
        ans = {}
        for qid, q in params["questions"].items():
            if q["type"] == "noul":
                ans[qid] = {"type": "noul", "noul": 0.8}
            elif q["type"] == "choice":
                labs = list(q["criteria"])
                ans[qid] = {"type": "choice", "choice": labs[0], "confidence": 0.7,
                            "probabilities": {k: (0.7 if i == 0 else 0.3 / max(1, len(labs) - 1)) for i, k in enumerate(labs)}}
            else:
                n = len(q["criteria"])
                ans[qid] = {"type": "score", "score": float(n - 1), "confidence": 0.6,
                            "legend": {str(i): t for i, t in enumerate(q["criteria"])},
                            "probabilities": {str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)}}
        if self.bad_answer:
            ans.pop(next(iter(ans)))
        n_in = sum(est_tokens(t) for t in _text_leaves(params))
        return {"raw": {"model": self.model_override or params["model"], "answers": ans,
                        "usage": {"input_tokens": n_in, "output_tokens": len(ans)}}, "request_id": "req_fake"}


# ------------------------------------------------------------------------------------- plan
def is_jev_plan(pd: Path) -> bool:
    try:
        return json.loads((Path(pd) / "plan.json").read_text(encoding="utf-8")).get("use") == USE
    except Exception:  # noqa: BLE001
        return False


def compute_plan_hash(manifest_sha256: str, hard_worst_usd: float) -> str:
    return sha256_text(canon_json({"use": USE, "manifest_sha256": manifest_sha256, "hard_worst_usd": round(hard_worst_usd, 6),
                                   "price_in_per_mtok": JEV_PRICE_IN_PER_MTOK, "price_safety": JEV_PRICE_SAFETY}))


def _resolve(builder: PackBuilder, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    for root in (builder.lab_root, builder.repo_root):
        if (Path(root) / p).is_file():
            return Path(root) / p
    raise Refused(f"jev: file {rel!r} not found")


def plan(spec_path: Path, builder: PackBuilder, ledger: Ledger) -> dict:
    from . import manifest as M
    from .ops import _limits_ok, _pack_py_sha
    spec_path = Path(spec_path)
    pd = spec_path.parent
    spec = load_spec(spec_path)
    builder.batch_dir = pd
    scanned = [builder.build_spec_text("jev-spec", spec_path.read_text(encoding="utf-8"))]
    questions = load_questions(spec["questions"])
    qpacks = {}
    for qid, q in questions.items():
        for nm, t in _q_texts(q):
            qpacks[f"{qid}.{nm}"] = builder.build_spec_text(f"q-{qid}-{nm}", t)
    rows = []
    for it in spec["items"]:
        state, src_paths = {}, []
        for i, rel in enumerate(it["state"], 1):
            src = _resolve(builder, rel)
            kind = DATA_KIND.get(src.suffix.lower())
            if kind is None:
                raise Refused(f"jev: state file {rel!r} has no allowed extension ({sorted(DATA_KIND)})")
            pk = builder.build_verify_only(kind, src.name, src.read_text(encoding="utf-8"), origin=src)
            state[f"doc{i}"] = pk                 # neutral names: no file name / slug reaches the model
            src_paths += [x["path"] for x in pk.sources]
            scanned.append(pk)
        params = build_params(spec["model"], state, questions, qpacks)
        packs = [*state.values(), *qpacks.values()]
        validate_params(params, packs)
        for j in range(1, spec.get("repeats", 1) + 1):
            cid = f"jev-{it['id']}-r{j}"
            if not M.CUSTOM_ID_RE.fullmatch(cid):
                raise Refused(f"jev: custom_id {cid!r} invalid")
            rows.append({"custom_id": cid, "params": params, "params_sha256": sha256_text(canon_json(params)),
                         "pack_sha256s": [p.pack_sha256 for p in packs],
                         "meta": {"item": it["id"], "repeat": j, "source_paths": sorted(set(src_paths))}})
    scanned += list(qpacks.values())
    per_req = [{"custom_id": r["custom_id"], "hard_worst_usd": round(request_hard_worst(r["params"]), 6)} for r in rows]
    worst = max(sum(request_hard_worst(r["params"]) for r in rows), 1e-6)
    cap = min(float(spec.get("budget_usd") or LIMIT_BATCH_USD), LIMIT_BATCH_USD)
    if worst > cap:
        raise Refused(f"jev plan refused: hard-worst ${worst:.6f} > cap ${cap:.2f}")
    _limits_ok(ledger, worst, now(), include_self=False)
    (pd / "pack").mkdir(exist_ok=True)
    seen = {p.pack_sha256: p for p in scanned}
    for p in seen.values():
        (pd / "pack" / f"{p.kind}-{p.pack_sha256[:8]}.txt").write_text(p.text, encoding="utf-8")
    data = ("\n".join(canon_json(r) for r in rows) + "\n").encode("utf-8")
    (pd / "manifest.jsonl").write_bytes(data)
    msha = sha256_text(data.decode("utf-8"))
    plan_hash = compute_plan_hash(msha, worst)
    d = {"name": spec["name"], "use": USE, "model": spec["model"], "n_requests": len(rows), "n_questions": len(questions),
         "est_usd_hard_worst": round(worst, 6), "cap_usd": cap, "manifest_sha256": msha, "plan_hash": plan_hash,
         "price_basis": f"input only, ${JEV_PRICE_IN_PER_MTOK}/MTok (docs, unconfirmed) x {JEV_PRICE_SAFETY} safety, "
                        "state counted once per question",
         "spec_sha256": sha256_file(spec_path), "pack_sha256": sha256_text(",".join(sorted(seen))),
         "deny_sha256": builder.deny.sha256, "pack_py_sha256": _pack_py_sha(), "requests": per_req,
         "egress": "TypeSafe /v1/systemone (third party): user confirms the sample list before approve",
         "approve_cmd": f"approve {pd} --usd {math.ceil(worst * 1e6) / 1e6:.6f}"}
    (pd / "plan.json").write_text(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    ledger.append("planned", lane=USE, batch_dir=str(pd), manifest_sha256=msha, plan_hash=plan_hash, n_requests=len(rows),
                  est_usd_hard_worst=d["est_usd_hard_worst"], model=spec["model"], deny_sha256=d["deny_sha256"])
    return d


def load_plan(pd: Path) -> tuple[dict, list[dict], float]:
    pd = Path(pd)
    d = json.loads((pd / "plan.json").read_text(encoding="utf-8"))
    if d.get("use") != USE:
        raise Refused(f"{pd} is not a jev plan")
    data = (pd / "manifest.jsonl").read_bytes()
    msha = sha256_text(data.decode("utf-8"))
    if msha != d.get("manifest_sha256"):
        raise Refused("jev: manifest.jsonl no longer matches plan.json (edited after plan)")
    rows = [json.loads(ln) for ln in data.decode("utf-8").splitlines() if ln.strip()]
    for r in rows:
        if sha256_text(canon_json(r["params"])) != r["params_sha256"]:
            raise Refused(f"jev: {r['custom_id']} params_sha256 mismatch")
    worst = max(sum(request_hard_worst(r["params"]) for r in rows), 1e-6)
    if compute_plan_hash(msha, worst) != d.get("plan_hash") or abs(worst - float(d.get("est_usd_hard_worst", -1))) > 1e-6:
        raise Refused("jev: plan_hash / hard-worst in plan.json do not match the manifest (tampered)")
    return d, rows, worst


def _done_ids(ledger: Ledger, key: str) -> set:
    return {x.get("custom_id") for x in ledger.rows("sync_done", manifest_sha256=key, lane=USE) if x.get("rc") == 0}


def sync_plan_for_approve(pd: Path, ledger: Ledger | None = None) -> dict:
    d, rows, worst = load_plan(pd)
    if ledger is not None and not [r for r in rows if r["custom_id"] not in _done_ids(ledger, d["plan_hash"])]:
        raise Refused(f"jev: nothing to do -- every request of {pd} is already done; no approval needed")
    return {"plan_hash": d["plan_hash"], "hard_worst_usd": round(worst, 6), "label": f"{USE}:{pd}"}


# ------------------------------------------------------------------------------------- run
def run(pd: Path, ledger: Ledger, client, builder: PackBuilder) -> dict:
    from .ops import _approved, _limits_ok, tests_passed_current
    pd = Path(pd)
    d, rows, worst = load_plan(pd)
    key = d["plan_hash"]
    done = _done_ids(ledger, key)
    pending = [r for r in rows if r["custom_id"] not in done]
    if not pending:
        return {"plan_hash": key, "nothing_to_do": True, "requests": [], "halted": None}
    ok, why = tests_passed_current(ledger)
    if not ok:
        raise Refused(f"jev run lapses: {why}")
    appr = _approved(ledger, key, worst, fresh=True, lane="sync")
    if float(appr.get("est_usd_hard_worst") or 0) + 1e-6 < worst:
        raise Refused(f"jev: approval records hard-worst ${appr.get('est_usd_hard_worst')} < plan ${worst:.6f}")
    _limits_ok(ledger, worst, now(), include_self=True)
    builder.batch_dir = pd
    for r in pending:          # deny source may have grown since plan: rescan text, identifiers and source paths
        params = r["params"]
        packs = [builder.build_spec_text(f"{r['custom_id']}-seg{i}", t) for i, t in enumerate(_text_leaves(params))]
        ids = [*params["state"], *params["questions"],
               *(k for q in params["questions"].values() if isinstance(q.get("criteria"), dict) for k in q["criteria"])]
        for i, k in enumerate(ids):                     # qids, choice labels, state keys reach the model too
            builder.build_spec_text(f"{r['custom_id']}-id{i}", k)
        validate_params(params, packs)
        paths = r["meta"].get("source_paths")
        if not isinstance(paths, list) or not paths:
            raise Refused(f"jev run: {r['custom_id']} has no recorded source paths; re-plan")
        for p in paths:
            if builder.deny.path_hit(p) or builder.deny.matcher.matches(p):
                raise Refused(f"jev run: source {p!r} now matches the deny source (added after plan)")
    for sub in ("raw", "out"):
        (pd / sub).mkdir(exist_ok=True)
    aid = approval_id(appr)
    results, halted, spent = [], None, 0.0
    budget = float(appr["approval"]["usd"])
    for r in pending:
        cid, params = r["custom_id"], r["params"]
        w = request_hard_worst(params)
        if spent + w > budget + 1e-9:
            halted = f"next request {cid} could exceed the approved ${budget}"
            break
        ledger.append("sync_sent", lane=USE, batch_dir=str(pd), plan_hash=key, manifest_sha256=key, custom_id=cid,
                      params_sha256=r["params_sha256"], approval_id=aid, est_usd_hard_worst=round(w, 6))
        try:
            got = client.send(params)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
            ledger.append("sync_done", lane=USE, plan_hash=key, manifest_sha256=key, custom_id=cid, rc=1, error=err)
            results.append({"custom_id": cid, "rc": 1})
            halted = f"send failed for {cid}: {err}"
            break
        # every path below ends in exactly one sync_done for this sync_sent (r2 G2/G3)
        (pd / "raw" / f"{cid}.json").write_text(json.dumps(got, ensure_ascii=False, sort_keys=True, indent=1, default=repr),
                                                encoding="utf-8")
        got_d = got if isinstance(got, dict) else {}
        raw = got_d.get("raw")
        usage = raw.get("usage") if isinstance(raw, dict) else None
        req_id = got_d.get("request_id")
        norm = None
        if not isinstance(raw, dict):
            rc, err, usage = 4, f"response is not a JSON object ({type(raw).__name__})", None
        elif not _usage_ok(usage):
            rc, err = 4, f"usage malformed: {str(usage)[:200]}"
            usage = None
        elif raw.get("model") != params["model"]:
            rc, err = 2, f"model mismatch: requested {params['model']}, returned {raw.get('model')}"
        else:
            try:
                norm, rc, err = normalize(raw, params), 0, None
            except Exception as e:  # noqa: BLE001  -- any malformed response closes its sync_sent with rc 4
                norm, rc, err = None, 4, f"{type(e).__name__}: {str(e)[:300]}"
        cost = cost_from_usage(usage) if usage is not None else w     # unknown usage: charge the request's hard-worst
        spent += cost
        if norm is not None:
            with (pd / "out" / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"custom_id": cid, "item": r["meta"]["item"], "repeat": r["meta"]["repeat"],
                                    "model": raw.get("model"), "answers": norm, "usage": usage,
                                    "request_id": req_id}, ensure_ascii=False, sort_keys=True) + "\n")
        ledger.append("sync_done", lane=USE, plan_hash=key, manifest_sha256=key, custom_id=cid, rc=rc, error=err,
                      model_returned=raw.get("model") if isinstance(raw, dict) else None,
                      request_id=req_id if isinstance(req_id, (str, type(None))) else repr(req_id)[:100], usage=usage,
                      cost_est_usd=round(cost, 8), sdk_version=getattr(client, "sdk_version", None))
        results.append({"custom_id": cid, "rc": rc})
        if rc == 2:
            halted = err
            break
    return {"plan_hash": key, "approval_id": aid, "spent_usd_est": round(spent, 8), "requests": results,
            "halted": halted, "nothing_to_do": False}
