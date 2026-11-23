"""Document-consistency pre-screen (pre-verifier lint) over the Claude API.

A tool-less API seat: the packet is one RESULTS document plus its frozen
pre-registration, nothing else. Output is a list of suspected document defects
(doc_lint_schema.json). Attention routing only — never a verdict, never a gate.

Metered lane: run under `with-anthropic-key` so ANTHROPIC_API_KEY reaches this
process only. Design: lab/33_integrations/11_lab-integrations/2026-10-13_claude-api-pipeline.md §7.

  with-anthropic-key uv run --with anthropic==1.13.0 python -I tools/api_seats/doc_lint.py \
      --plan PLAN.json --out RUN_DIR --models claude-sonnet-5-5,claude-opus-5-5 --n 3 \
      --budget-usd 10 [--dry-run]

PLAN.json: [{"label": "015_v_r1", "results": "<path>", "prereg": "<path>"}, ...]

Per call it writes, in order: <label>/<model>/s<k>.packet.json (exact request
params) and its sha256 into ledger.jsonl BEFORE the request is sent; then
s<k>.response.json (raw message), s<k>.parsed.json (findings + quote checks).
No server-side fallbacks (they would swap the model under comparison); a refusal,
max_tokens stop or unparseable output is recorded as a failed sample.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "doc_lint_prompt.md").read_text(encoding="utf-8")
SCHEMA = json.loads((HERE / "doc_lint_schema.json").read_text(encoding="utf-8"))
INSTRUCTION = ("Review RESULTS against itself and against PREREG and report every document "
               "defect you find in the required JSON format.")
EFFORT = "high"
MAX_TOKENS = 32000
# USD per MTok: input, output, cache write (5-min TTL), cache read. Claude API list
# prices as cached in the claude-api skill (2026-10-06).
PRICE = {"claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
         "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20)}
# Output tokens assumed for the pre-call budget estimate until a model has a
# measured call; then the model's largest observed call cost is used instead.
EST_OUTPUT_TOKENS = 16000


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def doc_block(title: str, text: str, cache: bool) -> dict:
    b = {"type": "document", "title": title,
         "source": {"type": "text", "media_type": "text/plain", "data": text}}
    if cache:
        b["cache_control"] = {"type": "ephemeral"}
    return b


def build_params(model: str, results: str, prereg: str) -> dict:
    # PREREG first: versions of the same experiment share it, so the prefix up to
    # its breakpoint is reusable across versions as well as across samples.
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": PROMPT}],
        "messages": [{"role": "user", "content": [
            doc_block("PREREG", prereg, cache=True),
            doc_block("RESULTS", results, cache=True),
            {"type": "text", "text": INSTRUCTION},
        ]}],
        # Explicit, not omitted: both models default to adaptive, but naming it removes any
        # doubt about which thinking mode the comparison ran under.
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def cost_usd(model: str, u) -> float:
    pin, pout, pw, pr = PRICE[model]
    get = (lambda k: (getattr(u, k, None) or 0)) if not isinstance(u, dict) else (lambda k: u.get(k) or 0)
    return (get("input_tokens") * pin + get("output_tokens") * pout
            + get("cache_creation_input_tokens") * pw + get("cache_read_input_tokens") * pr) / 1e6


def quote_checks(findings: list, results: str, prereg: str) -> list:
    src = {"RESULTS": results, "PREREG": prereg}
    squash = lambda s: " ".join(s.split())
    out = []
    for f in findings:
        text = src.get(f.get("quote_source"), "")
        q = f.get("quote", "")
        out.append({"exact": bool(q) and q in text,
                    "ws_normalized": bool(q) and squash(q) in squash(text)})
    return out


class Budget:
    def __init__(self, cap: float, ledger: Path):
        self.cap, self.spent, self.lock, self.ledger = cap, 0.0, threading.Lock(), ledger
        self.max_seen: dict[str, float] = {}

    def reserve(self, model: str, prior: float) -> bool:
        with self.lock:
            est = max(prior, self.max_seen.get(model, 0.0))
            return self.spent + est <= self.cap

    def charge(self, model: str, c: float) -> float:
        with self.lock:
            self.spent += c
            self.max_seen[model] = max(self.max_seen.get(model, 0.0), c)
            return self.spent

    def log(self, row: dict) -> None:
        with self.lock, self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_one(client, budget: Budget, out: Path, item: dict, model: str, k: int, texts: dict,
            prior_est: float) -> None:
    label = item["label"]
    d = out / label / model
    d.mkdir(parents=True, exist_ok=True)
    params = build_params(model, texts["results"], texts["prereg"])
    pbytes = json.dumps(params, ensure_ascii=False, sort_keys=True).encode("utf-8")
    (d / f"s{k}.packet.json").write_bytes(pbytes)
    base = {"label": label, "model": model, "sample": k, "packet_sha256": sha256(pbytes),
            "results_sha256": sha256(texts["results"].encode()), "prereg_sha256": sha256(texts["prereg"].encode())}
    if not budget.reserve(model, prior_est):
        budget.log({**base, "ts": now(), "event": "skipped_budget", "spent_usd": round(budget.spent, 4)})
        return
    budget.log({**base, "ts": now(), "event": "sent"})
    t0 = time.time()
    try:
        with client.messages.stream(**params) as stream:
            msg = stream.get_final_message()
            # The final message of a stream has no _request_id; the stream carries it.
            request_id = stream.request_id
    except anthropic.APIStatusError as e:
        budget.log({**base, "ts": now(), "event": "api_error", "status": e.status_code,
                    "error": str(e.message)[:300]})
        return
    except anthropic.APIConnectionError as e:
        budget.log({**base, "ts": now(), "event": "connection_error", "error": str(e)[:300]})
        return
    elapsed = time.time() - t0
    # Charge before any write: an error past this point must not leave spend uncounted.
    c = cost_usd(model, msg.usage)
    spent = budget.charge(model, c)
    (d / f"s{k}.response.json").write_text(msg.to_json(), encoding="utf-8")
    status, findings, perr = "ok", [], None
    if msg.stop_reason == "refusal":
        status = "refusal"
    elif msg.stop_reason == "max_tokens":
        status = "max_tokens"
    else:
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            findings = json.loads(text)["findings"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            status, perr = "parse_error", f"{type(e).__name__}: {e}"[:200]
    qc = quote_checks(findings, texts["results"], texts["prereg"])
    (d / f"s{k}.parsed.json").write_text(json.dumps(
        {"status": status, "parse_error": perr, "findings": findings, "quote_checks": qc},
        ensure_ascii=False, indent=1), encoding="utf-8")
    u = msg.usage
    budget.log({**base, "ts": now(), "event": "done", "status": status, "request_id": request_id,
                "model_returned": msg.model, "model_match": msg.model == model,
                "stop_reason": msg.stop_reason, "elapsed_s": round(elapsed, 1),
                "usage": {k2: getattr(u, k2, None) for k2 in
                          ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
                "cost_usd": round(c, 5), "spent_usd": round(spent, 5), "n_findings": len(findings),
                "quote_exact": sum(q["exact"] for q in qc), "quote_ws": sum(q["ws_normalized"] for q in qc)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", default="claude-sonnet-5-5,claude-opus-5-5")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true", help="count_tokens only (free); no messages are created")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        if m not in PRICE:
            sys.exit(f"no price row for {m}")
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    texts = {it["label"]: {"results": Path(it["results"]).read_text(encoding="utf-8"),
                           "prereg": Path(it["prereg"]).read_text(encoding="utf-8")} for it in plan}
    a.out.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(max_retries=3)

    # Pre-call estimate per (label, model): uncached input at cache-write price + EST_OUTPUT_TOKENS.
    est, dry = {}, []
    for it in plan:
        for m in models:
            p = build_params(m, **texts[it["label"]])
            n_in = client.messages.count_tokens(model=m, system=p["system"],
                                                messages=p["messages"]).input_tokens
            pin, pout, pw, _ = PRICE[m]
            est[(it["label"], m)] = (n_in * pw + EST_OUTPUT_TOKENS * pout) / 1e6
            dry.append({"label": it["label"], "model": m, "input_tokens": n_in,
                        "est_first_call_usd": round(est[(it["label"], m)], 4)})
    (a.out / "dryrun.json").write_text(json.dumps(dry, indent=1), encoding="utf-8")
    worst = sum(est[(it["label"], m)] for it in plan for m in models) * a.n
    print(json.dumps(dry, indent=1))
    print(f"upper estimate (no cache, {EST_OUTPUT_TOKENS} out tokens/call): ${worst:.2f}; cap ${a.budget_usd:.2f}")
    if a.dry_run:
        return 0

    budget = Budget(a.budget_usd, a.out / "ledger.jsonl")
    budget.log({"ts": now(), "event": "start", "models": models, "n": a.n, "cap_usd": a.budget_usd,
                "prompt_sha256": sha256(PROMPT.encode()), "schema_sha256": sha256(json.dumps(SCHEMA, sort_keys=True).encode()),
                "effort": EFFORT, "max_tokens": MAX_TOKENS, "thinking": "adaptive", "fallbacks": None,
                "anthropic_sdk": anthropic.__version__})

    def worker(m: str) -> None:
        # Within one model: labels in plan order, samples 1..n back to back, so samples
        # 2..n (and later versions sharing the PREREG prefix) read the cache sample 1 wrote.
        for it in plan:
            for k in range(1, a.n + 1):
                # One unexpected error must not end this model's remaining calls. If it struck
                # before parsed.json was written, the sample is scored as missing (00_PREREG.md §4).
                try:
                    run_one(client, budget, a.out, it, m, k, texts[it["label"]], est[(it["label"], m)])
                except Exception as e:
                    budget.log({"ts": now(), "event": "client_error", "label": it["label"], "model": m,
                                "sample": k, "error": f"{type(e).__name__}: {e}"[:300]})

    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        list(ex.map(worker, models))
    budget.log({"ts": now(), "event": "end", "spent_usd": round(budget.spent, 5)})
    print(f"spent ${budget.spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
