"""Offline prompt-cache audit of api-batch manifests (no network, no key, no deny scan needed: it reads only
files already written by `plan` / `collect` / `draft-sync run`).

For each manifest it renders every request into its cache-relevant block sequence (system blocks, then the
user-message content blocks, in API render order) and simulates the prompt cache in manifest order:

  ideal      -- the CURRENT breakpoints, requests served one after another (a breakpoint whose prefix an
                earlier request already wrote is a read; breakpoints below the model minimum never cache).
  potential  -- the same order of blocks but a breakpoint at the end of every sharing level (the deepest
                boundary still shared by the same >=2 requests of the group): upper bound for any placement that
                keeps the block order. The number of breakpoints this needs per request is reported
                (max_breakpoints_needed; F7 when above the API's 4).

The gap potential - ideal is the cache the current breakpoint placement leaves on the table. Requests are
grouped by (model, thinking, effort, output format): caches are model-scoped and thinking/effort/format
changes invalidate the messages cache, so requests in different groups never share an entry.

Findings (codes used in the report):
  F1 shared-prefix tokens per breakpoint                 F2 breakpoint below the model minimum
  F3 shared blocks after the last effective breakpoint,  F4 image block ahead of a block shared across requests
     plus blocks shared across requests off-prefix       F5 thinking/effort/format differ between requests
                                                            that share their first block (same model)
  F6 measured writes vs ideal writes (concurrency duplicates), from collected usage when present.
  F7 potential needs more than 4 breakpoints in some request  F8 a request carries more than 4 breakpoints
Not modelled: tool definitions (no lane sends tools) and the API's ~20-block lookback per breakpoint (no lane has
more than ~10 content blocks before its last breakpoint).

Token counts are common.est_tokens (bytes/1.5, deliberately an over-estimate); the model-minimum test uses a
bytes/4 under-estimate instead, so a prefix near the threshold is treated as "won't cache". When usage is present the
report also gives the calibration factor measured_input / estimated_input and the measured coverage
read / (read + write + input) -- unlike ledger `r_hit_measured` (read / (read + write)), uncached input
counts against it.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .common import BYTES_PER_TOKEN, canon_json, est_tokens, sha256_text
from .pack import image_tokens_b64

# Minimum cacheable prefix (tokens). claude-api skill shared/prompt-caching.md (cached 2026-10-06): 512 on every
# current 5.x model. Unknown models get the largest documented minimum (fail toward "won't cache").
MIN_CACHE_TOKENS = {"claude-opus-5-5": 512, "claude-sonnet-5-5": 512, "claude-fable-5-1": 512,
                   "claude-fable-5-2": 512, "claude-haiku-5-5": 512}   # fable-5-2: prompt-caching docs 2026-11-10
MIN_CACHE_UNKNOWN = 4096
MAX_BREAKPOINTS = 4
F6_WRITE_RATIO = 2.0
LOW_BYTES_PER_TOKEN = 4.0    # text under-estimate for the minimum test: near-threshold prefixes count as "won't cache"


@dataclass
class Block:
    kind: str
    sha: str
    tokens: int
    cc: dict | None
    title: str | None = None
    tokens_low: int = 0          # under-estimate (bytes/4) used only for the model-minimum test


@dataclass
class Req:
    custom_id: str
    model: str
    group: str
    settings: dict
    blocks: list
    cum: list = field(default_factory=list)      # cum[i] = tokens of blocks[0..i] (est_tokens, over-estimate)
    cum_low: list = field(default_factory=list)  # same with tokens_low: a breakpoint caches only if cum_low >= minimum

    @property
    def total(self) -> int:
        return self.cum[-1] if self.cum else 0


def _block(b: dict, kind_hint: str | None = None) -> Block:
    t = kind_hint or b.get("type")
    cc = b.get("cache_control")
    body = {k: v for k, v in b.items() if k != "cache_control"}
    if t == "image":
        data = (b.get("source") or {}).get("data", "")
        try:
            tok = image_tokens_b64(data)
        except Exception:  # noqa: BLE001 -- unparsable image: count it as text bytes rather than fail the audit
            tok = est_tokens(data)
    elif t == "document":
        tok = est_tokens((b.get("source") or {}).get("data", "")) + est_tokens(b.get("title") or "")
    else:
        tok = est_tokens(b.get("text", ""))
    low = tok if t == "image" else -int(-tok * BYTES_PER_TOKEN // LOW_BYTES_PER_TOKEN)
    return Block(t, sha256_text(canon_json(body)), tok, cc, b.get("title"), low)


def render(row: dict) -> Req:
    p = row["params"]
    blocks: list[Block] = []
    sysv = p.get("system")
    if isinstance(sysv, str):
        blocks.append(_block({"type": "text", "text": sysv}, "system"))
    elif isinstance(sysv, list):
        blocks += [_block(b, "system") for b in sysv]
    for m in p.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            blocks.append(_block({"type": "text", "text": c}))
        else:
            blocks += [_block(b) for b in c or []]
    oc = p.get("output_config") or {}
    settings = {"thinking": p.get("thinking"), "effort": oc.get("effort"),
                "format_sha": sha256_text(canon_json(oc.get("format")))[:12] if oc.get("format") else None}
    model = p["model"]
    req = Req(row["custom_id"], model, f"{model}|{canon_json(settings)}", settings, blocks)
    s = lo = 0
    for b in blocks:
        s += b.tokens
        lo += b.tokens_low
        req.cum.append(s)
        req.cum_low.append(lo)
    return req


def _prefix_keys(r: Req) -> list[str]:
    keys, acc = [], ""
    for b in r.blocks:
        acc = sha256_text(acc + b.sha)
        keys.append(acc)
    return keys


def simulate(reqs: list[Req], every_boundary: bool = False, per_request: list | None = None) -> dict:
    """Sequential cache simulation. Returns token totals {read, write, input, total}; when `per_request` is a
    list, (req, read, write, input) is appended for each request."""
    written: dict[str, set] = {}
    tot = {"read": 0, "write": 0, "input": 0, "total": 0}
    shared: dict[str, int] = {}
    if every_boundary:          # candidate boundaries = prefixes that at least two requests of the group share
        for r in reqs:
            for k in _prefix_keys(r):
                shared[r.group + k] = shared.get(r.group + k, 0) + 1
    tot["max_breakpoints_needed"] = 0
    for r in reqs:
        mn = MIN_CACHE_TOKENS.get(r.model, MIN_CACHE_UNKNOWN)
        keys = _prefix_keys(r)
        if every_boundary:
            # only the END of each sharing level is a candidate (the deepest boundary that the same set of
            # requests still shares); a boundary inside a level adds nothing a deeper one in it doesn't give
            cnt = [shared.get(r.group + k, 0) for k in keys]
            bps = [i for i in range(len(keys)) if cnt[i] >= 2 and (i + 1 == len(keys) or cnt[i + 1] < cnt[i])
                   and r.cum_low[i] >= mn]
            tot["max_breakpoints_needed"] = max(tot["max_breakpoints_needed"], len(bps))
        else:
            bps = [i for i, b in enumerate(r.blocks) if b.cc and r.cum_low[i] >= mn]
        seen = written.setdefault(r.group, set())
        hit = max((i for i in bps if keys[i] in seen), default=None)
        read = r.cum[hit] if hit is not None else 0
        last = bps[-1] if bps else None
        write = (r.cum[last] - read) if last is not None and (hit is None or last > hit) else 0
        tot["read"] += read
        tot["write"] += write
        tot["input"] += r.total - read - write
        tot["total"] += r.total
        if per_request is not None:
            per_request.append((r, read, write, r.total - read - write))
        seen.update(keys[i] for i in bps)
    return tot


def _usd(rows: list, scale: float) -> float:
    """Input-side cost of simulated requests at the batch price table (draft-sync custom_ids at 2x standard),
    5m write price; output tokens excluded (identical in both simulations)."""
    from . import prices
    usd = 0.0
    for r, rd, wr, inp in rows:
        pin, _pout, pcr, cw5, _cw1 = prices.price(r.model, r.total)
        f = 2.0 if r.custom_id.startswith("draft-") else 1.0
        usd += f * scale * (rd * pcr + wr * cw5 + inp * pin) / 1e6
    return usd


def _cov(t: dict) -> float | None:
    return round(t["read"] / t["total"], 4) if t["total"] else None


def findings(reqs: list[Req]) -> list[dict]:
    out: list[dict] = []
    by_group: dict[str, list[Req]] = {}
    for r in reqs:
        by_group.setdefault(r.group, []).append(r)
    for g, rs in by_group.items():
        keys = {r.custom_id: _prefix_keys(r) for r in rs}
        count: dict[str, int] = {}
        for ks in keys.values():
            for k in ks:
                count[k] = count.get(k, 0) + 1
        sha_count: dict[str, int] = {}
        for r in rs:
            for s in {b.sha for b in r.blocks}:
                sha_count[s] = sha_count.get(s, 0) + 1
        mn = MIN_CACHE_TOKENS.get(rs[0].model, MIN_CACHE_UNKNOWN)
        f1: dict[int, dict] = {}
        f3_tok = f3_off = 0
        f3_pos: set = set()
        for r in rs:
            ks = keys[r.custom_id]
            ncc = sum(1 for b in r.blocks if b.cc)
            if ncc > MAX_BREAKPOINTS:
                out.append({"code": "F8", "group": g, "custom_id": r.custom_id, "msg": f"{ncc} breakpoints > {MAX_BREAKPOINTS} (API 400)"})
            # shared prefix end, counted only where a breakpoint could actually cache (prefix >= model minimum)
            shared_end = max((i for i, k in enumerate(ks) if count[k] >= 2 and r.cum_low[i] >= mn), default=None)
            eff = [i for i, b in enumerate(r.blocks) if b.cc and r.cum_low[i] >= mn]
            for i, b in enumerate(r.blocks):
                if b.cc:
                    e = f1.setdefault(i, {"kind": b.kind, "title": b.title, "prefix_tokens": r.cum[i], "n_sharing": 0})
                    e["n_sharing"] = max(e["n_sharing"], count[ks[i]])
                    if r.cum_low[i] < mn:
                        out.append({"code": "F2", "group": g, "custom_id": r.custom_id, "pos": i,
                                    "msg": f"breakpoint at block {i} ({b.kind}): prefix may be below min {mn} "
                                           f"(est {r.cum_low[i]}-{r.cum[i]} tok)"})
            last_eff = eff[-1] if eff else -1
            if shared_end is not None and shared_end > last_eff:
                f3_tok += r.cum[shared_end] - (r.cum[last_eff] if last_eff >= 0 else 0)
                f3_pos.add((shared_end, r.blocks[shared_end].kind))
            off = [b for i, b in enumerate(r.blocks) if (shared_end is None or i > shared_end) and sha_count[b.sha] >= 2]
            f3_off += sum(b.tokens for b in off)
            for i, b in enumerate(r.blocks):
                if b.kind == "image" and sha_count[b.sha] < 2 and any(sha_count[x.sha] >= 2 for x in r.blocks[i + 1:]):
                    out.append({"code": "F4", "group": g, "custom_id": r.custom_id, "pos": i,
                                "msg": "an image used by this request alone precedes a block shared with other requests (move shared blocks ahead of it)"})
                    break
        for i, e in sorted(f1.items()):
            out.append({"code": "F1", "group": g, "pos": i, **e})
        if f3_tok:
            out.append({"code": "F3", "group": g, "missed_prefix_tokens": f3_tok,
                        "shared_until": sorted(f3_pos)[:5],
                        "msg": "requests share blocks past their last effective breakpoint"})
        if f3_off:
            out.append({"code": "F3", "group": g, "off_prefix_shared_tokens": f3_off,
                        "msg": "blocks repeated across requests but not on the shared prefix (reachable only by reordering)"})
    # F5: same model, same first block, different settings
    first: dict[tuple, set] = {}
    for r in reqs:
        if r.blocks:
            first.setdefault((r.model, r.blocks[0].sha), set()).add(canon_json(r.settings))
    for (m, _), ss in first.items():
        if len(ss) > 1:
            out.append({"code": "F5", "model": m, "settings": sorted(ss),
                        "msg": "requests share their first block but differ in thinking/effort/format -> separate caches"})
    return out


# ----------------------------------------------------------------------------------- measured usage
def load_usage(batch_dir: Path, ids: set) -> dict | None:
    """Collected usage keyed by custom_id: batch raw/results.jsonl, or draft-sync raw/<section>.json."""
    bd = Path(batch_dir)
    out: dict[str, dict] = {}
    rp = bd / "raw" / "results.jsonl"
    if rp.is_file():
        for ln in rp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                msg = (rec.get("result") or {}).get("message") or {}
                if msg.get("usage") is not None:
                    out[rec["custom_id"]] = msg["usage"]
    else:
        for cid in ids:
            if cid.startswith("draft-"):
                f = bd / "raw" / f"{cid[len('draft-'):]}.json"
                if f.is_file():
                    u = json.loads(f.read_text(encoding="utf-8")).get("usage")
                    if u is not None:
                        out[cid] = u
    return out or None


def audit_manifest(manifest: Path) -> dict:
    manifest = Path(manifest)
    rows = [json.loads(ln) for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    reqs = [render(r) for r in rows]
    pr_i, pr_p = [], []
    ideal, pot = simulate(reqs, per_request=pr_i), simulate(reqs, every_boundary=True, per_request=pr_p)
    ideal.pop("max_breakpoints_needed", None)
    res = {"batch": manifest.parent.name, "n_requests": len(reqs),
           "models": sorted({r.model for r in reqs}), "n_groups": len({r.group for r in reqs}),
           "est_input_tokens": ideal["total"],
           "ideal": {**ideal, "coverage": _cov(ideal)}, "potential": {**pot, "coverage": _cov(pot)},
           "findings": findings(reqs), "measured": None}
    usage = load_usage(manifest.parent, {r.custom_id for r in reqs})
    m: dict = {}
    if usage:
        g = lambda u, k: int(u.get(k) or 0)  # noqa: E731
        m = {"read": sum(g(u, "cache_read_input_tokens") for u in usage.values()),
             "write": sum(g(u, "cache_creation_input_tokens") for u in usage.values()),
             "input": sum(g(u, "input_tokens") for u in usage.values()), "n": len(usage)}
        tot = m["read"] + m["write"] + m["input"]
        est_cov = sum(r.total for r in reqs if r.custom_id in usage)
        m["coverage"] = round(m["read"] / tot, 4) if tot else None
        m["calibration"] = round(tot / est_cov, 4) if est_cov else None
        m["r_hit_ledger_def"] = round(m["read"] / (m["read"] + m["write"]), 4) if m["read"] + m["write"] else None
        if m["calibration"]:
            sub = [r for r in reqs if r.custom_id in usage]
            si = simulate(sub)
            ideal_read = si["read"] * m["calibration"]
            ideal_write = si["write"] * m["calibration"]
            m["miss_tokens_vs_ideal"] = round(ideal_read - m["read"])
            m["write_over_ideal"] = round(m["write"] / ideal_write, 3) if ideal_write else None
        res["measured"] = m
    scale = (res["measured"] or {}).get("calibration") or 1.0
    try:
        ui, up = _usd(pr_i, scale), _usd(pr_p, scale)
        res["input_usd"] = {"ideal": round(ui, 4), "potential": round(up, 4), "gap": round(ui - up, 4),
                            "basis": f"batch table (draft-* 2x), 5m writes, tokens x calibration {scale}"}
    except prices_unknown() as e:     # unknown model: no dollar figure, never a guessed one
        res["input_usd"] = {"error": str(e)}
    if (m.get("write_over_ideal") or 0) > F6_WRITE_RATIO:
        res["findings"].append({"code": "F6", "write_over_ideal": m["write_over_ideal"],
                                "msg": f"measured cache writes exceed the sequential ideal >{F6_WRITE_RATIO}x (concurrent starts); "
                                       "consider ttl 1h or a pre-warm"})
    if res["potential"]["max_breakpoints_needed"] > MAX_BREAKPOINTS:
        res["findings"].append({"code": "F7", "max_breakpoints_needed": res["potential"]["max_breakpoints_needed"],
                                "msg": f"potential needs more than {MAX_BREAKPOINTS} breakpoints in some request; "
                                       "it is an upper bound, not reachable as is"})
    return res


def prices_unknown():
    from .prices import UnknownModel
    return UnknownModel


def resolve(paths: list[Path]) -> list[Path]:
    """A directory without manifest.jsonl is skipped with a note (e.g. a doclint-sync dir); an explicit file
    path that does not exist is an error."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            m = p / "manifest.jsonl"
            if m.is_file():
                out.append(m)
            else:
                print(f"cache-audit: skip {p} (no manifest.jsonl)", file=sys.stderr)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"no manifest at {p}")
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def render_markdown(results: list[dict]) -> str:
    lines = ["| batch | n | models | est input tok | ideal cov (current bp) | potential cov (any bp, same order) | gap | input $ gap | measured cov | ledger r_hit | calib | findings |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        m = r["measured"] or {}
        codes = sorted({f["code"] for f in r["findings"] if f["code"] != "F1"})
        gap = (r["potential"]["coverage"] or 0) - (r["ideal"]["coverage"] or 0)
        lines.append(f"| {r['batch']} | {r['n_requests']} | {', '.join(x.replace('claude-', '') for x in r['models'])} | "
                     f"{r['est_input_tokens']:,} | {_pct(r['ideal']['coverage'])} | {_pct(r['potential']['coverage'])} | "
                     f"{_pct(gap)} | {('$%.2f' % r['input_usd']['gap']) if 'gap' in r.get('input_usd', {}) else '-'} | {_pct(m.get('coverage'))} | {_pct(m.get('r_hit_ledger_def'))} | "
                     f"{m.get('calibration') or '-'} | {' '.join(codes) or '-'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cache-audit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="batch dirs or manifest.jsonl files")
    ap.add_argument("--json", action="store_true", help="full JSON (findings included) instead of the table")
    a = ap.parse_args(argv)
    try:
        res = [audit_manifest(p) for p in resolve(a.paths)]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"cache-audit: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else render_markdown(res), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
