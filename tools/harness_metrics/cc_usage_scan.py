#!/usr/bin/env python3
"""Claude Code transcript usage scan: token mix, prompt-cache hit rate, cache breaks, per-agent-type cost, and the
1h-vs-5m cache-TTL counterfactual. Offline and read-only: it reads only `usage` rows of the local transcripts
(`~/.claude/projects/<project>/*.jsonl` and `<session>/subagents/**/*.jsonl` + `.meta.json`), never message text.

    python3 -m tools.harness_metrics.cc_usage_scan [--root DIR] [--days 14] [--json OUT]

Definitions (every number in the report follows these):
  request     one API call = one `requestId` (assistant rows repeat `usage` per content block; the last row wins).
  prompt      input_tokens + cache_creation_input_tokens + cache_read_input_tokens of one request.
  hit         sum(cache_read) / sum(prompt).
  break       a request whose cache_read < 0.5 * the previous request's prompt in the same file (previous prompt
              >= 20k). Cause, first match: model-switch | compact/shrink (prompt halved) | idle>1h | effort-change
              | other.
  cost        LIST-PRICE EQUIVALENT of Claude Code usage in USD (PRICE below, platform.claude.com pricing read
              2026-11-17). This is NOT API cost: Claude Code usage is excluded from cost accounting (user,
              2026-11-17); API cost lives in lab/38_api-batch ledgers. Use these numbers as a usage size.
  advisor     advisor-tool calls. Their tokens are recorded only in `usage.iterations` entries of type
              `advisor_message` (on the advisor model) and are NOT part of the request's top-level usage, so
              they are reported in their own columns and are not included in "$ actual" / "$ 5m (cf)". Advisor
              input carries no cache fields and is priced as uncached input.
  ttl 5m (cf) counterfactual cost had the file run on a 5-minute TTL: writes cost 1.25x input instead of 2x, but
              (a) a request more than 300 s after the previous one in its file re-writes its cached prefix, and
              (b) a spawn's FIRST request keeps its cache_read only if another spawn of the same
              (agent type, model) made a request in the 300 s before it (shared system+tools prefix); otherwise
              that read is re-written too. Gaps above 3600 s are breaks under both TTLs and are left as measured.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import glob
import json
import os
import statistics
import sys

# $/Mtok: input, output, cache read. 1h write = 2x input, 5m write = 1.25x input.
PRICE = {"claude-fable-5-2": (8.0, 40.0, 0.20), "claude-fable-5-1": (10.0, 50.0, 0.25),
         "claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.10),
         "claude-haiku-5-5": (0.10, 0.50, 0.01)}   # haiku: <=100k-prompt tier
W1H, W5M = 2.0, 1.25
BREAK_MIN_PROMPT, GAP_5M, GAP_1H = 20_000, 300.0, 3600.0


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def read_requests(path: str, since: float) -> list[dict]:
    """Deduplicated requests of one transcript, in file order, at or after `since` (epoch seconds)."""
    seen: dict[str, int] = {}
    out: list[dict] = []
    try:
        fh = open(path, errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            if '"usage"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "assistant" or not d.get("timestamp"):
                continue
            m = d.get("message") or {}
            u, rid = m.get("usage"), d.get("requestId") or m.get("id")
            if not u or not rid or m.get("model") in (None, "<synthetic>"):
                continue
            rec = {"t": _ts(d["timestamp"]), "model": m["model"], "effort": d.get("effort"),
                   "inp": u.get("input_tokens") or 0, "cw": u.get("cache_creation_input_tokens") or 0,
                   "cr": u.get("cache_read_input_tokens") or 0, "out": u.get("output_tokens") or 0,
                   "cw1h": (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0,
                   "adv": [(it.get("model"), (it.get("input_tokens") or 0) + (it.get("cache_read_input_tokens") or 0)
                            + (it.get("cache_creation_input_tokens") or 0), it.get("output_tokens") or 0)
                           for it in (u.get("iterations") or []) if it.get("type") == "advisor_message"]}
            if rid in seen:
                out[seen[rid]] = rec
            else:
                seen[rid] = len(out)
                out.append(rec)
    return [r for r in out if r["t"] >= since]


def prompt(r: dict) -> int:
    return r["inp"] + r["cw"] + r["cr"]


def cost_actual(r: dict) -> float:
    pin, pout, pcr = PRICE.get(r["model"], (0.0, 0.0, 0.0))
    w5 = r["cw"] - r["cw1h"]
    return (r["inp"] * pin + r["out"] * pout + r["cr"] * pcr + r["cw1h"] * W1H * pin + w5 * W5M * pin) / 1e6


def cost_advisor(r: dict) -> float:
    return sum((i * PRICE.get(m, (0.0, 0.0, 0.0))[0] + o * PRICE.get(m, (0.0, 0.0, 0.0))[1]) / 1e6
               for m, i, o in r.get("adv") or [])


def cost_5m(r: dict, reread_lost: bool) -> float:
    pin, pout, pcr = PRICE.get(r["model"], (0.0, 0.0, 0.0))
    if reread_lost:
        return (r["inp"] * pin + r["out"] * pout + (r["cw"] + r["cr"]) * W5M * pin) / 1e6
    return (r["inp"] * pin + r["out"] * pout + r["cr"] * pcr + r["cw"] * W5M * pin) / 1e6


def classify_break(prev: dict, r: dict) -> str | None:
    pp = prompt(prev)
    if pp < BREAK_MIN_PROMPT or r["cr"] >= 0.5 * pp:
        return None
    if r["model"] != prev["model"]:
        return "model-switch"
    if prompt(r) < 0.5 * pp:
        return "compact/shrink"
    if r["t"] - prev["t"] > GAP_1H:
        return "idle>1h"
    if r["effort"] != prev["effort"]:
        return "effort-change"
    return "other"


def summarize(files: dict[str, tuple[str, list[dict]]]) -> dict:
    """files: path -> (group, requests). Returns {group: stats}."""
    # index of request times per (group, model) for the first-request rule of the 5m counterfactual
    times: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for g, rs in files.values():
        for r in rs:
            times[(g, r["model"])].append(r["t"])
    for v in times.values():
        v.sort()
    groups: dict[str, dict] = {}
    for g, rs in files.values():
        s = groups.setdefault(g, {"spawns": 0, "reqs": 0, "inp": 0, "cw": 0, "cr": 0, "out": 0, "cost": 0.0,
                                  "cost_5m": 0.0, "gaps_5m": 0, "advisor_calls": 0, "advisor_cost": 0.0, "first": [], "peak": [],
                                  "nreq": [],
                                  "models": collections.Counter(), "cost_by_model": collections.Counter(),
                                  "breaks": collections.Counter(),
                                  "break_tokens": collections.Counter()})
        s["spawns"] += 1
        s["first"].append(prompt(rs[0]))
        s["peak"].append(max(prompt(r) for r in rs))
        s["nreq"].append(len(rs))
        prev = None
        for r in rs:
            s["reqs"] += 1
            for k in ("inp", "cw", "cr", "out"):
                s[k] += r[k]
            s["models"][f"{r['model'].replace('claude-', '')}@{r['effort']}"] += 1
            s["cost"] += cost_actual(r)
            s["advisor_calls"] += len(r.get("adv") or [])
            s["advisor_cost"] += cost_advisor(r)
            s["cost_by_model"][f"{r['model'].replace('claude-', '')}@{r['effort']}"] += cost_actual(r)
            if prev is None:
                tl = times[(g, r["model"])]
                i = bisect.bisect_left(tl, r["t"])
                warm = i > 0 and r["t"] - tl[i - 1] <= GAP_5M
                lost = r["cr"] > 0 and not warm
            else:
                lost = r["cr"] > 0 and GAP_5M < r["t"] - prev["t"] <= GAP_1H
                s["gaps_5m"] += int(lost)
                why = classify_break(prev, r)
                if why:
                    s["breaks"][why] += 1
                    s["break_tokens"][why] += r["cw"] + r["inp"]
            s["cost_5m"] += cost_5m(r, lost)
            prev = r
    return groups


def collect(root: str, days: float) -> tuple[dict, dict]:
    since = dt.datetime.now(dt.timezone.utc).timestamp() - days * 86400
    main, sub = {}, {}
    for p in glob.glob(os.path.join(root, "*.jsonl")):
        if os.path.getmtime(p) < since:
            continue
        rs = read_requests(p, since)
        if rs:
            main[p] = (collections.Counter(r["model"] for r in rs).most_common(1)[0][0].replace("claude-", ""), rs)
    for p in glob.glob(os.path.join(root, "*", "subagents", "**", "*.jsonl"), recursive=True):
        if os.path.getmtime(p) < since:
            continue
        rs = read_requests(p, since)
        if not rs:
            continue
        at = "?"
        try:
            with open(p[:-6] + ".meta.json") as fh:
                at = json.load(fh).get("agentType") or "?"
        except (OSError, ValueError):
            pass
        sub[p] = (at, rs)
    return summarize(main), summarize(sub)


def render(title: str, groups: dict) -> list[str]:
    out = [f"## {title}", "",
           "| group | spawns | req/spawn (med) | first prompt (med) | peak prompt (med) | write | read | out | hit % | "
           "$ actual | $ 5m (cf) | gaps>5m | advisor calls | $ advisor | top model@effort |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for g, s in sorted(groups.items(), key=lambda kv: -kv[1]["cost"]):
        den = s["inp"] + s["cw"] + s["cr"]
        ms = ", ".join(f"{k}:{v}" for k, v in s["models"].most_common(3))
        out.append(f"| {g} | {s['spawns']} | {statistics.median(s['nreq']):.0f} | "
                   f"{statistics.median(s['first']) / 1e3:.0f}k | {statistics.median(s['peak']) / 1e3:.0f}k | "
                   f"{s['cw'] / 1e6:.1f}M | {s['cr'] / 1e6:.0f}M | {s['out'] / 1e6:.2f}M | "
                   f"{100 * s['cr'] / max(den, 1):.1f} | {s['cost']:.0f} | {s['cost_5m']:.0f} | {s['gaps_5m']} | "
                   f"{s['advisor_calls']} | {s['advisor_cost']:.0f} | {ms} |")
        for k in ("cost", "cost_5m", "cw", "cr", "inp", "out", "advisor_calls", "advisor_cost"):
            tot[k] += s[k]
    den = tot["inp"] + tot["cw"] + tot["cr"]
    out += ["", f"total: $ actual {tot['cost']:.0f}, $ 5m (cf) {tot['cost_5m']:.0f}, "
                f"hit {100 * tot['cr'] / max(den, 1):.1f} %; advisor {tot['advisor_calls']:.0f} calls, "
                f"$ {tot['advisor_cost']:.0f} (not included in $ actual)", "",
                "cache breaks (count, re-written tokens):"]
    br, bt = collections.Counter(), collections.Counter()
    for s in groups.values():
        br.update(s["breaks"])
        bt.update(s["break_tokens"])
    out += [f"- {k}: {v}, {bt[k] / 1e6:.2f}M" for k, v in br.most_common()] or ["- none"]
    pairs = sorted(((c, g, m) for g, s in groups.items() for m, c in s["cost_by_model"].items()), reverse=True)
    out += ["", "$ actual by (group, model@effort), top 12:"]
    out += [f"- {g} / {m}: {c:.0f}" for c, g, m in pairs[:12]]
    return out + [""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects/-home-ayp-project-trm"))
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--json", help="also write the raw group stats here")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.root):
        print(f"no transcript dir: {a.root}", file=sys.stderr)
        return 2
    main_g, sub_g = collect(a.root, a.days)
    if not main_g and not sub_g:
        print("no requests in window (empty result is a failure, not a zero)", file=sys.stderr)
        return 1
    print(f"# Claude Code usage scan — last {a.days:g} days, root {a.root}")
    print("$ figures are list-price equivalents of Claude Code usage, not API cost; see the module docstring\n")
    print("\n".join(render("main sessions (grouped by dominant model)", main_g)))
    print("\n".join(render("subagents (grouped by agent type)", sub_g)))
    if a.json:
        def clean(g):
            return {k: {kk: (dict(vv) if isinstance(vv, collections.Counter) else vv) for kk, vv in s.items()
                        if kk not in ("first", "peak", "nreq")} for k, s in g.items()}
        with open(a.json, "w") as fh:
            json.dump({"days": a.days, "main": clean(main_g), "sub": clean(sub_g)}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
