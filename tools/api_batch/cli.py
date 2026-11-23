"""api-batch CLI (design §5.1). Run from the repo root:

    uv run --no-sync --with pyyaml python -m tools.api_batch.cli <command> ...

(never `python -I`: it drops the cwd from sys.path and sibling imports fail, design §8.1 / NB-10).

Offline commands (no network, no key): new, plan, approve, scan, selftest, collect --offline, cache-audit.
Metered commands (RealClient; run under `with-anthropic-key ...`): submit, status, collect, cancel,
probe-sync, draft-sync run. TypeSafe (run under a shell with TYPESAFE_API_KEY): jev run. LG AI Research (run under
tools/api_batch/bin/with-exaone-key): exaone run, exaone probe. `doclint-sync` hands the send to the frozen doc_lint.py.
`draft-sync plan`, `jev plan` and `exaone plan` are offline. An exaone plan is approved in tokens
(`approve <plan_dir> --tokens N`; no price is published), after reading its SEND_LIST.md.

`approve` is USER-ONLY. It refuses inside a Claude Code session. Marker (checked in this order, first
non-empty wins): CLAUDECODE, CLAUDE_CODE_SESSION_ID -- the same pair ~/.local/bin/anthropic-key-set:20 uses.
Observed set in the implementing session (CLAUDECODE=1, CLAUDE_CODE_SESSION_ID=<uuid>, CLAUDE_CODE_ENTRYPOINT=cli);
not verified for other Claude Code versions or headless/IDE modes, where a session that sets neither would pass the
guard. A plain terminal does not set them. Limit: it stops accidental self-approval; it cannot stop someone unsetting
the variable or editing the ledger (the stage verifier cross-checks `approved` rows against the user's message).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import draft, exaone, jev, ops
from .common import (DEFAULT_BATCH_ROOT, LAB_ROOT, REPO_ROOT, Halt, check_tools_path, in_claude_code_session)
from .deny import default_deny_files
from .ledger import Ledger
from .pack import PackBuilder

SKELETON_U3 = """name: {slug}
use: {use}
model: claude-fable-5-2
effort: high
max_tokens: 16000
cache_ttl: 5m
context_pack:
  sources:
    - {{path: 11_governance/11_lab-root/STATE.md, sections: ["0", "2"]}}
  cap_tokens: 60000
prompt:
  system_template: tools/api_batch/prompts/u3_discovery.md
  perspectives: [mechanism, falsifier]
  repeats: 1
output_schema: tools/api_batch/schemas/idea.json
refusal_fallback: null
budget:
  cap_usd_hard_worst: 20
"""


def _builder(a, batch_dir=None) -> PackBuilder:
    files = [Path(f) for f in a.deny_files] if a.deny_files else default_deny_files()
    return PackBuilder.from_files(files, batch_dir=batch_dir, lab_root=a.lab_root)


def _real_client():
    from .client import RealClient
    return RealClient()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="api-batch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    ap.add_argument("--lab-root", type=Path, default=LAB_ROOT, help="L1 root for lab/ sources (tests)")
    ap.add_argument("--ledger", type=Path, default=None, help="default: <batch-root>/ledger.jsonl")
    ap.add_argument("--deny-file", dest="deny_files", action="append", default=None, help="override the deny source file list (repeatable; tests)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    n = sp.add_parser("new"); n.add_argument("slug"); n.add_argument("--use", default="U3")
    for c in ("plan", "submit", "status", "cancel", "probe-sync"):
        p = sp.add_parser(c); p.add_argument("batch_dir", type=Path)
        if c == "submit":
            p.add_argument("--recover", action="store_true")
        if c == "probe-sync":
            p.add_argument("--ids", nargs="*")
    p = sp.add_parser("collect"); p.add_argument("batch_dir", type=Path)
    p.add_argument("--offline", action="store_true"); p.add_argument("--keep-server-copy", action="store_true")
    p = sp.add_parser("approve"); p.add_argument("batch_dir", type=Path, nargs="?"); p.add_argument("--usd", type=float, default=None)
    p.add_argument("--tokens", type=int, default=None, help="exaone plans only: token approval (price unpublished)")
    p.add_argument("--sync-plan-hash"); p.add_argument("--sync-worst", type=float)
    p.add_argument("--for-probe-sync", action="store_true", help="record 2x hard-worst (standard price)")
    p = sp.add_parser("scan"); p.add_argument("--verify-only", action="store_true", required=True)
    p.add_argument("paths", nargs="+", type=Path); p.add_argument("--kind", default="seat_input")
    p.add_argument("--lane", default=None, help="lane for allow_files exceptions (seat-doclint); default none = no exception")
    p = sp.add_parser("doclint-sync"); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--prereg", type=Path, required=True); p.add_argument("--p0a-verifier-ref", default="")
    p.add_argument("--plan-only", action="store_true")
    p = sp.add_parser("draft-sync", help="plan <spec> (offline) | run <plan_dir> (metered, standard price)")
    p.add_argument("action", choices=("plan", "run")); p.add_argument("target", type=Path)
    p = sp.add_parser("jev", help="TypeSafe Jev typed judgments: plan <spec> (offline) | run <plan_dir> (metered, third party)")
    p.add_argument("action", choices=("plan", "run")); p.add_argument("target", type=Path)
    p = sp.add_parser("exaone", help="K-EXAONE advisory reading: plan <spec> (offline) | show <plan_dir> (offline, "
                       "read-only) | run <plan_dir> (third party) | probe")
    p.add_argument("action", choices=("plan", "show", "run", "probe")); p.add_argument("target", type=Path, nargs="?")
    p.add_argument("--try-response-format", action="store_true", help="probe only: add the undocumented response_format")
    p.add_argument("--out", type=Path, default=None, help="probe only: where to save the response shape")
    sp.add_parser("selftest")
    p = sp.add_parser("cache-audit", help="offline prompt-cache audit of manifests (and collected usage, if present)")
    p.add_argument("paths", nargs="+", type=Path); p.add_argument("--json", action="store_true")
    for c in ("triage", "resubmit"):
        sp.add_parser(c, help="not implemented in P0a").add_argument("rest", nargs="*")
    return ap


def main(argv=None, client=None, runner=None) -> int:
    try:
        check_tools_path()
    except Halt as e:
        print(f"api-batch: {e}", file=sys.stderr)
        return 2
    a = build_parser().parse_args(argv)
    ledger = Ledger(a.ledger or (a.batch_root / "ledger.jsonl"))
    try:
        if a.cmd == "approve":
            if a.batch_dir and not a.sync_plan_hash and not in_claude_code_session() and exaone.is_exaone_plan(a.batch_dir):
                if a.usd is not None or a.for_probe_sync:
                    raise Halt("an exaone plan is approved in tokens (price unpublished): use --tokens N, not --usd")
                r = exaone.approve(a.batch_dir, a.tokens, ledger, builder=_builder(a, a.batch_dir))   # re-reads listed files
                print(f"approved {r['approval']['tokens']} tokens (price unpublished)\n"
                      f"SEND_LIST.md sha256 {r['send_list_sha256']}  plan_hash {r['manifest_sha256'][:12]}  "
                      "(must equal what `exaone show` printed)")
                print("listed files re-checked on disk (raw sha256 = `sha256sum <path>`; compare with `exaone show`):")
                for f in r["files"]:
                    print(f"  {f['raw_sha256']}  {f['path']}")
                return 0
            if a.tokens is not None and not in_claude_code_session():
                raise Halt(f"--tokens: {a.batch_dir} is not an exaone plan (no readable plan.json with use: exaone there; "
                           f"check the path, batch dirs live under {DEFAULT_BATCH_ROOT}/11_batches/). Other lanes take --usd.")
            if a.usd is None and not in_claude_code_session():
                raise Halt("approve needs --usd X (or --tokens N for an exaone plan)")
            sp = None
            if a.sync_plan_hash:
                if a.sync_worst is None:         # NB-3: a missing worst would record 0 and skip the limit check
                    raise Halt("approve --sync-plan-hash needs --sync-worst (the plan's hard_worst_usd from `doclint-sync --plan-only`)")
                sp = {"plan_hash": a.sync_plan_hash, "hard_worst_usd": a.sync_worst, "label": "doclint-sync"}
            elif a.batch_dir and not in_claude_code_session() and draft.is_draft_plan(a.batch_dir):
                # session guard first (in a session we fall through to ops.approve, which refuses before reading);
                # the draft-sync plan is then verified and its standard-price hard-worst used, never a typed figure.
                if a.for_probe_sync:
                    raise Halt("a draft-sync plan already carries the standard (2x) price; drop --for-probe-sync")
                sp = draft.sync_plan_for_approve(a.batch_dir, ledger)   # refuses an all-done plan
            elif a.batch_dir and not in_claude_code_session() and jev.is_jev_plan(a.batch_dir):
                if a.for_probe_sync:
                    raise Halt("a jev plan carries its own TypeSafe price; drop --for-probe-sync")
                sp = jev.sync_plan_for_approve(a.batch_dir, ledger)     # re-derived; refuses an all-done plan
            r = ops.approve(a.batch_dir, a.usd, ledger, sync_plan=sp, sync_factor=2.0 if a.for_probe_sync else 1.0)
            print(f"approved ${a.usd} for manifest {r['manifest_sha256'][:12]}")
        elif a.cmd == "selftest":
            rc = ops.selftest(ledger, runner)
            print("selftest rc=%d%s" % (rc, " (tests_passed row written)" if rc == 0 else ""))
            return rc
        elif a.cmd == "cache-audit":
            from . import cache_audit
            return cache_audit.main([*map(str, a.paths), *(["--json"] if a.json else [])])
        elif a.cmd == "new":
            bd = a.batch_root / "11_batches" / f"{__import__('datetime').date.today()}_{a.slug}"
            bd.mkdir(parents=True, exist_ok=False); (bd / "user").mkdir()
            (bd / "spec.yaml").write_text(SKELETON_U3.format(slug=a.slug, use=a.use), encoding="utf-8")
            print(bd)
        elif a.cmd == "plan":
            d = ops.plan(a.batch_dir, _builder(a, a.batch_dir), ledger)
            print(f"planned {d['n_requests']} requests; hard-worst ${d['est_usd_hard_worst']}; manifest {d['manifest_sha256'][:12]}")
        elif a.cmd == "scan":
            rc = ops.scan_verify_only(a.paths, _builder(a), ledger, a.kind, a.lane)
            print("scan rc=%d" % rc)
            return rc
        elif a.cmd == "doclint-sync":
            r = ops.doclint_sync(a.results, a.prereg, _builder(a), ledger, p0a_verifier_ref=a.p0a_verifier_ref,
                                 plan_only=a.plan_only, runner=runner)
            print(r)
        elif a.cmd == "draft-sync" and a.action == "plan":
            d = draft.plan(a.target, _builder(a, a.target.parent), ledger)
            print(f"draft-sync planned {d['n_requests']} sections; standard-price hard-worst ${d['est_usd_hard_worst']}; "
                  f"plan_hash {d['plan_hash'][:12]}\nuser, in your own terminal: ... -m tools.api_batch.cli {d['approve_cmd']}")
        elif a.cmd == "draft-sync":
            d, rows, _w = draft.load_plan(a.target)
            if not draft.pending_sections(a.target, ledger, d["plan_hash"], rows):
                print(f"draft-sync: nothing to do -- every section of {a.target} is already done (no approval used)")
                return 0                      # before _real_client(): no SDK import, no key, no ledger row
            r = draft.run(a.target, ledger, client or _real_client(), _builder(a, a.target))
            print(r)
            return 0 if r["halted"] is None and all(s["rc"] == 0 for s in r["sections"]) else 5
        elif a.cmd == "jev" and a.action == "plan":
            d = jev.plan(a.target, _builder(a, a.target.parent), ledger)
            print(f"jev planned {d['n_requests']} requests x {d['n_questions']} questions; hard-worst ${d['est_usd_hard_worst']}; "
                  f"plan_hash {d['plan_hash'][:12]}\nuser, in your own terminal: ... -m tools.api_batch.cli {d['approve_cmd']}")
        elif a.cmd == "jev":
            d, rows, _w = jev.load_plan(a.target)
            if not [r for r in rows if r["custom_id"] not in jev._done_ids(ledger, d["plan_hash"])]:
                print(f"jev: nothing to do -- every request of {a.target} is already done (no approval used)")
                return 0                      # before RealJevClient(): no SDK import, no key, no ledger row
            r = jev.run(a.target, ledger, client or jev.RealJevClient(), _builder(a, a.target))
            print(r)
            return 0 if r["halted"] is None and all(x["rc"] == 0 for x in r["requests"]) else 5
        elif a.cmd == "exaone" and a.action == "probe":
            marker = in_claude_code_session()
            if marker:                        # before constructing the client: only the user triggers a live call
                raise Halt(f"exaone probe refused: running inside a Claude Code session ({marker} is set)")
            out = a.out or (a.batch_root / "13_exaone-probe" / f"probe-{__import__('time').strftime('%Y%m%dT%H%M%SZ', __import__('time').gmtime())}.json")
            s = exaone.probe(ledger, client or exaone.RealExaoneClient(), out, try_response_format=a.try_response_format)
            print(__import__("json").dumps(s, ensure_ascii=False, indent=1, sort_keys=True))
            print(f"saved: {out}")
            return 0 if s.get("rc") == 0 else 5
        elif a.cmd == "exaone" and a.target is None:
            raise Halt(f"exaone {a.action} needs a target ({'spec.yaml' if a.action == 'plan' else 'plan dir'})")
        elif a.cmd == "exaone" and a.action == "show":   # offline, read-only: no ledger row, no key, any session
            try:
                text = exaone.show(a.target, _builder(a, a.target))
            except (OSError, ValueError) as e:
                raise Halt(f"exaone show: cannot read plan {a.target} ({type(e).__name__})")
            try:
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
            except BrokenPipeError:            # `show | less` quit early: nothing to report
                try:
                    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
                except (OSError, ValueError):
                    pass
            return 0
        elif a.cmd == "exaone" and a.action == "plan":
            if not a.target.is_file():
                raise Halt(f"exaone plan: spec {a.target} not found (paths are relative to the current directory; "
                           f"batch dirs live under {DEFAULT_BATCH_ROOT}/11_batches/)")
            d = exaone.plan(a.target, _builder(a, a.target.parent), ledger)
            print(f"exaone planned {d['n_requests']} / {exaone.DAY_REQUEST_LIMIT_OBSERVED} requests (observed daily "
                  f"limit, {exaone.DAY_LIMIT_SOURCE}); hard-worst {d['hard_worst_tokens']} tokens "
                  f"(cap {d['max_total_tokens']}; price unpublished); plan_hash {d['plan_hash'][:12]}\n"
                  f"read first: ... -m tools.api_batch.cli exaone show {a.target.parent} | less\n"
                  f"user, in your own terminal: ... -m tools.api_batch.cli {d['approve_cmd']}")
        elif a.cmd == "exaone":
            try:
                _d, pend = exaone.pending_rows(a.target, ledger)
            except (OSError, ValueError) as e:
                raise Halt(f"exaone run: cannot read plan {a.target} ({type(e).__name__}); paths are relative to the "
                           f"current directory, batch dirs live under {DEFAULT_BATCH_ROOT}/11_batches/")
            if not pend:
                print(f"exaone: nothing to do -- every request of {a.target} is already done (no approval used)")
                return 0                      # before RealExaoneClient(): no key read, no ledger row
            r = exaone.run(a.target, ledger, client or exaone.RealExaoneClient(), _builder(a, a.target))
            print(r)
            return 0 if r["halted"] is None and all(x["rc"] == 0 for x in r["requests"]) else 5
        elif a.cmd in ("submit", "status", "cancel", "collect", "probe-sync"):
            cl = client or (None if (a.cmd == "collect" and a.offline) else _real_client())
            if a.cmd == "submit":
                print(ops.submit(a.batch_dir, ledger, cl, recover=a.recover)["batch_id"])
            elif a.cmd == "status":
                print(ops.status(a.batch_dir, ledger, cl))
            elif a.cmd == "cancel":
                print(ops.cancel(a.batch_dir, ledger, cl))
            elif a.cmd == "probe-sync":
                print(len(ops.probe_sync(a.batch_dir, ledger, cl, a.ids)), "sent")
            else:
                print(ops.collect(a.batch_dir, ledger, cl, a.batch_root, _builder(a, a.batch_dir),
                                  offline=a.offline, delete=not a.keep_server_copy))
        else:
            print(f"{a.cmd}: not implemented in P0a", file=sys.stderr)
            return 4
    except Halt as e:
        print(f"api-batch: HALT: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
