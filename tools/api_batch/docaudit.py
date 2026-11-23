"""docaudit lane: a tool-less READER seat for documentation audits (2026-11-11; first use:
lab/24_audits/11_lab-audits/2026-11-11_full-doc-audit).

One request = one shard of lab markdown files (or line-chunks of an oversize file) behind a cached
[system, header] prefix; the model returns candidate findings as JSON (schema docaudit_findings).

What this lane is, and is not:
  * A seat, never a verifier (G1/G2). A batch request has no tools: the reader sees only the text in its own
    request, so every finding is a CANDIDATE. `basis: internal` means both sides are quoted from the supplied text;
    `basis: needs-check` means something outside the request (filesystem, another document, the queue) decides it.
    Nothing is edited from this output without a tool-equipped check.
  * Fail-closed per file. Every file passes the pipeline's verify-only scan (L1 allow-list, L2 path/frontmatter,
    L3b body, L4 secrets) on its RAW text before any piece of it is packed, and the text that is actually sent
    (marker lines + body, or a line-chunk) is scanned again. A file that fails either scan is NOT sent and NOT
    masked: it is listed in plan.json -> docaudit.skipped with the layer and reason. The same file reached twice
    (an alias, './x') is sent once. (Other lanes refuse the whole
    plan on one hit; a corpus-wide read would then never plan, so here the unit of refusal is the file.)
  * L1 limits the lane to lab/**/*.md|txt whose name does not contain DECISIONS. Repo docs, memory files and
    .claude/ cannot be sent; they stay with tool-equipped readers.

Spec (use: docaudit):
  name, use, models: [1..2 distinct ids from prices.py], effort, max_tokens, cache_ttl: 5m|1h,
  header: shared/<file>.md, prompt: {system_template: tools/api_batch/prompts/<file>.md},
  output_schema: <name under schemas/>, budget: {cap_usd_hard_worst},
  files: {list: shared/<file>.txt}            # one lab-relative path per line, '#' comments
       | {roots: [<lab-relative dir>...], exclude?: [<fnmatch on the lab-relative path>...]}
  shard?: {target_tokens: 40000, max_file_tokens: 120000, chunk_tokens: 100000}   # est_tokens units (bytes/1.5)

Post-check (offline, no network): `python -m tools.api_batch.docaudit report <batch_dir>` after `collect`.
It resolves every finding's `quote` to a line of the CURRENT file (exact substring, then whitespace-normalised)
and writes docaudit/findings.jsonl + docaudit/REPORT.md. A quote that is not in the file marks the finding
`quote_found: false` (a fabricated or paraphrased quote); such rows are kept and counted, never dropped silently.
`counter_quote` is resolved the same way (`counter_found`), an item that is not an object is kept as
`bad_finding`, and a request whose `files_read` does not reach the last line of a part it was sent is listed in
summary.json (`parts_not_read_to_end`). Limits: locate() returns the FIRST occurrence in the file, and the L1 rule
on DECISIONS is a file-NAME rule (pack.py), so a `decisions/` directory is not excluded by it.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

from . import manifest as M
from .common import Halt, est_tokens, sha256_bytes, sha256_text
from .prices import PRICES

LANE = "docaudit"
KEYS = ("name", "use", "models", "effort", "max_tokens", "cache_ttl", "header", "prompt", "output_schema", "files", "budget")
OPTIONAL_KEYS = ("shard",)
SHARD_DEFAULT = {"target_tokens": 40000, "max_file_tokens": 120000, "chunk_tokens": 100000}
# never walked by `files.roots`: vendored packages, generated cards, this pipeline's own outputs, backups, archives
ROOT_SKIP_DIRS = (".lake", "12_run-cards", "38_api-batch", "99_Archive", "before", "__pycache__", "node_modules")
ROOT_SKIP_RE = re.compile(r"(^|/)(backup[^/]*|[^/]*_backups|\.locks)(/|$)")
CATEGORIES = ("wrong", "inconsistent", "inefficient", "obsolete")
BASES = ("internal", "needs-check")


class DocAuditError(Halt):
    pass


# ------------------------------------------------------------------------------------------------ spec
def validate(spec: dict) -> dict:
    from .spec import EFFORTS, SpecError
    extra = set(spec) - set(KEYS) - set(OPTIONAL_KEYS)
    if extra:
        raise SpecError(f"docaudit: unknown spec keys {sorted(extra)}")
    ms = spec["models"]
    if not isinstance(ms, list) or not 1 <= len(ms) <= 2 or len(set(ms)) != len(ms) \
            or not all(isinstance(m, str) and m in PRICES for m in ms):
        raise SpecError("docaudit models must be a list of 1..2 distinct models from prices.py")
    if spec["effort"] not in EFFORTS:
        raise SpecError("effort must be explicit (low|medium|high|xhigh|max)")
    if not isinstance(spec["max_tokens"], int) or isinstance(spec["max_tokens"], bool) or spec["max_tokens"] <= 0:
        raise SpecError("max_tokens must be a positive integer")
    if spec["cache_ttl"] not in ("5m", "1h"):
        raise SpecError("docaudit needs a cached shared header: cache_ttl 5m | 1h")
    if not isinstance(spec["header"], str) or not re.fullmatch(r"shared/[^/]+\.md", spec["header"]):
        raise SpecError("docaudit header must be a batch-relative shared/<file>.md")
    pr = spec["prompt"]
    if not isinstance(pr, dict) or set(pr) != {"system_template"} \
            or not re.fullmatch(r"tools/api_batch/prompts/[^/]+\.md", str(pr["system_template"])):
        raise SpecError("docaudit prompt must be exactly {system_template: tools/api_batch/prompts/<file>.md}")
    if not isinstance(spec["output_schema"], str) or not spec["output_schema"] or "/" in spec["output_schema"]:
        raise SpecError("docaudit output_schema must be a name under tools/api_batch/schemas/")
    fs = spec["files"]
    if not isinstance(fs, dict) or (("list" in fs) == ("roots" in fs)) or set(fs) - {"list", "roots", "exclude"}:
        raise SpecError("docaudit files must be {list: shared/<file>.txt} or {roots: [...], exclude?: [...]}")
    if "list" in fs:
        if "exclude" in fs or not isinstance(fs["list"], str) or not re.fullmatch(r"shared/[^/]+\.txt", fs["list"]):
            raise SpecError("docaudit files.list must be a batch-relative shared/<file>.txt (no exclude)")
    else:
        rs = fs["roots"]
        if not isinstance(rs, list) or not rs or not all(_is_rel(r) for r in rs):
            raise SpecError("docaudit files.roots must be a non-empty list of lab-relative directories")
        ex = fs.get("exclude", [])
        if not isinstance(ex, list) or not all(isinstance(e, str) and e for e in ex):
            raise SpecError("docaudit files.exclude must be a list of fnmatch patterns")
    sh = spec.get("shard", {})
    if not isinstance(sh, dict) or set(sh) - set(SHARD_DEFAULT) \
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in sh.values()):
        raise SpecError(f"docaudit shard keys must be positive integers among {sorted(SHARD_DEFAULT)}")
    p = {**SHARD_DEFAULT, **sh}
    if p["chunk_tokens"] > p["max_file_tokens"]:
        raise SpecError("docaudit shard.chunk_tokens must be <= shard.max_file_tokens")
    return spec


def _is_rel(r) -> bool:
    return isinstance(r, str) and bool(r) and not r.startswith("/") and ".." not in r.split("/")


# ------------------------------------------------------------------------------------------- selection
def read_list(path: Path) -> list[str]:
    """Lab-relative paths, one per line; blank lines and '#' comments ignored; duplicates and absolute/.. paths halt."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        if not _is_rel(ln):
            raise DocAuditError(f"files list: {ln!r} is not a lab-relative path")
        out.append(ln)
    if not out:
        raise DocAuditError("files list is empty")
    if len(set(out)) != len(out):
        raise DocAuditError("files list has duplicate paths")
    return out


def walk_roots(lab_root: Path, roots: list[str], exclude: list[str]) -> list[str]:
    """Sorted lab-relative *.md under the roots. Symlinked directories are not followed; ROOT_SKIP_* never entered."""
    lab = Path(lab_root).resolve()
    seen = set()
    for r in roots:
        base = lab / r
        try:
            base.resolve().relative_to(lab)
        except ValueError:
            raise DocAuditError(f"files.roots: {r!r} resolves outside lab/") from None
        if not base.is_dir():
            raise DocAuditError(f"files.roots: {r!r} is not a directory under lab/")
        for dp, dns, fns in os.walk(base, followlinks=False):
            rel_dir = Path(dp).resolve().relative_to(lab).as_posix()
            dns[:] = sorted(d for d in dns if d not in ROOT_SKIP_DIRS and not ROOT_SKIP_RE.search(f"{rel_dir}/{d}")
                            and not os.path.islink(os.path.join(dp, d)))
            for fn in fns:
                if not fn.endswith(".md"):
                    continue
                rel = f"{rel_dir}/{fn}" if rel_dir != "." else fn
                if any(fnmatch.fnmatch(rel, e) for e in exclude):
                    continue
                seen.add(rel)
    if not seen:
        raise DocAuditError("files.roots matched no markdown file")
    return sorted(seen)


# --------------------------------------------------------------------------------------------- packing
def wrap(rel: str, a: int, b: int, n_lines: int, body: str) -> str:
    """The text of one sent piece. The marker lines are the only bytes that are not the file's own."""
    part = "whole file" if (a, b) == (1, n_lines) else f"lines {a}-{b} of {n_lines}"
    return f"===== FILE {rel} ({part}) =====\n{body}\n===== END FILE {rel} =====\n"


def chunk_lines(text: str, chunk_tokens: int) -> list[tuple[int, int, str]]:
    """(first_line, last_line, body) parts of at most ~chunk_tokens est tokens, cut at line ends, in order,
    covering every line exactly once (a single line longer than the budget is its own part)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    parts, cur, cur_t, start = [], [], 0, 1
    for i, ln in enumerate(lines, 1):
        t = est_tokens(ln + "\n")
        if cur and cur_t + t > chunk_tokens:
            parts.append((start, i - 1, "\n".join(cur)))
            cur, cur_t, start = [], 0, i
        cur.append(ln)
        cur_t += t
    if cur:
        parts.append((start, start + len(cur) - 1, "\n".join(cur)))
    return parts


def build_pieces(builder, rels: list[str], params: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """(pieces, skipped, chunked). A piece = {rel, a, b, n_lines, pack, tokens, file_sha256}."""
    from .pack import PackError
    lab = Path(builder.lab_root)
    pieces, skipped, chunked = [], [], []
    seen_real: dict = {}
    for rel in rels:
        p = lab / rel
        if not p.is_file():
            skipped.append({"rel": rel, "layer": "missing", "reason": "not a file"})
            continue
        real = str(p.resolve())
        if real in seen_real:                       # './a.md', a symlink alias: the same bytes must not be sent twice
            skipped.append({"rel": rel, "layer": "duplicate", "reason": f"same file as {seen_real[real]}"})
            continue
        seen_real[real] = rel
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append({"rel": rel, "layer": "decode", "reason": "not UTF-8"})
            continue
        if not text.strip():
            skipped.append({"rel": rel, "layer": "empty", "reason": "empty file"})
            continue
        try:
            # gate on the RAW text: L1 + L2 (path, name, frontmatter) + L3b (body) + L4 (secrets)
            builder.build_verify_only("seat_input", rel, text, origin=p, lane=LANE)
        except PackError as e:
            msg = str(e)
            skipped.append({"rel": rel, "layer": msg.split(":", 1)[0], "reason": msg[:200]})
            continue
        body = text[:-1] if text.endswith("\n") else text
        n_lines = body.count("\n") + 1
        if est_tokens(text) > params["max_file_tokens"]:
            parts = chunk_lines(text, params["chunk_tokens"])
            chunked.append({"rel": rel, "parts": len(parts), "est_tokens": est_tokens(text)})
        else:
            parts = [(1, n_lines, body)]
        mine = []
        try:
            # the text that is actually sent (marker lines + body, or a chunk) is scanned again; a hit that only
            # appears there (e.g. through the path in the marker line) skips the WHOLE file, like a raw-text hit
            for a, b, part in parts:
                name = rel if len(parts) == 1 else f"{rel}#L{a}-{b}"
                pk = builder.build_verify_only("seat_input", name, wrap(rel, a, b, n_lines, part), origin=p, lane=LANE)
                mine.append({"rel": rel, "a": a, "b": b, "n_lines": n_lines, "pack": pk,
                             "tokens": int(pk.report["est_tokens"]), "file_sha256": sha256_bytes(raw)})
        except PackError as e:
            msg = str(e)
            skipped.append({"rel": rel, "layer": msg.split(":", 1)[0], "reason": ("sent-text scan: " + msg)[:200]})
            if chunked and chunked[-1]["rel"] == rel:
                chunked.pop()
            continue
        pieces += mine
    return pieces, skipped, chunked


def make_shards(pieces: list[dict], target_tokens: int) -> list[list[dict]]:
    """Consecutive pieces (path order) packed up to target_tokens. A shard never spans two top-level lab
    categories; a piece larger than the target is a shard of its own."""
    shards, cur, cur_t, cur_top = [], [], 0, None
    for pc in pieces:
        top = pc["rel"].split("/", 1)[0]
        if cur and (top != cur_top or cur_t + pc["tokens"] > target_tokens):
            shards.append(cur)
            cur, cur_t = [], 0
        cur.append(pc)
        cur_t += pc["tokens"]
        cur_top = top
    if cur:
        shards.append(cur)
    return shards


def model_tag(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.replace("claude-", ""))


def plan_docaudit(spec: dict, bd: Path, builder, scanned: list, reqs: list, schema_of) -> dict:
    """ops.plan branch."""
    from .common import REPO_ROOT
    tpl = REPO_ROOT / spec["prompt"]["system_template"]
    system = builder.build_verify_only("spec_text", "system", tpl.read_text(encoding="utf-8"), origin=tpl)
    hp = Path(bd) / spec["header"]
    header = builder.build_verify_only("spec_text", hp.stem, hp.read_text(encoding="utf-8"), origin=hp)
    sch = schema_of(spec["output_schema"])
    scanned += [system, header]
    params = {**SHARD_DEFAULT, **(spec.get("shard") or {})}
    fs = spec["files"]
    list_sha = None
    if "list" in fs:
        lp = Path(bd) / fs["list"]
        rels = read_list(lp)
        list_sha = sha256_bytes(lp.read_bytes())
    else:
        rels = walk_roots(builder.lab_root, fs["roots"], fs.get("exclude", []))
    pieces, skipped, chunked = build_pieces(builder, rels, params)
    if not pieces:
        raise DocAuditError("no file passed the scan: nothing to send (see the skipped reasons with a dry build_pieces)")
    scanned += [pc["pack"] for pc in pieces]
    shards = make_shards(pieces, params["target_tokens"])
    shard_info = []
    for i, sh in enumerate(shards, 1):
        sid = f"sh{i:04d}"
        files = [{"rel": pc["rel"], "lines": [pc["a"], pc["b"]], "n_lines": pc["n_lines"],
                  "file_sha256": pc["file_sha256"], "pack_sha256": pc["pack"].pack_sha256} for pc in sh]
        src = "\n".join(f["pack_sha256"] for f in files)
        for model in spec["models"]:
            cid = M.make_custom_id("docaudit", src, f"{sid}{model_tag(model)}", 1)
            reqs.append(M.build_request(cid, model, spec["effort"], spec["max_tokens"], system, [header],
                                        [pc["pack"] for pc in sh], spec["cache_ttl"], sch,
                                        meta={"prompt_sha256": sha256_text(system.text), "header_sha256": header.pack_sha256,
                                              "shard": sid, "model": model, "files": files}))
        shard_info.append({"id": sid, "n_pieces": len(sh), "est_tokens": sum(pc["tokens"] for pc in sh),
                           "files": [f["rel"] + ("" if f["lines"] == [1, f["n_lines"]] else f"#L{f['lines'][0]}-{f['lines'][1]}")
                                     for f in files]})
    by_layer: dict = {}
    for s in skipped:
        by_layer[s["layer"]] = by_layer.get(s["layer"], 0) + 1
    return {"models": list(spec["models"]), "shard_params": params, "files_list": fs.get("list"),
            "files_list_sha256": list_sha, "roots": fs.get("roots"), "exclude": fs.get("exclude"),
            "n_files_listed": len(rels), "n_files_sent": len({pc["rel"] for pc in pieces}),
            "n_pieces": len(pieces), "n_shards": len(shards), "n_requests": len(shards) * len(spec["models"]),
            "est_tokens_sent_per_model": sum(pc["tokens"] for pc in pieces),
            "n_skipped": len(skipped), "skipped_by_layer": dict(sorted(by_layer.items())), "skipped": skipped,
            "chunked": chunked, "shards": shard_info,
            "skip_rule": "a file that fails the verify-only scan (L1/L2/L3b/L4), is missing, empty or not UTF-8 is not "
                         "sent and not masked; it is listed here",
            "seat_rule": "reader seat without tools: findings are candidates (basis internal | needs-check), never evidence"}


# ------------------------------------------------------------------------------------- offline post-check
_WS = re.compile(r"\s+")


def locate(text: str, quote: str) -> tuple[int, str]:
    """(1-indexed line of the quote's first character, how) or (0, 'not-found'). Exact substring first, then a
    whitespace-normalised match (the model may have re-wrapped a line)."""
    q = (quote or "").strip()
    if not q:
        return 0, "empty"
    i = text.find(q)
    if i >= 0:
        return text.count("\n", 0, i) + 1, "exact"
    nq = _WS.sub(" ", q)
    pos, norm = [], []
    for j, ch in enumerate(text):                         # normalised text with a map back to raw offsets
        if ch.isspace():
            if norm and norm[-1] == " ":
                continue
            norm.append(" ")
        else:
            norm.append(ch)
        pos.append(j)
    k = "".join(norm).find(nq)
    if k >= 0:
        return text.count("\n", 0, pos[k]) + 1, "normalised"
    return 0, "not-found"


def report(batch_dir: Path, lab_root: Path | None = None, batch_root: Path | None = None) -> dict:
    """Read the collected items of a docaudit batch and write docaudit/findings.jsonl + docaudit/REPORT.md."""
    from .common import LAB_ROOT
    bd = Path(batch_dir)
    lab = Path(lab_root or LAB_ROOT)
    rows, msha = M.load_manifest(bd / "manifest.jsonl")
    inbox_p = Path(batch_root or bd.parent.parent) / "INBOX.jsonl"
    if not inbox_p.exists():
        raise DocAuditError(f"{inbox_p} not found: run collect first")
    ok = {r["custom_id"]: r for r in (json.loads(ln) for ln in inbox_p.read_text(encoding="utf-8").splitlines() if ln.strip())}
    out, per_req = [], []
    texts: dict = {}
    for row in rows:
        cid, meta = row["custom_id"], row["meta"]
        ib = ok.get(cid)
        ip = bd / "items" / f"{cid}.md"
        if ib is None or not ip.exists():
            st = "quarantined" if (bd / "quarantine" / f"{cid}.md").exists() else "no-item"
            per_req.append({"custom_id": cid, "model": meta["model"], "shard": meta["shard"], "status": st, "n": 0})
            continue
        if ib.get("result_status") != "succeeded":
            per_req.append({"custom_id": cid, "model": meta["model"], "shard": meta["shard"], "status": ib.get("result_status"), "n": 0})
            continue
        body = ip.read_text(encoding="utf-8").split("\n---\n", 1)[-1]
        try:
            obj = json.loads(body)
            fnd = obj["findings"]
            assert isinstance(fnd, list)
        except Exception:  # noqa: BLE001
            per_req.append({"custom_id": cid, "model": meta["model"], "shard": meta["shard"], "status": "bad-json", "n": 0})
            continue
        sent = {f["rel"]: f for f in meta["files"]}

        def text_of(rel):
            """(text, changed-since-plan) of a sent file, from its BYTES (so CRLF files do not look changed)."""
            if rel not in texts:
                p = lab / rel
                raw = p.read_bytes() if p.is_file() else None
                texts[rel] = (raw.decode("utf-8", "replace") if raw is not None else "",
                              raw is None or sha256_bytes(raw) != sent[rel]["file_sha256"])
            return texts[rel]

        for k, f in enumerate(fnd):
            base = {"fid": f"{cid}-{k:02d}", "custom_id": cid, "model": meta["model"], "shard": meta["shard"]}
            if not isinstance(f, dict):                    # out-of-contract item: kept and flagged, never a crash
                out.append({**base, "bad_finding": True, "raw": str(f)[:300], "file_in_request": False,
                            "quote_found": False, "line": 0, "quote_match": "bad-finding"})
                continue
            rel = f.get("file") if isinstance(f.get("file"), str) else ""
            quote = f.get("quote") if isinstance(f.get("quote"), str) else ""
            row_o = {**f, **base}                  # provenance keys win over anything the model put in the item
            if rel not in sent:
                row_o.update(file_in_request=False, quote_found=False, line=0, quote_match="file-not-in-request")
            else:
                t, changed = text_of(rel)
                line, how = locate(t, quote)
                row_o.update(file_in_request=True, quote_found=line > 0, line=line, quote_match=how,
                             file_changed_since_plan=changed)
            cf, cq = f.get("counter_file"), f.get("counter_quote")
            if isinstance(cq, str) and cq.strip():        # the second side of an `internal` finding is checked too
                if isinstance(cf, str) and cf in sent:
                    cl, ch = locate(text_of(cf)[0], cq)
                    row_o.update(counter_found=cl > 0, counter_line=cl, counter_match=ch)
                else:
                    row_o.update(counter_found=False, counter_line=0, counter_match="file-not-in-request")
            out.append(row_o)
        # did the reader say it reached the end of every part it was given?
        fr = obj.get("files_read") if isinstance(obj.get("files_read"), list) else []
        last = {}
        for x in fr:
            if isinstance(x, dict) and isinstance(x.get("file"), str) and isinstance(x.get("last_line_read"), int):
                last[x["file"]] = max(last.get(x["file"], 0), x["last_line_read"])
        unread = sorted({f["rel"] for f in meta["files"] if last.get(f["rel"], 0) < f["lines"][1]})
        per_req.append({"custom_id": cid, "model": meta["model"], "shard": meta["shard"], "status": "succeeded",
                        "n": len(fnd), "parts_not_read_to_end": unread})
    od = bd / "docaudit"
    od.mkdir(exist_ok=True)
    (od / "findings.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out), encoding="utf-8")
    models = sorted({r["model"] for r in per_req})
    summ = {"manifest_sha256": msha, "n_requests": len(rows), "by_status": {}, "by_model": {}}
    for r in per_req:
        summ["by_status"][r["status"]] = summ["by_status"].get(r["status"], 0) + 1
    for m in models:
        fm = [r for r in out if r["model"] == m]
        summ["by_model"][m] = {
            "findings": len(fm), "quote_found": sum(1 for r in fm if r["quote_found"]),
            "quote_not_found": sum(1 for r in fm if not r["quote_found"]),
            "counter_checked": sum(1 for r in fm if "counter_found" in r),
            "counter_not_found": sum(1 for r in fm if r.get("counter_found") is False),
            "requests_with_unread_parts": sum(1 for r in per_req if r["model"] == m and r.get("parts_not_read_to_end")),
            "basis": {b: sum(1 for r in fm if r.get("basis") == b) for b in BASES},
            "category": {c: sum(1 for r in fm if r.get("category") == c) for c in CATEGORIES}}
    lines = ["---", "status: DRAFT (api-batch output — not evidence)", f"use: {LANE}", f"manifest_sha256: {msha}", "---", "",
             f"# docaudit report — {bd.name}", "",
             "Reader-seat candidates only. `quote_found: false` rows carry a quote that is not in the current file.", "",
             "| model | findings | quote found | quote not found | internal | needs-check |", "|---|---|---|---|---|---|"]
    for m, s in summ["by_model"].items():
        lines.append(f"| {m} | {s['findings']} | {s['quote_found']} | {s['quote_not_found']} | {s['basis']['internal']} | {s['basis']['needs-check']} |")
    lines += ["", "Request status: " + ", ".join(f"{k}={v}" for k, v in sorted(summ["by_status"].items())), ""]
    (od / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (od / "summary.json").write_text(json.dumps({**summ, "requests": per_req}, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return summ


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "report":
        print("usage: python -m tools.api_batch.docaudit report <batch_dir>", file=sys.stderr)
        return 2
    try:
        print(json.dumps(report(Path(argv[1])), indent=1, ensure_ascii=False, sort_keys=True))
    except Halt as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
