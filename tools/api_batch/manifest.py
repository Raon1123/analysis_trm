"""Requests, params builders, canonical hashes, manifest file (§5.5 single-path invariant, §8.2).

A Request is refused unless every text segment of its params is either a frozen template string
(doc-lint PROMPT/INSTRUCTION only) or exactly the text of a sealed, scanned Pack that the Request
references; every referenced Pack must actually appear. Non-text params are code-controlled: the
structured-output schema is compared by sha against an allow-set (NB-R3-7); an image block is accepted only as the
base64 of a referenced figqa_image Pack (fig-qa)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .common import Halt, canon_json, sha256_bytes, sha256_text
from .pack import Pack

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
USES = {"U0": "u0", "U1": "u1m", "U1-reduce": "u1r", "U2": "u2", "U3": "u3", "seat-doclint": "lint", "seat-figqa": "figqa",
        "triage": "tri", "probe": "probe", "SIZ-map": "sizm", "SIZ-reduce": "sizr"}
ALLOWED_TOP = {"model", "max_tokens", "system", "messages", "thinking", "output_config", "fallbacks"}
FORBIDDEN_TOP = {"stream", "speed"}     # F4 (fallbacks is allowed only so the P0b probe can send it)


class ManifestError(Halt):
    pass


def make_custom_id(use: str, src: str | bytes, cond: str, j: int) -> str:
    tok = USES.get(use, re.sub(r"[^a-z0-9]", "", use.lower()))
    b = src.encode("utf-8") if isinstance(src, str) else src
    cid = f"{tok}-{sha256_bytes(b)[:8]}-{re.sub(r'[^a-z0-9]', '', cond.lower()) or 'x'}-s{j}"
    if not CUSTOM_ID_RE.match(cid):
        raise ManifestError(f"custom_id {cid!r} violates F6")
    return cid


@dataclass
class Request:
    custom_id: str
    params: dict
    packs: tuple
    meta: dict = field(default_factory=dict)

    @property
    def params_sha256(self) -> str:
        return sha256_text(canon_json(self.params))


def _segments(params: dict, allow_image: bool = False) -> list[str]:
    segs: list[str] = []
    sysv = params.get("system")
    if isinstance(sysv, str):
        segs.append(sysv)
    elif isinstance(sysv, list):
        for b in sysv:
            if not isinstance(b, dict) or b.get("type") != "text":
                raise ManifestError("system block is not a text block")
            segs.append(b["text"])
    elif sysv is not None:
        raise ManifestError("system has an unsupported shape")
    for m in params.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            segs.append(c)
            continue
        for b in c or []:
            t = b.get("type")
            if t == "text":
                segs.append(b["text"])
            elif t == "document" and (b.get("source") or {}).get("type") == "text":
                segs.append(b["source"]["data"])
            elif t == "image" and allow_image:
                continue                       # validated against figqa_image Packs by validate_request
            else:
                raise ManifestError(f"content block type {t!r} is not allowed in P0a")
    return segs


VALID_CC = ({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "1h"})


def _all_blocks(params: dict) -> list[dict]:
    """System blocks then content blocks, in render order (dict blocks only)."""
    sysv = params.get("system")
    out = [b for b in sysv if isinstance(b, dict)] if isinstance(sysv, list) else []
    for m in params.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            out += [b for b in c if isinstance(b, dict)]
    return out


def _images(params: dict) -> list[dict]:
    out = []
    for m in params.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            out += [b for b in c if b.get("type") == "image"]
    return out


def validate_request(req: Request, frozen_texts=frozenset(), schema_shas=frozenset()) -> None:
    if not CUSTOM_ID_RE.match(req.custom_id or ""):
        raise ManifestError(f"custom_id {req.custom_id!r} violates F6")
    if not req.packs:
        raise ManifestError(f"{req.custom_id}: Request references no Pack (single-path invariant)")
    for p in req.packs:
        if not isinstance(p, Pack) or not p.sealed:
            raise ManifestError(f"{req.custom_id}: a referenced Pack is not a scanned, sealed Pack")
    keys = set(req.params)
    if keys & FORBIDDEN_TOP or not keys <= ALLOWED_TOP:
        raise ManifestError(f"{req.custom_id}: params keys {sorted(keys - ALLOWED_TOP | keys & FORBIDDEN_TOP)} not allowed")
    if not req.params.get("max_tokens") or req.params["max_tokens"] <= 0:
        raise ManifestError(f"{req.custom_id}: max_tokens must be explicit and > 0")
    img_texts = {p.text for p in req.packs if p.kind == "figqa_image"}
    pack_texts = {p.text for p in req.packs if p.kind != "figqa_image"}
    used = set()
    for b in _images(req.params):
        src = b.get("source") or {}
        if src.get("type") != "base64" or src.get("media_type") != "image/png" or src.get("data") not in img_texts \
                or set(b) - {"cache_control"} != {"type", "source"} or set(src) != {"type", "media_type", "data"}:
            raise ManifestError(f"{req.custom_id}: image block is not a base64 image/png of a referenced figqa_image Pack")
        used.add(src["data"])
    for seg in _segments(req.params, allow_image=True):
        if seg in frozen_texts:
            continue
        if seg in pack_texts:
            used.add(seg)
            continue
        raise ManifestError(f"{req.custom_id}: params text segment ({len(seg)} chars) is neither a frozen "
                            f"template nor the bytes of a referenced Pack")
    if used != pack_texts | img_texts:
        raise ManifestError(f"{req.custom_id}: a referenced Pack's bytes do not appear in params")
    ccs = [b["cache_control"] for b in _all_blocks(req.params) if "cache_control" in b]
    if len(ccs) > 4:
        raise ManifestError(f"{req.custom_id}: {len(ccs)} cache_control breakpoints > 4")
    if any(c not in VALID_CC for c in ccs):
        raise ManifestError(f"{req.custom_id}: cache_control must be ephemeral (5m) or ephemeral ttl 1h")
    ttls = ["1h" if c.get("ttl") == "1h" else "5m" for c in ccs]
    if "5m" in ttls and "1h" in ttls[ttls.index("5m"):]:
        raise ManifestError(f"{req.custom_id}: a 1h breakpoint after a 5m one (longer TTLs must come first)")
    fmt = ((req.params.get("output_config") or {}).get("format") or {})
    if fmt:
        if fmt.get("type") != "json_schema" or sha256_text(canon_json(fmt.get("schema"))) not in schema_shas:
            raise ManifestError(f"{req.custom_id}: output schema is not a registered code-controlled schema")


def _cc(cache_ttl: str) -> dict | None:
    if cache_ttl == "5m":
        return {"type": "ephemeral"}
    if cache_ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return None


def build_request(custom_id: str, model: str, effort: str, max_tokens: int, system_pack: Pack | None,
                  doc_packs: list[Pack] | None = None, var_packs: list[Pack] | None = None,
                  cache_ttl: str = "none", schema: dict | None = None, thinking: dict | None = None,
                  extra: dict | None = None, meta: dict | None = None, image_packs: list[Pack] | None = None,
                  cache_after: list[Pack] | None = None, images_after: Pack | None = None) -> Request:
    """§8.6 order: frozen system, [images], document blocks, variable tail.

    Breakpoints (only when cache_ttl is 5m|1h; with 'none' nothing is cached and cache_after is ignored):
      cache_after=None  -> legacy: the last document block carries cache_control (byte-identical to before).
      cache_after=[...] -> exactly those Packs' blocks (system, image, document or variable Pack; matched by
                           identity), at most 4. Put a breakpoint at the end of each shared stretch of the
                           prompt, never after the per-request tail.
    images_after=<doc Pack> places the image blocks right after that document instead of first (seat-figqa
    'ctx-first-v2' layout: the context shared across figures precedes the per-figure image)."""
    docs, imgs, var = list(doc_packs or []), list(image_packs or []), list(var_packs or [])
    if images_after is not None and not any(images_after is d for d in docs):
        raise ManifestError(f"{custom_id}: images_after is not one of the document Packs")
    blocks: list[tuple[Pack, dict]] = []

    def img_blocks():
        return [(p, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": p.text}}) for p in imgs]

    if images_after is None:
        blocks += img_blocks()
    for p in docs:
        blocks.append((p, {"type": "document", "title": p.name,
                           "source": {"type": "text", "media_type": "text/plain", "data": p.text}}))
        if images_after is p:
            blocks += img_blocks()
    blocks += [(p, {"type": "text", "text": p.text}) for p in var]
    sys_block = {"type": "text", "text": system_pack.text} if system_pack is not None else None
    cc = _cc(cache_ttl)
    bp_names: list[str] = []
    if cc is not None:
        if cache_after is None:
            last_doc = max((i for i, (p, b) in enumerate(blocks) if b["type"] == "document"), default=None)
            if last_doc is not None:
                blocks[last_doc][1]["cache_control"] = dict(cc)
        else:
            if len(cache_after) > 4:
                raise ManifestError(f"{custom_id}: {len(cache_after)} breakpoints > 4")
            for t in cache_after:
                if sys_block is not None and t is system_pack:
                    sys_block["cache_control"] = dict(cc)
                else:
                    hit = [b for p, b in blocks if p is t]
                    if len(hit) != 1:
                        raise ManifestError(f"{custom_id}: cache_after Pack {t.name!r} matches {len(hit)} blocks (need 1)")
                    hit[0]["cache_control"] = dict(cc)
                bp_names.append(t.name)
    content = [b for _, b in blocks]
    oc: dict = {"effort": effort}
    if schema is not None:
        oc["format"] = {"type": "json_schema", "schema": schema}
    params = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": content}], "output_config": oc}
    if sys_block is not None:
        params["system"] = [sys_block]
    if thinking:
        params["thinking"] = thinking
    if extra:
        params.update(extra)
    packs = tuple([*([system_pack] if system_pack is not None else []), *imgs, *docs, *var])
    m = {**(meta or {}), "cache_ttl": cache_ttl, "effort": effort}
    if bp_names:
        m["cache_after"] = bp_names
    return Request(custom_id, params, packs, m)


def cache_split(params: dict) -> tuple[int, int, str]:
    """(t_ctx, t_var, ttl) for cost formula 2: t_ctx = est tokens of every block up to and including the LAST
    cache_control block in render order (system, then content; each request may write all of it), t_var = the
    rest, ttl = '1h' if any breakpoint is 1h else '5m' ('none' and t_ctx = 0 when no block carries cache_control)."""
    from .common import est_tokens
    from .pack import image_tokens_b64
    seq = []
    sysv = params.get("system")
    if isinstance(sysv, str):
        seq.append((est_tokens(sysv), None))
    else:
        for b in sysv or []:
            seq.append((est_tokens(b.get("text", "")), b.get("cache_control")))
    for m in params.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            seq.append((est_tokens(c), None))
            continue
        for b in c or []:
            t = b.get("type")
            if t == "document":
                n = est_tokens(b["source"]["data"])
            elif t == "image":
                n = image_tokens_b64(b["source"]["data"])
            else:
                n = est_tokens(b.get("text", ""))
            seq.append((n, b.get("cache_control")))
    last = max((i for i, (_, cc) in enumerate(seq) if cc), default=None)
    total = sum(n for n, _ in seq)
    if last is None:
        return 0, total, "none"
    t_ctx = sum(n for n, _ in seq[:last + 1])
    ttl = "1h" if any((cc or {}).get("ttl") == "1h" for _, cc in seq) else "5m"
    return t_ctx, total - t_ctx, ttl


def build_doclint_request(custom_id: str, model: str, results_pack: Pack, prereg_pack: Pack):
    """Second permitted path: the frozen build_params applied to the Packs' bytes (§8.8)."""
    from .doclint_bridge import doc_lint_module
    dl = doc_lint_module()
    params = dl.build_params(model, results_pack.text, prereg_pack.text)
    req = Request(custom_id, params, (results_pack, prereg_pack), {"cache_ttl": "5m", "effort": dl.EFFORT})
    frozen = frozenset({dl.PROMPT, dl.INSTRUCTION})
    schemas = frozenset({sha256_text(canon_json(dl.SCHEMA))})
    validate_request(req, frozen, schemas)
    return req


def manifest_rows(requests: list[Request]) -> list[dict]:
    ids = [r.custom_id for r in requests]
    if len(set(ids)) != len(ids):
        raise ManifestError("duplicate custom_id in manifest")
    return [{"custom_id": r.custom_id, "params": r.params, "params_sha256": r.params_sha256,
             "pack_sha256s": [p.pack_sha256 for p in r.packs], "meta": r.meta} for r in requests]


def write_manifest(requests: list[Request], path: Path) -> str:
    data = ("\n".join(canon_json(r) for r in manifest_rows(requests)) + "\n").encode("utf-8")
    Path(path).write_bytes(data)
    return sha256_bytes(data)


def load_manifest(path: Path) -> tuple[list[dict], str]:
    data = Path(path).read_bytes()
    rows = [json.loads(ln) for ln in data.decode("utf-8").splitlines() if ln.strip()]
    ids = [r["custom_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ManifestError("duplicate custom_id in manifest file")
    for r in rows:
        if sha256_text(canon_json(r["params"])) != r["params_sha256"]:
            raise ManifestError(f"{r['custom_id']}: params_sha256 mismatch")
    return rows, sha256_bytes(data)
