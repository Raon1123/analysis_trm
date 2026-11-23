"""exaone: LG AI Research K-EXAONE 2.5 lane -- an advisory ("참고용") free-text or JSON reading of user-listed,
non-sensitive lab text, through the same rails as the Jev lane (2026-11-10 user: the user enters the key in their own
shell; only the non-sensitive samples they list are sent).

API (docs.exaone.ai, read 2026-11-10): POST https://api.exaone.ai/v1/chat/completions, OpenAI-compatible
chat.completion (+ message.reasoning_content, usage.reasoning_tokens). One model id `k-exaone_v2.5`, text only,
context 524,288 tokens with input + max_tokens <= context, max_tokens INCLUDES reasoning tokens, reasoning cannot be
turned off and `reasoning_effort` defaults to `max`. Not documented: price, seed, response_format, tools, batch, prompt
caching, numeric rate limits (HTTP 429 + x-*-rate-limit headers). ToS: outputs not reproducible, the model may change
without notice, keys expire 30 days after issue, inputs/outputs may be used for training by default.

  plan <spec>      offline. spec = {name, model (= k-exaone_v2.5), reasoning_effort (low|high|max, REQUIRED),
                   max_tokens (REQUIRED), max_total_tokens (REQUIRED run-wide cap), instructions (inline text ->
                   spec_text Pack), items: [{id, docs: [<path>...]}] (verify-only Packs, neutral doc1..docN labels),
                   output?: text|json, json_required_keys?: [token...], repeats?: 1..5, temperature?, top_p?}.
                   One chat/completions call per item x repeat. Cost gating is in TOKENS (no published price):
                   hard_worst_tokens = sum(est_input + max_tokens) <= max_total_tokens. Writes manifest.jsonl,
                   plan.json, SEND_LIST.md (what leaves the machine), ledger `planned` (lane exaone).
  approve          user-only (`approve <plan_dir> --tokens N`, refused inside a Claude Code session); one approval =
                   one run.
  run <plan_dir>   third-party send (LG AI Research). Gates as jev.run; per request `sync_sent` BEFORE the send,
                   raw/<id>.json, out/results.jsonl, `sync_done` rc 0 ok, 1 send raised, 2 model != pin (halt),
                   3 HTTP 429 (halt), 4 response failed validation (halts on non-2xx HTTP, missing/zero usage,
                   unencodable reply text, local write error). The model is checked before usage. SEND_LIST.md is bound into plan_hash and
                   re-rendered from the manifest by load_plan; display_safe (invisible / control characters,
                   HTML comments, heading spoofs, marker lookalikes) runs on every render.
  probe            one fixed code-controlled ping (no lab text, no Pack): shows the live response shape.

The API key is read from the environment only in RealExaoneClient._key (used by send; ready() tests presence before
any ledger row). See tools/api_batch/bin/.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import secrets
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from .common import Halt, canon_json, est_tokens, in_claude_code_session, sha256_bytes, sha256_file, sha256_text
from .jev import DATA_KIND, ITEM_RE
from .ledger import Ledger, approval_id, now
from .pack import Pack, PackBuilder
from .spec import SpecError

USE = "exaone"
PROBE_LANE = "exaone-probe"
MODEL = "k-exaone_v2.5"
URL = "https://api.exaone.ai/v1/chat/completions"
CONTEXT_TOKENS = 524_288
TIMEOUT_S = 900
EFFORTS = ("low", "high", "max")
OUTPUTS = ("text", "json")
# request params named in the API reference (2026-11-10); anything else is undocumented and refused
DOCUMENTED_PARAMS = frozenset({"model", "messages", "max_tokens", "stream", "stream_options", "temperature", "top_p",
                               "top_k", "presence_penalty", "reasoning_effort", "chat_template_kwargs"})
LANE_PARAMS = frozenset({"model", "messages", "max_tokens", "reasoning_effort", "temperature", "top_p"})
REQUIRED_PARAMS = frozenset({"model", "messages", "max_tokens", "reasoning_effort"})
assert LANE_PARAMS <= DOCUMENTED_PARAMS
SPEC_KEYS = {"name", "model", "reasoning_effort", "max_tokens", "max_total_tokens", "instructions", "items", "output",
             "json_required_keys", "repeats", "temperature", "top_p"}
SPEC_REQUIRED = ("name", "model", "reasoning_effort", "max_tokens", "max_total_tokens", "instructions", "items")
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CID_RE = re.compile(r"^exaone-[a-z0-9][a-z0-9_-]{0,40}-r[1-5]$")
MARKER_RE = re.compile(r"=====\s*(BEGIN|END)\s+doc", re.I)
# display safety (R3-1): categories that render as nothing / move the cursor / reorder text / are not characters
UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
# lines that would read, in a raw view, as one of SEND_LIST.md's own section headings
HEADING_SPOOF_RE = re.compile(r"^\s*#{1,6}\s*(SEND_LIST|기기\s*밖으로|파일\s*외에|system\s*메시지|user\s*메시지|요청별)", re.I)
GUTTER = "│ "
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
RATE_HDR_RE = re.compile(r"^(x-[a-z]+-rate-limit(-remaining)?|retry-after(-ms)?)$")
DESTINATION = "LG AI연구원 api.exaone.ai — 기본 약관상 입출력이 모델 학습에 쓰일 수 있음"
PRICE = "unpublished"

CAPABILITY = {"batch": False, "prefix_cache": False, "images": False, "output_kind": "free_text_advisory",
              "max_context_tokens": CONTEXT_TOKENS, "seed": False, "price": PRICE}

# ---- code-controlled scaffolding (the only bytes in a request that are not a sealed Pack's bytes)
USER_HEAD = ("The {n} document(s) below are the material for this task. Each one lies between its own BEGIN and END "
             "marker lines. The labels doc1..docN are neutral and carry no meaning.\n")
DOC_BLOCK = "\n===== BEGIN {label} =====\n{text}\n===== END {label} =====\n"
JSON_TAIL = ("\n\n---\nOutput format (fixed by the calling code): reply with exactly one JSON object and nothing else; "
             "one ```json fence around it is allowed. The object must contain these keys: {keys}.")
PROBE_SYSTEM, PROBE_USER, PROBE_MAX_TOKENS = "Reply with the single word OK.", "ping", 1024

# ---- measured by the user's live probe 2026-11-11 (lab/38_api-batch/13_exaone-probe/probe-20261111T064521Z.json):
# rate headers x-second-rate-limit 1, x-minute-rate-limit 60, x-day-rate-limit 100 (this account)
DAY_REQUEST_LIMIT_OBSERVED = 100     # requests/day, x-day-rate-limit in the 2026-11-11 probe; plan refuses more
DAY_LIMIT_SOURCE = "2026-11-11 probe"
MIN_SEND_INTERVAL_S = 1.2            # >= 1/s (x-second-rate-limit 1) + margin; measured from the previous send's START
DAY_REMAINING_HDR = "x-day-rate-limit-remaining"
WEIGHT_VERSION_MAX = 64
_sleep, _clock = time.sleep, time.monotonic   # run() defaults (tests inject their own)


class Refused(Halt):
    pass


class ExaoneError(Halt):
    pass


class ExaoneSendError(RuntimeError):
    """A send failure whose text has been scrubbed of the key."""


# ------------------------------------------------------------------------------------- spec
def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pos(text: str, i: int) -> str:
    line = text.count("\n", 0, i) + 1
    return f"line {line} col {i - (text.rfind(chr(10), 0, i) + 1) + 1}"


# blank-rendering code points outside the refused categories: combining grapheme joiner, Hangul fillers, Khmer
# inherent vowels, Mongolian free variation selectors
INVISIBLE_EXTRA = frozenset({0x034F, 0x115F, 0x1160, 0x3164, 0xFFA0, 0x17B4, 0x17B5, 0x180B, 0x180C, 0x180D, 0x180F})


# instructions only (R4-5): blank glyphs and combining Latin letters (stack into an unreadable smudge)
INSTR_BLANK = frozenset({0x2800, 0x1D159, 0x16FE4, 0x1BC9D, 0x1BC9E, *range(0x0363, 0x0370), *range(0x1DD3, 0x1DF5)})
INSTR_MAX_LINE = 400            # R4-4: characters per instructions line
INSTR_PAD_RE = re.compile(r"\S[\t \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]{9,}")   # > 8 blanks after text
MAX_MARKS = 2                   # combining marks allowed after one base letter / digit in instructions


def _is_jamo_l(o: int) -> bool:
    return 0x1100 <= o <= 0x115F or 0xA960 <= o <= 0xA97F


def _is_jamo_vt(o: int) -> bool:
    return 0x1160 <= o <= 0x11FF or 0xD7B0 <= o <= 0xD7FF


def _instructions_shape(text: str, where: str) -> None:
    """R4-4 / R4-5, instructions only (user-written, rendered in SEND_LIST.md): no line longer than INSTR_MAX_LINE
    characters, no run of more than 8 blanks after text on a line, no blank glyph / combining Latin letter, no Hangul
    vowel/final jamo that is not part of a sequence starting with an initial jamo (a lone one is zero-width), and no
    combining mark that does not follow a base letter/digit (at most MAX_MARKS in a row)."""
    for n, ln in enumerate(text.split("\n"), 1):
        if len(ln) > INSTR_MAX_LINE:
            raise Refused(f"exaone display check: {where} line {n} is {len(ln)} characters (max {INSTR_MAX_LINE}): a "
                          "wrapped line has continuation rows without a gutter")
        m = INSTR_PAD_RE.search(ln)
        if m:
            raise Refused(f"exaone display check: {where} line {n} col {m.start() + 2} has a run of "
                          f"{len(m.group(0)) - 1} blanks after text (max 8)")
    prev_base, marks, jamo = False, 0, False
    for i, ch in enumerate(text):
        o, cat = ord(ch), unicodedata.category(ch)
        if o in INSTR_BLANK:
            raise Refused(f"exaone display check: {where} contains U+{o:04X} ({cat}, blank glyph / combining letter) at "
                          f"{_pos(text, i)}")
        if _is_jamo_vt(o) and not jamo:
            raise Refused(f"exaone display check: {where} contains a Hangul jamo U+{o:04X} without an initial consonant "
                          f"jamo before it (zero-width) at {_pos(text, i)}")
        jamo = _is_jamo_l(o) or (jamo and _is_jamo_vt(o))
        if cat in ("Mn", "Me"):
            marks += 1
            if not prev_base or marks > MAX_MARKS:
                raise Refused(f"exaone display check: {where} has a combining mark U+{o:04X} at {_pos(text, i)} that does "
                              f"not follow a letter/digit (or more than {MAX_MARKS} in a row)")
        else:
            marks, prev_base = 0, cat[0] in "LN"


def _is_variation_selector(ch: str) -> bool:
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or o in INVISIBLE_EXTRA


def display_safe(text, where: str, *, multiline: bool = False, prose: bool = False, marker: bool = False) -> str:
    """R3-1 display-safety gate for every string SEND_LIST.md shows or vouches for. Refuses (Refused naming the field,
    the code point and line/col) any character in Unicode categories Cc (except \\n and \\t when multiline), Cf, Cs, Co,
    Cn, Zl, Zp, and the variation selectors. With prose (instructions: text SEND_LIST.md renders) also refuses an HTML
    comment opener / closer and a line that reads as one of SEND_LIST.md's own section headings. With prose or marker
    (listed doc text: sent, never rendered in SEND_LIST.md; 2026-11-10 coordinator decision (b) allows HTML comments and
    headings there) refuses doc-marker lookalikes (NFKC form, '=' runs collapsed), which would blur the request framing.
    Returns text unchanged: nothing is normalised in what is sent."""
    if not isinstance(text, str):
        raise Refused(f"exaone display check: {where} is not a string")
    for i, ch in enumerate(text):
        if multiline and ch in "\n\t":
            continue
        cat = unicodedata.category(ch)
        if cat in UNSAFE_CATEGORIES or _is_variation_selector(ch):
            raise Refused(f"exaone display check: {where} contains U+{ord(ch):04X} ({cat}"
                          f"{', invisible' if _is_variation_selector(ch) else ''}) at {_pos(text, i)} -- "
                          "invisible / control characters are refused so SEND_LIST.md shows what is sent")
    nf = unicodedata.normalize("NFKC", text)
    if prose:
        _instructions_shape(text, where)
        for tok in ("<!--", "-->"):
            for src in (text, nf):
                j = src.find(tok)
                if j >= 0:
                    raise Refused(f"exaone display check: {where} contains the HTML comment token {tok!r} at "
                                  f"{_pos(src, j)} (a markdown viewer would hide text)")
        for n, ln in enumerate(nf.split("\n"), 1):
            if HEADING_SPOOF_RE.match(ln):
                raise Refused(f"exaone display check: {where} line {n} reads as a SEND_LIST.md section heading "
                              f"({ln[:60]!r})")
    if prose or marker:
        m = MARKER_RE.search(re.sub(r"={2,}", "=====", nf))
        if m:
            raise Refused(f"exaone display check: {where} contains a doc-marker lookalike ({m.group(0)!r} after NFKC "
                          "and '=' run collapse)")
    return text


def load_spec(path: Path) -> dict:
    try:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SpecError(f"exaone spec unreadable: {type(e).__name__}")
    if not isinstance(spec, dict):
        raise SpecError("exaone spec is not a mapping")
    if set(spec) - SPEC_KEYS:
        raise SpecError(f"exaone spec: unknown keys {sorted(set(spec) - SPEC_KEYS)}")
    for k in SPEC_REQUIRED:
        if k not in spec:
            raise SpecError(f"exaone spec: missing key {k!r} (no defaults: the API default reasoning_effort=max "
                            "must never be used implicitly)")
    if not isinstance(spec["name"], str) or not NAME_RE.fullmatch(spec["name"]):
        raise SpecError(f"exaone spec: name must match {NAME_RE.pattern}")
    if spec["model"] != MODEL:
        raise SpecError(f"exaone spec: model must be exactly {MODEL!r}")
    if spec["reasoning_effort"] not in EFFORTS:
        raise SpecError(f"exaone spec: reasoning_effort must be one of {EFFORTS}")
    mt = spec["max_tokens"]
    if not _int(mt) or not 1 <= mt < CONTEXT_TOKENS:
        raise SpecError(f"exaone spec: max_tokens must be an integer 1..{CONTEXT_TOKENS - 1} (includes reasoning tokens)")
    mtt = spec["max_total_tokens"]
    if not _int(mtt) or mtt <= 0:
        raise SpecError("exaone spec: max_total_tokens must be a positive integer (run-wide token cap)")
    if not isinstance(spec["instructions"], str) or not spec["instructions"].strip():
        raise SpecError("exaone spec: instructions must be a non-empty string")
    display_safe(spec["instructions"], "instructions", multiline=True, prose=True)   # early; re-run on every render
    out = spec.get("output", "text")
    if out not in OUTPUTS:
        raise SpecError("exaone spec: output must be text | json")
    keys = spec.get("json_required_keys")
    if keys is not None:
        if out != "json":
            raise SpecError("exaone spec: json_required_keys needs output: json")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and KEY_RE.fullmatch(k) for k in keys) \
                or len(set(keys)) != len(keys):
            raise SpecError(f"exaone spec: json_required_keys must be a non-empty list of distinct tokens {KEY_RE.pattern}")
    rep = spec.get("repeats", 1)
    if not _int(rep) or not 1 <= rep <= 5:
        raise SpecError("exaone spec: repeats must be an integer 1..5")
    t = spec.get("temperature")
    if t is not None and (not _num(t) or not 0 <= t <= 2):
        raise SpecError("exaone spec: temperature (optional) must be a number 0..2")
    tp = spec.get("top_p")
    if tp is not None and (not _num(tp) or not 0 < tp <= 1):
        raise SpecError("exaone spec: top_p (optional) must be a number in (0, 1]")
    items = spec["items"]
    if not isinstance(items, list) or not items:
        raise SpecError("exaone spec: items must be a non-empty list")
    ids = []
    for it in items:
        if not isinstance(it, dict) or set(it) != {"id", "docs"} or not isinstance(it["id"], str) or not ITEM_RE.fullmatch(it["id"]):
            raise SpecError("exaone spec: each item is exactly {id: [a-z0-9_-], docs: [<path>...]}")
        if not isinstance(it["docs"], list) or not it["docs"] or not all(isinstance(x, str) and x for x in it["docs"]):
            raise SpecError(f"exaone spec: item {it['id']} docs must be a non-empty list of paths")
        for x in it["docs"]:
            display_safe(x, f"item {it['id']} doc path")
        ids.append(it["id"])
    if len(set(ids)) != len(ids):
        raise SpecError("exaone spec: duplicate item id")
    return spec


# ------------------------------------------------------------------------------------- request build / validate
def render_messages(instructions: str, docs: list[str], output: str, json_keys: list[str]) -> list[dict]:
    """The only way a request's text is assembled: Pack bytes + the fixed scaffolding above."""
    system = instructions + (JSON_TAIL.format(keys=", ".join(json_keys) if json_keys else "(none fixed)")
                             if output == "json" else "")
    user = USER_HEAD.format(n=len(docs)) + "".join(DOC_BLOCK.format(label=f"doc{i}", text=t) for i, t in enumerate(docs, 1))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_params(spec: dict, instructions: Pack, docs: list[Pack]) -> dict:
    p = {"model": spec["model"], "reasoning_effort": spec["reasoning_effort"], "max_tokens": spec["max_tokens"],
         "messages": render_messages(instructions.text, [d.text for d in docs], spec.get("output", "text"),
                                     list(spec.get("json_required_keys") or []))}
    for k in ("temperature", "top_p"):
        if spec.get(k) is not None:
            p[k] = spec[k]
    return p


def est_input_tokens(params: dict) -> int:
    """Over-estimate (common.est_tokens) of every message string, scaffolding included, plus 8 per message."""
    return sum(est_tokens(m.get("content") or "") + 8 for m in params.get("messages") or [] if isinstance(m, dict))


def request_worst_tokens(params: dict) -> int:
    return est_input_tokens(params) + int(params["max_tokens"])


def validate_params(params: dict, instructions: Pack, docs: list[Pack], output: str = "text",
                    json_keys: list[str] | tuple = ()) -> None:
    """Single-path invariant: params keys are documented lane keys; the messages are EXACTLY render_messages() of the
    referenced sealed Packs (so every text byte is a Pack's bytes or fixed scaffolding); request fits the context."""
    if not isinstance(params, dict):
        raise ExaoneError("exaone params is not a mapping")
    if set(params) - DOCUMENTED_PARAMS:
        raise ExaoneError(f"exaone params carry undocumented keys {sorted(set(params) - DOCUMENTED_PARAMS)}")
    if set(params) - LANE_PARAMS or not REQUIRED_PARAMS <= set(params):
        raise ExaoneError(f"exaone params keys must be {sorted(REQUIRED_PARAMS)} + optional temperature/top_p "
                          f"(got {sorted(params)})")
    if params["model"] != MODEL:
        raise ExaoneError(f"exaone model {params['model']!r} is not {MODEL!r}")
    if params["reasoning_effort"] not in EFFORTS:
        raise ExaoneError("exaone reasoning_effort must be explicit low|high|max")
    mt = params["max_tokens"]
    if not _int(mt) or mt < 1:
        raise ExaoneError("exaone max_tokens must be a positive integer")
    if "temperature" in params and (not _num(params["temperature"]) or not 0 <= params["temperature"] <= 2):
        raise ExaoneError("exaone temperature out of range")
    if "top_p" in params and (not _num(params["top_p"]) or not 0 < params["top_p"] <= 1):
        raise ExaoneError("exaone top_p out of range")
    allp = [instructions, *docs]
    if not docs or any(not isinstance(p, Pack) or not p.sealed for p in allp):
        raise ExaoneError("exaone request references no doc Pack or an unsealed Pack")
    if output not in OUTPUTS or (json_keys and output != "json") \
            or not all(isinstance(k, str) and KEY_RE.fullmatch(k) for k in json_keys):
        raise ExaoneError("exaone output / json_required_keys invalid (keys are code-controlled tokens)")
    for d in docs:
        if MARKER_RE.search(d.text) or MARKER_RE.search(re.sub(r"={2,}", "=====", unicodedata.normalize("NFKC", d.text))):
            raise ExaoneError("a doc contains the scaffolding marker '===== BEGIN/END doc' (would blur doc boundaries)")
    if params["messages"] != render_messages(instructions.text, [d.text for d in docs], output, list(json_keys)):
        raise ExaoneError("exaone request text is not the bytes of its referenced Packs + fixed scaffolding")
    n_in = est_input_tokens(params)
    if n_in + mt > CONTEXT_TOKENS:
        raise ExaoneError(f"exaone request too large: est input {n_in} + max_tokens {mt} > context {CONTEXT_TOKENS}")


# ------------------------------------------------------------------------------------- response
def usage_tokens(usage) -> int | None:
    """Tokens consumed per usage, or None (fail closed). MEASURED (2026-11-11 probe: prompt 25, completion 13,
    reasoning 10, total 38): reasoning tokens are inside completion_tokens and total = prompt + completion. So:
    prompt/completion must be ints >= 0; total, if given, must equal prompt + completion; a reported reasoning count
    must not exceed completion_tokens. Anything else is a usage shape we have not seen -> None (rc 4 + halt)."""
    if not isinstance(usage, dict):
        return None
    p, c, t = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens", None)
    if not (_int(p) and p >= 0 and _int(c) and c >= 0):
        return None
    if t is not None and not (_int(t) and t == p + c):
        return None
    r = _reasoning_tokens(usage)
    if r is not None and not 0 <= r <= c:
        return None
    return p + c


def _reported_tokens(usage) -> int:
    """For charging a request whose usage failed usage_tokens: the largest integer the server reported (0 if none)."""
    if not isinstance(usage, dict):
        return 0
    v = [usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
    v = [x for x in v if _int(x) and x > 0]
    return max([sum(v[:2]) if len(v) >= 2 else 0, *v, 0])


def weight_version(raw) -> str | None:
    """metadata.weight_version of a reply (provenance), or None: absent, not a string, longer than WEIGHT_VERSION_MAX,
    or not display-safe (invisible / control characters)."""
    md = raw.get("metadata") if isinstance(raw, dict) else None
    wv = md.get("weight_version") if isinstance(md, dict) else None
    if not isinstance(wv, str) or not wv or len(wv) > WEIGHT_VERSION_MAX:
        return None
    try:
        display_safe(wv, "weight_version")
    except Halt:
        return None
    return wv


def day_remaining(rate) -> int | None:
    """x-day-rate-limit-remaining as a non-negative int, or None (missing / unparseable: carry on)."""
    v = rate.get(DAY_REMAINING_HDR) if isinstance(rate, dict) else None
    if isinstance(v, str) and re.fullmatch(r"[0-9]{1,9}", v.strip()):
        return int(v.strip())
    return None


def _reasoning_tokens(usage: dict):
    r = usage.get("reasoning_tokens")
    if r is None and isinstance(usage.get("completion_tokens_details"), dict):
        r = usage["completion_tokens_details"].get("reasoning_tokens")
    return r if _int(r) else None


def _reject_constant(c):
    raise ValueError(f"non-standard JSON constant {c}")


def strip_fence(s: str) -> str:
    t = s.strip()
    m = re.fullmatch(r"```(?:json)?[ \t]*\n(.*)\n```", t, flags=re.S)
    return m.group(1).strip() if m else t


def normalize(raw: dict, output: str, json_keys) -> dict:
    """Validate one chat.completion body: {content (stripped: the probe reply was "\\n\\nOK"), parsed,
    finish_reason, reasoning_tokens}. raw/ keeps the untouched string. Raises ExaoneError."""
    if not isinstance(raw, dict):
        raise ExaoneError("response is not a JSON object")
    ch = raw.get("choices")
    if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict):
        raise ExaoneError("response has no choices")
    msg = ch[0].get("message")
    fr = ch[0].get("finish_reason")
    if not isinstance(msg, dict):
        raise ExaoneError("choices[0].message missing")
    content = msg.get("content")
    if fr == "length":
        raise ExaoneError("truncated: finish_reason=length (max_tokens includes reasoning tokens)")
    if not isinstance(content, str) or not content.strip():
        raise ExaoneError("empty content (reasoning may have used the whole max_tokens)")
    parsed = None
    if output == "json":
        try:
            parsed = json.loads(strip_fence(content), parse_constant=_reject_constant)
        except Exception as e:  # noqa: BLE001
            raise ExaoneError(f"output: json but content is not JSON ({type(e).__name__})")
        if not isinstance(parsed, dict):
            raise ExaoneError("output: json but content is not a JSON object")
        miss = [k for k in json_keys if k not in parsed]
        if miss:
            raise ExaoneError(f"JSON object lacks required keys {miss}")
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return {"content": content.strip(), "parsed": parsed, "finish_reason": fr if isinstance(fr, (str, type(None))) else repr(fr),
            "reasoning_tokens": _reasoning_tokens(usage)}


# ------------------------------------------------------------------------------------- clients
def _key_variants(key: str) -> list[str]:
    """The key as it may be echoed: raw, JSON-escaped, JSON-escaped with '/' written as '\\/'."""
    j = json.dumps(key)[1:-1]
    return sorted({key, j, j.replace("/", "\\/"), key.replace("/", "\\/")}, key=len, reverse=True)


def _scrub_key(s: str, key: str) -> str:
    """Remove every echoed form of the key (applied to the response body before parsing; model text is untouched)."""
    if key:
        for v in _key_variants(key):
            s = s.replace(v, "[REDACTED]")
    return s


def _scrub(s: str, key: str) -> str:
    """Error-path scrub (exception text, error bodies, headers): the key plus any 'Bearer <token>'."""
    return re.sub(r"(?i)\bBearer\s+(?!\[REDACTED\])[A-Za-z0-9._~+/=\-]{8,}", "Bearer [REDACTED]", _scrub_key(s, key))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a 3xx: urllib would copy the Authorization header to the redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect refused ({code})", headers, fp)


def _open_no_redirect(req, timeout=None):
    return urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout)


def _rate_headers(hdrs, key: str) -> dict:
    out = {}
    try:
        items = list(hdrs.items()) if hdrs is not None else []
    except Exception:  # noqa: BLE001
        items = []
    for k, v in items:
        if isinstance(k, str) and RATE_HDR_RE.fullmatch(k.lower()):
            out[k.lower()] = _scrub(str(v), key)[:200]
    return out


class RealExaoneClient:
    """stdlib urllib, non-streaming. Returns {status, rate_limit, raw, body_error}; HTTP error statuses are returned,
    not raised. Transport failures raise ExaoneSendError with the key scrubbed. Request headers are never stored."""

    def __init__(self, urlopen=None, timeout: float = TIMEOUT_S):
        self._urlopen = urlopen or _open_no_redirect
        self.timeout = timeout
        self.sdk_version = "urllib"

    @staticmethod
    def _key() -> str:
        """The ONE place the API key is read (send uses it; ready() only tests presence)."""
        return os.environ.get("EXAONE_API_KEY") or ""

    def ready(self) -> bool:
        """Key present? Checked by run/probe before the first sync_sent, so a forgotten wrapper consumes nothing."""
        return bool(self._key())

    def send(self, params: dict) -> dict:
        key = self._key()
        if not key:
            raise ExaoneSendError("EXAONE_API_KEY is not set: run under tools/api_batch/bin/with-exaone-key")
        try:
            req = urllib.request.Request(URL, data=json.dumps(params, ensure_ascii=False).encode("utf-8"), method="POST",
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                resp = self._urlopen(req, timeout=self.timeout)
                try:
                    status, hdrs, body = int(getattr(resp, "status", None) or resp.getcode()), resp.headers, resp.read()
                finally:
                    close = getattr(resp, "close", None)
                    if close:
                        close()
            except urllib.error.HTTPError as e:
                status, hdrs = int(e.code), e.headers
                try:
                    body = e.read() or b""
                except Exception:  # noqa: BLE001
                    body = b""
            body_s = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
            text = _scrub_key(body_s, key)
            err_text = _scrub(body_s, key)
            rate = _rate_headers(hdrs, key)
        except Exception as e:  # noqa: BLE001
            raise ExaoneSendError(_scrub(f"{type(e).__name__}: {e}", key)[:500]) from None
        finally:
            key = ""  # noqa: F841
        try:
            raw = json.loads(text)
        except Exception:  # noqa: BLE001
            raw = None
        return {"status": status, "rate_limit": rate, "raw": raw,
                "body_error": None if (isinstance(raw, dict) and 200 <= status < 300) else err_text[:1000]}


class FakeExaoneClient:
    """Offline stand-in. Knobs: model_override, fail, bad_json (non-JSON content for output json), no_usage,
    rate_limited (429), usage_total (forced total_tokens; completion_tokens follows it so total = prompt + completion
    as measured, unless that would be negative or total is not an int), status (forced HTTP status), content,
    day_remaining (x-day-rate-limit-remaining after the FIRST reply, counting down; None = header absent),
    weight_version (metadata.weight_version; None = no metadata). Rate headers default to the measured shape
    (2026-11-11 probe: second 1, minute 60, day 100)."""

    def __init__(self, model_override=None, fail=False, bad_json=False, no_usage=False, rate_limited=False,
                 usage_total=None, status=200, content=None, day_remaining="default", weight_version="default"):
        self.calls = []
        self.model_override, self.fail, self.bad_json, self.no_usage = model_override, fail, bad_json, no_usage
        self.rate_limited, self.usage_total, self.status, self.content = rate_limited, usage_total, status, content
        self.day_remaining = DAY_REQUEST_LIMIT_OBSERVED - 1 if day_remaining == "default" else day_remaining
        self.weight_version = weight_version
        self.sdk_version = "fake"
        self.not_ready = False

    def ready(self) -> bool:
        return not self.not_ready

    def send(self, params: dict) -> dict:
        self.calls.append(params)
        if self.fail:
            raise ExaoneSendError("fake send failure")
        n = len(self.calls)
        rate = {"x-second-rate-limit": "1", "x-second-rate-limit-remaining": "0",
                "x-minute-rate-limit": "60", "x-minute-rate-limit-remaining": str(max(60 - n, 0)),
                "x-day-rate-limit": str(DAY_REQUEST_LIMIT_OBSERVED)}
        if self.day_remaining is not None:
            dr = self.day_remaining
            rate[DAY_REMAINING_HDR] = str(max(dr - (n - 1), 0)) if _int(dr) else str(dr)
        if self.rate_limited:
            return {"status": 429, "rate_limit": {**rate, "retry-after": "30"}, "raw": {"error": "rate limited"},
                    "body_error": '{"error": "rate limited"}'}
        sysmsg = params["messages"][0]["content"]
        if self.content is not None:
            content = self.content
        elif "Output format (fixed by the calling code)" in sysmsg and not self.bad_json:
            keys = sysmsg.rsplit("these keys: ", 1)[-1].rstrip(".").split(", ")
            content = "```json\n" + json.dumps({k: "지적할 것 없음" for k in keys if k != "(none fixed)"}, ensure_ascii=False) + "\n```"
        else:
            content = "지적할 것 없음." if not self.bad_json else "not json {"
        n_in = est_input_tokens(params) // 2
        t = self.usage_total
        comp = t - n_in if (_int(t) and t >= n_in) else 40
        usage = None if self.no_usage else {"prompt_tokens": n_in, "completion_tokens": comp,
                                            "reasoning_tokens": min(30, comp),
                                            "total_tokens": t if t is not None else n_in + comp}
        raw = {"id": "chatcmpl-fake", "object": "chat.completion", "model": self.model_override or params["model"],
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content, "reasoning_content": "REASONING-SECRET"}}]}
        if usage is not None:
            raw["usage"] = usage
        if self.weight_version is not None:
            wv = "default" if self.weight_version == "default" else self.weight_version
            raw["metadata"] = {"weight_version": wv, "weight_versions": [{"start": 0, "end": 1, "version": wv}]}
        return {"status": self.status, "rate_limit": rate, "raw": raw, "body_error": None}


# ------------------------------------------------------------------------------------- plan
def is_exaone_plan(pd: Path) -> bool:
    try:
        return json.loads((Path(pd) / "plan.json").read_text(encoding="utf-8")).get("use") == USE
    except Exception:  # noqa: BLE001
        return False


def compute_plan_hash(manifest_sha256: str, hard_worst_tokens: int, max_total_tokens: int, send_list_sha256: str) -> str:
    """The approval key: manifest bytes, both token figures, the price marker AND the SEND_LIST.md the user reads."""
    return sha256_text(canon_json({"use": USE, "manifest_sha256": manifest_sha256, "hard_worst_tokens": int(hard_worst_tokens),
                                   "max_total_tokens": int(max_total_tokens), "price": PRICE,
                                   "send_list_sha256": send_list_sha256}))


def _resolve(builder: PackBuilder, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    for root in (builder.lab_root, builder.repo_root):
        if (Path(root) / p).is_file():
            return Path(root) / p
    raise Refused(f"exaone: file {rel!r} not found")


PATH_CHARS_RE = re.compile(r"^[!-+\--\[\]-_a-{}~]+$")   # R4-3: printable ASCII without space , | ` \\


def _rel_ok(rel) -> bool:
    """A doc_files rel must be a plain relative posix path inside its root (no '..', no absolute, no backslash) made of
    printable ASCII other than space, ',', '|', '`' (R4-3: no Unicode spaces, full-width forms or homoglyphs; a ',' or
    '|' could make the SEND_LIST tables read as two files). 2026-11-11 census: no lab file name needs more."""
    if not isinstance(rel, str) or not rel or "\\" in rel or rel.startswith("/") or not PATH_CHARS_RE.fullmatch(rel):
        return False
    parts = rel.split("/")
    return all(x not in ("", ".", "..") for x in parts)


def _shared(r: dict) -> dict:
    """Everything a row sends besides its doc texts: params except messages, output, JSON keys, instructions."""
    seg = r["segments"]
    return {"params": {k: v for k, v in r["params"].items() if k != "messages"}, "output": seg["output"],
            "json_required_keys": seg["json_required_keys"], "instructions": seg["instructions"]}


def check_rows(rows: list[dict]) -> None:
    """Every manifest row is tied to what SEND_LIST.md displays: doc segment i hashes to doc_files[i].sha256 (and its
    est_tokens / label / shown path agree), source_paths are exactly the doc_files rels (plain relative paths), and
    every row shares row 0's params-except-messages, output, json_required_keys and instructions (which SEND_LIST shows)."""
    base, seen = None, set()
    for r in rows:
        seg, meta = r.get("segments"), r.get("meta") or {}
        if not isinstance(seg, dict) or set(seg) != {"instructions", "docs", "output", "json_required_keys"} \
                or not isinstance(seg.get("docs"), list) or not isinstance(seg.get("instructions"), str) \
                or not isinstance(seg.get("json_required_keys"), list) or not isinstance(r.get("params"), dict):
            raise Refused(f"exaone: {r.get('custom_id')} has no valid segments; re-plan")
        cid, item, rep = r.get("custom_id"), meta.get("item"), meta.get("repeat")
        if not isinstance(cid, str) or not CID_RE.fullmatch(cid) or not isinstance(item, str) \
                or not ITEM_RE.fullmatch(item) or not _int(rep) or cid != f"exaone-{item}-r{rep}":
            raise Refused(f"exaone: custom_id {cid!r} / item {item!r} / repeat {rep!r} are not "
                          f"exaone-<item>-r<1..5> with item {ITEM_RE.pattern} (custom_id names raw/<id>.json); re-plan")
        if cid in seen:
            raise Refused(f"exaone: duplicate custom_id {cid} (would be sent twice); re-plan")
        seen.add(cid)
        df = meta.get("doc_files")
        if not isinstance(df, list) or len(df) != len(seg["docs"]) or not seg["docs"]:
            raise Refused(f"exaone: {r.get('custom_id')} doc_files do not match its segments; re-plan")
        for i, (t, f) in enumerate(zip(seg["docs"], df), 1):
            if not isinstance(f, dict) or f.get("root") not in ("lab", "repo") or not _rel_ok(f.get("rel")):
                raise Refused(f"exaone: {r.get('custom_id')} doc{i} has an invalid root/rel (must be a plain relative path)")
            if not isinstance(f.get("raw_sha256"), str) or not HEX64_RE.fullmatch(f["raw_sha256"]):
                raise Refused(f"exaone: {r.get('custom_id')} doc{i} has no raw-byte sha256; re-plan")
            if f.get("label") != f"doc{i}" or not isinstance(t, str) or sha256_text(t) != f.get("sha256") \
                    or f.get("est_tokens") != est_tokens(t) or f.get("path") != _shown(f["root"], f["rel"]):
                raise Refused(f"exaone: {r.get('custom_id')} doc{i} segment text is not the listed file's text (sha256)")
        sp = meta.get("source_paths")
        if isinstance(sp, list) and sp and sorted(set(sp)) != sorted({f["rel"] for f in df}):
            raise Refused(f"exaone: {r.get('custom_id')} source_paths differ from its listed doc files; re-plan")
        sh = canon_json(_shared(r))        # R4-7: canonical JSON, so 1 vs 1.0 vs -0.0 differ
        if base is None:
            base = sh
        elif sh != base:
            raise Refused(f"exaone: {r.get('custom_id')} differs from the first request in params / output / JSON keys / "
                          "instructions (SEND_LIST shows the first); re-plan")


def _shown(root: str, rel: str) -> str:
    return f"lab/{rel}" if root == "lab" else rel


def decode_listed(raw: bytes) -> str:
    """The text of a listed file as it is sent: UTF-8, CRLF and lone CR turned into LF (what Path.read_text does)."""
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def read_listed(builder: PackBuilder, f: dict, who: str) -> tuple[str, bytes]:
    """R4-1 / R4-2: read ONE listed file once and check it against its manifest entry -- plain relative rel, real path
    inside its root, deny path check on the listed AND the real path, root/rel re-derived from disk (builder._rel, batch
    dir first) equal to the recorded ones and under lab/ or the repo, L1 for the re-derived root, raw-byte sha256 and
    text sha256 equal to the manifest. Returns (text, raw bytes): the caller sends / shows exactly these."""
    if not _rel_ok(f.get("rel")) or f.get("root") not in ("lab", "repo"):
        raise Refused(f"{who}: {f.get('path')!r} is not a plain relative path under lab/ or the repo")
    root = Path(builder.lab_root if f["root"] == "lab" else builder.repo_root)
    fp = root / f["rel"]
    try:
        real_rel = fp.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        raise Refused(f"{who}: {f['path']!r} resolves outside its root")
    for p in dict.fromkeys([f["rel"], real_rel, _shown(f["root"], real_rel)]):   # in-root symlink to a denied path
        if builder.deny.path_hit(p) or builder.deny.matcher.matches(p):
            raise Refused(f"{who}: {f['path']!r} resolves to {p!r}, which matches the deny source")
    try:
        tag, rel2 = builder._rel(fp)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"{who}: {f['path']!r}: {e}")
    if (tag, rel2) != (f["root"], f["rel"]):
        raise Refused(f"{who}: {f['path']!r} is really {tag}:{rel2} (the manifest records {f['root']}:{f['rel']}; "
                      "a file inside the plan dir or behind a symlink is not a listable document)")
    kind = DATA_KIND.get(Path(f["rel"]).suffix.lower())
    if kind is None:
        raise Refused(f"{who}: {f['path']!r} has no allowed extension")
    builder._l1(kind, tag, rel2)                                  # L1 for the re-derived root
    try:
        raw = fp.read_bytes()
        text = decode_listed(raw)
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"{who}: {f['path']} unreadable ({type(e).__name__})")
    if sha256_bytes(raw) != f.get("raw_sha256") or sha256_text(text) != f.get("sha256"):
        raise Refused(f"{who}: {f['path']} does not match the listed file (raw sha256 {sha256_bytes(raw)[:12]} / text "
                      f"sha256 {sha256_text(text)[:12]} on disk vs listed {str(f.get('raw_sha256'))[:12]} / "
                      f"{str(f.get('sha256'))[:12]}; changed since plan, or the manifest text is not from it); "
                      "re-plan and re-read with `exaone show`")
    return text, raw


def listed_files(rows: list[dict]) -> list[dict]:
    """The distinct listed files in first-seen order (the SEND_LIST table order)."""
    out: dict[str, dict] = {}
    for r in rows:
        for f in r["meta"]["doc_files"]:
            out.setdefault(f["path"], f)
    return list(out.values())


def _fenced(text: str) -> list[str]:
    """A backtick fence longer than any backtick run in text, so the text cannot close it or spoof list structure, and
    a line-number gutter ('   1│ ') on every line, so a fake fence or heading inside the text is visibly inside the
    block in a raw view (cat / less / editor). The gutter is display only; it is not sent."""
    n = max([3] + [len(m) for m in re.findall(r"`+", text)]) + 1
    lines = text.split("\n")
    w = max(4, len(str(len(lines))))
    return ["`" * n + "text", *(f"{i:>{w}}{GUTTER}{ln}" for i, ln in enumerate(lines, 1)), "`" * n]


def _display_gate(name, rows: list[dict]) -> None:
    """R3-1: display_safe over everything SEND_LIST.md shows or stands for -- the plan name, every custom_id / item id,
    every listed doc rel / shown path, the instructions, every listed doc's text, the JSON keys, and row 0's params
    values (which the params line prints). Run at plan and again on every re-render (load_plan -> approve, run)."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise Refused(f"exaone display check: plan name {name!r} must match {NAME_RE.pattern}")
    display_safe(name, "name")
    p0 = rows[0].get("params")
    if not isinstance(p0, dict) or set(p0) - LANE_PARAMS or p0.get("model") != MODEL \
            or p0.get("reasoning_effort") not in EFFORTS or not _int(p0.get("max_tokens")) \
            or any(k in p0 and not _num(p0[k]) for k in ("temperature", "top_p")):
        raise Refused("exaone display check: row 0 params (model / reasoning_effort / max_tokens / temperature / top_p) "
                      "are not the lane's typed values")
    for r in rows:
        cid, seg, meta = r.get("custom_id"), r["segments"], r["meta"]
        display_safe(cid, "custom_id")
        display_safe(meta.get("item"), f"{cid} item id")
        display_safe(seg["instructions"], "instructions", multiline=True, prose=True)
        if seg["output"] not in OUTPUTS:
            raise Refused(f"exaone display check: {cid} output {seg['output']!r} is not text|json")
        for k in seg["json_required_keys"]:
            display_safe(k, f"{cid} json_required_keys")
            if not isinstance(k, str) or not KEY_RE.fullmatch(k):
                raise Refused(f"exaone display check: {cid} json_required_keys entry {k!r} must match {KEY_RE.pattern}")
        for i, (t, f) in enumerate(zip(seg["docs"], meta["doc_files"]), 1):
            display_safe(f.get("rel"), f"{cid} doc{i} rel")
            display_safe(f.get("path"), f"{cid} doc{i} path")
            display_safe(t, f"{cid} doc{i} text ({f.get('path')})", multiline=True, marker=True)


def render_send_list(name: str, rows: list[dict], hard_worst: int, max_total: int, manifest_sha256: str) -> str:
    """SEND_LIST.md, derived ONLY from the manifest rows (+ name and the two token figures, all bound by plan_hash).
    It shows every byte a request carries other than the listed files' texts: the system message verbatim and the user
    scaffolding verbatim. check_rows makes row 0 representative. load_plan re-renders it and refuses any byte difference.
    The display gate runs first on every render, so a plan dir carrying invisible / control text cannot be rendered."""
    _display_gate(name, rows)
    p0, s0 = rows[0]["params"], rows[0]["segments"]
    files: dict[str, dict] = {}
    for r in rows:
        for f in r["meta"]["doc_files"]:
            e = files.setdefault(f["path"], {"sha256": f["sha256"], "raw_sha256": f["raw_sha256"],
                                             "est_tokens": f["est_tokens"], "items": []})
            if e["sha256"] != f["sha256"] or e["raw_sha256"] != f["raw_sha256"]:
                raise Refused(f"exaone: {f['path']} listed with two different sha256 values")
            if r["meta"]["item"] not in e["items"]:
                e["items"].append(r["meta"]["item"])
    extra = {k: p0[k] for k in ("temperature", "top_p") if k in p0}
    system = render_messages(s0["instructions"], [], s0["output"], list(s0["json_required_keys"]))[0]["content"]
    L = [f"# SEND_LIST — {name} (exaone, 승인 전 확인용)", "",
         f"- 목적지: {DESTINATION}",
         f"- 모델: `{p0['model']}`, reasoning_effort `{p0['reasoning_effort']}`, max_tokens {p0['max_tokens']} (추론 토큰 포함)"
         + (f", 그 밖의 파라미터 {json.dumps(extra, sort_keys=True)}" if extra else ""),
         f"- 요청 수 {len(rows)}건 (관측된 일일 한도 {DAY_REQUEST_LIMIT_OBSERVED}건/일, {DAY_LIMIT_SOURCE})",
         f"- 요청 구성: item × repeats, 출력: {s0['output']}"
         + (f", 필수 JSON 키: {', '.join(s0['json_required_keys'])}" if s0["json_required_keys"] else ""),
         "- 모든 요청이 위 파라미터, 아래 system 메시지, 아래 틀을 공유한다(다르면 plan/approve/run이 거부).",
         f"- 최악 토큰(입력 추정 + max_tokens 합): **{hard_worst}** / 상한 max_total_tokens {max_total} (run 1회당)",
         f"- 가격: 미공개(price: {PRICE}). 비용 게이트는 토큰으로만 건다.",
         f"- manifest sha256: `{manifest_sha256}` (승인 키 plan_hash가 이 파일 SEND_LIST.md의 sha256도 함께 묶는다)", "",
         "## 기기 밖으로 나가는 파일", "",
         "`raw sha256`은 파일 바이트 그대로의 값(`sha256sum <경로>`로 같은 값), `text sha256`은 보내는 텍스트(UTF-8, CRLF·CR→LF)의 "
         "값이다. 문서는 경로를 따로 열지 말고 `exaone show <plan_dir> | less`로 읽는다(한 번 읽어 두 sha를 확인한 바로 그 바이트를 보여준다).", "",
         "| 경로 | raw sha256 | text sha256 | est tokens | items |", "|---|---|---|---|---|"]
    for path, f in files.items():
        L.append(f"| `{path}` | `{f['raw_sha256']}` | `{f['sha256']}` | {f['est_tokens']} | {', '.join(f['items'])} |")
    L += ["", "## 파일 외에 함께 나가는 텍스트 (전문)", "",
          "아래 블록의 각 줄 앞 `   N│ `은 표시용 줄 번호이고 전송되지 않는다(그 뒤가 전송되는 바이트). "
          "보이지 않는·제어 문자는 plan/approve/run이 거부한다. 이 파일은 마크다운 미리보기가 아니라 `less`나 편집기로 읽는다.", "",
          "위 표의 문서는 HTML 주석(`<!-- … -->`)을 포함해 전문이 그대로 나간다. 마크다운 미리보기는 주석을 숨기므로 문서는 "
          "`exaone show <plan_dir> | less`로 원문을 읽는다.", "",
          f"### system 메시지 전문 (sha256 `{sha256_text(system)}`, est tokens {est_tokens(system)})", "",
          *_fenced(system), "",
          "### user 메시지 틀 (코드 고정; `{n}`은 문서 수, `{label}`은 doc1..docN, `{text}` 자리에 위 표의 파일 텍스트가 들어간다)", "",
          *_fenced(USER_HEAD + DOC_BLOCK), "",
          "파일명·경로는 요청에 들어가지 않는다.",
          "", "## 요청별", "", "| custom_id | docs (순서대로 doc1..) | worst tokens |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['custom_id']} | {', '.join(x['path'] for x in r['meta']['doc_files'])} | {request_worst_tokens(r['params'])} |")
    L += ["", "출력은 참고용(advisory)이다: verifier rung이 아니고 인용 가능한 근거가 아니다(seed 없음, 모델이 예고 없이 바뀔 수 있음).", ""]
    return "\n".join(L)


def _day_cap(n: int, who: str) -> None:
    if n > DAY_REQUEST_LIMIT_OBSERVED:
        raise Refused(f"exaone {who} refused: {n} requests > observed daily limit {DAY_REQUEST_LIMIT_OBSERVED}/day "
                      f"({DAY_LIMIT_SOURCE}); put several docs into one item (fewer, larger requests; context "
                      f"{CONTEXT_TOKENS} tokens) or split the spec")


def plan(spec_path: Path, builder: PackBuilder, ledger: Ledger) -> dict:
    from . import manifest as M
    from .ops import _pack_py_sha
    spec_path = Path(spec_path)
    pd = spec_path.parent
    spec = load_spec(spec_path)
    builder.batch_dir = pd
    output, keys = spec.get("output", "text"), list(spec.get("json_required_keys") or [])
    scanned = [builder.build_spec_text("exaone-spec", spec_path.read_text(encoding="utf-8"))]
    ipack = builder.build_spec_text("exaone-instructions", spec["instructions"])
    for i, k in enumerate(keys):
        builder.build_spec_text(f"exaone-key{i}", k)
    rows = []
    for it in spec["items"]:
        docs, src_paths, doc_files = [], [], []
        for i, rel in enumerate(it["docs"], 1):
            src = _resolve(builder, rel)
            kind = DATA_KIND.get(src.suffix.lower())
            if kind is None:
                raise Refused(f"exaone: doc {rel!r} has no allowed extension ({sorted(DATA_KIND)})")
            raw = src.read_bytes()
            try:
                text = decode_listed(raw)
            except UnicodeDecodeError as e:
                raise Refused(f"exaone: doc {rel!r} is not valid UTF-8 (byte {e.start}); send a UTF-8 copy")
            pk = builder.build_verify_only(kind, src.name, text, origin=src)
            tag, prel = builder._rel(src)
            if tag not in ("lab", "repo"):
                raise Refused(f"exaone: doc {rel!r} must be under lab/ or the repo")
            docs.append(pk)
            src_paths += [x["path"] for x in pk.sources]
            if pk.pack_sha256 != sha256_text(pk.text):
                raise Refused(f"exaone: doc {rel!r}: Pack sha256 is not the sha256 of its text")
            doc_files.append({"label": f"doc{i}", "root": tag, "rel": prel, "path": _shown(tag, prel),
                              "sha256": pk.pack_sha256, "raw_sha256": sha256_bytes(raw), "est_tokens": est_tokens(pk.text)})
            scanned.append(pk)
        params = build_params(spec, ipack, docs)
        validate_params(params, ipack, docs, output, keys)
        for j in range(1, spec.get("repeats", 1) + 1):
            cid = f"exaone-{it['id']}-r{j}"
            if not M.CUSTOM_ID_RE.fullmatch(cid):
                raise Refused(f"exaone: custom_id {cid!r} invalid")
            rows.append({"custom_id": cid, "params": params, "params_sha256": sha256_text(canon_json(params)),
                         "pack_sha256s": [p.pack_sha256 for p in (ipack, *docs)],
                         "segments": {"instructions": ipack.text, "docs": [p.text for p in docs], "output": output,
                                      "json_required_keys": keys},
                         "meta": {"item": it["id"], "repeat": j, "source_paths": sorted(set(src_paths)), "doc_files": doc_files}})
    scanned.append(ipack)
    _day_cap(len(rows), "plan")
    check_rows(rows)
    _display_gate(spec["name"], rows)      # before anything is written (render_send_list re-runs it)
    worst = sum(request_worst_tokens(r["params"]) for r in rows)
    if worst > spec["max_total_tokens"]:
        raise Refused(f"exaone plan refused: hard-worst {worst} tokens > max_total_tokens {spec['max_total_tokens']}")
    (pd / "pack").mkdir(exist_ok=True)
    seen = {p.pack_sha256: p for p in scanned}
    for p in seen.values():
        (pd / "pack" / f"{p.kind}-{p.pack_sha256[:8]}.txt").write_text(p.text, encoding="utf-8")
    data = ("\n".join(canon_json(r) for r in rows) + "\n").encode("utf-8")
    (pd / "manifest.jsonl").write_bytes(data)
    msha = sha256_text(data.decode("utf-8"))
    sl = render_send_list(spec["name"], rows, worst, spec["max_total_tokens"], msha)
    (pd / "SEND_LIST.md").write_bytes(sl.encode("utf-8"))
    slsha = sha256_text(sl)
    plan_hash = compute_plan_hash(msha, worst, spec["max_total_tokens"], slsha)
    d = {"name": spec["name"], "use": USE, "model": spec["model"], "reasoning_effort": spec["reasoning_effort"],
         "max_tokens": spec["max_tokens"], "output": output, "json_required_keys": keys, "n_requests": len(rows),
         "hard_worst_tokens": worst, "max_total_tokens": spec["max_total_tokens"],
         "est_usd_hard_worst": 0.0, "price": PRICE,
         "price_basis": "no price published (docs.exaone.ai 2026-11-10); gated in tokens: sum(est_input + max_tokens)",
         "manifest_sha256": msha, "plan_hash": plan_hash, "spec_sha256": sha256_file(spec_path),
         "pack_sha256": sha256_text(",".join(sorted(seen))), "deny_sha256": builder.deny.sha256,
         "pack_py_sha256": _pack_py_sha(),
         "requests": [{"custom_id": r["custom_id"], "worst_tokens": request_worst_tokens(r["params"])} for r in rows],
         "egress": DESTINATION, "advisory": True, "send_list": "SEND_LIST.md", "send_list_sha256": slsha,
         "approve_cmd": f"approve {pd} --tokens {worst}", "day_request_limit_observed": DAY_REQUEST_LIMIT_OBSERVED}
    (pd / "plan.json").write_text(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    ledger.append("planned", lane=USE, batch_dir=str(pd), manifest_sha256=msha, plan_hash=plan_hash, n_requests=len(rows),
                  est_usd_hard_worst=0.0, price=PRICE, hard_worst_tokens=worst, max_total_tokens=spec["max_total_tokens"],
                  send_list_sha256=slsha, model=spec["model"], deny_sha256=d["deny_sha256"])
    return d


def load_plan(pd: Path) -> tuple[dict, list[dict], int]:
    """Re-derive everything the approval key covers; refuse on any mismatch (manifest, params, segment<->listed file,
    token figures, SEND_LIST.md bytes, plan_hash)."""
    pd = Path(pd)
    try:
        d = json.loads((pd / "plan.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Refused(f"exaone: {pd / 'plan.json'} unreadable ({type(e).__name__}); re-plan")
    if not isinstance(d, dict) or d.get("use") != USE:
        raise Refused(f"{pd} is not an exaone plan")
    try:
        text = (pd / "manifest.jsonl").read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"exaone: manifest.jsonl unreadable ({type(e).__name__}); re-plan")
    msha = sha256_text(text)
    if msha != d.get("manifest_sha256"):
        raise Refused("exaone: manifest.jsonl no longer matches plan.json (edited after plan)")
    try:   # split on "\n" only (as written by plan): U+2028 / U+0085 / VT / FF inside a JSON string must not split a row
        rows = [json.loads(ln) for ln in text.split("\n") if ln]
    except json.JSONDecodeError as e:
        raise Refused(f"exaone: manifest.jsonl has a line that is not JSON ({e.msg}, line {e.lineno}); re-plan")
    if not rows:
        raise Refused("exaone: empty manifest")
    _day_cap(len(rows), "plan load")
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("params"), dict) or not isinstance(r.get("meta"), dict):
            raise Refused("exaone: a manifest row is not {custom_id, params, segments, meta, ...}; re-plan")
        if sha256_text(canon_json(r["params"])) != r.get("params_sha256"):
            raise Refused(f"exaone: {r.get('custom_id')} params_sha256 mismatch")
    check_rows(rows)
    worst = sum(request_worst_tokens(r["params"]) for r in rows)
    mtt = d.get("max_total_tokens")
    if not _int(mtt):
        raise Refused("exaone: plan.json max_total_tokens is not an integer (tampered)")
    if worst > mtt:
        raise Refused(f"exaone: hard-worst {worst} tokens > max_total_tokens {mtt}")
    sl = render_send_list(d.get("name"), rows, worst, mtt, msha)
    try:
        on_disk = (pd / "SEND_LIST.md").read_bytes()
    except OSError:
        on_disk = None
    if on_disk != sl.encode("utf-8") or sha256_text(sl) != d.get("send_list_sha256"):
        raise Refused("exaone: SEND_LIST.md does not match the manifest (edited, or the manifest changed after plan); re-plan")
    if compute_plan_hash(msha, worst, mtt, d["send_list_sha256"]) != d.get("plan_hash") or worst != d.get("hard_worst_tokens"):
        raise Refused("exaone: plan_hash / hard_worst_tokens / max_total_tokens in plan.json do not match the manifest (tampered)")
    return d, rows, worst


def _done_ids(ledger: Ledger, key: str) -> set:
    return {x.get("custom_id") for x in ledger.rows("sync_done", manifest_sha256=key, lane=USE) if x.get("rc") == 0}


def pending_rows(pd: Path, ledger: Ledger) -> tuple[dict, list[dict]]:
    d, rows, _w = load_plan(pd)
    done = _done_ids(ledger, d["plan_hash"])
    return d, [r for r in rows if r["custom_id"] not in done]


def sync_plan_for_approve(pd: Path, ledger: Ledger | None = None, loaded: tuple | None = None) -> dict:
    """The approval facts of ONE load_plan snapshot (pass `loaded` to reuse the caller's snapshot, R5-1)."""
    d, rows, worst = loaded if loaded is not None else load_plan(pd)
    if ledger is not None and not [r for r in rows if r["custom_id"] not in _done_ids(ledger, d["plan_hash"])]:
        raise Refused(f"exaone: nothing to do -- every request of {pd} is already done; no approval needed")
    return {"plan_hash": d["plan_hash"], "hard_worst_tokens": worst, "max_total_tokens": d["max_total_tokens"],
            "send_list_sha256": d["send_list_sha256"], "label": f"{USE}:{pd}"}


def approve(pd: Path, tokens, ledger: Ledger, *, builder: PackBuilder, env=None) -> dict:
    """User-only token approval. Refuses inside a Claude Code session BEFORE reading anything. Re-reads every listed
    file (read_listed: raw and text sha256, root, L1, deny path) so a change between plan and approve is refused here
    (and one between approve and run is refused by run). The row has the sync-approval shape ops._approved matches
    (lane sync, by user-cli, approval_id) plus approval.tokens and the per-file raw sha256s it checked."""
    marker = in_claude_code_session(env)
    if marker:
        raise Refused(f"approve refused: running inside a Claude Code session ({marker} is set). "
                      f"Run it yourself from your own terminal.")
    loaded = load_plan(pd)                # R5-1: ONE snapshot -- key, SEND_LIST sha, token figure and the files re-read
    sp = sync_plan_for_approve(pd, ledger, loaded=loaded)
    _d, rows, _w = loaded
    builder.batch_dir = Path(pd)
    files = []
    for f in listed_files(rows):
        read_listed(builder, f, "exaone approve")
        files.append({"path": f["path"], "raw_sha256": f["raw_sha256"]})
    if not _int(tokens) or tokens < sp["hard_worst_tokens"]:
        raise Refused(f"approve refused: an exaone plan is approved in tokens (price {PRICE}); "
                      f"--tokens {tokens} < hard-worst {sp['hard_worst_tokens']} (read {Path(pd) / 'SEND_LIST.md'} first)")
    ts = now()
    return ledger.append("approved", batch_dir=sp["label"], manifest_sha256=sp["plan_hash"], est_usd_hard_worst=0.0,
                         est_tokens_hard_worst=sp["hard_worst_tokens"], price=PRICE, send_list_sha256=sp["send_list_sha256"],
                         approval={"by": "user-cli", "usd": 0.0, "tokens": tokens, "manifest_sha256": sp["plan_hash"], "ts": ts},
                         ts=ts, lane="sync", approval_id=secrets.token_hex(6), files=files)


def show(pd: Path, builder: PackBuilder) -> str:
    """Offline, read-only, no ledger row, no key (allowed inside a session): SEND_LIST.md (verified against the manifest
    by load_plan) followed by every listed document, each read ONCE into memory, checked against its manifest raw and
    text sha256 (Refused on mismatch), and printed from that same read with the line gutter and both sha256s. What the
    user reads here is exactly the text that was hashed; approve and run refuse any later change on disk."""
    pd = Path(pd)
    d, rows, worst = load_plan(pd)        # one snapshot: the shas printed and the files read come from it
    builder.batch_dir = pd
    sl = render_send_list(d["name"], rows, worst, d["max_total_tokens"], d["manifest_sha256"])
    out = [f"SEND_LIST.md sha256 {sha256_text(sl)}  plan_hash {d['plan_hash'][:12]}  "
           "(approve prints the same two values; compare them)", "", sl, "",
           "# 문서 전문 (exaone show)", "",
           "각 파일을 한 번 읽어 raw·text sha256이 위 표와 같은지 확인한 뒤, 그 같은 읽기의 텍스트를 보여준다. 이 텍스트가 "
           "보내는 텍스트다. 줄 앞 `   N│ `은 표시용 줄 번호이고 전송되지 않는다.", ""]
    for f in listed_files(rows):
        text, raw = read_listed(builder, f, "exaone show")
        display_safe(text, f"{f['path']} text", multiline=True, marker=True)
        out += [f"## {f['path']} (raw sha256 `{sha256_bytes(raw)}`, text sha256 `{sha256_text(text)}`, {len(raw)} bytes)",
                "", *_fenced(text), ""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------- run
def _unencodable(obj, depth: int = 0) -> bool:
    """True if any string (key or value) anywhere in a reply cannot be encoded as UTF-8 (a lone surrogate, e.g. from a
    JSON '\\ud800' escape): such text would make the raw / results / ledger writes fail."""
    if depth > 200:
        return True
    if isinstance(obj, str):
        try:
            obj.encode("utf-8")
        except UnicodeEncodeError:
            return True
        return False
    if isinstance(obj, dict):
        return any(_unencodable(k, depth + 1) or _unencodable(v, depth + 1) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(_unencodable(v, depth + 1) for v in obj)
    return False


UNENCODABLE = "reply contains unencodable text (lone surrogate); raw saved with \\u escapes"


def _safe_text(s: str) -> str:
    return s.encode("utf-8", "backslashreplace").decode("utf-8")


def _client_ready(client) -> bool:
    ready = getattr(client, "ready", None)
    return bool(ready()) if callable(ready) else True


def _evaluate(got, params: dict, seg: dict):
    """-> (rc, err, norm, usage, consumed_tokens|None, halt)."""
    if not isinstance(got, dict):
        return 4, f"send result is not an object ({type(got).__name__})", None, None, None, True
    if _unencodable(got):                            # content / reasoning / model / usage / headers: halt, charge worst
        return 4, UNENCODABLE, None, None, None, True
    status = got.get("status")
    if status == 429:
        rl = got.get("rate_limit") if isinstance(got.get("rate_limit"), dict) else {}
        return 3, f"HTTP 429 rate limited (retry-after {rl.get('retry-after')})", None, None, None, True
    if not _int(status) or not 200 <= status < 300:
        return 4, f"HTTP {status!r}: {str(got.get('body_error'))[:300]}", None, None, None, True
    raw = got.get("raw")
    if not isinstance(raw, dict):
        return 4, f"response is not a JSON object ({type(raw).__name__})", None, None, None, True   # model unchecked: halt
    usage = raw.get("usage")
    consumed = usage_tokens(usage)
    if raw.get("model") != params["model"]:          # checked first: a changed model halts whatever its usage looks like
        return 2, f"model mismatch: requested {params['model']}, returned {raw.get('model')!r}", None, \
            (usage if consumed is not None else None), consumed, True
    if consumed is None:                             # the token cap cannot be enforced without usage: halt
        return 4, f"usage missing or malformed: {str(usage)[:200]}", None, None, None, True
    if consumed == 0:                                # an under-reporting server would never trip the cap: halt
        return 4, "usage reports 0 tokens", None, None, None, True
    try:
        norm = normalize(raw, seg["output"], seg["json_required_keys"])
    except Exception as e:  # noqa: BLE001
        return 4, f"{type(e).__name__}: {str(e)[:300]}", None, usage, consumed, False
    return 0, None, norm, usage, consumed, False


def last_day_remaining(ledger: Ledger) -> tuple[int | None, str | None]:
    """(x-day-rate-limit-remaining, ts) from the newest exaone / exaone-probe sync_done that recorded it, else (None, None)."""
    for row in reversed(ledger.rows("sync_done")):
        if row.get("lane") in (USE, PROBE_LANE):
            v = day_remaining(row.get("rate_limit"))
            if v is not None:
                return v, row.get("ts")
    return None, None


def run(pd: Path, ledger: Ledger, client, builder: PackBuilder, *, sleep=None, clock=None, warn=None) -> dict:
    """Send the approved plan, one request at a time, at least MIN_SEND_INTERVAL_S apart (start to start). No retry:
    429 is rc 3 + halt; a reply whose x-day-rate-limit-remaining is 0 halts cleanly before the next send."""
    from .ops import _approved, tests_passed_current
    sleep, clock = sleep or _sleep, clock or _clock
    warn = warn or (lambda m: print(m, file=sys.stderr))
    pd = Path(pd)
    d, rows, worst = load_plan(pd)
    key = d["plan_hash"]
    done = _done_ids(ledger, key)
    pending = [r for r in rows if r["custom_id"] not in done]
    if not pending:
        return {"plan_hash": key, "nothing_to_do": True, "requests": [], "halted": None}
    if not _client_ready(client):                    # before any ledger row: a forgotten key wrapper consumes nothing
        raise Refused("exaone run: EXAONE_API_KEY is not set -- run under tools/api_batch/bin/with-exaone-key "
                      "(nothing was sent, the approval is not consumed)")
    ok, why = tests_passed_current(ledger)
    if not ok:
        raise Refused(f"exaone run lapses: {why}")
    appr = _approved(ledger, key, 0.0, fresh=True, lane="sync")
    tok = (appr.get("approval") or {}).get("tokens")
    if not _int(tok) or tok < worst:
        raise Refused(f"exaone: approval records tokens {tok!r} < plan hard-worst {worst} (approve with --tokens)")
    builder.batch_dir = pd
    checked = []
    for r in pending:          # deny source may have grown since plan: rescan text, identifiers, rendered messages, paths
        params, seg = r["params"], r.get("segments")
        if not isinstance(seg, dict) or set(seg) != {"instructions", "docs", "output", "json_required_keys"} \
                or not isinstance(seg["docs"], list) or not isinstance(seg["json_required_keys"], list):
            raise Refused(f"exaone run: {r['custom_id']} has no valid segments; re-plan")
        ip = builder.build_spec_text(f"{r['custom_id']}-instructions", seg["instructions"])
        dps = [builder.build_spec_text(f"{r['custom_id']}-doc{i}", t) for i, t in enumerate(seg["docs"], 1)]
        for i, k in enumerate(seg["json_required_keys"]):
            builder.build_spec_text(f"{r['custom_id']}-key{i}", k)
        for m in params.get("messages") or []:
            builder.build_spec_text(f"{r['custom_id']}-{m.get('role')}", m.get("content") or "")
        validate_params(params, ip, dps, seg["output"], seg["json_required_keys"])
        paths = r["meta"].get("source_paths")
        if not isinstance(paths, list) or not paths:
            raise Refused(f"exaone run: {r['custom_id']} has no recorded source paths; re-plan")
        rels = [f["rel"] for f in r["meta"]["doc_files"]]
        if sorted(set(paths)) != sorted(set(rels)):              # check_rows did too; the rescan below walks the listed rels
            raise Refused(f"exaone run: {r['custom_id']} source_paths differ from its listed doc files; re-plan")
        for p in rels:
            if builder.deny.path_hit(p) or builder.deny.matcher.matches(p):
                raise Refused(f"exaone run: source {p!r} now matches the deny source (added after plan)")
        for f, t in zip(r["meta"]["doc_files"], seg["docs"]):   # the text sent must be the listed file's current text
            text, _raw = read_listed(builder, f, "exaone run")
            if text != t:
                raise Refused(f"exaone run: {f['path']} text differs from the manifest segment; re-plan")
        checked.append(r)
    for sub in ("raw", "out"):
        (pd / sub).mkdir(exist_ok=True)
    aid = approval_id(appr)
    budget = min(int(d["max_total_tokens"]), tok)
    results, halted, spent, warnings = [], None, 0, []
    known, known_ts = last_day_remaining(ledger)
    if known is None:
        warnings.append(f"exaone run: daily requests remaining unknown (no earlier reply recorded {DAY_REMAINING_HDR}); "
                        f"observed limit {DAY_REQUEST_LIMIT_OBSERVED}/day ({DAY_LIMIT_SOURCE}); sending "
                        f"{len(checked)} request(s), halting if a reply reports 0 remaining")
    elif known < len(checked):
        warnings.append(f"exaone run: last recorded {DAY_REMAINING_HDR} was {known} (at {known_ts}) < {len(checked)} "
                        f"pending request(s); the run halts when a reply reports 0 remaining")
    for m in warnings:
        warn(m)
    last_start = None
    for idx, r in enumerate(checked):
        cid, params, seg = r["custom_id"], r["params"], r["segments"]
        w = request_worst_tokens(params)
        if spent + w > budget:
            halted = f"next request {cid} could exceed the token cap ({spent} + {w} > {budget})"
            break
        if last_start is not None:                   # pacing: start-to-start >= MIN_SEND_INTERVAL_S, before sync_sent
            wait = MIN_SEND_INTERVAL_S - (clock() - last_start)
            if wait > 0:
                sleep(wait)
        ledger.append("sync_sent", lane=USE, batch_dir=str(pd), plan_hash=key, manifest_sha256=key, custom_id=cid,
                      params_sha256=r["params_sha256"], approval_id=aid, worst_tokens=w, est_usd_hard_worst=0.0)
        last_start = clock()
        try:
            got = client.send(params)
        except Exception as e:  # noqa: BLE001
            err = _safe_text(f"{type(e).__name__}: {str(e)[:200]}")
            spent += w                                       # spend unknown: charge the request's worst case
            ledger.append("sync_done", lane=USE, plan_hash=key, manifest_sha256=key, custom_id=cid, rc=1, error=err,
                          tokens_charged=w)
            results.append({"custom_id": cid, "rc": 1})
            halted = f"send failed for {cid}: {err}"
            break
        # every path below ends in exactly one sync_done for this sync_sent (local I/O errors included: rc 4 + halt)
        rc, err, norm, usage, consumed, halt = _evaluate(got, params, seg)
        got_d = got if isinstance(got, dict) and err != UNENCODABLE else {}   # unencodable: nothing of it reaches the ledger
        raw = got_d.get("raw") if isinstance(got_d.get("raw"), dict) else {}
        # usage that failed usage_tokens: charge the worst case, or more if the server reported more
        charged = consumed if consumed is not None else max(w, _reported_tokens(raw.get("usage")))
        spent += charged
        wv = weight_version(raw)
        rate = got_d.get("rate_limit") if isinstance(got_d.get("rate_limit"), dict) else {}
        status = got_d.get("status") if _int(got_d.get("status")) else None
        ts = now()
        try:
            (pd / "raw" / f"{cid}.json").write_text(json.dumps(got, ensure_ascii=True, sort_keys=True, indent=1, default=repr),
                                                    encoding="utf-8")   # ascii: a lone surrogate cannot lose the response
            if norm is not None:
                line = json.dumps({"custom_id": cid, "item": r["meta"]["item"], "repeat": r["meta"]["repeat"],
                                   "advisory": True, "plan_hash": key, "params_sha256": r["params_sha256"],
                                   "model": raw.get("model"), "ts": ts, "output": seg["output"],
                                   "content": norm["content"], "parsed": norm["parsed"],
                                   "finish_reason": norm["finish_reason"], "reasoning_tokens": norm["reasoning_tokens"],
                                   "usage": usage, "prompt_tokens": usage.get("prompt_tokens"),
                                   "est_input_tokens": est_input_tokens(params), "weight_version": wv,
                                   "http_status": status, "rate_limit": rate},
                                  ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                with (pd / "out" / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:  # noqa: BLE001  -- the response is kept only if both writes succeed
            rc, halt = 4, True
            err = f"local write failed: {type(e).__name__}: {str(e)[:200]}" + (f" (after: {err})" if err else "")
        ledger.append("sync_done", lane=USE, plan_hash=key, manifest_sha256=key, custom_id=cid, rc=rc, error=err,
                      http_status=status, model_returned=raw.get("model") if isinstance(raw.get("model"), (str, type(None))) else repr(raw.get("model"))[:100],
                      usage=usage, tokens_charged=charged, rate_limit=rate, weight_version=wv,
                      transport=getattr(client, "sdk_version", None))
        results.append({"custom_id": cid, "rc": rc})
        if halt:
            halted = err
            break
        if day_remaining(rate) == 0 and idx + 1 < len(checked):    # quota: stop cleanly, never send into a 429
            halted = (f"daily request limit reached ({DAY_REMAINING_HDR} 0 after {cid}); "
                      f"{len(checked) - idx - 1} request(s) not sent -- run again after the daily reset (new approval)")
            break
    return {"plan_hash": key, "approval_id": aid, "tokens_spent": spent, "token_budget": budget, "requests": results,
            "halted": halted, "nothing_to_do": False, "warnings": warnings}


# ------------------------------------------------------------------------------------- probe
def probe_params(try_response_format: bool = False) -> dict:
    p = {"model": MODEL, "messages": [{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": PROBE_USER}],
         "max_tokens": PROBE_MAX_TOKENS, "reasoning_effort": "low"}
    if try_response_format:
        p["response_format"] = {"type": "json_object"}      # undocumented: probe only, never in a plan
    return p


def probe(ledger: Ledger, client, out_path: Path, try_response_format: bool = False, env=None) -> dict:
    """One fixed code-controlled ping (no lab text). User-only: refused inside a Claude Code session."""
    marker = in_claude_code_session(env)
    if marker:
        raise Refused(f"exaone probe refused: running inside a Claude Code session ({marker} is set). "
                      f"Run it yourself from your own terminal.")
    if not _client_ready(client):
        raise Refused("exaone probe: EXAONE_API_KEY is not set -- run under tools/api_batch/bin/with-exaone-key "
                      "(nothing was sent or recorded)")
    params = probe_params(try_response_format)
    key = sha256_text(canon_json(params))
    ledger.append("sync_sent", lane=PROBE_LANE, plan_hash=key, manifest_sha256=key, params_sha256=key,
                  try_response_format=bool(try_response_format), est_usd_hard_worst=0.0)
    summary = {"ts": now(), "try_response_format": bool(try_response_format), "pinned_model": MODEL}
    got = None
    try:
        got = client.send(copy.deepcopy(params))
    except Exception as e:  # noqa: BLE001
        summary.update(rc=1, send_error=_safe_text(f"{type(e).__name__}: {str(e)[:300]}"))
    if got is not None and _unencodable(got):
        summary.update(rc=4, body_error=UNENCODABLE)
    elif got is not None:
        try:
            g = got if isinstance(got, dict) else {}
            raw = g.get("raw") if isinstance(g.get("raw"), dict) else None
            status = g.get("status")
            ch = (raw or {}).get("choices")
            c0 = ch[0] if isinstance(ch, list) and ch and isinstance(ch[0], dict) else {}
            msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
            usage = (raw or {}).get("usage")
            summary.update(
                http_status=status, model=(raw or {}).get("model"), model_matches_pin=(raw or {}).get("model") == MODEL,
                top_level_keys=sorted(raw) if raw else None, message_keys=sorted(msg), finish_reason=c0.get("finish_reason"),
                content=(msg.get("content") or "")[:200] if isinstance(msg.get("content"), (str, type(None))) else repr(msg.get("content"))[:200],
                has_reasoning_content="reasoning_content" in msg,
                usage_keys=sorted(usage) if isinstance(usage, dict) else None, usage=usage,
                usage_tokens_ok=bool(usage_tokens(usage)), rate_limit=g.get("rate_limit"),
                weight_version=weight_version(raw),
                body_error=(str(g.get("body_error"))[:500] if g.get("body_error") else None),
                response_format_accepted=(isinstance(status, int) and 200 <= status < 300) if try_response_format else None,
                rc=0 if (_int(status) and 200 <= status < 300) else (3 if status == 429 else 4))
        except Exception as e:  # noqa: BLE001  -- malformed reply: still one sync_done
            summary.update(rc=4, body_error=_safe_text(f"response shape unreadable: {type(e).__name__}: {str(e)[:200]}"))
    out_path = Path(out_path)
    write_err = None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "response": got}, ensure_ascii=True, indent=1, sort_keys=True,
                                       default=repr), encoding="utf-8")
    except Exception as e:  # noqa: BLE001  -- the ping was sent: close sync_sent first, then report
        write_err = f"local write failed: {type(e).__name__}: {str(e)[:200]}"
        summary["write_error"] = write_err
    ledger.append("sync_done", lane=PROBE_LANE, plan_hash=key, manifest_sha256=key, rc=summary.get("rc", 4),
                  error=write_err or summary.get("send_error") or summary.get("body_error"),
                  http_status=summary.get("http_status"), model_returned=summary.get("model"), usage=summary.get("usage"),
                  rate_limit=summary.get("rate_limit") if isinstance(summary.get("rate_limit"), dict) else None,
                  weight_version=summary.get("weight_version"), out=None if write_err else str(out_path))
    if write_err:
        raise Refused(f"exaone probe: response not saved ({write_err}); summary: {json.dumps(summary, default=repr)[:1500]}")
    return summary
