"""Shared pieces of tools/api_agents (Claude Managed Agents lane).

Control plane (sync.py): version-controlled agent / environment definitions under defs/ are pushed to the
API project and their ids + versions are recorded in lock.json.
Data plane (run.py): one run directory per task under lab/38_api-batch/14_managed-agents/runs/.

Every free-text field and file we send (agent system prompt, name and description; environment name and
description; run name; kickoff text; the run's shared worker text; mounted files) passes the api_batch verify-only
scan (deny sources WITHHELD_RUNS.yaml + API_BATCH_DENY.yaml, secrets) and an IPv4-literal check first; a hit
refuses, never masks.
Not scanned: structured values we write ourselves (environment config booleans, model ids, metadata keys).
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import yaml

from tools.api_batch.common import LAB_ROOT, REPO_ROOT, Halt, canon_json, est_tokens, sha256_bytes  # noqa: F401

PKG_DIR = Path(__file__).resolve().parent
DEFS_DIR = PKG_DIR / "defs"
TASKS_DIR = PKG_DIR / "tasks"
LOCK_PATH = PKG_DIR / "lock.json"
STORE = LAB_ROOT / "38_api-batch" / "14_managed-agents"
RUNS_DIR = STORE / "runs"
LEDGER_PATH = STORE / "ledger.jsonl"

SDK_PIN = "anthropic==2.8.0"
TOOLSET = "agent_toolset_20260401"
TOOL_NAMES = ("bash", "read", "write", "edit", "glob", "grep", "web_fetch", "web_search")
MULTIAGENT_TYPE = "multiagent_20261001"
METADATA_KEY = "trm_key"          # agents/environments carry metadata {trm_key: <defs key>} so a lost lock can be re-adopted

# A run whose spec marks a context file `inline: true` gives its workers one shared text (context files + the
# task's rules block, built by run.prepare as worker_system.md). It is appended to each rostered worker's system
# prompt below this line, so every thread of the run shares one cached prefix.
WORKER_SYSTEM_SEP = "\n========== Shared material for this run ==========\n\n"

IPV4 = re.compile(r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])")
IPV4_OK = {"0.0.0.0", "127.0.0.1"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ definitions
def split_md(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise Halt("agent definition must start with a YAML frontmatter block")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise Halt("agent definition frontmatter is not closed")
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:].strip() + "\n"


def load_agent_defs(defs_dir: Path = DEFS_DIR) -> dict[str, dict]:
    out = {}
    for p in sorted((defs_dir / "agents").glob("*.md")):
        fm, body = split_md(p.read_text(encoding="utf-8"))
        key = fm.get("key")
        if not key or key != p.stem:
            raise Halt(f"{p}: frontmatter key must equal the file stem")
        for k in ("name", "description", "model", "tools"):
            if k not in fm:
                raise Halt(f"{p}: missing {k!r}")
        enable = fm["tools"].get("enable") or []
        bad = [t for t in enable if t not in TOOL_NAMES]
        if bad:
            raise Halt(f"{p}: unknown tools {bad}")
        out[key] = {**fm, "system": body, "path": str(p)}
    return out


def load_env_defs(defs_dir: Path = DEFS_DIR) -> dict[str, dict]:
    out = {}
    for p in sorted((defs_dir / "environments").glob("*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        if d.get("key") != p.stem:
            raise Halt(f"{p}: key must equal the file stem")
        out[d["key"]] = d
    return out


def worker_order(defs: dict[str, dict]) -> list[str]:
    """Agents without multiagent first (they are referenced by id), coordinators after."""
    plain = [k for k, d in defs.items() if not d.get("multiagent")]
    coord = [k for k, d in defs.items() if d.get("multiagent")]
    for k in coord:
        for ref in (defs[k]["multiagent"].get("subagents") or []) + (defs[k]["multiagent"].get("workflows") or []):
            if ref != "self" and ref not in plain:
                raise Halt(f"{k}: roster entry {ref!r} is not a plain (non-coordinator) agent definition")
    return plain + coord


def toolset(tools_fm: dict) -> list[dict]:
    enable = set(tools_fm.get("enable") or [])
    policy = {"type": tools_fm.get("permission", "auto")}
    return [{
        "type": TOOLSET,
        "default_config": {"enabled": False, "permission_policy": policy},
        "configs": [{"name": n, "enabled": n in enable} for n in TOOL_NAMES],
    }]


def roster(d: dict) -> list[str]:
    """Keys a coordinator lists under multiagent (subagents, then workflows), without 'self' or repeats."""
    ma = d.get("multiagent") or {}
    out: list[str] = []
    for k in (ma.get("subagents") or []) + (ma.get("workflows") or []):
        if k != "self" and k not in out:
            out.append(k)
    return out


def system_text(d: dict, worker_system: str | None = None) -> str:
    """The system prompt sent for d: its body, plus the run's shared text under WORKER_SYSTEM_SEP when given."""
    return d["system"] + WORKER_SYSTEM_SEP + worker_system if worker_system else d["system"]


def agent_params(d: dict, lock: dict, worker_system: str | None = None) -> dict:
    """API parameters of one agent; roster entries resolve to pinned {id, version} from the lock.
    Without worker_system the result is exactly what it was before that parameter existed."""
    p = {
        "name": d["name"],
        "description": " ".join(str(d["description"]).split()),
        "model": {"id": d["model"]["id"], "effort": d["model"]["effort"]},
        "system": system_text(d, worker_system),
        "tools": toolset(d["tools"]),
        "metadata": {METADATA_KEY: d["key"], "lane": "tools/api_agents"},
    }
    ma = d.get("multiagent")
    if ma:
        def ref(k):
            if k == "self":
                return {"type": "self"}
            ent = (lock.get("agents") or {}).get(k)
            if not ent:
                raise Halt(f"roster entry {k!r} has no lock entry yet (sync workers first)")
            return {"type": "agent", "id": ent["id"], "version": ent["version"]}

        def member(keys):
            if not keys:
                return {"type": "disabled"}
            return {"type": "enabled", "predefined_agents": [ref(k) for k in keys], "inline_agents": {"type": "disabled"}}

        p["multiagent"] = {"type": MULTIAGENT_TYPE, "subagents": member(ma.get("subagents")),
                           "workflows": member(ma.get("workflows")), "advisor": {"type": "disabled"}}
    return p


def env_params(d: dict) -> dict:
    return {"name": d["name"], "description": d.get("description"), "config": d["config"],
            "metadata": {METADATA_KEY: d["key"], "lane": "tools/api_agents"}}


def params_sha(p: dict) -> str:
    return sha256_bytes(canon_json(p).encode())


# ------------------------------------------------------------------ lock / ledger
def read_lock(path: Path = LOCK_PATH) -> dict:
    if not path.exists():
        return {"agents": {}, "environments": {}}
    d = json.loads(path.read_text())
    d.setdefault("agents", {}); d.setdefault("environments", {})
    return d


def write_lock(lock: dict, path: Path = LOCK_PATH) -> None:
    path.write_text(json.dumps(lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def ledger(event: str, path: Path | None = None, **fields) -> dict:
    path = Path(path) if path else globals()["LEDGER_PATH"]      # resolved at call time (tests redirect it)
    row = {"ts": now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


# ------------------------------------------------------------------ egress scan
def builder():
    """api_batch PackBuilder over both deny sources (fail-closed: Halt if either is missing or malformed)."""
    from tools.api_batch.deny import default_deny_files
    from tools.api_batch.pack import PackBuilder
    return PackBuilder.from_files(default_deny_files())


def scan_text(b, name: str, text: str) -> None:
    """Free text we wrote (system prompts, kickoff): L2 on the name, L3b body, L4 secrets. Raises PackError."""
    b.build_verify_only("spec_text", name, text)
    ip = ipv4_hits(text)
    if ip:
        from tools.api_batch.pack import PackError
        raise PackError(f"IPV4: literal address(es) {ip[:3]} in {name}")


def ipv4_hits(text: str) -> list[str]:
    return sorted({m.group(0) for m in IPV4.finditer(text)} - IPV4_OK)


def canonical_rel(rel) -> bool:
    """A lab-relative posix path in canonical form: no leading '/', no '.', '..' or empty components."""
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel:
        return False
    parts = rel.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def scan_lab_file(b, rel: str) -> tuple[dict | None, dict | None]:
    """(entry, None) when lab/<rel> may be mounted, else (None, skipped-record). Per-file fail-closed.
    The entry's sha256 is of the very bytes that were scanned."""
    from tools.api_batch.pack import PackError
    if not canonical_rel(rel):
        return None, {"rel": str(rel), "layer": "path", "reason": "not a canonical lab-relative path"}
    p = LAB_ROOT / rel
    if not p.is_file():
        return None, {"rel": rel, "layer": "missing", "reason": "not a file"}
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, {"rel": rel, "layer": "decode", "reason": "not UTF-8"}
    if not text.strip():
        return None, {"rel": rel, "layer": "empty", "reason": "empty file"}
    try:
        b.build_verify_only("seat_input", rel, text, origin=p)      # L1 (lab .md/.txt) + L2 + L3b + L4
    except PackError as e:
        msg = str(e)
        return None, {"rel": rel, "layer": msg.split(":", 1)[0], "reason": msg[:200]}
    ip = ipv4_hits(text)
    if ip:
        return None, {"rel": rel, "layer": "IPV4", "reason": f"literal address(es) {ip[:3]}"}
    return {"rel": rel, "sha256": sha256_bytes(raw), "bytes": len(raw), "est_tokens": est_tokens(text)}, None
