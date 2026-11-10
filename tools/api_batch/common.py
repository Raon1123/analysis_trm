"""Shared constants and helpers for tools/api_batch (P0a, offline).

No network, no API key access. The only environment variables read are
TRM_MCP_SRC (optional path override), and the Claude-Code-session markers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
PKG_DIR = Path(__file__).resolve().parent
PROBES_DIR = PKG_DIR / "probes"
LAB_ROOT = REPO_ROOT / "lab"
GOV_DIR = LAB_ROOT / "11_governance" / "11_lab-root"
WITHHELD_RUNS_PATH = GOV_DIR / "WITHHELD_RUNS.yaml"
API_BATCH_DENY_PATH = GOV_DIR / "API_BATCH_DENY.yaml"
DEFAULT_BATCH_ROOT = LAB_ROOT / "38_api-batch"
# NB-R3-8c: not hard-coded for another host; override with TRM_MCP_SRC.
TRM_MCP_SRC = Path(os.environ.get("TRM_MCP_SRC", "/home/ayp/project/trm-mcp/src"))
FROZEN_DOC_LINT = [REPO_ROOT / "tools/api_seats" / n
                   for n in ("doc_lint.py", "doc_lint_prompt.md", "doc_lint_schema.json")]

BYTES_PER_TOKEN = 1.5      # design §4: conservative byte/token ratio (G5 wants an over-estimate)

# Claude Code session markers. Precedent: ~/.local/bin/anthropic-key-set:20 refuses when
# CLAUDECODE or CLAUDE_CODE_SESSION_ID is non-empty. Both are set inside a Claude Code
# session (verified in the implementing session: CLAUDECODE=1, CLAUDE_CODE_SESSION_ID=<uuid>,
# CLAUDE_CODE_ENTRYPOINT=cli).
SESSION_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID")


def in_claude_code_session(env=None) -> str | None:
    """Name of the first session marker that is set and non-empty, else None."""
    env = os.environ if env is None else env
    for k in SESSION_MARKERS:
        if env.get(k):
            return k
    return None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    """Same as doc_lint.py:130-131: UTF-8 re-encoding of the text as read."""
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canon_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def est_tokens(text: str) -> int:
    return -int(-len(text.encode("utf-8")) // BYTES_PER_TOKEN)


class Halt(RuntimeError):
    """Fail-closed stop. Never caught to continue with a partial result."""


def check_tools_path() -> None:
    """§8.1: `tools` is a namespace package; a regular top-level `tools` in site-packages
    would shadow it. Require that the repo's tools/ is among tools.__path__ and that
    `tools` has no __file__ (i.e. is a namespace package)."""
    import importlib
    tools = sys.modules.get("tools") or importlib.import_module("tools")
    paths = [Path(p).resolve() for p in getattr(tools, "__path__", [])]
    if TOOLS_DIR.resolve() not in paths or getattr(tools, "__file__", None):
        raise Halt(f"`tools` resolves to {list(map(str, paths))} / file={getattr(tools, '__file__', None)!r}, "
                   f"not the repo namespace package {TOOLS_DIR}")


_TRM_MCP_LOADED: dict = {}


def load_trm_mcp():
    """(withheld module, research_digest module) from trm-mcp, unmodified, via a stub
    package (make_digest.py:254-274). trm_mcp/__init__ imports the MCP app, so a bare
    package stands in for it. Any failure -> Halt (fail-closed rule 3)."""
    if _TRM_MCP_LOADED:
        return _TRM_MCP_LOADED["w"], _TRM_MCP_LOADED["r"]
    try:
        if "trm_mcp" not in sys.modules:
            pkg = types.ModuleType("trm_mcp")
            pkg.__path__ = [str(TRM_MCP_SRC / "trm_mcp")]
            sys.modules["trm_mcp"] = pkg
        import importlib
        w = importlib.import_module("trm_mcp.withheld")
        r = importlib.import_module("trm_mcp.research_digest")
    except Exception as e:  # noqa: BLE001
        raise Halt(f"trm-mcp matcher load failed: {type(e).__name__}: {str(e)[:200]}")
    _TRM_MCP_LOADED.update(w=w, r=r)
    return w, r


def trm_mcp_module_files() -> list[Path]:
    return [TRM_MCP_SRC / "trm_mcp" / "withheld.py", TRM_MCP_SRC / "trm_mcp" / "research_digest.py"]
