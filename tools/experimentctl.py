from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

RUN_ROOT = Path("/home/ayp/project/trm/results/sigma-k-tau-grid-N10retro").resolve()
RUN_ID_RE = re.compile(r"^k[1-9][0-9]*_z[01]_it[01]_N[1-9][0-9]*_s[0-9]+$")
PROTOCOL_VERSION = "1.0"


class CtlError(Exception):
    def __init__(self, code: str, message: str, exit_code: int = 2): self.code, self.message, self.exit_code = code, message, exit_code


def emit(value: dict[str, Any], exit_code: int = 0) -> int:
    print(json.dumps({"protocol_version": PROTOCOL_VERSION, **value}, sort_keys=True))
    return exit_code


def safe_run(run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id): raise CtlError("INVALID_RUN_ID", "run_id is not an allowed cell identifier")
    path = (RUN_ROOT / "runs" / run_id).resolve()
    if RUN_ROOT not in path.parents or path.parent != (RUN_ROOT / "runs").resolve(): raise CtlError("PATH_ESCAPE", "run path escapes the configured root")
    if not path.is_dir(): raise CtlError("RUN_NOT_FOUND", "run does not exist", 3)
    return path


def safe_file(run: Path, relative: str) -> Path:
    path = (run / relative).resolve()
    if run not in path.parents or not path.is_file(): raise CtlError("PATH_ESCAPE", "artifact is not a regular file inside the run")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def artifact_list(run: Path) -> list[dict[str, Any]]:
    items = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and not path.is_symlink():
            safe = safe_file(run, path.relative_to(run).as_posix())
            items.append({"path": safe.relative_to(run).as_posix(), "size": safe.stat().st_size, "sha256": sha256(safe)})
    return items


def status(run: Path) -> str:
    if (run / "DONE").is_file(): return "SUCCEEDED"
    if (run / "FAILED").is_file(): return "FAILED"
    if (run / "RUNNING").is_file(): return "RUNNING"
    return "UNKNOWN"


def get_run(run_id: str) -> dict[str, Any]:
    run = safe_run(run_id)
    return {"run_id": run_id, "status": status(run), "artifacts": artifact_list(run)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["server-info", "list-runs", "get-run", "status", "tail-log", "list-artifacts", "manifest", "validate", "get-artifact"])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--lines", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--path")
    parser.add_argument("--max-bytes", type=int, default=65536)
    try:
        args = parser.parse_args(argv)
        if not args.json: raise CtlError("JSON_REQUIRED", "--json is required")
        if args.command == "server-info": return emit({"server": {"run_root": str(RUN_ROOT), "controller": "experimentctl", "read_only": True}})
        if args.command == "list-runs":
            runs = [] if not (RUN_ROOT / "runs").is_dir() else [get_run(p.name) for p in sorted((RUN_ROOT / "runs").iterdir()) if p.is_dir() and RUN_ID_RE.fullmatch(p.name)]
            return emit({"runs": runs})
        if not args.run_id: raise CtlError("RUN_ID_REQUIRED", "run_id is required")
        run = safe_run(args.run_id)
        if args.command == "get-run": return emit({"run": get_run(args.run_id)})
        if args.command == "get-artifact":
            if not args.path: raise CtlError("PATH_REQUIRED", "--path is required")
            if not 1 <= args.max_bytes <= 1_000_000: raise CtlError("INVALID_MAX_BYTES", "max_bytes must be 1 through 1000000")
            allowed_ext = {".json", ".jsonl", ".txt", ".log", ".yaml", ".yml", ".csv", ".md"}
            target = safe_file(run, args.path)
            if target.suffix.lower() not in allowed_ext: raise CtlError("ARTIFACT_NOT_ALLOWED", "artifact type is not an allowlisted text format")
            size = target.stat().st_size
            content = target.read_bytes()[:args.max_bytes].decode("utf-8", errors="replace")
            return emit({"run_id": args.run_id, "path": target.relative_to(run).as_posix(), "size": size, "sha256": sha256(target), "truncated": size > args.max_bytes, "content": content})
        if args.command == "status": return emit({"run_id": args.run_id, "status": status(run)})
        if args.command == "list-artifacts": return emit({"run_id": args.run_id, "artifacts": artifact_list(run)})
        if args.command == "manifest":
            config = safe_file(run, "config.json")
            return emit({"run_id": args.run_id, "manifest": {"schema_version": "1.0", "status": status(run), "config": json.loads(config.read_text(encoding="utf-8")), "config_sha256": sha256(config)}})
        if args.command == "tail-log":
            if not 1 <= args.lines <= 500: raise CtlError("INVALID_LINES", "lines must be 1 through 500")
            candidates = ["train.log", "stdout.log", "stderr.log"]
            log = next((safe_file(run, name) for name in candidates if (run / name).is_file()), None)
            if log is None: raise CtlError("LOG_NOT_FOUND", "no allowlisted log exists", 3)
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-args.lines:]
            return emit({"run_id": args.run_id, "lines": lines})
        if args.command == "validate":
            required = ["config.json", "tau_curve.jsonl", "DONE"]
            missing = [name for name in required if not (run / name).is_file()]
            return emit({"run_id": args.run_id, "status": "PASS" if not missing else "FAIL", "missing": missing})
    except CtlError as exc:
        return emit({"error": {"code": exc.code, "message": exc.message}}, exc.exit_code)
    except Exception:
        return emit({"error": {"code": "INTERNAL_ERROR", "message": "controller failed safely"}}, 1)
    return 1


if __name__ == "__main__": raise SystemExit(main())
