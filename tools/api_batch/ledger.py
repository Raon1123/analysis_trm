"""Append-only JSONL batch ledger with flock (§8.2). Not the goal-graph or trm-mcp event ledger."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

EVENTS = ("planned", "approved", "submitting", "submitted", "status", "ended", "collected", "deleted",
          "resubmit_planned", "canceled", "scanned", "tests_passed", "sync_sent", "sync_done", "error")
# Q4 (a): batch / day / month limits on summed hard-worst values.
# 2026-10-30 user: batch/day 190, month 200 (was 20/30/150); day = batch is this edit's reading of "한도숫자 190".
LIMIT_BATCH_USD, LIMIT_DAY_USD, LIMIT_MONTH_USD = 190.0, 190.0, 200.0


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def approval_id(row: dict) -> str:
    """Identity of one `approved` row (rows written before `approval_id` existed fall back to their ts)."""
    return row.get("approval_id") or row["ts"]


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, event: str, **fields) -> dict:
        if event not in EVENTS:
            raise ValueError(f"unknown ledger event {event!r}")
        row = {"ts": fields.pop("ts", None) or now(), "event": event, **fields}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return row

    def rows(self, event: str | None = None, **match) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if event and r.get("event") != event:
                continue
            if all(r.get(k) == v for k, v in match.items()):
                out.append(r)
        return out

    def last(self, event: str, **match) -> dict | None:
        r = self.rows(event, **match)
        return r[-1] if r else None

    def committed_usd(self, prefix: str | None = None) -> float:
        """Sum of hard-worst values minus released_usd of collected rows, restricted to rows whose ts
        starts with `prefix` ('YYYY-MM-DD' for the day, 'YYYY-MM' for the month). Batch lane: the MAX per
        manifest over its approved/submitted rows (one submit per manifest, so an upgraded approval
        replaces, never adds; NB-2). Sync lane (`lane: sync` approvals): one approval = one send, so every
        approval row counts on its own."""
        worst: dict[str, float] = {}
        for r in self.rows():
            if r["event"] in ("approved", "submitted") and r.get("manifest_sha256") \
                    and (prefix is None or r["ts"].startswith(prefix)):
                v = float(r.get("est_usd_hard_worst") or 0)
                k = f"{r['manifest_sha256']}#{approval_id(r)}" if r.get("lane") == "sync" else r["manifest_sha256"]
                worst[k] = max(worst.get(k, 0.0), v)
        released = sum(float(r.get("released_usd") or 0) for r in self.rows("collected")
                       if prefix is None or r["ts"].startswith(prefix))
        return max(0.0, sum(worst.values()) - released)
