#!/usr/bin/env python3
"""AD-5: read-only freshness alarm.

Thresholds are fixed by lab/11_governance/11_lab-root/AUTORESEARCH.md section 1 item 7:
STATE.md mtime > 48 h = stale; while processing/ is non-empty, newest
`stage == "monitor"` event in events.jsonl > 6 h = stale (else "n/a").
Exit 0 fresh, 1 stale, 2 file missing/unreadable.
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_status  # noqa: E402

REPO = queue_status.REPO
STATE = os.path.join(REPO, "lab/11_governance/11_lab-root/STATE.md")
EVENTS = os.path.join(REPO, "lab/93_monitoring/11_lab-monitoring/15_research-dashboard/events.jsonl")


def parse_ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def newest_events(path):
    """Return (newest_any, newest_monitor) as (datetime, stage) / datetime|None."""
    any_ts, any_stage, mon = None, None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                ts = parse_ts(e["ts_utc"])
            except (ValueError, KeyError, TypeError):
                continue
            if any_ts is None or ts > any_ts:
                any_ts, any_stage = ts, e.get("stage")
            if e.get("stage") == "monitor" and (mon is None or ts > mon):
                mon = ts
    return any_ts, any_stage, mon


def evaluate(state, events, queue_dir, state_max_h, monitor_max_h, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    snap = queue_status.snapshot(queue_dir)
    if snap is None:
        raise OSError(f"queue dir missing: {queue_dir}")
    st_mtime = dt.datetime.fromtimestamp(os.stat(state).st_mtime, dt.timezone.utc)
    state_age = (now - st_mtime).total_seconds() / 3600
    any_ts, any_stage, mon = newest_events(events)
    running = bool(snap["processing"])
    r = {
        "state_age_h": round(state_age, 2), "state_max_h": state_max_h,
        "state_status": "stale" if state_age > state_max_h else "fresh",
        "cohort_running": running, "monitor_max_h": monitor_max_h,
        "monitor_age_h": None, "monitor_status": "n/a",
        "newest_event_age_h": round((now - any_ts).total_seconds() / 3600, 2) if any_ts else None,
        "newest_event_stage": any_stage,
    }
    if running:
        if mon is None:
            r["monitor_status"] = "stale"
        else:
            age = (now - mon).total_seconds() / 3600
            r["monitor_age_h"] = round(age, 2)
            r["monitor_status"] = "stale" if age > monitor_max_h else "fresh"
    r["stale"] = "stale" in (r["state_status"], r["monitor_status"])
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-max-h", type=float, default=48)
    ap.add_argument("--monitor-max-h", type=float, default=6)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--events", default=EVENTS)
    ap.add_argument("--queue-dir")
    a = ap.parse_args(argv)
    try:
        r = evaluate(a.state, a.events, queue_status.resolve_queue_dir(a.queue_dir),
                     a.state_max_h, a.monitor_max_h)
    except (OSError, UnicodeDecodeError) as e:
        print(f"unreadable: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f'STATE.md age: {r["state_age_h"]} h (max {r["state_max_h"]}) -> {r["state_status"]}')
        ma = "" if r["monitor_age_h"] is None else f' age {r["monitor_age_h"]} h (max {r["monitor_max_h"]})'
        print(f'monitor event: cohort_running={r["cohort_running"]}{ma} -> {r["monitor_status"]}')
        print(f'newest event (info): stage={r["newest_event_stage"]} age {r["newest_event_age_h"]} h')
        print("RESULT: " + ("STALE" if r["stale"] else "FRESH"))
    return 1 if r["stale"] else 0


if __name__ == "__main__":
    sys.exit(main())
