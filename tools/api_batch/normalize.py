"""raw batch result -> item (§8.2, §8.4): status classification, usage cost, item rendering, L5."""
from __future__ import annotations

import json

from .prices import cost_from_usage

STATUSES = ("succeeded", "refusal", "max_tokens", "pause_turn", "parse_error", "errored", "expired", "canceled")
UNBILLED = ("errored", "expired", "canceled")        # F7


def message_text(msg: dict) -> str:
    return "".join(b.get("text", "") for b in (msg or {}).get("content", []) if b.get("type") == "text")


def classify(rec: dict, expect_json: bool) -> str:
    res = rec["result"]
    t = res["type"]
    if t in UNBILLED:
        return t
    if t != "succeeded":
        return "errored"
    msg = res["message"]
    sr = msg.get("stop_reason")
    if sr == "refusal":
        return "refusal"
    if sr == "max_tokens":
        return "max_tokens"
    if sr == "pause_turn":
        return "pause_turn"
    if expect_json:
        try:
            json.loads(message_text(msg))
        except Exception:  # noqa: BLE001
            return "parse_error"
    return "succeeded"


def cost_est(model: str, rec: dict) -> float:
    msg = (rec["result"] or {}).get("message")
    return cost_from_usage(model, msg.get("usage") or {}) if msg else 0.0


def render_item(*, use, batch_id, custom_id, row, rec, status, cost, ended_at, l5, deleted_at=None, retention_until=None, extra=None) -> str:
    msg = (rec["result"] or {}).get("message") or {}
    hdr = {
        "status": "DRAFT (api-batch output — not evidence)", "use": use, "batch_id": batch_id, "custom_id": custom_id,
        "message_id": msg.get("id"), "model_requested": row["params"]["model"], "model_returned": msg.get("model"),
        "prompt_sha256": row["meta"].get("prompt_sha256"), "pack_sha256": ",".join(row["pack_sha256s"]),
        "params_sha256": row["params_sha256"], "stop_reason": msg.get("stop_reason"), "result_status": status,
        "usage": json.dumps(msg.get("usage") or {}, sort_keys=True), "cost_est_usd": f"{cost:.6f}",
        "ended_at": ended_at, "l5": l5, "deleted_at": deleted_at, "retention_until": retention_until,
        **(extra or {}),
    }
    lines = ["---"] + [f"{k}: {v}" for k, v in hdr.items() if v is not None] + ["---", ""]
    return "\n".join(lines) + message_text(msg) + "\n"
