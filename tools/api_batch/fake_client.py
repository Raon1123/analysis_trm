"""Offline stand-in for client.RealClient (same interface). No network, no SDK, no key."""
from __future__ import annotations

import random
from datetime import datetime, timezone


class FakeApiError(Exception):
    pass


def _usage(model, cache="none"):
    u = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    if cache == "hit":
        u["cache_read_input_tokens"] = 1000
    elif cache == "write":
        u["cache_creation_input_tokens"] = 1000
        u["cache_creation"] = {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 0}
    return u


class FakeClient:
    def __init__(self, seed: int = 0, fallbacks: str = "create_error"):
        self.rng = random.Random(seed)
        self.batches: dict[str, dict] = {}
        self.fallbacks = fallbacks            # create_error | errored | accept
        self.calls: list[str] = []
        self.sync_sent: list[dict] = []
        self.sdk_version = "fake"
        self._n = 0

    # -- API surface
    def create(self, requests):
        self.calls.append("create")
        errored_ids = set()
        for r in requests:
            p = r["params"]
            if p.get("stream") or p.get("speed") or p.get("max_tokens") == 0:
                raise FakeApiError("F4: stream/speed/max_tokens:0 rejected")
            if "fallbacks" in p:
                if self.fallbacks == "create_error":
                    raise FakeApiError("validation error: fallbacks")
                if self.fallbacks == "errored":
                    errored_ids.add(r["custom_id"])
        self._n += 1
        bid = f"msgbatch_fake_{self._n:04d}"
        self.batches[bid] = {"id": bid, "requests": list(requests), "status": "in_progress", "outcomes": {},
                             "forced_errored": errored_ids, "created_at": datetime.now(timezone.utc).isoformat(),
                             "deleted": False, "opts": {}}
        return self._view(bid)

    def _view(self, bid):
        b = self.batches[bid]
        n = len(b["requests"])
        return {"id": bid, "processing_status": b["status"], "created_at": b["created_at"],
                "request_counts": {"processing": n if b["status"] == "in_progress" else 0,
                                   "succeeded": n if b["status"] == "ended" else 0, "errored": 0, "canceled": 0, "expired": 0}}

    def retrieve(self, bid):
        self.calls.append("retrieve")
        if self.batches[bid]["deleted"]:
            raise FakeApiError("not found")
        return self._view(bid)

    def finish(self, bid, outcomes: dict | None = None, drop=(), duplicate=(), shuffle=True, texts: dict | None = None):
        """outcomes: custom_id -> succeeded|refusal|max_tokens|errored|expired|canceled|cache_hit|cache_write."""
        b = self.batches[bid]
        b["status"] = "ended"
        b["outcomes"] = dict(outcomes or {})
        b["opts"] = {"drop": set(drop), "dup": set(duplicate), "shuffle": shuffle, "texts": dict(texts or {})}

    def results(self, bid):
        self.calls.append("results")
        b = self.batches[bid]
        if b["status"] != "ended" or b["deleted"]:
            raise FakeApiError("results not available")
        recs = []
        for r in b["requests"]:
            cid = r["custom_id"]
            kind = "errored" if cid in b["forced_errored"] else b["outcomes"].get(cid, "succeeded")
            recs.append((cid, self._record(cid, r["params"], kind, b["opts"]["texts"].get(cid))))
        out = [rec for cid, rec in recs if cid not in b["opts"]["drop"]]
        out += [rec for cid, rec in recs if cid in b["opts"]["dup"]]
        if b["opts"]["shuffle"]:
            self.rng.shuffle(out)
        yield from out

    def _record(self, cid, params, kind, text_override=None):
        if kind in ("errored", "expired", "canceled"):
            res = {"type": kind, "message": None,
                   "error": {"type": "error", "error": {"type": "api_error" if kind == "errored" else kind}} if kind == "errored" else None}
            return {"custom_id": cid, "result": res}
        text = '{"answer": "fake"}' if (params.get("output_config") or {}).get("format") else "fake output text"
        text = text_override or text
        stop = {"refusal": "refusal", "max_tokens": "max_tokens"}.get(kind, "end_turn")
        cache = {"cache_hit": "hit", "cache_write": "write"}.get(kind, "none")
        msg = {"id": f"msg_fake_{cid}", "model": params["model"], "stop_reason": stop,
               "content": [] if stop == "refusal" else [{"type": "text", "text": text}], "usage": _usage(params["model"], cache)}
        if stop == "refusal":
            msg["stop_details"] = {"type": "refusal", "category": "other"}
        return {"custom_id": cid, "result": {"type": "succeeded", "message": msg, "error": None}}

    def cancel(self, bid):
        self.calls.append("cancel")
        self.batches[bid]["status"] = "canceling"
        return self._view(bid)

    def list(self):
        self.calls.append("list")
        return [self._view(b) for b in self.batches if not self.batches[b]["deleted"]]

    def delete(self, bid):
        self.calls.append("delete")
        self.batches[bid]["deleted"] = True
        return {"id": bid, "type": "message_batch_deleted"}

    def _sync_reply(self, p):
        return {"id": "msg_fake_sync", "model": p["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "fake sync"}], "usage": _usage(p["model"])}

    def send_sync(self, request):
        self.calls.append("send_sync")
        self.sync_sent.append(request)
        return self._sync_reply(request["params"])

    def send_stream(self, request):
        """Streaming-path stand-in for RealClient.send_stream: same final-message dict as send_sync, recorded
        under its own call name; refuses a 'stream' param like messages.stream(**params) would."""
        self.calls.append("send_stream")
        self.sync_sent.append(request)
        p = request["params"]
        if "stream" in p:
            raise FakeApiError("send_stream: params must not carry 'stream'")
        return self._sync_reply(p)
