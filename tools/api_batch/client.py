"""Thin SDK wrapper (create / retrieve / results / cancel / list / delete / send_sync / send_stream). The SDK is
imported lazily and only when a RealClient is constructed -- nothing in P0a does that.
All methods return plain dicts so tests can inject a FakeClient with the same interface.
Result record shape: {"custom_id", "result": {"type": succeeded|errored|canceled|expired,
"message": {...}|None, "error": {...}|None}}."""
from __future__ import annotations


def _dump(o):
    return o.model_dump(mode="json") if hasattr(o, "model_dump") else o


class RealClient:
    def __init__(self):
        import anthropic                      # lazy: SDK needed only for the metered lane
        self._c = anthropic.Anthropic()
        self.sdk_version = anthropic.__version__

    def create(self, requests: list[dict]) -> dict:
        return _dump(self._c.messages.batches.create(requests=requests))

    def retrieve(self, batch_id: str) -> dict:
        return _dump(self._c.messages.batches.retrieve(batch_id))

    def results(self, batch_id: str):
        for rec in self._c.messages.batches.results(batch_id):
            yield _dump(rec)

    def cancel(self, batch_id: str) -> dict:
        return _dump(self._c.messages.batches.cancel(batch_id))

    def list(self) -> list[dict]:
        return [_dump(b) for b in self._c.messages.batches.list()]

    def delete(self, batch_id: str) -> dict:
        return _dump(self._c.messages.batches.delete(batch_id))

    def send_sync(self, request: dict) -> dict:
        """Standard-price synchronous send of one manifest request (probe-sync only)."""
        return _dump(self._c.messages.create(**request["params"]))

    def send_stream(self, request: dict) -> dict:
        """Standard-price streamed send of one manifest request (draft-sync). Same params and the same
        plain-dict final message as send_sync, but over SSE: SDK 1.13.0 refuses (or times out) a
        non-streaming create whose max_tokens implies a long generation. probe-sync keeps send_sync."""
        params = request["params"]
        if "stream" in params:
            raise ValueError("send_stream: params must not carry 'stream' (messages.stream sets it)")
        with self._c.messages.stream(**params) as s:
            return _dump(s.get_final_message())
