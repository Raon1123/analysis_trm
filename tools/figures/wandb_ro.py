"""Read-only reader for local wandb ``*.wandb`` datastore files.

wandb 0.25's ``DataStore.open_for_scan`` opens the file read-write even though
it only reads. Run files under ``wandb/`` are symlinks to the only copy on the
NAS, so this module subclasses the datastore and opens the file with mode
"rb". The parent scan method is never called.

``iter_history(path, wanted_keys)`` yields ``(step, {key: value})`` for every
history record that carries at least one wanted key. A clean end of file ends
the iteration; any scan or parse error before the end raises ``TruncatedRun``.

Importing this module first sets WANDB_MODE=disabled and points the wandb
directories that are unset at fresh temporary directories, so that importing
wandb cannot write under the repository's wandb/ or the home directory.
"""

from __future__ import annotations

import os
import tempfile

os.environ["WANDB_MODE"] = "disabled"
for _var in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
    if not os.environ.get(_var):
        os.environ[_var] = tempfile.mkdtemp(prefix="wandb_ro_")

import json  # noqa: E402
import logging  # noqa: E402
from typing import Iterable, Iterator  # noqa: E402

from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore as _ds  # noqa: E402

logger = logging.getLogger(__name__)


class TruncatedRun(Exception):
    """A scan or parse error before the end of a ``*.wandb`` file."""

    def __init__(self, path: str, bytes_read: int, file_size: int, cause: str = ""):
        self.path = path
        self.bytes_read = bytes_read
        self.file_size = file_size
        self.cause = cause
        super().__init__(
            f"TruncatedRun: {path}: stopped at byte {bytes_read} of {file_size}"
            + (f" ({cause})" if cause else "")
        )


class ReadOnlyDataStore(_ds.DataStore):
    """DataStore whose scan open uses mode "rb" (body copied from wandb 0.25)."""

    def open_for_scan(self, fname):
        self._fname = fname
        logger.info("open for scan: %s", fname)
        self._fp = open(fname, "rb")
        self._index = 0
        self._size_bytes = os.stat(fname).st_size
        self._opened_for_scan = True
        self._read_header()


def _item_key(item) -> str:
    return item.key if item.key else "/".join(item.nested_key)


def iter_history(path: str, wanted_keys: Iterable[str], stats: dict | None = None
                 ) -> Iterator[tuple[int, dict]]:
    """Yield (step, {key: value}) for history records with >= 1 wanted key.

    ``stats`` (optional dict) is filled with ``last_step`` (largest ``_step`` of
    any history record seen so far), ``n_history`` and ``bytes_read``; it stays
    valid when TruncatedRun is raised.
    """
    wanted = set(wanted_keys)
    if stats is None:
        stats = {}
    stats.setdefault("last_step", None)
    stats.setdefault("n_history", 0)
    stats.setdefault("bytes_read", 0)
    file_size = os.stat(path).st_size
    ds = ReadOnlyDataStore()
    try:
        try:
            ds.open_for_scan(path)
        except Exception as exc:  # header unreadable
            raise TruncatedRun(path, 0, file_size, f"header: {exc!r}") from exc
        while True:
            pos = ds._fp.tell()
            stats["bytes_read"] = pos
            if pos >= file_size:
                break  # clean end of file
            try:
                data = ds.scan_data()
                if data is None:
                    pos = ds._fp.tell()
                    stats["bytes_read"] = pos
                    if pos != file_size:
                        raise ValueError("scan ended before end of file")
                    break
                rec = pb.Record()
                rec.ParseFromString(data)
                if rec.WhichOneof("record_type") != "history":
                    continue
                step = None
                values = {}
                for item in rec.history.item:
                    k = _item_key(item)
                    if k == "_step":
                        step = json.loads(item.value_json)
                    elif k in wanted:
                        values[k] = json.loads(item.value_json)
                if step is None:
                    step = rec.history.step.num
                step = int(step)
            except TruncatedRun:
                raise
            except Exception as exc:
                raise TruncatedRun(path, ds._fp.tell(), file_size, repr(exc)) from exc
            stats["n_history"] += 1
            if stats["last_step"] is None or step > stats["last_step"]:
                stats["last_step"] = step
            if values:
                yield step, values
    finally:
        ds.close()
