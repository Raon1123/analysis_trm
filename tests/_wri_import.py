"""Import tools/wandb_run_index modules in tests without leaking their bare-name ``common``.

tools/wandb_run_index/*.py are scripts: they put their own directory on ``sys.path`` and import their
siblings by bare name (``from common import ...``). dataset/*.py do the same with a different
``common`` (dataset/common.py). In one pytest process whichever is imported first stays cached as
``sys.modules["common"]`` and the other side then fails with an ImportError.

``wri(name)`` imports one run-index module with the run-index ``common`` in place for the duration
of the import only, then restores whatever ``common`` was cached before and takes the run-index
directory off ``sys.path`` again. Every run-index module imported through here shares one ``common``
module object.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

WRI_DIR = str(Path(__file__).resolve().parents[1] / "tools" / "wandb_run_index")

_wri_common: ModuleType | None = None


def wri(name: str) -> ModuleType:
    """Return tools/wandb_run_index/<name>.py as a module, leaving ``sys.modules["common"]`` as it was."""
    global _wri_common
    outside = sys.modules.pop("common", None)
    if _wri_common is not None:
        sys.modules["common"] = _wri_common
    sys.path.insert(0, WRI_DIR)
    try:
        module = importlib.import_module(name)
        _wri_common = sys.modules.get("common", _wri_common)
        return module
    finally:
        sys.path[:] = [p for p in sys.path if p != WRI_DIR]
        sys.modules.pop("common", None)
        if outside is not None and outside is not _wri_common:
            sys.modules["common"] = outside
