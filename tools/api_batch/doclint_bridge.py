"""Import-only bridge to the frozen tools/api_seats/doc_lint.py (§8.8). Never edits it.
doc_lint.py does `import anthropic` at module top; build_params/quote_checks never use it, so when the
SDK is absent a throw-away stub module stands in for the import only."""
from __future__ import annotations

import importlib.util
import sys
import types

from .common import FROZEN_DOC_LINT


def doc_lint_module():
    if "doc_lint_frozen" in sys.modules:
        return sys.modules["doc_lint_frozen"]
    stubbed = False
    if importlib.util.find_spec("anthropic") is None:
        sys.modules["anthropic"] = types.ModuleType("anthropic")
        stubbed = True
    try:
        spec = importlib.util.spec_from_file_location("doc_lint_frozen", FROZEN_DOC_LINT[0])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["doc_lint_frozen"] = mod
        spec.loader.exec_module(mod)
    finally:
        if stubbed:
            sys.modules.pop("anthropic", None)
    return mod
