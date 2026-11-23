"""Build one registered figure in every language of its entry.

Run only as `python -m tools.figures.run_one --registry-file <yaml> --out <dir> --cache-dir <dir>`.
"""
import argparse
import importlib
import sys
import traceback
from pathlib import Path

from . import io as fio
from . import registry, style


def run(registry_file, out, cache_dir):
    entry = registry.load_entry(registry_file)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mod = importlib.import_module(entry.module)
    if not callable(getattr(mod, "build", None)):
        raise RuntimeError(f"{entry.module} has no build(ctx)")
    for lang in entry.langs:
        style.apply_style(lang)
        ctx = fio.FigureContext(entry.id, lang, out, cache_dir, entry, registry_file=registry_file)
        mod.build(ctx)
        style.plt.close("all")
    expected = [f"{entry.id}.values.csv", f"{entry.id}.source.json"]
    for lang in entry.langs:
        expected += [f"{entry.id}.{lang}.png", f"{entry.id}.{lang}.pdf"]
    bad = [n for n in expected if not (out / n).is_file() or (out / n).stat().st_size == 0]
    if bad:
        raise RuntimeError(f"missing or empty outputs: {bad}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.figures.run_one")
    ap.add_argument("--registry-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True)
    a = ap.parse_args(argv)
    try:
        run(a.registry_file, a.out, a.cache_dir)
    except BaseException:  # noqa: BLE001 - any failure must give a non-zero exit
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
