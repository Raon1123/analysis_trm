"""U1 experiment-data review (design §6.2): per-cell slicing for the map stage and item grouping for the reduce stage.

Pure functions over text the PackBuilder has already scanned. Nothing here reads a file the scanner has not seen, and
nothing recomputes a number: rows are copied verbatim (raw CSV strings), only columns that repeat the cell identity are
dropped. Cell = one run (arm x k x seed); the join key across the four sources is `run_name`.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from .common import Halt

RUN_RE = re.compile(r"pp_ltf_(g7|fullgrad)_lay2_k(\d+)_s(\d+)\Z")
IDENT = ("run_name", "arm", "k", "seed", "exp_id", "unit")        # repeat the cell identity: dropped from the slice rows
SOURCES = ("train_curves", "decode_curves", "cell_stats", "tstar")
# (expected rows per cell) sanity bounds: a short file means the input changed shape -> halt, do not review a fragment
EXPECT = {"train_curves": (50, 60), "decode_curves": (42, 42), "cell_stats": (2, 4), "tstar": (3, 6)}


class U1Error(Halt):
    pass


def parse_csv(text: str, label: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "run_name" not in rows[0]:
        raise U1Error(f"{label}: no rows or no run_name column")
    return rows


def _csv_block(rows: list[dict], drop=IDENT) -> str:
    cols = [c for c in rows[0] if c not in drop]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
    return buf.getvalue().rstrip("\n")


def cell_key(run_name: str) -> dict:
    m = RUN_RE.match(run_name)
    if not m:
        raise U1Error(f"run_name {run_name!r} is not a pp_ltf_<g7|fullgrad>_lay2_k<k>_s<seed> name")
    return {"arm": m.group(1), "k": int(m.group(2)), "seed": int(m.group(3)), "run_name": run_name}


def build_cells(texts: dict[str, str]) -> list[dict]:
    """texts: {train_curves, decode_curves, cell_stats, tstar} -> [{key, text, rows}] sorted by (arm, k, seed).
    Halts if a source disagrees on the set of runs or a cell has an out-of-range number of rows."""
    by: dict[str, dict[str, list[dict]]] = {}
    for src in SOURCES:
        rows = parse_csv(texts[src], src)
        for r in rows:
            by.setdefault(r["run_name"], {s: [] for s in SOURCES})[src].append(r)
    names = sorted(by, key=lambda n: (cell_key(n)["arm"], cell_key(n)["k"], cell_key(n)["seed"]))
    out = []
    for n in names:
        parts, counts = by[n], {}
        for src, (lo, hi) in EXPECT.items():
            counts[src] = len(parts[src])
            if not lo <= counts[src] <= hi:
                raise U1Error(f"{n}: {src} has {counts[src]} rows (expected {lo}..{hi}); inputs changed shape")
        key = cell_key(n)
        tr0 = parts["train_curves"][0]
        meta = f"arm={key['arm']} k={key['k']} seed={key['seed']} frozen_class={tr0.get('frozen_class', '')}"
        z = parts["decode_curves"][0].get("cell_class_Z", "")
        text = "\n".join([
            f"# CELL {n} ({meta} cell_class_Z={z})",
            "## cell_stats rows (one per split x readout kind)", _csv_block(parts["cell_stats"]),
            "## tstar rows (first-correct-loop distribution, per split x gcd class)", _csv_block(parts["tstar"]),
            "## train_curves rows (step,train_loss,probe_test_exact)", _csv_block(parts["train_curves"], IDENT + ("frozen_class",)),
            "## decode_curves rows (t,a_t,v_t)", _csv_block(parts["decode_curves"], IDENT + ("cell_class_Z",)),
        ]) + "\n"
        out.append({"key": key, "text": text, "rows": counts})
    return out


def cell_cond(key: dict) -> str:
    return f"{key['arm']}k{key['k']}s{key['seed']}"


# ----------------------------------------------------------------------------------------- reduce
def read_map_batch(map_dir: Path, inbox: Path) -> list[dict]:
    """The map batch's requests (from its verified manifest) joined with its INBOX rows.
    Strict: every request must have a collected, succeeded item."""
    from . import manifest as M
    plan = json.loads((map_dir / "plan.json").read_text(encoding="utf-8"))
    rows, msha = M.load_manifest(map_dir / "manifest.jsonl")
    if msha != plan["manifest_sha256"]:
        raise U1Error("map manifest.jsonl no longer matches its plan.json")
    if plan.get("use") != "U1":
        raise U1Error(f"map batch use is {plan.get('use')!r}, not 'U1'")
    inb = {}
    if Path(inbox).exists():
        for ln in Path(inbox).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                inb[r["custom_id"]] = r
    out, missing = [], []
    for r in rows:
        i = inb.get(r["custom_id"])
        if not i or i.get("result_status") != "succeeded" or not (map_dir / i["item"]).exists():
            missing.append(r["custom_id"])
            continue
        m = r["meta"]
        out.append({"custom_id": r["custom_id"], "arm": m["arm"], "k": m["k"], "seed": m["seed"], "cell": m["cell"],
                    "item": map_dir / i["item"]})
    if missing:
        raise U1Error(f"map batch not fully collected: {len(missing)} of {len(rows)} requests have no succeeded item "
                      f"(first: {missing[:3]}); the reduce batch is planned only after the map is collected")
    return sorted(out, key=lambda x: (x["arm"], x["k"], x["seed"]))


def group_items(items: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = {}
    for it in items:
        g.setdefault((it["arm"], it["k"]), []).append(it)
    return g
