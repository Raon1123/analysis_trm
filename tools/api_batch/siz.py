"""SIZ-map / SIZ-reduce (state/iteration/z synthesis, PREREG 2026-10-30 §9 S2/S3; S23_templates/PACKET_SPEC.md §5).

Helpers for the ops.plan branches, modelled on u1.py. The packets themselves come from
tools/state_iter_z_synthesis/packets.py (pure functions over text the PackBuilder has already scanned); nothing here
reads a metric value or computes a number. Every refusal is a Halt (fail-closed, nothing written by plan).
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import Halt, sha256_file

MODEL_TAG = {"claude-sonnet-5-5": "sonnet", "claude-opus-5-5": "opus"}   # custom_id tag: twin ids must not collide


class SIZError(Halt):
    pass


def packets():
    from tools.state_iter_z_synthesis import packets as P
    return P


def call(fn, *a, **kw):
    """packets.PacketError is a plain Exception: re-raise it as a Halt so the CLI exits with a code."""
    P = packets()
    try:
        return fn(*a, **kw)
    except P.PacketError as e:
        raise SIZError(f"packets.py refused: {e}") from None


def check_templates(bd: Path, spec: dict, lab_root: Path, repo_root: Path) -> dict:
    """The staged header and system prompt (<batch>/shared/*.md) and the output schema must carry the sha256 that
    S23_templates/HASHES.sha256 witnesses. The hashes file never enters a request."""
    P = packets()
    hp = Path(spec["templates_hashes"])
    hp = hp if hp.is_absolute() else Path(lab_root) / hp
    if not hp.is_file():
        raise SIZError(f"templates_hashes {spec['templates_hashes']!r} not found")
    table = P.read_hashes(hp)
    out = {}
    for label, rel in (("header", spec["header"]), ("system", spec["prompt"]["system_template"])):
        keys = [k for k in table if k.endswith(f"/S23_templates/shared/{Path(rel).name}")]
        if len(keys) != 1:
            raise SIZError(f"{label} {Path(rel).name!r} is not listed exactly once in {hp.name}")
        f = Path(bd) / rel
        if not f.is_file():
            raise SIZError(f"{label} {rel!r} is not staged in the batch dir (packets.py stage)")
        if sha256_file(f) != table[keys[0]]:
            raise SIZError(f"{label} {rel!r} sha256 differs from {hp.name} (edited after witness)")
        out[label] = table[keys[0]]
    sch = spec["output_schema"]
    if sch not in table:
        raise SIZError(f"output_schema {sch!r} is not listed in {hp.name}")
    if sha256_file(Path(repo_root) / sch) != table[sch]:
        raise SIZError(f"output_schema {sch!r} sha256 differs from {hp.name}")
    out["schema"] = table[sch]
    # the packet generator (PREREG §9 S2 witness list): the listed file must carry the witnessed sha256 AND be the
    # module this process imported, so the code that builds the packets is the code that was witnessed
    gen = spec["packets"]["generator"]
    if gen not in table:
        raise SIZError(f"packets generator {gen!r} is not listed in {hp.name}")
    gp = Path(repo_root) / gen
    if not gp.is_file() or sha256_file(gp) != table[gen]:
        raise SIZError(f"packets generator {gen!r} sha256 differs from {hp.name} (edited after witness)")
    if Path(P.__file__).resolve() != gp.resolve():
        raise SIZError(f"imported packets module {P.__file__} is not the witnessed generator {gen!r}")
    out["generator"] = table[gen]
    return out


def read_map_batch(map_dir: Path, inbox: Path, model: str) -> tuple[list[dict], list[dict]]:
    """(manifest rows, items) of one SIZ-map batch. Strict, as u1.read_map_batch: the manifest must match its
    plan.json, every request must be for `model`, and every request needs a collected, succeeded item in INBOX."""
    from . import manifest as M
    map_dir = Path(map_dir)
    if not (map_dir / "plan.json").is_file():
        raise SIZError(f"map batch {map_dir} has no plan.json (not planned)")
    plan = json.loads((map_dir / "plan.json").read_text(encoding="utf-8"))
    rows, msha = M.load_manifest(map_dir / "manifest.jsonl")
    if msha != plan["manifest_sha256"]:
        raise SIZError(f"map manifest.jsonl of {map_dir.name} no longer matches its plan.json")
    if plan.get("use") != "SIZ-map":
        raise SIZError(f"map batch {map_dir.name} use is {plan.get('use')!r}, not 'SIZ-map'")
    bad = [r["custom_id"] for r in rows if r["params"]["model"] != model or r["meta"].get("model") != model]
    if bad or not rows:
        raise SIZError(f"map batch {map_dir.name} is not a {model} batch (first: {bad[:2]})")
    inb = {}
    if Path(inbox).exists():
        for ln in Path(inbox).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                inb[r["custom_id"]] = r
    items, missing = [], []
    for r in rows:
        i = inb.get(r["custom_id"])
        if not i or i.get("result_status") != "succeeded" or not (map_dir / i["item"]).exists():
            missing.append(r["custom_id"])
            continue
        m = r["meta"]
        items.append({"custom_id": r["custom_id"], "q": m["q"], "cell_id": m["cell_id"], "model": model,
                      "item": map_dir / i["item"]})
    if missing:
        raise SIZError(f"map batch {map_dir.name} not fully collected: {len(missing)} of {len(rows)} requests have no "
                       f"succeeded item (first: {missing[:3]}); the reduce batch is planned only after both maps are collected")
    return rows, items


def check_twins(rows_by_model: dict[str, list[dict]]) -> None:
    """The two map batches must be twins: same cells, and per cell the same params except `model` (same packet,
    header, system prompt, schema, effort, max_tokens, cache_ttl)."""
    def norm(rows):
        out = {}
        for r in rows:
            p = {k: v for k, v in r["params"].items() if k != "model"}
            out[r["meta"]["cell_id"]] = (p, r["pack_sha256s"])
        return out
    (ma, ra), (mb, rb) = rows_by_model.items()
    a, b = norm(ra), norm(rb)
    if set(a) != set(b):
        raise SIZError(f"map twins cover different cells: {sorted(set(a) ^ set(b))[:4]}")
    diff = sorted(c for c in a if a[c] != b[c])
    if diff:
        raise SIZError(f"map twins {ma} / {mb} differ beyond the model (packet/header/prompt/params) at {diff[:4]}")
