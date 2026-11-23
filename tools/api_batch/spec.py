"""spec.yaml validation (§5.2): required keys, allowed use, explicit effort, explicit caps."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .common import Halt
from .manifest import USES
from .prices import PRICES

EFFORTS = ("low", "medium", "high", "xhigh", "max")
TTLS = ("5m", "1h", "none")
DOCLINT_MODEL = "claude-opus-5-5"
FIGQA_KEYS = ("name", "use", "models", "effort", "max_tokens", "cache_ttl", "prompt", "context", "output_schema", "figures", "budget")
FIGQA_FIG_KEYS = {"slug", "exp_id", "origin", "png", "caption"}


class SpecError(Halt):
    pass


def load_spec(path: Path) -> dict:
    try:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SpecError(f"spec unreadable: {type(e).__name__}")
    return validate_spec(spec)


def validate_spec(spec) -> dict:
    if not isinstance(spec, dict):
        raise SpecError("spec is not a mapping")
    if spec.get("use") == "probe":
        need = ["name", "use", "budget", "probe_requests"]
    elif spec.get("use") == "seat-doclint":
        need = ["name", "use", "model", "n", "inputs", "budget"]
    elif spec.get("use") == "seat-figqa":
        need = list(FIGQA_KEYS)
    elif spec.get("use") == "U1":
        need = ["name", "use", "model", "effort", "max_tokens", "cache_ttl", "header", "prompt", "output_schema", "data", "budget"]
    elif spec.get("use") == "U1-reduce":
        need = ["name", "use", "model", "effort", "max_tokens", "digest_max_tokens", "cache_ttl", "header", "prompt",
                "map_batch", "group_by", "budget"]
    elif spec.get("use") == "SIZ-map":
        need = ["name", "use", "model", "effort", "max_tokens", "cache_ttl", "header", "prompt", "output_schema", "packets",
                "data", "templates_hashes", "budget"]
    elif spec.get("use") == "SIZ-reduce":
        need = ["name", "use", "model", "effort", "max_tokens", "cache_ttl", "header", "prompt", "output_schema", "packets",
                "group_by", "digest", "exclude_q_tokens", "map_batches", "data", "templates_hashes", "budget"]
    elif spec.get("use") == "runcard":
        need = ["name", "use", "model", "effort", "max_tokens", "cache_ttl", "header", "prompt", "output_schema", "cards", "budget"]
    elif spec.get("use") == "docaudit":
        need = ["name", "use", "models", "effort", "max_tokens", "cache_ttl", "header", "prompt", "output_schema", "files", "budget"]
    else:
        need = ["name", "use", "model", "effort", "max_tokens", "cache_ttl", "context_pack", "prompt", "budget"]
    for k in need:
        if k not in spec:
            raise SpecError(f"missing key {k!r}")
    if spec["use"] not in USES:
        raise SpecError(f"use {spec['use']!r} not in {sorted(USES)}")
    cap = (spec["budget"] or {}).get("cap_usd_hard_worst")
    if not isinstance(cap, (int, float)) or cap <= 0:
        raise SpecError("budget.cap_usd_hard_worst must be a positive number")
    if spec["use"] == "probe":
        for pr in spec["probe_requests"]:
            _check_req(pr)
        return spec
    if spec["use"] == "seat-doclint":
        # effort/max_tokens/schema/prompt are the frozen doc_lint.py values (§8.8), so none are spec keys.
        if spec["model"] != DOCLINT_MODEL:
            raise SpecError(f"seat-doclint model must be {DOCLINT_MODEL!r} (only Opus 5.5 is shadow-eligible)")
        if not isinstance(spec["n"], int) or not 1 <= spec["n"] <= 3:
            raise SpecError("seat-doclint n must be an integer 1..3")
        ins = spec["inputs"]
        if not isinstance(ins, dict) or set(ins) != {"results", "prereg"} or not all(isinstance(v, str) and v for v in ins.values()):
            raise SpecError("seat-doclint inputs must be exactly {results: <path>, prereg: <path>}")
        return spec
    if spec["use"] == "seat-figqa":
        return _validate_figqa(spec)
    if spec["use"] in ("U1", "U1-reduce"):
        _check_req(spec)
        if not isinstance(spec["header"], str) or not spec["header"].startswith("shared/"):
            raise SpecError("U1 header must be a path under the batch directory's shared/")
        if not (spec["prompt"] or {}).get("system_template"):
            raise SpecError("prompt.system_template required")
        if spec["use"] == "U1":
            d = spec["data"]
            if not isinstance(d, dict) or set(d) != {"train_curves", "decode_curves", "cell_stats", "tstar"} \
                    or not all(isinstance(v, str) and v for v in d.values()):
                raise SpecError("U1 data must be exactly {train_curves, decode_curves, cell_stats, tstar} -> path")
        else:
            if spec["group_by"] != ["arm", "k"]:
                raise SpecError("U1-reduce group_by must be [arm, k]")
            if not isinstance(spec["digest_max_tokens"], int) or spec["digest_max_tokens"] <= 0:
                raise SpecError("digest_max_tokens must be a positive integer")
            if not isinstance(spec["map_batch"], str) or not spec["map_batch"]:
                raise SpecError("map_batch must be the map batch directory (lab-relative)")
            sc = spec.get("scopes")
            if sc is not None and (not isinstance(sc, list) or not sc or not all(isinstance(s, str) and s for s in sc)):
                raise SpecError("scopes (optional) must be a non-empty list of scope names, e.g. [g7k3, digest]")
        return spec
    if spec["use"] in ("SIZ-map", "SIZ-reduce"):
        return _validate_siz(spec)
    if spec["use"] == "runcard":
        return _validate_runcard(spec)
    if spec["use"] == "docaudit":
        from .docaudit import validate as _validate_docaudit
        return _validate_docaudit(spec)
    _check_req(spec)
    cp = spec["context_pack"]
    if not isinstance(cp, dict) or not isinstance(cp.get("sources"), list) or not cp["sources"]:
        raise SpecError("context_pack.sources must be a non-empty list")
    if not isinstance(cp.get("cap_tokens", 60000), int):
        raise SpecError("context_pack.cap_tokens must be an integer")
    pr = spec["prompt"]
    if not pr.get("system_template") or not isinstance(pr.get("perspectives", ["_"]), list):
        raise SpecError("prompt.system_template required; perspectives must be a list")
    if int(pr.get("repeats", 1)) < 1:
        raise SpecError("prompt.repeats must be >= 1")
    return spec


def _validate_figqa(spec: dict) -> dict:
    """seat-figqa: B1/B2/B3 blind readers (fig-qa.md) through the batch path. Exact key set; no refusal_fallback
    (a refusal is recorded as REFUSED at collect, never re-sent)."""
    if set(spec) != set(FIGQA_KEYS):
        raise SpecError(f"seat-figqa keys must be exactly {sorted(FIGQA_KEYS)} (extra: {sorted(set(spec) - set(FIGQA_KEYS))})")
    ms = spec["models"]
    if not isinstance(ms, dict) or set(ms) != {"b1", "b2", "b3"} or not all(isinstance(v, str) and v in PRICES for v in ms.values()):
        raise SpecError("seat-figqa models must be exactly {b1, b2, b3} -> a model in prices.py")
    if spec["effort"] not in EFFORTS:
        raise SpecError("effort must be explicit (low|medium|high|xhigh|max)")
    if not isinstance(spec["max_tokens"], int) or isinstance(spec["max_tokens"], bool) or spec["max_tokens"] <= 0:
        raise SpecError("max_tokens must be a positive integer")
    if spec["cache_ttl"] not in TTLS:
        # 'none' = legacy layout (image first, no breakpoint); 5m|1h = ctx-first-v2 layout with breakpoints on the
        # shared context and the image (2026-11-05 user decision; plan api-joyful-island B)
        raise SpecError("seat-figqa cache_ttl must be 5m | 1h | none")
    pr = spec["prompt"]
    if not isinstance(pr, dict) or set(pr) != {"system_template"} or not isinstance(pr["system_template"], str) or not pr["system_template"]:
        raise SpecError("seat-figqa prompt must be exactly {system_template: <path>}")
    ctx = spec["context"]
    if not isinstance(ctx, str) or not re.fullmatch(r"tools/api_batch/prompts/[^/]+\.md", ctx):
        raise SpecError("seat-figqa context must be a repo-relative tools/api_batch/prompts/<file>.md")
    if not isinstance(spec["output_schema"], str) or not spec["output_schema"] or "/" in spec["output_schema"]:
        raise SpecError("seat-figqa output_schema must be a name under tools/api_batch/schemas/")
    figs = spec["figures"]
    if not isinstance(figs, list) or not figs:
        raise SpecError("seat-figqa figures must be a non-empty list")
    slugs = set()
    for f in figs:
        if not isinstance(f, dict) or set(f) != FIGQA_FIG_KEYS or not all(isinstance(v, str) and v for v in f.values()):
            raise SpecError(f"seat-figqa figure must be exactly {sorted(FIGQA_FIG_KEYS)} (non-empty strings)")
        if f["slug"] in slugs:
            raise SpecError(f"duplicate figure slug {f['slug']!r}")
        slugs.add(f["slug"])
        if not re.fullmatch(r"user/[^/]+\.png", f["png"]):
            raise SpecError(f"figure png must be batch-relative user/<name>.png, got {f['png']!r}")
        if not f["caption"].endswith(".md") or f["caption"].startswith("/") or ".." in f["caption"].split("/"):
            raise SpecError("figure caption must be a lab-relative .md path")
    return spec


def _validate_runcard(spec: dict) -> dict:
    """runcard (wandb run-card prose, Stage B): one request per card; the card set is enumerated at plan time from
    cards.root + cards.cohorts (re-plan = plan again)."""
    from .runcard import COHORTS
    _check_req(spec)
    if not isinstance(spec["header"], str) or not re.fullmatch(r"shared/[^/]+\.md", spec["header"]):
        raise SpecError("runcard header must be a batch-relative shared/<file>.md")
    pr = spec["prompt"]
    if not isinstance(pr, dict) or set(pr) != {"system_template"} \
            or not re.fullmatch(r"tools/api_batch/prompts/[^/]+\.md", str(pr["system_template"])):
        raise SpecError("runcard prompt must be exactly {system_template: tools/api_batch/prompts/<file>.md}")
    if not isinstance(spec["output_schema"], str) or not spec["output_schema"] or "/" in spec["output_schema"]:
        raise SpecError("runcard output_schema must be a name under tools/api_batch/schemas/")
    c = spec["cards"]
    if not isinstance(c, dict) or "root" not in c or not isinstance(c["root"], str) or not c["root"] \
            or not set(c) <= {"root", "cohorts", "ids_file", "include_injected"}:
        raise SpecError("runcard cards must be {root: <lab-relative dir>, cohorts?: [...], ids_file?: shared-or-batch-relative path, "
                        "include_injected?: bool}")
    if "cohorts" not in c and "ids_file" not in c:
        raise SpecError("runcard cards needs cohorts and/or ids_file")
    if "cohorts" in c and (not isinstance(c["cohorts"], list) or not c["cohorts"] or not all(x in COHORTS for x in c["cohorts"])
                           or len(set(c["cohorts"])) != len(c["cohorts"])):
        raise SpecError(f"runcard cards.cohorts must be a non-empty list of distinct values from {list(COHORTS)}")
    f = c.get("ids_file")
    if f is not None and (not isinstance(f, str) or not re.fullmatch(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*", f) or ".." in f.split("/")):
        raise SpecError("runcard cards.ids_file must be a relative path inside the batch directory")
    if "include_injected" in c and not isinstance(c["include_injected"], bool):
        raise SpecError("runcard cards.include_injected must be a boolean")
    if spec["cache_ttl"] == "none":
        raise SpecError("runcard needs a cached shared header: cache_ttl 5m | 1h")
    return spec


def _validate_siz(spec: dict) -> dict:
    """SIZ-map / SIZ-reduce (state/iteration/z synthesis, PREREG §9 S2/S3; PACKET_SPEC.md §5). One model per spec;
    the packet generator, its function and the map classes are fixed literals (never looked up from the spec)."""
    from tools.state_iter_z_synthesis import packets as P
    if not isinstance(spec.get("model"), str):
        raise SpecError("SIZ spec takes exactly one model (a string), one spec per model")
    _check_req(spec)
    if not isinstance(spec["header"], str) or not spec["header"].startswith("shared/"):
        raise SpecError("SIZ header must be a path under the batch directory's shared/")
    st = (spec["prompt"] or {}).get("system_template") if isinstance(spec["prompt"], dict) else None
    if not isinstance(st, str) or not st.startswith("shared/") or not st.endswith(".md"):
        raise SpecError("SIZ prompt.system_template must be a batch-relative shared/<file>.md (staged copy)")
    if not isinstance(spec["output_schema"], str) or "/" not in spec["output_schema"]:
        raise SpecError("SIZ output_schema must be a repo-relative path")
    if not isinstance(spec["templates_hashes"], str) or not spec["templates_hashes"].endswith("HASHES.sha256"):
        raise SpecError("SIZ templates_hashes must be the lab-relative S23_templates/HASHES.sha256")
    pk = spec["packets"]
    if not isinstance(pk, dict) or pk.get("generator") != "tools/state_iter_z_synthesis/packets.py":
        raise SpecError("SIZ packets.generator must be tools/state_iter_z_synthesis/packets.py")
    d = spec["data"]
    if spec["use"] == "SIZ-map":
        if spec["model"] not in P.MAP_MODELS:
            raise SpecError(f"SIZ-map model must be one of {P.MAP_MODELS} (one spec per model)")
        if pk.get("function") != "build_map_packets" or pk.get("classes") != list(P.S2_CLASSES):
            raise SpecError(f"SIZ-map packets must be build_map_packets over classes {list(P.S2_CLASSES)}")
        if not isinstance(pk.get("r3_ii_cite", "") or "", str):
            raise SpecError("SIZ-map packets.r3_ii_cite must be a string")
        if not isinstance(d, dict) or set(d) != {"s1_table", "s1_cells"} or not all(isinstance(v, str) and v for v in d.values()):
            raise SpecError("SIZ-map data must be exactly {s1_table, s1_cells} -> path")
    else:
        if spec["model"] != P.REDUCE_MODEL:
            raise SpecError(f"SIZ-reduce model must be {P.REDUCE_MODEL!r}")
        if pk.get("function") != "build_reduce_packets":
            raise SpecError("SIZ-reduce packets.function must be build_reduce_packets")
        if spec["group_by"] != ["q"]:
            raise SpecError("SIZ-reduce group_by must be [q]")
        if spec["digest"] is not False:
            raise SpecError("SIZ-reduce digest must be false (PREREG §9 S3: no global digest)")
        if spec["exclude_q_tokens"] != list(P.REDUCE_EXCLUDED):
            raise SpecError(f"SIZ-reduce exclude_q_tokens must be {list(P.REDUCE_EXCLUDED)}")
        mb = spec["map_batches"]
        if not isinstance(mb, dict) or set(mb) != set(P.MAP_MODELS) or not all(isinstance(v, str) and v for v in mb.values()):
            raise SpecError(f"SIZ-reduce map_batches must map exactly {list(P.MAP_MODELS)} -> map batch dir")
        if len(set(mb.values())) != len(mb):
            raise SpecError("SIZ-reduce map_batches must name two different map batch dirs")
        if not isinstance(d, dict) or set(d) != {"s1_cells"} or not isinstance(d["s1_cells"], str) or not d["s1_cells"]:
            raise SpecError("SIZ-reduce data must be exactly {s1_cells} -> path (no S1_table rows, PREREG §9 S3)")
        sc = spec.get("scopes")
        if sc is not None and (not isinstance(sc, list) or not sc or not all(isinstance(s, str) and s for s in sc)):
            raise SpecError("scopes (optional) must be a non-empty list of question IDs, e.g. [Q-I2]")
    return spec


def _check_req(d: dict) -> None:
    if not isinstance(d.get("model"), str) or d.get("model") not in PRICES:
        raise SpecError(f"model {d.get('model')!r} not in prices.py")
    if d.get("effort") not in EFFORTS:
        raise SpecError("effort must be explicit (low|medium|high|xhigh|max)")
    if not isinstance(d.get("max_tokens"), int) or d["max_tokens"] <= 0:
        raise SpecError("max_tokens must be a positive integer")
    if d.get("cache_ttl", "none") not in TTLS:
        raise SpecError("cache_ttl must be 5m | 1h | none")
