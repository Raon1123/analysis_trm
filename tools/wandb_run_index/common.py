"""Pure helpers shared by the wandb run-card pipeline (no wandb / trm_mcp imports).

Pipeline (Stage A, deterministic; see lab/22_plans/11_lab-plans/23_2026-11-02_wandb-runs-results-docs/):
  1. scan.py   (trm-mcp venv)  .wandb  -> _meta/scan_cache.jsonl   (sha256, header, exit, metric stats)
  2. build.py  (uv run)        metadata/config/summary + scan cache -> _wandb_runs_index.jsonl
  3. links.py  (uv run)        index + read-only wandb API listing + logs/queue -> _meta/links.jsonl
  4. cards.py  (uv run)        index + scan + links -> lab/21_experiments/12_run-cards/<project>/<name>__<id>.md

Everything in this module is a pure function of its arguments so that tests/test_wandb_run_cards.py
can exercise tiering, cohort, sanitising and rendering without wandb, network or real data.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from typing import Any, Iterable

UTC = dt.timezone.utc
ENTITY = "raon1123"
CARDS_REL = "lab/21_experiments/12_run-cards"
META_REL = CARDS_REL + "/_meta"
NO_PROJECT_DIR = "_no-project"
PROSE_START, PROSE_END = "<!-- prose:auto start -->", "<!-- prose:auto end -->"

# --------------------------------------------------------------------------------------------
# Link tiers
# --------------------------------------------------------------------------------------------
TIERS = ("api-confirmed", "queue-log", "constructed-unverified", "no-remote")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}  # lower = stronger


def run_url(entity: str, project: str, rid: str) -> str:
    return f"https://wandb.ai/{entity}/{project}/runs/{rid}"


def assign_tier(run: dict, api: dict, queue_urls: dict, listed_projects: dict) -> dict:
    """Assign exactly one link tier to one run.

    run:   {id, project (header project, authoritative), mode ('online'|'offline'), synced_marker(bool)}
    api:   {run_id: {project, name, state, url, created_at}}  -- union of all listings
    queue_urls: {run_id: [(entity, project, url, log_path), ...]}  literal URLs in logs/queue
    listed_projects: {project: {"ok": bool, "n": int, "error": str|None}}

    Returns a dict with tier, url (None unless the tier licenses one), url_status, evidence, and a list
    `errors` of inconsistencies (non-empty -> caller must exit non-zero).

    Precedence: api-confirmed > queue-log > constructed-unverified > no-remote.
      * api-confirmed: id present in the API listing of the run's own project. URL = API run.url.
      * queue-log: URL literally printed in logs/queue (same project), id NOT in a complete listing.
      * constructed-unverified: online run, no API hit, no queue-log URL. URL is a guess.
      * no-remote: offline run with no API hit, no queue URL and no .synced marker -- no URL emitted.
    Inconsistencies: id listed under another project; queue URL with another project/entity;
    listing failed/empty for a project that has local online runs (recorded per run as
    `listing_ok=False`; such runs can never be "confirmed absent").
    """
    rid, proj = run["id"], run.get("project") or ""
    errors: list[str] = []
    out: dict[str, Any] = {"id": rid, "project": proj, "url": None, "evidence": [], "errors": errors}
    hit = api.get(rid)
    lst = listed_projects.get(proj) or {"ok": False, "n": 0, "error": "project not listed"}
    out["listing_ok"] = bool(lst.get("ok")) and lst.get("n", 0) > 0
    qs = queue_urls.get(rid, [])
    for ent, qproj, url, path in qs:
        if qproj != proj or ent != ENTITY:
            errors.append(f"queue-log URL project/entity mismatch: {ent}/{qproj} != {ENTITY}/{proj} ({path})")
    if hit is not None and hit.get("project") != proj:
        errors.append(f"API lists id under project {hit.get('project')!r}, local header project {proj!r}")
    if hit is not None and hit.get("project") == proj:
        out["tier"] = "api-confirmed"
        out["url"] = hit.get("url") or run_url(ENTITY, proj, rid)
        out["api_name"], out["api_state"] = hit.get("name"), hit.get("state")
        out["name_mismatch"] = (hit.get("name") or "") != (run.get("name") or "")
        out["evidence"].append(f"api listing {ENTITY}/{proj}")
    elif qs and not errors:
        out["tier"] = "queue-log"
        out["url"] = qs[0][2]
        out["evidence"] += [f"logs/queue: {p}" for _, _, _, p in qs]
        if out["listing_ok"]:
            out["finding"] = "queue-log URL but id absent from a complete API listing"
        else:
            out["finding"] = "queue-log URL; API listing unavailable for project (not confirmed absent)"
    elif run.get("mode") == "offline" and not run.get("synced_marker"):
        out["tier"] = "no-remote"
        out["evidence"].append("offline-run dir, no .synced marker, no API hit, no queue-log URL")
    else:
        out["tier"] = "constructed-unverified"
        out["url"] = run_url(ENTITY, proj, rid) if proj and proj not in ("?", "unknown") else None
        out["evidence"].append("URL constructed from pattern; not confirmed")
        if not out["listing_ok"]:
            out["evidence"].append("API listing unavailable/empty for this project -- not confirmed absent")
        else:
            out["listing_complete_id_absent"] = True
    out["url_status"] = {"api-confirmed": "confirmed", "queue-log": "printed-in-log-not-api-confirmed",
                         "constructed-unverified": "constructed-guess", "no-remote": "none"}[out["tier"]]
    if out["tier"] == "no-remote" and out["url"]:
        errors.append("URL emitted for a no-remote run")
    return out


# --------------------------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------------------------
# Primary sources: on-disk mtimes of the rebuilt dataset files (checked 2026-11-10, all UTC):
#   data/sigma_k_10/<k> (k != 11, 13): files written 2026-07-20 13:18:37..13:18:39Z (ord>k fix rebuild)
#   data/sigma_k_10/11: 2026-07-21 01:59:20Z (k=11 stayed on the 07-04 buggy build until then)
#   data/sigma_k_10/13: 2026-08-13 07:58:06Z (dir created then; no earlier build exists on disk)
#   data/sigma_k/<k>  : 2026-07-20 13:29:38..39Z (path regenerated as n=10; before that it was the
#                        deprecated n=20 build -> legacy, "deprecated-n20")
#   data/power_permutation/{d1,d0_sealed}/<k>: 2026-08-06 09:41:09..13Z (clean-era builder)
# plus memory sigmak-data-filter-off-disk.md and trm-mcp provenance._PATH_CUTOFFS.
# Rule: legacy  iff started < window_start (pre-fix build was on disk when the run loaded data);
#       clean   iff started >= window_end (fixed build complete) and the k-dir was not rewritten
#                   after the run started;
#       unknown otherwise (inside the rebuild window, no start time, unrecognised data path,
#                   multiple paths disagreeing).
#       other   ARC / Sudoku data (sigma^k data-epoch concept does not apply).
def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


COHORT_WINDOWS = [  # (prefix regex, window_start, window_end, legacy_note) most specific first
    (r"^data/sigma_k_10/11$", _t("2026-07-21T01:59:20"), _t("2026-07-21T01:59:21"), "pre-fix build (07-04)"),
    (r"^data/sigma_k_10/13$", None, _t("2026-08-13T07:58:07"), None),
    (r"^data/sigma_k_10/\d+$", _t("2026-07-20T13:18:00"), _t("2026-07-20T13:18:40"), "pre-fix ord>2 filter"),
    (r"^data/sigma_k/\d+$", _t("2026-07-20T13:29:00"), _t("2026-07-20T13:29:40"), "deprecated-n20 pre-07-20 build"),
    (r"^data/power_permutation/(d1|d0_sealed)/\d+$", None, _t("2026-08-06T09:41:14"), None),
]
COHORT_RULE_TEXT = (
    "cohort from config data_paths + the run's own recorded start time (UTC) against the on-disk rebuild "
    "window of that data dir: sigma_k_10/<k> legacy if started < 2026-07-20T13:18:00Z, clean if >= 13:18:40Z "
    "(k=11: legacy < 2026-07-21T01:59:20Z, clean >= 01:59:21Z; k=13: clean >= 2026-08-13T07:58:07Z, else unknown); "
    "sigma_k/<k> legacy (deprecated n=20) if started < 2026-07-20T13:29:00Z, clean (n=10) if >= 13:29:40Z; "
    "power_permutation/{d1,d0_sealed}/<k> clean if started >= 2026-08-06T09:41:14Z else unknown; "
    "ARC/Sudoku -> other; inside a window, no start time, no data_paths, unrecognised path, or disagreeing "
    "multi-path -> unknown. Project name is NOT used."
)


def parse_started(s: Any) -> dt.datetime | None:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return dt.datetime.fromtimestamp(s, UTC)
    if not isinstance(s, str) or not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def norm_data_paths(dp: Any) -> list[str]:
    if dp is None:
        return []
    if isinstance(dp, str):
        s = dp.strip().strip("'\"")
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        items = [x.strip().strip("'\"") for x in s.split(",")]
    elif isinstance(dp, (list, tuple)):
        items = [str(x).strip().strip("'\"") for x in dp]
    else:
        return []
    return [x.rstrip("/") for x in items if x]


def cohort_for_path(p: str, started: dt.datetime | None, kdir_mtime: dt.datetime | None = None):
    low = p.lower()
    if "arc" in low.split("/")[-1] or "sudoku" in low:
        return "other", "ARC/Sudoku data domain"
    for rx, ws, we, note in COHORT_WINDOWS:
        if re.match(rx, p):
            if started is None:
                return "unknown", "no recorded start time"
            if kdir_mtime is not None and kdir_mtime > started and (ws is None or started >= ws):
                return "unknown", f"data dir rewritten after run start ({kdir_mtime.isoformat()})"
            if ws is not None and started < ws:
                return "legacy", note
            if started >= we:
                return "clean", "started after rebuild window"
            return "unknown", "started inside rebuild window or before data dir existed"
    return "unknown", f"unrecognised data path {p!r}"


def cohort(data_paths: Any, started: Any, kdir_mtimes: dict | None = None) -> tuple[str, str]:
    paths = norm_data_paths(data_paths)
    if not paths:
        return "unknown", "no data_paths"
    st = parse_started(started)
    res = [cohort_for_path(p, st, (kdir_mtimes or {}).get(p)) for p in paths]
    labels = {r[0] for r in res}
    if len(labels) == 1:
        return res[0][0], "; ".join(sorted({r[1] for r in res}))
    return "unknown", "multi-path disagreement: " + "; ".join(f"{p}={r[0]}" for p, r in zip(paths, res))


# --------------------------------------------------------------------------------------------
# File names / identity
# --------------------------------------------------------------------------------------------
_SAFE = re.compile(r"[^A-Za-z0-9._=+-]+")


def sanitize(s: Any, fallback: str = "_unnamed") -> str:
    s = "" if s is None else str(s)
    s = _SAFE.sub("_", s).strip("._")
    s = re.sub(r"_+", "_", s)
    return (s or fallback)[:150]


def project_dir(project: Any) -> str:
    if project in (None, "", "?", "unknown"):
        return NO_PROJECT_DIR
    return sanitize(project, NO_PROJECT_DIR)


def card_relpath(project: Any, name: Any, rid: str) -> str:
    return f"{project_dir(project)}/{sanitize(name)}__{sanitize(rid)}.md"


def collisions(paths: Iterable[str]) -> list[list[str]]:
    by: dict[str, list[str]] = {}
    for p in paths:
        by.setdefault(p.lower(), []).append(p)
    return [v for v in by.values() if len(v) > 1]


def sibling_key(row: dict) -> str:
    """Same project + same config except seed. Ignored besides seed: run_name, run_group,
    checkpoint paths, wandb/output paths (anything in SIBLING_IGNORE)."""
    cfg = {k: v for k, v in (row.get("config_flat") or {}).items() if k not in SIBLING_IGNORE}
    return row.get("project", "") + "|" + json.dumps(cfg, sort_keys=True, default=str)


SIBLING_IGNORE = {"seed", "run_name", "run_group", "cell_id", "checkpoint_path", "load_checkpoint",
                  "project_name", "entity", "wandb_dir", "output_dir", "_wandb"}


def filter_cohort(rows: list[dict], exclude: Iterable[str]) -> list[dict]:
    ex = set(exclude or ())
    return [r for r in rows if r.get("cohort") not in ex]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# Card rendering
# --------------------------------------------------------------------------------------------
METRIC_DEFS = [
    ("probe/test_exact", "exact-match accuracy on a fixed 512-example probe subset of the test split, logged "
     "during training (memory test-exact-metric-provenance); NOT the full test set and NOT a sealed eval"),
    ("probe/train_exact", "exact-match accuracy on a fixed probe subset of the train split, logged during training"),
    ("all/exact_accuracy", "exact-match accuracy over the full evaluation set at an eval step, online (raw, "
     "non-EMA) weights; differs from probe/test_exact in example set and weights"),
    ("all/accuracy", "per-token accuracy over the full evaluation set (online weights)"),
    ("all/lm_loss", "token cross-entropy loss over the evaluation set"),
    ("all/q_halt_accuracy", "accuracy of the halting head's prediction on the evaluation set"),
    ("all/q_halt_loss", "halting-head loss on the evaluation set"),
    ("all/steps", "mean number of recurrent (ACT) steps used on the evaluation set"),
    ("train/exact_accuracy", "exact-match accuracy on the current training batch (online weights)"),
    ("train/accuracy", "per-token accuracy on the current training batch"),
    ("train/lm_loss", "token cross-entropy loss on the current training batch"),
    ("train/q_halt_accuracy", "halting-head accuracy on the training batch"),
    ("train/q_halt_loss", "halting-head loss on the training batch"),
    ("train/steps", "mean recurrent (ACT) steps on the training batch"),
    ("train/lr", "learning rate at that step"),
    ("train/count", "value logged under train/count (normaliser of the training-batch metrics)"),
    ("num_params", "model parameter count"),
    ("z/eff_rank", "effective rank of the latent z (logged only when log_z_dynamics=true)"),
    ("z/pca_top2_var", "fraction of z variance in the top-2 principal components (log_z_dynamics only)"),
    ("z/mean_norm", "mean L2 norm of z (log_z_dynamics only)"),
    ("phase/index", "phase index derived from z dynamics (log_z_dynamics only)"),
    ("z/delta_step_<N>", "change in z at recurrent step N (log_z_dynamics only)"),
]


# Definitions carry one of three labels (memory test-exact-metric-provenance.md is the only evidence):
#  * VERIFIED_DEFS: self-evident keys (num_params, train/lr): verified for every project.
#  * "verified" for probe/test_exact and all/exact_accuracy: ONLY the run tf noz_iter k2_s1 of project Sigma_k_fig12
#    (`fig1_tf_noz_iter_k2_s1`), where the example-set sizes (500/512) and weights were measured.
#  * "assumed extension": the other tf cells of the fig1 grid in Sigma_k_fig12 (`fig1_tf_<z|noz>_<iter|noiter>_k*_s*`;
#    values of the three metrics agree on 74 manifest cells, but sizes/weights were not measured per cell, and the 22
#    noz_noiter cells are wandb-probe-only). mlp variants and every other run: not covered by the note -> gloss.
#  * everything else: working gloss from the key name (generic text, no size/weight claims).
VERIFIED_DEFS = {"num_params", "train/lr"}
VERIFIED_BY_PROJECT = {"probe/test_exact": {"Sigma_k_fig12"}, "all/exact_accuracy": {"Sigma_k_fig12"}}
VERIFIED_RUN = "fig1_tf_noz_iter_k2_s1"
FIG1_TF_GRID_RE = re.compile(r"^fig1_tf_(z|noz)_(iter|noiter)_k\d+_s\d+$")
FIG1_GRID_RE = FIG1_TF_GRID_RE            # kept name; the scope no longer includes mlp
GENERIC_DEFS = {
    "probe/test_exact": "exact-match accuracy on a probe subset of the test split, logged during training "
                        "(subset size and weights not verified for this project)",
    "all/exact_accuracy": "exact-match accuracy over the evaluation set at an eval step "
                          "(example set and weights not verified for this project)",
}
UNVERIFIED_SUFFIX = " *(working gloss from the key name / code reference; not verified against pretrain.py)*"
VERIFIED_SCOPE_SUFFIX = (" *(verified on this run only (tf noz_iter k2_s1: example-set sizes and weights measured); "
                         "memory test-exact-metric-provenance)*")
ASSUMED_SUFFIX = (" *(assumed extension of a verified definition \u2014 not independently checked; the definition was "
                  "measured on run tf noz_iter k2_s1 only)*")
LABELS = ("verified on this run only", "assumed extension of a verified definition \u2014 not independently checked",
          "working gloss from the key name / code reference; not verified against pretrain.py")


def metric_def_tier(key: str, project: str | None, name: str | None) -> str:
    """verified | assumed | gloss for the two memory-sourced keys; verified for VERIFIED_DEFS; gloss otherwise."""
    if key in VERIFIED_DEFS:
        return "verified"
    if key in VERIFIED_BY_PROJECT and project in VERIFIED_BY_PROJECT[key] and name:
        if name == VERIFIED_RUN:
            return "verified"
        if FIG1_TF_GRID_RE.match(name):
            return "assumed"
    return "gloss"


def metric_def(key: str, project: str | None = None, name: str | None = None) -> str | None:
    for k, d in METRIC_DEFS:
        if k == key or (k.endswith("<N>") and key.startswith(k[:-3]) and key[len(k) - 3:].isdigit()):
            if k in VERIFIED_DEFS:
                return d
            if k in VERIFIED_BY_PROJECT:
                t = metric_def_tier(k, project, name)
                if t == "verified":
                    return d + VERIFIED_SCOPE_SUFFIX
                if t == "assumed":
                    return d + ASSUMED_SUFFIX
                return GENERIC_DEFS[k] + UNVERIFIED_SUFFIX
            return d + UNVERIFIED_SUFFIX
    return None


def metric_def_is_verified(key: str, project: str | None, name: str | None = None) -> bool:
    return metric_def_tier(key, project, name) == "verified"


def fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def yaml_scalar(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


def to_yaml(d: Any, indent: int = 0) -> str:
    """Tiny deterministic YAML emitter (JSON-quoted scalars => always valid YAML)."""
    pad = "  " * indent
    lines = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and v:
                lines.append(f"{pad}{k}:")
                lines.append(to_yaml(v, indent + 1))
            elif isinstance(v, list) and v and any(isinstance(x, (dict, list)) for x in v):
                lines.append(f"{pad}{k}:")
                for x in v:
                    sub = to_yaml(x, indent + 2).lstrip()
                    lines.append(f"{pad}  - {sub}")
            else:
                lines.append(f"{pad}{k}: {yaml_scalar(v)}")
    else:
        lines.append(pad + yaml_scalar(d))
    return "\n".join(lines)


def extract_prose(old_text: str | None) -> str:
    if not old_text or PROSE_START not in old_text or PROSE_END not in old_text:
        return ""
    return old_text.split(PROSE_START, 1)[1].split(PROSE_END, 1)[0].strip("\n")


# ---------------------------------------------------------------------------------------------- Summary (script-rendered)
SUMMARY_TITLE = "## Summary (script-rendered from this card's own fields; no interpretation)"
SUMMARY_PRIORITY = ("probe/test_exact", "probe/train_exact", "all/exact_accuracy", "train/exact_accuracy", "train/accuracy",
                    "all/lm_loss", "train/lm_loss", "train/lr", "num_params", "phase/index", "z/eff_rank")
# Fixed caveat templates keyed by frontmatter flags. {x} are filled from the card's own fields only.
T_NO_SUMMARY = "The file files/wandb-summary.json is absent in the local run directory; the local files do not record an end state."
T_NO_SUMMARY_API = "The wandb API record for this run lists api_state {api_state}."
T_END_STATE = ("The recorded end-state fields are as listed in the first sentence (local status, local exit_code, wandb API api_state); "
               "this card does not reconcile them.")
END_STATE_RULE_TEXT = (
    "end_state_fields_differ is true iff (status == 'ended' and exit_code is present and != 0) or link.api_state is one of "
    "crashed / failed / killed; false otherwise. It marks cards whose recorded end-state fields should be read together; "
    "it is not a verdict about the run.")
T_NO_META = "No files/wandb-metadata.json is present; identity was recovered from the .wandb record header."
T_HIST_EMPTY = "The local .wandb file contains no history records."
T_TRUNC = ("Local history ends at step {history_last_step}, before the summary step {summary_last_step}; "
           "the last values quoted here are not the run's final step.")
T_SNAP = "Partial snapshot taken at {snapshot_utc}; the run may still have been in progress, so these values are not final."
T_DUP = "Other runs share this project and run name: {ids}."
T_MULTI = "The attribution map assigns this run to several experiments ({ids}); this is unresolved here."
T_LEGACY = ("Cohort legacy per the cohort rule: data path {path} and start time {started} are before {window_start}, the start of "
            "the rebuild window of that data dir (rule: cohort_rule in the frontmatter, common.py COHORT_WINDOWS, _meta/README.md).")
T_LEGACY_GENERIC = "Cohort legacy per the cohort rule (cohort_rule and cohort_reason in the frontmatter; rule text in _meta/README.md)."
T_COHORT_UNKNOWN = "Cohort unknown: the cohort_reason field of this card gives the rule outcome."
T_TIER_OFFLINE = "The link tier is no-remote: this is an offline run with no remote URL."
T_TIER = "The link tier is {tier}; the URL is not api-confirmed."
T_NO_ROWS = "No metric rows were recorded in the local history."
T_ATTR_NONE = "The card lists no experiment or hypothesis attribution."
T_ATTR = "DRAFT attribution map, not evidence: {items}."


def _bt(v: Any) -> str:
    """Backticked value as the frontmatter spells it: YAML booleans true/false, every other value verbatim (a string "True" stays "True")."""
    if isinstance(v, bool):
        v = "true" if v else "false"
    return f"`{v}`"


def end_state_fields_differ(status: Any, exit_code: Any, api_state: Any) -> bool:
    nonzero = isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and exit_code != 0
    return bool((status == "ended" and nonzero) or api_state in ("crashed", "failed", "killed"))


def legacy_sentence(fm: dict) -> str:
    st = parse_started(fm.get("started"))
    for p in norm_data_paths((fm.get("config") or {}).get("data_paths")):
        for rx, ws, _we, _note in COHORT_WINDOWS:
            if re.match(rx, p):
                if ws is not None and st is not None and st < ws:
                    return T_LEGACY.format(path=_bt(p), started=_bt(st.strftime("%Y-%m-%dT%H:%M:%SZ")),
                                           window_start=_bt(ws.strftime("%Y-%m-%dT%H:%M:%SZ")))
                break
    return T_LEGACY_GENERIC


def summary_parts(fm: dict, stats: dict | None) -> dict:
    """{'config': str, 'metrics': [str], 'caveats': [str], 'attribution': str}; pure function of the frontmatter and the
    stats rows (the same rows and fmt_num as the stats table, so no value is re-rounded)."""
    cfg = fm.get("config") or {}
    bits = [f"Run {_bt(fm['name'])}" if fm.get("name") else "Run with no name", f"id {_bt(fm['id'])}", f"in project {_bt(fm['project'])}"]
    cfgs = []
    if cfg.get("arch_preset"):
        cfgs.append(f"model variant {_bt(cfg['arch_preset'])}")
    if cfg.get("arch_name"):
        cfgs.append(f"arch name {_bt(cfg['arch_name'])}")
    for lab, key in (("k", "k"), ("seed", "seed"), ("H cycles", "H_cycles"), ("L cycles", "L_cycles")):
        if cfg.get(key) is not None:
            cfgs.append(f"{lab} {_bt(cfg[key])}")
    if cfg.get("data_paths"):
        dp = cfg["data_paths"] if isinstance(cfg["data_paths"], list) else [cfg["data_paths"]]
        cfgs.append("data path " + ", ".join(_bt(x) for x in dp))
    for lab, key in (("ema", "ema"), ("log_z_dynamics", "log_z_dynamics")):
        if cfg.get(key) is not None:
            cfgs.append(f"{lab} {_bt(cfg[key])}")
    if fm.get("started"):
        cfgs.append(f"started {_bt(str(fm['started'])[:10])}")
    api_state = (fm.get("link") or {}).get("api_state")
    cfgs.append(f"local status {_bt(fm.get('status'))}")
    if fm.get("exit_code") is not None:
        cfgs.append(f"local exit_code {_bt(fm['exit_code'])}")
    if api_state is not None:
        cfgs.append(f"wandb API api_state {_bt(api_state)}")
    config = " ".join(bits) + "; " + "; ".join(cfgs) + "."

    metrics = []
    keys = ((stats or {}).get("keys") or {}) if (stats or {}).get("status") == "ok" and not fm.get("history_empty") else {}
    order = [k for k in SUMMARY_PRIORITY if k in keys] + sorted(k for k in keys if k not in SUMMARY_PRIORITY)
    for k in order:
        d = keys[k]
        cell = lambda a, b: f"{fmt_num(d.get(a))} (@{fmt_num(d.get(b))})"      # noqa: E731  identical to the stats table
        parts = [f"first {cell('first', 'first_step')}", f"last {cell('last', 'last_step')}"]
        seen = {fmt_num(d.get("first")), fmt_num(d.get("last"))}
        for lab, a, b in (("min", "min", "min_step"), ("max", "max", "max_step")):
            if fmt_num(d.get(a)) not in seen:
                parts.append(f"{lab} {cell(a, b)}")
        metrics.append(f"- {_bt(k)}: " + ", ".join(parts))
    cav = []
    if fm.get("status") == "no-summary":
        cav.append(T_NO_SUMMARY + (" " + T_NO_SUMMARY_API.format(api_state=_bt(api_state)) if api_state is not None else ""))
    if fm.get("end_state_fields_differ"):
        cav.append(T_END_STATE)
    if fm.get("has_metadata") is False:
        cav.append(T_NO_META)
    if fm.get("history_empty"):
        cav.append(T_HIST_EMPTY)
    if fm.get("history_truncated"):
        cav.append(T_TRUNC.format(**{k: fm.get(k) for k in ("history_last_step", "summary_last_step")}))
    if fm.get("snapshot"):
        cav.append(T_SNAP.format(snapshot_utc=fm.get("snapshot_utc")))
    if fm.get("duplicates"):
        cav.append(T_DUP.format(ids=", ".join(str(d["id"]) for d in fm["duplicates"])))
    at = fm.get("attribution") or {}
    if fm.get("multi_attribution"):
        cav.append(T_MULTI.format(ids=", ".join(str(e) for e in at.get("exps") or [])))
    tier = (fm.get("link") or {}).get("tier")
    if tier == "no-remote":
        cav.append(T_TIER_OFFLINE)
    elif tier != "api-confirmed":
        cav.append(T_TIER.format(tier=tier))
    if fm.get("cohort") == "legacy":
        cav.append(legacy_sentence(fm))
    elif fm.get("cohort") == "unknown":
        cav.append(T_COHORT_UNKNOWN)
    items = []
    for lst, det in ((at.get("exps") or [], at.get("exp_attribution") or []), (at.get("hyps") or [], at.get("hyp_attribution") or [])):
        info = {str(x.get("id")): x for x in det if isinstance(x, dict)}
        for i in lst:
            x = info.get(str(i))
            items.append(f"{i} (source {x.get('source')}, confidence {x.get('confidence')})" if x else str(i))
    attribution = T_ATTR.format(items="; ".join(items)) if items else T_ATTR_NONE
    return {"config": config, "metrics": metrics, "caveats": cav, "attribution": attribution}


def render_summary(fm: dict, stats: dict | None) -> list[str]:
    p = summary_parts(fm, stats)
    out = [SUMMARY_TITLE, "", p["config"], ""]
    out += ["Values below are the first and last (and, where different, min and max) entries of the stats table, copied unchanged:", ""]
    out += p["metrics"] or [T_NO_ROWS]
    out.append("")
    out += [f"- {c}" for c in p["caveats"]]
    if p["caveats"]:
        out.append("")
    out += [p["attribution"], ""]
    return out


def render_card(fm: dict, stats: dict | None, related: list[tuple[str, str]], old_text: str | None,
                notes: list[str], attr_lines: list[str] | None = None) -> str:
    """fm: frontmatter dict; stats: {"keys": {key: {...}}, "status": str, ...} or None."""
    out = ["---", to_yaml(fm), "---", ""]
    out.append(f"# Run card — `{fm['name']}` ({fm['project']} / {fm['id']})")
    out.append("")
    out.append("> **Script-generated** by `tools/wandb_run_index/cards.py` (Stage A). EXP/H attribution is a "
               "**DRAFT** map (`confidence` verbatim), **not evidence**. Numbers below are computed by script "
               "from the local `.wandb` file; no interpretation is made here. Link tier: "
               f"**{fm['link']['tier']}** ({fm['link']['url_status']}).")
    out.append("")
    for n in notes:
        out.append(f"- ⚠ {n}")
    if notes:
        out.append("")
    for ln in attr_lines or []:
        out.append(f"- {ln}")
    if attr_lines:
        out.append("")
    out += render_summary(fm, stats)
    out.append("## Symbols")
    out.append("")
    out.append("| symbol | definition |")
    out.append("|---|---|")
    base = [("step", "the wandb `_step` value of a history record (training-loop logging step)"),
            ("first / last", "value at the smallest / largest step at which the key was logged"),
            ("min / max (@step)", "extreme finite value over ALL logged points (full scan, no downsampling); "
             "ties -> earliest step"),
            ("n", "number of logged points for the key; `nonfinite` = NaN/inf points, excluded from min/max"),
            ("cohort", "data-epoch cohort (clean / legacy / other / unknown), rule in frontmatter `cohort_rule`")]
    keys = sorted((stats or {}).get("keys", {}) or {})
    for k, d in base:
        out.append(f"| {k} | {d} |")
    for k in keys:
        out.append(f"| `{k}` | {metric_def(k, fm.get('project'), fm.get('name')) or 'allow-listed key (definition not recorded)'} |")
    out.append("")
    out.append("## Precomputed stats (full-resolution, script-computed)")
    out.append("")
    if fm.get("snapshot"):
        out.append(f"Snapshot: this run was possibly still training at snapshot time. The table numbers come from "
                   f"the scan cache (`_meta/scan_cache.jsonl`, scanned at snapshot `{fm.get('snapshot_utc')}`); "
                   "frontmatter `last_step`/`local.wandb_*` come from the index (`_wandb_runs_index.jsonl`), which may "
                   "be older (see the sha256 note above if present). Values will move until the run ends.")
        out.append("")
    if not stats or stats.get("status") != "ok":
        reason = (stats or {}).get("status") or "no scan result"
        out.append(f"Stats unavailable: `{reason}`.")
    elif fm.get("history_empty"):
        out.append("The local .wandb file contains no history records "
                   f"(history_records = 0; wandb_size = {fm.get('local', {}).get('wandb_size')} bytes; "
                   f"summary `_step` = {fm.get('last_step')}).")
    elif not keys:
        out.append("History records exist, but no allow-listed metric key was logged in this run's history.")
    else:
        if fm.get("history_truncated"):
            out.append(f"Note: the `last` values in this table come from the local `.wandb` history, which ends "
                       f"before the summary step (`wandb-summary.json` `_step` = {fm.get('summary_last_step')}; "
                       f"largest `_step` in the local history = {fm.get('history_last_step')}; "
                       f"largest step in this table = {fm.get('stats_table_last_step')}).")
            out.append("")
        out.append("| key | n | first (@step) | last (@step) | min (@step) | max (@step) | nonfinite |")
        out.append("|---|---:|---|---|---|---|---:|")
        for k in keys:
            s = stats["keys"][k]
            cell = lambda a, b: f"{fmt_num(s.get(a))} (@{fmt_num(s.get(b))})"
            out.append(f"| `{k}` | {s.get('n', 0)} | {cell('first', 'first_step')} | {cell('last', 'last_step')} | "
                       f"{cell('min', 'min_step')} | {cell('max', 'max_step')} | {s.get('nonfinite', 0)} |")
    out.append("")
    out.append("## Related documents")
    out.append("")
    if related:
        for label, rel in related:
            out.append(f"- [{label}]({rel})")
    else:
        out.append("- (none found on disk)")
    out.append("")
    out.append("## Prose (filled by a later batch stage from the numbers above only)")
    out.append("")
    out.append(PROSE_START)
    prose = extract_prose(old_text)
    if prose:
        out.append(prose)
    out.append(PROSE_END)
    out.append("")
    return "\n".join(out)
