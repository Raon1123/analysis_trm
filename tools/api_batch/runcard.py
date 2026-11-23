"""runcard lane: wandb run-card prose (Stage B of lab/22_plans/.../23_2026-11-02_wandb-runs-results-docs).

One request per run card (lab/21_experiments/12_run-cards/<project>/<run_name>__<id>.md), shared cached
[system, header] prefix, the whole card as the uncached tail. The model writes prose only: every number it may quote
is already in the card's script-computed stats table (the deterministic post-check in
tools/wandb_run_index/prose_check.py enforces that; nothing here computes a number).

The card set is enumerated at plan time from the spec (`cards.root` + `cards.cohorts`), so a re-plan after the cards
change is `plan <batch_dir>` and nothing else. Cards with an empty stats table get NO request (a script fills a fixed
sentence at inject time; they are listed in plan.json -> runcard.skipped).

Pure helpers (parse_card, blanked_text, ...) are shared with the post-check/inject scripts.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from . import manifest as M
from .common import Halt, sha256_bytes, sha256_text

COHORTS = ("clean", "other", "legacy")
MARK_S, MARK_E = "<!-- prose:auto start -->", "<!-- prose:auto end -->"
CARDS_ROOT_REL = "21_experiments/12_run-cards"          # lab-relative default
_REGION = re.compile(re.escape(MARK_S) + r".*?" + re.escape(MARK_E), re.S)
_CELL = re.compile(r"^(\S+) \(@(\d+)\)$")
STATS = ("first", "last", "min", "max")
LANE = "runcard"
RULES_VERSION = "r4"        # post-check rule set the prompt describes (tools/wandb_run_index/prose_check.py RULE_SETS)


class RunCardError(Halt):
    pass


def blanked_text(text: str) -> str:
    """The card with its prose:auto region emptied to the Stage-A form (start / end on adjacent lines)."""
    if text.count(MARK_S) != 1 or text.count(MARK_E) != 1 or text.index(MARK_S) > text.index(MARK_E):
        raise RunCardError("card does not contain exactly one well-ordered prose:auto region")
    return _REGION.sub(f"{MARK_S}\n{MARK_E}", text, count=1)


def card_sha(text: str) -> str:
    """sha256 of the blanked card: stable across injection, changes when anything outside the region changes."""
    return sha256_text(blanked_text(text))


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise RunCardError("card has no frontmatter")
    end = text.index("\n---\n", 4)
    fm = yaml.safe_load(text[4:end]) or {}
    if not isinstance(fm, dict):
        raise RunCardError("card frontmatter is not a mapping")
    return fm, text[end + 5:]


def stats_table(text: str) -> dict[str, dict]:
    """key -> {n, first: (value, step), last, min, max, nonfinite}; all strings exactly as printed in the card."""
    if "## Precomputed stats" not in text:
        return {}
    sec = text.split("## Precomputed stats", 1)[1].split("\n## ", 1)[0]
    out = {}
    for ln in sec.splitlines():
        if not ln.startswith("| `"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 7:
            raise RunCardError(f"stats row with {len(cells)} cells: {ln[:60]!r}")
        row = {"n": cells[1], "nonfinite": cells[6]}
        for stat, c in zip(STATS, cells[2:6]):
            m = _CELL.match(c)
            if not m:
                raise RunCardError(f"stats cell {c!r} is not '<value> (@<step>)'")
            row[stat] = (m.group(1), m.group(2))
        out[cells[0].strip("`")] = row
    return out


def parse_card(text: str) -> dict:
    fm, _ = split_frontmatter(text)
    return {"fm": fm, "stats": stats_table(text)}


def region_text(text: str) -> str:
    blanked_text(text)  # validates the markers
    a, b = text.index(MARK_S) + len(MARK_S), text.index(MARK_E)
    return text[a:b].strip()


def read_ids(path: Path) -> list[str]:
    """One run id per line; blank lines and '#' comments (whole-line or trailing) ignored; duplicates halt."""
    ids = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            if not re.fullmatch(r"[A-Za-z0-9]{4,16}", ln):
                raise RunCardError(f"ids file: {ln!r} is not a run id")
            ids.append(ln)
    if len(set(ids)) != len(ids):
        raise RunCardError("ids file has duplicate ids")
    if not ids:
        raise RunCardError("ids file is empty")
    return ids


def select_cards(cards_root: Path, cohorts: list[str] | None, rel_to: Path | None = None, ids: list[str] | None = None,
                 include_injected: bool = True) -> tuple[list[dict], list[dict]]:
    """(selected, skipped) card descriptors, sorted by relative path. Selection = cohort filter (if `cohorts`) AND id
    list (if `ids`). Halts on a cohort outside clean/other/legacy (anywhere in the tree, not only in the selection: a
    changed rule must not slip through), an id/filename mismatch, an unreadable card, or an id that matches no card.
    Cards with no stats rows or `history_empty: true` are returned in `skipped` (reason recorded), never selected.
    With include_injected=False (the default for every batch: pilot -> inject -> re-plan clean/legacy drops pilot ids) a card whose prose region already holds model prose is dropped from
    both lists into `skipped` with reason 'already_injected'."""
    bad = [c for c in (cohorts or []) if c not in COHORTS]
    if bad or (cohorts is not None and not cohorts) or (cohorts is None and ids is None):
        raise RunCardError(f"need cards.cohorts (subset of {list(COHORTS)}) and/or an ids list; got {cohorts!r}")
    root = Path(cards_root)
    files = sorted(p for p in root.glob("*/*.md") if p.parent.name != "_meta")
    if not files:
        raise RunCardError(f"no cards under {root}")
    want = set(ids) if ids is not None else None
    found = set()
    sel, skip = [], []
    for p in files:
        text = p.read_text(encoding="utf-8")
        try:
            c = parse_card(text)
            sha = card_sha(text)
            reg = region_text(text)
        except RunCardError as e:
            raise RunCardError(f"{p.name}: {e}") from None
        fm = c["fm"]
        if fm.get("cohort") not in COHORTS:
            raise RunCardError(f"{p.name}: cohort {fm.get('cohort')!r} not in {list(COHORTS)}")
        if not p.name.endswith(f"__{fm.get('id')}.md"):
            raise RunCardError(f"{p.name}: frontmatter id {fm.get('id')!r} does not match the file name")
        rid = str(fm["id"])
        if want is not None:
            if rid not in want:
                continue
            found.add(rid)
        if cohorts is not None and fm["cohort"] not in cohorts:
            continue
        d = {"rel": p.relative_to(rel_to or root.parent.parent).as_posix(), "path": p, "id": rid, "name": fm.get("name"),
             "project": fm.get("project"), "cohort": fm["cohort"], "status": fm.get("status"),
             "n_rows": len(c["stats"]), "card_sha256": sha, "bytes_sha256": sha256_bytes(text.encode("utf-8")),
             "text": text}
        if fm.get("history_empty") is True:
            d["reason"] = "history_empty"
        elif not d["n_rows"]:
            d["reason"] = "no_stats_rows"
        elif not include_injected and reg:
            d["reason"] = "already_injected"
        (skip if d.get("reason") else sel).append(d)
    if want is not None and want != found:
        raise RunCardError(f"ids not found among cards: {sorted(want - found)[:5]}")
    return sel, skip


def plan_runcard(spec: dict, bd: Path, builder, scanned: list, reqs: list, schema_of) -> dict:
    """ops.plan branch. System prompt (repo prompts/), header (batch shared/), and each card go through
    build_verify_only (any L2/L3b hit refuses the whole plan, nothing is masked). The pack text is the card with its
    prose region blanked (a re-run never shows the model earlier prose). Selection = cards.cohorts and/or
    cards.ids_file (batch-relative, one run id per line); every batch skips cards that already hold injected prose
    (reported in plan.json skipped_already_injected) unless cards.include_injected is true."""
    from .common import REPO_ROOT
    tpl = REPO_ROOT / spec["prompt"]["system_template"]
    system = builder.build_verify_only("spec_text", "system", tpl.read_text(encoding="utf-8"), origin=tpl)
    hp = Path(bd) / spec["header"]
    header = builder.build_verify_only("spec_text", hp.stem, hp.read_text(encoding="utf-8"), origin=hp)
    sch = schema_of(spec["output_schema"])
    scanned += [system, header]
    cs = spec["cards"]
    croot = Path(cs["root"])
    croot = croot if croot.is_absolute() else Path(builder.lab_root) / croot
    ids, ids_sha = None, None
    if cs.get("ids_file"):
        ip = Path(bd) / cs["ids_file"]
        ids = read_ids(ip)
        ids_sha = sha256_bytes(ip.read_bytes())
    sel, skip = select_cards(croot, cs.get("cohorts"), Path(builder.lab_root), ids=ids,
                             include_injected=bool(cs.get("include_injected", False)))
    if not sel:
        raise RunCardError("no card with a non-empty stats table matched the selection")
    for d in sel:
        blank = blanked_text(d["text"])
        pk = builder.build_verify_only("seat_input", d["path"].name, blank, origin=d["path"], lane=LANE)
        scanned.append(pk)
        cid = M.make_custom_id("runcard", pk.text, d["id"], 1)
        reqs.append(M.build_request(cid, spec["model"], spec["effort"], spec["max_tokens"], system, [header], [pk],
                                    spec["cache_ttl"], sch,
                                    meta={"prompt_sha256": sha256_text(system.text), "header_sha256": header.pack_sha256,
                                          "rules": RULES_VERSION, "run_id": d["id"], "card_rel": d["rel"], "card_sha256": d["card_sha256"],
                                          "card_bytes_sha256": d["bytes_sha256"], "cohort": d["cohort"],
                                          "project": d["project"]}))
    keep = ("rel", "id", "name", "project", "cohort", "status", "card_sha256", "reason")
    count = lambda ds, k: {v: sum(1 for x in ds if x[k] == v) for v in sorted({x[k] for x in ds})}  # noqa: E731
    fixed = [d for d in skip if d["reason"] != "already_injected"]
    info = {"cards_root": str(cs["root"]), "cohorts": cs.get("cohorts"), "ids_file": cs.get("ids_file"),
            "ids_file_sha256": ids_sha, "n_ids": len(ids) if ids else None,
            "n_selected": len(sel), "n_skipped_no_stats": len(fixed),
            "n_skipped_already_injected": len(skip) - len(fixed),
            "selected_by_cohort": count(sel, "cohort"), "selected_by_status": count(sel, "status"),
            "skipped": [{k: d[k] for k in keep} for d in fixed],
            "skipped_already_injected": [{k: d[k] for k in keep} for d in skip if d["reason"] == "already_injected"],
            "skip_rule": "history_empty or stats table with 0 rows -> no request; prose_inject writes a fixed sentence"}
    return info
