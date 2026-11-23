# tools/timeline — stage-1 EXP-/H- chronological extractor

Deterministic and LLM-free. Reads the lab tree and writes JSONL, JSON, and text sidecars.
No judgement is produced here. `status_token` holds a verbatim status that appears in its own event's quote, and only for the self-recording cases listed below. Every other status text is `doc_status_at_read`, read at build time and not dated to the event. The authoritative rule text is `EXTRACTION_NOTES.md` in the report directory.

## Run

```bash
cd /home/ayp/project/trm
uv run --no-sync python tools/timeline/build_timeline.py                 # default out dir
uv run --no-sync python tools/timeline/build_timeline.py --out-dir <dir>  # scratch run
CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m pytest tests/test_build_timeline.py -q
```

Exit code 0 on success. Exit code 3 on a fail-closed configuration error (withheld registry missing or malformed).

Default output: `lab/23_reports/11_lab-reports/2026-10-31_exp-h-timeline/`

| file | content |
|---|---|
| `timeline_events.jsonl` | canonical events (copy-class sources excluded) |
| `timeline_copy_events.jsonl` | copy-class events (`is_copy: true`, `copy_reason`) |
| `timeline_copy_files.txt` | copy-class source paths, reason, event count |
| `timeline_no_event_files.txt` | every read file with zero events: path, reason, group |
| `timeline_stats.json` | counts, accounting, unparsed list, `source_tree_hash`, `same_day_mixed_precision_groups` |
| `EXTRACTION_NOTES.md` | rules, before/after lists, limits (authoritative) |

## Event fields

`event_id, date, date_tz, date_precision, date_basis, date_source, type, lane, ids{exp,h,atoms,other}, ids_ambiguous, aliases_seen, title, status_token, doc_status_at_read, quote, angle_tokens_in_doc, source_path, withheld, notes`

- `date`: normalised ISO text without the timezone marker. `date_tz` holds the marker that followed it in the source (`Z`, `+00:00`, `KST`, `UTC`) or null.
- `status_token`: only on `status-change` events from a registry Status cell that has exactly one ISO date with the token in the same clause, and on `verify` events whose verdict comes from the `verdict:` key of a verify return. Otherwise null.
- `doc_status_at_read`: the document's status text as read at build time, verbatim. It is not attributed to the event date.
- `type`: `register, prereg, freeze, enqueue, run-complete, judge, results, verify, verdict, status-change, decision, tool, plan, handoff, witness`.
  - `freeze`: a witness line with a freeze keyword (`freeze`, `frozen`, `동결`) after file-like tokens and code spans are removed, and without a negation or reference phrase (`not frozen`, `unfrozen`, `before/after the freeze`, `freeze manifest`, `미동결`, `동결 전`).
  - `witness`: any other timestamped witness-file line.
  - `judge`: only from `judged` or `judged_at` keys with an ISO date outside a path.
- Verify returns: result-time keys `audit_date_utc`, `audit_date`, `date_utc`, `audited_at`, and `verified_on` make verify events.
- `ids_ambiguous`: raw tokens not bound. These are paper-position EXP tokens, and tokens that are both a registry canonical and an alias string of another entry (EXP-019 under EXP-020). A registry section's own canonical is never ambiguous for its own section.

## Ordering

Events are sorted by normalised date, then `event_id`. A day-precision event sorts at 00:00:00, so on the same date it can sort before a timed event whose time is earlier in the day. The relative order of same-day events of different precision carries no meaning. `same_day_mixed_precision_groups` in stats counts the groups where this happens. No sort hack is applied.

## Source scope

Registry, hypothesis log, experiment docs (`21_experiments`), reports (`23_reports`), theory (`27_theory`), atoms INDEX, autoresearch plans and goal ledger, AUTORESEARCH.md, handoff INDEX (ARG/ARB rows only). Not scanned: `24_audits/`, plan directories outside the autoresearch list, and goal-state subprojects other than the goal ledger. Body text is not scanned for ids.

## Copy-class rule

`is_copy_path(rel)` returns a reason if a directory component is `benchmark`, `transcript_extracts`, `volatile_sources*`, `scratchpad_*`, `backup*`, `*_snapshot`, or `recovered`, or if the file name has a token (split on `-`, `_`, `.`) equal to `copy`. The earlier clause that treated VERIFY returns in `*_impl` directories as copies is removed. Matching events go to `timeline_copy_events.jsonl`. Canonical files never carry `is_copy`.

## Tests

`tests/test_build_timeline.py` (68 test cases including parametrised ones): date parser and tz, alias conversion and ambiguity, status anchoring and the registry clause rule, quote containment for status and doc status, withheld fail-closed, copy-path patterns and false-positive guards, `pc1_impl` VERIFY returns as canonical, copy-token file names, copy file separation, frontmatter id fallback, result-time precedence, verify result-time keys, run_date results as doc status, path-embedded dates, unknown-key reasons, nested date keys, freeze/witness typing with negation, handoff scope, no-event accounting, byte-identical rebuild, source-tree hash sensitivity to inputs, sort and id uniqueness.

## Layout

- `build_timeline.py` — extractor (single module; `build()` and `main()`).
- `__init__.py` — package marker.
