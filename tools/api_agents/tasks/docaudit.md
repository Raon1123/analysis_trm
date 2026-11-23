# Task: audit {{N_FILES}} living documents of the TRM documentation tree

Date: {{DATE}}. {{RUN_NOTE}}

## Inputs (read-only)
- Corpus: {{N_FILES}} files under {{CORPUS_ROOT}}/, about {{EST_TOKENS}} tokens in total. Their list, one absolute path per line: {{FILES_LIST}}.
- Project context: {{CONTEXT_FILE}}. Read it first. It explains the tree's layout, which document wins when two disagree, which documents are living and which are frozen records, the identifier scheme, and dated facts that make older text stale.

## What to do
Start a workflow run with two phases.

1. **read** — Split the file list into shards of at most {{SHARD_TOKENS}} estimated tokens (estimate 1 token per 1.5 bytes; `wc -c` gives bytes). Keep files of the same top-level category (the first path component under {{CORPUS_ROOT}}/) together and in list order. Give each shard to one TRM Reader with the shard's absolute paths and the output path `/workspace/work/read/<shard_id>.json`.
2. **check** — For every shard, one TRM Checker gets the reader's result path, the shard's source paths and the output path `/workspace/work/check/<shard_id>.json`. It reviews every finding, not only some, from the reader's result and the places it cites: for each finding it opens the quoted line and a few lines around it in the cited file, and does the same for the counter-quote and evidence. It does not re-read the shard. It adds a finding (`added`) only for a clear defect it meets inside a window it opened.

{{WORKER_BRIEFS}}

Then, outside the run, combine: keep findings marked `keep`, `revise` (with the checker's corrections applied) or `added`; drop the rest. Re-check every kept quote and counter-quote with a script (exact substring of the named file, or the same text with whitespace runs collapsed); drop any that fail and count them. Limits for the run: at most {{MAX_PARALLEL}} threads working at once, a one-hour time limit, one agent failing does not fail the run (list its shard as not covered).

## Result formats
Reader result (`/workspace/work/read/<shard_id>.json`):
`{"shard_id": str, "files_read": [{"file": <absolute path>, "last_line_read": int, "total_lines": int}], "findings": [<finding>], "notes": str}`

A finding:
`{"file": <absolute path>, "quote": str, "category": "wrong"|"inconsistent"|"inefficient"|"obsolete", "basis": "verified"|"needs-check", "severity": "high"|"medium"|"low", "frozen": bool, "claim": str, "counter_file": <absolute path or "">, "counter_quote": str, "evidence": [<"path:line">], "check": str}`

Checker result (`/workspace/work/check/<shard_id>.json`; `added` holds only clear defects met inside a window the checker opened, each with an exact quote):
`{"shard_id": str, "reviews": [{"index": <position in the reader's findings>, "verdict": "keep"|"drop"|"revise", "reason": str, "revised": <finding or null>}], "added": [<finding>], "notes": str}`

## Deliverables in {{OUTPUT_DIR}}/
- `findings.jsonl` — one kept finding per line, the finding fields above plus `"shard_id"` and `"checker_verdict"` (`keep` | `revise` | `added`). Paths stay absolute as mounted.
- `coverage.json` — `{"shards": [{"shard_id", "files": [...], "reader": "ok"|"failed", "checker": "ok"|"failed", "n_reader_findings", "n_kept", "n_dropped"}], "files_not_read_to_end": [...], "quote_check_dropped": int, "not_covered": [...]}`.
- `REPORT.md` — in Korean, at most 80 lines: counts by category and severity, the high-severity findings (one line each: file, claim), what was not covered, and how many findings the checker and the quote script dropped.

## Rules for every agent
<!-- WORKER_RULES_BEGIN -->
- Every file in this task is a LIVING document (an index, a README, a governance page, a registry, a schema, a template or a runbook) that people and agents read to find out what is true now. Report what in them is wrong, inconsistent, inefficient or no longer needed.
- Categories: `wrong` = a statement contradicted by another file, by arithmetic, or by its own cited source. `inconsistent` = two places that should agree and do not (a status word for the same identifier, a rule stated two ways, frontmatter against body, an index against the entries it lists). `inefficient` = the same content kept in several places, a required step that duplicates another or cannot be followed, a living document that keeps closed history inline. `obsolete` = superseded versions, finished one-off scratch, text about components the context file says were retired, a completed step described as still to come.
- `basis: "verified"` only when you found the contradicting or confirming text yourself in the mounted corpus (give it in `counter_file` / `counter_quote` and `evidence`). `basis: "needs-check"` when settling it needs something outside the corpus (a queue, a remote service, a file that is not mounted); say in `check` exactly what to look at.
- Quote `quote` and `counter_quote` exactly, character for character, from a single line where possible, at most 200 characters, no ellipsis, no correction, no translation. A script searches the file for the exact string.
- Not a finding: anything you examined and found fine or already corrected by a banner or a later section of the same file; a history, changelog, "이력" or dated-entry section inside a living document; a dated record that was true when written; old flat-style paths (such as `lab/reports/`) inside a dated record; typos, wording, formatting; anything you would have to guess at. Before reporting a number from a table, copy the column heading it sits under into `claim`.
- `frozen`: true for a historical or generated record that must not be edited in place (the context file lists the kinds), false for a living document.
- `severity`: `high` if a wrong reading would misstate the state of an experiment or send someone to the wrong next step, `medium` if it would waste a session's time, `low` otherwise.
- At most {{MAX_FINDINGS_PER_FILE}} findings per file, the most consequential first. Write `claim`, `check`, `reason` and `notes` in Korean; keep identifiers, paths and quotes verbatim.
- Keep the text short (what you write is paid for as output and then carried as context on every later turn): `claim` at most two sentences; `check`, `reason` and `notes` one sentence each. Do not restate file contents and do not list what you found fine.
<!-- WORKER_RULES_END -->
