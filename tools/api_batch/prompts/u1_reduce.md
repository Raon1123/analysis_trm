You receive collected outputs of an earlier per-cell review: each item is one cell's `cell_summary` and `notes` (JSON), under
a short provenance header, preceded by the shared header (definitions and checklist). The items are machine-written drafts,
not evidence, and may contain errors. A final TASK line says which items you are to organise.

Rules
1. Use only the items and the shared header. Quote values exactly as they appear in the items; never recompute, round or
   combine numbers, and never go back to guess at raw data you cannot see. If you need arithmetic, say it needs a check.
2. Do not drop a note because it seems minor. Merge only notes that point at the same thing, and keep every cell's run name
   next to the merged note so a reader can open the item. If two items contradict each other, report both and the conflict.
3. Do not decide which scenario of the header's table applies and do not state a mechanism. Report agreements and
   disagreements among the cells and mark what each item's own wording supports and what it does not.
4. Write Markdown with exactly these sections and no others:
   - `## Scope` - which items you were given (count and run names).
   - `## What the curves show` - the common pattern across the given cells, and the differences between seeds (or arms, k).
   - `## Anomalies and inconsistencies` - a table: run name / locator, kind, severity, evidence (verbatim), suggested check.
   - `## Disagreements between items` - or "none".
   - `## Open questions for the verifier` - an ordered list, most useful first, each naming the file and columns to check.
   For a TASK that covers all cells, add `## Cross-cell digest` after `## Scope`: at most 10 bullets, each starting with the
   run names it rests on, covering the pattern across arms and k, outliers, and the notes that recur most.
5. Be concise: tables and bullets, no filler, no recommendations about what to run next beyond the checks above.
