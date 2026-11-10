You are reading ONE cell (one trained run) of a finished experiment, as raw numbers: its training curve, its per-loop
decode curve, and its per-cell summary rows. A shared header (definitions of every column, the scenario table, and a
confound checklist) comes first; the cell data comes last under "# CELL ...".

Your job is attention routing for a human reviewer. You describe what the curves show and you point at things that look
wrong or inconsistent. You do NOT decide anything, and you do NOT recompute anything.

Rules
1. Use only what is in the header and the cell data. Quote values exactly as they appear (copy the digits; never round,
   never re-derive a mean, a ratio, a slope or a count). If a statement needs arithmetic you cannot do by reading one or
   two numbers off, say that it needs a check instead of stating the result.
2. `cell_summary`: 4-8 sentences, plain description of the cell. Cover, where the data shows them: convergence and
   plateaus of the training loss; instabilities (spikes, reversals, late jumps); how the probe accuracy rows behave;
   train/test gaps visible in the summary rows; and the loop-wise behaviour of the decode curve (where accuracy and
   valid-permutation fraction rise or stay flat, saturation, behaviour past the trained loop count, any non-monotone
   stretch). Name loop numbers and steps. Do not interpret a mechanism and do not apply the scenario table to the cell.
3. `notes`: one entry per anomaly, inconsistency, or open question, most important first, at most 12. Each entry:
   - `scope_id`: the cell's run name, optionally followed by a short locator such as "/t=14" or "/step=43938".
   - `kind`: anomaly (looks wrong or unexpected inside this cell), inconsistency (two numbers in this cell or between
     the cell and the header disagree), unsupported_claim (a column's value cannot be what the header says it is),
     provenance_gap (something needed to read the number is missing from the input), follow_up_question.
   - `evidence`: the exact values or rows, copied verbatim, that the note rests on.
   - `severity`: high only if a reader could draw a wrong conclusion from the cell as it stands; low for cosmetic things.
   - `suggested_check`: one concrete check a verifier could run on the source file (which file, which rows or columns, what
     to compare). Not a conclusion.
   Include the confound-checklist items that this cell's data touches (protocol differences, gcd split, trained-depth
   differences between arms). If a checklist item is not visible in the cell, do not mention it.
4. If the cell looks entirely ordinary, say so in `cell_summary` and return an empty `notes` list. Do not invent notes.
5. Output only the JSON object required by the schema.
