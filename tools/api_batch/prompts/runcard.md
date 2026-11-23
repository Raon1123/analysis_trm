You write the short prose section of ONE wandb run card for a research lab notebook. The shared header (definitions of
cohorts, status values, link tiers, metric keys, and the output contract) comes first; the run card itself comes last.
The card's frontmatter and its "Precomputed stats" table are the only material. The numbers in that table were computed
by a script. You describe; you never compute, estimate, round, compare or interpret.

Rules (a deterministic checker enforces every one of them; an output that breaks one is discarded)
0. Write in English, plain ASCII only (printable characters 0x20-0x7E; no accents, no symbols such as arrows or the minus
   sign U+2212, no emoji, no angle brackets, dashes written as the ASCII hyphen-minus "-", no HTML, no markdown headings). Describe only: "the table records X at
   step Y". Allowed neutral verbs: records, logs, reads, is, was, has, reaches, drops to, rises to, stays at, at step.
   No interpretation, no evaluation, no explanation of why, no claim that anything is evidence for or against anything.
1. Numbers. A number may appear in your output only if it is printed in the card, and then as digits copied exactly as
   printed (5e-08, 0.000881774; never round, never reformat). No spelled-out numbers (zero, one, two ... twenty, thirty ...
   ninety, hundred, thousand, million, half, double, twice, triple, dozen), no unit suffix glued to digits (no 244k, 1e5,
   3x, 10M, 5%) unless that exact string is printed in the card, no ordinals such as 3rd. No difference, ratio, percentage,
   rate, average, "n-fold" or any other derived quantity.
2. `run_id`: the card's `id` value, verbatim.
3. `what_this_run_is`: 1-2 sentences from the frontmatter only: run name, project, model preset (`config.arch_preset`), the
   config fields that identify it (`k`, `seed`, H/L cycles, data path, flags such as `ema`, `log_z_dynamics`), start date,
   and the card's `status` value, written exactly as "status <value>" with the value as printed. Say nothing else about how or whether the run ended. No
   purpose, no goal, no result.
4. `attribution`: copy ONE of these three sentences word for word (the script prints the ids and confidences itself; you
   never list, restate, rank or choose among them, and you never write an EXP or H identifier here):
   - if `attribution.exps` and `attribution.hyps` are both empty: "The card lists no experiment or hypothesis attribution."
   - if `multi_attribution` is not true and a list is non-empty: "DRAFT attribution map, not evidence; the script prints the listed ids below."
   - if `multi_attribution` is true: "DRAFT attribution map, not evidence; the map lists several experiments and is unresolved here; the script prints the listed ids below."
5. `what_the_numbers_show`: 3-8 entries, one per (metric key, statistic) you choose, most informative first. Each entry is
   `{key, stat, value, step}` where `key` is a key of the stats table, `stat` is first|last|min|max, and `value` and `step`
   are copied character for character from that cell, which is printed as `value (@step)`: `value` is the part before the parenthesis and `step` is only the digits after the @ (write 292, never @292, never parentheses). Prefer the evaluation/probe
   keys and the loss keys; include `num_params` or `phase/index` only when present and informative.
6. `numbers_summary`: at most 2 sentences walking through the entries above in plain words, using only the neutral verbs of
   rule 0, for example "<key> reads <value> at step <step> and <value> at step <step>" with printed values. Every number
   must be one of the printed values or steps. No adjective of judgement (good, poor, strong, weak, robust, perfect, best,
   worst, converged, collapsed, learned), no comparison words (better, worse, higher than), no causal or mechanism language.
7. `caveats`: one entry per applicable item; `kind` must be one of the kinds below. Where a sentence is given in quotes,
   copy it word for word with the card's values substituted; add nothing to it that restates, softens or contradicts it.
   - `status` if the card `status` is not `ended`: "The card's status value is <status>." (the exact status string printed in the card). Do not explain what the status means.
   - `no_metadata` if `has_metadata` is false: one short factual sentence saying the card has no metadata.
   - `truncated_history` if frontmatter `history_truncated` is true: "Local history ends at step <history_last_step>, before the summary step <summary_last_step>; the last values quoted here are not the run's final step." Also write "local history" in `numbers_summary` when you quote a `last` value.
   - `snapshot` if `snapshot` is true: "Partial snapshot taken at <snapshot_utc>; the run may still have been in progress, so these values are not final."
   - `multi_attribution` if `multi_attribution` is true: "The attribution map assigns this run to several experiments (<EXP ids from exps, in card order, joined by ", ">); this is unresolved here."
   - `legacy_cohort` if `cohort` is `legacy`: one short sentence that numbers of this cohort are not comparable with the clean cohort.
   - `duplicate_run_name` if `duplicates` is non-empty: one short sentence naming the ids exactly as listed.
   - `link_tier` if `link.tier` is not `api-confirmed`: one short sentence naming the tier and saying the URL is "not api-confirmed".
   - `other` for a warning line in the card that none of these covers (quote it briefly).
   An empty list is correct when none applies. Do not add caveats the card does not support.
8. `see_also`: ids, file names or paths printed in the card (sibling ids, duplicate ids, "Related documents" file names), each
   one token without spaces, copied exactly. This is the ONLY field in which an EXP or H identifier may appear. An empty
   list is fine. Never write a URL, and never write a path or id that is not in the card.
9. Hypotheses and meaning. Never mention hypotheses, H-ids, claims, theories, predictions or expectations, and never say
   what a result means. Banned word classes (any form): hypothesis, claim, theory, conjecture, predict, expect; proven,
   falsified, refuted, supported, confirmed, verified, validated, favors, holds, bolsters, robust, strong or weak evidence,
   contradicts, corroborates, in line with, agrees with, demonstrates, shows that, indicates, suggests, evidence for,
   consistent with, generalizes, perfect, good, poor, better, worse, best, worst, likely; because, due to, caused, explains,
   mechanism, leads to. The link-tier names `api-confirmed` / `constructed-unverified` and the metric-definition labels
   printed in the card may be quoted exactly as printed; do not use the word "verified" yourself otherwise.
10. Run status. Describe how a run stands ONLY by quoting the card's `status` value (rule 3) and the sentences of rule 7. Do not
    use any word of these classes anywhere else: crash, fail, diverge, abort, kill, stall, hang, running, in progress,
    ongoing, complete, finish, done, over, ended, wrapped, terminated, succeed, success, converge, final, stopped, concluded.
    Use "last" and "first" for the statistics; never "final". On a card with `history_truncated` or `snapshot` the same holds with
    no exception outside the quoted sentences.
11. Never mention any run id or run name other than this card's own and those printed in it; never write a URL.
12. Write "H and L cycles", "min and max", "first and last" in words. The only slash-joined tokens you may write are metric keys
    and paths printed in the card.
13. Output only the JSON object required by the schema. Keep every string short (one or two sentences).
