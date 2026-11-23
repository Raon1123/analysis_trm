"""Synthetic run cards in the Stage-A format (tools/wandb_run_index/cards.py) for the runcard lane tests. No real runs."""
import json

HEADER = "# SHARED HEADER\nDefinitions only; nothing denied here.\n"
ROWS = {  # key -> (n, first, last, min, max), each "value@step"
    "probe/test_exact": ("50", "0.00390625@4882", "1@244100", "0@63466", "1@43938"),
    "train/lm_loss": ("244100", "2.54363@1", "0.002931@244100", "0.000471849@105258", "7.58577@66745"),
    "train/lr": ("244100", "5e-08@1", "0.0001@244100", "5e-08@1", "0.0001@2000"),
}


def make_card(id="abc12345", name="demo_k4_s1", project="Demo", cohort="clean", status="ended", has_metadata=True,
              tier="api-confirmed", rows=ROWS, duplicates=(), exps=("EXP-010",), region="", truncated=False, snapshot=False, multi=False, history_empty=False, z=False):
    url = f"https://wandb.ai/someone/{project}/runs/{id}"
    dup = "duplicates: []" if not duplicates else "duplicates:\n" + "".join(f'  - {{id: "{d}", status: "ended"}}\n' for d in duplicates).rstrip("\n")
    trunc_fields = "summary_last_step: 292\nhistory_last_step: 264\nstats_table_last_step: 264\n" if truncated else ""
    snap_fields = 'snapshot_utc: "2026-11-10T11:11:34+00:00"\n' if snapshot else ""
    he = "history_empty: true\n" if history_empty else ""
    if z:
        rows = {**rows, "z/eff_rank": ("50", "36.834@4882", "3.1089@244100", "1.07478@126932", "36.834@4882")}
    exps_l = list(exps) + (["EXP-015", "EXP-023"] if multi else [])
    ea = "  exp_attribution:\n" + "".join(f'    - {{id: "{e}", source: "map", confidence: "registry"}}\n' for e in exps_l) if exps_l else "  exp_attribution: []\n"
    stats = ""
    if rows:
        stats = ("| key | n | first (@step) | last (@step) | min (@step) | max (@step) | nonfinite |\n|---|---:|---|---|---|---|---:|\n"
                 + "".join("| `%s` | %s | %s | 0 |\n" % (k, v[0], " | ".join(f"{c.split('@')[0]} (@{c.split('@')[1]})" for c in v[1:])) for k, v in rows.items()))
    else:
        stats = "No allow-listed metric key was logged in this run's history.\n"
    return f'''---
id: "{id}"
name: "{name}"
project: "{project}"
started: "2026-07-22T10:29:14.791746Z"
status: "{status}"
has_metadata: {str(has_metadata).lower()}
history_truncated: {str(truncated).lower()}
{trunc_fields}snapshot: {str(snapshot).lower()}
{snap_fields}multi_attribution: {str(multi).lower()}
{he}last_step: 244100
git_commit: "c02df84c11329d92860add6b8f37a8fd68532244"
wandb_sha256: "75186aa5aea414ab30afee6272c39e74bc99c3bbe582f697ae3c378d01ea57b2"
cohort: "{cohort}"
link:
  tier: "{tier}"
  url_status: "confirmed"
  url: "{url}"
config:
  arch_preset: "trm"
  k: 4
  seed: 1
  data_paths: ["data/sigma_k_10/4"]
attribution:
  status: "DRAFT (run-family map; not evidence)"
  exps: {json.dumps(exps_l)}
  hyps: []
{ea}  hyp_attribution: []
siblings: {{n: 0, runs: []}}
{dup}
---

# Run card — `{name}` ({project} / {id})

## Symbols

| symbol | definition |
|---|---|
| `probe/test_exact` | exact-match accuracy on a fixed 512-example probe subset |

## Precomputed stats (full-resolution, script-computed)

{stats}
## Related documents

- [EXP-010: EXP-010_demo.md](../../11_lab-experiments/EXP-010_demo.md)

## Prose (filled by a later batch stage from the numbers above only)

<!-- prose:auto start -->{region}
<!-- prose:auto end -->
'''


def good_output(card_id="abc12345", cohort="clean", status="ended", extra_caveats=(), attribution=None, texts=None, multi=False,
                truncated=False, snapshot=False, exps=("EXP-010",)):
    """A compliant output (r4 contract). Caveat texts: the script sentences for status / truncated / snapshot / multi
    (substituted from the fixture card's own numbers), a plain 'Card flag: <kind>.' for the others; override with texts=."""
    if attribution is None:
        attribution = ("DRAFT attribution map, not evidence; the map lists several experiments and is unresolved here; the script "
                       "prints the listed ids below." if multi else
                       "DRAFT attribution map, not evidence; the script prints the listed ids below." if exps else
                       "The card lists no experiment or hypothesis attribution.")
    ids = ", ".join(list(exps) + (["EXP-015", "EXP-023"] if multi else []))
    sent = {"status": f"The card's status value is {status}.",
            "truncated_history": "Local history ends at step 264, before the summary step 292; the last values quoted here are not the run's final step.",
            "snapshot": "Partial snapshot taken at 2026-11-10T11:11:34+00:00; the run may still have been in progress, so these values are not final.",
            "multi_attribution": f"The attribution map assigns this run to several experiments ({ids}); this is unresolved here."}
    sent.update(texts or {})
    summ = "probe/test_exact was 0.00390625 at step 4882 and 1 at step 244100."
    if truncated:
        summ = "From the local history, " + summ
    return {
        "run_id": card_id,
        "what_this_run_is": f"Run demo_k4_s1 in project Demo, arch preset trm, k 4, seed 1, started 2026-07-22, status {status}.",
        "attribution": attribution,
        "numbers_summary": summ,
        "what_the_numbers_show": [
            {"key": "probe/test_exact", "stat": "first", "value": "0.00390625", "step": "4882"},
            {"key": "probe/test_exact", "stat": "last", "value": "1", "step": "244100"},
            {"key": "train/lr", "stat": "min", "value": "5e-08", "step": "1"}],
        "caveats": [{"kind": k, "text": sent.get(k, f"Card flag: {k}.")} for k in extra_caveats],
        "see_also": ["EXP-010_demo.md"]}
