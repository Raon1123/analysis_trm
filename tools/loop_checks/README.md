# tools/loop_checks — read-only loop checks

All five tools are pure stdlib, **read-only** (write nothing but stdout/stderr), support
`--json` and `--help`, follow NAS symlinks, and use the exit code as the verdict.
Run with `uv run python tools/loop_checks/<tool>.py` (stdlib only; `uv run` is the project convention).

| Tool | Audit decision | Purpose | Exit codes |
|---|---|---|---|
| `queue_status.py` | AD-2 | `queue_run.sh status` branch as JSON/text (honours `QUEUE_DIR`) | 0 ok; 2 queue dir missing |
| `freshness.py` | AD-5 | STATE.md mtime and newest `monitor` event age (monitor only checked while `processing/` is non-empty; else `n/a`); newest event of any stage is info only | 0 fresh; 1 stale; 2 file missing/unreadable |
| `ledger_regen.py` | AD-7 | Parse a RESULTS "Claim ledger" regeneration-command column. Dry-run by default; `--execute` runs ONLY rows classed `pure` (never `gpu`/`writes`) and compares an expected sha256 prefix | 0 ok / nothing to compare; 1 mismatch or nonzero rc; 2 no ledger table |
| `memory_last_cited.py` | AD-11 | Dead backticked repo paths in memory notes + `last_cited` (newest lab/**/*.md mtime containing `[[name]]`) | 0 no dead paths; 1 dead paths; 2 memory dir missing |
| `enqueue_guard.py` | AD-4 | Warning-only pre-enqueue guard for an EXP card: `literature_link` (or lab/28_literature in a 문헌/literature section), `EXP-NNN_FREEZE.sha256`, `_FREEZE_WITNESS.md`, `sha256sum -c --quiet` | 0 always (WARN: on stderr); `--strict` 1 if any item missing; 2 card not found |

Freshness thresholds (STATE.md 48 h, monitor events 6 h) come from
`lab/11_governance/11_lab-root/AUTORESEARCH.md` §1 item 7; the flags
`--state-max-h` / `--monitor-max-h` only override them for testing.

Notes: `ledger_regen.py` classes are `gpu` (CUDA/nvidia/torch/measure_rho/pretrain.py in
command or row), `writes` (redirect `>` in command; 덮어씀/overwrite/tee/--save/--out in
command or row), `pure`, `nocmd`. Never `--execute` against EXP-016/EXP-020 results.
Tests: `CUDA_VISIBLE_DEVICES="" rtk uv run pytest tests/test_loop_checks.py -q`.
