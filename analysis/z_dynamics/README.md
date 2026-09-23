# analysis/z_dynamics — EXP-017 per-call readout analysis (pre-registered, unrun)

This package implements the analysis that EXP-017 pre-registers
(`/mnt/ayp/agents/trm/21_experiments/11_lab-experiments/EXP-017_z-dynamics-latent-formation.md`, H-201).
It covers C-Z-1 (staged formation), C-Z-2 (partial powers σ^m) and C-Z-3 (E_L of ln PR(z_L), G vs C).
The analysis is CPU only, uses no GPU and does no training.

`analysis/` is gitignored, so this package is tracked with `git add -f analysis/z_dynamics`.

## Order of operations (EXP-017 §4.6, binding)

1. The prereg and H-201 are frozen and witnessed, and the independent-verifier PASS is in hand (r1 PASS).
2. `tests/test_z_dynamics.py` passes.
3. **Before any run**, the sha256 of `judge.py`, `stats.py` and `readouts.py` goes into the daily
   rollup and the `EXP-017_RESULTS.md` header, together with the statement that zero values exist.
   The orchestrator does this step, not the implementer.
4. Run `run_zdyn.py`, then `judge.py`, then `make_figures.py`, in that order and from the repo root.
5. Results go into `EXP-017_RESULTS.md` only. **They never go into the EXP-017 doc, which is frozen.**
   `reports/` holds data and figures only, never prose.

## Commands (repo root, `uv run`)

```bash
TODAY=$(date +%F)
uv run python analysis/z_dynamics/run_zdyn.py --cells analysis/z_dynamics/EXP017_cells.csv \
    --ckpt-root checkpoints/power_permutation --device cpu --batch 250 \
    --test-n 1000 --probe-n 512 --lens-train-n 2048 --n-perm 2000 --seed 20260923 \
    --out reports/figures/${TODAY}_z-dynamics/data
uv run python analysis/z_dynamics/judge.py --data reports/figures/${TODAY}_z-dynamics/data \
    --prereg /mnt/ayp/agents/trm/21_experiments/11_lab-experiments/EXP-017_z-dynamics-latent-formation.md \
    --out reports/figures/${TODAY}_z-dynamics/data/verdict.json
uv run python analysis/z_dynamics/make_figures.py --data reports/figures/${TODAY}_z-dynamics/data \
    --out reports/figures/${TODAY}_z-dynamics
```

Then copy the figures to `/mnt/ayp/agents/trm/25_figures/11_lab-curated/${TODAY}_z-dynamics/`
(memory `figures-copy-to-lab`).

- `run_zdyn.py` processes the G5 unit `pp_base_tf_z_iter_k5_s1` first and prints an extrapolated
  wall time (VERIFY-r1 N10). If that estimate is more than 10× the prereg's 2–3 CPU-h, report it
  before continuing. GPU is **not** pre-authorised (I-11).
- `run_zdyn.py` refuses an `--out` that is not empty (I-8 requires a run from scratch). It also
  refuses arguments that deviate from the frozen protocol (`--n-perm`, `--seed`, `--test-n`, `--probe-n`).
- `run_manifest.json` says `"status": "complete"` only after every cell and the G5 re-run have
  finished. `judge.py` refuses any other status. `--only` debug runs are marked `partial` and are
  never judged.

Smoke test (loading and hook shapes only; it computes no accuracy, PR or statistic):

```bash
uv run python analysis/z_dynamics/run_zdyn.py --smoke --only pp_base_tf_z_iter_k5_s1 --smoke-n 8 \
    --out /home/ayp/.claude/jobs/<job>/tmp/zdyn_smoke     # refuses reports/ and lab/
```

Tests (the worktree has no .venv, so run from the main checkout):

```bash
cd /home/ayp/project/trm && PYTHONPATH=<checkout-with-this-package>:/home/ayp/project/trm \
    rtk uv run pytest <checkout>/tests/test_z_dynamics.py -q
```

## Files

| File | Role |
|---|---|
| `judge.py` | **Hash-witnessed.** The single `FROZEN_RULES` dict holds the verbatim §2.5 YAML block (re-parsed from `--prereg` at run time, which must be deep- and byte-equal and match the frozen sha256 `567e6a40…`), plus the `gates` sub-block of §6 prose tolerances (G3 ≤ 0.02, G4 ≤ 0.01, I-11 > 3 cells, terminal steps, G5 unit, G7 keys), each anchored to a verbatim prereg substring (VERIFY-r1 N2). Applies §2 and writes `verdict.json`. |
| `stats.py` | **Hash-witnessed.** C-Z-1 cell_M / subclass, pooled γ + shuffled-call test, mixture qualifier, P_int and chance ω, E_L / Θ_L / Λ_L, exact stratified enumeration, D-1 Vis. |
| `readouts.py` | **Hash-witnessed.** Decode, validity, triviality, powers and order, nearest power (unique argmin, HAM_MAX), Ham_mix, A_i, D-9 residues, soft mass. |
| `instrumented_forward.py` | `forward_trace`, the §4.3 re-execution (18 preliminary readouts, native argument structure, G1/G2), and `load_inner`. |
| `gates.py` | G0 (byte `cmp` of the code copy), G6/G7 records, snapshot reference for G3/G4, and the sealed-path guard (I-2). |
| `pr_traj.py` | PR per call (the `pr_recompute.pr_of_latent` composition over `utils.z_logging`). |
| `tuned_lens.py` | D-7 (§4.4). Interpretive only; never read by `judge.py`. |
| `cells.py`, `EXP017_cells.csv` | The frozen 41-cell manifest (§4.1). The CSV is an export and is asserted equal to `cells.MANIFEST`. |
| `run_zdyn.py` | CLI orchestrator. |
| `make_figures.py` | §5 figures. Only **Z1** is implemented so far. |

## Judge semantics

- Thresholds are read only through `judge.rule()` / `judge.num()`. A key missing from `FROZEN_RULES`
  raises `FrozenRuleMissing`, and the judge refuses (I-9). Numbers inside rule strings such as
  `ceil(2n'/3)`, `floor(n_frozen/3)`, `>= 0.9` and `'6->7'` are parsed from those strings.
- The judge recomputes every cell class (M, Z, thr) from the numeric `cell_stats.csv` columns and
  aborts if the stats labels disagree.
- Decisions are made in this order:
  1. I-7 / I-8 checks (abort, exit 2).
  2. A G5 failure or I-11 → HALT (exit 4). Under N2 a determinism failure is a pipeline stop, not a cell exclusion.
  3. INVALID-EMPTY, then INVALID-GATE, then S1-0 / S2-0 (N2 order), then the branch.
  4. C-Z-3: S3-0 (any stratum cell excluded or U), then S3-1, S3-2, S3-3.
  5. Anything the frozen branches do not cover is reported as `UNCOVERED` (I-10).
  6. S1-6 is additionally flagged `uncovered_mechanism`.
- Exit codes: 0 means clean. 3 means a verdict was written but at least one cell was excluded
  (gate, I-3, I-4 or U). 2 means refused or aborted, with no verdict written. 4 means HALT.

## Implementation choices where the prereg left room (report these in RESULTS)

- **Loader.** `load_inner` mirrors `pr_recompute.load_wrapped` step by step but returns the
  missing and unexpected keys, which `load_wrapped` only prints. I-3 needs those keys.
- **Chance ω.** ω depends on σ only through its cycle type (conjugation invariance), so it is
  estimated once per (cycle type, k) on a canonical representative, with seed `[PERM_SEED, k, *cycle_type]`.
- **Permutation test.** Each (cell, split, readout kind) gets a fresh `default_rng(PERM_SEED)`, and
  rows are shuffled by `argsort(rng.random)`, as in the null check.
- **Column `Z`.** This is the C-Z-2 cell class, `PF` / `stepwise` / `jump` / `unordered` (N5).
  `Z_raw` is the class before PF tagging.
- **G7 within a family or stratum.** A violation of arch.name / epochs / data-root parity makes the
  whole family INVALID-GATE, and the stratum S3-0.
- **INVALID-EMPTY vs INVALID-GATE.** When n = 0, INVALID-EMPTY wins as the more specific token.
- **D-8.** The "float32 logits" are `.float()` of the natively computed bf16 logits.
- **Exact boundaries.** a_j, v_j, R, maxdrop and maxrise are formed from integer counts and divided
  once. A boundary value such as R = 0.2 therefore equals the literal 0.20.
- **j\*.** The first j whose modal m* (over defined m*) is uniquely k.
- **Test-split PR / E_L.** These are also computed, on the first 512 test rows, as descriptive D-5
  values. The judge uses train only.
