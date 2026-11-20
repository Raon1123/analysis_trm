#!/usr/bin/env bash
# Declarative grid-search enqueuer for scripts/queue_run.sh.
#
# A sweep is a list of "alias=hydra.param: candidate1 candidate2 ..." lines.
# The cartesian product (grid search) of all candidate lists is enqueued for
# every k in K_LIST. Run names are built automatically from the swept values:
#
#   - 1 candidate   -> fixed value, NOT in the run name
#   - 2+ candidates -> swept, run name gets _<alias><value>
#
#   e.g. TRM_SWEEP below produces run names like  k6_trm_halt16_H3_L6
#
# Usage:
#   scripts/sigma_enqueue.sh [run_prefix]              # write job files
#   scripts/sigma_enqueue.sh --dry-run [run_prefix]    # print grid, write nothing
#   scripts/sigma_enqueue.sh [run_prefix] --dry-run    # same; order does not matter
#
# WARNING: with no --dry-run this writes real job bodies into scripts/queue/jobs/,
# where the runner can claim them immediately.  --dry-run is accepted in any
# position and an unrecognised flag aborts rather than being read as a prefix.
#
# Re-running appends after existing jobs (sequence numbers continue), so you
# can enqueue more sweeps while the runner is going.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing.
#
# --dry-run is honoured in ANY position, and an unrecognised flag is a HARD
# ERROR rather than a run prefix.  Both halves matter, and the reason is an
# incident, not a style preference:
#
#   2026-08-06, `sigma_enqueue.sh pp --dry-run` was run intending a dry run.
#   The old parser tested only "$1" for --dry-run, so "pp" became RUN_PREFIX,
#   "--dry-run" was silently ignored as a stray argument, DRY_RUN stayed 0, and
#   216 real job bodies were written into the live scripts/queue/jobs/ where
#   they sat claimable by the runner for ~39 s before being pulled back out.
#
# A silently-swallowed flag on a script whose default action is "enqueue real
# training jobs" is a one-keystroke path to an unintended production launch, so
# anything starting with "-" that is not understood now stops the script.
# ---------------------------------------------------------------------------
DRY_RUN=0
_positional=()
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            # Print the leading comment block by STRUCTURE, not by a hardcoded
            # line range: a range goes stale the moment the header is edited
            # (and silently truncates the queue-writing warning mid-sentence,
            # which is the one line a --help reader most needs to see).
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
                "$0" >&2
            exit 0
            ;;
        --) shift; _positional+=("$@"); break ;;
        -*)
            echo "ERROR: unknown option '$1'." >&2
            echo "       Refusing to continue: this script ENQUEUES REAL TRAINING" >&2
            echo "       JOBS by default, and treating an unrecognised flag as a run" >&2
            echo "       prefix is how 216 jobs were once enqueued by a command that" >&2
            echo "       was meant to be a dry run.  Did you mean --dry-run?" >&2
            exit 2
            ;;
        *) _positional+=("$1") ;;
    esac
    shift
done
if (( ${#_positional[@]} > 1 )); then
    echo "ERROR: expected at most one run prefix, got: ${_positional[*]}" >&2
    exit 2
fi
set -- "${_positional[@]+"${_positional[@]}"}"
DRY_RUN_SHOW_BODY="${DRY_RUN_SHOW_BODY:-0}"

QUEUE_DIR="${QUEUE_DIR:-scripts/queue}"
JOBS_DIR="$QUEUE_DIR/jobs"

RUN_PREFIX="${1:-}"
prefix=""
[[ -n "$RUN_PREFIX" ]] && prefix="${RUN_PREFIX}_"

# ====================  EDIT BELOW: fig1 cohort grid  ====================
# Target = the fig1 accuracy grid (analysis/make_mlp_grid_figure.py + tf_vs_mlp_grid),
# re-run on the ORDER-FILTER-FIXED data (data/sigma_k_10 = ord(σ)>k, EXP-007 fix).
# Legacy fig1 lived in project Sigma_k_fig12 on data/sigma_k_10; this is the clean
# re-run under Sigma_k_new. Run names match legacy so the figure code is unchanged.
#
# 8 cohorts = block {mlp,tf} × z {z,noz} × iter {iter,noiter}:
#   block:  mlp -> arch.mlp_t=True    | tf  -> arch.mlp_t=False
#   z:      z   -> arch=trm (z-carry) | noz -> arch=trm_singlez (no z)
#   iter:   iter-> H_cycles=3 L_cycles=6 | noiter -> H_cycles=1 L_cycles=1
#   fixed across all: arch.L_layers=2, arch.halt_max_steps=1  (matches legacy fig1)
# Single seed (=1) for a fast grid; add 2 3 to SEEDS later for min/max bands.

# Campaign identity. Every value here is env-overridable so a re-target does not
# need a source edit:  WANDB_PROJECT=... K_LIST="3 4" ./sigma_enqueue.sh
#
# 2026-08-06: default project moved Sigma_k_new -> power_permutation. The old
# name now denotes a SEALED generation (see
# scripts/queue/sealed-20260806-pre-power-permutation/README.md); nothing new
# should be written into it. wandb's `project` is flat, so campaign arms are
# separated by RUN_GROUP (wandb group), not by more project names.
WANDB_PROJECT="${WANDB_PROJECT:-power_permutation}"
RUN_GROUP="${RUN_GROUP:-}"           # per-stage default assigned at emit time

# 2026-08-06: extended to EVERY k directory that exists under D0 (user
# decision "1부터 전부로 D0 확장"). All 14 are content-verified clean --
# min ord(sigma) = the smallest order actually achievable above k, and
# frac(ord<=k) = 0.000 in both train and test for every one of them.
#
# The achievable orders in S_10 are {1,2,...,10,12,14,15,20,21,30}, which
# makes the pools coarser as k grows and produces two facts worth knowing
# before reading any k-sweep as a smooth curve:
#   k=1   -> ord>1  : sigma^1 = sigma, an identity-copy task; near-trivial.
#   k=9   -> ord>9  : {10,12,14,15,20,21,30}
#   k=10  -> ord>10 : {12,14,15,20,21,30}
#   k=11  -> ord>11 : {12,14,15,20,21,30}  <-- IDENTICAL to k=10, because
#           S_10 has NO permutation of order 11 (11 is prime > 10). k=11 is a
#           same-distribution exponent comparison against k=10, not new data.
#   k=12  -> ord>12 : {14,15,20,21,30}
#   k=16  -> ord>16 : {20,21,30}
#   k=20  -> ord>20 : {21,30}      <-- narrowest pool in the sweep
# So k=10/k=11 are a matched pair, and k>=12 progressively narrows the
# distribution rather than merely raising difficulty.
K_LIST=(${K_LIST:-1 2 3 4 5 6 7 8 9 10 11 12 16 20})
SEEDS=(${SEEDS:-1})

# ---- Staged (one-factor-at-a-time) exploration, 2026-08-06 ----------------
# Replaces "enqueue the whole 378-cell cartesian product at once".
#
# Why. The full product spends most of its budget on cells whose answer is
# already implied by cheaper ones: k=1 is an identity-copy task, k=10 and k=11
# draw from an IDENTICAL pool (S_10 has no order-11 element), and the k>=12
# pools narrow monotonically. Meanwhile the phenomena actually under
# investigation -- the H-collapse band and the L3 non-monotonicity -- sit in a
# narrow mid-k window. A grid answers every question at the same low
# resolution; running one factor at a time answers the load-bearing ones first
# and lets each stage choose the next stage's k values.
#
# K_DIAG = the diagnostic k set the ABLATION stages sweep, deliberately much
# smaller than K_LIST. The default brackets the observed transition: 3 is
# reliably learnable, 5/6/7 is where collapse and non-monotonicity appear, 10
# is the far end. Override once stage 1 has told you where the real boundary
# is on THIS data generation -- that is the entire point of staging.
# WARNING (SP-04-L03 / G-20260813-inverse-learnability, 2026-08-13): do NOT
# extend K_DIAG to include 12 or 16 as a shortcut for covering the SP-04
# judgment cells. K_DIAG feeds the LIVE ablation_act / ablation_arch / tfb
# stages, which run under D1 semantics (epochs=50000, see the "2026-08-06:
# epochs 100000 -> 50000" comment above) -- but the SP-04 judgment protocol
# for k=12/16 is D0 + epochs=100000, a DIFFERENT budget on the SAME
# architecture cohort (tf_noz_iter). Folding 12/16 into K_DIAG would silently
# launch a wrong-protocol live run at those k values before the dedicated
# `invcells` stage (below) ever fires, producing two non-comparable results
# at the same k with no run_name collision to catch the mismatch
# (abl_*_k12_s1 vs pp_inv_tf_noz_iter_k12_s1 are different names, so
# enqueue()'s idempotency skip will NOT protect against this).
K_DIAG=(${K_DIAG:-3 5 6 7 10})

# The single reference cell every ablation perturbs by exactly one factor.
# Matches the legacy fig1 baseline (tf = attention block, z-carry, iterating).
BASELINE_COHORT="${BASELINE_COHORT:-tf_z_iter}"
DATA_ROOT="${DATA_ROOT:-data/sigma_k_10}"   # canonical n=10, ord(σ)>k-clean (EXP-007 fixed)

# Protocol-matched to legacy fig1 all_config.yaml (Sigma_k_fig12), verified 2026-07-21:
#   eval_interval=2000 (NOT 5000), log_z_dynamics=True (gates probe/test_exact —
#   the figure's primary metric, z_logging.py:378), z_snapshot=False,
#   checkpoint_every_eval=False (cfg default True would checkpoint every eval).
# (+ prefix: log_z_dynamics / z_snapshot are pretrain.py pydantic fields NOT in
#  cfg_pretrain.yaml — Hydra struct mode rejects bare overrides for absent keys.)
# 2026-08-06: z_snapshot flipped False -> True for the power_permutation
# generation. Rationale (design doc §4.2b): the open question is whether an
# H-collapse cell (train_exact~1.0, test_exact~0.0) is ALSO a representational
# collapse (low participation ratio) or a purely accuracy-side failure.
# `z/eff_rank` = PR(z_H) already answers that and is already logged -- the only
# thing missing was that the snapshot path was switched off. This is additive
# telemetry: z_snapshot only controls whether snapshots are written to
# checkpoint_path (pretrain.py `_snap_path`), it does not alter training or any
# existing metric. Legacy fig1 protocol-match is therefore preserved for every
# metric the old figures consume.
# 2026-08-06: per-eval prediction dump ENABLED (eval_save_outputs).
# Requested by intake 20260806T052122Z-c65f6d6d0f, which asks for a raw
# per-eval prediction artifact so that E_comm, valid-permutation rate and the
# d(k) trivial-position split can be computed ENTIRELY DOWNSTREAM -- keeping
# server-side metric code at zero lines. No new code was needed: the machinery
# already exists and was merely inert, gated on this one empty list.
#   models/losses.py:63    preds = argmax(logits)
#   models/losses.py:100   filtered to return_keys
#   pretrain.py:531-535    collects any key named in eval_save_outputs
#   pretrain.py:558-565    torch.save -> checkpoint_path/step_<N>_all_preds.<rank>
# It fires inside evaluate(), so it is independent of checkpoint_every_eval:
# dumps land at EVERY eval interval even though full weights are only written
# at the final iteration.
# Keys chosen: preds/labels/inputs/puzzle_identifiers -- enough to make each
# dump self-describing without a join back to the dataset. Deliberately NOT
# `logits` (vastly larger, and every requested statistic needs only the decoded
# sequence). Cost ~136 KB/eval, ~7 MB/run, ~2.6 GB across the 378-job grid.
# 2026-08-06: epochs 100000 -> 50000 (grid-search budget).
# `epochs` is NOT the optimizer step count. pretrain.py derives
#   total_steps = epochs * total_groups * mean_puzzle_examples / global_batch_size
# and with total_groups=5000, mean_puzzle_examples=1.0 (dataset.json) and
# global_batch_size=2048 (config/cfg_pretrain.yaml):
#   50000 -> 122,070 steps    (100000 -> 244,140;  10000 -> only 24,414)
# NOTE eval_interval is in EPOCHS, not steps (pretrain.py:717), so
# eval_interval=2000 gives epochs/2000 evaluations: 25 here, 50 before.
#
# Why 50000 and not less. Empirically, from the 31 runs in
# reports/figures/2026-08-04_ablation-backlog/ablation_terminal.csv that reached
# peak_test_exact >= 0.99, the step at which they peaked is strongly structured
# BY k -- and truncation therefore is not uniform noise, it is a k-dependent
# bias that manufactures a fake k-curve:
#     budget      k=5    k=6    k=7    total captured
#     24,414 st   0/3    0/3    0/4    15/31  (48%)   <- kills mid-k entirely
#     48,828 st   1/3    2/3    1/4    22/31  (71%)
#     97,656 st   1/3    3/3    3/4    28/31  (90%)
#    122,070 st   3/3    3/3    3/4    30/31  (97%)   <- this budget
# k=5 (median peak 107,404 steps) only becomes whole at this budget, and k=5 is
# precisely the cell where the L3 non-monotone collapse was observed -- a
# shorter budget would kill the one cell the campaign most wants to read.
# The single remaining loss is one k=7 run peaking at 161,106; re-run that cell
# individually at a longer budget if it matters.
# Cost: ~1,142 GPU-h for the 378-job grid (~12 days on 4 GPUs), half of the
# 100000-epoch budget.
# CAVEAT: those 52 abl_* runs are cloud-only, so their data_epoch is `unknown`
# (not confirmed clean). The peak_step distribution is used here only as a
# measure of how long this task/architecture takes to converge, which is
# argued to be usable regardless of the ord-filter contamination question.
# NOTE the `+` on eval_save_outputs. It is a PretrainConfig field
# (pretrain.py `eval_save_outputs: List[str] = []`) but is NOT declared in
# config/cfg_pretrain.yaml, so Hydra's struct mode rejects a bare assignment
# with "Could not override 'eval_save_outputs' ... not in struct" and the job
# dies before the first step. Same reason log_z_dynamics/z_snapshot carry `+`.
# Caught 2026-08-06 by `--cfg job` pre-flight on a real enqueued body; the key
# was added to this line earlier the same day without the prefix, which would
# have failed every job in the campaign at launch.
common_args="${COMMON_ARGS:-epochs=50000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=True checkpoint_every_eval=False +eval_save_outputs=[inputs,labels,preds,puzzle_identifiers]}"

# tag | arch | mlp_t | H_cycles | L_cycles
#
# mlp_t toggles the token-mixing sublayer only: True swaps self-attention for a
# SwiGLU applied across the sequence axis (MLP-Mixer style); the channel-mixing
# SwiGLU, RMSNorm, recursion, z carry and halting are identical either way
# (models/recursive_reasoning/trm.py:70-104).
COHORTS=(
    "tf_z_iter|trm|False|3|6"
    "tf_z_noiter|trm|False|1|1"
    "tf_noz_iter|trm_singlez|False|3|6"
    "tf_noz_noiter|trm_singlez|False|1|1"
    "mlp_z_iter|trm|True|3|6"
    "mlp_z_noiter|trm|True|1|1"
    "mlp_noz_iter|trm_singlez|True|3|6"
    "mlp_noz_noiter|trm_singlez|True|1|1"
)
# FIFO priority tiers for the fig1 grid (PI 2026-07-22): the primary questions
# are z-presence and iteration-presence, and those are read off the attention
# (tf) half of the grid — so the ENTIRE tf half is emitted, across every k,
# before any mlp cell. Within a tier the loop stays k-major so each k gets its
# complete z × iter 2×2 before moving deeper.
COHORT_TIERS=(tf mlp)

# COHORT_FILTER (2026-08-26, pp-campaign closeout supplement P1.2/design doc
# `lab/23_reports/11_lab-reports/2026-08-26_pp-campaign-closeout-supplement-design.md`):
# optional comma-separated allowlist of full COHORTS tags (e.g. "tf_noz_iter"),
# restricting the cohorts/fig1 stage (1c, below) to exactly those cohorts.
# UNSET (default) is IDENTICAL to prior behaviour -- every cohort in the
# matching COHORT_TIERS tier is emitted, same as before this existed.
#
# Exists because COHORT_TIERS only filters by tier PREFIX (tf/mlp), not by the
# full z/iter tag, so there was no way to enqueue e.g. a single-cohort seed
# top-up (tf_noz_iter k10 s2/s3, to actually firm up the n=1 recursion-
# attribution reading in the closeout design doc's P1.2 -- the already-built
# `seedext` stage looks like it would do this but runs D0/epochs=100000, a
# different protocol from the D1/122k fig1 cell the claim depends on) without
# emitting the other 6-7 unwanted cohorts at that k too. Same allowlist
# pattern as INV_CELLS_FILTER above (cohort_allowed mirrors inv_cell_allowed).
#   COHORT_FILTER="tf_noz_iter" STAGES=fig1 K_LIST="10" SEEDS="2 3" \
#       DATA_ROOT=data/power_permutation/d1 scripts/sigma_enqueue.sh pp
COHORT_FILTER="${COHORT_FILTER:-}"

# cohort_allowed <tag> -- true iff COHORT_FILTER is unset, or <tag> is one of
# its comma-separated entries. See COHORT_FILTER above.
cohort_allowed() {
    local tag="$1" c
    [[ -z "$COHORT_FILTER" ]] && return 0
    local -a _tags
    IFS=',' read -r -a _tags <<< "$COHORT_FILTER"
    for c in "${_tags[@]}"; do
        [[ "$(trim "$c")" == "$tag" ]] && return 0
    done
    return 1
}

# ---- Module ablations (FIFO priority AFTER the fig1 grid) -----------------
# One-factor-at-a-time around the canonical TRM baseline fig1_tf_z_iter
# (arch=trm, mlp_t=False, H3, L6, L_layers=2, halt=1); axis values taken from
# the pre-fig1 exploratory sweep (halt {8,16}, H {6}, L {3}).  Cells equal to
# the baseline itself are already covered by fig1 — not re-enqueued.
#   tag | extra arch overrides (vs baseline)
# 2026-08-06: the single TRM_ABLATIONS list was split along the two axes the
# campaign actually reasons about, so each lands in its own wandb group and can
# be enqueued independently (STAGES="ablation_act" vs "ablation_arch").
# Union of the two below == the pre-split list plus the finer ACT rungs.

# --- ACT / halting axis (adaptive computation time: how many halt steps) ------
# The old list only had halt=8 and 16 against the fig1 baseline halt=1, leaving
# a 1 -> 8 gap with nothing in it. halt2/halt4 fill it so a monotone-in-halt
# claim can actually be checked rather than interpolated across one octave.
ACT_ABLATIONS=(
    "halt2|arch.halt_max_steps=2"
    "halt4|arch.halt_max_steps=4"
    "halt8|arch.halt_max_steps=8"
    "halt16|arch.halt_max_steps=16"
)

# --- model-structure axis (recurrence depth / width) --------------------------
# H_cycles = outer recurrence, L_cycles = inner, L_layers = block depth.
# L3 is retained because it is the one cell with a KNOWN non-monotone failure
# (k=5,6 collapsed while k=3,4,7,8,10 reached 1.0) -- see design doc §4.1.
#
# THE ONE THING TO KNOW BEFORE READING THIS AXIS (trm.py:207-216): the first
# H_cycles-1 outer cycles run under `torch.no_grad()`; only the last one is in
# the gradient path.  So define, per ACT step,
#
#     blocks applied  C = H_cycles * (L_cycles + 1) * L_layers
#     blocks w/ grad  G =            (L_cycles + 1) * L_layers
#
# and H_cycles moves C but NOT G.  Measured on this repo (n=10, k=5 metadata,
# CPU instantiation, 2026-08-06):
#
#     variant     params        C     G
#     lay1         3,420,162    21     7
#     baseline     6,828,034    42    14      H3 L6 lay2
#     L12          6,828,034    78    26
#     lay4        13,643,778    84    28
#     L24          6,828,034   150    50      <- added
#     lay8        27,275,266   168    56      <- added
#     H1 / H6      6,828,034  14/84   14      G identical to baseline
#
# That yields three distinct, individually well-posed questions instead of one
# muddled "depth" question:
#
#   (a) H1 / baseline / H6 hold G = 14 fixed and vary C by 6x.  This is NOT a
#       depth ladder -- it asks whether extra no-grad forward refinement helps
#       at all.  Well-posed as-is; do not read it as "more loops = deeper".
#   (b) The G-matched ladder pairs a params-FIXED recurrence arm against a
#       params-DOUBLING layer arm at each rung:
#           G ~  7-8 : L3   (6.83M) vs lay1 ( 3.42M)
#           G =   14 : baseline, the shared origin of both arms
#           G ~ 26-28: L12  (6.83M) vs lay4 (13.64M)
#           G ~ 50-56: L24  (6.83M) vs lay8 (27.28M)   <- the two new cells
#       L24/lay8 exist because the ladder previously stopped at rung 3, where
#       the params gap is only 2x; at rung 4 it is 4x, which is where a
#       weight-tying claim either survives or does not.
#   (c) tfb (below) supplies the no-recurrence arm.
#
# COST is NOT uniform across this list: runtime tracks C, so L24 (~3.6x
# baseline) and lay8 (~4x) are the two expensive cells.  lay8 is also the
# largest activation footprint in the campaign (G = 56 blocks kept for
# backward at global_batch_size=2048); if anything OOMs it is this cell, and
# it will land in scripts/queue/failed/ rather than corrupting its neighbours.
ARCH_ABLATIONS=(
    "H6|arch.H_cycles=6"
    "H1|arch.H_cycles=1"
    "L3|arch.L_cycles=3"
    "L12|arch.L_cycles=12"
    "L24|arch.L_cycles=24"
    "lay1|arch.L_layers=1"
    "lay4|arch.L_layers=4"
    "lay8|arch.L_layers=8"
)
# transformers_baseline depth/width ablation (old TRANSFORMER_SWEEP grid).
# arch.halt_max_steps=1 pinned — tfb yaml defaults to 16, which the old sweep
# left in place (protocol mismatch vs fig1); here every run is halt=1.
#
# tfb has no inner cycles at all (transformers_baseline.py:7 "REMOVED inner
# cycles"), so for it C = G = H_layers -- a plain stack.  Two of these layer
# counts are chosen against the trm numbers tabulated above, not by round
# numbers:
#
#   lay2  (6,795,266 params) is PARAMETER-MATCHED to the trm baseline
#         (6,828,034, a 0.5% gap).  Same weights budget; the baseline applies
#         them 42 times, tfb twice.  This is the cleanest read on recurrence.
#   lay14 (47,493,122 params) is GRADIENT-DEPTH-MATCHED to the trm baseline
#         (G = 14 both).  Added 2026-08-06.  Without it the campaign can say
#         "recurrence beats a same-size stack" but cannot say anything about
#         a stack that is genuinely as deep, because tfb topped out at 6 --
#         and the answer costs 7x the parameters, which is the actual claim.
#
# Deliberately NOT included: lay26 (88,190,978 params) would match L12's
# G = 26, but an 88M-parameter model on 5,000 training examples is a
# different regime (capacity-saturated), so the comparison would confound
# depth with overfitting.  Matching trm baseline's C = 42 is worse still:
# ~42 distinct layers, ~140M params.  The compute-matched pure-transformer
# comparison is therefore OUT OF SCOPE by construction, not an oversight --
# say so when reporting rather than implying the sweep covers it.
TFB_LAYERS=(${TFB_LAYERS:-1 2 6 14})

# 2026-08-06 — TFB_CYCLES REMOVED. `arch.H_cycles` is a NO-OP for
# arch=transformers_baseline: in models/recursive_reasoning/transformers_baseline.py
# the identifier occurs exactly twice, at the docstring (L7, "REMOVED inner
# cycles (no H_cycles/L_cycles loops within reasoning)") and the pydantic field
# declaration (L52). The only range() in the file is over H_layers (L168), and
# Model_ACTV2ReasoningModule.forward (L109-116) applies each layer once with no
# cycle loop. So the old cyc1/cyc6 sweep built IDENTICAL models and burned half
# its budget on duplicates.
#
# What that accident may have measured is worth keeping. `abl_tfb_lay2_cyc1_k3_s1`
# and `abl_tfb_lay2_cyc6_k3_s1` reached final_test_exact 0.980 vs 0.164 (peak 1.0
# both). H_cycles occurs 0 times in pretrain.py / utils/ / models/layers.py /
# models/common.py -- only in the arch YAMLs, where transformers_baseline.yaml:9
# says "H_cycles: 1  # kept for compatibility" -- so parameter count, init path,
# LR schedule and total_steps are all independent of it, and the two runs were
# structurally the same model.
# CAVEAT: both runs are cloud-only (no local wandb dir, no local checkpoint), so
# their all_config.yaml could NOT be diffed directly, and "same seed" rests on
# the ablation CSV's seed column. Treat the 0.980/0.164 gap as strong evidence
# of run-to-run nondeterminism, not as confirmed.
# Either way the cycles axis had to go (it is a no-op), and spending that budget
# on a real seed axis is right under both readings: if the variance is real the
# seeds are required, and if it is not, the repeats explain what actually
# differed.
TFB_SEEDS=(${TFB_SEEDS:-1 2 3})

# ---- Looped-transformer control (stage `ltf`, added 2026-09-23) ------------
# Closes the "compute-matched looped stack is OUT OF SCOPE" gap noted at the
# TFB_LAYERS comment above, WITHOUT the 42-distinct-layer / ~140M-param model:
# transformers_baseline gained a weight-tied loop (models/recursive_reasoning/
# transformers_baseline.py, config/arch/transformers_baseline.yaml):
#   arch.loops=N               re-apply the whole H_layers stack N times per forward
#   arch.loop_input_injection  re-add the input embedding at every loop (else loop 0 only)
#   arch.loop_grad_cycles=n    0 = full backprop; n>0 = no_grad for the first N-n loops
# With H_layers=2 and loops=21, C = 21*2 = 42 blocks per forward = the trm
# baseline's C (tf_z_iter: H3 L6 lay2), at lay2's 6,795,266 params (0.5% under
# trm's 6,828,034). Arms (each a one-factor contrast):
#   fullgrad : C=42, G=42  -- compute-matched, full backprop through every loop
#   g7       : C=42, G=14  -- ALSO gradient-depth-matched (TRM-style truncation:
#              last 7 loops = 14 blocks carry grad, like trm's last H cycle)
#   noinj    : C=42, G=42, input embedding added at loop 0 only -- isolates
#              TRM's per-step input injection
# What an ltf arm does NOT have relative to trm: the z_H/z_L two-state split
# and the L-inside-H schedule. That difference is the one being measured.
# Protocol parity with pp_base_tf_z_iter / pp_abl_tfb requires the enqueue-time
# env DATA_ROOT=data/power_permutation/d1 (n=10, ord(σ)>k-clean D1 build) -- the
# script default DATA_ROOT is data/sigma_k_10, same as every other stage here.
#   tag | arch overrides (appended after arch.H_layers / arch.halt_max_steps=1)
LTF_ARMS=(
    "fullgrad|arch.loops=21 arch.loop_grad_cycles=0 arch.loop_input_injection=True"
    "g7|arch.loops=21 arch.loop_grad_cycles=7 arch.loop_input_injection=True"
    "noinj|arch.loops=21 arch.loop_grad_cycles=0 arch.loop_input_injection=False"
)
LTF_LAYERS=(${LTF_LAYERS:-2})
LTF_SEEDS=(${LTF_SEEDS:-1 2 3})   # private seed array (EXP-013 lesson, see INV_SEEDS)

# ---- heads8 control of the ltf g7 arm (stage `ltfh8`, prereg DRAFT 2026-11-14) ----
# lab/23_reports/11_lab-reports/2026-11-14_heads8-g7-control-prereg-DRAFT.md
# The EXP-015 g7 arm with ONE added key: arch.num_heads 12 -> 8 (TRM's attention
# geometry: head_dim 64, width 512, 6,828,034 params = trm tf_z_iter).
# A SEPARATE stage, array and name family on purpose:
#   - LTF_ARMS / STAGES=ltf must keep emitting exactly the 45 EXP-015 names
#     (exp015_gate_bodies.py, pg1_body_gate.py);
#   - run names are pp_ltfh8_*, NOT pp_ltf_*, so the pp_ltf_* name globs and the
#     run-family rule ^pp_ltf_*_lay2_k\d+_s\d+$ (EXP-015 / EXP-023) do not match.
# Private k and seed arrays. Needs DATA_ROOT=data/power_permutation/d1 like `ltf`.
#   tag | arch overrides (appended after arch.H_layers=2 arch.halt_max_steps=1)
LTFH8_ARMS=(
    "g7|arch.loops=21 arch.loop_grad_cycles=7 arch.loop_input_injection=True arch.num_heads=8"
)
LTFH8_K=(${LTFH8_K:-3 5 6 7})
LTFH8_SEEDS=(${LTFH8_SEEDS:-1 2 3})

# ---- ACT axis (stage `act`) — EXP-016 / H-200 --------------------------------
# Grid is FROZEN by lab/21_experiments/11_lab-experiments/
# EXP-016_act-deep-supervision-axis.md §4.2 / §4.4 (ACT_ARMS below is the §4.4
# declaration verbatim). Emitted bodies must stay byte-equal to §4.2.
# Every live cohort pins arch.halt_max_steps=1, so ACT / deep supervision has
# been OFF everywhere that matters. This stage turns it on (halt=16) for:
#   noz_iter           : trm_singlez, the H-collapse cohort (A1)
#   ltf_fullgrad       : looped transformers_baseline, adaptive halting in
#                        training only (act_enabled=True act_inference=False) (A2)
#   ltf_fullgrad_fixed : same, act_enabled=False -> always 16 steps (A3,
#                        fixed-16 control for A2)
# With halt_max_steps>1 the inner model is re-applied up to halt_max_steps
# times, carry DETACHED between ACT steps, loss on every step (deep supervision).
# Halting: trm_singlez has NO act_enabled / act_inference fields
# (trm_singlez.py:272): adaptive only in training (q_halt > 0), always the full
# halt_max_steps at eval. transformers_baseline with act_inference=False also
# runs the full halt_max_steps at eval; in training (act_enabled=True) it halts
# on q_halt > q_continue, and q_continue gets NO loss term (compute_target_q has
# no caller), so the rule is effectively q_halt > a near-constant from bias -5.
# Same DATA_ROOT note as LTF_ARMS (env DATA_ROOT=data/power_permutation/d1).
# A0 (the halt=1 comparator top-up) is NOT emitted here: stage fig1 with
# COHORT_FILTER=tf_noz_iter emits it (EXP-016 §4.4 enqueue procedure).
# NOTE cell_id reuses arm=act, which ablation_act also emits -- filter on the
# extra `cohort=` key to separate them.
# one arm per entry:  tag | arch | halt | k list (emission order) | arch overrides
# (halt is appended as arch.halt_max_steps=<halt> AFTER the overrides; Hydra last-key-wins)
ACT_ARMS=(
  "noz_iter|trm_singlez|16|5 6 7 3 10|arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 arch.halt_exploration_prob=0.1"
  "ltf_fullgrad|transformers_baseline|16|5 6 7 3 10|arch.H_layers=2 arch.loops=21 arch.loop_grad_cycles=0 arch.loop_input_injection=True arch.halt_exploration_prob=0.1 arch.act_enabled=True arch.act_inference=False"
  "ltf_fullgrad_fixed|transformers_baseline|16|5 6 7|arch.H_layers=2 arch.loops=21 arch.loop_grad_cycles=0 arch.loop_input_injection=True arch.halt_exploration_prob=0.1 arch.act_enabled=False arch.act_inference=False"
)
ACT_SEEDS=(${ACT_SEEDS:-1 2 3})   # private seed array

# ---- SP-04-L03 judgment cells (G-20260813-inverse-learnability) ----------
# tf_noz_iter (the H-collapse cohort) at k in {9,12,13,16}: does it learn AT
# ALL under D0 data given a longer budget than the live campaign default?
# User-confirmed design 2026-08-13. See the invcells stage body in main() for
# why epochs=100000 (not the live default 50000) and why RUN_GROUP is a
# dedicated "inverse-cells-d0".
#
# INV_SEEDS is intentionally its own array -- never SEEDS / ABLATION_SEEDS /
# TFB_SEEDS. A previous campaign (EXP-013) shared a seed array across two
# runs that were meant to be independent, which silently coupled their seed
# semantics; giving every campaign its own private seed array is how that
# does not repeat here.
#
# INV_K_LIST is ordered with 16 first, not for grid semantics (order does not
# change WHICH 12 cells are emitted) but because the intended real-launch
# tripwire is k=16 s=1 fired alone first: k=16 has the narrowest achievable
# ord(sigma) pool in the whole campaign (K_LIST comment above: ord>16 ->
# only {20,21,30}), so it is the most likely of the four k values to expose a
# data or config problem early, before the other 11 cells are released. This
# file does not enforce the tripwire (this task is dry-run-definition only,
# no enqueue) -- it is an operator obligation at real-launch time, recorded
# here so the ordering intent is not lost.
INV_K_LIST=(${INV_K_LIST:-16 12 13 9})
INV_SEEDS=(${INV_SEEDS:-1 2 3})

# INV_CELLS_FILTER (real-launch tripwire, added 2026-08-13): optional
# "k:seed[,k:seed,...]" allowlist restricting invcells emission to exactly
# the listed cells. UNSET (default) is IDENTICAL to the prior behaviour --
# every (k,s) in INV_K_LIST x INV_SEEDS is emitted, all 12. Exists because
# under FIFO + 4-worker parallelism, "enqueue all 12 invcells jobs" makes any
# single-cell tripwire (fire k=16 s=1 alone, confirm it lands in
# scripts/queue/done/, THEN release the other 11) purely aspirational unless
# the enqueuer itself can be told to emit fewer than all 12.
#   INV_CELLS_FILTER="16:1" scripts/sigma_enqueue.sh   # tripwire cell only
INV_CELLS_FILTER="${INV_CELLS_FILTER:-}"

# inv_cell_allowed <k> <seed> -- true iff INV_CELLS_FILTER is unset, or "k:s"
# is one of its comma-separated entries. See INV_CELLS_FILTER above.
inv_cell_allowed() {
    local k="$1" s="$2" pair
    [[ -z "$INV_CELLS_FILTER" ]] && return 0
    local -a _pairs
    IFS=',' read -r -a _pairs <<< "$INV_CELLS_FILTER"
    for pair in "${_pairs[@]}"; do
        [[ "$(trim "$pair")" == "${k}:${s}" ]] && return 0
    done
    return 1
}

# ---- Seed-extension grid (SP-04-L04 / GATE-EXP013-LAUNCH narrowed alt) ----
# tf_noz_iter, D0, at the 7 k values where seed=1 already fit
# (k=3,4,5,6,7,8,10 -- these are the baseline/fig1 k's known to reach the
# collapse-diagnostic band, NOT the invcells k's above): seeds 2,3 only.
# seed=1 is deliberately excluded from the array (not left in and relied on
# to be idempotency-skipped) -- SEEDEXT_SEEDS={2,3} IS the grid definition,
# per the same EXP-013 lesson as INV_SEEDS: a seed array is campaign
# semantics, not a superset to be filtered down at enqueue time.
# User-approved 2026-08-13. See the seedext stage body in main() for why
# epochs=100000 and RUN_GROUP="seedext-d0".
SEEDEXT_K_LIST=(${SEEDEXT_K_LIST:-3 4 5 6 7 8 10})
SEEDEXT_SEEDS=(${SEEDEXT_SEEDS:-2 3})

# ---- EXP-018 stages (`e018_*`) — separation-signal gates, 129 runs ----------
# Declared from lab/21_experiments/11_lab-experiments/EXP-018_separation-signal-gates.md
# §4.3 (grid) / §4.4 (stage order) + 19_DECISIONS_2026-10-07.md (b,j,l).
# Added 2026-10-20 (prep lane B). NOT yet enqueued; draft prereg, not frozen.
# Selected ONLY with run prefix `e018` (hard error otherwise) and explicit
# STAGES tokens; there is deliberately NO aggregate token:
#   e018_g1   57  G1 NULL: NT4a 30 + NT4b 12 + NT4c 6 + noiter 9 (config only)
#   e018_g3    9  G3 OPT: fullgrad 3 + cyclelossdetach 3 + learninit_fullgrad 3
#                 (flags merge needed; learninit = learnable_init + full_grad_cycles,
#                 requester decision B1, intake 9d5fd91cab / witness W4)
#   e018_g4   21  G4: cap 6 + per 3 + bcopy 6 + carry 3 + stb_zhreset 3
#   e018_g2   39  G2: sym2/singlefull/singletok x k5,6,7,8 x 3 + noz k8 x 3
#   e018_lb    3  BLOCKED (LB2 zH_freeze x3: B2 stop-gradient is a no-op at
#                 halt_max_steps=1, arm held per requester, witness W4). Real
#                 enqueue refused unless E018_ALLOW_BLOCKED=1. Never part of
#                 any other token.
# 129 = 57 + 9 + 21 + 39 + 3. Regression runs (2) and the G1-hold seed 13/14
# extension (12) are NOT emitted here.
# Common: 1.5x budget = epochs=150000 (int(150000*5000/2048)=366,210 steps; 75
# evals at eval_interval=2000), checkpoint_every_eval=True, project
# EXP018_sigma_k, data data/sigma_k_10 (ds2/ds3 for NT4b k=6). Seeds start at 8
# (private arrays; seeds 1-7 refused unless E018_ALLOW_LOW_SEEDS=1).
# Flag keys (inject_H, inject_last, step_token, full_grad_cycles, per_cycle_loss,
# learnable_init, zH_*, zL_reset, boundary_copy, x_copy) exist only in worktree
# trm-exp018-flags_claude until merge; real enqueue refuses an arm whose key is
# absent from config/arch/<arch>.yaml (struct mode would kill the job anyway).
# The per-stage `df` gate (decision l) is checked here on real runs only; the
# stage-wise order (decision j) remains an operator obligation.
# spec: block|tag|arch|k list|seedset(A=5 seeds,B=3 seeds)|data seeds|arch overrides
E018_SEEDS_A=(${E018_SEEDS_A:-8 9 10 11 12})
E018_SEEDS_B=(${E018_SEEDS_B:-8 9 10})
E018_EPOCHS="${E018_EPOCHS:-150000}"
E018_DATA_ROOT="${E018_DATA_ROOT:-data/sigma_k_10}"
E018_BASE="arch.mlp_t=False arch.L_layers=2 arch.halt_max_steps=1"
E018_CYC="arch.H_cycles=3 arch.L_cycles=6"
E018_G1_ARMS=(
  "nt4a|asym2|trm|5 6 7|A|1|${E018_CYC}"
  "nt4a|noz|trm_singlez|5 6 7|A|1|${E018_CYC}"
  "nt4b|asym2|trm|6|B|2 3|${E018_CYC}"
  "nt4b|noz|trm_singlez|6|B|2 3|${E018_CYC}"
  "nt4c|asym2|trm|8 10|B|1|${E018_CYC}"
  "noiter|noz|trm_singlez|8 10|B|1|arch.H_cycles=1 arch.L_cycles=1"
  "noiter|asym2|trm|10|B|1|arch.H_cycles=1 arch.L_cycles=1"
)
E018_G3_ARMS=(
  "g3|noz_fullgrad|trm_singlez|6|B|1|${E018_CYC} arch.full_grad_cycles=True"
  "g3|noz_cyclelossdetach|trm_singlez|6|B|1|${E018_CYC} arch.per_cycle_loss=True"
  "g3|noz_learninit_fullgrad|trm_singlez|6|B|1|${E018_CYC} arch.learnable_init=True arch.full_grad_cycles=True"
)
E018_G4_ARMS=(
  "g4|cap_noz1024|trm_singlez|6|B|1|${E018_CYC} arch.hidden_size=1024"
  "g4|cap_asym256|trm|6|B|1|${E018_CYC} arch.hidden_size=256"
  "g4|per_zhevery|trm|6|B|1|${E018_CYC} arch.zH_every_call=True"
  "g4|bcopy_boundary|trm_singlez|6|B|1|${E018_CYC} arch.boundary_copy=True"
  "g4|bcopy_xcopy|trm_singlez|6|B|1|${E018_CYC} arch.x_copy=True"
  "g4|carry_zlreset|trm|6|B|1|${E018_CYC} arch.zL_reset=True"
  "g4|stb_zhreset|trm|6|B|1|${E018_CYC} arch.zH_reset=True"
)
E018_G2_ARMS=(
  "g2|sym2|trm|5 6 7 8|B|1|${E018_CYC} arch.inject_H=True"
  "g2|singlefull|trm_singlez|5 6 7 8|B|1|${E018_CYC} arch.inject_last=True"
  "g2|singletok|trm_singlez|5 6 7 8|B|1|${E018_CYC} arch.inject_last=True arch.step_token=True"
  "g2|noz|trm_singlez|8|B|1|${E018_CYC}"
)
E018_LB_ARMS=(
  "g4|stb_zhfreeze|trm|6|B|1|${E018_CYC} arch.zH_freeze=True"
)

# e018_preflight -- no-op unless the e018 prefix or an e018_* token is in play.
e018_preflight() {
    local has_tok=0
    [[ " $STAGES " == *" e018_"* ]] && has_tok=1
    if [[ "$RUN_PREFIX" == "e018" ]] && (( ! has_tok )); then
        echo "ERROR: prefix e018 needs explicit STAGES with e018_* tokens" >&2
        echo "       (e018_g1 e018_g3 e018_g4 e018_g2 e018_lb); default STAGES would emit e018_base_*." >&2
        exit 2
    fi
    if (( has_tok )) && [[ "$RUN_PREFIX" != "e018" ]]; then
        echo "ERROR: e018_* stages require run prefix 'e018' (got '${RUN_PREFIX}')" >&2
        exit 2
    fi
    return 0
}

# ---- EXP-020 stages (`e020_*`) -- EXP-016 supplement, 27 runs (DRAFT prereg) ----
# Declared from lab/23_reports/11_lab-reports/2026-10-22_exp020-act-budget-extension-prereg.md §4.
# NOT enqueued; prereg not frozen. Selected ONLY with run prefix `e020` (hard error otherwise)
# and explicit STAGES tokens; there is deliberately NO aggregate token:
#   e020_s1  18  C-EXT-B: per (k, seed) the pair Cx (trm_singlez halt 1) then A1x (trm_singlez
#                h_max 16), epochs 76000 (38 evals; last logged step 38*4882 = 185,516)
#   e020_s2   9  C-EXT-H: A2n looped TF h_max 16 adaptive, k 2 3 4 x seeds 4 5 6, epochs 50000
# DATA_ROOT and WANDB_PROJECT are pinned INSIDE the stages (EXP-016 parity: D1 and project
# power_permutation). The script default data/sigma_k_10 is a different dataset (other files).
# Grids are fixed by the prereg: no env override of k, seeds or epochs.
E020_DATA_ROOT="data/power_permutation/d1"
E020_PROJECT="power_permutation"
E020_S1_EPOCHS=76000
E020_S1_K=(5 6 7)
E020_S1_SEEDS=(1 2 3)
E020_S2_K=(2 3 4)
E020_S2_SEEDS=(4 5 6)
E020_NOZ="arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6"
E020_LTF="arch.H_layers=2 arch.loops=21 arch.loop_grad_cycles=0 arch.loop_input_injection=True arch.halt_exploration_prob=0.1 arch.act_enabled=True arch.act_inference=False"

# e020_preflight -- no-op unless the e020 prefix or an e020_* token is in play.
e020_preflight() {
    local has_tok=0
    [[ " $STAGES " == *" e020_"* ]] && has_tok=1
    if [[ "$RUN_PREFIX" == "e020" ]] && (( ! has_tok )); then
        echo "ERROR: prefix e020 needs explicit STAGES with e020_* tokens (e020_s1 e020_s2)" >&2
        exit 2
    fi
    if (( has_tok )) && [[ "$RUN_PREFIX" != "e020" ]]; then
        echo "ERROR: e020_* stages require run prefix 'e020' (got '${RUN_PREFIX}')" >&2
        exit 2
    fi
    return 0
}

# e018_gate <stage> <est_gb> -- df gate (decision l). Real run: hard fail.
# est_gb = 1.96 GB/run (prereg 4.5) with hidden-1024 arm taken as ~8 GB/run
# (prereg: "약 4배 추정, 미실측"). Margin E018_DF_MARGIN_GB (default 50).
e018_gate() {
    local stage="$1" need="$2" avail margin="${E018_DF_MARGIN_GB:-50}"
    avail="$(df --output=avail -BG / | tail -n 1 | tr -dc '0-9')"
    [[ -n "$avail" ]] || { echo "ERROR: e018 df gate could not read df" >&2; exit 1; }
    if (( DRY_RUN )); then
        echo "e018 df gate (info) stage=${stage} avail=${avail}G need=${need}G margin=${margin}G" >&2
    elif (( avail < need + margin )); then
        echo "ERROR: e018 df gate FAILED stage=${stage} avail=${avail}G need=${need}G margin=${margin}G" >&2
        exit 1
    fi
    return 0
}

# e018_emit_arms <ARMS array name>
e018_emit_arms() {
    local -n _e018_arms="$1"
    local spec blk tag arch ks sset dss ov k d s key
    local -a seeds
    local _save_root="$DATA_ROOT" _save_proj="$WANDB_PROJECT" _save_grp="$RUN_GROUP"
    WANDB_PROJECT="${E018_WANDB_PROJECT:-EXP018_sigma_k}"
    for spec in "${_e018_arms[@]}"; do
        IFS='|' read -r blk tag arch ks sset dss ov <<< "$spec"
        if [[ "$sset" == "A" ]]; then seeds=("${E018_SEEDS_A[@]}"); else seeds=("${E018_SEEDS_B[@]}"); fi
        for s in "${seeds[@]}"; do
            if (( s < 8 )) && [[ "${E018_ALLOW_LOW_SEEDS:-0}" != "1" ]]; then
                echo "ERROR: e018 seed ${s} < 8 (H108N seeds 1-7 are not reusable)" >&2; exit 2
            fi
        done
        for key in $(grep -oE 'arch\.(inject_H|inject_last|step_token|full_grad_cycles|per_cycle_loss|learnable_init|zH_every_call|zH_freeze|zL_reset|zH_reset|boundary_copy|x_copy)=' <<< "$ov" || true); do
            key="${key#arch.}"; key="${key%=}"
            if ! grep -q "^${key}:" "config/arch/${arch}.yaml" 2>/dev/null; then
                if (( DRY_RUN )); then
                    echo "e018 WARN (info): ${key} absent from config/arch/${arch}.yaml (flags not merged)" >&2
                else
                    echo "ERROR: e018 flag ${key} absent from config/arch/${arch}.yaml; merge flags first" >&2; exit 1
                fi
            fi
        done
        RUN_GROUP="e018_${blk}"
        for k in $ks; do
            for d in $dss; do
                if [[ "$d" == "1" ]]; then DATA_ROOT="$E018_DATA_ROOT"; else DATA_ROOT="${E018_DATA_ROOT}_ds${d}"; fi
                for s in "${seeds[@]}"; do
                    CELL_ID="arm=e018,block=${blk},cohort=${tag},n=10,k=${k},s=${s},ds=${d}"
                    emit_job "${prefix}${blk}_${tag}_k${k}_s${s}$( [[ "$d" == "1" ]] || echo "_ds${d}" )" \
                        "$arch" "$k" "$s" \
                        "${E018_BASE} ${ov}" \
                        "epochs=${E018_EPOCHS} checkpoint_every_eval=True"
                done
            done
        done
    done
    DATA_ROOT="$_save_root"; WANDB_PROJECT="$_save_proj"; RUN_GROUP="$_save_grp"
    return 0
}

# CELL_ID is set by each stage before calling emit_job (see the stages below)
# and is emitted into the run config verbatim. It is the machine-readable grid
# coordinate; run_name remains the human-facing label. Intake
# 20260806T052122Z-c65f6d6d0f asked for `k{K}_z{0|1}_it{0|1}_N{N}_s{SEED}`,
# which only spans the old fig1 tau-grid axes and cannot express the halt/H/L/
# lay/tfb ablation arms at all -- so the intent (mechanical run -> cell
# mapping) is preserved here in a form that covers every arm.
CELL_ID=""

# emit_job <run_name> <arch> <k> <seed> <arch_args...>
#
# NOTE the '"..."' nesting on +cell_id, which is NOT cosmetic.  The value is a
# comma-separated key=value string ("arm=baseline,block=tf,z=1,..."), and Hydra
# parses each override's VALUE with its own grammar, in which a bare inner '='
# is a syntax error:
#
#     +cell_id=arm=baseline,block=tf,...
#       -> "mismatched input '=' expecting <EOF>"
#
# A plain "${CELL_ID}" does not help, because the job body is a bash script:
# bash strips those quotes and execs argv WITHOUT them, so Hydra still sees the
# bare form.  The quotes have to survive bash and reach Hydra, hence single
# quotes wrapping double quotes.  Verified with `--cfg job`: the parsed value
# is the clean string 'arm=baseline,block=tf,k=5,s=1', no quotes retained.
# The other +key="..." lines below are safe unquoted (no '=' or ',' in their
# values) and are left as they are so this comment marks the one real hazard.
emit_job() {
    local run_name="$1" arch="$2" k="$3" s="$4"; shift 4
    enqueue "$run_name" <<EOF
uv run pretrain.py arch=${arch} ${common_args} \\
    $* \\
    evaluators="[]" \\
    data_paths="[${DATA_ROOT}/${k}]" \\
    seed=${s} \\
    +k=${k} \\
    +project_name="${WANDB_PROJECT}" \\
    +run_group="${RUN_GROUP}" \\
    +run_name="${run_name}" \\
    +cell_id='"${CELL_ID}"' \\
    ema=True
EOF
}

# ---- PERF-001 P0.1: profiler-only preset (registered M3) -----------------
# Selected EXCLUSIVELY by run prefix `perf0`; STAGES is ignored in that mode,
# so this can never emit a fig1/ablation cell. Field text is byte-identical
# to the registered M3 command in
# lab/reports/2026-07-26_experiment-speed-profiling.md §4.2.1, with two
# documented deviations: (1) the job body never sets CUDA_VISIBLE_DEVICES —
# the queue worker owns it (queue_run.sh:165); (2) the body uses bare
# `uv run` like every other job body — `rtk` is an operator-shell output
# filter, not a registered Hydra field, and would mangle the queue log.
# perf_benchmark stays at its disabled defaults: M3 is a profiler run.
PERF0_PROFILE_DIR="reports/figures/2026-07-26_experiment-speed-profiling/profiles/m3_tf_z_iter_k6_s1"

emit_perf0_m3_job() {
    local run_name="${prefix}m3_tf_z_iter_k6_s1"
    enqueue "$run_name" <<EOF
uv run pretrain.py arch=trm \\
  global_batch_size=2048 epochs=20 eval_interval=20 min_eval_interval=0 \\
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \\
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \\
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 \\
  arch.halt_max_steps=1 arch.forward_dtype=bfloat16 \\
  data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \\
  +log_z_dynamics=True +z_snapshot=False \\
  +project_name=Sigma_k_perf +run_name=${run_name} \\
  perf_profiler.enabled=True perf_profiler.performance_only=True \\
  perf_profiler.output_dir=${PERF0_PROFILE_DIR} \\
  perf_profiler.wait=10 perf_profiler.warmup=10 perf_profiler.active=20 \\
  perf_profiler.repeat=1 perf_profiler.max_steps=40 \\
  perf_profiler.record_shapes=True perf_profiler.profile_memory=True \\
  perf_profiler.with_stack=True perf_profiler.export_chrome_trace=True \\
  perf_profiler.export_tensorboard_trace=True
EOF
}

# ---- PERF-001 P0.1a: unprofiled benchmark preset (registered M2) ---------
# Closes PERF-DEV-14 ("no runnable M2 path ships"): before this, `--dry-run
# perf0` emitted the M3 profiler job only, so the M2 baseline the plan requires
# to run *as a queue job body* had nothing to enqueue.
#
# Field text is byte-identical to the registered M2 command in
# lab/reports/2026-07-26_experiment-speed-action-plan.md §P0.1a (lines 99-109),
# with the same two documented deviations as the M3 emitter above:
# (1) the job body never sets CUDA_VISIBLE_DEVICES — the queue worker owns it
# (queue_run.sh:165); (2) the body drops the operator-shell `rtk` prefix
# (PERF-DEV-16 in the 2026-07-28 ratification ledger of the same document).
#
# The timing contract registers three repeats r1..r3 with "seed=1 and same
# reserved GPU; repeat/seed/hardware differences are provenance, not pooled
# silently". This emitter discharges the repeat/provenance half ONLY: it emits
# three separate jobs whose +run_name=..._r<n> and
# perf_benchmark.output_dir=..._r<n> differ and whose every other field is
# identical, so the three repeats can never be silently pooled into one
# artifact set. The `_r<n>` suffix is the convention utils/perf_benchmark.py
# already parses — derive_condition_id() strips it to the shared condition,
# derive_repeat() reads the repeat index off it (see _REPEAT_SUFFIX and those
# two functions).
#
# The "same reserved GPU" half is NOT discharged here and is NOT recorded as a
# numbered deviation in the 2026-07-28 PERF-DEV ledger. Emitting three jobs
# gives no placement or ordering guarantee whatsoever: queue_run.sh:180-183
# spawns one independent worker PER GPU (default GPUS="4 5 6 7"), each worker
# claims the oldest job it can claim (queue_run.sh:144-146) regardless of which
# condition it belongs to, and runs it on its own device (queue_run.sh:165).
# With the default worker set, r1/r2/r3 are claimable concurrently by three
# workers on three different GPUs — three timing baselines contending for the
# same host CPU/PCIe/power while each measures steady state. A single-worker
# invocation does not fix this either: worker_loop takes the oldest job in the
# WHOLE FIFO, not the oldest perf0 job, so unrelated science jobs (and the M3
# profiler job emitted just above, which precedes r1) would interleave on the
# reserved device. Satisfying "same reserved GPU" is therefore an operator
# obligation at run time, outside anything this file can enforce, and the gap
# belongs to the pre-registration owner to number and rule on.
#
# PERF0_BENCH_DIR_BASE is deliberately UNprefixed (like PERF0_PROFILE_DIR):
# the registered artifact path is .../data/m2_tf_z_iter_k6_s1_r<n>, whereas the
# run name carries the `perf0_` queue prefix. Do not build one from the other.
#
# Caveat carried from PERF-DEV-06: perf_benchmark.max_steps bounds ROW
# COLLECTION, not training length (unlike perf_profiler.max_steps). The real
# training bound of this preset is Hydra `epochs=2000`.
PERF0_BENCH_DIR_BASE="reports/figures/2026-07-26_experiment-speed-profiling/data/m2_tf_z_iter_k6_s1"
PERF0_M2_REPEATS=(1 2 3)

# emit_perf0_m2_job <repeat_index>
emit_perf0_m2_job() {
    local r="$1"
    local run_name="${prefix}m2_tf_z_iter_k6_s1_r${r}"
    enqueue "$run_name" <<EOF
uv run pretrain.py arch=trm \\
  global_batch_size=2048 epochs=2000 eval_interval=2000 min_eval_interval=0 \\
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \\
  ema=True ema_rate=0.999 evaluators="[]" checkpoint_every_eval=False \\
  arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 \\
  arch.halt_max_steps=1 arch.forward_dtype=bfloat16 \\
  data_paths="[data/sigma_k_10/6]" seed=1 +k=6 \\
  +log_z_dynamics=True +z_snapshot=False \\
  +project_name=Sigma_k_perf +run_name=${run_name} \\
  perf_profiler.enabled=False \\
  perf_benchmark.enabled=True perf_benchmark.performance_only=True \\
  perf_benchmark.warmup_steps=50 perf_benchmark.measured_steps=200 \\
  perf_benchmark.eval_event_step=2000 perf_benchmark.max_steps=2000 \\
  perf_benchmark.output_dir=${PERF0_BENCH_DIR_BASE}_r${r}
EOF
}

# Which stage groups to emit — space-separated subset of "fig1 ablation".
# Lets two machines split the grid cleanly (e.g. STAGES=fig1 on one host,
# STAGES=ablation on another) without touching the grid definitions above.
# 2026-08-06: "ablation" split into ablation_act / ablation_arch / tfb (each its
# own wandb group). The bare legacy token "ablation" is still accepted and
# expands to all three, so any existing caller keeps its old meaning.
# 2026-08-06: the DEFAULT is now stage 1 alone, not the whole product.
# Running `sigma_enqueue.sh pp` used to emit 378 jobs; it now emits 14 -- the
# baseline k-sweep. Each later stage is enqueued deliberately, after reading
# the previous one:
#
#   STAGES=baseline       14 jobs   baseline cohort across the full K_LIST.
#                                   Establishes WHERE the task breaks on this
#                                   data generation. Everything downstream
#                                   picks its k values from this result.
#   STAGES=seedvar        ~10 jobs  seeds 2,3 on K_DIAG for the baseline only.
#                                   Answers "is a single-seed cell readable at
#                                   all?" -- and there is direct reason to
#                                   doubt it: two runs of an identical config
#                                   at the same seed reached final_test_exact
#                                   0.980 vs 0.164 (see the TFB_SEEDS note).
#                                   If dispersion is large, EVERY single-seed
#                                   comparison below is uninterpretable and
#                                   the campaign must widen seeds before
#                                   spending anything on ablations.
#   STAGES=ablation_act   20 jobs   halt in {2,4,8,16} on K_DIAG. The ONLY
#                                   stage that yields a per-ACT-step
#                                   trajectory, so it is the only substrate
#                                   for the tau(h)/PR(h)/MI(h) instrumentation.
#   STAGES=ablation_arch  30 jobs   one architecture factor at a time on K_DIAG.
#   STAGES=cohorts        40 jobs   the 2x2x2 block x z x iter factorial on
#                                   K_DIAG. Genuinely a grid -- kept last
#                                   because it is the only stage whose
#                                   questions need joint variation.
#   STAGES=tfb            45 jobs   transformer baseline, 3 seeds on K_DIAG.
#   STAGES=invcells       12 jobs   SP-04-L03 judgment cells
#                                   (G-20260813-inverse-learnability):
#                                   tf_noz_iter at k in {9,12,13,16}, D0 data,
#                                   epochs=100000 (NOT the live default
#                                   50000), 3 seeds from the PRIVATE INV_SEEDS
#                                   array. run_group "inverse-cells-d0". Not
#                                   folded into any legacy token expansion --
#                                   must be selected explicitly.
#   STAGES=seedext        14 jobs   SP-04-L04 seed-extension grid
#                                   (GATE-EXP013-LAUNCH narrowed alt): same
#                                   tf_noz_iter / D0 / epochs=100000 protocol
#                                   as invcells, but at the 7 ALREADY-FITTING
#                                   baseline k's (3,4,5,6,7,8,10), seeds 2,3
#                                   only (seed=1 already done/, excluded from
#                                   the array by design, not filtered at
#                                   enqueue time). PRIVATE SEEDEXT_SEEDS array.
#                                   run_group "seedext-d0". Also must be
#                                   selected explicitly.
#   STAGES=ltf            45 jobs   looped-transformer control (LTF_ARMS
#                                   fullgrad/g7/noinj x K_DIAG x LTF_LAYERS=2 x
#                                   LTF_SEEDS 1,2,3): transformers_baseline
#                                   with a weight-tied loop, C=42 matched to
#                                   the trm baseline, halt=1. Arm-major FIFO
#                                   (whole fullgrad arm first). run_group
#                                   "ltf". Must be selected explicitly.
#   STAGES=act            39 jobs   ACT axis, EXP-016 §4.2 (ACT_ARMS, halt=16,
#                                   ACT_SEEDS 1,2,3): noz_iter k 5,6,7,3,10 (15)
#                                   -> ltf_fullgrad k 5,6,7,3,10 (15) ->
#                                   ltf_fullgrad_fixed k 5,6,7 (9). Arm-major
#                                   FIFO. Halting rules differ between arches --
#                                   see the ACT_ARMS comment. run_group "act".
#                                   Must be selected explicitly.
#   (ltf/act need DATA_ROOT=data/power_permutation/d1 for parity with the
#    pp_base / pp_abl_tfb cohorts they are compared against.)
#
# Legacy tokens still work: "fig1" == "cohorts" over the full K_LIST, and
# "ablation" expands to all three ablation stages, so any existing caller
# keeps its old meaning.
STAGES="${STAGES:-baseline}"
[[ " $STAGES " == *" ablation "* ]] && STAGES="$STAGES ablation_act ablation_arch tfb"

main() {
    local tier k spec tag arch arch_args mlp_t Hc Lc s lay cyc r
    e018_preflight
    e020_preflight
    # -- e026 (2026-11-10, user instruction): halt16 two-state TRM vs looped TF --
    # Prereg: lab/23_reports/11_lab-reports/2026-11-10_exp026-halt16-two-state-prereg.md
    # T16 seeds 2-3 copy pp_abl_halt16's flags (seed 1 = existing pp_abl_halt16_k*_s1);
    # G16 copies e016_act_ltf_fullgrad_h16 with loop_grad_cycles 0 -> 7 (EXP-015 g7).
    # D1 data, 50000 epochs, K_DIAG. Emitted only with STAGES=e026 and an empty prefix.
    if [[ " $STAGES " == *" e026 "* ]]; then
        if [[ -n "$RUN_PREFIX" ]]; then
            echo "ERROR: e026 uses fixed run names; run with no prefix" >&2
            exit 1
        fi
        local e026_common="epochs=50000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=True checkpoint_every_eval=False +eval_save_outputs=[inputs,labels,preds,puzzle_identifiers]"
        local ek es
        for ek in 3 5 6 7 10; do
            for es in 2 3; do
                enqueue "e026_trm_h16_k${ek}_s${es}" <<EOF
uv run pretrain.py arch=trm ${e026_common} \\
    arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 arch.halt_max_steps=16 +z_seq_metrics=True \\
    evaluators="[]" \\
    data_paths="[data/power_permutation/d1/${ek}]" \\
    seed=${es} \\
    +k=${ek} \\
    +project_name="power_permutation" \\
    +run_group="e026" \\
    +run_name="e026_trm_h16_k${ek}_s${es}" \\
    +cell_id='"arm=e026,cohort=trm,halt=16,n=10,k=${ek},s=${es}"' \\
    ema=True
EOF
            done
            for es in 1 2 3; do
                enqueue "e026_ltf_g7_h16_k${ek}_s${es}" <<EOF
uv run pretrain.py arch=transformers_baseline ${e026_common} \\
    arch.H_layers=2 arch.loops=21 arch.loop_grad_cycles=7 arch.loop_input_injection=True arch.halt_exploration_prob=0.1 arch.act_enabled=True arch.act_inference=False arch.halt_max_steps=16 \\
    evaluators="[]" \\
    data_paths="[data/power_permutation/d1/${ek}]" \\
    seed=${es} \\
    +k=${ek} \\
    +project_name="power_permutation" \\
    +run_group="e026" \\
    +run_name="e026_ltf_g7_h16_k${ek}_s${es}" \\
    +cell_id='"arm=e026,cohort=ltf_g7,halt=16,n=10,k=${ek},s=${es}"' \\
    ema=True
EOF
            done
        done
        return 0
    fi
    # -- cc27 (2026-11-18): ICML 2027 confirmation cohort, D1 k in K_F={5,6,7} (+k=3 for R3) --
    # Prereg: lab/23_reports/11_lab-reports/2026-11-18_icml27-confirm-cohort-prereg-DRAFT.md (§2, §3.1).
    # Dedicated stages (no existing array touched); explicit selection only, empty prefix:
    #   cc27_r2  TRM + g7 halt 1,  k 5 6 7, seeds 41-46   (36 jobs)
    #   cc27_r4  trm_singlez halt 1, k 5 6 7, seeds 61-63 ( 9)
    #   cc27_r3  g7s (arch=transformers_baseline_skip, skip period 7) halt 1, k 5 6 7 3, seeds 51-53 (12)
    #   cc27_r1  TRM + g7 halt 16, k 5 6 7, seeds 21-35   (90)
    # Bodies = the landed reference bodies of the same arm (pp_base_tf_z_iter, pp_ltf_g7_lay2,
    # e016_fig1_tf_noz_iter, e026_trm_h16, e026_ltf_g7_h16) with only seed/k and identity fields
    # (run_name, run_group, cell_id) changed; R3 = g7 body with arch=transformers_baseline_skip plus the
    # explicit arch.loop_input_injection_skip_period=7. Order: k -> seed -> arm. Data root fixed to D1.
    # Enqueue each stage, then rename per prereg G7 (R2 0722c_, R4 0722d_, R3 0722e_, R1 0722f_).
    if [[ " $STAGES " == *" cc27_r1 "* || " $STAGES " == *" cc27_r2 "* || " $STAGES " == *" cc27_r3 "* || " $STAGES " == *" cc27_r4 "* ]]; then
        if [[ -n "$RUN_PREFIX" ]]; then
            echo "ERROR: cc27 uses fixed run names; run with no prefix" >&2
            exit 1
        fi
        local cc_common="epochs=50000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=True checkpoint_every_eval=False +eval_save_outputs=[inputs,labels,preds,puzzle_identifiers]"
        local cc_trm_args="arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2"
        local cc_g7_h1="arch.H_layers=2 arch.halt_max_steps=1 arch.loops=21 arch.loop_grad_cycles=7 arch.loop_input_injection=True"
        local cc_g7_h16="arch.H_layers=2 arch.loops=21 arch.loop_grad_cycles=7 arch.loop_input_injection=True arch.halt_exploration_prob=0.1 arch.act_enabled=True arch.act_inference=False arch.halt_max_steps=16"
        # cc27_emit <tag: trm|g7|g7s|noz> <halt> <k> <s>
        cc27_emit() {
            local tag="$1" halt="$2" ck="$3" cs="$4" carch cargs
            case "${tag}_${halt}" in
                trm_1)   carch="trm";                    cargs="${cc_trm_args} arch.halt_max_steps=1" ;;
                trm_16)  carch="trm";                    cargs="${cc_trm_args} arch.halt_max_steps=16 +z_seq_metrics=True" ;;
                noz_1)   carch="trm_singlez";            cargs="${cc_trm_args} arch.halt_max_steps=1" ;;
                g7_1)    carch="transformers_baseline";  cargs="${cc_g7_h1}" ;;
                g7_16)   carch="transformers_baseline";  cargs="${cc_g7_h16}" ;;
                g7s_1)   carch="transformers_baseline_skip"; cargs="${cc_g7_h1} arch.loop_input_injection_skip_period=7" ;;
                *) echo "ERROR: cc27_emit: bad arm ${tag}_${halt}" >&2; exit 1 ;;
            esac
            enqueue "cc27_${tag}_h${halt}_k${ck}_s${cs}" <<EOF
uv run pretrain.py arch=${carch} ${cc_common} \\
    ${cargs} \\
    evaluators="[]" \\
    data_paths="[data/power_permutation/d1/${ck}]" \\
    seed=${cs} \\
    +k=${ck} \\
    +project_name="power_permutation" \\
    +run_group="cc27" \\
    +run_name="cc27_${tag}_h${halt}_k${ck}_s${cs}" \\
    +cell_id='"arm=cc27,cohort=${tag},halt=${halt},n=10,k=${ck},s=${cs}"' \\
    ema=True
EOF
        }
        local ck cs
        if [[ " $STAGES " == *" cc27_r2 "* ]]; then
            for ck in 5 6 7; do for cs in 41 42 43 44 45 46; do
                cc27_emit trm 1 "$ck" "$cs"; cc27_emit g7 1 "$ck" "$cs"
            done; done
        fi
        if [[ " $STAGES " == *" cc27_r4 "* ]]; then
            for ck in 5 6 7; do for cs in 61 62 63; do
                cc27_emit noz 1 "$ck" "$cs"
            done; done
        fi
        if [[ " $STAGES " == *" cc27_r3 "* ]]; then
            for ck in 5 6 7 3; do for cs in 51 52 53; do
                cc27_emit g7s 1 "$ck" "$cs"
            done; done
        fi
        if [[ " $STAGES " == *" cc27_r1 "* ]]; then
            for ck in 5 6 7; do for cs in 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
                cc27_emit trm 16 "$ck" "$cs"; cc27_emit g7 16 "$ck" "$cs"
            done; done
        fi
        return 0
    fi
    # -- cx27 (2026-11-19): ICML 2027 addendum cohort, D1 k in {5,6,7}, 3 seeds per k per cell --
    # Prereg: lab/23_reports/11_lab-reports/2026-11-19_icml27-addendum-prereg-DRAFT.md.
    # Dedicated stages (no existing array or stage touched); explicit selection only, empty prefix:
    #   cx27_xf  looped factorial, arch=transformers_baseline_px, halt 16, 7 cells x 3 seeds x k 5 6 7 (63)
    #            body = the cc27_g7_h16 body with arch=transformers_baseline -> transformers_baseline_px and
    #            three explicit overrides arch.num_heads / arch.puzzle_emb_len /
    #            arch.loop_input_injection_skip_period appended after the last arch override.
    #            cell (heads,prefix,period) seeds: px12p1s7 74-76; px8p1s0 77-79; px8p1s7 81-83;
    #            px12p16s0 84-86; px12p16s7 87-89; px8p16s0 91-93; px8p16s7 94-96
    #   cx27_x5  trm_singlez halt 16, k 5 6 7, seeds 71-73 (9); body = cc27_trm_h16 body with only arch swapped
    #   cx27_x7  trm_singlez halt 1, lr=3e-5 puzzle_emb_lr=3e-5, k 5 6 7, seeds 97-99 (9); body = cc27_noz_h1
    #            body with only the two learning-rate values changed (weight-decay overrides kept)
    # Run names: cx27_px{H}p{P}s{S}_h16_k{k}_s{s}, cx27_noz_h16_k{k}_s{s}, cx27_nozlr_h1_k{k}_s{s}.
    # Order inside a stage: k -> seed. Data root fixed to D1. Nothing is emitted under the default STAGES.
    if [[ " $STAGES " == *" cx27_xf "* || " $STAGES " == *" cx27_x5 "* || " $STAGES " == *" cx27_x7 "* ]]; then
        if [[ -n "$RUN_PREFIX" ]]; then
            echo "ERROR: cx27 uses fixed run names; run with no prefix" >&2
            exit 1
        fi
        local cx_common="epochs=50000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +log_z_dynamics=True +z_snapshot=True checkpoint_every_eval=False +eval_save_outputs=[inputs,labels,preds,puzzle_identifiers]"
        local cx_trm_args="arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2"
        local cx_g7_h16="arch.H_layers=2 arch.loops=21 arch.loop_grad_cycles=7 arch.loop_input_injection=True arch.halt_exploration_prob=0.1 arch.act_enabled=True arch.act_inference=False arch.halt_max_steps=16"
        # cx27_emit <tag> <halt> <k> <s> <heads> <prefix> <period>   (tag: pxHpPsS | noz | nozlr)
        cx27_emit() {
            local tag="$1" halt="$2" ck="$3" cs="$4" xh="$5" xp="$6" xs="$7" carch ccommon cargs
            ccommon="$cx_common"
            case "$tag" in
                noz)   carch="trm_singlez"; cargs="${cx_trm_args} arch.halt_max_steps=16 +z_seq_metrics=True" ;;
                nozlr) carch="trm_singlez"; cargs="${cx_trm_args} arch.halt_max_steps=1"
                       ccommon="${cx_common/lr=1e-4 puzzle_emb_lr=1e-4/lr=3e-5 puzzle_emb_lr=3e-5}" ;;
                px*)   carch="transformers_baseline_px"
                       cargs="${cx_g7_h16} arch.num_heads=${xh} arch.puzzle_emb_len=${xp} arch.loop_input_injection_skip_period=${xs}" ;;
                *) echo "ERROR: cx27_emit: bad arm ${tag}" >&2; exit 1 ;;
            esac
            enqueue "cx27_${tag}_h${halt}_k${ck}_s${cs}" <<EOF
uv run pretrain.py arch=${carch} ${ccommon} \\
    ${cargs} \\
    evaluators="[]" \\
    data_paths="[data/power_permutation/d1/${ck}]" \\
    seed=${cs} \\
    +k=${ck} \\
    +project_name="power_permutation" \\
    +run_group="cx27" \\
    +run_name="cx27_${tag}_h${halt}_k${ck}_s${cs}" \\
    +cell_id='"arm=cx27,cohort=${tag},halt=${halt},n=10,k=${ck},s=${cs}"' \\
    ema=True
EOF
        }
        local ck cs cxspec
        # seed|heads|prefix|period   (cell tag = px{heads}p{prefix}s{period})
        local -a cx_xf_cells=(
            "74|12|1|7" "75|12|1|7" "76|12|1|7"
            "77|8|1|0"  "78|8|1|0"  "79|8|1|0"
            "81|8|1|7"  "82|8|1|7"  "83|8|1|7"
            "84|12|16|0" "85|12|16|0" "86|12|16|0"
            "87|12|16|7" "88|12|16|7" "89|12|16|7"
            "91|8|16|0" "92|8|16|0" "93|8|16|0"
            "94|8|16|7" "95|8|16|7" "96|8|16|7"
        )
        local cxh cxp cxs
        if [[ " $STAGES " == *" cx27_x5 "* ]]; then
            for ck in 5 6 7; do for cs in 71 72 73; do
                cx27_emit noz 16 "$ck" "$cs" 0 0 0
            done; done
        fi
        if [[ " $STAGES " == *" cx27_xf "* ]]; then
            for ck in 5 6 7; do for cxspec in "${cx_xf_cells[@]}"; do
                IFS='|' read -r cs cxh cxp cxs <<< "$cxspec"
                cx27_emit "px${cxh}p${cxp}s${cxs}" 16 "$ck" "$cs" "$cxh" "$cxp" "$cxs"
            done; done
        fi
        if [[ " $STAGES " == *" cx27_x7 "* ]]; then
            for ck in 5 6 7; do for cs in 97 98 99; do
                cx27_emit nozlr 1 "$ck" "$cs" 0 0 0
            done; done
        fi
        return 0
    fi
    # -- legacy_redo (2026-11-03, user decision; PIP handoff 03:42 §3) --
    # Re-run the 11 legacy Sigma_k_fig12 cells that never completed:
    #   EXP-006 2x-extension arm (fig1x2_*, epochs=200000): noz_iter k7 s1-3,
    #     z_iter k4_s2, z_iter k5_s1  (4/9 of the arm completed in 2026-07)
    #   EXP-007 k11 discriminator arm (fig1d_*, epochs=100000): {noz,z}_iter
    #     k11 s1-3  (the 2026-07-20 runs died at start: 0 eval, 0 ckpt)
    # Bodies reproduce checkpoints/Sigma_k_fig12/<run>/all_config.yaml of the
    # original runs field-for-field (python difflib 2026-11-03 against
    # fig1x2_tf_noz_iter_k5_s1, which completed): legacy fig1 protocol, D0
    # data/sigma_k_10/<k>, flat LR, checkpoint_every_eval=True,
    # log_z_dynamics=True, z_snapshot=False, no eval_save_outputs/+k/+cell_id.
    # Run names are the ORIGINAL names (no prefix) so analysis/pr_extract.py
    # and the tau harness parse them; wandb will hold a second run id per
    # name -- local_tau_from_datastore.py keeps the id reaching the highest
    # step (restart guard). Emitted only with STAGES=legacy_redo and an EMPTY
    # run prefix.  k11 data: data/sigma_k_10/11 (exists, built 2026-07).
    if [[ " $STAGES " == *" legacy_redo "* ]]; then
        if [[ -n "$RUN_PREFIX" ]]; then
            echo "ERROR: legacy_redo uses fixed original run names; run with no prefix" >&2
            exit 1
        fi
        local legacy_common="eval_interval=2000 lr=1e-4 lr_min_ratio=1.0 lr_warmup_steps=2000 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 checkpoint_every_eval=True +log_z_dynamics=True +z_snapshot=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 arch.halt_max_steps=1"
        local lspec lname larch lk ls lep
        for lspec in \
            "fig1x2_tf_noz_iter_k7_s1|trm_singlez|7|1|200000" \
            "fig1x2_tf_noz_iter_k7_s2|trm_singlez|7|2|200000" \
            "fig1x2_tf_noz_iter_k7_s3|trm_singlez|7|3|200000" \
            "fig1x2_tf_z_iter_k4_s2|trm|4|2|200000" \
            "fig1x2_tf_z_iter_k5_s1|trm|5|1|200000" \
            "fig1d_tf_noz_iter_k11_s1|trm_singlez|11|1|100000" \
            "fig1d_tf_noz_iter_k11_s2|trm_singlez|11|2|100000" \
            "fig1d_tf_noz_iter_k11_s3|trm_singlez|11|3|100000" \
            "fig1d_tf_z_iter_k11_s1|trm|11|1|100000" \
            "fig1d_tf_z_iter_k11_s2|trm|11|2|100000" \
            "fig1d_tf_z_iter_k11_s3|trm|11|3|100000"; do
            IFS='|' read -r lname larch lk ls lep <<< "$lspec"
            enqueue "$lname" <<EOF
uv run pretrain.py arch=${larch} epochs=${lep} ${legacy_common} \\
    evaluators="[]" \\
    data_paths="[data/sigma_k_10/${lk}]" \\
    seed=${ls} \\
    +project_name="Sigma_k_fig12" \\
    +run_name="${lname}" \\
    ema=True
EOF
        done
        return 0
    fi
    # -- 0) PERF-001 profiler-only preset: prefix-selected and exclusive --
    if [[ "$RUN_PREFIX" == "perf0" ]]; then
        if (( ! DRY_RUN )) && [[ "${PERF0_ALLOW_ENQUEUE:-0}" != "1" ]]; then
            echo "ERROR: perf0 is dry-run-only until the G3 enqueue gate is signed" >&2
            echo "       (profiling doc §4.1: 'no production enqueue in this registration';" >&2
            echo "        §4.2 G3 requires an independent verifier PASS)." >&2
            echo "       Re-run with --dry-run, or set PERF0_ALLOW_ENQUEUE=1 after G3." >&2
            exit 1
        fi
        DRY_RUN_SHOW_BODY=1
        emit_perf0_m3_job
        for r in "${PERF0_M2_REPEATS[@]}"; do
            emit_perf0_m2_job "$r"
        done
        return 0
    fi
    # -- 1a) STAGE 1: baseline k-sweep. ONE cohort, every k, one seed. --
    # This is the map the rest of the campaign is read against: it locates the
    # k at which the baseline stops learning. Nothing else should be enqueued
    # until this is read, because every later stage picks its K_DIAG from it.
    if [[ " $STAGES " == *" baseline "* ]]; then
        RUN_GROUP="${RUN_GROUP_BASELINE:-baseline}"
        for k in "${K_LIST[@]}"; do
            for spec in "${COHORTS[@]}"; do
                IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                [[ "$tag" == "$BASELINE_COHORT" ]] || continue
                local blk zf itf
                blk="${tag%%_*}"
                case "$tag" in *_noz_*) zf=0;; *) zf=1;; esac
                case "$tag" in *_noiter) itf=0;; *) itf=1;; esac
                for s in "${SEEDS[@]}"; do
                    CELL_ID="arm=baseline,block=${blk},z=${zf},it=${itf},n=10,k=${k},s=${s}"
                    emit_job "${prefix}base_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                        "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                        "arch.L_layers=2 arch.halt_max_steps=1"
                done
            done
        done
    fi
    # -- 1b) STAGE 2: seed variance on the baseline, at K_DIAG only. --
    # Gate for everything after it. If identical configs disperse widely, no
    # single-seed ablation cell below can be read as a real effect.
    if [[ " $STAGES " == *" seedvar "* ]]; then
        RUN_GROUP="${RUN_GROUP_SEEDVAR:-seedvar}"
        for k in "${K_DIAG[@]}"; do
            for spec in "${COHORTS[@]}"; do
                IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                [[ "$tag" == "$BASELINE_COHORT" ]] || continue
                local blk zf itf
                blk="${tag%%_*}"
                case "$tag" in *_noz_*) zf=0;; *) zf=1;; esac
                case "$tag" in *_noiter) itf=0;; *) itf=1;; esac
                for s in ${SEEDVAR_SEEDS:-2 3}; do
                    CELL_ID="arm=seedvar,block=${blk},z=${zf},it=${itf},n=10,k=${k},s=${s}"
                    emit_job "${prefix}base_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                        "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                        "arch.L_layers=2 arch.halt_max_steps=1"
                done
            done
        done
    fi
    # -- 1c) cohort factorial (legacy name: fig1). Genuinely a grid, so it --
    #    runs LAST and only over K_DIAG unless the legacy "fig1" token is used.
    if [[ " $STAGES " == *" cohorts "* || " $STAGES " == *" fig1 "* ]]; then
        RUN_GROUP="${RUN_GROUP_FIG1:-fig1}"
        local -a COHORT_KS
        if [[ " $STAGES " == *" fig1 "* ]]; then COHORT_KS=("${K_LIST[@]}"); else COHORT_KS=("${K_DIAG[@]}"); fi
        for tier in "${COHORT_TIERS[@]}"; do
            for k in "${COHORT_KS[@]}"; do
                for spec in "${COHORTS[@]}"; do
                    IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                    [[ "$tag" == "${tier}_"* ]] || continue
                    cohort_allowed "$tag" || continue
                    # tag is "<block>_<z>_<iter>", e.g. tf_z_iter / mlp_noz_noiter.
                    # Decode it into the axes the old cell_id convention named,
                    # rather than shipping the tag string and making every
                    # consumer re-learn the grammar.
                    local blk zf itf
                    blk="${tag%%_*}"                       # mlp | tf
                    case "$tag" in *_noz_*) zf=0;; *) zf=1;; esac
                    case "$tag" in *_noiter) itf=0;; *) itf=1;; esac
                    for s in "${SEEDS[@]}"; do
                        CELL_ID="arm=fig1,block=${blk},z=${zf},it=${itf},n=10,k=${k},s=${s}"
                        emit_job "${prefix}fig1_${tag}_k${k}_s${s}" "$arch" "$k" "$s" \
                            "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                            "arch.L_layers=2 arch.halt_max_steps=1"
                    done
                done
            done
        done
    fi
    # -- 2a) ACT / halting ablation --
    # NOTE the ${arch_args} suffix: it comes AFTER the pinned
    # `arch.halt_max_steps=1` baseline, so a later duplicate key wins under
    # Hydra. That ordering is what makes halt2/4/8/16 actually take effect.
    #
    # z_seq_metrics is set HERE, in the stage body, rather than in common_args
    # or at the call site, because it is a property of the stage and not of the
    # invocation: the depth instrumentation (zseq/ zmi/ ztau/ zperm/) reduces
    # over the per-ACT-step trajectory, and ablation_act is the only stage that
    # produces one.  Every other stage pins arch.halt_max_steps=1, i.e. exactly
    # one ACT step, where those keys would be emitted as flat single-point
    # series -- readable as "the metric is constant" when the truth is "there
    # was nothing to vary".  Setting it stage-locally makes that impossible to
    # get wrong by forgetting an env var.  ACT_SEQ_METRICS=False turns it off
    # without touching this file.
    if [[ " $STAGES " == *" ablation_act "* ]]; then
        RUN_GROUP="${RUN_GROUP_ACT:-ablation_act}"
        for k in "${K_DIAG[@]}"; do
            for spec in "${ACT_ABLATIONS[@]}"; do
                IFS='|' read -r tag arch_args <<< "$spec"
                for s in "${SEEDS[@]}"; do
                    CELL_ID="arm=act,halt=${tag#halt},n=10,k=${k},s=${s}"
                    emit_job "${prefix}abl_${tag}_k${k}_s${s}" "trm" "$k" "$s" \
                        "arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6" \
                        "arch.L_layers=2 arch.halt_max_steps=1 ${arch_args}" \
                        "+z_seq_metrics=${ACT_SEQ_METRICS:-True}"
                done
            done
        done
    fi
    # -- 2b) model-structure ablation --
    if [[ " $STAGES " == *" ablation_arch "* ]]; then
        RUN_GROUP="${RUN_GROUP_ARCH:-ablation_arch}"
        for k in "${K_DIAG[@]}"; do
            for spec in "${ARCH_ABLATIONS[@]}"; do
                IFS='|' read -r tag arch_args <<< "$spec"
                for s in "${SEEDS[@]}"; do
                    # Decode the tag into semantic keys rather than shipping
                    # the tag string -- otherwise a consumer still has to know
                    # the tag grammar, which is exactly what cell_id exists to
                    # avoid (the other three arms already decode).
                    local akey aval
                    case "$tag" in
                        H*)   akey=Hc;     aval="${tag#H}"   ;;
                        L*[0-9]) akey=Lc;  aval="${tag#L}"   ;;
                        lay*) akey=layers; aval="${tag#lay}" ;;
                        *)    akey=axis;   aval="$tag"       ;;
                    esac
                    CELL_ID="arm=arch,${akey}=${aval},n=10,k=${k},s=${s}"
                    emit_job "${prefix}abl_${tag}_k${k}_s${s}" "trm" "$k" "$s" \
                        "arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6" \
                        "arch.L_layers=2 arch.halt_max_steps=1 ${arch_args}"
                done
            done
        done
    fi
    # -- 2c) transformer-baseline depth/width grid --
    if [[ " $STAGES " == *" tfb "* ]]; then
        RUN_GROUP="${RUN_GROUP_TFB:-tfb}"
        for k in "${K_DIAG[@]}"; do
            for lay in "${TFB_LAYERS[@]}"; do
                for s in "${TFB_SEEDS[@]}"; do
                    # arch.H_cycles deliberately NOT passed -- it is a no-op for
                    # this arch (see TFB_SEEDS note above). Passing it would only
                    # write a misleading value into the wandb config.
                    CELL_ID="arm=tfb,lay=${lay},n=10,k=${k},s=${s}"
                    emit_job "${prefix}abl_tfb_lay${lay}_k${k}_s${s}" \
                        "transformers_baseline" "$k" "$s" \
                        "arch.H_layers=${lay} arch.halt_max_steps=1"
                done
            done
        done
    fi
    # -- 2c') looped-transformer control (see LTF_ARMS above) --
    # Arm-major: the whole first arm is emitted across every k before the
    # second, so FIFO finishes one complete arm first. Arm overrides are
    # appended AFTER arch.halt_max_steps=1 (Hydra last-key-wins; they never
    # touch halt anyway). arch.H_cycles is not passed: still a no-op for tfb.
    if [[ " $STAGES " == *" ltf "* ]]; then
        RUN_GROUP="${RUN_GROUP_LTF:-ltf}"
        for spec in "${LTF_ARMS[@]}"; do
            IFS='|' read -r tag arch_args <<< "$spec"
            for k in "${K_DIAG[@]}"; do
                for lay in "${LTF_LAYERS[@]}"; do
                    for s in "${LTF_SEEDS[@]}"; do
                        CELL_ID="arm=ltf,ltf=${tag},lay=${lay},n=10,k=${k},s=${s}"
                        emit_job "${prefix}ltf_${tag}_lay${lay}_k${k}_s${s}" \
                            "transformers_baseline" "$k" "$s" \
                            "arch.H_layers=${lay} arch.halt_max_steps=1 ${arch_args}"
                    done
                done
            done
        done
    fi
    # -- 2c'-h8) heads8 control of the ltf g7 arm (see LTFH8_ARMS above) --
    # k-major, then seed (k=3 first: the trainability gate is the first observation).
    # Bodies equal the landed g7 bodies plus arch.num_heads=8 and the identity
    # fields (run_group, run_name, cell_id). Explicit selection only.
    if [[ " $STAGES " == *" ltfh8 "* ]]; then
        RUN_GROUP="${RUN_GROUP_LTFH8:-ltfh8}"
        for spec in "${LTFH8_ARMS[@]}"; do
            IFS='|' read -r tag arch_args <<< "$spec"
            for k in "${LTFH8_K[@]}"; do
                for s in "${LTFH8_SEEDS[@]}"; do
                    CELL_ID="arm=ltfh8,ltf=${tag},lay=2,n=10,k=${k},s=${s}"
                    emit_job "${prefix}ltfh8_${tag}_lay2_k${k}_s${s}" \
                        "transformers_baseline" "$k" "$s" \
                        "arch.H_layers=2 arch.halt_max_steps=1 ${arch_args}"
                done
            done
        done
    fi
    # -- 2c'') ACT axis (see ACT_ARMS above; EXP-016 §4.4) --
    # RUN_GROUP override is RUN_GROUP_ACTAX, NOT RUN_GROUP_ACT: the latter
    # already belongs to stage ablation_act, and sharing it would silently
    # regroup that stage too.
    # emission: arm-major -> k in the arm's listed order -> ACT_SEEDS.
    # act_enabled/act_inference live in the ACT_ARMS overrides (only
    # transformers_baseline declares them); halt_max_steps goes LAST.
    if [[ " $STAGES " == *" act "* ]]; then
        RUN_GROUP="${RUN_GROUP_ACTAX:-act}"
        local halt act_k overrides
        for spec in "${ACT_ARMS[@]}"; do
            IFS='|' read -r tag arch halt act_k overrides <<< "$spec"
            for k in $act_k; do
                for s in "${ACT_SEEDS[@]}"; do
                    CELL_ID="arm=act,cohort=${tag},halt=${halt},n=10,k=${k},s=${s}"
                    emit_job "${prefix}act_${tag}_h${halt}_k${k}_s${s}" \
                        "$arch" "$k" "$s" \
                        "${overrides}" \
                        "arch.halt_max_steps=${halt}"
                done
            done
        done
    fi
    # -- 2d) SP-04-L03 judgment cells (G-20260813-inverse-learnability) -----
    # tf_noz_iter (the H-collapse cohort) at k in {9,12,13,16}, D0 data
    # (DATA_ROOT is already data/sigma_k_10, the n=10 canonical corpus), with
    # a LONGER budget than the live campaign default: epochs=100000 (244,140
    # steps) instead of the live default epochs=50000 baked into common_args
    # (see the "2026-08-06: epochs 100000 -> 50000" comment above for why the
    # live default sits where it does -- this stage is a deliberate,
    # user-confirmed exception to it, not a reversion of that decision).
    # `epochs=100000` is passed as an emit_job extra arg, which lands AFTER
    # common_args in the job body, so it overrides common_args' epochs=50000
    # under Hydra's last-key-wins rule (the same ordering pattern documented
    # at the ablation_act arch_args note above). eval_save_outputs is already
    # inherited unmodified from common_args -- verified 2026-08-13 that it
    # only filters which already-computed tensors evaluate() detaches for
    # saving (pretrain.py:511,553; models/losses.py:100), entirely inside
    # torch.inference_mode() at a call site (pretrain.py:867) separate from
    # train_batch(), so it cannot affect the optimization trajectory.
    #
    # RUN_GROUP is a dedicated "inverse-cells-d0" (distinct from baseline/
    # ablation_*/tfb/fig1), and the seed array is the PRIVATE INV_SEEDS
    # defined above, never SEEDS/ABLATION_SEEDS/TFB_SEEDS (EXP-013 lesson).
    #
    # TRIPWIRE: fire k=16 s=1 alone first (INV_CELLS_FILTER="16:1") and
    # confirm it reaches scripts/queue/done/ before releasing the other 11
    # cells (unset INV_CELLS_FILTER); see the INV_K_LIST comment above for
    # why 16 is ordered first, and the INV_CELLS_FILTER comment above for why
    # the filter itself is needed under FIFO + 4-worker parallelism.
    if [[ " $STAGES " == *" invcells "* ]]; then
        RUN_GROUP="${RUN_GROUP_INV:-inverse-cells-d0}"
        for k in "${INV_K_LIST[@]}"; do
            for spec in "${COHORTS[@]}"; do
                IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                [[ "$tag" == "tf_noz_iter" ]] || continue
                for s in "${INV_SEEDS[@]}"; do
                    inv_cell_allowed "$k" "$s" || continue
                    CELL_ID="arm=invcells,block=tf,z=0,it=1,n=10,k=${k},s=${s}"
                    emit_job "${prefix}pp_inv_tf_noz_iter_k${k}_s${s}" "$arch" "$k" "$s" \
                        "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                        "arch.L_layers=2 arch.halt_max_steps=1" \
                        "epochs=100000"
                done
            done
        done
    fi
    # -- 2e) SP-04-L04 seed-extension grid (GATE-EXP013-LAUNCH narrowed alt) -
    # Same tf_noz_iter / D0 / epochs=100000 protocol as invcells (2d, above),
    # but at the 7 k values where seed=1 ALREADY FITS (the baseline/fig1 k's
    # 3,4,5,6,7,8,10 -- distinct from invcells' k in {9,12,13,16}), seeds 2,3
    # only. seed=1 is not in SEEDEXT_SEEDS at all (excluded by the array
    # itself, not filtered post hoc): the array IS the grid definition, per
    # the same EXP-013 lesson INV_SEEDS above already documents -- a shared
    # or over-broad seed array silently redefines what a downstream reader
    # thinks the campaign covers.
    #
    # run_name is DELIBERATELY `pp_seedext_...`, not the `fig1_...` legacy
    # pattern the k-sweep stages above use: fig1_* run_names read as
    # continuations of the SEALED Sigma_k_new generation (see the
    # "2026-08-06: default project moved" comment above WANDB_PROJECT), and
    # this grid is neither that generation nor its protocol (D0 +
    # epochs=100000 vs D1 + epochs=50000).
    #
    # RUN_GROUP is a dedicated "seedext-d0" (distinct from inverse-cells-d0
    # and every other group above).
    if [[ " $STAGES " == *" seedext "* ]]; then
        RUN_GROUP="${RUN_GROUP_SEEDEXT:-seedext-d0}"
        for k in "${SEEDEXT_K_LIST[@]}"; do
            for spec in "${COHORTS[@]}"; do
                IFS='|' read -r tag arch mlp_t Hc Lc <<< "$spec"
                [[ "$tag" == "tf_noz_iter" ]] || continue
                for s in "${SEEDEXT_SEEDS[@]}"; do
                    CELL_ID="arm=seedext,block=tf,z=0,it=1,n=10,k=${k},s=${s}"
                    emit_job "${prefix}pp_seedext_tf_noz_iter_k${k}_s${s}" "$arch" "$k" "$s" \
                        "arch.mlp_t=${mlp_t} arch.H_cycles=${Hc} arch.L_cycles=${Lc}" \
                        "arch.L_layers=2 arch.halt_max_steps=1" \
                        "epochs=100000"
                done
            done
        done
    fi
    # -- 2f) EXP-018 stages (see the e018 declaration block above) ---------
    # Stage-wise: one token per invocation, `df` gate first (real runs).
    if [[ " $STAGES " == *" e018_g1 "* ]]; then
        e018_gate g1 114; e018_emit_arms E018_G1_ARMS
    fi
    if [[ " $STAGES " == *" e018_g3 "* ]]; then
        e018_gate g3 18; e018_emit_arms E018_G3_ARMS
    fi
    if [[ " $STAGES " == *" e018_g4 "* ]]; then
        e018_gate g4 54; e018_emit_arms E018_G4_ARMS
    fi
    if [[ " $STAGES " == *" e018_g2 "* ]]; then
        e018_gate g2 78; e018_emit_arms E018_G2_ARMS
    fi
    if [[ " $STAGES " == *" e018_lb "* ]]; then
        if (( ! DRY_RUN )) && [[ "${E018_ALLOW_BLOCKED:-0}" != "1" ]]; then
            echo "ERROR: e018_lb = LB2 (zH_freeze) runs, held: requester B2 (stop-gradient) is a no-op here" >&2
            echo "       (EXP-018_FREEZE_WITNESS.md W4). Dry-run only unless E018_ALLOW_BLOCKED=1." >&2
            exit 1
        fi
        e018_gate lb 6; e018_emit_arms E018_LB_ARMS
    fi
    # -- 2g) EXP-020 stages (see the e020 declaration block above) ---------
    # epochs=<n> is an emit_job extra arg, so it lands AFTER common_args' epochs=50000 and wins
    # under Hydra last-key-wins (same pattern as invcells: job 0418 body has both, all_config 100000).
    if [[ " $STAGES " == *" e020_s1 "* || " $STAGES " == *" e020_s2 "* ]]; then
        local _e020_root="$DATA_ROOT" _e020_proj="$WANDB_PROJECT" _e020_grp="$RUN_GROUP"
        DATA_ROOT="$E020_DATA_ROOT"; WANDB_PROJECT="$E020_PROJECT"
        if [[ " $STAGES " == *" e020_s1 "* ]]; then
            RUN_GROUP="e020_s1"
            for k in "${E020_S1_K[@]}"; do
                for s in "${E020_S1_SEEDS[@]}"; do
                    CELL_ID="arm=fig1,block=tf,z=0,it=1,n=10,k=${k},s=${s},epochs=${E020_S1_EPOCHS}"
                    emit_job "${prefix}fig1_tf_noz_iter_ep76k_k${k}_s${s}" trm_singlez "$k" "$s" \
                        "${E020_NOZ}" "arch.L_layers=2 arch.halt_max_steps=1" \
                        "epochs=${E020_S1_EPOCHS}"
                    CELL_ID="arm=act,cohort=noz_iter,halt=16,n=10,k=${k},s=${s},epochs=${E020_S1_EPOCHS}"
                    emit_job "${prefix}act_noz_iter_h16_ep76k_k${k}_s${s}" trm_singlez "$k" "$s" \
                        "${E020_NOZ} arch.L_layers=2 arch.halt_exploration_prob=0.1" \
                        "arch.halt_max_steps=16" "epochs=${E020_S1_EPOCHS}"
                done
            done
        fi
        if [[ " $STAGES " == *" e020_s2 "* ]]; then
            RUN_GROUP="e020_s2"
            for s in "${E020_S2_SEEDS[@]}"; do
                for k in "${E020_S2_K[@]}"; do
                    CELL_ID="arm=act,cohort=ltf_fullgrad,halt=16,n=10,k=${k},s=${s},epochs=50000"
                    emit_job "${prefix}act_ltf_fullgrad_h16_ep50k_k${k}_s${s}" transformers_baseline "$k" "$s" \
                        "${E020_LTF}" "arch.halt_max_steps=16"
                done
            done
        fi
        DATA_ROOT="$_e020_root"; WANDB_PROJECT="$_e020_proj"; RUN_GROUP="$_e020_grp"
    fi
}

# =================  machinery below, no need to edit  ====================

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Next sequence number across queued/running/done/failed, so appended jobs
# keep FIFO order even after earlier ones complete.
seq_next() {
    local max=0 f n
    for f in "$JOBS_DIR"/*.job "$QUEUE_DIR"/processing/*.job.gpu* \
             "$QUEUE_DIR"/done/*.job "$QUEUE_DIR"/failed/*.job; do
        [[ -e "$f" ]] || continue
        n="$(basename "$f")"
        n="${n%%_*}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( 10#$n > max )) && max=$(( 10#$n ))
    done
    echo $(( max + 1 ))
}

# enqueue <name>  — job body comes from stdin (heredoc)
# Idempotent: a cell whose run_name already exists anywhere in the queue
# lifecycle (queued / running / done / failed) is skipped, so re-running the
# script after edits only adds the missing cells.
enqueue() {
    if compgen -G "$JOBS_DIR/*_$1.job" >/dev/null \
       || compgen -G "$QUEUE_DIR/processing/*_$1.job.gpu*" >/dev/null \
       || compgen -G "$QUEUE_DIR/done/*_$1.job" >/dev/null \
       || compgen -G "$QUEUE_DIR/failed/*_$1.job" >/dev/null; then
        echo "skip (already in queue lifecycle): $1"
        cat > /dev/null
        return
    fi
    if (( DRY_RUN )); then
        printf '%04d %s\n' "$SEQ" "$1"
        if (( DRY_RUN_SHOW_BODY )); then
            while IFS= read -r _line; do printf '    | %s\n' "$_line"; done
        else
            cat > /dev/null
        fi
    else
        local file
        file="$(printf '%s/%04d_%s.job' "$JOBS_DIR" "$SEQ" "$1")"
        cat > "$file"
        echo "enqueued: $file"
    fi
    SEQ=$(( SEQ + 1 ))
}

# enqueue_grid <name_prefix> <tag> <base_cmd> <data_path> <sweep_array_name>
# Parses the sweep spec into the GRID_* globals, then recursively enqueues
# the full cartesian product.
enqueue_grid() {
    GRID_NAME_PREFIX="$1" GRID_TAG="$2" GRID_BASE="$3" GRID_DATA="$4"
    local -n spec_ref="$5"

    GRID_ALIASES=() GRID_PARAMS=() GRID_VALUES=()
    local line rest
    for line in "${spec_ref[@]}"; do
        GRID_ALIASES+=("$(trim "${line%%=*}")")
        rest="${line#*=}"
        GRID_PARAMS+=("$(trim "${rest%%:*}")")
        GRID_VALUES+=("$(trim "${rest#*:}")")
    done

    emit_grid 0 "" ""
}

# emit_grid <depth> <name_acc> <args_acc>  — one recursion level per param
emit_grid() {
    local depth="$1" name_acc="$2" args_acc="$3"

    if (( depth == ${#GRID_PARAMS[@]} )); then
        local run_name="${GRID_NAME_PREFIX}_${GRID_TAG}${name_acc}"
        enqueue "$run_name" <<EOF
$GRID_BASE \\
    evaluators="[]" \\
    data_paths="[${GRID_DATA}]" \\
    ${args_acc# } \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${run_name}" \\
    ema=True
EOF
        return
    fi

    local vals v suffix
    read -r -a vals <<< "${GRID_VALUES[$depth]}"
    for v in "${vals[@]}"; do
        suffix=""
        (( ${#vals[@]} > 1 )) && suffix="_${GRID_ALIASES[$depth]}${v}"
        emit_grid "$(( depth + 1 ))" \
            "${name_acc}${suffix}" \
            "${args_acc} ${GRID_PARAMS[$depth]}=${v}"
    done
}

mkdir -p "$JOBS_DIR"
SEQ="$(seq_next)"
SEQ_START="$SEQ"

main

echo
echo "jobs: $(( SEQ - SEQ_START ))$( (( DRY_RUN )) && echo ' (dry run, nothing written)' )"
echo "now run:  scripts/queue_run.sh        (GPUS=\"4 5 6 7\" by default)"
echo "status:   scripts/queue_run.sh status"
