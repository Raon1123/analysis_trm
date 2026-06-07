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
#
# Re-running appends after existing jobs (sequence numbers continue), so you
# can enqueue more sweeps while the runner is going.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; shift; }

QUEUE_DIR="${QUEUE_DIR:-scripts/queue}"
JOBS_DIR="$QUEUE_DIR/jobs"

RUN_PREFIX="${1:-}"
prefix=""
[[ -n "$RUN_PREFIX" ]] && prefix="${RUN_PREFIX}_"

# ====================  EDIT BELOW: sweep definitions  ====================

WANDB_PROJECT="Sigma_k"
K_LIST=(6 7 8 10 12 16 20)

common_args="epochs=100000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0"

# "alias=hydra.param: candidates..."  — add/remove candidates to change the grid.
TRM_SWEEP=(
    "halt=arch.halt_max_steps: 1 8 16"
    "Llay=arch.L_layers: 2"
    "H=arch.H_cycles: 3 6"
    "L=arch.L_cycles: 3 6"
)

TRANSFORMER_SWEEP=(
    "lay=arch.H_layers: 1 2 6"
    "cyc=arch.H_cycles: 1 6"
)

main() {
    local k
    for k in "${K_LIST[@]}"; do
        # enqueue_grid <name_prefix> <tag> <base_cmd> <data_path> <sweep_array_name>
        enqueue_grid "${prefix}k${k}" "trm" \
            "uv run pretrain.py arch=trm $common_args" \
            "data/sigma_k_10/${k}" TRM_SWEEP

        # NOTE: the old script used data/sigma_k/ (not _10) for the deep
        # transformer baselines — unify or change here as needed.
        enqueue_grid "${prefix}k${k}" "tf" \
            "uv run pretrain.py arch=transformers_baseline $common_args" \
            "data/sigma_k_10/${k}" TRANSFORMER_SWEEP
    done
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
enqueue() {
    if (( DRY_RUN )); then
        printf '%04d %s\n' "$SEQ" "$1"
        cat > /dev/null
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
