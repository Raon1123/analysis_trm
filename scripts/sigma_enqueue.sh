#!/usr/bin/env bash
# Enqueue the sigma_k experiment grid as *.job files for scripts/queue_run.sh.
# (Job definitions ported from scripts/sigma_refactor.sh.)
#
# Usage:
#   scripts/sigma_enqueue.sh [run_prefix]    # then: scripts/queue_run.sh
#
# Re-running appends after existing jobs (sequence numbers continue), so you
# can enqueue more k values / variants while the runner is already going.
# To change the grid: edit K_LIST or the enqueue blocks below — each block is
# one job, written verbatim into its .job file.

set -euo pipefail

QUEUE_DIR="${QUEUE_DIR:-scripts/queue}"
JOBS_DIR="$QUEUE_DIR/jobs"
mkdir -p "$JOBS_DIR"

WANDB_PROJECT="Sigma_k"
RUN_PREFIX="${1:-}"
K_LIST=(6 7 8 10 12 16 20)

prefix=""
[[ -n "$RUN_PREFIX" ]] && prefix="${RUN_PREFIX}_"

uv_trm="uv run pretrain.py arch=trm epochs=100000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0"
uv_transformer="uv run pretrain.py arch=transformers_baseline epochs=100000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0"

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
SEQ="$(seq_next)"

# enqueue <name>  — job body comes from stdin (heredoc)
enqueue() {
    local file
    file="$(printf '%s/%04d_%s.job' "$JOBS_DIR" "$SEQ" "$1")"
    cat > "$file"
    echo "enqueued: $file"
    SEQ=$(( SEQ + 1 ))
}

for k in "${K_LIST[@]}"; do
    data_path="[data/sigma_k_10/${k}]"

    enqueue "${prefix}k${k}_transformer_1layer_1cycle" <<EOF
$uv_transformer \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.H_layers=1 \\
    arch.H_cycles=1 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_transformer_1layer_1cycle" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_baseline" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=16 \\
    arch.L_layers=2 \\
    arch.H_cycles=3 \\
    arch.L_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_baseline" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_H6L3" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=16 \\
    arch.L_layers=2 \\
    arch.H_cycles=6 \\
    arch.L_cycles=3 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_H6L3" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_halt_one" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=1 \\
    arch.L_layers=2 \\
    arch.H_cycles=3 \\
    arch.L_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_halt_one" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_halt_half" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=8 \\
    arch.L_layers=2 \\
    arch.H_cycles=3 \\
    arch.L_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_halt_half" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_ablation_L_cycles" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=16 \\
    arch.L_layers=2 \\
    arch.H_cycles=3 \\
    arch.L_cycles=1 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_ablation_L_cycles" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_ablation_H_cycles" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=16 \\
    arch.L_layers=2 \\
    arch.H_cycles=1 \\
    arch.L_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_ablation_H_cycles" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_half_layer" <<EOF
$uv_trm \\
    evaluators="[]" \\
    data_paths="${data_path}" \\
    arch.halt_max_steps=16 \\
    arch.L_layers=1 \\
    arch.H_cycles=3 \\
    arch.L_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_half_layer" \\
    ema=True
EOF

    # --- transformer baselines (NOTE: sigma_refactor.sh used data/sigma_k/,
    #     not data/sigma_k_10/, for these three — preserved as-is) ---
    baseline_data_path="[data/sigma_k/${k}]"

    enqueue "${prefix}k${k}_transformer_6layer" <<EOF
$uv_transformer \\
    evaluators="[]" \\
    data_paths="${baseline_data_path}" \\
    arch.H_layers=6 \\
    arch.H_cycles=1 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_transformer_6layer" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_transformer_1layer_half_cycle" <<EOF
$uv_transformer \\
    evaluators="[]" \\
    data_paths="${baseline_data_path}" \\
    arch.H_layers=1 \\
    arch.H_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_transformer_1layer_half_cycle" \\
    ema=True
EOF

    enqueue "${prefix}k${k}_transformer_2layer_half_cycle" <<EOF
$uv_transformer \\
    evaluators="[]" \\
    data_paths="${baseline_data_path}" \\
    arch.H_layers=2 \\
    arch.H_cycles=6 \\
    +project_name="${WANDB_PROJECT}" \\
    +run_name="${prefix}k${k}_transformer_2layer_half_cycle" \\
    ema=True
EOF
done

echo
echo "now run:  scripts/queue_run.sh        (GPUS=\"4 5 6 7\" by default)"
echo "status:   scripts/queue_run.sh status"
