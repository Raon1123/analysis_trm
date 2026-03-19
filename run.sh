#!/bin/bash

echo "Starting distributed training with torchrun..."
echo ${prefix}

torchrun_cmd="torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1"

OMP_NUM_THREADS=4 uv run ${torchrun_cmd} \
pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
global_batch_size=256 \
arch.halt_max_steps=8 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+project_name="reasoning_nanoGPT" \
+run_name=${1} ema=True