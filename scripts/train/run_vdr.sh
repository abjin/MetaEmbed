#!/bin/bash
# Launch a VDR training run (see scripts/VDR_EXPERIMENT.md).
#
# Usage: scripts/train/run_vdr.sh <config.yaml> [log_dir]
#   GPUS          comma-separated GPU ids            (default 3,4,5,6,7)
#   FSDP_CONFIG   accelerate config; num_processes must equal the GPU count
#                                                   (default fsdp_5gpu.yaml)
#   DATA_DIR      parent of MoCa_train_with_image/, trailing slash required
#                                                   (default ../data_dir/)
#
# The run is detached (setsid + nohup) so it survives the launching shell.
# Logs: <log_dir>/<config name>.log and <config name>_gpu.log (nvidia-smi every 30 s).
set -euo pipefail

CONFIG=${1:?usage: run_vdr.sh <config.yaml> [log_dir]}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOG_DIR=$(realpath -m "${2:-$REPO/../runs}")
GPUS=${GPUS:-3,4,5,6,7}
FSDP_CONFIG=${FSDP_CONFIG:-fsdp_5gpu.yaml}
DATA_DIR=${DATA_DIR:-$(realpath -m "$REPO/../data_dir")/}
NAME=$(basename "$CONFIG" .yaml)
mkdir -p "$LOG_DIR"

cd "$REPO"
setsid nohup bash -c '
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    -i "$0" -l 30 > "$3/$4_gpu.log" 2>&1 &
  SMI_PID=$!
  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="$0" DATA_DIR="$5" \
    TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/accelerate launch --config_file "$1" \
    scripts/train/train_colbert.py "$2" > "$3/$4.log" 2>&1
  echo "EXIT $?" >> "$3/$4.log"
  kill "$SMI_PID"
' "$GPUS" "$FSDP_CONFIG" "$CONFIG" "$LOG_DIR" "$NAME" "$DATA_DIR" > /dev/null 2>&1 < /dev/null &

echo "launched $CONFIG on GPUs $GPUS -> $LOG_DIR/$NAME.log"
