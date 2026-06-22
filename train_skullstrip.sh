#!/bin/bash
# Skull Stripping Training Launch Script
# 0.5mm resolution, patch_size=96x96x96, DiceFocalLoss

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/weiyahui/software/miniconda3/envs/macabrain/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/weiyahui/software/miniconda3/envs/macabrain/bin/torchrun}"
TRAIN_SCRIPT="$(dirname "$0")/train_skullstrip.py"

# GPU config
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

# CPU threads
TOTAL_CPUS=$(nproc 2>/dev/null || echo 32)
OMP_NUM_THREADS=$(( TOTAL_CPUS / NPROC_PER_NODE / 2 ))
if [ "${OMP_NUM_THREADS}" -lt 1 ]; then OMP_NUM_THREADS=1; fi
export OMP_NUM_THREADS
export CUDA_VISIBLE_DEVICES

# Default paths — override via environment
DATA_ROOT="${DATA_ROOT:-/home/weiyahui/projects/monkey/dataset/skull_stripping}"
JSON_PATH="${JSON_PATH:-${DATA_ROOT}/cross_val_fold_${FOLD:-1}.json}"
PRETRAINED="${PRETRAINED:-/home/weiyahui/projects/monkey/macaBrainNet/pretrain/LocalGlobal_B_step_17000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/weiyahui/projects/monkey/macaBrainNet_v2/swinunetr_models/skull_stripping/fold_${FOLD:-1}}"

echo "========================================"
echo "Skull Stripping Training"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "JSON=${JSON_PATH}"
echo "PRETRAINED=${PRETRAINED}"
echo "OUTPUT=${OUTPUT_DIR}"
echo "========================================"

"${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_SCRIPT}" \
    --pretrained "${PRETRAINED}" \
    --json-path "${JSON_PATH}" \
    --data-root "${DATA_ROOT}" \
    --num-samples 12 \
    --accumulation-steps 4 \
    --num-workers 32 \
    --output-dir "${OUTPUT_DIR}" \
    --resume
