#!/bin/bash
# Example: Run the full MacaBrainNet pipeline on example data
#
# Prerequisites:
#   1. Install dependencies (see README.md)
#   2. Download model checkpoints: python download_from_hf.py
#
# Usage:
#   bash src/run_example.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

EXAMPLE_IMG="${SCRIPT_DIR}/example/sub-01_T1w.nii.gz"
OUT_DIR="${PROJECT_DIR}/example_output"

echo "============================================"
echo "MacaBrainNet Pipeline Example"
echo "============================================"
echo "Input:  ${EXAMPLE_IMG}"
echo "Output: ${OUT_DIR}"
echo ""

cd "${PROJECT_DIR}"

python pipeline.py \
    --img "${EXAMPLE_IMG}" \
    --out-dir "${OUT_DIR}" \
    --padding 16 \
    --overlap 0.60 \
    --device cuda

echo ""
echo "Example complete! Results in: ${OUT_DIR}"
echo "  - *_brain_mask.nii.gz      # skull stripping result"
echo "  - *_cropped.nii.gz         # cropped + brain-masked image"
echo "  - *_tissue_seg_cropped.nii.gz  # tissue seg (cropped space)"
echo "  - *_tissue_seg.nii.gz      # tissue seg (original space)"
