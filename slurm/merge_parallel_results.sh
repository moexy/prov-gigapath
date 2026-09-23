#!/bin/bash
#SBATCH -A naiss2026-4-1499-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J gigapath-merge
#SBATCH -o /nobackup/proj/disk/muc/personal/%u/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59/logs/%x-%j.out

set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "Usage: $0 SHARDS OUTPUT ARRAY_JOB_ID SIF TILE_CKPT SLIDE_CKPT REQUESTED_NODES REQUESTED_GPUS" >&2
    exit 2
fi

SHARDS=$1
OUTPUT=$2
ARRAY_JOB_ID=$3
SIF=$4
TILE_CKPT=$5
SLIDE_CKPT=$6
REQUESTED_NODES=$7
REQUESTED_GPUS=$8
REPO_DIR="${REPO_DIR:-/nobackup/proj/disk/muc/personal/${USER}/gigapath}"

cd "$REPO_DIR"
pixi run python scripts/merge_parallel_results.py \
    --shards "$SHARDS" \
    --output "$OUTPUT" \
    --array-job-id "$ARRAY_JOB_ID" \
    --sif "$SIF" \
    --tile-checkpoint "$TILE_CKPT" \
    --slide-checkpoint "$SLIDE_CKPT" \
    --requested-nodes "$REQUESTED_NODES" \
    --requested-gpus "$REQUESTED_GPUS"
