#!/bin/bash
#SBATCH -A naiss2026-4-1499-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 72
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH -J gigapath-cohort
#SBATCH -o %x-%j.out

set -euo pipefail

INPUT=${1:?Usage: evaluate_cohort.sh <input-batch-dir> <output-results-dir>}
OUTPUT=${2:?Usage: evaluate_cohort.sh <input-batch-dir> <output-results-dir>}

# HF cache under project storage per AGENTS.md
export HF_HOME=${HF_HOME:-/nobackup/proj/disk/muc/personal/$USER/.cache/huggingface}
export TMPDIR=/scratch/local/$SLURM_JOB_ID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

RUN_ROOT=/nobackup/proj/disk/muc/personal/$USER/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59
REPO=/nobackup/proj/disk/muc/personal/$USER/gigapath

mkdir -p "$TMPDIR" "$OUTPUT" "$HF_HOME"
cd "$REPO"

pixi run python scripts/evaluate_wsi.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --tile-checkpoint "$RUN_ROOT/models/pytorch_model.bin" \
  --slide-checkpoint "$RUN_ROOT/models/slide_encoder.pth" \
  --scratch "$TMPDIR/tiles" \
  --level 1 \
  --source-mpp 0.243093922651934 \
  --target-mpp 0.5 \
  --batch-size 128
