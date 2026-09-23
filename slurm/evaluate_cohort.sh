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

RUN_ROOT=/nobackup/proj/disk/muc/personal/moe/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59
REPO=/nobackup/proj/disk/muc/personal/moe/gigapath
INPUT=${1:?Usage: evaluate_cohort.sh <input-batch-dir> <output-results-dir>}
OUTPUT=${2:?Usage: evaluate_cohort.sh <input-batch-dir> <output-results-dir>}

# HF cache under project storage per AGENTS.md
export HF_HOME=${HF_HOME:-/nobackup/proj/disk/muc/personal/$USER/.cache/huggingface}
export TMPDIR=/scratch/local/$SLURM_JOB_ID/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO
mkdir -p "$TMPDIR" "$OUTPUT" "$HF_HOME"
cd "$REPO"

pixi run check-gpu
pixi run python scripts/evaluate_wsi.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --tile-checkpoint "$RUN_ROOT/models/pytorch_model.bin" \
  --slide-checkpoint "$RUN_ROOT/models/slide_encoder.pth" \
  --scratch "$TMPDIR/gigapath" \
  --level 1 \
  --source-mpp 0.243093922651934 \
  --target-mpp 0.5 \
  --batch-size 128 \
  --model-revision 685a3c816fb7bfd4fec697d7f1ed7da57f2e8e86
