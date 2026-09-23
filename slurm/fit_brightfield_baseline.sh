#!/bin/bash
#SBATCH -A naiss2026-4-1499-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J gigapath-clf
#SBATCH -o /nobackup/proj/disk/muc/personal/moe/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59/classifier-%j.out

set -euo pipefail

RUN_ROOT=/nobackup/proj/disk/muc/personal/moe/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59
REPO=/nobackup/proj/disk/muc/personal/moe/gigapath
OUTPUT=$RUN_ROOT/baseline-classifiers

export PYTHONPATH=$REPO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT"
cd "$REPO"

pixi run python scripts/fit_brightfield_baseline.py \
  --results-csv "$RUN_ROOT/results/results.csv" \
  --output "$OUTPUT" \
  --block-size 4096 \
  --folds 5 \
  --seed 42
