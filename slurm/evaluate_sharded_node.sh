#!/bin/bash
#SBATCH -A naiss2026-4-1499-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH -t 02:00:00
#SBATCH -J gigapath-node
#SBATCH -o /nobackup/proj/disk/muc/personal/%u/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59/logs/%x-%A_%a.out

set -uo pipefail

if [[ $# -ne 8 ]]; then
    echo "Usage: $0 SHARDS SIF SIF_SHA TILE_CKPT TILE_SHA SLIDE_CKPT SLIDE_SHA OUTPUT" >&2
    exit 2
fi

SHARDS=$1
SIF=$2
SIF_SHA=$3
TILE_CKPT=$4
TILE_SHA=$5
SLIDE_CKPT=$6
SLIDE_SHA=$7
OUTPUT=$8
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
JOB_ID=${SLURM_JOB_ID:-manual}
NODE=${SLURMD_NODENAME:-$(hostname)}
SCRATCH=/scratch/local/${JOB_ID}
COMMON=$SCRATCH/common
ROWS=$SCRATCH/rows.tsv
WORKER=$SCRATCH/run-worker.sh

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

for path in "$SHARDS" "$SIF" "$TILE_CKPT" "$SLIDE_CKPT"; do
    [[ -f $path ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
mkdir -p "$COMMON" "$OUTPUT/.incoming" "$OUTPUT/.failed"

python3 - "$SHARDS" "$TASK_ID" "$ROWS" <<'PY' || exit 1
import csv
import sys
from pathlib import Path

source, task, target = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
with source.open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle, delimiter="\t") if int(row["node_shard"]) == task]
if not 1 <= len(rows) <= 4:
    raise SystemExit(f"node_shard {task} has {len(rows)} rows; expected 1-4")
slots = [int(row["gpu_slot"]) for row in rows]
if len(set(slots)) != len(slots) or any(slot not in range(4) for slot in slots):
    raise SystemExit(f"invalid GPU slots for node_shard {task}: {slots}")
with target.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["node_shard", "gpu_slot", "slide_id", "source_path", "weight"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY

cp "$SIF" "$COMMON/gigapath.sif" || exit 1
cp "$TILE_CKPT" "$COMMON/pytorch_model.bin" || exit 1
cp "$SLIDE_CKPT" "$COMMON/slide_encoder.pth" || exit 1

verify_hash() {
    local actual
    actual=$(sha256sum "$1" | cut -d' ' -f1)
    [[ $actual == "$2" ]] || { echo "SHA-256 mismatch for $1: $actual != $2" >&2; return 1; }
}
verify_hash "$COMMON/gigapath.sif" "$SIF_SHA" || exit 1
verify_hash "$COMMON/pytorch_model.bin" "$TILE_SHA" || exit 1
verify_hash "$COMMON/slide_encoder.pth" "$SLIDE_SHA" || exit 1

python3 - "$ROWS" "$COMMON" "$SCRATCH" <<'PY' || exit 1
import csv
import shutil
import sys
from pathlib import Path

rows, common, scratch = map(Path, sys.argv[1:])
with rows.open(newline="") as handle:
    sources = [Path(row["source_path"]) for row in csv.DictReader(handle, delimiter="\t")]
common_bytes = sum(path.stat().st_size for path in common.iterdir())
# One source copy plus generated tiles; require 20% free headroom.
required = int((common_bytes + 2 * sum(path.stat().st_size for path in sources if path.is_file())) * 1.2)
free = shutil.disk_usage(scratch).free
if free < required:
    raise SystemExit(f"Insufficient scratch: need {required} bytes with headroom, have {free}")
print(f"scratch preflight: required={required} free={free}")
PY
cat > "$WORKER" <<'WORKER'
#!/bin/bash
set -uo pipefail
slot=$1
source_path=$2
output=$3
scratch=$4
log=$5
common=$6
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm did not assign a GPU}"
exec apptainer exec --nv --cleanenv \
    --bind "$source_path:$source_path:ro" \
    --bind "$common:$common:ro" \
    --bind "$scratch:$scratch:rw" \
    --bind "$output:$output:rw" \
    "$common/gigapath.sif" \
    python /opt/gigapath/scripts/evaluate_wsi.py \
    --input "$source_path" \
    --output "$output" \
    --tile-checkpoint "$common/pytorch_model.bin" \
    --slide-checkpoint "$common/slide_encoder.pth" \
    --scratch "$scratch" \
    --level 1 \
    --source-mpp 0.243093922651934 \
    --target-mpp 0.5 \
    --batch-size 128 \
    --model-revision 685a3c816fb7bfd4fec697d7f1ed7da57f2e8e86 \
    >"$log" 2>&1
WORKER
chmod +x "$WORKER"

declare -A PIDS SLIDES SOURCES OUTDIRS LOGS
while IFS=$'\t' read -r node slot slide source weight; do
    [[ $node == node_shard ]] && continue
    if [[ ! -f $source ]]; then
        mkdir -p "$OUTPUT/.failed/${slide}-${ARRAY_JOB_ID}-${TASK_ID}"
        python3 - "$OUTPUT/.failed/${slide}-${ARRAY_JOB_ID}-${TASK_ID}/failure.json" "$slide" "$source" "$slot" "$NODE" "$ARRAY_JOB_ID" "$TASK_ID" <<'PY'
import json, sys
from pathlib import Path
path, slide, source, slot, node, job, task = sys.argv[1:]
Path(path).write_text(json.dumps({"slide_id": slide, "source_path": source, "gpu_slot": int(slot), "node": node, "job_id": job, "task_id": task, "exit_code": 1, "error": "Source WSI file is missing"}, indent=2) + "\n")
PY
        continue
    fi
    slot_scratch=$SCRATCH/gpu-$slot
    slot_output=$SCRATCH/output-$slot
    slot_log=$SCRATCH/worker-$slot.log
    mkdir -p "$slot_scratch" "$slot_output"
    srun --exclusive -N1 -n1 --gpus-per-task=1 --gpu-bind=single:1 -c72 --mem=105G \
        "$WORKER" "$slot" "$source" "$slot_output" "$slot_scratch" "$slot_log" "$COMMON" &
    PIDS[$slot]=$!
    SLIDES[$slot]=$slide
    SOURCES[$slot]=$source
    OUTDIRS[$slot]=$slot_output
    LOGS[$slot]=$slot_log
done < "$ROWS"

rc=0
declare -A EXITS
for slot in "${!PIDS[@]}"; do
    if wait "${PIDS[$slot]}"; then EXITS[$slot]=0; else EXITS[$slot]=$?; fi
done

record_failure() {
    local slide=$1 source=$2 slot=$3 exit_code=$4 error=$5
    local dir=$OUTPUT/.failed/${slide}-${ARRAY_JOB_ID}-${TASK_ID}
    mkdir -p "$dir"
    python3 - "$dir/failure.json" "$slide" "$source" "$slot" "$NODE" "$ARRAY_JOB_ID" "$TASK_ID" "$exit_code" "$error" <<'PY'
import json, sys
from pathlib import Path
path, slide, source, slot, node, job, task, exit_code, error = sys.argv[1:]
Path(path).write_text(json.dumps({"slide_id": slide, "source_path": source, "gpu_slot": int(slot), "node": node, "job_id": job, "task_id": task, "exit_code": int(exit_code), "error": error}, indent=2) + "\n")
PY
}

verify_output() {
    local dir=$1 slide=$2 source=$3
    apptainer exec --cleanenv --bind "$dir:$dir:ro" "$COMMON/gigapath.sif" \
        python -c 'import sys; from pathlib import Path; from scripts.merge_parallel_results import verify_slide_output; ok, error, _ = verify_slide_output(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]); print(error, file=sys.stderr) if not ok else None; raise SystemExit(0 if ok else 1)' \
        "$dir" "$slide" "$source" "$TILE_SHA" "$SLIDE_SHA"
}

for slot in "${!PIDS[@]}"; do
    slide=${SLIDES[$slot]}
    source=${SOURCES[$slot]}
    step_rc=${EXITS[$slot]}
    generated=${OUTDIRS[$slot]}/$slide
    final=$OUTPUT/$slide
    incoming=$OUTPUT/.incoming/${slide}-${ARRAY_JOB_ID}-${TASK_ID}

    if [[ -d $final ]]; then
        if verify_output "$final" "$slide" "$source"; then
            continue
        fi
        record_failure "$slide" "$source" "$slot" "$step_rc" "Existing durable output is invalid"
        rc=1
        continue
    fi
    if (( step_rc != 0 )); then
        record_failure "$slide" "$source" "$slot" "$step_rc" "Inference worker exited nonzero; see array log and retained failure record"
        rc=1
        continue
    fi
    mkdir -p "$generated/logs"
    if ! cp "${LOGS[$slot]}" "$generated/logs/evaluation.log"; then
        record_failure "$slide" "$source" "$slot" 1 "Could not retain worker evaluation log"
        rc=1
        continue
    fi
    if ! verify_output "$generated" "$slide" "$source"; then
        record_failure "$slide" "$source" "$slot" 1 "Generated output failed durable contract verification"
        rc=1
        continue
    fi
    rm -rf "$incoming"
    if ! cp -a "$generated" "$incoming" || ! verify_output "$incoming" "$slide" "$source"; then
        record_failure "$slide" "$source" "$slot" 1 "Could not copy and verify durable incoming output"
        rm -rf "$incoming"
        rc=1
        continue
    fi
    if ! mv "$incoming" "$final"; then
        record_failure "$slide" "$source" "$slot" 1 "Could not atomically publish durable output"
        rm -rf "$incoming"
        rc=1
    fi
done

# Missing-source rows never entered PIDS.
expected=$(($(wc -l < "$ROWS") - 1))
(( ${#PIDS[@]} == expected )) || rc=1
exit "$rc"
