#!/bin/bash
#SBATCH -A naiss2026-4-1499-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J gigapath-build
#SBATCH -o /nobackup/proj/disk/muc/personal/%u/gigapath-containers/build-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/nobackup/proj/disk/muc/personal/${USER}/gigapath}"
PROJECT_CONTAINERS_DIR="${PROJECT_CONTAINERS_DIR:-/nobackup/proj/disk/muc/personal/${USER}/gigapath-containers}"
CONTAINER_DEF="${CONTAINER_DEF:-${REPO_DIR}/containers/gigapath-flash.def}"
LOCK_FILE="${LOCK_FILE:-${REPO_DIR}/pixi.lock}"
RUN_ROOT="${RUN_ROOT:-/nobackup/proj/disk/muc/personal/${USER}/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59}"
TILE_CHECKPOINT="${TILE_CHECKPOINT:-${RUN_ROOT}/models/pytorch_model.bin}"
SLIDE_CHECKPOINT="${SLIDE_CHECKPOINT:-${RUN_ROOT}/models/slide_encoder.pth}"

cd "${REPO_DIR}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/local/${SLURM_JOB_ID:-$$}/apptainer}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_CONTAINERS_DIR}/.apptainer-cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
mkdir -p "${PROJECT_CONTAINERS_DIR}"

if [[ ! -f "${CONTAINER_DEF}" ]]; then
    echo "ERROR: Container definition not found: ${CONTAINER_DEF}" >&2
    exit 1
fi

if [[ ! -f "${LOCK_FILE}" ]]; then
    echo "ERROR: Pixi lock file not found: ${LOCK_FILE}" >&2
    exit 1
fi

ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" ]]; then
    echo "ERROR: Build host architecture must be aarch64, got: ${ARCH}" >&2
    exit 1
fi

DEF_SHA256=$(sha256sum "${CONTAINER_DEF}" | awk '{print $1}')
LOCK_SHA256=$(sha256sum "${LOCK_FILE}" | awk '{print $1}')
LOCK_PREFIX="${LOCK_SHA256:0:12}"
GIT_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")

FINAL_SIF="${PROJECT_CONTAINERS_DIR}/gigapath-flash-${LOCK_PREFIX}-aarch64.sif"
FINAL_MANIFEST="${PROJECT_CONTAINERS_DIR}/gigapath-flash-${LOCK_PREFIX}-aarch64.sif.manifest.json"
if [[ -e "${FINAL_SIF}" || -e "${FINAL_MANIFEST}" ]]; then
    echo "ERROR: Immutable container target already exists: ${FINAL_SIF}" >&2
    exit 1
fi

TMP_SIF="${PROJECT_CONTAINERS_DIR}/.tmp-gigapath-flash-${SLURM_JOB_ID:-$$}.sif"
TMP_MANIFEST="${PROJECT_CONTAINERS_DIR}/.tmp-gigapath-flash-${SLURM_JOB_ID:-$$}.sif.manifest.json"

# Clean up temp files if anything fails
trap 'rm -f "${TMP_SIF}" "${TMP_MANIFEST}"' EXIT

echo "Building Apptainer image to temporary file ${TMP_SIF}..."
if ! apptainer build --fakeroot "${TMP_SIF}" "${CONTAINER_DEF}"; then
    echo "fakeroot build failed; retrying the site-configured setuid builder" >&2
    rm -f "${TMP_SIF}"
    apptainer build "${TMP_SIF}" "${CONTAINER_DEF}"
fi

echo "Verifying image architecture and CUDA GH200 availability..."
APP_ARCH=$(apptainer exec "${TMP_SIF}" uname -m)
if [[ "${APP_ARCH}" != "aarch64" ]]; then
    echo "ERROR: Container internal architecture is not aarch64 (got ${APP_ARCH})" >&2
    exit 1
fi
for checkpoint in "${TILE_CHECKPOINT}" "${SLIDE_CHECKPOINT}"; do
    [[ -f "${checkpoint}" ]] || { echo "ERROR: Checkpoint not found: ${checkpoint}" >&2; exit 1; }
done
PROBE_JSON=$(apptainer exec --nv --bind "${RUN_ROOT}:${RUN_ROOT}:ro" "${TMP_SIF}" python - "${TILE_CHECKPOINT}" "${SLIDE_CHECKPOINT}" <<'PY'
import contextlib
import json
import subprocess
import sys
import torch
from gigapath import slide_encoder, tile_encoder

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false inside container")
device = torch.cuda.get_device_name(0)
if "GH200" not in device:
    raise SystemExit(f"Expected GH200 device, got {device}")
with contextlib.redirect_stdout(sys.stderr):
    tile = tile_encoder.create_model(sys.argv[1], "gigapath_tile_enc_dinov2s")
    slide = slide_encoder.create_model(sys.argv[2], "gigapath_slide_enc12l384d", 384, global_pool=True)
tile_state = torch.load(sys.argv[1], map_location="cpu")
slide_state = torch.load(sys.argv[2], map_location="cpu")["model"]
for name, model, state in (("tile", tile, tile_state), ("slide", slide, slide_state)):
    missing = sorted(set(model.state_dict()) - set(state))
    unexpected = sorted(set(state) - set(model.state_dict()))
    if missing or unexpected:
        raise SystemExit(f"{name} checkpoint mismatch: missing={missing}, unexpected={unexpected}")
tile.eval()
slide.eval()
print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "device_name": device,
    "pixi_version": subprocess.check_output(["pixi", "--version"], text=True).strip().split()[-1],
}))
PY
)

SIF_SHA256=$(sha256sum "${TMP_SIF}" | cut -d' ' -f1)
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
readarray -t PROBE_VALUES < <(python3 - "${PROBE_JSON}" <<'PY'
import json, sys
probe = json.loads(sys.argv[1])
for key in ("torch_version", "cuda_version", "device_name", "pixi_version"):
    print(probe[key])
PY
)
TORCH_VER=${PROBE_VALUES[0]}
CUDA_VER=${PROBE_VALUES[1]}
DEVICE_NAME=${PROBE_VALUES[2]}
PIXI_VER=${PROBE_VALUES[3]}
TILE_SHA256=$(sha256sum "${TILE_CHECKPOINT}" | cut -d' ' -f1)
SLIDE_SHA256=$(sha256sum "${SLIDE_CHECKPOINT}" | cut -d' ' -f1)

cat <<EOF > "${TMP_MANIFEST}"
{
  "definition_sha256": "${DEF_SHA256}",
  "tile_checkpoint_sha256": "${TILE_SHA256}",
  "slide_checkpoint_sha256": "${SLIDE_SHA256}",
  "lock_sha256": "${LOCK_SHA256}",
  "git_commit": "${GIT_COMMIT}",
  "sif_sha256": "${SIF_SHA256}",
  "pixi_version": "${PIXI_VER}",
  "pytorch_version": "${TORCH_VER}",
  "cuda_version": "${CUDA_VER}",
  "device_name": "${DEVICE_NAME}",
  "build_time": "${BUILD_TIME}",
  "architecture": "aarch64"
}
EOF

echo "Publishing SIF and manifest atomically..."
mv "${TMP_SIF}" "${FINAL_SIF}"
mv "${TMP_MANIFEST}" "${FINAL_MANIFEST}"

# Clear trap on success
trap - EXIT

echo "Build complete and verified:"
echo "SIF: ${FINAL_SIF}"
echo "Manifest: ${FINAL_MANIFEST}"
