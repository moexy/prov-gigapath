# GigaPath WSI Evaluation on Arrhenius

Follow this file for work in this repository. Compute runs on Arrhenius. Keep changes minimal and reuse `gigapath/` before adding code.

## Goal and meaning of zero-shot

Default task: run frozen GigaPath-Flash tile and slide encoders on one or more H&E whole-slide images (WSIs), without training or fitting, and save reproducible tile and slide representations plus quality-control artifacts.

GigaPath-Flash is not a vision-language model and has no text classifier. In this repository, “zero-shot” means frozen feature extraction, similarity, retrieval, clustering, or comparison against an already supplied reference representation. It does not mean diagnosis from class names. Do not report accuracy, QWK, or balanced accuracy without labels and an explicit evaluation rule. The PANDA and EBRAINS results in the paper used supervised downstream training for five epochs; they are not zero-shot results.

Research use only. Never present output as clinical evidence, diagnosis, or a validated biomarker.

Fluorescence files are out of scope for this repository. Exclude them from input discovery, feature analysis, classifier fitting, evaluation, plots, and aggregate metrics. Record an exclusion count only when auditing a mixed source receipt.

Current classifier baseline predicts source-slide identity from tile embeddings with spatial-block cross-validation. Treat it only as a representation separability check; one slide per class cannot estimate biological, patient-level, or external generalization.

## Arrhenius rules

- Work under `/nobackup/proj/disk/muc/personal/$USER`; use `/nobackup/proj/disk/muc/shared` only for group data.
- Never store environments, WSI collections, checkpoints, tiles, or outputs in `$HOME`.
- Login nodes are x86_64 and only for editing, transfers, dependency solving, and Slurm submission. Run preprocessing and inference in Slurm jobs.
- This project can use only `-A naiss2026-4-1499-gpu -p gpu`.
- GPU nodes are aarch64 GH200 nodes. Build and solve `linux-aarch64`; do not reuse linux-64 binaries.
- Prefer Pixi. Use `platforms = [{ platform = "linux-aarch64", cuda = "13" }]`. Run a GPU check inside the allocation before evaluation.
- Use one GPU first. Request more only after measured single-GPU throughput shows a need. One node has four GPUs.
- Set wall time explicitly. Validate new batch scripts with `sbatch --test-only` before submission.
- Use `/scratch/local/$SLURM_JOB_ID` for temporary WSI copies and generated PNG tiles. Copy durable outputs back to project storage before exit. Scratch is deleted after the job.
- Keep `HF_TOKEN` in the environment. Never commit or print it. Accept the gated Hugging Face model terms before use.
- Put Hugging Face caches under project storage, for example `HF_HOME=/nobackup/proj/disk/muc/personal/$USER/.cache/huggingface`.

Minimum Slurm shape for a first run:

```bash
#SBATCH -A naiss2026-4-1499-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 72
#SBATCH --mem=100G
#SBATCH -t 04:00:00
```

## Model contract

Use GigaPath-Flash unless the user explicitly asks for original GigaPath.

- Hugging Face repository: `prov-gigapath/prov-gigapath-flash`.
- Tile encoder: `gigapath_tile_enc_dinov2s`, ViT-S/16, 22M parameters, 384-dimensional output.
- Slide encoder: `gigapath_slide_enc12l384d`, 12-layer LongNet, 21M parameters, `in_chans=384`, 384-dimensional output.
- Load the Flash tile encoder through `gigapath.tile_encoder.create_model`; importing `gigapath.tile_encoder` registers its timm architecture.
- Load the Flash slide encoder through `gigapath.slide_encoder.create_model`.
- Set `global_pool=True`. Repository demos state that the slide encoder CLS token was not trained.
- Use local checkpoint paths in batch jobs after downloading once. The current Hub-loading helpers force downloads.
- Fail if weights are absent or randomly initialized. Treat missing or unexpected state-dict keys as an error unless a known release-format exception is documented.

Canonical loading:

```python
import gigapath.tile_encoder as tile_encoder
import gigapath.slide_encoder as slide_encoder

tile_model = tile_encoder.create_model(
    "/project/cache/prov-gigapath-flash/pytorch_model.bin",
    "gigapath_tile_enc_dinov2s",
)
slide_model = slide_encoder.create_model(
    "/project/cache/prov-gigapath-flash/slide_encoder.pth",
    "gigapath_slide_enc12l384d",
    384,
    global_pool=True,
)
```

Replace `/project/cache` with a project-storage path. Never add a second model-loading implementation.

## WSI input contract

- Use OpenSlide-readable H&E WSI files such as SVS, NDPI, MRXS, or pyramidal TIFF.
- For pyramidal OME-TIFF, use `scripts/evaluate_wsi.py`; it reads TIFF pyramid levels with `tifffile` because OpenSlide can expose only level 0 for these files.
- Evaluate at 20×, normally 0.5 microns per pixel (MPP), matching repository guidance and paper protocol.
- Start with `gigapath.preprocessing.data.slide_utils.find_level_for_target_mpp`. It only understands TIFF centimeter-resolution fields. If it cannot parse the slide, inspect OpenSlide `openslide.mpp-x`/`openslide.mpp-y`, vendor metadata, and level downsampling directly.
- If MPP metadata is missing or inconsistent, stop for an explicit scanner-specific level or MPP override. Never silently assume level 0.
- Tile tissue into non-overlapping 256×256-pixel tiles. Use the repository tiler and its default occupancy threshold of 0.1 unless the user requests another documented protocol.
- Preserve level-0 `(x, y)` coordinates. The slide encoder converts these coordinates using a 256-pixel tile grid.
- Apply the repository transform exactly: resize to 256, center-crop to 224, convert to tensor, and normalize with ImageNet mean and standard deviation.
- Never silently subsample tiles. If memory or runtime fails, stream or batch tiles; do not change magnification, tile size, tissue threshold, or tile count to hide the failure.
- Keep slide identity separate from file extension. Avoid collisions when different source files share a stem.

## Required evaluation flow

1. Inventory inputs. Record path, byte size, OpenSlide vendor, level dimensions, level downsampling, native MPP, selected level, and effective MPP.
2. Run one representative WSI interactively before cohort submission.
3. Tile with `gigapath.pipeline.tile_one_slide` or the same repository primitives. Save the original, ROI, and tile-overlay thumbnails.
4. Verify at least one tile exists and `failed_tiles.csv` has no entries. Visually inspect the overlay and several tiles.
5. Encode tiles with `gigapath.pipeline.run_inference_with_tile_encoder`. Use batch size 128 as a starting point; lower it only after an observed out-of-memory error.
6. Verify tile features have shape `[N, 384]`, coordinates have shape `[N, 2]`, counts match, and all values are finite.
7. Encode the full slide with `gigapath.pipeline.run_inference_with_slide_encoder`. Save `last_layer_embed`; expected shape after removing the batch dimension is `[384]`.
8. Remove temporary PNG tiles only after durable features, embeddings, manifests, and QC images are copied to project storage.
9. Resume by skipping only outputs whose manifest and shape checks pass. Never treat file existence alone as completion.

If an end-to-end command is needed, add one small script rather than a new framework. Preferred interface:

```bash
python scripts/evaluate_wsi.py \
  --input /path/to/slide-or-directory \
  --output /nobackup/proj/disk/muc/personal/$USER/gigapath-results \
  --tile-checkpoint /path/to/pytorch_model.bin \
  --slide-checkpoint /path/to/slide_encoder.pth \
  --target-mpp 0.5 \
  --tile-size 256 \
  --batch-size 128
```

The script must accept one file or a directory, process slides independently, continue after a per-slide failure, and return nonzero if any slide fails.

## Durable outputs

Write one directory per slide:

```text
<output>/<slide-id>/
  manifest.json
  tile_features.h5
  slide_embedding.npy
  qc/
    original.png
    roi.png
    tile-overlay.png
  logs/
    evaluation.log
```

`tile_features.h5` must contain `features` with shape `[N, 384]` and `coords` with shape `[N, 2]`, matching the repository fine-tuning dataset format. Use float32 unless storage pressure is measured; if float16 is used, record it in the manifest. Save the final pooled slide embedding as float32.

`manifest.json` must record:

- source path and stable slide ID;
- repository commit;
- model repository and resolved revision or checkpoint SHA-256;
- model architecture names;
- selected level, native MPP, target MPP, tile size, occupancy threshold, transform, and tile count;
- feature and embedding shapes and dtypes;
- device name, PyTorch/CUDA versions, Slurm job ID, start/end times, and status;
- failure message when status is not successful.

For a cohort, also write `results.csv` with one row per slide and status, tile count, runtime, output paths, and concise error text.

## Verification gate

Do not call a run successful until all checks pass:

- `uname -m` is `aarch64` inside the job.
- `torch.cuda.is_available()` is true and device is GH200.
- Both checkpoints load pretrained weights.
- Effective sampling is 20×/0.5 MPP, or the manifest records an explicit approved deviation.
- Tile count is positive; failed tile count is zero.
- Tile features are finite `[N, 384]`; coordinates are finite `[N, 2]`.
- Slide embedding is finite `[384]`.
- QC overlay shows tissue coverage and coordinate alignment.
- A repeated smoke run of the same slide produces equivalent output within the chosen mixed-precision tolerance.
- Durable outputs exist on project storage after scratch cleanup.

For labeled downstream evaluation, follow the paper protocol only when explicitly requested: fixed common splits, five epochs, final-epoch checkpoint, PANDA QWK, and EBRAINS balanced accuracy. Keep those supervised results separate from frozen zero-shot representations.

## Sources

- Repository inference and preprocessing: `README.md`, `gigapath/pipeline.py`, `gigapath/tile_encoder.py`, `gigapath/slide_encoder.py`.
- GigaPath-Flash paper: https://arxiv.org/html/2607.18218v2
- Arrhenius site guidance inherited from `/home/moe/AGENTS.md`; repository instructions here take precedence for GigaPath-specific work.
