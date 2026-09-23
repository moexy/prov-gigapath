#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import tifffile
from PIL import Image, ImageDraw
from tqdm import tqdm

from gigapath import slide_encoder, tile_encoder
from gigapath.preprocessing.data.create_tiles_dataset import generate_tiles, save_image
from gigapath.pipeline import (
    run_inference_with_slide_encoder,
    run_inference_with_tile_encoder,
    tile_one_slide,
)

WSI_SUFFIXES = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_revision():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_receipt(input_path):
    receipt_path = input_path / "LOCAL_VALIDATION_RECEIPT.json"
    if not receipt_path.exists():
        return {}
    receipt = json.loads(receipt_path.read_text())
    return {slide["file"]: slide for slide in receipt.get("slides", [])}


def find_slides(input_path):
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.name.lower().endswith(WSI_SUFFIXES)
    )


def validate_checkpoint(model, checkpoint_path, wrapped):
    state = torch.load(checkpoint_path, map_location="cpu")
    if wrapped:
        state = state["model"]
    missing = sorted(set(model.state_dict()) - set(state))
    unexpected = sorted(set(state) - set(model.state_dict()))
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )


def write_results(path, rows):
    fields = ["slide_id", "source", "modality", "status", "tile_count", "runtime_seconds", "output", "error"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rgb_chw(array):
    array = np.squeeze(array)
    if array.ndim != 3:
        raise RuntimeError(f"Expected RGB image, got shape {array.shape}")
    if array.shape[-1] in (3, 4):
        array = array[..., :3].transpose(2, 0, 1)
    elif array.shape[0] in (3, 4):
        array = array[:3]
    else:
        raise RuntimeError(f"Expected three RGB channels, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 RGB pixels, got {array.dtype}")
    return array


def tile_ome_tiff(source, scratch, level, tile_size=256):
    local_source = scratch / source.name
    shutil.copy2(source, local_source)
    output_root = scratch / "output"
    tile_dir = output_root / source.name
    thumbnail_dir = scratch / "thumbnails"
    tile_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(local_source) as tif:
        levels = tif.series[0].levels
        if level >= len(levels):
            raise RuntimeError(f"OME-TIFF has {len(levels)} pyramid levels; requested level {level}")
        level0_height, level0_width = levels[0].shape[:2]
        selected_height, selected_width = levels[level].shape[:2]
        slide = _rgb_chw(levels[level].asarray())
        thumbnail = Image.fromarray(_rgb_chw(levels[-1].asarray()).transpose(1, 2, 0))

    height = slide.shape[1] // tile_size * tile_size
    width = slide.shape[2] // tile_size * tile_size
    slide = slide[:, :height, :width]
    scale_x = level0_width / selected_width
    scale_y = level0_height / selected_height
    image_tiles, relative_coords, occupancies, _ = generate_tiles(
        slide, tile_size=tile_size, foreground_threshold=None, occupancy_threshold=0.1
    )
    if len(relative_coords) == 0:
        raise RuntimeError("No tissue tiles selected")
    level0_coords = np.rint(relative_coords * np.array([scale_x, scale_y])).astype(np.int64)

    dataset_path = tile_dir / "dataset.csv"
    with open(dataset_path, "w", newline="") as dataset_file:
        writer = csv.writer(dataset_file)
        writer.writerow(["slide_id", "tile_id", "image", "label", "tile_x", "tile_y", "occupancy"])
        for image, (x, y), occupancy in tqdm(
            zip(image_tiles, level0_coords, occupancies),
            total=len(image_tiles),
            desc=f"Tiles ({source.name[:12]})",
        ):
            descriptor = f"{x:05d}x_{y:05d}y"
            relative_path = f"{source.name}/{descriptor}.png"
            save_image(image, output_root / relative_path)
            writer.writerow([
                source.name,
                f"{source.name}.{descriptor}",
                relative_path,
                "",
                x,
                y,
                float(occupancy),
            ])
    (tile_dir / "failed_tiles.csv").write_text("tile_id\\n")

    thumbnail.save(thumbnail_dir / f"{source.name}_original.png")
    thumbnail.save(thumbnail_dir / f"{source.name}_roi.png")
    overlay = thumbnail.copy()
    draw = ImageDraw.Draw(overlay)
    thumb_scale_x = overlay.width / level0_width
    thumb_scale_y = overlay.height / level0_height
    tile_width = tile_size * scale_x * thumb_scale_x
    tile_height = tile_size * scale_y * thumb_scale_y
    for x, y in level0_coords:
        left = x * thumb_scale_x
        top = y * thumb_scale_y
        draw.rectangle((left, top, left + tile_width, top + tile_height), outline="red")
    overlay.save(thumbnail_dir / f"{source.name}_roi_tiles.png")



def evaluate_slide(args, source, metadata, tile_model, slide_model, model_hashes, revision):
    started = time.time()
    slide_id = metadata.get("slide", source.stem)
    output_dir = args.output / slide_id
    qc_dir = output_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "slide_id": slide_id,
        "source": str(source.resolve()),
        "source_sha256": metadata.get("sha256"),
        "modality": metadata.get("modality", "unknown"),
        "repository_revision": revision,
        "model": {
            "repository": "prov-gigapath/prov-gigapath-flash",
            "revision": args.model_revision,
            "tile_architecture": "gigapath_tile_enc_dinov2s",
            "slide_architecture": "gigapath_slide_enc12l384d",
            **model_hashes,
        },
        "preprocessing": {
            "level": args.level,
            "source_mpp": args.source_mpp,
            "effective_mpp": args.source_mpp * (2 ** args.level) if args.source_mpp else None,
            "target_mpp": args.target_mpp,
            "tile_size": 256,
            "occupancy_threshold": 0.1,
            "transform": "resize-256_center-crop-224_imagenet-normalize",
        },
        "runtime": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": args.slurm_job_id,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    scratch = args.scratch / slide_id
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    try:
        if source.name.lower().endswith(".ome.tiff"):
            tile_ome_tiff(source, scratch, args.level)
        else:
            tile_one_slide(str(source), str(scratch), level=args.level, tile_size=256)
        tile_dir = scratch / "output" / source.name
        dataset = pd.read_csv(tile_dir / "dataset.csv")
        failed = pd.read_csv(tile_dir / "failed_tiles.csv")
        if dataset.empty:
            raise RuntimeError("No tissue tiles selected")
        if not failed.empty:
            raise RuntimeError(f"{len(failed)} tiles failed during preprocessing")

        image_paths = [str(scratch / "output" / path) for path in dataset["image"]]
        tile_outputs = run_inference_with_tile_encoder(image_paths, tile_model, args.batch_size)
        features = tile_outputs["tile_embeds"].float().numpy()
        coords = tile_outputs["coords"].float().numpy()
        if features.shape != (len(dataset), 384) or coords.shape != (len(dataset), 2):
            raise RuntimeError(f"Unexpected feature shapes: features={features.shape}, coords={coords.shape}")
        if not np.isfinite(features).all() or not np.isfinite(coords).all():
            raise RuntimeError("Non-finite tile output")

        slide_outputs = run_inference_with_slide_encoder(
            torch.from_numpy(features), torch.from_numpy(coords), slide_model
        )
        embedding = slide_outputs["last_layer_embed"].squeeze(0).float().numpy()
        if embedding.shape != (384,) or not np.isfinite(embedding).all():
            raise RuntimeError(f"Unexpected slide embedding: shape={embedding.shape}")

        with h5py.File(output_dir / "tile_features.h5", "w") as handle:
            handle.create_dataset("features", data=features, compression="gzip")
            handle.create_dataset("coords", data=coords)
        np.save(output_dir / "slide_embedding.npy", embedding)

        thumbnail_dir = scratch / "thumbnails"
        qc_sources = {
            "original.png": thumbnail_dir / f"{source.name}_original.png",
            "roi.png": thumbnail_dir / f"{source.name}_roi.png",
            "tile-overlay.png": thumbnail_dir / f"{source.name}_roi_tiles.png",
        }
        for target_name, source_path in qc_sources.items():
            shutil.copy2(source_path, qc_dir / target_name)

        manifest["status"] = "success"
        manifest["tile_count"] = len(dataset)
        manifest["tile_features"] = {"shape": list(features.shape), "dtype": str(features.dtype)}
        manifest["slide_embedding"] = {"shape": list(embedding.shape), "dtype": str(embedding.dtype)}
        return {
            "slide_id": slide_id,
            "source": str(source),
            "modality": manifest["modality"],
            "status": "success",
            "tile_count": len(dataset),
            "runtime_seconds": round(time.time() - started, 3),
            "output": str(output_dir),
            "error": "",
        }
    except Exception as error:  # noqa: BLE001 - each slide must report failure and let cohort continue
        traceback.print_exc()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        return {
            "slide_id": slide_id,
            "source": str(source),
            "modality": manifest["modality"],
            "status": "failed",
            "tile_count": 0,
            "runtime_seconds": round(time.time() - started, 3),
            "output": str(output_dir),
            "error": manifest["error"],
        }
    finally:
        manifest["runtime"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["runtime"]["seconds"] = round(time.time() - started, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not args.keep_tiles:
            shutil.rmtree(scratch, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen GigaPath-Flash WSI representations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-checkpoint", type=Path, required=True)
    parser.add_argument("--slide-checkpoint", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--source-mpp", type=float)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--keep-tiles", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.input = args.input.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.slurm_job_id = __import__("os").environ.get("SLURM_JOB_ID")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if torch.cuda.get_device_capability(0)[0] < 9:
        raise RuntimeError(f"Expected GH200-class GPU, found {torch.cuda.get_device_name(0)}")

    receipt = load_receipt(args.input) if args.input.is_dir() else {}
    slides = find_slides(args.input)
    if not slides:
        raise RuntimeError(f"No WSI files found under {args.input}")

    rows = []
    brightfield = []
    seen = set()
    for relative, metadata in receipt.items():
        if metadata.get("modality") == "fluorescence":
            rows.append({
                "slide_id": metadata.get("slide", Path(relative).stem),
                "source": str(args.input / relative),
                "modality": "fluorescence",
                "status": "unsupported-modality",
                "tile_count": 0,
                "runtime_seconds": 0,
                "output": "",
                "error": "GigaPath-Flash is trained for RGB H&E brightfield slides, not fluorescence channels",
            })
    for source in slides:
        relative = str(source.relative_to(args.input)) if args.input.is_dir() else source.name
        seen.add(relative)
        metadata = receipt.get(relative, {})
        if metadata.get("modality") != "fluorescence":
            brightfield.append((source, metadata))
    for relative, metadata in receipt.items():
        if metadata.get("modality") == "brightfield" and relative not in seen:
            rows.append({
                "slide_id": metadata.get("slide", Path(relative).stem),
                "source": str(args.input / relative),
                "modality": "brightfield",
                "status": "failed",
                "tile_count": 0,
                "runtime_seconds": 0,
                "output": "",
                "error": "Input file is missing",
            })
    write_results(args.output / "results.csv", rows)

    tile_model = tile_encoder.create_model(str(args.tile_checkpoint), "gigapath_tile_enc_dinov2s")
    slide_model = slide_encoder.create_model(
        str(args.slide_checkpoint), "gigapath_slide_enc12l384d", 384, global_pool=True
    )
    validate_checkpoint(tile_model, args.tile_checkpoint, wrapped=False)
    validate_checkpoint(slide_model, args.slide_checkpoint, wrapped=True)
    tile_model.eval()
    slide_model.eval()

    hashes = {
        "tile_checkpoint_sha256": sha256(args.tile_checkpoint),
        "slide_checkpoint_sha256": sha256(args.slide_checkpoint),
    }
    revision = repo_revision()
    for source, metadata in brightfield:
        row = evaluate_slide(args, source, metadata, tile_model, slide_model, hashes, revision)
        rows.append(row)
        write_results(args.output / "results.csv", rows)

    failures = [row for row in rows if row["status"] == "failed"]
    print(json.dumps({"results": str(args.output / "results.csv"), "rows": rows}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
