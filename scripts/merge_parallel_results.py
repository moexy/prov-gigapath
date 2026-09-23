#!/usr/bin/env python3
"""Merge parallel array results into canonical results.csv and run-manifest.json.

Verifies durable output contract for each slide in shard TSV, captures failure records
from .failed/<slide-id>-<jobid>/failure.json, writes sorted results.csv and run-manifest.json.
Exits nonzero unless all expected slides succeed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

EXPECTED_QC_FILES = ("original.png", "roi.png", "tile-overlay.png")


def sha256_file(path: Path) -> str:
    """Compute hex sha256 of file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_slide_output(
    output_dir: Path,
    expected_slide_id: str,
    expected_source: Path | None = None,
    tile_checkpoint_sha256: str | None = None,
    slide_checkpoint_sha256: str | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify durable slide output contract.

    Returns (is_valid, error_message, manifest_data).
    """
    if not output_dir.is_dir():
        return False, f"Output directory missing: {output_dir}", {}

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, f"Missing manifest.json in {output_dir}", {}

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return False, f"Corrupt manifest.json: {exc}", {}

    if manifest.get("status") != "success":
        return False, f"Manifest status is not 'success': {manifest.get('status')}", manifest

    actual_id = manifest.get("slide_id")
    if actual_id != expected_slide_id:
        return False, f"Slide ID mismatch: expected '{expected_slide_id}', got '{actual_id}'", manifest
    if expected_source is not None:
        manifest_source = manifest.get("source")
        if not manifest_source or Path(manifest_source).resolve() != expected_source.resolve():
            return False, f"Source mismatch: expected '{expected_source}', got '{manifest_source}'", manifest
    model = manifest.get("model", {})
    expected_hashes = {
        "tile_checkpoint_sha256": tile_checkpoint_sha256,
        "slide_checkpoint_sha256": slide_checkpoint_sha256,
    }
    for key, expected in expected_hashes.items():
        if expected is not None and model.get(key) != expected:
            return False, f"Model hash mismatch for {key}: expected '{expected}', got '{model.get(key)}'", manifest

    # Tile features H5 verification: finite [N, 384] and coords [N, 2]
    h5_path = output_dir / "tile_features.h5"
    if not h5_path.is_file():
        return False, f"Missing tile_features.h5 in {output_dir}", manifest

    try:
        with h5py.File(h5_path, "r") as h5:
            if "features" not in h5 or "coords" not in h5:
                return False, "tile_features.h5 missing 'features' or 'coords' dataset", manifest
            features = h5["features"][:]
            coords = h5["coords"][:]
    except Exception as exc:
        return False, f"Failed to read tile_features.h5: {exc}", manifest

    if features.ndim != 2 or features.shape[1] != 384 or features.shape[0] == 0:
        return False, f"Invalid features shape {features.shape}, expected [N, 384] with N > 0", manifest

    n_tiles = features.shape[0]
    if coords.ndim != 2 or coords.shape != (n_tiles, 2):
        return False, f"Invalid coords shape {coords.shape}, expected ({n_tiles}, 2)", manifest

    if not np.isfinite(features).all() or not np.isfinite(coords).all():
        return False, "Non-finite values in tile features or coords", manifest

    # Slide embedding NPY verification: finite [384]
    npy_path = output_dir / "slide_embedding.npy"
    if not npy_path.is_file():
        return False, f"Missing slide_embedding.npy in {output_dir}", manifest

    try:
        embedding = np.load(npy_path)
    except Exception as exc:
        return False, f"Failed to read slide_embedding.npy: {exc}", manifest

    if embedding.shape != (384,) or not np.isfinite(embedding).all():
        return False, f"Invalid slide embedding: shape={embedding.shape}, expected finite (384,)", manifest

    # QC images verification: original.png, roi.png, tile-overlay.png
    qc_dir = output_dir / "qc"
    if not qc_dir.is_dir():
        return False, f"Missing qc directory in {output_dir}", manifest

    for qc_name in EXPECTED_QC_FILES:
        qc_file = qc_dir / qc_name
        if not qc_file.is_file() or qc_file.stat().st_size == 0:
            return False, f"Missing or empty QC image: {qc_file}", manifest
    log_path = output_dir / "logs" / "evaluation.log"
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return False, f"Missing or empty evaluation log: {log_path}", manifest

    return True, "", manifest


def read_shards_tsv(shards_path: Path) -> List[Dict[str, str]]:
    """Read and validate shard TSV.

    Columns strictly: node_shard,gpu_slot,slide_id,source_path,weight.
    Rejects malformed header, invalid types, or duplicate slide IDs.
    """
    if not shards_path.is_file():
        raise ValueError(f"Shard TSV not found: {shards_path}")

    expected_cols = ["node_shard", "gpu_slot", "slide_id", "source_path", "weight"]
    rows: List[Dict[str, str]] = []
    seen_ids = set()
    seen_slots = set()

    with open(shards_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None or header != expected_cols:
            raise ValueError(f"Shard TSV header must be exactly {expected_cols}, got {header}")

        for line_num, parts in enumerate(reader, start=2):
            if len(parts) != 5:
                raise ValueError(f"Line {line_num} has {len(parts)} columns, expected 5")
            node_shard, gpu_slot, slide_id, source_path, weight = parts

            try:
                node = int(node_shard)
                slot = int(gpu_slot)
                float(weight)
            except ValueError as exc:
                raise ValueError(f"Line {line_num} has invalid numeric value: {exc}") from exc
            if node < 0 or slot not in range(4):
                raise ValueError(f"Line {line_num} has invalid node_shard/gpu_slot: {node}/{slot}")
            worker = (node, slot)
            if worker in seen_slots:
                raise ValueError(f"Duplicate GPU slot {slot} in node_shard {node}")
            seen_slots.add(worker)
            if not source_path:
                raise ValueError(f"Line {line_num} has empty source_path")
            if not slide_id:
                raise ValueError(f"Line {line_num} has empty slide_id")
            if slide_id in seen_ids:
                raise ValueError(f"Duplicate slide_id '{slide_id}' at line {line_num}")
            seen_ids.add(slide_id)
            rows.append({
                "node_shard": node_shard,
                "gpu_slot": gpu_slot,
                "slide_id": slide_id,
                "source_path": source_path,
                "weight": weight,
            })

    if not rows:
        raise ValueError("Shard TSV contains no data rows")

    return rows


def find_latest_failure(failed_root: Path, slide_id: str) -> Optional[Dict[str, Any]]:
    """Locate latest matching durable failure JSON in output/.failed/<slide-id>-<jobid>/failure.json."""
    if not failed_root.is_dir():
        return None

    matching_dirs = [
        d for d in failed_root.iterdir()
        if d.is_dir() and (d.name == slide_id or d.name.startswith(f"{slide_id}-"))
    ]
    if not matching_dirs:
        return None

    # Sort by mtime descending
    matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for target_dir in matching_dirs:
        fail_json = target_dir / "failure.json"
        if fail_json.is_file():
            try:
                return json.loads(fail_json.read_text())
            except Exception:
                continue
    return None


def merge_results(
    shards_path: Path,
    output_root: Path,
    array_job_id: str,
    sif_path: Optional[Path] = None,
    tile_checkpoint: Optional[Path] = None,
    slide_checkpoint: Optional[Path] = None,
    requested_nodes: Optional[int] = None,
    requested_gpus: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Tuple[bool, Path, Path, Dict[str, Any]]:
    """Verify all shard outputs and write deterministic aggregate artifacts."""
    shard_rows = read_shards_tsv(shards_path)
    shard_hash = sha256_file(shards_path)
    sif_hash = sha256_file(sif_path) if sif_path and sif_path.is_file() else None
    tile_hash = sha256_file(tile_checkpoint) if tile_checkpoint and tile_checkpoint.is_file() else None
    slide_hash = sha256_file(slide_checkpoint) if slide_checkpoint and slide_checkpoint.is_file() else None
    failed_root = output_root / ".failed"
    slide_results = []
    status_counts = {"success": 0, "failed": 0, "missing": 0}
    per_slide_paths: Dict[str, Optional[str]] = {}

    for shard in shard_rows:
        slide_id = shard["slide_id"]
        source_path = shard["source_path"]
        slide_dir = output_root / slide_id
        valid, error, manifest = verify_slide_output(
            slide_dir,
            slide_id,
            Path(source_path),
            tile_hash,
            slide_hash,
        )
        if valid:
            status_counts["success"] += 1
            per_slide_paths[slide_id] = str(slide_dir)
            slide_results.append({
                "slide_id": slide_id,
                "source": manifest.get("source", source_path),
                "modality": manifest.get("modality", "brightfield"),
                "status": "success",
                "tile_count": manifest.get("tile_count", 0),
                "runtime_seconds": round(manifest.get("runtime", {}).get("seconds", 0.0), 3),
                "output": str(slide_dir),
                "error": "",
            })
            continue

        per_slide_paths[slide_id] = None
        failure = find_latest_failure(failed_root, slide_id)
        if failure or slide_dir.is_dir():
            status = "failed"
            status_counts["failed"] += 1
            detail = failure.get("error", error) if failure else error
        else:
            status = "missing"
            status_counts["missing"] += 1
            detail = error or "Slide output directory missing"
        slide_results.append({
            "slide_id": slide_id,
            "source": source_path,
            "modality": "brightfield",
            "status": status,
            "tile_count": 0,
            "runtime_seconds": 0.0,
            "output": str(slide_dir) if slide_dir.is_dir() else "",
            "error": detail,
        })

    slide_results.sort(key=lambda row: row["slide_id"])
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "results.csv"
    fields = ["slide_id", "source", "modality", "status", "tile_count", "runtime_seconds", "output", "error"]
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(slide_results)

    unique_nodes = {int(row["node_shard"]) for row in shard_rows}
    run_manifest = {
        "array_job_id": array_job_id,
        "shards": {
            "path": str(shards_path.resolve()),
            "sha256": shard_hash,
            "total_slides": len(shard_rows),
        },
        "artifacts": {
            "sif": {"path": str(sif_path.resolve()) if sif_path else None, "sha256": sif_hash},
            "tile_checkpoint": {"path": str(tile_checkpoint.resolve()) if tile_checkpoint else None, "sha256": tile_hash},
            "slide_checkpoint": {"path": str(slide_checkpoint.resolve()) if slide_checkpoint else None, "sha256": slide_hash},
        },
        "resources": {
            "requested_nodes": requested_nodes or len(unique_nodes),
            "requested_gpus": requested_gpus or len(shard_rows),
            "node_shards": len(unique_nodes),
        },
        "timing": {
            "started_at": start_time,
            "ended_at": end_time or datetime.now(timezone.utc).isoformat(),
        },
        "status_counts": status_counts,
        "outputs": per_slide_paths,
    }
    run_manifest_path = output_root / "run-manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    return status_counts["success"] == len(shard_rows), results_csv, run_manifest_path, status_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge parallel array results into results.csv and run-manifest.json")
    parser.add_argument("--shards", type=Path, help="Path to input shard TSV")
    parser.add_argument("--output", type=Path, help="Durable cohort output directory")
    parser.add_argument("--array-job-id", type=str, help="Slurm array job ID")
    parser.add_argument("--sif", type=Path, default=None, help="Apptainer SIF image path")
    parser.add_argument("--tile-checkpoint", type=Path, default=None, help="Tile model checkpoint path")
    parser.add_argument("--slide-checkpoint", type=Path, default=None, help="Slide model checkpoint path")
    parser.add_argument("--requested-nodes", type=int, default=None, help="Requested node count")
    parser.add_argument("--requested-gpus", type=int, default=None, help="Requested GPU count")
    parser.add_argument("--start-time", type=str, default=None, help="Cohort run start timestamp (ISO format)")
    parser.add_argument("--end-time", type=str, default=None, help="Cohort run end timestamp (ISO format)")
    parser.add_argument("--self-test", action="store_true", help="Run compact self-check and exit")
    return parser.parse_args()


def demo() -> None:
    """Self-check testing verification and aggregation logic."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output_dir = root / "output"
        output_dir.mkdir()

        # 1. Create fake shard TSV
        shards_tsv = root / "shards.tsv"
        tsv_content = (
            "node_shard\tgpu_slot\tslide_id\tsource_path\tweight\n"
            "0\t0\tBF-0002\t/path/to/BF-0002.ome.tiff\t1000\n"
            "0\t1\tBF-0001\t/path/to/BF-0001.ome.tiff\t2000\n"
            "1\t0\tBF-0003\t/path/to/BF-0003.ome.tiff\t1500\n"
        )
        shards_tsv.write_text(tsv_content)

        # 2. Build valid slide 1 (BF-0001)
        s1 = output_dir / "BF-0001"
        s1.mkdir()
        (s1 / "qc").mkdir()
        for qc_name in EXPECTED_QC_FILES:
            (s1 / "qc" / qc_name).write_bytes(b"PNG")
        manifest1 = {
            "status": "success",
            "slide_id": "BF-0001",
            "source": "/path/to/BF-0001.ome.tiff",
            "modality": "brightfield",
            "tile_count": 5,
            "runtime": {"seconds": 12.34},
        }
        (s1 / "logs").mkdir()
        (s1 / "logs" / "evaluation.log").write_text("ok\n")
        (s1 / "manifest.json").write_text(json.dumps(manifest1))
        with h5py.File(s1 / "tile_features.h5", "w") as h5:
            h5.create_dataset("features", data=np.ones((5, 384), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((5, 2), dtype=np.float32))
        np.save(s1 / "slide_embedding.npy", np.zeros(384, dtype=np.float32))

        # 3. Build invalid slide 2 (BF-0002) - missing embedding
        s2 = output_dir / "BF-0002"
        s2.mkdir()
        (s2 / "qc").mkdir()
        for qc_name in EXPECTED_QC_FILES:
            (s2 / "qc" / qc_name).write_bytes(b"PNG")
        (s2 / "manifest.json").write_text(json.dumps({
            "status": "success",
            "slide_id": "BF-0002",
            "source": "/path/to/BF-0002.ome.tiff",
        }))
        with h5py.File(s2 / "tile_features.h5", "w") as h5:
            h5.create_dataset("features", data=np.ones((2, 384), dtype=np.float32))
            h5.create_dataset("coords", data=np.zeros((2, 2), dtype=np.float32))

        (s2 / "logs").mkdir()
        (s2 / "logs" / "evaluation.log").write_text("failed\n")
        # 4. Slide 3 (BF-0003) missing, with durable failure record
        fail_dir = output_dir / ".failed" / "BF-0003-99999"
        fail_dir.mkdir(parents=True)
        (fail_dir / "failure.json").write_text(json.dumps({
            "slide_id": "BF-0003",
            "error": "OOM while tiling",
            "exit_code": 137,
        }))

        # Run merge
        all_ok, csv_p, manifest_p, stats = merge_results(
            shards_path=shards_tsv,
            output_root=output_dir,
            array_job_id="12345",
        )

        assert not all_ok, "Expected failure because 2 slides failed/missing"
        assert stats["success"] == 1
        assert stats["failed"] == 2
        assert stats["missing"] == 0

        # Check results.csv order (canonical ID sort: BF-0001, BF-0002, BF-0003)
        with open(csv_p, "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert [r["slide_id"] for r in rows] == ["BF-0001", "BF-0002", "BF-0003"]
        assert rows[0]["status"] == "success"
        assert rows[1]["status"] == "failed"
        assert "Missing slide_embedding.npy" in rows[1]["error"]
        assert "OOM while tiling" in rows[2]["error"]

        # Check run-manifest.json
        manifest_data = json.loads(manifest_p.read_text())
        assert manifest_data["array_job_id"] == "12345"
        assert manifest_data["shards"]["total_slides"] == 3
        assert manifest_data["resources"]["node_shards"] == 2

    print("Self-check demo() passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        demo()
        return 0
    if args.shards is None or args.output is None or args.array_job_id is None:
        raise SystemExit("--shards, --output, and --array-job-id are required")

    try:
        all_success, csv_path, manifest_path, stats = merge_results(
            shards_path=args.shards,
            output_root=args.output,
            array_job_id=args.array_job_id,
            sif_path=args.sif,
            tile_checkpoint=args.tile_checkpoint,
            slide_checkpoint=args.slide_checkpoint,
            requested_nodes=args.requested_nodes,
            requested_gpus=args.requested_gpus,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    except Exception as exc:
        print(f"Error during merge: {exc}", file=sys.stderr)
        return 1

    print(f"Results merged: {csv_path}")
    print(f"Run manifest: {manifest_path}")
    print(f"Status summary: {stats}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
