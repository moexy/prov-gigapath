#!/usr/bin/env python3
from __future__ import annotations
"""Submit parallel WSI inference cohort as a 4-GPU-per-node Slurm job array.

Preflight validates input files, checksums, shard TSV schemas, slide IDs, and paths.
Outputs passing verification are excluded to generate a filtered resume TSV.
Submits an array job, followed by an aggregate merge job dependent on completion.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.merge_parallel_results import (
    read_shards_tsv as merge_read_shards_tsv,
    sha256_file as merge_sha256_file,
    verify_slide_output as merge_verify_slide_output,
)
from scripts.wsi_paths import slide_id_from_path

REQUIRED_TSV_COLUMNS = ["node_shard", "gpu_slot", "slide_id", "source_path", "weight"]


def sha256_file(path: Path) -> str:
    return merge_sha256_file(path)


def verify_slide_output(
    output_dir: Path,
    expected_slide_id: str,
    expected_source: Path | None = None,
    tile_sha256: str | None = None,
    slide_sha256: str | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    return merge_verify_slide_output(
        output_dir / expected_slide_id,
        expected_slide_id,
        expected_source,
        tile_sha256,
        slide_sha256,
    )


def load_expected_sha256(file_path: Path, explicit_sha: Optional[str] = None) -> str:
    """Resolve expected sha256 from explicit argument, .manifest.json, or file hash."""
    if explicit_sha:
        return explicit_sha.strip().lower()

    manifest_path = file_path.parent / f"{file_path.name}.manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            for k in ("sif_sha256", "sha256", "file_sha256"):
                if k in meta and isinstance(meta[k], str):
                    return meta[k].strip().lower()
        except Exception:
            pass

    return sha256_file(file_path)


def validate_shards_tsv(shards_path: Path) -> List[Dict[str, str]]:
    if not shards_path.is_file():
        raise FileNotFoundError(f"Shards TSV file not found: {shards_path}")
    rows = merge_read_shards_tsv(shards_path)
    node_shards = sorted({int(row["node_shard"]) for row in rows})
    if node_shards != list(range(len(node_shards))):
        raise ValueError(f"node_shard values must be contiguous from zero, got {node_shards}")
    seen_sources = set()
    for row in rows:
        source_path = Path(row["source_path"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Slide '{row['slide_id']}' source_path does not exist: {source_path}"
            )
        if row["slide_id"] != slide_id_from_path(source_path):
            raise ValueError(
                f"Noncanonical slide_id '{row['slide_id']}' for source '{source_path}'"
            )
        if source_path in seen_sources:
            raise ValueError(f"Duplicate source_path in shard plan: {source_path}")
        seen_sources.add(source_path)
    return rows


def filter_resume_shards(
    rows: List[Dict[str, str]],
    output_root: Path,
    tile_sha256: str | None = None,
    slide_sha256: str | None = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    pending_rows: List[Dict[str, str]] = []
    skipped_ids: List[str] = []
    for row in rows:
        slide_id = row["slide_id"]
        valid, _, _ = verify_slide_output(
            output_root,
            slide_id,
            Path(row["source_path"]),
            tile_sha256,
            slide_sha256,
        )
        (skipped_ids if valid else pending_rows).append(slide_id if valid else row)
    return pending_rows, skipped_ids


def recompact_node_shards(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Regenerate contiguous 0-based node_shard and 0-3 gpu_slot assignments."""
    recompacted: List[Dict[str, str]] = []
    for idx, r in enumerate(rows):
        node = idx // 4
        slot = idx % 4
        new_row = dict(r)
        new_row["node_shard"] = str(node)
        new_row["gpu_slot"] = str(slot)
        recompacted.append(new_row)
    return recompacted


def write_shards_tsv(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write shards TSV to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def run_sbatch_command(cmd: List[str]) -> str:
    """Execute sbatch command and return stripped stdout."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"sbatch failed with exit code {res.returncode}:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
        )
    return res.stdout.strip()


def parse_job_id(sbatch_stdout: str) -> str:
    """Extract job id from 'Submitted batch job 12345'."""
    m = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not m:
        raise ValueError(f"Could not parse job ID from sbatch output: {sbatch_stdout}")
    return m.group(1)


def generate_monitor_commands(array_job_id: str) -> Dict[str, str]:
    """Generate exact 5-second monitoring commands."""
    return {
        "squeue": f"while true; do date; squeue -j {array_job_id} -O 'JobID,ArrayJobID,ArrayTaskID,NodeList:20,State,TimeUsed,Reason'; sleep 5; done",
        "sstat": f"while true; do date; sstat -j {array_job_id}.batch --format=JobID,AveCPU,AveRSS,MaxRSS,AveDiskRead,AveDiskWrite -a 2>/dev/null || true; sleep 5; done",
        "nvidia_smon": f"for node in $(squeue -j {array_job_id} -h -t R -O 'NodeList' | sort -u | grep -v '^$'); do ssh -o StrictHostKeyChecking=no \"$node\" 'nvidia-smi dmon -s pucm -d 5' & done",
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight, resume check, and submission of 4-GPU-per-node Slurm inference array and merge job."
    )
    parser.add_argument("--shards", type=Path, required=True, help="Path to input shards.tsv")
    parser.add_argument("--sif", type=Path, required=True, help="Path to Apptainer SIF image")
    parser.add_argument(
        "--tile-checkpoint", type=Path, required=True, help="Path to local tile checkpoint"
    )
    parser.add_argument(
        "--slide-checkpoint", type=Path, required=True, help="Path to local slide checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Durable output directory for cohort results"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Max simultaneous nodes / array concurrency limit %%max_nodes (default: all shard nodes)",
    )
    parser.add_argument("--sif-sha256", type=str, default=None, help="Expected sha256 for SIF")
    parser.add_argument(
        "--tile-sha256", type=str, default=None, help="Expected sha256 for tile checkpoint"
    )
    parser.add_argument(
        "--slide-sha256", type=str, default=None, help="Expected sha256 for slide checkpoint"
    )
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=Path("slurm/evaluate_sharded_node.sh"),
        help="Path to node evaluation sbatch script",
    )
    parser.add_argument(
        "--merge-script",
        type=Path,
        default=Path("slurm/merge_parallel_results.sh"),
        help="Path to merge sbatch script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and display commands without invoking sbatch",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=== Step 1: Preflight Validation ===")
    shards_path = args.shards.resolve()
    sif_path = args.sif.resolve()
    tile_ckpt = args.tile_checkpoint.resolve()
    slide_ckpt = args.slide_checkpoint.resolve()
    output_dir = args.output.resolve()

    if not sif_path.is_file():
        raise FileNotFoundError(f"SIF container file does not exist: {sif_path}")
    if not tile_ckpt.is_file():
        raise FileNotFoundError(f"Tile checkpoint file does not exist: {tile_ckpt}")
    if not slide_ckpt.is_file():
        raise FileNotFoundError(f"Slide checkpoint file does not exist: {slide_ckpt}")

    print("Validating input shards TSV...")
    original_rows = validate_shards_tsv(shards_path)
    print(f"Validated {len(original_rows)} rows in {shards_path}")

    print("Validating file checksums...")
    expected_sif_sha = load_expected_sha256(sif_path, args.sif_sha256)
    actual_sif_sha = sha256_file(sif_path)
    if actual_sif_sha != expected_sif_sha:
        raise ValueError(
            f"SIF sha256 mismatch!\nExpected: {expected_sif_sha}\nActual:   {actual_sif_sha}"
        )
    print(f"SIF verified: {actual_sif_sha}")

    expected_tile_sha = load_expected_sha256(tile_ckpt, args.tile_sha256)
    actual_tile_sha = sha256_file(tile_ckpt)
    if actual_tile_sha != expected_tile_sha:
        raise ValueError(
            f"Tile checkpoint sha256 mismatch!\nExpected: {expected_tile_sha}\nActual:   {actual_tile_sha}"
        )
    print(f"Tile checkpoint verified: {actual_tile_sha}")

    expected_slide_sha = load_expected_sha256(slide_ckpt, args.slide_sha256)
    actual_slide_sha = sha256_file(slide_ckpt)
    if actual_slide_sha != expected_slide_sha:
        raise ValueError(
            f"Slide checkpoint sha256 mismatch!\nExpected: {expected_slide_sha}\nActual:   {actual_slide_sha}"
        )
    print(f"Slide checkpoint verified: {actual_slide_sha}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    print("\n=== Step 2: Resume Check and Filtering ===")
    pending_rows, skipped_ids = filter_resume_shards(
        original_rows, output_dir, actual_tile_sha, actual_slide_sha
    )
    print(f"Total slides: {len(original_rows)}")
    print(f"Already verified: {len(skipped_ids)}")
    print(f"Pending execution: {len(pending_rows)}")

    merge_script = args.merge_script.resolve()
    if not merge_script.is_file():
        raise FileNotFoundError(f"Merge script not found: {merge_script}")
    if not pending_rows:
        nodes = len({int(row["node_shard"]) for row in original_rows})
        command = [
            "sbatch",
            f"--output={output_dir}/logs/gigapath-merge-%j.out",
            str(merge_script),
            str(shards_path),
            str(output_dir),
            "aggregate-only",
            str(sif_path),
            str(tile_ckpt),
            str(slide_ckpt),
            str(nodes),
            str(nodes * 4),
        ]
        if args.dry_run:
            print(" ".join(command))
            return 0
        run_sbatch_command(["sbatch", "--test-only", *command[1:]])
        merge_job_id = parse_job_id(run_sbatch_command(command))
        print(f"All slides verified; submitted aggregate-only merge job {merge_job_id}.")
        return 0

    if skipped_ids:
        # Re-compact node shards to avoid idle array nodes
        active_rows = recompact_node_shards(pending_rows)
        run_shards_tsv = output_dir / f"shards.resume.{len(active_rows)}.tsv"
        write_shards_tsv(run_shards_tsv, active_rows)
        print(f"Generated compacted resume shards TSV with {len(active_rows)} rows: {run_shards_tsv}")
    else:
        active_rows = original_rows
        run_shards_tsv = shards_path

    total_node_shards = max(int(r["node_shard"]) for r in active_rows) + 1
    max_nodes = args.max_nodes if args.max_nodes is not None else total_node_shards
    if max_nodes <= 0:
        raise ValueError(f"--max-nodes must be positive, got {max_nodes}")

    array_spec = f"0-{total_node_shards - 1}%{max_nodes}"
    print(f"Slurm array spec: --array={array_spec} (Total node shards: {total_node_shards})")

    eval_script = args.evaluate_script.resolve()
    merge_script = args.merge_script.resolve()

    if not eval_script.is_file():
        raise FileNotFoundError(f"Evaluate script not found: {eval_script}")
    if not merge_script.is_file():
        raise FileNotFoundError(f"Merge script not found: {merge_script}")

    array_args = [
        str(run_shards_tsv),
        str(sif_path),
        actual_sif_sha,
        str(tile_ckpt),
        actual_tile_sha,
        str(slide_ckpt),
        actual_slide_sha,
        str(output_dir),
    ]

    array_cmd = [
        "sbatch",
        f"--array={array_spec}",
        f"--output={output_dir}/logs/gigapath-node-%A_%a.out",
        str(eval_script),
        *array_args,
    ]

    print("\n=== Step 3: Submission ===")
    if args.dry_run:
        print("[Dry-run] Validated successfully. Commands that would run:")
        print("Array submit:")
        print("  " + " ".join(array_cmd))
        print("Merge submit (dependency on array job):")
        print(
            f"  sbatch --dependency=afterany:<ARRAY_JOB_ID> {merge_script} {shards_path} {output_dir} <ARRAY_JOB_ID> {sif_path} {tile_ckpt} {slide_ckpt} {total_node_shards} {total_node_shards * 4}"
        )
        return 0

    print("Running sbatch --test-only validation...")
    test_cmd = [
        "sbatch", "--test-only", f"--array={array_spec}",
        f"--output={output_dir}/logs/gigapath-node-%A_%a.out",
        str(eval_script), *array_args,
    ]
    run_sbatch_command(test_cmd)
    print("sbatch --test-only passed.")

    print("Submitting array job...")
    array_stdout = run_sbatch_command(array_cmd)
    array_job_id = parse_job_id(array_stdout)
    print(f"Array job submitted: {array_job_id}")

    merge_cmd = [
        "sbatch",
        f"--dependency=afterany:{array_job_id}",
        f"--output={output_dir}/logs/gigapath-merge-%j.out",
        str(merge_script),
        str(shards_path),
        str(output_dir),
        array_job_id,
        str(sif_path),
        str(tile_ckpt),
        str(slide_ckpt),
        str(total_node_shards),
        str(total_node_shards * 4),
    ]

    print("Submitting merge job...")
    merge_stdout = run_sbatch_command(merge_cmd)
    merge_job_id = parse_job_id(merge_stdout)
    print(f"Merge job submitted: {merge_job_id}")

    print("\n=== Submission Summary ===")
    print(f"Array Job ID: {array_job_id}")
    print(f"Merge Job ID: {merge_job_id}")
    print(f"Active Shards TSV: {run_shards_tsv}")
    print(f"Durable Output: {output_dir}")

    mon_cmds = generate_monitor_commands(array_job_id)
    print("\n=== 5-Second Monitoring Commands ===")
    print("# 1. Slurm tasks and placement:")
    print(f"  {mon_cmds['squeue']}\n")
    print("# 2. Step resource utilization (CPU/RSS/Disk):")
    print(f"  {mon_cmds['sstat']}\n")
    print("# 3. GPU metrics (pucm) on allocated nodes:")
    print(f"  {mon_cmds['nvidia_smon']}\n")

    return 0


def demo() -> None:
    """Self-check testing validation, resume filtering, and command generation."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sif = tmp / "test.sif"
        sif.write_text("sif content")
        tile_ckpt = tmp / "tile.pt"
        tile_ckpt.write_text("tile ckpt")
        slide_ckpt = tmp / "slide.pt"
        slide_ckpt.write_text("slide ckpt")
        wsi1 = tmp / "BF-0001.ome.tiff"
        wsi2 = tmp / "BF-0002.ome.tiff"
        wsi1.write_text("wsi one")
        wsi2.write_text("wsi two")

        sif_hash = sha256_file(sif)
        tile_hash = sha256_file(tile_ckpt)
        slide_hash = sha256_file(slide_ckpt)

        shards_tsv = tmp / "shards.tsv"
        rows = [
            {
                "node_shard": "0",
                "gpu_slot": "0",
                "slide_id": "BF-0001",
                "source_path": str(wsi1),
                "weight": "100",
            },
            {
                "node_shard": "0",
                "gpu_slot": "1",
                "slide_id": "BF-0002",
                "source_path": str(wsi2),
                "weight": "100",
            },
        ]
        write_shards_tsv(shards_tsv, rows)

        validated = validate_shards_tsv(shards_tsv)
        assert len(validated) == 2

        out_dir = tmp / "output"
        pending, skipped = filter_resume_shards(validated, out_dir)
        assert len(pending) == 2
        assert len(skipped) == 0

        import h5py
        import numpy as np

        slide_dir = out_dir / "BF-0001"
        qc_dir = slide_dir / "qc"
        qc_dir.mkdir(parents=True)
        (slide_dir / "manifest.json").write_text(json.dumps({
            "status": "success",
            "slide_id": "BF-0001",
            "source": str(wsi1.resolve()),
            "tile_count": 1,
        }))
        (slide_dir / "logs").mkdir()
        (slide_dir / "logs" / "evaluation.log").write_text("ok\n")
        for name in ("original.png", "roi.png", "tile-overlay.png"):
            (qc_dir / name).write_bytes(b"PNG")
        with h5py.File(slide_dir / "tile_features.h5", "w") as handle:
            handle.create_dataset("features", data=np.zeros((1, 384), dtype=np.float32))
            handle.create_dataset("coords", data=np.zeros((1, 2), dtype=np.float32))
        np.save(slide_dir / "slide_embedding.npy", np.zeros(384, dtype=np.float32))

        pending2, skipped2 = filter_resume_shards(validated, out_dir)
        assert len(pending2) == 1
        assert len(skipped2) == 1
        assert skipped2 == ["BF-0001"]
        assert pending2[0]["slide_id"] == "BF-0002"

        compacted = recompact_node_shards(pending2)
        assert compacted[0]["node_shard"] == "0"
        assert compacted[0]["gpu_slot"] == "0"

        cmds = generate_monitor_commands("99999")
        assert "squeue -j 99999" in cmds["squeue"]
        assert "sstat -j 99999.batch" in cmds["sstat"]
        assert "nvidia-smi dmon" in cmds["nvidia_smon"]

    print("Self-check demo() passed successfully.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
        sys.exit(0)
    sys.exit(main())
