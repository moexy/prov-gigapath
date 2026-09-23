#!/usr/bin/env python3
from __future__ import annotations
"""Plan deterministic, maximally packed node shards for parallel WSI inference."""

import argparse
import csv
import heapq
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.wsi_paths import WSI_SUFFIXES, slide_id_from_path

TSV_FIELDS = ["node_shard", "gpu_slot", "slide_id", "source_path", "weight"]


def discover_slides(slides_arg: Path) -> list[Path]:
    if slides_arg.is_file():
        # Either a single WSI file or a manifest/text file listing slide paths
        if slides_arg.name.lower().endswith(WSI_SUFFIXES):
            return [slides_arg.resolve()]
        # Read lines from text file or csv/tsv
        paths = []
        with open(slides_arg, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (slides_arg.parent / p).resolve()
                if not p.exists():
                    raise FileNotFoundError(f"Slide file not found: {p}")
                if p.is_file() and p.name.lower().endswith(WSI_SUFFIXES):
                    paths.append(p)
        return sorted(paths)

    if slides_arg.is_dir():
        excluded = set()
        receipt_path = slides_arg / "LOCAL_VALIDATION_RECEIPT.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            excluded = {
                str(entry["file"])
                for entry in receipt.get("slides", [])
                if entry.get("modality") == "fluorescence"
            }
        return sorted(
            path.resolve()
            for path in slides_arg.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(WSI_SUFFIXES)
            and str(path.relative_to(slides_arg)) not in excluded
        )

    raise FileNotFoundError(f"Input path does not exist: {slides_arg}")


def get_slide_weight(slide_path: Path, slide_id: str, results_root: Path | None) -> int:
    if results_root is not None:
        manifest_path = results_root / slide_id / "manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                if manifest.get("status") == "success" and "tile_count" in manifest:
                    return int(manifest["tile_count"])
            except Exception:
                pass
    return slide_path.stat().st_size


def plan_shards(
    slides: list[tuple[str, Path, int]],
    gpus_per_node: int = 4,
    nodes: int | None = None,
) -> list[dict]:
    if gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be positive")
    if not slides:
        return []

    # Validate uniqueness of slide IDs
    seen_ids = set()
    for sid, spath, _ in slides:
        if sid in seen_ids:
            raise ValueError(f"Duplicate slide ID detected: {sid}")
        seen_ids.add(sid)

    n_slides = len(slides)
    default_nodes = math.ceil(n_slides / gpus_per_node)
    if nodes is not None and nodes <= 0:
        raise ValueError("nodes must be positive")
    total_nodes = default_nodes if nodes is None else min(nodes, n_slides)
    total_workers = total_nodes * gpus_per_node

    if total_workers < n_slides:
        raise ValueError(
            f"Not enough GPU slots ({total_workers}) for {n_slides} slides across {total_nodes} nodes with {gpus_per_node} GPUs/node."
        )

    # Sort slides descending by weight (Longest Processing Time first)
    # Tie-break by slide_id for determinism
    sorted_slides = sorted(slides, key=lambda x: (x[2], x[0]), reverse=True)

    # Each worker is identified by (node_shard, gpu_slot)
    # Heap stores: (current_weight, count_assigned, node_shard, gpu_slot, assignments)
    # We maintain max capacity of 1 slide per worker slot if slots >= slides,
    # or general greedy partition. Per plan: each worker takes at most 1 slide (or balanced).
    # Since total_workers >= n_slides and slots per worker is 1 for cohort (e.g. 91 slides on 92 slots):
    # If n_slides <= total_workers, each worker gets at most ceil(n_slides / total_workers) = 1 slide!
    # Let's support arbitrary packing:
    # Priority queue of workers: (load, node_shard, gpu_slot)
    worker_heap = []
    worker_slides = {}
    for node in range(total_nodes):
        for slot in range(gpus_per_node):
            heapq.heappush(worker_heap, (0, node, slot))
            worker_slides[(node, slot)] = []

    for sid, spath, weight in sorted_slides:
        load, node, slot = heapq.heappop(worker_heap)
        worker_slides[(node, slot)].append((sid, spath, weight))
        heapq.heappush(worker_heap, (load + weight, node, slot))

    # Produce rows sorted deterministically by (node_shard, gpu_slot, slide_id)
    rows = []
    for node in range(total_nodes):
        for slot in range(gpus_per_node):
            for sid, spath, weight in worker_slides[(node, slot)]:
                rows.append({
                    "node_shard": node,
                    "gpu_slot": slot,
                    "slide_id": sid,
                    "source_path": str(spath),
                    "weight": weight,
                })

    # Sort final rows by node_shard, gpu_slot
    rows.sort(key=lambda r: (r["node_shard"], r["gpu_slot"], r["slide_id"]))
    return rows


def write_shards_tsv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def self_check() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_counts = [1, 4, 5, 91, 1001]
        for count in test_counts:
            items = []
            for i in range(1, count + 1):
                name = f"slide_{i:04d}.ome.tiff" if i % 2 == 0 else f"slide_{i:04d}.svs"
                sp = tmp_path / name
                sid = slide_id_from_path(sp)
                items.append((sid, sp, 1000 + (i * 37) % 500))

            rows = plan_shards(items, gpus_per_node=4)

            # Check 1: exactly count rows
            assert len(rows) == count, f"Expected {count} rows, got {len(rows)}"

            # Check 2: unique slide_ids
            row_ids = [r["slide_id"] for r in rows]
            assert len(set(row_ids)) == count, f"Duplicate slide_ids in output for count {count}"

            # Check 3: gpu_slot only in 0..3
            slots = {r["gpu_slot"] for r in rows}
            assert slots.issubset({0, 1, 2, 3}), f"Invalid slots: {slots}"

            # Check 4: node_shard count == ceil(count / 4)
            expected_shards = math.ceil(count / 4)
            node_shards = {r["node_shard"] for r in rows}
            assert len(node_shards) == expected_shards, (
                f"Expected {expected_shards} node shards, got {len(node_shards)}"
            )
            assert max(node_shards) == expected_shards - 1
            assert min(node_shards) == 0

            # Check 5: no worker has > 1 slide when total_workers >= count
            worker_allocations = {}
            for r in rows:
                key = (r["node_shard"], r["gpu_slot"])
                worker_allocations[key] = worker_allocations.get(key, 0) + 1
            assert all(v == 1 for v in worker_allocations.values()), (
                f"Slot received multiple slides unexpectedly when total_workers >= count for {count}"
            )

            # Check 6: for 91 slides, exactly 23 shards and 91 rows
            if count == 91:
                assert expected_shards == 23
                assert len(rows) == 91

            tsv_path = tmp_path / f"shards_{count}.tsv"
            write_shards_tsv(tsv_path, rows)
            with open(tsv_path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
                assert header == TSV_FIELDS, f"TSV header mismatch: {header} vs {TSV_FIELDS}"
                lines = [line.strip().split("\t") for line in f if line.strip()]
                assert len(lines) == count
                for line in lines:
                    assert len(line) == 5

    print("Self-check passed successfully.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slides",
        type=Path,
        default=None,
        help="Path to WSI file, directory containing WSIs, or text file listing paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="Number of GPUs per node (default: 4)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Total node shards to plan (default: ceil(slide_count / gpus_per_node))",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional root directory of prior run containing slide_id/manifest.json for tile count weights",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run self-check assertions and exit",
    )

    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if not args.slides or not args.output:
        parser.error("--slides and --output are required unless --self-check is passed")

    slide_paths = discover_slides(args.slides)
    if not slide_paths:
        raise RuntimeError(f"No valid WSI files found at {args.slides}")

    items = []
    for sp in slide_paths:
        sid = slide_id_from_path(sp)
        weight = get_slide_weight(sp, sid, args.results_root)
        items.append((sid, sp, weight))

    rows = plan_shards(items, gpus_per_node=args.gpus_per_node, nodes=args.nodes)
    write_shards_tsv(args.output, rows)
    print(f"Wrote {len(rows)} slide shard assignments to {args.output}")


if __name__ == "__main__":
    main()
