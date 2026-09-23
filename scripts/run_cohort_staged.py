#!/usr/bin/env python3
"""Staged cohort processing runner for GigaPath-Flash.

Drives brightfield WSI batches from local machine to Arrhenius cluster,
respecting cluster disk quotas by uploading, processing, downloading,
verifying, and deleting per-batch slides.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def plan_batches(slides, batch_bytes=30e9):
    """Plan batches greedily by file size so each batch stays under batch_bytes.

    slides: list of (path, size_bytes)
    Returns: list of lists of (path, size_bytes)
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for path, size in slides:
        # If adding this slide exceeds batch_bytes and we already have slides in batch,
        # close current batch and start a new one.
        if current_batch and (current_bytes + size > batch_bytes):
            batches.append(current_batch)
            current_batch = []
            current_bytes = 0

        # Oversized single slide still forms its own batch
        current_batch.append((path, size))
        current_bytes += size

    if current_batch:
        batches.append(current_batch)

    return batches


def verify_slide_dir(slide_dir):
    """Verify slide dir contains expected outputs and valid tile_features.h5 shapes."""
    if not slide_dir.is_dir():
        return False
    h5_path = slide_dir / "tile_features.h5"
    if not h5_path.is_file():
        return False
    try:
        import h5py
        with h5py.File(h5_path, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                return False
            feat_shape = handle["features"].shape
            coord_shape = handle["coords"].shape
            if len(feat_shape) != 2 or feat_shape[1] != 384:
                return False
            if len(coord_shape) != 2 or coord_shape[1] != 2:
                return False
            if feat_shape[0] != coord_shape[0]:
                return False
            if feat_shape[0] == 0:
                return False
        return True
    except Exception:
        return False


def run_cmd(cmd, check=True):
    print(f"+ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def append_remote_results(batch_results_path, cohort_csv_path):
    """Append rows from batch results.csv to cohort results.csv."""
    if not batch_results_path.is_file():
        return []

    fields = ["slide_id", "source", "modality", "status", "tile_count", "runtime_seconds", "output", "error"]
    cohort_rows = []
    seen_ids = set()

    if cohort_csv_path.is_file():
        with open(cohort_csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cohort_rows.append(row)
                seen_ids.add(row.get("slide_id"))

    new_rows = []
    with open(batch_results_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = row.get("slide_id")
            if sid not in seen_ids:
                cohort_rows.append(row)
                seen_ids.add(sid)
                new_rows.append(row)

    with open(cohort_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cohort_rows)

    return new_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Staged cohort processing runner for GigaPath-Flash")
    parser.add_argument("--slides", type=Path, default=Path("wsi_2/brightfield"), help="Path to slides dir or glob")
    parser.add_argument("--remote-host", default="arrhenius-1", help="Remote SSH host")
    parser.add_argument("--ssh-socket", default=str(Path.home() / ".ssh/cm-arrhenius-1.sock"), help="SSH control socket")
    parser.add_argument("--remote-root", default="/nobackup/proj/disk/muc/personal/moe/gigapath-runs/wsi-pilot-2026-09-19-f81d8f59", help="Remote run root")
    parser.add_argument("--results-root", type=Path, default=Path("results/cohort"), help="Local results root")
    parser.add_argument("--batch-bytes", type=float, default=30e9, help="Max batch bytes (default 30 GB)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    return parser.parse_args()


def demo():
    """Assert-based self-check covering the batch planner."""
    # 1. Sizes bin under the cap
    slides = [
        (Path(f"slide_{i}.tiff"), 10) for i in range(10)
    ]
    batches = plan_batches(slides, batch_bytes=25)
    for b in batches:
        assert sum(s[1] for s in b) <= 25, f"Batch exceeded cap: {sum(s[1] for s in b)}"

    # 2. Every slide appears exactly once
    all_planned = [s[0] for b in batches for s in b]
    assert len(all_planned) == len(slides), "Slide count mismatch"
    assert set(all_planned) == {s[0] for s in slides}, "Slide identity mismatch"

    # 3. Oversized single slide still forms its own batch
    slides_oversized = [
        (Path("small_1.tiff"), 10),
        (Path("giant.tiff"), 50),
        (Path("small_2.tiff"), 10),
    ]
    batches_ov = plan_batches(slides_oversized, batch_bytes=25)
    assert len(batches_ov) == 3
    assert len(batches_ov[1]) == 1
    assert batches_ov[1][0][0] == Path("giant.tiff")
    all_ov_planned = [s[0] for b in batches_ov for s in b]
    assert len(all_ov_planned) == len(slides_oversized)

    # Empty list
    assert plan_batches([], batch_bytes=100) == []
    print("Self-check demo() passed.")


def main():
    demo()
    args = parse_args()

    slides_dir = args.slides.resolve() if args.slides.exists() else args.slides
    if slides_dir.is_dir():
        all_slides = sorted(
            p for p in slides_dir.iterdir()
            if p.is_file() and p.name.lower().endswith((".tiff", ".tif", ".svs", ".ndpi", ".mrxs"))
        )
    else:
        all_slides = sorted(slides_dir.parent.glob(slides_dir.name))

    if not all_slides:
        print(f"No slides found under {args.slides}")
        return 0

    slide_sizes = [(p, p.stat().st_size) for p in all_slides]
    total_bytes = sum(s for _, s in slide_sizes)
    print(f"Found {len(slide_sizes)} slides totaling {total_bytes / 1e9:.2f} GB")

    # Check resume status for all slides
    args.results_root.mkdir(parents=True, exist_ok=True)
    cohort_csv = args.results_root / "results.csv"

    pending_slides = []
    completed_slides = []
    for p, sz in slide_sizes:
        slide_id = p.stem
        # Handle .ome.tiff double suffix
        if slide_id.endswith(".ome"):
            slide_id = slide_id[:-4]
        slide_out = args.results_root / slide_id
        if verify_slide_dir(slide_out):
            completed_slides.append(p)
        else:
            pending_slides.append((p, sz))

    print(f"Resume status: {len(completed_slides)} completed, {len(pending_slides)} pending")

    batches = plan_batches(pending_slides, batch_bytes=args.batch_bytes)
    print(f"Planned {len(batches)} batches (cap: {args.batch_bytes / 1e9:.1f} GB):")
    for idx, b in enumerate(batches, 1):
        b_bytes = sum(s for _, s in b)
        print(f"  Batch {idx}: {len(b)} slides, {b_bytes / 1e9:.2f} GB ({b[0][0].name} .. {b[-1][0].name})")

    if args.dry_run:
        return 0

    if not batches:
        print("All slides already completed successfully.")
        return 0

    ssh_opts = ["-o", f"ControlPath={args.ssh_socket}"]
    remote_host = args.remote_host
    remote_root = args.remote_root
    input_batch_remote = f"{remote_root}/input-batch"
    results_batch_remote = f"{remote_root}/results-batch"

    overall_failure = False

    for b_idx, batch in enumerate(batches, 1):
        b_bytes = sum(s for _, s in batch)
        print(f"\n=== Processing Batch {b_idx}/{len(batches)}: {len(batch)} slides, {b_bytes / 1e9:.2f} GB ===")

        # 1. Clean remote input-batch and results-batch dirs
        clean_cmd = ["ssh", *ssh_opts, remote_host, f"rm -rf '{input_batch_remote}' '{results_batch_remote}' && mkdir -p '{input_batch_remote}' '{results_batch_remote}'"]
        run_cmd(clean_cmd)

        # 2. rsync batch slides to remote input-batch
        slide_paths = [str(p) for p, _ in batch]
        rsync_ssh = f"ssh -o ControlPath={args.ssh_socket}"
        rsync_up_cmd = ["rsync", "-av", "-e", rsync_ssh, *slide_paths, f"{remote_host}:{input_batch_remote}/"]
        run_cmd(rsync_up_cmd)

        # 3. Submit slurm/evaluate_cohort.sh with sbatch --wait
        sbatch_cmd = [
            "ssh", *ssh_opts, remote_host,
            f"sbatch --wait /nobackup/proj/disk/muc/personal/$USER/gigapath/slurm/evaluate_cohort.sh '{input_batch_remote}' '{results_batch_remote}'"
        ]
        sbatch_res = run_cmd(sbatch_cmd, check=False)
        if sbatch_res.returncode != 0:
            print(f"Warning: sbatch returned {sbatch_res.returncode}; checking per-slide outputs.")

        # 4. rsync the per-slide output dirs back into local results-root.
        #    results.csv is excluded: the cohort CSV is cumulative and merged separately.
        rsync_down_cmd = [
            "rsync", "-av", "--exclude", "results.csv", "-e", rsync_ssh,
            f"{remote_host}:{results_batch_remote}/",
            f"{args.results_root}/"
        ]
        run_cmd(rsync_down_cmd, check=False)

        # 5. Merge this batch's remote results.csv into the cohort results.csv
        temp_batch_csv = args.results_root / f".batch_{b_idx}_results.csv"
        rsync_csv_cmd = [
            "rsync", "-av", "-e", rsync_ssh,
            f"{remote_host}:{results_batch_remote}/results.csv",
            str(temp_batch_csv)
        ]
        run_cmd(rsync_csv_cmd, check=False)
        if temp_batch_csv.is_file():
            append_remote_results(temp_batch_csv, cohort_csv)
            temp_batch_csv.unlink(missing_ok=True)

        # 6. Verify each expected slide dir arrived and its tile_features.h5 opens with valid shapes
        batch_failed_slides = []
        verified_slides_to_delete = []

        for p, _ in batch:
            slide_id = p.stem
            if slide_id.endswith(".ome"):
                slide_id = slide_id[:-4]
            slide_dir = args.results_root / slide_id
            if verify_slide_dir(slide_dir):
                verified_slides_to_delete.append(p.name)
            else:
                batch_failed_slides.append(slide_id)
                overall_failure = True
                print(f"ERROR: Slide {slide_id} failed output verification!")

        # 7. Delete ONLY verified slides from remote input-batch
        if verified_slides_to_delete:
            del_targets = " ".join(f"'{input_batch_remote}/{name}'" for name in verified_slides_to_delete)
            del_cmd = ["ssh", *ssh_opts, remote_host, f"rm -f {del_targets}"]
            run_cmd(del_cmd)
            print(f"Verified and cleaned {len(verified_slides_to_delete)} slides on remote.")

        if batch_failed_slides:
            print(f"Batch {b_idx} had failures: {batch_failed_slides}")

    return 1 if overall_failure else 0


if __name__ == "__main__":
    sys.exit(main())
