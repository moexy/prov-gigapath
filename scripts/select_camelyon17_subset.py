#!/usr/bin/env python3
"""Select a balanced multi-centre CAMELYON17 subset and emit a download manifest.

Public CC0 bucket, no credentials: https://camelyon-dataset.s3.us-west-2.amazonaws.com/
Selection is deterministic (sorted slide IDs) so the manifest is reproducible.
"""

import argparse
import csv
import re
import urllib.request
from pathlib import Path

BUCKET = "https://camelyon-dataset.s3.us-west-2.amazonaws.com"
FIELDS = ["slide_id", "centre", "label", "size_bytes", "md5", "url"]


def fetch(path):
    with urllib.request.urlopen(f"{BUCKET}/{path}") as response:
        return response.read().decode()


def image_sizes():
    listing = fetch("?list-type=2&prefix=CAMELYON17/images/&max-keys=1000")
    if "<IsTruncated>true" in listing:
        raise RuntimeError("image listing truncated; paginate before trusting sizes")
    return {
        key.rsplit("/", 1)[-1][:-4]: int(size)
        for key, size in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", listing, re.DOTALL)
        if key.endswith(".tif")
    }


def checksums():
    digests = {}
    for line in fetch("CAMELYON17/checksums.md5").splitlines():
        digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        if name.endswith(".tif") and "images/" in name:
            digests[Path(name).stem] = digest
    return digests


def select(per_class, labels):
    rows = [row for row in csv.DictReader(fetch("CAMELYON17/stages.csv").splitlines()) if "node" in row["patient"]]
    groups = {}
    for row in rows:
        if row["stage"] in labels:
            groups.setdefault((row["center"], row["stage"]), []).append(Path(row["patient"]).stem)
    selected = []
    for centre in sorted({centre for centre, _ in groups}):
        for label in labels:
            available = sorted(groups.get((centre, label), []))
            # One node per patient first, so a class is never carried by a couple of patients.
            by_patient = {}
            for slide in available:
                by_patient.setdefault(slide.rsplit("_node_", 1)[0], []).append(slide)
            ordered = [slide for rank in range(5) for _, nodes in sorted(by_patient.items()) if len(nodes) > rank for slide in [nodes[rank]]]
            if len(ordered) < per_class:
                raise RuntimeError(f"centre {centre} has {len(ordered)} '{label}' slides, need {per_class}")
            selected.extend((centre, label, slide) for slide in ordered[:per_class])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=6, help="slides per label per centre")
    parser.add_argument("--labels", nargs="+", default=["negative", "macro"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sizes = image_sizes()
    digests = checksums()
    rows = [
        {
            "slide_id": slide,
            "centre": centre,
            "label": label,
            "size_bytes": sizes[slide],
            "md5": digests[slide],
            "url": f"{BUCKET}/CAMELYON17/images/{slide}.tif",
        }
        for centre, label, slide in select(args.per_class, args.labels)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} slides, {sum(int(r['size_bytes']) for r in rows) / 1e9:.1f} GB -> {args.output}")


if __name__ == "__main__":
    main()
