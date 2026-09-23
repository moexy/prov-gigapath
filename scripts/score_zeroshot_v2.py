#!/usr/bin/env python3
"""Unblind protocol-v2 zero-shot predictions and score them against filename-derived labels.

Key chain: blind_id -> BF-XXXX.ome.tiff (.source-map.json) -> original slide file
(matched on OME SizeX/SizeY/PhysicalSizeX, written by --write-key).
Ground truth comes only from the source filename convention
<slide-id>__<stain>__<organ>__<geometry>__<objective>__<uuid8>.ome.tiff.
No diagnosis or grade labels exist for this cohort, so no diagnostic endpoint is scored.
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

V1 = Path("wsi_2/zero-shot-eval-publication-priority-v1")
V2 = Path("wsi_2/zero-shot-eval-publication-priority-v2")
KEY = V2 / "adjudication-key.json"


def ome_meta(path):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - (48 << 20)))
        blob = fh.read()
    i = blob.find(b"<OME")
    if i < 0:
        return None
    head = blob[i : i + 4000].decode("utf8", "replace")
    attr = lambda k: (re.search(r'(?<![A-Za-z])' + k + r'="([^"]+)"', head) or [None, None])[1]
    return (attr("SizeX"), attr("SizeY"), attr("PhysicalSizeX"))


def write_key(source_root):
    source_root = Path(source_root)
    index = collections.defaultdict(list)
    for p in sorted(source_root.rglob("*.ome.tiff")):
        index[ome_meta(p)].append(str(p.relative_to(source_root)))
    blind = json.loads((V1 / ".source-map.json").read_text())
    key = {}
    for blind_id, bf_name in blind.items():
        cands = index.get(ome_meta(Path("wsi_2/brightfield") / bf_name), [])
        if len(cands) != 1:
            raise SystemExit(f"{blind_id}: {len(cands)} source candidates; refuse ambiguous key")
        slide, stain, organ, geometry, objective, _ = Path(cands[0]).name.split("__")
        key[blind_id] = {
            "deidentified_file": bf_name,
            "source_file": cands[0],
            "slide_id": slide,
            "stain": stain,
            "organ": organ,
            "geometry": geometry,
            "objective": objective,
        }
    if len(set(v["source_file"] for v in key.values())) != len(key):
        raise SystemExit("key is not injective")
    KEY.write_text(json.dumps(key, indent=2) + "\n")
    return key


# Preregistered claim-to-label rules. Applied to claim + evidence text, first match wins.
STAIN_RULES = [("ab-pas", r"\bpas\b|periodic acid|alcian"), ("he", r"h&e|h\s*&\s*e|hematoxylin|haematoxylin|eosin")]
ORGAN_EXACT = r"colon|colorect|large intestin|caec|cec"
ORGAN_PARTIAL = r"intestin|bowel|gastrointestin|\bgi\b|gut"
GEOMETRY_EXACT = r"cross[- ]section|transverse"


def text_of(claims):
    if isinstance(claims, dict):
        claims = [claims]
    if not isinstance(claims, list):
        return ""
    keep = [c for c in claims if isinstance(c, dict) and c.get("confidence") != "unsupported"]
    return " ".join(f"{c.get('claim','')} {c.get('evidence','')}" for c in keep).lower()


def score(key):
    frozen = json.loads((V2 / "PREDICTIONS_FROZEN.json").read_text())
    rows = []
    for entry in frozen["predictions"]:
        bid = entry["blind_id"]
        pred = json.loads((V2 / entry["prediction_file"]).read_text())["parsed"]
        truth = key[bid]

        stain_text = text_of(pred.get("stain_family"))
        stain_pred = next(
            (label for label, rx in STAIN_RULES if re.search(rx, stain_text)),
            "abstain" if not stain_text else "other",
        )

        organ_text = text_of(pred.get("tissue_or_organ"))
        organ_pred = (
            "colon" if re.search(ORGAN_EXACT, organ_text)
            else "intestine-unspecified" if re.search(ORGAN_PARTIAL, organ_text)
            else "abstain" if not organ_text else "other"
        )

        geom_text = text_of(pred.get("specimen_morphology"))
        geom_pred = (
            "transverse-cross-section" if re.search(GEOMETRY_EXACT, geom_text)
            else "abstain" if not geom_text else "other"
        )

        rows.append({
            "blind_id": bid,
            "source_file": truth["source_file"],
            "stain_true": truth["stain"], "stain_pred": stain_pred,
            "organ_true": truth["organ"], "organ_pred": organ_pred,
            "geometry_true": truth["geometry"], "geometry_pred": geom_pred,
        })

    stain_hits = sum(r["stain_pred"] == r["stain_true"] for r in rows)
    per_stain = collections.Counter((r["stain_true"], r["stain_pred"]) for r in rows)
    classes = sorted({r["stain_true"] for r in rows})
    recalls = {c: sum(v for (t, p), v in per_stain.items() if t == c and p == c) / sum(r["stain_true"] == c for r in rows) for c in classes}
    report = {
        "n": len(rows),
        "label_source": "source filename convention; no diagnosis or grade labels exist for this cohort",
        "primary_strata_evaluable": {"panda_like_prostate": 0, "ebrains_like_brain": 0},
        "stain_family": {
            "accuracy": stain_hits / len(rows),
            "balanced_accuracy": sum(recalls.values()) / len(recalls),
            "recall_per_class": recalls,
            "confusion": {f"{t}->{p}": v for (t, p), v in sorted(per_stain.items())},
        },
        "tissue_or_organ": {
            "truth": "colon (constant, 91/91)",
            "counts": dict(collections.Counter(r["organ_pred"] for r in rows)),
            "exact_accuracy": sum(r["organ_pred"] == "colon" for r in rows) / len(rows),
            "exact_or_partial": sum(r["organ_pred"] in ("colon", "intestine-unspecified") for r in rows) / len(rows),
        },
        "specimen_morphology": {
            "truth": "transverse-cross-section (constant, 91/91)",
            "counts": dict(collections.Counter(r["geometry_pred"] for r in rows)),
            "accuracy": sum(r["geometry_pred"] == "transverse-cross-section" for r in rows) / len(rows),
        },
    }
    (V2 / "scoring.json").write_text(json.dumps({"report": report, "rows": rows}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="/Volumes/SSD/Imaging/wsi/brightfield")
    ap.add_argument("--write-key", action="store_true", help="rebuild adjudication-key.json from source slides")
    args = ap.parse_args()
    key = write_key(args.source_root) if args.write_key or not KEY.exists() else json.loads(KEY.read_text())
    print(json.dumps(score(key), indent=2))
