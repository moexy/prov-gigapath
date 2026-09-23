#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def make_classifiers():
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        ),
    }


def parse_adjudication_key(key_path):
    raw = json.loads(Path(key_path).read_text())
    key_mapping = {}
    for entry in raw.values():
        filename = entry.get("deidentified_file", "")
        stem = filename.replace(".ome.tiff", "").replace(".tiff", "").replace(".tif", "")
        if stem:
            key_mapping[stem] = entry["stain"]
    return key_mapping


def load_cohort_data(results_root, key_path, level):
    results_root = Path(results_root)
    key_mapping = parse_adjudication_key(key_path)

    slide_embeddings = []
    slide_labels = []
    slide_ids_slide_level = []

    tile_features = []
    tile_labels = []
    tile_groups = []

    skipped_count = 0
    loaded_slides = 0

    for stem, stain in sorted(key_mapping.items()):
        slide_dir = results_root / stem
        emb_file = slide_dir / "slide_embedding.npy"
        tile_file = slide_dir / "tile_features.h5"

        need_slide = level in ("slide", "both")
        need_tile = level in ("tile", "both")

        slide_ok = emb_file.is_file() if need_slide else True
        tile_ok = tile_file.is_file() if need_tile else True

        if not (slide_dir.is_dir() and slide_ok and tile_ok):
            skipped_count += 1
            continue

        loaded_slides += 1

        if need_slide:
            emb = np.load(emb_file).astype(np.float32)
            slide_embeddings.append(emb)
            slide_labels.append(stain)
            slide_ids_slide_level.append(stem)

        if need_tile:
            with h5py.File(tile_file, "r") as handle:
                tf = handle["features"][:].astype(np.float32)
            tile_features.append(tf)
            tile_labels.extend([stain] * len(tf))
            tile_groups.extend([stem] * len(tf))

    classes = sorted(list({*slide_labels, *tile_labels}))
    if loaded_slides < 10:
        raise RuntimeError(f"Fewer than 10 slides available (loaded {loaded_slides}, need >= 10).")
    if len(classes) < 2:
        raise RuntimeError(f"Need at least two stain classes, found {len(classes)}: {classes}")

    res = {
        "classes": classes,
        "loaded_slides": loaded_slides,
        "skipped_count": skipped_count,
    }

    if level in ("slide", "both"):
        res["slide_x"] = np.stack(slide_embeddings)
        res["slide_y"] = np.asarray(slide_labels)
        res["slide_ids"] = np.asarray(slide_ids_slide_level)

    if level in ("tile", "both"):
        res["tile_x"] = np.concatenate(tile_features)
        res["tile_y"] = np.asarray(tile_labels)
        res["tile_groups"] = np.asarray(tile_groups)

    return res


def run_slide_level(x, y, slide_ids, classes, folds, seed, output_dir=None):
    label_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_int[lbl] for lbl in y], dtype=np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    models = make_classifiers()
    results = {}

    for name, estimator in models.items():
        preds = np.full_like(y_int, -1)
        for train_idx, test_idx in splitter.split(x, y_int):
            m = clone(estimator).fit(x[train_idx], y_int[train_idx])
            preds[test_idx] = m.predict(x[test_idx])

        if np.any(preds < 0):
            raise RuntimeError(f"Missing out-of-fold predictions for slide {name}")

        final_m = clone(estimator).fit(x, y_int)
        if output_dir:
            joblib.dump(final_m, Path(output_dir) / f"slide_{name}.joblib")

        int_labels = list(range(len(classes)))
        results[name] = compute_metrics(y_int, preds, int_labels)

    return results


def run_tile_level(x, y, groups, classes, folds, seed, output_dir=None):
    label_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_int[lbl] for lbl in y], dtype=np.int64)

    # Unique slide groups and their slide-level label
    unique_groups, group_first_idx = np.unique(groups, return_index=True)
    slide_labels_int = y_int[group_first_idx]

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    # Verify no slide appears in both train and test across any fold
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y_int, groups=groups)):
        train_slides = set(groups[train_idx])
        test_slides = set(groups[test_idx])
        overlap = train_slides.intersection(test_slides)
        assert len(overlap) == 0, f"Fold {fold} has slide leakage across train/test: {overlap}"

    models = make_classifiers()
    results = {}

    for name, estimator in models.items():
        tile_preds = np.full_like(y_int, -1)
        for train_idx, test_idx in splitter.split(x, y_int, groups=groups):
            m = clone(estimator).fit(x[train_idx], y_int[train_idx])
            tile_preds[test_idx] = m.predict(x[test_idx])

        if np.any(tile_preds < 0):
            raise RuntimeError(f"Missing out-of-fold predictions for tile {name}")

        final_m = clone(estimator).fit(x, y_int)
        if output_dir:
            joblib.dump(final_m, Path(output_dir) / f"tile_{name}.joblib")

        int_labels = list(range(len(classes)))
        tile_metrics = compute_metrics(y_int, tile_preds, int_labels)

        # Slide-level majority vote over out-of-fold tile predictions
        slide_voted_preds = []
        slide_true = []
        for g in unique_groups:
            mask = groups == g
            slide_tile_preds = tile_preds[mask]
            # Majority vote (break ties consistently)
            counts = np.bincount(slide_tile_preds, minlength=len(classes))
            voted = int(np.argmax(counts))
            slide_voted_preds.append(voted)
            slide_true.append(y_int[mask][0])

        slide_vote_metrics = compute_metrics(np.array(slide_true), np.array(slide_voted_preds), int_labels)

        results[name] = {
            "tile": tile_metrics,
            "slide_majority_vote": slide_vote_metrics,
        }

    return results


def run_probe(results_root, key_path, output_dir, folds=5, seed=42, level="both"):
    data = load_cohort_data(results_root, key_path, level)
    classes = data["classes"]
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    report = {
        "target": "stain",
        "classes": classes,
        "n_slides_loaded": data["loaded_slides"],
        "n_slides_skipped": data["skipped_count"],
        "folds": folds,
        "seed": seed,
        "level": level,
        "results": {},
    }

    if level in ("slide", "both"):
        report["results"]["slide"] = run_slide_level(
            data["slide_x"],
            data["slide_y"],
            data["slide_ids"],
            classes,
            folds,
            seed,
            output_dir=out_path,
        )

    if level in ("tile", "both"):
        report["results"]["tile"] = run_tile_level(
            data["tile_x"],
            data["tile_y"],
            data["tile_groups"],
            classes,
            folds,
            seed,
            output_dir=out_path,
        )

    if out_path:
        (out_path / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    return report


def demo():
    print("Running self-check demo with synthetic separable slide/tile embeddings...")
    rng = np.random.default_rng(42)
    tiles_per_slide = 50
    dim = 384
    folds = 5

    # 10 slides class 0 (e.g. 'ab-pas'), 10 slides class 1 (e.g. 'he')
    slide_classes = [0] * 10 + [1] * 10
    classes = ["ab-pas", "he"]

    # Signal lives on a shared random direction spanning many dimensions; a single
    # informative dimension out of 384 is unlearnable from 16 training slides.
    direction = rng.normal(size=dim)
    direction[64:] = 0.0
    direction /= np.linalg.norm(direction)

    tile_feats = []
    tile_labels = []
    tile_groups = []

    slide_embs = []
    slide_labels = []
    slide_ids = []

    for i, cls in enumerate(slide_classes):
        s_id = f"BF-{i+1:04d}"
        center = (direction * (3.0 if cls == 1 else -3.0)).astype(np.float32)

        # Slide embedding
        s_emb = center + rng.normal(0, 0.2, size=dim).astype(np.float32)
        slide_embs.append(s_emb)
        slide_labels.append(classes[cls])
        slide_ids.append(s_id)

        # Tile features
        t_feat = center + rng.normal(0, 0.5, size=(tiles_per_slide, dim)).astype(np.float32)
        tile_feats.append(t_feat)
        tile_labels.extend([classes[cls]] * tiles_per_slide)
        tile_groups.extend([s_id] * tiles_per_slide)

    x_tile = np.concatenate(tile_feats)
    y_tile = np.asarray(tile_labels)
    groups = np.asarray(tile_groups)

    x_slide = np.stack(slide_embs)
    y_slide = np.asarray(slide_labels)

    # 1. Slide level
    slide_res = run_slide_level(x_slide, y_slide, np.asarray(slide_ids), classes, folds=folds, seed=42)
    assert slide_res["logistic_regression"]["balanced_accuracy"] > 0.8, (
        f"Slide level logistic regression balanced accuracy too low: {slide_res['logistic_regression']['balanced_accuracy']}"
    )

    # 2. Tile level
    tile_res = run_tile_level(x_tile, y_tile, groups, classes, folds=folds, seed=42)
    assert tile_res["logistic_regression"]["tile"]["balanced_accuracy"] > 0.8, (
        f"Tile level logistic regression balanced accuracy too low: {tile_res['logistic_regression']['tile']['balanced_accuracy']}"
    )
    assert tile_res["logistic_regression"]["slide_majority_vote"]["balanced_accuracy"] > 0.8, (
        f"Slide majority vote balanced accuracy too low: {tile_res['logistic_regression']['slide_majority_vote']['balanced_accuracy']}"
    )

    # 3. Explicit check on StratifiedGroupKFold non-leakage
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    checked_folds = 0
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x_tile, y_tile, groups=groups)):
        train_s = set(groups[train_idx])
        test_s = set(groups[test_idx])
        assert len(train_s & test_s) == 0, f"Slide overlap detected in fold {fold}: {train_s & test_s}"
        assert len(test_s) > 0, f"Empty test fold {fold}"
        checked_folds += 1
    assert checked_folds == folds, f"Expected {folds} folds checked, got {checked_folds}"

    print(json.dumps({"demo": "success", "slide_metrics": slide_res, "tile_metrics": tile_res}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Stain-label linear probe over GigaPath slide/tile embeddings")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Root directory containing per-slide subdirectories (BF-00NN)")
    parser.add_argument("--key", type=Path,
                        default=Path("wsi_2/zero-shot-eval-publication-priority-v2/adjudication-key.json"),
                        help="Path to adjudication-key.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory to save metrics.json and fitted models")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--level", choices=["slide", "tile", "both"], default="both",
                        help="Evaluation level: slide, tile, or both")
    args = parser.parse_args()

    if args.results_root is None:
        demo()
        sys.exit(0)

    report = run_probe(
        results_root=args.results_root,
        key_path=args.key,
        output_dir=args.output,
        folds=args.folds,
        seed=args.seed,
        level=args.level,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
