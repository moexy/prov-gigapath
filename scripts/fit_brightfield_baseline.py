#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def load_brightfield_tiles(results_csv, block_size):
    features = []
    labels = []
    coords = []
    groups = []
    slide_ids = []
    rows = list(csv.DictReader(results_csv.open()))
    included = [row for row in rows if row["status"] == "success" and row["modality"] == "brightfield"]
    if len(included) < 2:
        raise RuntimeError("Need at least two successful brightfield slides")

    for label, row in enumerate(included):
        slide_id = row["slide_id"]
        with h5py.File(Path(row["output"]) / "tile_features.h5", "r") as handle:
            slide_features = handle["features"][:]
            slide_coords = handle["coords"][:]
        if slide_features.shape[0] != slide_coords.shape[0] or slide_features.shape[1] != 384:
            raise RuntimeError(f"Invalid tile arrays for {slide_id}: {slide_features.shape}, {slide_coords.shape}")
        block_xy = np.floor_divide(slide_coords.astype(np.int64), block_size)
        features.append(slide_features)
        coords.append(slide_coords)
        labels.append(np.full(len(slide_features), label, dtype=np.int64))
        groups.extend(f"{slide_id}:{x}:{y}" for x, y in block_xy)
        slide_ids.extend([slide_id] * len(slide_features))

    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(coords),
        np.asarray(groups),
        np.asarray(slide_ids),
        [row["slide_id"] for row in included],
    )


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser(description="Spatially blocked tile-source classifier baselines")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    x, y, coords, groups, slide_ids, classes = load_brightfield_tiles(args.results_csv, args.block_size)
    group_counts = {class_name: len(set(groups[y == index])) for index, class_name in enumerate(classes)}
    if min(group_counts.values()) < args.folds:
        raise RuntimeError(f"Each class needs at least {args.folds} spatial blocks: {group_counts}")

    classifiers = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", dual=False, max_iter=5000),
        ),
    }
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    report = {
        "target": "source_slide_identity",
        "unit": "tile",
        "warning": "One source slide per class. Metrics measure tile-level source discrimination, not biological or patient-level generalization.",
        "brightfield_only": True,
        "n_slides": len(classes),
        "classes": classes,
        "n_tiles": len(y),
        "feature_dimension": int(x.shape[1]),
        "block_size_level0_pixels": args.block_size,
        "spatial_blocks_per_class": group_counts,
        "folds": args.folds,
        "seed": args.seed,
        "classifiers": {},
    }

    for name, estimator in classifiers.items():
        predictions = np.full_like(y, -1)
        fold_ids = np.full_like(y, -1)
        fold_metrics = []
        for fold, (train_indices, test_indices) in enumerate(splitter.split(x, y, groups)):
            model = clone(estimator)
            model.fit(x[train_indices], y[train_indices])
            fold_predictions = model.predict(x[test_indices])
            predictions[test_indices] = fold_predictions
            fold_ids[test_indices] = fold
            fold_metrics.append({"fold": fold, **metrics(y[test_indices], fold_predictions)})
        if np.any(predictions < 0):
            raise RuntimeError(f"Missing out-of-fold predictions for {name}")

        final_model = clone(estimator).fit(x, y)
        joblib.dump(final_model, args.output / f"{name}.joblib")
        aggregate = metrics(y, predictions)
        matrix = confusion_matrix(y, predictions, labels=np.arange(len(classes)))
        report["classifiers"][name] = {
            **aggregate,
            "fold_metrics": fold_metrics,
            "confusion_matrix": matrix.tolist(),
        }

        with open(args.output / f"{name}-predictions.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["slide_id", "x", "y", "spatial_block", "fold", "true_label", "predicted_label"])
            for slide_id, (tile_x, tile_y), group, fold, true, predicted in zip(
                slide_ids, coords, groups, fold_ids, y, predictions
            ):
                writer.writerow([
                    slide_id,
                    float(tile_x),
                    float(tile_y),
                    group,
                    int(fold),
                    classes[int(true)],
                    classes[int(predicted)],
                ])

    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
