"""Test a domain-informed residual hybrid requested in the 7 October review.

The tree branch receives explicit incoming-wafer radial descriptors and predicts
outgoing-minus-incoming thickness.  Its out-of-fold prediction is averaged
50/50 with the existing deep ensemble using the same wafer folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from run_baselines import group_name, load_process_features, load_wafer_arrays
from run_region_aware_fusion import radial_regions, region_metrics
from run_shape_decomposed_fusion import shape_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deep-oof", type=Path, default=Path(
        "outputs/shape_decomposed_2026-10-07/oof_equal_weight_deep_ensemble.npz"
    ))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metrics(actual, predicted):
    wafer_r2 = np.asarray([r2_score(a, p) for a, p in zip(actual, predicted)])
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2_global": float(r2_score(actual.ravel(), predicted.ravel())),
        "r2_mean_wafer": float(wafer_r2.mean()),
        "r2_median_wafer": float(np.median(wafer_r2)),
        "wafer_r2_positive_fraction": float(np.mean(wafer_r2 > 0)),
    }


def radial_descriptors(spatial, radius, regions):
    normalized_radius = radius / radius.max()
    rows = []
    for wafer in spatial:
        means = [wafer[regions == region].mean() for region in range(3)]
        stds = [wafer[regions == region].std() for region in range(3)]
        quadratic, linear, _ = np.polyfit(normalized_radius, wafer, 2)
        rows.append(means + stds + [means[2] - means[0], means[1] - means[0], linear, quadratic])
    return np.asarray(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", "postox_thickness")
    processes, _ = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", 64
    )
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    sequences = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    target = np.stack([record["target"] for record in matched])
    removal = target - spatial
    keys = np.asarray([record["experiment_key"] for record in matched])
    lots = np.asarray([record["lot"] for record in matched])
    radius, regions = radial_regions(coordinates)
    process_summary = np.concatenate([
        sequences.mean(axis=1), sequences.std(axis=1),
        sequences.min(axis=1), sequences.max(axis=1),
    ], axis=1)
    domain = radial_descriptors(spatial, radius, regions)
    features = np.concatenate([spatial, process_summary, domain], axis=1)
    deep_data = np.load(args.deep_oof)
    deep = deep_data["predicted"]

    tree = np.full_like(target, np.nan)
    for train, test in KFold(5, shuffle=True, random_state=args.seed).split(features):
        model = ExtraTreesRegressor(
            n_estimators=700, min_samples_leaf=2, max_features=.7,
            random_state=args.seed, n_jobs=-1,
        )
        model.fit(features[train], removal[train])
        tree[test] = spatial[test] + model.predict(features[test])
    hybrid = .5 * deep + .5 * tree

    predictions = {
        "deep_ensemble": deep,
        "domain_residual_tree": tree,
        "fixed_domain_residual_hybrid": hybrid,
    }
    rows, region_rows = [], []
    for name, predicted in predictions.items():
        rows.append({"model": name, **metrics(target, predicted), **shape_metrics(target, predicted)})
        region_rows.extend(region_metrics(target, predicted, regions, name))
    pd.DataFrame(rows).to_csv(args.output_dir / "domain_shape_summary.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir / "domain_shape_regions.csv", index=False)
    np.savez_compressed(
        args.output_dir / "oof_fixed_domain_residual_hybrid.npz", keys=keys, lots=lots,
        actual=target, predicted=hybrid, deep_predicted=deep, tree_predicted=tree,
        coordinates=coordinates, radius=radius, regions=regions,
    )
    metadata = {
        "samples": len(target),
        "tree_target": "outgoing minus incoming thickness at 88 locations",
        "domain_features": [
            "centre, middle and edge incoming means", "centre, middle and edge incoming standard deviations",
            "edge minus centre", "middle minus centre", "radial linear coefficient", "radial quadratic coefficient",
        ],
        "hybrid": "Fixed 50% deep ensemble and 50% domain residual tree",
        "validation": "Five-fold wafer-level out-of-fold validation",
    }
    (args.output_dir / "domain_shape_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
