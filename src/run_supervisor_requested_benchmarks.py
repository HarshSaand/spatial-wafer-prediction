"""Benchmarks requested in the 7 October review.

Gaussian Process and AdaBoost models predict removal/change (outgoing minus
incoming), then reconstruct the outgoing 88-location thickness profile.  All
reported predictions are out of fold at wafer level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from run_baselines import group_name, load_process_features, load_wafer_arrays
from run_region_aware_fusion import radial_regions, region_metrics
from run_shape_decomposed_fusion import shape_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def prepare_fold(x, removal, train, test, components=20):
    x_scaler = StandardScaler().fit(x[train])
    x_train = x_scaler.transform(x[train])
    x_test = x_scaler.transform(x[test])
    pca = PCA(n_components=min(components, len(train) - 1), random_state=42).fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    y_mean = removal[train].mean(axis=0, keepdims=True)
    y_std = removal[train].std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0
    y_train = (removal[train] - y_mean) / y_std
    return x_train, x_test, y_train, y_mean, y_std, float(pca.explained_variance_ratio_.sum())


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
    process_summary = np.concatenate([
        sequences.mean(axis=1), sequences.std(axis=1),
        sequences.min(axis=1), sequences.max(axis=1),
    ], axis=1)
    x = np.concatenate([spatial, process_summary], axis=1)
    target = np.stack([record["target"] for record in matched])
    removal = target - spatial
    keys = np.asarray([record["experiment_key"] for record in matched])
    lots = np.asarray([record["lot"] for record in matched])
    splits = list(KFold(5, shuffle=True, random_state=args.seed).split(x))

    predictions = {
        "gaussian_process_removal": np.full_like(target, np.nan),
        "adaboost_removal": np.full_like(target, np.nan),
    }
    fold_rows = []
    for fold, (train, test) in enumerate(splits, 1):
        x_train, x_test, y_train, y_mean, y_std, explained = prepare_fold(x, removal, train, test)
        models = {
            "gaussian_process_removal": GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(5.0, length_scale_bounds="fixed")
                + WhiteKernel(.1, noise_level_bounds="fixed"),
                alpha=1e-6, normalize_y=False, optimizer=None,
            ),
            "adaboost_removal": MultiOutputRegressor(
                AdaBoostRegressor(n_estimators=80, learning_rate=.04, loss="square", random_state=args.seed),
                n_jobs=-1,
            ),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            predicted_removal = model.predict(x_test) * y_std + y_mean
            prediction = spatial[test] + predicted_removal
            predictions[name][test] = prediction
            fold_rows.append({
                "model": name, "fold": fold, "test_wafers": len(test),
                "pca_explained_variance": explained, **metrics(target[test], prediction),
            })

    rows = []
    _, regions = radial_regions(coordinates)
    region_rows = []
    for name, prediction in predictions.items():
        rows.append({"model": name, **metrics(target, prediction), **shape_metrics(target, prediction)})
        region_rows.extend(region_metrics(target, prediction, regions, name))
        np.savez_compressed(
            args.output_dir / f"oof_{name}.npz", keys=keys, lots=lots, actual=target,
            predicted=prediction, coordinates=coordinates, regions=regions,
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "requested_benchmark_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "requested_benchmark_folds.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir / "requested_benchmark_regions.csv", index=False)
    metadata = {
        "samples": len(target),
        "input": "88 incoming thickness values plus four summaries for each of 31 process signals",
        "target": "88 outgoing-minus-incoming thickness values",
        "reconstruction": "outgoing thickness equals incoming thickness plus predicted change",
        "validation": "Five-fold shuffled wafer-level out-of-fold validation",
        "preprocessing": "Training-fold standardization followed by 20-component PCA",
        "models": ["Gaussian Process with fixed RBF and white-noise kernel", "Multi-output AdaBoost"],
    }
    (args.output_dir / "requested_benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
