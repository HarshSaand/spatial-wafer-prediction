"""Evaluate a fixed deep learning and Extra Trees hybrid without test-set tuning.

The deep component is the existing out-of-fold MLP/Conv1D plus graph/Conv1D
ensemble.  The tree component is refit on exactly the same five wafer folds.
A 50/50 average is declared in advance; additional weights are reported only
as a sensitivity analysis and are not used to select the recommended result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", "postox_thickness")
    processes, _ = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", 64
    )
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    process_sequence = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    process_summary = np.concatenate([
        process_sequence.mean(axis=1), process_sequence.std(axis=1),
        process_sequence.min(axis=1), process_sequence.max(axis=1),
    ], axis=1)
    features = np.concatenate([spatial, process_summary], axis=1)
    target = np.stack([record["target"] for record in matched])
    keys = np.asarray([record["experiment_key"] for record in matched])
    lots = np.asarray([record["lot"] for record in matched])

    deep_data = np.load(args.deep_oof)
    if not np.array_equal(keys.astype(str), deep_data["keys"].astype(str)):
        raise ValueError("Deep OOF wafer order does not match the public dataset")
    deep = deep_data["predicted"]

    splits = list(KFold(5, shuffle=True, random_state=args.seed).split(features))
    tree = np.full_like(target, np.nan)
    fold_rows = []
    for fold, (train, test) in enumerate(splits, 1):
        model = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=.7,
            random_state=args.seed, n_jobs=-1,
        )
        model.fit(features[train], target[train])
        tree[test] = model.predict(features[test])
        hybrid_fold = .5 * deep[test] + .5 * tree[test]
        fold_rows.append({"fold": fold, "test_wafers": len(test), **metrics(target[test], hybrid_fold)})

    predictions = {
        "deep_ensemble": deep,
        "extra_trees": tree,
        "fixed_50_50_hybrid": .5 * deep + .5 * tree,
    }
    rows = []
    for name, predicted in predictions.items():
        rows.append({"model": name, **metrics(target, predicted), **shape_metrics(target, predicted)})

    sensitivity_rows = []
    for deep_weight in [0.0, .25, .5, .75, 1.0]:
        predicted = deep_weight * deep + (1 - deep_weight) * tree
        sensitivity_rows.append({"deep_weight": deep_weight, "tree_weight": 1 - deep_weight, **metrics(target, predicted)})

    _, regions = radial_regions(coordinates)
    region_rows = []
    for name, predicted in predictions.items():
        region_rows.extend(region_metrics(target, predicted, regions, name))

    pd.DataFrame(rows).to_csv(args.output_dir / "hybrid_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "hybrid_fold_metrics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "hybrid_weight_sensitivity.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir / "hybrid_region_metrics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "oof_fixed_50_50_hybrid.npz", keys=keys, lots=lots,
        actual=target, predicted=predictions["fixed_50_50_hybrid"],
        deep_predicted=deep, tree_predicted=tree, coordinates=coordinates, regions=regions,
    )

    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4), constrained_layout=True)
    axes[0].bar(summary["model"], summary["r2_mean_wafer"], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_ylabel("Mean individual wafer R²")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=.2)
    sensitivity = pd.DataFrame(sensitivity_rows)
    axes[1].plot(sensitivity["deep_weight"], sensitivity["r2_mean_wafer"], marker="o", color="#1F4E78")
    axes[1].axvline(.5, color="#777777", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Deep model weight", ylabel="Mean individual wafer R²", title="Weight sensitivity")
    axes[1].grid(alpha=.2)
    fig.savefig(args.output_dir / "deep_tree_hybrid_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "samples": len(target),
        "validation": "Five-fold shuffled wafer-level out-of-fold validation",
        "recommended_weight": "Fixed 50% deep ensemble and 50% Extra Trees; not optimized on test folds",
        "deep_model": "MLP/Conv1D plus mean-shape graph MLP/Conv1D ensemble",
        "tree_model": "Extra Trees on 88 spatial values plus mean, standard deviation, minimum and maximum of 31 process signals",
        "caution": "Public plasma-etching development data; transfer to internal CMP data remains to be validated",
    }
    (args.output_dir / "hybrid_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nWeight sensitivity")
    print(pd.DataFrame(sensitivity_rows).to_string(index=False))


if __name__ == "__main__":
    main()
