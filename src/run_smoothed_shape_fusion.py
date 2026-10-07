"""Controlled incoming-profile smoothing test for the hybrid graph MLP Conv1D model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from run_baselines import group_name, load_process_features, load_wafer_arrays
from run_region_aware_fusion import overall_metrics, radial_regions, region_metrics
from run_shape_decomposed_fusion import shape_metrics, train_fold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="postox_thickness")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothing-strength", type=float, default=0.25)
    return parser.parse_args()


def neighbour_matrix(coordinates, neighbours=6):
    xy = torch.tensor(coordinates, dtype=torch.float32)
    distance = torch.cdist(xy, xy)
    nearest = distance.argsort(dim=1)[:, 1:neighbours + 1]
    adjacency = torch.zeros(len(xy), len(xy))
    adjacency.scatter_(1, nearest, 1.0 / neighbours)
    return adjacency.numpy()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", args.target)
    processes, _ = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", args.time_steps
    )
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    process = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    target = np.stack([record["target"] for record in matched])
    keys = np.asarray([record["experiment_key"] for record in matched])
    lots = np.asarray([record["lot"] for record in matched])
    radius, regions = radial_regions(coordinates)

    adjacency = neighbour_matrix(coordinates)
    neighbour_average = spatial @ adjacency.T
    smoothed = (1.0 - args.smoothing_strength) * spatial + args.smoothing_strength * neighbour_average
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(spatial))
    oof = np.full_like(target, np.nan)
    fold_rows = []
    parameter_count = None
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        prediction, epoch, validation_loss, parameter_count = train_fold(
            spatial, process, target, coordinates, train_idx, test_idx, args, fold,
            shape_weight=2.0, input_spatial=smoothed,
        )
        oof[test_idx] = prediction
        fold_rows.append({
            "model": "smoothed_mean_shape_graph", "fold": fold, "test_wafers": len(test_idx),
            "best_epoch": epoch, "validation_loss_scaled": validation_loss,
            **overall_metrics(target[test_idx], prediction), **shape_metrics(target[test_idx], prediction),
        })

    previous = np.load("outputs/deep_fusion_2026-09-23/oof_wafer_fused_spatial_process.npz")["predicted"]
    ensemble = 0.5 * previous + 0.5 * oof
    summaries = [
        {"model": "smoothed_mean_shape_graph", "parameters": parameter_count,
         **overall_metrics(target, oof), **shape_metrics(target, oof)},
        {"model": "smoothed_equal_weight_ensemble", "parameters": 18976 + parameter_count,
         **overall_metrics(target, ensemble), **shape_metrics(target, ensemble)},
    ]
    region_rows = []
    for name, predicted in [("smoothed_mean_shape_graph", oof), ("smoothed_equal_weight_ensemble", ensemble)]:
        region_rows.extend(region_metrics(target, predicted, regions, name))
    pd.DataFrame(summaries).to_csv(args.output_dir / "smoothing_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "smoothing_fold_metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir / "smoothing_region_metrics.csv", index=False)
    np.savez_compressed(args.output_dir / "oof_smoothed_mean_shape_graph.npz", keys=keys, lots=lots,
                        actual=target, predicted=oof, coordinates=coordinates)
    np.savez_compressed(args.output_dir / "oof_smoothed_equal_weight_ensemble.npz", keys=keys, lots=lots,
                        actual=target, predicted=ensemble, coordinates=coordinates)
    metadata = {
        "architecture": "Dual-input hybrid six-neighbour graph MLP plus temporal 1D CNN with mean and shape output heads",
        "smoothing": "Per-wafer convex blend of raw incoming profile and six-neighbour spatial average",
        "smoothing_strength": args.smoothing_strength,
        "validation": "Five-fold out-of-fold wafer validation; scaling and early stopping fitted within each fold",
        "seed": args.seed,
    }
    (args.output_dir / "smoothing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(region_rows).to_string(index=False))


if __name__ == "__main__":
    main()
