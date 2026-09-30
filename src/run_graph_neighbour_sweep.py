"""Compare graph neighbourhood sizes and uniform versus distance weighting."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_baselines import group_name, load_process_features, load_wafer_arrays
from run_coordinate_fusion import coordinate_features, metric_set, seed_all, train_fold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="postox_thickness")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    records, coordinates = load_wafer_arrays(
        args.data_dir / "Si_Oxide_etch_89_points.csv", args.target
    )
    processes, _ = load_process_features(
        args.data_dir / "Process_data.nc",
        args.data_dir / "Dictionary_process.nc",
        args.time_steps,
    )
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    process = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    target = np.stack([record["target"] for record in matched])
    keys = np.asarray([record["experiment_key"] for record in matched])
    lots = np.asarray([record["lot"] for record in matched])
    coords = coordinate_features(coordinates)
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(spatial))
    models = [
        "graph_uniform_k3", "graph_uniform_k6", "graph_uniform_k10",
        "graph_distance_k3", "graph_distance_k6", "graph_distance_k10",
    ]

    fold_rows = []
    summary_rows = []
    for model_name in models:
        oof = np.full_like(target, np.nan)
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            prediction, epoch, validation_loss = train_fold(
                spatial, process, target, coords, coordinates,
                train_idx, test_idx, model_name, args, fold,
            )
            oof[test_idx] = prediction
            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "test_wafers": len(test_idx),
                "best_epoch": epoch,
                "validation_loss_scaled": validation_loss,
                **metric_set(target[test_idx], prediction),
            })
        summary_rows.append({"model": model_name, **metric_set(target, oof)})
        np.savez_compressed(
            args.output_dir / f"oof_{model_name}.npz",
            keys=keys, lots=lots, actual=target, predicted=oof, coordinates=coordinates,
        )

    summary = pd.DataFrame(summary_rows).sort_values("rmse")
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "graph_neighbour_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "graph_neighbour_summary.csv", index=False)
    metadata = {
        "matched_wafers": len(matched),
        "models": models,
        "neighbour_counts": [3, 6, 10],
        "weighting": ["uniform", "inverse distance normalized per point"],
        "validation": "five fold wafer cross validation with per wafer R squared",
        "seed": args.seed,
    }
    (args.output_dir / "graph_neighbour_metadata.json").write_text(json.dumps(metadata, indent=2))

    labels = [name.replace("graph_", "").replace("_", " ") for name in summary["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].barh(labels[::-1], summary["rmse"][::-1], color="#2f80b7")
    axes[0].set_xlabel("RMSE (µm); lower is better")
    axes[0].set_title("Prediction error")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(labels[::-1], summary["r2_mean_wafer"][::-1], color="#4c956c")
    axes[1].set_xlabel("Mean wafer R²; higher is better")
    axes[1].set_title("Individual wafer spatial fit")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Graph neighbourhood and weighting comparison", fontsize=13)
    fig.savefig(args.output_dir / "graph_neighbour_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
