"""Diagnose and improve within-wafer shape prediction by radial region.

The experiment follows the 30 September review: quantify centre, middle and
edge errors, produce predicted-versus-measured diagnostics, and compare a
region-aware multi-head graph model with the existing graph-feature model.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_baselines import group_name, load_process_features, load_wafer_arrays
from run_coordinate_fusion import coordinate_features, standardize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="postox_thickness")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def radial_regions(coordinates):
    radius = np.sqrt((coordinates.astype(float) ** 2).sum(axis=1))
    order = np.argsort(radius)
    groups = np.empty(len(radius), dtype=int)
    for region, indices in enumerate(np.array_split(order, 3)):
        groups[indices] = region
    return radius, groups


class RegionAwareGraphFusionNet(nn.Module):
    """Graph fusion with separate centre, middle and edge residual heads."""

    def __init__(self, coordinates, region_ids, neighbours=6):
        super().__init__()
        xy = torch.tensor(coordinates, dtype=torch.float32)
        distance = torch.cdist(xy, xy)
        nearest = distance.argsort(dim=1)[:, 1:neighbours + 1]
        adjacency = torch.zeros(len(xy), len(xy))
        adjacency.scatter_(1, nearest, 1.0 / neighbours)
        self.register_buffer("adjacency", adjacency)

        ordered = np.concatenate([np.flatnonzero(region_ids == i) for i in range(3)])
        inverse = np.argsort(ordered)
        self.register_buffer("inverse_order", torch.tensor(inverse, dtype=torch.long))
        sizes = [int(np.sum(region_ids == i)) for i in range(3)]

        self.spatial = nn.Sequential(
            nn.Linear(176, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(48, 24), nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(31, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 24, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(96, 24), nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Dropout(0.20), nn.Linear(32, size))
            for size in sizes
        ])

    def forward(self, spatial, process):
        neighbour_mean = spatial @ self.adjacency.T
        context = torch.cat([
            self.spatial(torch.cat([spatial, neighbour_mean], dim=1)),
            self.temporal(process),
        ], dim=1)
        grouped = torch.cat([head(context) for head in self.heads], dim=1)
        return grouped[:, self.inverse_order]


def elementwise_smooth_l1(predicted, actual, beta=0.5):
    difference = torch.abs(predicted - actual)
    return torch.where(
        difference < beta,
        0.5 * difference.square() / beta,
        difference - 0.5 * beta,
    )


def train_fold(spatial, process, target, coordinates, regions, train_idx, test_idx, args, fold, edge_weight):
    rng = np.random.default_rng(args.seed + fold)
    shuffled = rng.permutation(train_idx)
    n_val = max(8, round(0.15 * len(train_idx)))
    val_idx, fit_idx = shuffled[:n_val], shuffled[n_val:]

    s_fit, s_val, s_test = standardize(
        spatial[fit_idx], spatial[fit_idx], spatial[val_idx], spatial[test_idx]
    )
    p_fit, p_val, p_test = standardize(
        process[fit_idx], process[fit_idx], process[val_idx], process[test_idx], axes=(0, 1)
    )
    residual = target - spatial
    residual_mean = residual[fit_idx].mean(axis=0, keepdims=True)
    residual_std = residual[fit_idx].std(axis=0, keepdims=True)
    residual_std[residual_std < 1e-8] = 1.0
    r_fit = (residual[fit_idx] - residual_mean) / residual_std
    r_val = (residual[val_idx] - residual_mean) / residual_std

    def tensors(s, p, r=None):
        values = [
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(p.transpose(0, 2, 1), dtype=torch.float32),
        ]
        if r is not None:
            values.append(torch.tensor(r, dtype=torch.float32))
        return values

    seed_all(args.seed + fold)
    model = RegionAwareGraphFusionNet(coordinates, regions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    point_weights = torch.ones(88, dtype=torch.float32)
    point_weights[torch.tensor(regions == 2)] = edge_weight
    point_weights /= point_weights.mean()

    loader = DataLoader(
        TensorDataset(*tensors(s_fit, p_fit, r_fit)),
        batch_size=min(16, len(fit_idx)),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + fold),
    )
    vs, vp, vr = tensors(s_val, p_val, r_val)
    best_state, best_loss, best_epoch, wait = None, float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bs, bp, by in loader:
            optimizer.zero_grad()
            loss = (elementwise_smooth_l1(model(bs, bp), by) * point_weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = (elementwise_smooth_l1(model(vs, vp), vr) * point_weights).mean().item()
        if val_loss < best_loss - 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_loss, best_epoch, wait = val_loss, epoch, 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    ts, tp = tensors(s_test, p_test)
    model.eval()
    with torch.no_grad():
        predicted_residual = model(ts, tp).numpy() * residual_std + residual_mean
    prediction = spatial[test_idx] + predicted_residual
    return prediction, best_epoch, best_loss, sum(p.numel() for p in model.parameters())


def overall_metrics(actual, predicted):
    wafer_r2 = np.asarray([r2_score(a, p) for a, p in zip(actual, predicted)])
    return {
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mae": mean_absolute_error(actual, predicted),
        "r2_global": r2_score(actual.ravel(), predicted.ravel()),
        "r2_mean_wafer": wafer_r2.mean(),
        "r2_median_wafer": np.median(wafer_r2),
        "wafer_r2_positive_fraction": np.mean(wafer_r2 > 0),
    }


def region_metrics(actual, predicted, regions, model_name):
    names = ["centre", "middle", "edge"]
    rows = []
    for region, name in enumerate(names):
        mask = regions == region
        region_wafer_r2 = np.asarray([r2_score(a[mask], p[mask]) for a, p in zip(actual, predicted)])
        error = predicted[:, mask] - actual[:, mask]
        rows.append({
            "model": model_name,
            "region": name,
            "locations": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mean_bias": float(np.mean(error)),
            "mean_wafer_r2_within_region": float(region_wafer_r2.mean()),
            "median_wafer_r2_within_region": float(np.median(region_wafer_r2)),
        })
    return rows


def diagnostic_figure(path, actual, predictions, coordinates, regions):
    names = ["centre", "middle", "edge"]
    colors = ["#2B8CBE", "#41AB5D", "#E6550D"]
    radius = np.sqrt((coordinates.astype(float) ** 2).sum(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    model_name = "region_heads_edge_weighted"
    predicted = predictions[model_name]
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    for region, name in enumerate(names):
        mask = regions == region
        axes[0].scatter(actual[:, mask], predicted[:, mask], s=13, alpha=0.48, color=colors[region], label=name)
    axes[0].plot([low, high], [low, high], color="black", lw=1.2, linestyle="--")
    axes[0].set(xlabel="Measured outgoing thickness (µm)", ylabel="Predicted outgoing thickness (µm)", title="Out-of-fold predicted versus measured")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18)

    for label, values, color in [
        ("Existing graph", predictions["graph_feature_fused"], "#777777"),
        ("Region-aware graph", predicted, "#1F4E78"),
    ]:
        point_mae = np.mean(np.abs(values - actual), axis=0)
        order = np.argsort(radius)
        axes[1].plot(radius[order], point_mae[order], marker="o", ms=3.5, lw=1.4, label=label, color=color)
    axes[1].set(xlabel="Radial distance from wafer centre", ylabel="Mean absolute error (µm)", title="Error as a function of radius")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
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
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(spatial))

    existing = np.load("outputs/coordinate_fusion_2026-09-30/oof_graph_feature_fused.npz")
    predictions = {"graph_feature_fused": existing["predicted"]}
    summary_rows = [{"model": "graph_feature_fused", "parameters": 23200, **overall_metrics(target, existing["predicted"])}]
    fold_rows = []

    for model_name, edge_weight in [("region_heads", 1.0), ("region_heads_edge_weighted", 1.5)]:
        oof = np.full_like(target, np.nan)
        parameter_count = None
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            prediction, epoch, val_loss, parameter_count = train_fold(
                spatial, process, target, coordinates, regions, train_idx, test_idx, args, fold, edge_weight
            )
            oof[test_idx] = prediction
            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "test_wafers": len(test_idx),
                "best_epoch": epoch,
                "validation_loss_scaled": val_loss,
                **overall_metrics(target[test_idx], prediction),
            })
        predictions[model_name] = oof
        summary_rows.append({"model": model_name, "parameters": parameter_count, **overall_metrics(target, oof)})
        np.savez_compressed(
            args.output_dir / f"oof_{model_name}.npz",
            keys=keys, lots=lots, actual=target, predicted=oof,
            coordinates=coordinates, radius=radius, regions=regions,
        )

    region_rows = []
    for model_name, predicted in predictions.items():
        region_rows.extend(region_metrics(target, predicted, regions, model_name))

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "region_model_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "region_model_fold_metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir / "region_error_metrics.csv", index=False)
    diagnostic_figure(args.output_dir / "wafer_region_diagnostics.png", target, predictions, coordinates, regions)

    lot_counts = pd.Series(lots).value_counts().sort_index().astype(int).to_dict()
    metadata = {
        "matched_wafers": len(matched),
        "lot_counts": {str(key): value for key, value in lot_counts.items()},
        "region_definition": "Equal-count radial groups based on fixed coordinate radius",
        "region_location_counts": {name: int(np.sum(regions == i)) for i, name in enumerate(["centre", "middle", "edge"])},
        "edge_loss_weight": 1.5,
        "validation": "Five-fold out-of-fold wafer validation; preprocessing fitted within each training fold",
        "seed": args.seed,
    }
    (args.output_dir / "region_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(region_rows).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
