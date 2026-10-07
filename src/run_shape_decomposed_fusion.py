"""Deep multi-task model separating wafer mean change from spatial shape change."""

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
from run_coordinate_fusion import standardize
from run_region_aware_fusion import overall_metrics, radial_regions, region_metrics


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
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ShapeDecomposedGraphFusionNet(nn.Module):
    def __init__(self, coordinates, neighbours=6):
        super().__init__()
        xy = torch.tensor(coordinates, dtype=torch.float32)
        distance = torch.cdist(xy, xy)
        nearest = distance.argsort(dim=1)[:, 1:neighbours + 1]
        adjacency = torch.zeros(len(xy), len(xy))
        adjacency.scatter_(1, nearest, 1.0 / neighbours)
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Sequential(
            nn.Linear(176, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(48, 24), nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(31, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 24, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(96, 24), nn.GELU(),
        )
        self.mean_head = nn.Sequential(nn.Linear(48, 24), nn.GELU(), nn.Dropout(0.15), nn.Linear(24, 1))
        self.shape_head = nn.Sequential(nn.Linear(48, 48), nn.GELU(), nn.Dropout(0.20), nn.Linear(48, 88))

    def forward(self, spatial, process):
        neighbour_mean = spatial @ self.adjacency.T
        context = torch.cat([
            self.spatial(torch.cat([spatial, neighbour_mean], dim=1)),
            self.temporal(process),
        ], dim=1)
        mean_change = self.mean_head(context)
        shape_change = self.shape_head(context)
        shape_change = shape_change - shape_change.mean(dim=1, keepdim=True)
        return mean_change, shape_change


def train_fold(spatial, process, target, coordinates, train_idx, test_idx, args, fold, shape_weight):
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
    mean_change = residual.mean(axis=1, keepdims=True)
    shape_change = residual - mean_change
    mean_mu = mean_change[fit_idx].mean(axis=0, keepdims=True)
    mean_sd = mean_change[fit_idx].std(axis=0, keepdims=True)
    mean_sd[mean_sd < 1e-8] = 1.0
    shape_mu = shape_change[fit_idx].mean(axis=0, keepdims=True)
    shape_sd = shape_change[fit_idx].std(axis=0, keepdims=True)
    shape_sd[shape_sd < 1e-8] = 1.0

    def targets(indices):
        return (
            torch.tensor((mean_change[indices] - mean_mu) / mean_sd, dtype=torch.float32),
            torch.tensor((shape_change[indices] - shape_mu) / shape_sd, dtype=torch.float32),
        )

    def inputs(s, p):
        return torch.tensor(s, dtype=torch.float32), torch.tensor(p.transpose(0, 2, 1), dtype=torch.float32)

    seed_all(args.seed + fold)
    model = ShapeDecomposedGraphFusionNet(coordinates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    fs, fp = inputs(s_fit, p_fit)
    fm, fshape = targets(fit_idx)
    loader = DataLoader(
        TensorDataset(fs, fp, fm, fshape), batch_size=min(16, len(fit_idx)), shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + fold),
    )
    vs, vp = inputs(s_val, p_val)
    vm, vshape = targets(val_idx)
    best_state, best_loss, best_epoch, wait = None, float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bs, bp, bm, bshape in loader:
            optimizer.zero_grad()
            pm, pshape = model(bs, bp)
            loss = loss_fn(pm, bm) + shape_weight * loss_fn(pshape, bshape)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pm, pshape = model(vs, vp)
            validation_loss = (loss_fn(pm, vm) + shape_weight * loss_fn(pshape, vshape)).item()
        if validation_loss < best_loss - 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_loss, best_epoch, wait = validation_loss, epoch, 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    ts, tp = inputs(s_test, p_test)
    model.eval()
    with torch.no_grad():
        predicted_mean_scaled, predicted_shape_scaled = model(ts, tp)
    predicted_mean = predicted_mean_scaled.numpy() * mean_sd + mean_mu
    predicted_shape = predicted_shape_scaled.numpy() * shape_sd + shape_mu
    predicted_shape -= predicted_shape.mean(axis=1, keepdims=True)
    prediction = spatial[test_idx] + predicted_mean + predicted_shape
    return prediction, best_epoch, best_loss, sum(p.numel() for p in model.parameters())


def shape_metrics(actual, predicted):
    actual_shape = actual - actual.mean(axis=1, keepdims=True)
    predicted_shape = predicted - predicted.mean(axis=1, keepdims=True)
    wafer_shape_r2 = np.asarray([r2_score(a, p) for a, p in zip(actual_shape, predicted_shape)])
    return {
        "shape_rmse": float(np.sqrt(np.mean((actual_shape - predicted_shape) ** 2))),
        "shape_r2_mean_wafer": float(wafer_shape_r2.mean()),
        "shape_r2_median_wafer": float(np.median(wafer_shape_r2)),
        "mean_thickness_rmse": float(np.sqrt(np.mean((actual.mean(axis=1) - predicted.mean(axis=1)) ** 2))),
    }


def plot_comparison(path, actual, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    labels = {
        "flat_mlp_conv1d": "Previous MLP Conv1D",
        "graph_feature_fused": "Existing graph",
        "shape_decomposed_w1": "Mean and shape heads",
        "shape_decomposed_w2": "Shape-weighted heads",
        "equal_weight_deep_ensemble": "Equal-weight deep ensemble",
    }
    colors = ["#777777", "#D95F0E", "#2B8CBE"]
    displayed = ["flat_mlp_conv1d", "shape_decomposed_w2", "equal_weight_deep_ensemble"]
    for name, color in zip(displayed, colors):
        predicted = predictions[name]
        wafer_shape_r2 = []
        actual_shape = actual - actual.mean(axis=1, keepdims=True)
        predicted_shape = predicted - predicted.mean(axis=1, keepdims=True)
        for a, p in zip(actual_shape, predicted_shape):
            wafer_shape_r2.append(r2_score(a, p))
        axes[0].hist(wafer_shape_r2, bins=np.linspace(-2, 1, 25), histtype="step", lw=2, label=labels[name], color=color)
        axes[1].scatter(actual.mean(axis=1), predicted.mean(axis=1), s=20, alpha=0.65, label=labels[name], color=color)
    axes[0].set(xlabel="Wafer-level shape R²", ylabel="Number of wafers", title="Within-wafer shape performance")
    axes[0].legend(frameon=False, fontsize=8)
    low, high = actual.mean(axis=1).min(), actual.mean(axis=1).max()
    axes[1].plot([low, high], [low, high], "k--", lw=1)
    axes[1].set(xlabel="Measured mean thickness (µm)", ylabel="Predicted mean thickness (µm)", title="Wafer mean prediction")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(spatial))

    flat = np.load("outputs/deep_fusion_2026-09-23/oof_wafer_fused_spatial_process.npz")["predicted"]
    existing = np.load("outputs/coordinate_fusion_2026-09-30/oof_graph_feature_fused.npz")["predicted"]
    predictions = {"flat_mlp_conv1d": flat, "graph_feature_fused": existing}
    summaries = [
        {"model": "flat_mlp_conv1d", "parameters": 18976, **overall_metrics(target, flat), **shape_metrics(target, flat)},
        {"model": "graph_feature_fused", "parameters": 23200, **overall_metrics(target, existing), **shape_metrics(target, existing)},
    ]
    fold_rows = []
    for name, shape_weight in [("shape_decomposed_w1", 1.0), ("shape_decomposed_w2", 2.0)]:
        oof = np.full_like(target, np.nan)
        parameter_count = None
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            prediction, epoch, validation_loss, parameter_count = train_fold(
                spatial, process, target, coordinates, train_idx, test_idx, args, fold, shape_weight
            )
            oof[test_idx] = prediction
            fold_rows.append({
                "model": name, "fold": fold, "test_wafers": len(test_idx),
                "best_epoch": epoch, "validation_loss_scaled": validation_loss,
                **overall_metrics(target[test_idx], prediction), **shape_metrics(target[test_idx], prediction),
            })
        predictions[name] = oof
        summaries.append({"model": name, "parameters": parameter_count, **overall_metrics(target, oof), **shape_metrics(target, oof)})
        np.savez_compressed(
            args.output_dir / f"oof_{name}.npz", keys=keys, lots=lots, actual=target,
            predicted=oof, coordinates=coordinates, radius=radius, regions=regions,
        )

    ensemble = 0.5 * predictions["flat_mlp_conv1d"] + 0.5 * predictions["shape_decomposed_w2"]
    predictions["equal_weight_deep_ensemble"] = ensemble
    summaries.append({
        "model": "equal_weight_deep_ensemble", "parameters": 18976 + 24401,
        **overall_metrics(target, ensemble), **shape_metrics(target, ensemble),
    })
    np.savez_compressed(
        args.output_dir / "oof_equal_weight_deep_ensemble.npz", keys=keys, lots=lots,
        actual=target, predicted=ensemble, coordinates=coordinates, radius=radius, regions=regions,
    )

    region_rows = []
    for name, predicted in predictions.items():
        region_rows.extend(region_metrics(target, predicted, regions, name))
    pd.DataFrame(summaries).to_csv(args.output_dir / "shape_model_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "shape_model_fold_metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir / "shape_model_region_metrics.csv", index=False)
    plot_comparison(args.output_dir / "mean_shape_decomposition.png", target, predictions)
    metadata = {
        "matched_wafers": len(matched),
        "architecture": "Six-neighbour graph MLP plus temporal Conv1D with separate mean-change and zero-mean shape-change heads",
        "shape_loss_weights": [1.0, 2.0],
        "ensemble": "Fixed 0.5 previous MLP Conv1D plus 0.5 shape-decomposed graph Conv1D; no test-set weight optimization",
        "validation": "Five-fold out-of-fold wafer validation",
        "seed": args.seed,
    }
    (args.output_dir / "shape_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(region_rows).to_string(index=False))


if __name__ == "__main__":
    main()
