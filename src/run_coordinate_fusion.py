"""Coordinate-aware phase-one benchmark for irregular wafer measurements.

Each of the 88 points is represented by thickness and its normalized x, y and
radial position. A shared point encoder preserves geometry without pretending
that the irregular samples form a regular image. The process branch remains a
temporal Conv1D encoder. The decoder predicts one residual per physical point.
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


def coordinate_features(coordinates):
    xy = coordinates.astype(np.float32)
    scale = np.abs(xy).max(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    xy = xy / scale
    radius = np.sqrt((xy ** 2).sum(axis=1, keepdims=True))
    return np.concatenate([xy, radius], axis=1).astype(np.float32)


class CoordinateFusionNet(nn.Module):
    def __init__(self, fused=True):
        super().__init__()
        self.fused = fused
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU()
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(31, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 24, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(96, 24), nn.GELU(),
        )
        # local point embedding + global mean/max spatial context + process context
        decoder_width = 16 + 32 + (24 if fused else 0)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_width, 40), nn.GELU(), nn.Dropout(0.20), nn.Linear(40, 1)
        )

    def forward(self, spatial, process, coordinates):
        coords = coordinates.unsqueeze(0).expand(spatial.shape[0], -1, -1)
        local = self.point_encoder(torch.cat([spatial.unsqueeze(-1), coords], dim=-1))
        global_context = torch.cat([local.mean(dim=1), local.amax(dim=1)], dim=-1)
        repeated = global_context.unsqueeze(1).expand(-1, spatial.shape[1], -1)
        pieces = [local, repeated]
        if self.fused:
            process_context = self.temporal(process)
            pieces.append(process_context.unsqueeze(1).expand(-1, spatial.shape[1], -1))
        return self.decoder(torch.cat(pieces, dim=-1)).squeeze(-1)


class GraphFeatureFusionNet(nn.Module):
    """Flat residual predictor augmented with coordinate-derived neighbours."""

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
        self.decoder = nn.Sequential(
            nn.Linear(48, 48), nn.GELU(), nn.Dropout(0.20), nn.Linear(48, 88)
        )

    def forward(self, spatial, process, coordinates):
        neighbour_mean = spatial @ self.adjacency.T
        spatial_context = self.spatial(torch.cat([spatial, neighbour_mean], dim=1))
        process_context = self.temporal(process)
        return self.decoder(torch.cat([spatial_context, process_context], dim=1))


def standardize(train, *arrays, axes=(0,)):
    median = np.nanmedian(train, axis=axes, keepdims=True)
    train_filled = np.where(np.isnan(train), median, train)
    mean = train_filled.mean(axis=axes, keepdims=True)
    std = train_filled.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return [(np.where(np.isnan(a), median, a) - mean) / std for a in arrays]


def metric_set(actual, predicted):
    wafer_r2 = np.asarray([r2_score(a, p) for a, p in zip(actual, predicted)])
    return {
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mae": mean_absolute_error(actual, predicted),
        "r2_global": r2_score(actual.ravel(), predicted.ravel()),
        "r2_mean_location": r2_score(actual, predicted, multioutput="uniform_average"),
        "r2_mean_wafer": wafer_r2.mean(),
        "r2_median_wafer": np.median(wafer_r2),
        "wafer_r2_positive_fraction": np.mean(wafer_r2 > 0),
    }


def train_fold(spatial, process, target, coords, raw_coordinates, train_idx, test_idx, model_name, args, fold):
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
    mean = residual[fit_idx].mean(axis=0, keepdims=True)
    std = residual[fit_idx].std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    r_fit = (residual[fit_idx] - mean) / std
    r_val = (residual[val_idx] - mean) / std

    def tensorize(s, p, r=None):
        values = [
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(p.transpose(0, 2, 1), dtype=torch.float32),
        ]
        if r is not None:
            values.append(torch.tensor(r, dtype=torch.float32))
        return values

    seed_all(args.seed + fold)
    if model_name == "coordinate_spatial_only":
        model = CoordinateFusionNet(fused=False)
    elif model_name == "coordinate_fused":
        model = CoordinateFusionNet(fused=True)
    else:
        model = GraphFeatureFusionNet(raw_coordinates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    coord_tensor = torch.tensor(coords, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(*tensorize(s_fit, p_fit, r_fit)),
        batch_size=min(16, len(fit_idx)), shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + fold),
    )
    vs, vp, vr = tensorize(s_val, p_val, r_val)
    best_state, best_loss, best_epoch, wait = None, float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bs, bp, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bs, bp, coord_tensor), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(vs, vp, coord_tensor), vr).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    ts, tp = tensorize(s_test, p_test)
    model.eval()
    with torch.no_grad():
        predicted_residual = model(ts, tp, coord_tensor).numpy() * std + mean
    return spatial[test_idx] + predicted_residual, best_epoch, best_loss


def spatial_figure(path, coords, actual, predicted, title):
    error = predicted - actual
    limits = (min(actual.min(), predicted.min()), max(actual.max(), predicted.max()))
    vmax = max(abs(error.min()), abs(error.max()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for ax, values, label in zip(axes[:2], [actual, predicted], ["Measured output", "Predicted output"]):
        plot = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=65, cmap="viridis", vmin=limits[0], vmax=limits[1])
        ax.set_title(label)
        fig.colorbar(plot, ax=ax, shrink=0.8, label="Thickness (µm)")
    plot = axes[2].scatter(coords[:, 0], coords[:, 1], c=error, s=65, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Prediction error")
    fig.colorbar(plot, ax=axes[2], shrink=0.8, label="Error (µm)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("Wafer x coordinate")
        ax.set_ylabel("Wafer y coordinate")
        ax.grid(alpha=0.15)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", args.target)
    processes, process_names = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", args.time_steps
    )
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    process = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    target = np.stack([record["target"] for record in matched])
    keys = np.asarray([record["experiment_key"] for record in matched])
    lots = np.asarray([record["lot"] for record in matched])
    coords = coordinate_features(coordinates)
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(spatial))

    folds, summaries = [], []
    saved = {}
    for name in ["coordinate_spatial_only", "coordinate_fused", "graph_feature_fused"]:
        oof = np.full_like(target, np.nan)
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            prediction, epoch, val_loss = train_fold(
                spatial, process, target, coords, coordinates, train_idx, test_idx, name, args, fold
            )
            oof[test_idx] = prediction
            folds.append({
                "model": name, "fold": fold, "train_wafers": len(train_idx),
                "test_wafers": len(test_idx), "best_epoch": epoch,
                "validation_loss_scaled": val_loss, **metric_set(target[test_idx], prediction),
            })
        summaries.append({"model": name, **metric_set(target, oof)})
        saved[name] = oof
        np.savez_compressed(
            args.output_dir / f"oof_{name}.npz", keys=keys, lots=lots,
            actual=target, predicted=oof, coordinates=coordinates,
        )

    pd.DataFrame(folds).to_csv(args.output_dir / "coordinate_cv_metrics.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "coordinate_summary_metrics.csv", index=False)

    best_name = summary.sort_values("rmse").iloc[0]["model"]
    errors = np.sqrt(((target - saved[best_name]) ** 2).mean(axis=1))
    idx = np.argsort(errors)[len(errors) // 2]
    spatial_figure(
        args.output_dir / "coordinate_fusion_spatial_output.png", coordinates,
        target[idx], saved[best_name][idx],
        f"{best_name.replace('_', ' ').title()} out-of-fold prediction for wafer {keys[idx]}",
    )

    metadata = {
        "matched_wafers": len(matched), "lots": sorted(map(int, set(lots))),
        "coordinate_features": ["normalized_x", "normalized_y", "normalized_radius"],
        "architecture": "shared point MLP on [thickness,x,y,radius], global mean/max pooling, temporal Conv1D process encoder, pointwise residual decoder",
        "parameter_counts": {
            "coordinate_spatial_only": sum(p.numel() for p in CoordinateFusionNet(False).parameters()),
            "coordinate_fused": sum(p.numel() for p in CoordinateFusionNet(True).parameters()),
            "graph_feature_fused": sum(p.numel() for p in GraphFeatureFusionNet(coordinates).parameters()),
        },
        "validation": "shuffled five-fold wafer cross-validation with fold-only preprocessing and out-of-fold predictions",
        "process_signals": process_names,
        "seed": args.seed,
    }
    (args.output_dir / "coordinate_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps({key: value for key, value in metadata.items() if key != "process_signals"}, indent=2))


if __name__ == "__main__":
    main()
