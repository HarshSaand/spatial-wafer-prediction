"""Graph-attention spatial encoder plus temporal Conv1D for 88-point thickness prediction."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_baselines import group_name, load_process_features, load_wafer_arrays
from run_coordinate_fusion import standardize
from run_region_aware_fusion import overall_metrics, radial_regions, region_metrics
from run_shape_decomposed_fusion import shape_metrics


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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class GraphAttentionBlock(nn.Module):
    def __init__(self, width, adjacency):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.register_buffer("mask", adjacency.bool())

    def forward(self, x):
        scores = torch.matmul(self.query(x), self.key(x).transpose(1, 2)) / math.sqrt(x.shape[-1])
        scores = scores.masked_fill(~self.mask.unsqueeze(0), -1e9)
        weights = torch.softmax(scores, dim=-1)
        message = torch.matmul(weights, self.value(x))
        return torch.nn.functional.gelu(self.norm(x + self.output(message)))


class GraphAttentionTemporalNet(nn.Module):
    def __init__(self, coordinates, neighbours=6):
        super().__init__()
        xy = torch.tensor(coordinates, dtype=torch.float32)
        distance = torch.cdist(xy, xy)
        nearest = distance.argsort(dim=1)[:, 1:neighbours + 1]
        adjacency = torch.eye(len(xy), dtype=torch.bool)
        adjacency.scatter_(1, nearest, True)
        adjacency = adjacency | adjacency.T
        self.register_buffer("coordinates", xy)
        self.node_input = nn.Sequential(nn.Linear(4, 24), nn.LayerNorm(24), nn.GELU())
        self.attention1 = GraphAttentionBlock(24, adjacency)
        self.attention2 = GraphAttentionBlock(24, adjacency)
        self.temporal = nn.Sequential(
            nn.Conv1d(31, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 24, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(96, 24), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(96, 48), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1),
        )

    def forward(self, spatial, process):
        batch = spatial.shape[0]
        coords = self.coordinates.unsqueeze(0).expand(batch, -1, -1)
        radius = torch.sqrt((coords ** 2).sum(dim=-1, keepdim=True))
        nodes = self.node_input(torch.cat([spatial.unsqueeze(-1), coords, radius], dim=-1))
        nodes = self.attention2(self.attention1(nodes))
        global_context = torch.cat([nodes.mean(dim=1), nodes.amax(dim=1)], dim=1)
        process_context = self.temporal(process)
        shared = torch.cat([global_context, process_context], dim=1).unsqueeze(1).expand(-1, nodes.shape[1], -1)
        return self.decoder(torch.cat([nodes, shared], dim=-1)).squeeze(-1)


def train_fold(spatial, process, target, coordinates, train_idx, test_idx, args, fold):
    rng = np.random.default_rng(args.seed + fold)
    shuffled = rng.permutation(train_idx)
    n_val = max(8, round(0.15 * len(train_idx)))
    val_idx, fit_idx = shuffled[:n_val], shuffled[n_val:]
    s_fit, s_val, s_test = standardize(spatial[fit_idx], spatial[fit_idx], spatial[val_idx], spatial[test_idx])
    p_fit, p_val, p_test = standardize(process[fit_idx], process[fit_idx], process[val_idx], process[test_idx], axes=(0, 1))
    residual = target - spatial
    y_mu = residual[fit_idx].mean(axis=0, keepdims=True)
    y_sd = residual[fit_idx].std(axis=0, keepdims=True); y_sd[y_sd < 1e-8] = 1.0
    y_fit = (residual[fit_idx] - y_mu) / y_sd
    y_val = (residual[val_idx] - y_mu) / y_sd

    def tensors(s, p, y=None):
        values = [torch.tensor(s, dtype=torch.float32), torch.tensor(p.transpose(0, 2, 1), dtype=torch.float32)]
        if y is not None: values.append(torch.tensor(y, dtype=torch.float32))
        return values

    seed_all(args.seed + fold)
    model = GraphAttentionTemporalNet(coordinates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    loader = DataLoader(TensorDataset(*tensors(s_fit, p_fit, y_fit)), batch_size=min(16, len(fit_idx)),
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed + fold))
    vs, vp, vy = tensors(s_val, p_val, y_val)
    best_state, best_loss, best_epoch, wait = None, float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bs, bp, by in loader:
            optimizer.zero_grad(); prediction = model(bs, bp); loss = loss_fn(prediction, by)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        model.eval()
        with torch.no_grad(): validation_loss = loss_fn(model(vs, vp), vy).item()
        if validation_loss < best_loss - 1e-5:
            best_state, best_loss, best_epoch, wait = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            wait += 1
            if wait >= args.patience: break
    model.load_state_dict(best_state); ts, tp = tensors(s_test, p_test); model.eval()
    with torch.no_grad(): predicted_scaled = model(ts, tp).numpy()
    prediction = spatial[test_idx] + predicted_scaled * y_sd + y_mu
    return prediction, best_epoch, best_loss, sum(p.numel() for p in model.parameters())


def plot_output(path, actual, predicted, coordinates):
    wafer_r2 = np.asarray([r2_score(a, p) for a, p in zip(actual, predicted)])
    representative = int(np.argsort(wafer_r2)[len(wafer_r2) // 2])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].scatter(actual.ravel(), predicted.ravel(), s=7, alpha=.35, color="#2B6F8E")
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[0].plot([low, high], [low, high], "k--", lw=1)
    axes[0].set(xlabel="Measured post-process thickness", ylabel="Predicted post-process thickness",
                title="All out-of-fold predictions")
    order = np.argsort(np.arctan2(coordinates[:, 1], coordinates[:, 0]))
    axes[1].plot(actual[representative, order], label="Measured", lw=2)
    axes[1].plot(predicted[representative, order], label="Predicted", lw=2)
    axes[1].set(xlabel="Spatial location ordered by angle", ylabel="Thickness",
                title=f"Representative wafer R² {wafer_r2[representative]:.3f}")
    axes[1].legend(frameon=False)
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", args.target)
    processes, _ = load_process_features(args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", args.time_steps)
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([r["spatial"] for r in matched]); target = np.stack([r["target"] for r in matched])
    process = np.stack([processes[group_name(r["experiment_key"])] for r in matched])
    keys = np.asarray([r["experiment_key"] for r in matched]); lots = np.asarray([r["lot"] for r in matched])
    radius, regions = radial_regions(coordinates)
    splits = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(spatial))
    oof = np.full_like(target, np.nan); fold_rows = []; parameter_count = None
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        prediction, epoch, validation_loss, parameter_count = train_fold(spatial, process, target, coordinates, train_idx, test_idx, args, fold)
        oof[test_idx] = prediction
        fold_rows.append({"model":"graph_attention_temporal_cnn", "fold":fold, "test_wafers":len(test_idx),
                          "best_epoch":epoch, "validation_loss_scaled":validation_loss,
                          **overall_metrics(target[test_idx], prediction), **shape_metrics(target[test_idx], prediction)})
    flat = np.load("outputs/deep_fusion_2026-09-23/oof_wafer_fused_spatial_process.npz")["predicted"]
    ensemble = 0.5 * flat + 0.5 * oof
    summaries = [
        {"model":"graph_attention_temporal_cnn", "parameters":parameter_count, **overall_metrics(target,oof), **shape_metrics(target,oof)},
        {"model":"graph_attention_equal_weight_ensemble", "parameters":18976+parameter_count, **overall_metrics(target,ensemble), **shape_metrics(target,ensemble)},
    ]
    region_rows=[]
    for name,predicted in [("graph_attention_temporal_cnn",oof),("graph_attention_equal_weight_ensemble",ensemble)]: region_rows.extend(region_metrics(target,predicted,regions,name))
    pd.DataFrame(summaries).to_csv(args.output_dir/"graph_attention_summary.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir/"graph_attention_fold_metrics.csv",index=False)
    pd.DataFrame(region_rows).to_csv(args.output_dir/"graph_attention_region_metrics.csv",index=False)
    np.savez_compressed(args.output_dir/"oof_graph_attention_temporal_cnn.npz",keys=keys,lots=lots,actual=target,predicted=oof,coordinates=coordinates,radius=radius,regions=regions)
    np.savez_compressed(args.output_dir/"oof_graph_attention_equal_weight_ensemble.npz",keys=keys,lots=lots,actual=target,predicted=ensemble,coordinates=coordinates,radius=radius,regions=regions)
    plot_output(args.output_dir/"graph_attention_prediction_output.png",target,oof,coordinates)
    metadata={"architecture":"Two graph-attention layers over a symmetric six-neighbour wafer graph plus temporal 1D CNN and pointwise residual decoder","matched_wafers":len(matched),"validation":"Five-fold out-of-fold wafer validation","seed":args.seed,"parameters":parameter_count}
    (args.output_dir/"graph_attention_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False)); print(pd.DataFrame(region_rows).to_string(index=False))


if __name__ == "__main__": main()
