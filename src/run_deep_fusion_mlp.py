"""Small-sample deep fusion baseline for 88 + 31x64 -> 88 wafer prediction.

The network uses separate spatial and process-summary encoders, then fuses their
representations.  All scalers are fitted inside each CV fold.  Early stopping,
dropout, weight decay, and a compact architecture limit overfitting on 88 wafers.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_baselines import group_name, load_process_features, load_wafer_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", default="postox_thickness")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--time-steps", type=int, default=64)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--direct-target", action="store_true", help="Predict post-process values directly instead of the physical change")
    return p.parse_args()


class DeepFusionMLP(nn.Module):
    def __init__(self, spatial_dim: int, process_dim: int, output_dim: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Linear(spatial_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.process = nn.Sequential(
            nn.Linear(process_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(64, output_dim),
        )

    def forward(self, spatial: torch.Tensor, process: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.spatial(spatial), self.process(process)], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_fold(x_spatial, x_process, y, train_idx, test_idx, seed, epochs, patience, residual=True):
    set_seed(seed)
    # Keep a validation subset wholly inside the outer training fold.
    inner = np.random.default_rng(seed).permutation(train_idx)
    n_val = max(8, round(0.18 * len(inner)))
    val_idx, fit_idx = inner[:n_val], inner[n_val:]

    learned_target = y - x_spatial if residual else y
    ss, ps, ys = StandardScaler(), StandardScaler(), StandardScaler()
    xs_fit = ss.fit_transform(x_spatial[fit_idx])
    xp_fit = ps.fit_transform(x_process[fit_idx])
    y_fit = ys.fit_transform(learned_target[fit_idx])
    xs_val, xp_val, y_val = ss.transform(x_spatial[val_idx]), ps.transform(x_process[val_idx]), ys.transform(learned_target[val_idx])

    ds = TensorDataset(*(torch.tensor(a, dtype=torch.float32) for a in (xs_fit, xp_fit, y_fit)))
    loader = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True)
    model = DeepFusionMLP(xs_fit.shape[1], xp_fit.shape[1], y_fit.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    val_tensors = tuple(torch.tensor(a, dtype=torch.float32) for a in (xs_val, xp_val, y_val))

    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb_s, xb_p, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb_s, xb_p), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_tensors[0], val_tensors[1]), val_tensors[2]).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_z = model(
            torch.tensor(ss.transform(x_spatial[test_idx]), dtype=torch.float32),
            torch.tensor(ps.transform(x_process[test_idx]), dtype=torch.float32),
        ).numpy()
    prediction = ys.inverse_transform(prediction_z)
    if residual:
        prediction = x_spatial[test_idx] + prediction
    return prediction, best_epoch, best_loss


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", args.target)
    processes, process_names = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", args.time_steps
    )
    matched = [r for r in records if group_name(r["experiment_key"]) in processes]
    spatial = np.stack([r["spatial"] for r in matched])
    sequence = np.stack([processes[group_name(r["experiment_key"])] for r in matched])
    process = np.concatenate([sequence.mean(1), sequence.std(1), sequence.min(1), sequence.max(1)], axis=1)
    target = np.stack([r["target"] for r in matched])

    splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
    all_predictions = np.full_like(target, np.nan)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(spatial), 1):
        pred, best_epoch, best_val = train_fold(
            spatial, process, target, train_idx, test_idx,
            args.seed + fold, args.epochs, args.patience, residual=not args.direct_target,
        )
        all_predictions[test_idx] = pred
        rows.append({
            "model": "deep_fusion_mlp", "fold": fold,
            "train_wafers": len(train_idx), "test_wafers": len(test_idx),
            "best_epoch": best_epoch, "best_validation_loss_scaled": best_val,
            "rmse": mean_squared_error(target[test_idx], pred) ** 0.5,
            "mae": mean_absolute_error(target[test_idx], pred),
            "r2_global": r2_score(target[test_idx].ravel(), pred.ravel()),
            "r2_mean_location": r2_score(target[test_idx], pred, multioutput="uniform_average"),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary = metrics[["rmse", "mae", "r2_global", "r2_mean_location", "best_epoch"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "summary_metrics.csv")
    prediction_frame = pd.DataFrame({
        "experiment_key": np.repeat([r["experiment_key"] for r in matched], target.shape[1]),
        "location_index": np.tile(np.arange(target.shape[1]), len(matched)),
        "x": np.tile(coordinates[:, 0], len(matched)),
        "y": np.tile(coordinates[:, 1], len(matched)),
        "actual": target.ravel(), "predicted": all_predictions.ravel(),
        "error": (all_predictions - target).ravel(),
    })
    prediction_frame.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    overall = {
        "model": "Residual DeepFusionMLP (two encoders + fused decoder)" if not args.direct_target else "DeepFusionMLP (two encoders + fused decoder)",
        "prediction_parameterization": "post-process minus incoming thickness, then add incoming thickness back" if not args.direct_target else "direct post-process thickness",
        "validation": "5-fold shuffled wafer-level CV; inner validation and early stopping per fold",
        "wafers": len(matched), "spatial_input_shape": list(spatial.shape),
        "process_sequence_shape": list(sequence.shape), "process_summary_shape": list(process.shape),
        "target_shape": list(target.shape), "parameter_count": sum(p.numel() for p in DeepFusionMLP(88, 124, 88).parameters()),
        "oof_rmse": mean_squared_error(target, all_predictions) ** 0.5,
        "oof_mae": mean_absolute_error(target, all_predictions),
        "oof_r2_global": r2_score(target.ravel(), all_predictions.ravel()),
        "oof_r2_mean_location": r2_score(target, all_predictions, multioutput="uniform_average"),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(overall, indent=2))
    print(metrics.to_string(index=False))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
