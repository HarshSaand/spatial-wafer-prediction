"""Compact two-branch neural model for 88 + 31x64 -> 88 wafer prediction.

The experiment is intentionally small because only 88 matched wafers are
available.  All imputers/scalers are fitted inside each outer CV training fold.
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
from sklearn.model_selection import GroupKFold, KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_baselines import group_name, load_process_features, load_wafer_arrays


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", default="postox_thickness")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--time-steps", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class FusionNet(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.spatial = nn.Sequential(
            nn.Linear(88, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(48, 24), nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(31, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 24, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(96, 24), nn.GELU(),
        )
        width = 48 if mode == "fused_spatial_process" else 24
        self.head = nn.Sequential(
            nn.Linear(width, 48), nn.GELU(), nn.Dropout(0.20), nn.Linear(48, 88)
        )

    def forward(self, spatial, process):
        if self.mode == "spatial_only":
            z = self.spatial(spatial)
        elif self.mode == "process_only":
            z = self.temporal(process)
        else:
            z = torch.cat([self.spatial(spatial), self.temporal(process)], dim=1)
        return self.head(z)


def outer_splits(n, y, groups, folds, mode, seed):
    x = np.zeros((n, 1))
    if mode == "lot":
        return list(GroupKFold(folds).split(x, y, groups))
    return list(KFold(folds, shuffle=True, random_state=seed).split(x, y))


def inner_split(train_idx, groups, mode, seed):
    if mode == "lot":
        # Keep a complete lot for early-stopping validation.
        lots = np.unique(groups[train_idx])
        rng = np.random.default_rng(seed)
        val_lot = rng.choice(lots)
        val = train_idx[groups[train_idx] == val_lot]
        fit = train_idx[groups[train_idx] != val_lot]
        return fit, val
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_val = max(8, round(0.15 * len(train_idx)))
    return shuffled[n_val:], shuffled[:n_val]


def med_scale(train, *arrays, axes=(0,)):
    med = np.nanmedian(train, axis=axes, keepdims=True)
    train_i = np.where(np.isnan(train), med, train)
    mean = train_i.mean(axis=axes, keepdims=True)
    std = train_i.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    out = []
    for a in arrays:
        out.append((np.where(np.isnan(a), med, a) - mean) / std)
    return out, (med, mean, std)


def train_fold(spatial, process, target, groups, train_idx, test_idx, mode, split_mode, args, fold):
    fit_idx, val_idx = inner_split(train_idx, groups, split_mode, args.seed + fold)
    (s_fit, s_val, s_test), _ = med_scale(spatial[fit_idx], spatial[fit_idx], spatial[val_idx], spatial[test_idx])
    # Per-signal scaling; retain the ordered time axis for temporal convolutions.
    (p_fit, p_val, p_test), _ = med_scale(
        process[fit_idx], process[fit_idx], process[val_idx], process[test_idx], axes=(0, 1)
    )
    # Spatial and fused models learn the physical pre-to-post change.  The
    # process-only ablation must not use the incoming wafer, even as a skip.
    learning_target = target if mode == "process_only" else target - spatial
    r_mean = learning_target[fit_idx].mean(axis=0, keepdims=True)
    r_std = learning_target[fit_idx].std(axis=0, keepdims=True)
    r_std[r_std < 1e-8] = 1.0
    r_fit = (learning_target[fit_idx] - r_mean) / r_std
    r_val = (learning_target[val_idx] - r_mean) / r_std

    def tensors(s, p, r=None):
        vals = [torch.tensor(s, dtype=torch.float32), torch.tensor(p.transpose(0, 2, 1), dtype=torch.float32)]
        if r is not None:
            vals.append(torch.tensor(r, dtype=torch.float32))
        return vals

    seed_all(args.seed + fold)
    model = FusionNet(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    fit_t = tensors(s_fit, p_fit, r_fit)
    loader = DataLoader(TensorDataset(*fit_t), batch_size=min(16, len(fit_idx)), shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed + fold))
    vs, vp, vr = tensors(s_val, p_val, r_val)
    best, best_loss, wait, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bs, bp, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bs, bp), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(vs, vp), vr).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best)
    ts, tp = tensors(s_test, p_test)
    model.eval()
    with torch.no_grad():
        pred_resid = model(ts, tp).numpy() * r_std + r_mean
    prediction = pred_resid if mode == "process_only" else spatial[test_idx] + pred_resid
    return prediction, best_epoch, best_loss, len(fit_idx), len(val_idx)


def metrics(y, p):
    return {
        "rmse": mean_squared_error(y, p) ** 0.5,
        "mae": mean_absolute_error(y, p),
        "r2_global": r2_score(y.ravel(), p.ravel()),
        "r2_mean_location": r2_score(y, p, multioutput="uniform_average"),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", args.target)
    processes, process_names = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", args.time_steps
    )
    matched = [r for r in records if group_name(r["experiment_key"]) in processes]
    spatial = np.stack([r["spatial"] for r in matched])
    process = np.stack([processes[group_name(r["experiment_key"])] for r in matched])
    target = np.stack([r["target"] for r in matched])
    groups = np.asarray([r["lot"] for r in matched])
    keys = np.asarray([r["experiment_key"] for r in matched])

    fold_rows, summary_rows = [], []
    for split_mode in ["wafer", "lot"]:
        splits = outer_splits(len(matched), target, groups, args.folds, split_mode, args.seed)
        for mode in ["spatial_only", "process_only", "fused_spatial_process"]:
            oof = np.full_like(target, np.nan)
            for fold, (train_idx, test_idx) in enumerate(splits, 1):
                pred, epoch, val_loss, nfit, nval = train_fold(
                    spatial, process, target, groups, train_idx, test_idx, mode, split_mode, args, fold
                )
                oof[test_idx] = pred
                row = {"validation": split_mode, "model": mode, "fold": fold,
                       "fit_wafers": nfit, "validation_wafers": nval, "test_wafers": len(test_idx),
                       "test_lots": ",".join(map(str, sorted(set(groups[test_idx])))),
                       "best_epoch": epoch, "validation_loss_scaled": val_loss}
                row.update(metrics(target[test_idx], pred))
                fold_rows.append(row)
            total = {"validation": split_mode, "model": mode, **metrics(target, oof)}
            total["best_epoch_mean"] = np.mean([r["best_epoch"] for r in fold_rows
                                                 if r["validation"] == split_mode and r["model"] == mode])
            summary_rows.append(total)
            np.savez_compressed(args.output_dir / f"oof_{split_mode}_{mode}.npz",
                                keys=keys, lots=groups, actual=target, predicted=oof, coordinates=coordinates)

    pd.DataFrame(fold_rows).to_csv(args.output_dir / "deep_cv_metrics.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "deep_summary_metrics.csv", index=False)

    # One concise progress figure: actual vs OOF prediction for a representative wafer.
    chosen = np.load(args.output_dir / "oof_wafer_fused_spatial_process.npz")
    errors = np.sqrt(((chosen["actual"] - chosen["predicted"]) ** 2).mean(axis=1))
    idx = np.argsort(errors)[len(errors) // 2]
    order = np.argsort(np.arctan2(coordinates[:, 1], coordinates[:, 0]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(chosen["actual"][idx, order], label="Actual outgoing", lw=2)
    ax.plot(chosen["predicted"][idx, order], label="Deep-fusion prediction", lw=2)
    ax.set(title=f"Representative out-of-fold wafer ({keys[idx]})", xlabel="Spatial point (angular order)",
           ylabel="Post-process thickness (µm)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(args.output_dir / "deep_fusion_representative_output.png", dpi=180)
    plt.close(fig)

    metadata = {
        "matched_wafers": len(matched), "lots": sorted(map(int, set(groups))),
        "input_spatial": list(spatial.shape), "input_process": list(process.shape),
        "target": list(target.shape), "process_signals": process_names,
        "architecture": "88->48->24 spatial MLP; 31x64 temporal Conv1D encoder; fusion MLP -> 88 outputs (residual for models with spatial input)",
        "parameter_counts": {m: sum(p.numel() for p in FusionNet(m).parameters()) for m in
                             ["spatial_only", "process_only", "fused_spatial_process"]},
        "seed": args.seed, "early_stopping_patience": args.patience,
        "leakage_control": "imputation, scaling and residual target normalization fitted within each outer training fold",
    }
    (args.output_dir / "deep_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps({k: v for k, v in metadata.items() if k != "process_signals"}, indent=2))


if __name__ == "__main__":
    main()
