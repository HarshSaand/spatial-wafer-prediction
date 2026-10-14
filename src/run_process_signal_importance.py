"""Estimate held-out group permutation importance for process signals.

Each signal is represented by four summaries across its 64 time steps: mean,
standard deviation, minimum and maximum.  All four columns are shuffled
together within each held-out fold.  The resulting RMSE increase is an
exploratory association, not a causal process conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from run_baselines import group_name, load_process_features, load_wafer_arrays


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, _ = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", "postox_thickness")
    processes, signal_names = load_process_features(
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
    y = np.stack([record["target"] for record in matched])
    splits = KFold(5, shuffle=True, random_state=args.seed).split(x)
    rng = np.random.default_rng(args.seed)
    rows = []

    groups = [("Incoming thickness profile", np.arange(88), "spatial")]
    for signal_index, signal_name in enumerate(signal_names):
        columns = np.asarray([88 + signal_index + block * len(signal_names) for block in range(4)])
        groups.append((signal_name.replace("Stat3_Etch_MV_", ""), columns, "process"))

    for fold, (train, test) in enumerate(splits, 1):
        model = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=.7,
            random_state=args.seed, n_jobs=-1,
        )
        model.fit(x[train], y[train])
        baseline = np.sqrt(mean_squared_error(y[test], model.predict(x[test])))
        for group_name_value, columns, group_type in groups:
            increases = []
            for _ in range(args.repeats):
                permuted = x[test].copy()
                order = rng.permutation(len(test))
                permuted[:, columns] = permuted[order][:, columns]
                rmse = np.sqrt(mean_squared_error(y[test], model.predict(permuted)))
                increases.append(rmse - baseline)
            rows.append({
                "fold": fold,
                "feature_group": group_name_value,
                "group_type": group_type,
                "baseline_rmse": baseline,
                "rmse_increase_mean": float(np.mean(increases)),
                "rmse_increase_sd": float(np.std(increases)),
            })

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["feature_group", "group_type"], as_index=False).agg(
        rmse_increase_mean=("rmse_increase_mean", "mean"),
        rmse_increase_sd_across_folds=("rmse_increase_mean", "std"),
    ).sort_values("rmse_increase_mean", ascending=False)
    detail.to_csv(args.output_dir / "process_signal_importance_by_fold.csv", index=False)
    summary.to_csv(args.output_dir / "process_signal_importance_summary.csv", index=False)

    process_only = summary.loc[summary["group_type"] == "process"].head(10).sort_values("rmse_increase_mean")
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.barh(process_only["feature_group"], process_only["rmse_increase_mean"], color="#1F77B4")
    axis.set(xlabel="Held-out RMSE increase after grouped permutation (µm)",
             title="Top process-signal associations in the public development data")
    axis.grid(axis="x", alpha=.2)
    fig.savefig(args.output_dir / "top_process_signal_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "samples": len(y),
        "process_signals": len(signal_names),
        "summaries_per_signal": ["mean", "standard deviation", "minimum", "maximum"],
        "validation": "Five held-out wafer folds with ten within-fold permutations per feature group",
        "interpretation": "Positive RMSE increase indicates predictive association in this public plasma-etch dataset; it does not establish causality or CMP transferability",
    }
    (args.output_dir / "process_signal_importance_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
