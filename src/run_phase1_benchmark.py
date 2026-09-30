"""Create a consistent phase-one benchmark including wafer-level R-squared."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from run_baselines import estimator, group_name, load_process_features, load_wafer_arrays


DATA = Path("work/public_data")
OUT = Path("outputs/phase1_benchmark_2026-09-30")


def metrics(name, family, actual, predicted, parameters="not applicable"):
    wafer_r2 = np.asarray([r2_score(a, p) for a, p in zip(actual, predicted)])
    return {
        "model": name,
        "family": family,
        "parameters": parameters,
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mae": mean_absolute_error(actual, predicted),
        "r2_global": r2_score(actual.ravel(), predicted.ravel()),
        "r2_mean_wafer": wafer_r2.mean(),
        "r2_median_wafer": np.median(wafer_r2),
        "wafer_r2_positive_fraction": np.mean(wafer_r2 > 0),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, _ = load_wafer_arrays(DATA / "Si_Oxide_etch_89_points.csv", "postox_thickness")
    processes, _ = load_process_features(DATA / "Process_data.nc", DATA / "Dictionary_process.nc", 64)
    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    sequence = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    process_summary = np.concatenate(
        [sequence.mean(1), sequence.std(1), sequence.min(1), sequence.max(1)], axis=1
    )
    fused = np.concatenate([spatial, process_summary], axis=1)
    target = np.stack([record["target"] for record in matched])
    splits = list(KFold(5, shuffle=True, random_state=42).split(spatial))
    fold_ids = np.zeros(len(spatial), dtype=int)
    for fold, (_, test) in enumerate(splits, 1):
        fold_ids[test] = fold

    rows = []
    predictions = {}
    mean_oof = np.full_like(target, np.nan)
    for train, test in splits:
        mean_oof[test] = target[train].mean(axis=0)
    predictions["Mean profile baseline"] = mean_oof

    for label, model_name in [("Ridge fusion", "ridge"), ("Extra Trees fusion", "extra_trees")]:
        oof = np.full_like(target, np.nan)
        for train, test in splits:
            model = clone(estimator(10.0, model_name))
            model.fit(fused[train], target[train])
            oof[test] = model.predict(fused[test])
        predictions[label] = oof

    deep_files = {
        "Flat MLP Conv1D fusion": Path("outputs/deep_fusion_2026-09-23/oof_wafer_fused_spatial_process.npz"),
        "Coordinate point fusion": Path("outputs/coordinate_fusion_2026-09-30/oof_coordinate_fused.npz"),
        "Graph feature fusion": Path("outputs/coordinate_fusion_2026-09-30/oof_graph_feature_fused.npz"),
    }
    for label, path in deep_files.items():
        predictions[label] = np.load(path)["predicted"]

    descriptions = {
        "Mean profile baseline": ("baseline", "not applicable"),
        "Ridge fusion": ("linear", "varies by transformed multi-output fit"),
        "Extra Trees fusion": ("tree ensemble", "500 trees"),
        "Flat MLP Conv1D fusion": ("deep learning", "18,976"),
        "Coordinate point fusion": ("coordinate-aware deep learning", "10,081"),
        "Graph feature fusion": ("geometry-aware deep learning", "23,200"),
    }
    for name, prediction in predictions.items():
        family, parameters = descriptions[name]
        rows.append(metrics(name, family, target, prediction, parameters))
    results = pd.DataFrame(rows).sort_values("rmse")
    results.to_csv(OUT / "phase1_model_benchmark.csv", index=False)

    fold_rows = []
    for name, prediction in predictions.items():
        wafer_r2 = np.asarray([r2_score(a, p) for a, p in zip(target, prediction)])
        wafer_rmse = np.sqrt(((target - prediction) ** 2).mean(axis=1))
        for fold in range(1, 6):
            selected = fold_ids == fold
            fold_rows.append({
                "model": name,
                "fold": fold,
                "test_wafers": int(selected.sum()),
                "mean_wafer_r2": wafer_r2[selected].mean(),
                "median_wafer_r2": np.median(wafer_r2[selected]),
                "positive_wafer_r2_fraction": np.mean(wafer_r2[selected] > 0),
                "mean_wafer_rmse": wafer_rmse[selected].mean(),
            })
    pd.DataFrame(fold_rows).to_csv(OUT / "phase1_per_fold_wafer_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    order = results.sort_values("rmse", ascending=False)
    colors = ["#829ab1" if "fusion" not in name.lower() else "#2f80b7" for name in order["model"]]
    axes[0].barh(order["model"], order["rmse"], color=colors)
    axes[0].set_xlabel("Out-of-fold RMSE (µm); lower is better")
    axes[0].set_title("Prediction error")
    axes[0].grid(axis="x", alpha=0.2)
    order_r2 = results.sort_values("r2_median_wafer")
    axes[1].barh(order_r2["model"], order_r2["r2_median_wafer"], color="#4c956c")
    axes[1].axvline(0, color="#444444", lw=0.8)
    axes[1].set_xlabel("Median wafer-level R²; higher is better")
    axes[1].set_title("Spatial fit on individual wafers")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Phase A model benchmark using five-fold wafer cross-validation", fontsize=13)
    fig.savefig(OUT / "phase1_model_benchmark.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
