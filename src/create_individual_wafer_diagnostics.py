"""Create individual held-out wafer colour maps and per-wafer metrics."""

from pathlib import Path
import re
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


SOURCE = Path("outputs/shape_decomposed_2026-10-07/oof_equal_weight_deep_ensemble.npz")
OUT_DIR = Path("outputs/individual_wafer_diagnostics_2026-10-07")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def save_all_wafer_maps(actual, predicted, coordinates, keys, lots, metrics, out_dir):
    """Save one measured, predicted and error triptych for every held-out wafer."""
    image_dir = out_dir / "all_wafer_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    thickness_low = min(actual.min(), predicted.min())
    thickness_high = max(actual.max(), predicted.max())
    error_limit = np.quantile(np.abs(predicted - actual), .98)
    metrics_by_index = metrics.set_index("wafer_index")

    for index, (measured, estimate) in enumerate(zip(actual, predicted)):
        error = estimate - measured
        row = metrics_by_index.loc[index]
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
        panels = [
            (measured, "Measured post CMP thickness", "viridis", thickness_low, thickness_high),
            (estimate, "Predicted post CMP thickness", "viridis", thickness_low, thickness_high),
            (error, "Prediction error", "coolwarm", -error_limit, error_limit),
        ]
        for axis, (values, title, cmap, low, high) in zip(axes, panels):
            plot = axis.scatter(
                coordinates[:, 0], coordinates[:, 1], c=values, cmap=cmap,
                vmin=low, vmax=high, s=88, edgecolor="white", linewidth=.35,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(title, fontsize=10.5)
            fig.colorbar(plot, ax=axis, fraction=.046, pad=.03)
        fig.suptitle(
            f"Held-out wafer {keys[index]} | lot {lots[index]} | "
            f"R² {row['wafer_r2']:.3f} | RMSE {row['wafer_rmse']:.4f}",
            fontsize=12,
        )
        filename = f"{index + 1:03d}_{safe_filename(keys[index])}_lot_{safe_filename(lots[index])}.png"
        fig.savefig(image_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    manifest = metrics.sort_values("wafer_index").copy()
    manifest["image_file"] = [
        f"{int(row.wafer_index) + 1:03d}_{safe_filename(row.wafer_key)}_lot_{safe_filename(row.lot)}.png"
        for row in manifest.itertuples()
    ]
    manifest.to_csv(image_dir / "image_manifest.csv", index=False)
    return image_dir


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.source)
    actual = data["actual"]
    predicted = data["predicted"]
    coordinates = data["coordinates"]
    keys = data["keys"].astype(str)
    lots = data["lots"]

    rows = []
    for index, (measured, estimate) in enumerate(zip(actual, predicted)):
        rows.append({
            "wafer_index": index,
            "wafer_key": keys[index],
            "lot": lots[index],
            "wafer_r2": r2_score(measured, estimate),
            "wafer_rmse": np.sqrt(mean_squared_error(measured, estimate)),
            "mean_bias": float(np.mean(estimate - measured)),
        })
    metrics = pd.DataFrame(rows).sort_values("wafer_r2", ascending=False)
    metrics.to_csv(args.output_dir / "individual_wafer_metrics.csv", index=False)
    image_dir = save_all_wafer_maps(actual, predicted, coordinates, keys, lots, metrics, args.output_dir)

    ordered = metrics.sort_values("wafer_r2").reset_index(drop=True)
    selections = [
        ("Lower performing", int(ordered.iloc[max(0, round(.10 * (len(ordered)-1)))]["wafer_index"])),
        ("Representative", int(ordered.iloc[len(ordered)//2]["wafer_index"])),
        ("Higher performing", int(ordered.iloc[min(len(ordered)-1, round(.90 * (len(ordered)-1)))]["wafer_index"])),
    ]
    thickness_low = min(actual.min(), predicted.min())
    thickness_high = max(actual.max(), predicted.max())
    error_limit = np.quantile(np.abs(predicted - actual), .98)
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12), constrained_layout=True)
    for row, (label, index) in enumerate(selections):
        measured, estimate = actual[index], predicted[index]
        error = estimate - measured
        score = r2_score(measured, estimate)
        rmse = np.sqrt(mean_squared_error(measured, estimate))
        panels = [
            (measured, "Measured post CMP thickness", "viridis", thickness_low, thickness_high),
            (estimate, "Predicted post CMP thickness", "viridis", thickness_low, thickness_high),
            (error, "Prediction error", "coolwarm", -error_limit, error_limit),
        ]
        for col, (values, title, cmap, low, high) in enumerate(panels):
            plot = axes[row, col].scatter(coordinates[:, 0], coordinates[:, 1], c=values, cmap=cmap,
                                          vmin=low, vmax=high, s=92, edgecolor="white", linewidth=.35)
            axes[row, col].set_aspect("equal"); axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            axes[row, col].set_title(title if row == 0 else "", fontsize=12)
            if col == 0:
                axes[row, col].set_ylabel(f"{label}\n{keys[index]}\nR² {score:.3f}  RMSE {rmse:.4f}", fontsize=10)
            fig.colorbar(plot, ax=axes[row, col], fraction=.046, pad=.03)
    fig.suptitle("Individual held-out wafer diagnosis for the recommended prediction model", fontsize=16)
    fig.savefig(args.output_dir / "individual_wafer_colour_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(metrics[["wafer_key", "lot", "wafer_r2", "wafer_rmse", "mean_bias"]].describe().to_string())
    print("\nSelected wafers")
    for label, index in selections:
        row = pd.DataFrame(rows).iloc[index]
        print(label, row["wafer_key"], f"R2={row['wafer_r2']:.3f}", f"RMSE={row['wafer_rmse']:.4f}")
    print(f"\nSaved {len(actual)} individual wafer images to {image_dir}")


if __name__ == "__main__":
    main()
