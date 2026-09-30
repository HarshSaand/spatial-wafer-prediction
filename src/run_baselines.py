"""Five-fold ablation baselines for 88 + 31x64 -> 88 wafer prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=["postox_thickness", "oxide_etch", "si_etch"],
        default="postox_thickness",
    )
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--split-mode", choices=["wafer", "lot"], default="wafer")
    parser.add_argument("--model", choices=["ridge", "extra_trees"], default="ridge")
    return parser.parse_args()


def group_name(experiment_key: str) -> str:
    date, wafer = experiment_key.rsplit("_", 1)
    return f"Day_{date.replace('-', '_')}_Wafer_{wafer}"


def resample_columns(values: np.ndarray, steps: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, values.shape[0])
    new_x = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(new_x, old_x, values[:, j]) for j in range(values.shape[1])], axis=1)


def load_process_features(data_path: Path, dictionary_path: Path, steps: int):
    with Dataset(dictionary_path) as dictionary:
        decoder = np.asarray(dictionary["data"][:])

    result: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    with Dataset(data_path) as dataset:
        feature_sets = [set(map(str, group["feature"][:])) for group in dataset.groups.values()]
        common = sorted(set.intersection(*feature_sets))
        if len(common) != 31:
            raise ValueError(f"Expected 31 common process signals, found {len(common)}")
        feature_names = common

        for name, group in dataset.groups.items():
            names = list(map(str, group["feature"][:]))
            positions = [names.index(feature) for feature in common]
            encoded = np.asarray(group["data"][:, positions])
            decoded = decoder[encoded].astype(float)
            result[name] = resample_columns(decoded, steps)
    return result, feature_names


def load_wafer_arrays(csv_path: Path, target: str):
    frame = pd.read_csv(csv_path)
    frame = frame.loc[~((frame["X"] == 0) & (frame["Y"] == 0))].copy()
    coordinates = frame[["X", "Y"]].drop_duplicates().sort_values(["Y", "X"])
    if len(coordinates) != 88:
        raise ValueError(f"Expected 88 non-centre locations, found {len(coordinates)}")
    coordinate_index = pd.MultiIndex.from_frame(coordinates)

    records = []
    for experiment_key, wafer in frame.groupby("experiment_key", sort=True):
        indexed = wafer.set_index(["X", "Y"]).reindex(coordinate_index)
        if indexed[["preox_thickness", target]].isna().any().any():
            continue
        records.append(
            {
                "experiment_key": experiment_key,
                "lot": int(wafer["lot_number"].iloc[0]),
                "spatial": indexed["preox_thickness"].to_numpy(float),
                "target": indexed[target].to_numpy(float),
            }
        )
    return records, coordinates.to_numpy(int)


def estimator(alpha: float, model_name: str):
    if model_name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features=0.7,
                random_state=42,
                n_jobs=-1,
            ),
        )
    regressor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def make_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, split_mode: str):
    if split_mode == "lot":
        return list(GroupKFold(n_splits=folds).split(x, y, groups))
    return list(KFold(n_splits=folds, shuffle=True, random_state=42).split(x, y))


def evaluate(name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, splits, alpha: float, model_name: str):
    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        model = clone(estimator(alpha, model_name))
        model.fit(x[train], y[train])
        prediction = model.predict(x[test])
        rows.append(
            {
                "model": name,
                "fold": fold,
                "train_wafers": len(train),
                "test_wafers": len(test),
                "test_lots": ",".join(map(str, sorted(set(groups[test])))),
                "rmse": mean_squared_error(y[test], prediction) ** 0.5,
                "mae": mean_absolute_error(y[test], prediction),
                "r2_global": r2_score(y[test].ravel(), prediction.ravel()),
                "r2_mean_location": r2_score(y[test], prediction, multioutput="uniform_average"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, coordinates = load_wafer_arrays(args.data_dir / "Si_Oxide_etch_89_points.csv", args.target)
    processes, process_names = load_process_features(
        args.data_dir / "Process_data.nc", args.data_dir / "Dictionary_process.nc", args.time_steps
    )

    matched = [record for record in records if group_name(record["experiment_key"]) in processes]
    spatial = np.stack([record["spatial"] for record in matched])
    process_sequence = np.stack([processes[group_name(record["experiment_key"])] for record in matched])
    # Ridge is a small-data baseline. Summaries avoid fitting 1,984 process
    # coefficients from only 88 wafers. The 64-step sequences are retained for
    # the later temporal fusion model.
    process = np.concatenate(
        [
            process_sequence.mean(axis=1),
            process_sequence.std(axis=1),
            process_sequence.min(axis=1),
            process_sequence.max(axis=1),
        ],
        axis=1,
    )
    target = np.stack([record["target"] for record in matched])
    groups = np.asarray([record["lot"] for record in matched])

    inputs = {
        "mean_map_baseline": np.ones((len(matched), 1)),
        "spatial_only": spatial,
        "process_only": process,
        "fused_spatial_process": np.concatenate([spatial, process], axis=1),
    }

    metric_rows = []
    splits = make_splits(spatial, target, groups, args.folds, args.split_mode)
    for name, x in inputs.items():
        if name != "mean_map_baseline":
            metric_rows.extend(evaluate(name, x, target, groups, splits, args.alpha, args.model))
            continue
        for fold, (train, test) in enumerate(splits, start=1):
            prediction = np.repeat(target[train].mean(axis=0, keepdims=True), len(test), axis=0)
            metric_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_wafers": len(train),
                    "test_wafers": len(test),
                    "test_lots": ",".join(map(str, sorted(set(groups[test])))),
                    "rmse": mean_squared_error(target[test], prediction) ** 0.5,
                    "mae": mean_absolute_error(target[test], prediction),
                    "r2_global": r2_score(target[test].ravel(), prediction.ravel()),
                    "r2_mean_location": r2_score(target[test], prediction, multioutput="uniform_average"),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary = metrics.groupby("model")[["rmse", "mae", "r2_global", "r2_mean_location"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "summary_metrics.csv")

    exact_spatial_duplicates = int(pd.DataFrame(spatial).duplicated(keep=False).sum())
    exact_process_duplicates = int(
        pd.DataFrame(process_sequence.reshape(len(matched), -1)).duplicated(keep=False).sum()
    )
    audit = {
        "target": args.target,
        "model_family": args.model,
        "measurement_wafers": len(records),
        "matched_wafers": len(matched),
        "lots": sorted(map(int, set(groups))),
        "spatial_input_shape": list(spatial.shape),
        "process_sequence_shape": [len(matched), len(process_names), args.time_steps],
        "process_ridge_summary_shape": list(process.shape),
        "target_shape": list(target.shape),
        "process_signals": len(process_names),
        "time_steps_per_signal": args.time_steps,
        "coordinates": coordinates.tolist(),
        "exact_duplicate_spatial_inputs": exact_spatial_duplicates,
        "exact_duplicate_process_inputs": exact_process_duplicates,
        "validation": (
            f"{args.folds}-fold GroupKFold by lot"
            if args.split_mode == "lot"
            else f"{args.folds}-fold shuffled KFold by wafer (random_state=42)"
        ),
        "r2_definition": "sklearn.metrics.r2_score; can be negative and has maximum 1",
    }
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(summary.to_string())
    print(json.dumps({key: audit[key] for key in audit if key != "coordinates"}, indent=2))


if __name__ == "__main__":
    main()
