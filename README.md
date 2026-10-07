# Spatial wafer profile prediction

This repository contains the reproducible modelling work for predicting an
outgoing wafer thickness profile from an incoming spatial profile and temporal
process measurements. The public BOSCH plasma-etching dataset is used as a
development proxy while internal CMP data is unavailable; these results are
not final CMP validation.

## Modelling task

- Independent samples: 88 wafers from 10 lots.
- Spatial input: 88 incoming thickness measurements. The original layout has
  89 points; the centre point is excluded.
- Process input: 31 process signals resampled to 64 ordered time steps.
- Output: outgoing thickness at the same 88 spatial locations.
- Primary target: `postox_thickness`.
- Evaluation: five-fold out-of-fold wafer validation with 18, 18, 18, 17 and
  17 test wafers.
- Wafer-level R²: calculated separately across the 88 locations of each test
  wafer. Wafer profiles are not concatenated for this metric.

## Data

The source files are included in `work/public_data/` and originate from:

<https://zenodo.org/records/17122442>

Files:

- `Si_Oxide_etch_89_points.csv`
- `Process_data.nc`
- `Dictionary_process.nc`

## Models

- Mean-profile reference.
- Multi-output Ridge regression.
- Extra Trees fusion benchmark.
- Flat residual MLP–Conv1D temporal fusion.
- Coordinate-point fusion using thickness, x, y and radius.
- Graph-feature fusion using each location's six nearest physical neighbours.
- Region-aware graph fusion with separate centre, middle and edge heads.
- Mean-shape graph fusion with separate wafer-mean and zero-mean shape heads.
- Fixed equal-weight ensemble of the flat MLP–Conv1D and mean-shape graph model.
- Controlled incoming-profile smoothing using a fixed 25% six-neighbour blend.
- Learnable graph-attention spatial encoder combined with a temporal 1D CNN.

## Main wafer-level results

| Model | RMSE µm | Mean wafer R² | Median wafer R² | Positive wafer R² |
|---|---:|---:|---:|---:|
| Extra Trees fusion | 0.0391 | 0.583 | 0.742 | 92.0% |
| Equal-weight deep ensemble | 0.0411 | 0.537 | 0.730 | 85.2% |
| Smoothed equal-weight ensemble | 0.0410 | 0.538 | 0.728 | 86.4% |
| Graph attention–1D CNN | 0.0540 | 0.272 | 0.516 | 77.3% |
| Flat MLP–Conv1D fusion | 0.0420 | 0.519 | 0.669 | 86.4% |
| Graph-feature fusion | 0.0450 | 0.460 | 0.677 | 84.1% |
| Coordinate-point fusion | 0.0534 | 0.275 | 0.384 | 81.8% |

The complete fold-by-fold wafer-level R² values are in
[`outputs/phase1_benchmark_2026-09-30/phase1_per_fold_wafer_metrics.csv`](outputs/phase1_benchmark_2026-09-30/phase1_per_fold_wafer_metrics.csv).

## Run the experiments

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the baseline models:

```bash
python src/run_baselines.py --data-dir work/public_data --output-dir outputs/baseline_postox
```

Run the original residual MLP–Conv1D fusion network:

```bash
python src/run_deep_fusion.py \
  --data-dir work/public_data \
  --output-dir outputs/deep_fusion_run
```

Run the coordinate and graph-feature experiments:

```bash
python src/run_coordinate_fusion.py \
  --data-dir work/public_data \
  --output-dir outputs/coordinate_fusion_run
```

Run the graph-neighbourhood sensitivity comparison:

```bash
python src/run_graph_neighbour_sweep.py \
  --data-dir work/public_data \
  --output-dir outputs/graph_neighbour_sweep_run
```

Run the centre, middle and edge diagnostic models:

```bash
python src/run_region_aware_fusion.py \
  --data-dir work/public_data \
  --output-dir outputs/region_aware_run
```

Run the mean-shape multi-task model and fixed deep ensemble:

```bash
python src/run_shape_decomposed_fusion.py \
  --data-dir work/public_data \
  --output-dir outputs/shape_decomposed_run
```

Run the fixed incoming-profile smoothing test:

```bash
python src/run_smoothed_shape_fusion.py \
  --data-dir work/public_data \
  --output-dir outputs/smoothed_shape_run
```

Run the learnable graph-attention and temporal 1D CNN model:

```bash
python src/run_graph_attention_fusion.py \
  --data-dir work/public_data \
  --output-dir outputs/graph_attention_run
```

Rebuild the common benchmark and per-fold wafer-level metrics:

```bash
python src/run_phase1_benchmark.py
```

## Result files

- `outputs/phase1_benchmark_2026-09-30/phase1_model_benchmark.csv` contains the
  overall comparison.
- `outputs/phase1_benchmark_2026-09-30/phase1_per_fold_wafer_metrics.csv`
  contains mean and median wafer-level R² for every model and fold.
- `outputs/coordinate_fusion_2026-09-30/` contains out-of-fold predictions,
  model metadata and the representative spatial output.
- `outputs/graph_neighbour_sweep_2026-09-30/graph_neighbour_summary.csv`
  compares three, six and ten neighbours using uniform and inverse-distance
  weighting. Uniform six-neighbour averaging has the best RMSE and mean
  wafer-level R² among these graph variants.
- `outputs/deep_fusion_2026-09-23/` contains the previous temporal fusion run.
- `outputs/region_aware_2026-10-07/` contains centre, middle and edge error
  metrics and the predicted-versus-measured radial diagnostic.
- `outputs/shape_decomposed_2026-10-07/shape_model_summary.csv` contains the
  mean-shape model and fixed equal-weight deep-ensemble results.
- `outputs/smoothed_shape_2026-10-07/` contains the fixed smoothing test,
  fold metrics, regional metrics and out-of-fold predictions.
- `outputs/graph_attention_2026-10-07/` contains the graph-attention results,
  regional metrics and out-of-fold predictions.

Progress reports are intentionally excluded from this repository and retained
locally.
