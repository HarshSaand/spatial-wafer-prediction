# Wafer thickness surrogate explorer

Open this project in VS Code and run:

```bash
python -m pip install -r requirements.txt -r demo/requirements.txt
python -m streamlit run demo/app.py
```

The app reads the public source data in `work/public_data/`, fits a residual Extra Trees demonstration model on all 88 wafers, and shows actual, predicted and scenario thickness at 88 wafer positions. Use the wafer selector and process-family slider to inspect model sensitivity. Scenario changes are perturbations of sensor summary features, not physically validated recipe changes. In-app predictions for selected wafers are training-set demonstrations; benchmark accuracy must be read from the held-out results in `outputs/wafer_improvement_2026-11-06/summary.csv`. That benchmark is exploratory because model settings were compared on the same folds.

A nested setting-selection check is in `outputs/wafer_improvement_2026-11-06/nested_summary.csv`; it gives mean wafer R² 0.667 on outer wafer folds. This does not test unseen lots.

Source dataset: Sayyed et al., BOSCH plasma-etch dataset, [Zenodo record 17122442](https://zenodo.org/records/17122442). Incoming maps were interpolated from 15 measurements. Some outgoing oxide readings were filled by the data creators.
