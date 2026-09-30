# Deep fusion MLP result

The compact two-branch network used 88 incoming spatial measurements and 124
fold-safe process summaries derived from 31 signals over 64 time steps. It
predicted all 88 outgoing wafer locations. Five-fold shuffled wafer-level
cross-validation included an inner validation subset for early stopping.

Out-of-fold result:

- RMSE: 0.04658 micrometres
- MAE: 0.02241 micrometres
- Global R-squared: 0.6283
- Mean location-wise R-squared: 0.5640
- Trainable parameters: 28,120

For comparison, the existing fused Extra Trees result was 0.0390 micrometres
RMSE and 0.718 global R-squared under the same outer five-fold split. The deep
model therefore demonstrates a working nonlinear neural pipeline, but it does
not yet outperform Extra Trees on only 88 wafers. This is an expected and useful
small-data finding rather than evidence that the neural implementation failed.

`out_of_fold_predictions.csv` contains every held-out actual value, prediction,
coordinate, and residual for reproducible plots and error analysis.
