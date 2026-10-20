# Published fixed-model results (internal holdout)

The following values are copied from the archived result CSV files. They have been **inspected previously** and should not be used to tune or reselect models.

## Numerical AQI regression (n = 4,707)

| Prespecified model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| **log-HGB + A5 (primary)** | 16.937 | 29.665 | 0.690 |
| log-HGB + A4 (comparator) | 16.975 | 29.553 | 0.692 |
| Huber stack A4 (comparator) | **16.872** | **29.037** | **0.703** |

MAE contrasts: HGB+A5 − HGB+A4 = −0.0380 [−0.1545, 0.0731]; Huber − HGB+A5 = −0.0645 [−0.2476, 0.1059]. These conditional descriptive intervals include zero; numerical differences cannot establish stable superiority.

## Direct six-class classification (n = 4,707)

| Prespecified model | Macro-F1 ↑ | Accuracy ↑ | Balanced accuracy ↑ |
|---|---:|---:|---:|
| **LightGBM + A4 (primary)** | 0.4977 | **0.7038** | 0.4829 |
| Balanced RF + A4 (safety comparator) | **0.5537** | 0.6892 | **0.5966** |
| HGB+A5 → AQI categories (comparator) | 0.4609 | 0.6926 | 0.4541 |

On the **29** Hazardous observations, the primary LightGBM detected **10**, whereas balanced RF detected **17**. RF achieved higher severe-class recall at the cost of lower Hazardous precision; this is a trade-off, not a new post-holdout primary selection.

## Geographic transfer and limitations

On three separate country-held-out development folds, HGB+A5 MAE was approximately **32.855**, **28.282**, and **18.910**; LightGBM macro-F1 was **0.307**, **0.290**, and **0.421**. These demonstrate substantial geographic sensitivity, not validated predictions for every unseen location.

The source has no sufficient time series/station history for forecasting. All results are from a single source dataset that had previous exploratory scrutiny; the outer holdout is **not independent external validation**.

## Canonical machine-readable records

- `results/batch/final_regression_fixed.csv`
- `results/batch/final_classification_fixed.csv`
- `results/batch/final_paired_pattern_CIs.csv`
- `results/batch/final_six_class_metrics.csv`
- `results/batch/final_calibration_bins.csv`
- `results/batch/country_transfer_fixed_models.csv`
- `results/batch/FINAL_HOLDOUT_RECEIPT.json`

Figures are stored in `results/batch/figures/`. Use `python scripts/verify_release.py` for a hash and numeric consistency check.
