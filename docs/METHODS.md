# Methods and experimental contract

This repository implements **cross-sectional estimation of numerical PM2.5 AQI** and **six-category severity classification**. It does **not** forecast future PM2.5 concentration and does not infer causal effects.

## Inputs and leakage prevention

Admitted original predictors: `CO AQI Value`, `Ozone AQI Value`, `NO2 AQI Value`, `Country`. City, overall AQI, PM2.5 numerical target, PM2.5 category and pollutant category fields are excluded. In the source dataset overall `AQI Value` equals PM2.5 AQI in approximately 78.34% of rows, so using it would constitute target-proxy leakage.

Five feature sets are defined in `pm25aqi/config.py`:

- A1: three gaseous AQI values.
- A2: A1 + Country.
- A3: A1 + CO×NO2 and CO×Ozone interactions.
- A4: A3 + Country.
- A5: A4 + three log transforms, gas maximum/range/sum and normalized gas fractions.

Imputation, categorical encoding and scaling are fit only within the corresponding model training fold. Target transformation uses `log1p` with inverse `expm1` for the fixed HGB regressor; reported errors are original AQI units.

## Fixed partitions

- All rows: **23,463**.
- Pattern-disjoint development: **18,756**; fixed internal holdout **4,707**, using `GroupShuffleSplit(test_size=.20, random_state=42)` on exact raw input tuples.
- Development screening **14,112**; prior inspected confirmation **4,644**. New candidate screening excludes confirmation and holdout labels.
- Screening classifiers and regressors use grouped development CV; seven stackers use inner out-of-fold and outer grouped evaluation.
- Country-held-out CV is separate, more difficult geographic sensitivity assessment, not external validation.

## Prespecified models

The frozen model contract is `LOCKED_PROTOCOL.json`. It declares **log-HGB+A5** (primary numerical, MAE) and **LightGBM+A4** (primary six-class, macro-F1); fixed numerical comparators **log-HGB+A4** and **Huber stacking (A4)**, classification comparators **balanced RF+A4** and **HGB+A5 mapped to AQI categories**. The published primary model identity must not change after inspecting holdout results.

## Diagnostics

The study reports MAE, RMSE, R²; macro-F1, accuracy, balanced accuracy, ordinal tier error, rare-class precision/recall; paired and absolute exact-pattern-cluster bootstrap intervals; reliability and ten-bin ECE/Brier; category-boundary sensitivity; and country transfer. Intervals are **descriptive and conditional on the fitted models**; they do not account for historical partition inspection, all candidate screening or novel country/station dependence.

## Frozen evaluation and provenance

`results/batch/FINAL_HOLDOUT_RECEIPT.json` includes the SHA-256 of every fixed final result. `scripts/verify_release.py` verifies the original protocol, receipt and numerical table values **without accessing holdout labels**. `run_batch.py` may be used with the verified original dataset but will read the original receipt and **never automatically rescore the holdout**.

This is internal validation from one dataset. Results do not establish independent external, prospective temporal or causal generalization.
