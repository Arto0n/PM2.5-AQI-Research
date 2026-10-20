# PM2.5 AQI: prespecified batch validation and publication audit

## Scope and status

This is a *fixed-candidate single-pass outer-holdout evaluation*, not a newly tuned
all-model test leaderboard. Exact dataset fingerprint: `e1fc6e682edb8a2102d87e182b844fc8cde49e10763c0ef4293e510bbe240f69`.

The locked protocol (`LOCKED_PROTOCOL.json`, SHA-256 `3a44256ae396182495f61f8c293572456b465b92d2807adf045af3d85977f556`)
preceded all new holdout-label access. It freezes the model families, feature sets,
primary metrics, safety contrasts, data partition and training plan. Do not use
these results to revise the candidates or modify the feature sets and then rescore
this same holdout. An evaluation receipt prevents automatic repetition.

**Critical provenance caveat:** the original historical random-row test was
already inspected in earlier research and the v3 confirmation labels were also
inspected. This new 4,707-row outer split is pattern-disjoint, but it comes from
the same dataset and can share original rows or countries with those earlier
inspected partitions. Its results are **not independent external validation**.

## Dataset, protocol and model-selection contract

- Dataset: **23,463** records; three gaseous pollutant AQIs and `Country` permitted.
- Outer holdout: **4,707** exact-pattern-disjoint rows; development: **18,756**.
- Development: screening **14,112** (candidate selection and all exploratory CV);
  prior v3 confirmation **4,644** (previously inspected, not used for v4 selection).
- Final fit: **all 18,756 development rows** after freezing candidate choices.
- Primary regression: **HGB on log target with A5 features**, primary metric MAE.
  Prespecified comparators: same HGB+A4, and Huber stacked ridge/ExtraTrees/HGB+A4.
- Primary direct classification: **LightGBM+A4**, primary metric six-class Macro-F1.
  Safety comparator: **balanced RF+A4**, including Hazardous recall;
  additional fixed comparator: HGB+A5 numerical prediction mapped to AQI categories.
- 95% intervals: input-pattern cluster bootstrap, **descriptive conditional intervals**;
  they do not account for hyperparameter screening, geographic clustering, the
  earlier historical test examination, or model-fitting uncertainty.
- No model is selected based on holdout outcomes, irrespective of which comparator wins.

## Table 1 — one-shot fixed regression holdout

| model       |    n |     mae |    rmse |     r2 |
|:------------|-----:|--------:|--------:|-------:|
| log_hgb_A5  | 4707 | 16.937  | 29.6646 | 0.69   |
| log_hgb_A4  | 4707 | 16.975  | 29.5531 | 0.6923 |
| huber_stack | 4707 | 16.8725 | 29.0372 | 0.7029 |

## Table 2 — one-shot fixed classification holdout

| model           |    n |   macro_f1 |   weighted_f1 |   accuracy |   balanced_accuracy |   ordinal_mae |   quadratic_kappa |
|:----------------|-----:|-----------:|--------------:|-----------:|--------------------:|--------------:|------------------:|
| lightgbm_A4     | 4707 |     0.4977 |        0.6914 |     0.7038 |              0.4829 |        0.3397 |            0.7818 |
| rf_balanced_A4  | 4707 |     0.5537 |        0.6896 |     0.6892 |              0.5966 |        0.3535 |            0.7896 |
| hgb_A5_to_class | 4707 |     0.4609 |        0.6854 |     0.6926 |              0.4541 |        0.3369 |            0.7894 |

## Table 3 — preregistered paired holdout comparisons

| task           | a           | b               | metric                        |   point_estimate |   ci_low |   ci_high |
|:---------------|:------------|:----------------|:------------------------------|-----------------:|---------:|----------:|
| regression     | log_hgb_A5  | log_hgb_A4      | MAE(a)-MAE(b)                 |          -0.038  |  -0.1545 |    0.0731 |
| regression     | huber_stack | log_hgb_A5      | MAE(a)-MAE(b)                 |          -0.0645 |  -0.2476 |    0.1059 |
| classification | lightgbm_A4 | rf_balanced_A4  | Macro-F1(a)-Macro-F1(b)       |          -0.056  |  -0.095  |   -0.0185 |
| classification | lightgbm_A4 | hgb_A5_to_class | Macro-F1(a)-Macro-F1(b)       |           0.0368 |  -0.0044 |    0.0736 |
| safety         | lightgbm_A4 | rf_balanced_A4  | Hazardous recall(a)-recall(b) |          -0.2414 |  -0.4033 |   -0.0968 |

**Reading paired intervals:** Regression contrasts are MAE(a)-MAE(b), so
negative is in favor of A. Macro-F1 and recall contrasts are A minus B,
so positive favors A. These intervals are exploratory/descriptive, not
proof of generalization superiority or independent hypothesis tests.

## Table 4 — absolute holdout metrics with conditional pattern-bootstrap intervals

| task           | model           | metric           |   point_estimate |   ci_low |   ci_high |
|:---------------|:----------------|:-----------------|-----------------:|---------:|----------:|
| regression     | log_hgb_A5      | MAE              |          16.937  |  15.7859 |   18.1423 |
| regression     | log_hgb_A4      | MAE              |          16.975  |  15.8283 |   18.121  |
| regression     | huber_stack     | MAE              |          16.8725 |  15.7419 |   17.9707 |
| classification | lightgbm_A4     | Macro-F1         |           0.4977 |   0.4534 |    0.5374 |
| classification | lightgbm_A4     | Hazardous recall |           0.3448 |   0.1507 |    0.5304 |
| classification | rf_balanced_A4  | Macro-F1         |           0.5537 |   0.5158 |    0.5891 |
| classification | rf_balanced_A4  | Hazardous recall |           0.5862 |   0.375  |    0.7778 |
| classification | hgb_A5_to_class | Macro-F1         |           0.4609 |   0.437  |    0.4875 |
| classification | hgb_A5_to_class | Hazardous recall |           0      |   0      |    0      |

## Table 5 — six-class classification performance

| model           | category                       |   support |   precision |   recall |     f1 |
|:----------------|:-------------------------------|----------:|------------:|---------:|-------:|
| lightgbm_A4     | Good                           |      1998 |      0.7382 |   0.8328 | 0.7827 |
| lightgbm_A4     | Moderate                       |      1880 |      0.6944 |   0.667  | 0.6804 |
| lightgbm_A4     | Unhealthy for Sensitive Groups |       330 |      0.5211 |   0.2242 | 0.3136 |
| lightgbm_A4     | Unhealthy                      |       420 |      0.6538 |   0.7286 | 0.6892 |
| lightgbm_A4     | Very Unhealthy                 |        50 |      0.5556 |   0.1    | 0.1695 |
| lightgbm_A4     | Hazardous                      |        29 |      0.3571 |   0.3448 | 0.3509 |
| rf_balanced_A4  | Good                           |      1998 |      0.7409 |   0.8258 | 0.7811 |
| rf_balanced_A4  | Moderate                       |      1880 |      0.718  |   0.6122 | 0.6609 |
| rf_balanced_A4  | Unhealthy for Sensitive Groups |       330 |      0.3771 |   0.4697 | 0.4184 |
| rf_balanced_A4  | Unhealthy                      |       420 |      0.7278 |   0.5857 | 0.6491 |
| rf_balanced_A4  | Very Unhealthy                 |        50 |      0.3378 |   0.5    | 0.4032 |
| rf_balanced_A4  | Hazardous                      |        29 |      0.3148 |   0.5862 | 0.4096 |
| hgb_A5_to_class | Good                           |      1998 |      0.7257 |   0.8764 | 0.7939 |
| hgb_A5_to_class | Moderate                       |      1880 |      0.724  |   0.6043 | 0.6587 |
| hgb_A5_to_class | Unhealthy for Sensitive Groups |       330 |      0.3653 |   0.4273 | 0.3939 |
| hgb_A5_to_class | Unhealthy                      |       420 |      0.7432 |   0.5167 | 0.6096 |
| hgb_A5_to_class | Very Unhealthy                 |        50 |      0.3191 |   0.3    | 0.3093 |
| hgb_A5_to_class | Hazardous                      |        29 |      0      |   0      | 0      |

## Table 6 — model performance near AQI category boundaries

| model           | zone                     |    n |   accuracy |   ordinal_ge2_rate |
|:----------------|:-------------------------|-----:|-----------:|-------------------:|
| lightgbm_A4     | within_5_AQI_of_boundary |  935 |     0.5241 |             0.0481 |
| lightgbm_A4     | outside_boundary_band    | 3772 |     0.7484 |             0.0353 |
| rf_balanced_A4  | within_5_AQI_of_boundary |  935 |     0.4984 |             0.0385 |
| rf_balanced_A4  | outside_boundary_band    | 3772 |     0.7365 |             0.0334 |
| hgb_A5_to_class | within_5_AQI_of_boundary |  935 |     0.4845 |             0.0299 |
| hgb_A5_to_class | outside_boundary_band    | 3772 |     0.7442 |             0.0252 |

## Table 7 — seven nested stacking methods on screening only

| model          |     mae |    rmse |     r2 |
|:---------------|--------:|--------:|-------:|
| huber          | 17.2589 | 31.3987 | 0.6883 |
| inverse_mae    | 17.5624 | 31.421  | 0.6879 |
| mean           | 17.6117 | 31.5059 | 0.6862 |
| simplex        | 17.6909 | 30.978  | 0.6963 |
| positive_ridge | 17.7706 | 30.8389 | 0.6991 |
| elastic_net    | 17.783  | 30.8443 | 0.699  |
| ridge          | 17.7831 | 30.8443 | 0.699  |

These stacking scores were obtained on screening observations only.
Huber stacking was designated a fixed secondary comparator *before* test access;
its newer screening result is not allowed to change final primary-model selection.

## Table 8 — predeclared country-held-out development checks

| model          |   fold |   n_validation |      mae |     rmse |       r2 |   macro_f1 |   weighted_f1 |   balanced_accuracy |   accuracy |   ordinal_mae |   quadratic_kappa |
|:---------------|-------:|---------------:|---------:|---------:|---------:|-----------:|--------------:|--------------------:|-----------:|--------------:|------------------:|
| log_hgb_A5     |      1 |           6871 |  32.8552 |  61.6042 |   0.1592 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |      1 |           6871 | nan      | nan      | nan      |     0.307  |        0.4826 |              0.2946 |     0.4967 |        0.6944 |            0.4168 |
| rf_balanced_A4 |      1 |           6871 | nan      | nan      | nan      |     0.3059 |        0.4489 |              0.3043 |     0.4494 |        0.7353 |            0.4598 |
| log_hgb_A5     |      2 |           1723 |  28.2824 |  45.6929 |   0.2654 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |      2 |           1723 | nan      | nan      | nan      |     0.2901 |        0.527  |              0.2939 |     0.5572 |        0.5862 |            0.4911 |
| rf_balanced_A4 |      2 |           1723 | nan      | nan      | nan      |     0.285  |        0.5017 |              0.2977 |     0.5305 |        0.6053 |            0.5079 |
| log_hgb_A5     |      3 |           5518 |  18.9096 |  25.3162 |   0.3333 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |      3 |           5518 | nan      | nan      | nan      |     0.4207 |        0.6118 |              0.3922 |     0.6109 |        0.4087 |            0.4703 |
| rf_balanced_A4 |      3 |           5518 | nan      | nan      | nan      |     0.4248 |        0.6439 |              0.4117 |     0.6618 |        0.3693 |            0.5103 |

## Table 9 — predeclared seed checks (screening only)

| model          |   seed |   fold |   n_validation |      mae |     rmse |       r2 |   macro_f1 |   weighted_f1 |   balanced_accuracy |   accuracy |   ordinal_mae |   quadratic_kappa |
|:---------------|-------:|-------:|---------------:|---------:|---------:|---------:|-----------:|--------------:|--------------------:|-----------:|--------------:|------------------:|
| log_hgb_A5     |     42 |      1 |           7057 |  17.2469 |  31.4212 |   0.6897 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |     42 |      1 |           7057 | nan      | nan      | nan      |     0.5139 |        0.6899 |              0.4902 |     0.702  |        0.3422 |            0.7887 |
| rf_balanced_A4 |     42 |      1 |           7057 | nan      | nan      | nan      |     0.5136 |        0.6865 |              0.5517 |     0.689  |        0.3684 |            0.7886 |
| log_hgb_A5     |     42 |      2 |           7055 |  17.9871 |  32.7149 |   0.662  |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |     42 |      2 |           7055 | nan      | nan      | nan      |     0.5073 |        0.6853 |              0.4785 |     0.6953 |        0.3536 |            0.773  |
| rf_balanced_A4 |     42 |      2 |           7055 | nan      | nan      | nan      |     0.5104 |        0.6701 |              0.5398 |     0.6683 |        0.3949 |            0.7637 |
| log_hgb_A5     |     43 |      1 |           7056 |  17.1976 |  30.6782 |   0.7041 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |     43 |      1 |           7056 | nan      | nan      | nan      |     0.5228 |        0.6946 |              0.4954 |     0.7073 |        0.3396 |            0.7838 |
| rf_balanced_A4 |     43 |      1 |           7056 | nan      | nan      | nan      |     0.5116 |        0.689  |              0.5678 |     0.6888 |        0.3798 |            0.7733 |
| log_hgb_A5     |     43 |      2 |           7056 |  17.7541 |  32.9748 |   0.6567 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |     43 |      2 |           7056 | nan      | nan      | nan      |     0.4885 |        0.6823 |              0.4638 |     0.6942 |        0.3576 |            0.7697 |
| rf_balanced_A4 |     43 |      2 |           7056 | nan      | nan      | nan      |     0.5034 |        0.6696 |              0.5392 |     0.6685 |        0.4004 |            0.7603 |
| log_hgb_A5     |     44 |      1 |           7055 |  17.5962 |  31.5294 |   0.6842 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |     44 |      1 |           7055 | nan      | nan      | nan      |     0.5109 |        0.6909 |              0.4835 |     0.7026 |        0.3423 |            0.7825 |
| rf_balanced_A4 |     44 |      1 |           7055 | nan      | nan      | nan      |     0.4921 |        0.6759 |              0.5233 |     0.6773 |        0.3819 |            0.7761 |
| log_hgb_A5     |     44 |      2 |           7057 |  17.9007 |  32.5666 |   0.6686 |   nan      |      nan      |            nan      |   nan      |      nan      |          nan      |
| lightgbm_A4    |     44 |      2 |           7057 | nan      | nan      | nan      |     0.5128 |        0.687  |              0.4899 |     0.6994 |        0.3493 |            0.7837 |
| rf_balanced_A4 |     44 |      2 |           7057 | nan      | nan      | nan      |     0.5159 |        0.6781 |              0.5613 |     0.6786 |        0.3881 |            0.7721 |

## Calibration, class rarity and geographic interpretation

- LightGBM 10-bin ECE on outer holdout: **0.0199**.
- LightGBM multiclass Brier on outer holdout: **0.3918**.
- Outer holdout class counts: {'Good': 1998, 'Moderate': 1880, 'Unhealthy for Sensitive Groups': 330, 'Unhealthy': 420, 'Very Unhealthy': 50, 'Hazardous': 29}.
- Severe categories have few examples; uncertainty is high. In addition,
  input-pattern cluster bootstrap does not capture dependence among countries.
- The model estimates category/AQI cross-sectionally. **This dataset lacks
  timestamps and station/meteorological covariates and cannot justify time-series
  forecasting or sensor-station external generalization claims.**
- Do not promote the best holdout comparator to a new selected model. If the
  primary model loses on a secondary metric, report that result unchanged.

## Paper-ready figure files

- `results/batch/figures/FINAL_fixed_models.png`
- `results/batch/figures/FINAL_confusion_matrices.png`
- `results/batch/figures/FINAL_per_category_recall.png`
- `results/batch/figures/FINAL_calibration.png`
- `results/batch/figures/DEVELOPMENT_country_transfer.png`
- `results/batch/figures/DEVELOPMENT_seven_stackers.png`

## Recommended reporting and reviewer considerations

1. Explain why exact-pattern-disjoint splitting was chosen, distinguish it from
   country-grouped validation and from the historical random-row protocol.
2. Give ALL six class supports and classwise precision/recall/F1; avoid citing
   accuracy alone for rare high-risk AQI labels.
3. Clearly state candidate-selection optimism: v4's 10 regression and 12
   classification candidates were searched on screening; paired screening CIs
   are conditional, not confirmatory.
4. The original paper's already-inspected test cannot be rehabilitated by
   splitting the same dataset again. Independent external monitoring data
   would still be needed for strong deployment/generalization claims.
5. Discuss that AQI thresholds are tied to numeric PM2.5 AQI; direct class
   and regression-derived predictions serve related but different objectives.
6. Share software/library versions and dataset fingerprint, and call the
   locked holdout pass a precommitted **internal** evaluation rather than
   claiming a never-before-seen external cohort.
7. Do not claim causal effects, future forecasting, or intervention policies.
8. Treat numerical model ranking on holdout as description of FIXED models,
   not as a basis for another tuning cycle.

## Reproduction safeguard

The package contains `LOCKED_PROTOCOL.json` and `results/batch/FINAL_HOLDOUT_RECEIPT.json`.
Calling the default batch runner again verifies these files and **skips holdout scoring**.
Test predictions and individual holdout labels are not exported for post-test tuning.
All processing other than the final fixed scoring used screening or frozen development rows.
