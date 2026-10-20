# PM2.5 AQI v4 — Verified exploratory scientific results

**Date:** 2026-10-19. **Data:** the same unchanged 23,463-row CSV, SHA256 `e1fc6e682edb8a2102d87e182b844fc8cde49e10763c0ef4293e510bbe240f69`. **No outside dataset.** CPU experiments on Python 3.13.5, pandas 2.2.3, numpy 2.3.5. **Elapsed full CLI comparison: 465.8 seconds**.

> **Interpretation boundary:** all new results below are **exploratory grouped cross-validation** on 14,112 SCREENING records. The 4,644-row v3 development-confirmation and 4,707-row outer holdout are NOT scored, tuned on or ranked using their labels. This is not a fresh independent test of improvements. The v3 confirmation had already been inspected. Never use its outcomes to pick v4 models.

## Key findings

- **Regression:** HGB with target-independent engineered A5 features had mean 3-fold CV **MAE 17.217**, versus fixed log-HGB A4 **17.329**, within the **same v4 grouped folds**. Absolute decrease ~**0.111 AQI points (0.64%)**. Paired cluster-bootstrap (conditional screening) difference in MAE(A5)−MAE(A4) -0.111, 95% descriptive CI [-0.216, -0.015]. This does **not** account for candidate selection/multiple comparisons.
- **Classification:** LightGBM A4 had best *mean fold* macro-F1 **0.522**. The pre-existing balanced RF A4 reference achieved 0.517. For LightGBM−RF, the group-bootstrap difference in *pooled OOF* macro-F1 is **0.701**, 95% descriptive CI [0.469, 0.926], **including zero**. No independent evidence of a genuine superiority.
- **MLP:** MLP A4 achieved regression MAE 18.123 and classification macro-F1 0.361. A deep neural network is not automatically superior on this tabular dataset; its regression training raised occasional convergence warnings at the prespecified 125-iteration budget.
- **Country stress:** predeclared country-disjoint HGB MAE 19.040–32.788; balanced RF macro-F1 0.285–0.425. Geographic transfer is a major limitation.
- **Rare-category uncertainty:** v4 screening balanced RF Hazardous F1 0.371 on 121 samples; bootstrap 95% conditional interval [0.277, 0.471]. The earlier v3 confirmation F1 was substantially lower (0.114 on 22 cases): another warning against overstating stable severe-class performance.

## 1. Complete prespecified regression comparison (shared 3 folds)

| name | features | mae | mae_sd | rmse | r2 |
|---|---|---|---|---|---|
| log_hgb_A5 | A5 | 17.217 | 0.273 | 31.604 | 0.685 |
| log_hgb_A4 | A4 | 17.329 | 0.265 | 31.673 | 0.684 |
| lightgbm_A5 | A5 | 17.947 | 0.182 | 34.053 | 0.635 |
| lightgbm_A4 | A4 | 18.029 | 0.193 | 34.314 | 0.629 |
| mlp_A4 | A4 | 18.123 | 1.128 | 32.919 | 0.658 |
| xgboost_A5 | A5 | 18.243 | 0.232 | 31.028 | 0.697 |
| xgboost_A4 | A4 | 18.324 | 0.203 | 31.417 | 0.689 |
| extra_trees_A4 | A4 | 18.485 | 0.150 | 31.172 | 0.694 |
| catboost_A5 | A5 | 18.792 | 0.108 | 35.407 | 0.605 |
| catboost_A4 | A4 | 18.886 | 0.256 | 35.446 | 0.604 |


**Decision:** preserve HGB A4 as the reproducible fixed reference; A5 shows a small *exploratory* gain. LightGBM, CatBoost, XGBoost, Extra Trees and MLP were all run but did not beat HGB in MAE under these fixed settings. The models were not subjected to unlimited or equal-budget hyperparameter optimization, so these are comparisons of prespecified configurations, not algorithm families' theoretical upper bounds.

## 2. Complete prespecified direct-classification comparison

| name | features | macro_f1 | macro_f1_sd | accuracy | balanced_accuracy | ordinal_mae |
|---|---|---|---|---|---|---|
| lightgbm_A4 | A4 | 0.522 | 0.022 | 0.703 | 0.496 | 0.345 |
| rf_balanced_A4 | A4 | 0.517 | 0.022 | 0.682 | 0.565 | 0.379 |
| xgboost_A5 | A5 | 0.511 | 0.030 | 0.701 | 0.487 | 0.346 |
| xgboost_A4 | A4 | 0.507 | 0.033 | 0.699 | 0.487 | 0.350 |
| rf_balanced_A5 | A5 | 0.506 | 0.014 | 0.670 | 0.543 | 0.404 |
| hgb_balanced_A4 | A4 | 0.506 | 0.010 | 0.671 | 0.554 | 0.400 |
| lightgbm_A5 | A5 | 0.504 | 0.016 | 0.707 | 0.487 | 0.339 |
| catboost_A4 | A4 | 0.405 | 0.023 | 0.678 | 0.396 | 0.380 |
| catboost_A5 | A5 | 0.397 | 0.027 | 0.680 | 0.391 | 0.381 |
| mlp_A4 | A4 | 0.361 | 0.013 | 0.679 | 0.373 | 0.386 |
| rf_balanced_A1 | A1 | 0.314 | 0.008 | 0.481 | 0.335 | 1.006 |
| rf_balanced_A3 | A3 | 0.313 | 0.011 | 0.481 | 0.340 | 1.004 |


**Important:** means of per-fold nonlinear macro-F1 need not equal macro-F1 calculated from pooled OOF predictions. Paired bootstrap is computed on the pooled predictions of the same validation rows. Class imbalance weights matter: the high-accuracy LightGBM gives lower balanced accuracy than balanced RF, so selecting a model depends on the agreed clinical/environmental error objective, not accuracy alone.

## 3. Paired grouped uncertainty (descriptive, screening predictions)

| task | a | b | metric | estimate | CI_low | CI_high |
|---|---|---|---|---|---|---|
| regression | catboost_A4 | log_hgb_A4 | MAE(a)-MAE(b) | 1.558 | 1.250 | 1.865 |
| regression | xgboost_A4 | log_hgb_A4 | MAE(a)-MAE(b) | 0.995 | 0.785 | 1.206 |
| regression | lightgbm_A4 | log_hgb_A4 | MAE(a)-MAE(b) | 0.701 | 0.469 | 0.926 |
| regression | log_hgb_A5 | log_hgb_A4 | MAE(a)-MAE(b) | -0.111 | -0.216 | -0.015 |
| classification | catboost_A4 | rf_balanced_A4 | macro-F1(a)-macro-F1(b) | -0.112 | -0.140 | -0.087 |
| classification | xgboost_A4 | rf_balanced_A4 | macro-F1(a)-macro-F1(b) | -0.009 | -0.032 | 0.013 |
| classification | lightgbm_A4 | rf_balanced_A4 | macro-F1(a)-macro-F1(b) | 0.006 | -0.020 | 0.029 |
| classification | rf_balanced_A5 | rf_balanced_A4 | macro-F1(a)-macro-F1(b) | -0.013 | -0.025 | -0.001 |


Pattern-cluster resampling of paired predictions was repeated 350 times. This is **not** a formal multiple-comparisons-adjusted hypothesis test, not an unbiased selected-model CI, and not an independently sampled test result. Reused screening folds and prior research exploration impose a selection bias; avoid claiming statistical significance. In particular, A5 was one of several explored representations.

## 4. Direct classifier versus numeric-regression-to-class comparator

Six class categories are a deterministic thresholding of the *same* numeric PM2.5 AQI response. Both approaches were evaluated with identical grouped folds and only screening observations:

| method | macro_f1 | balanced_accuracy | accuracy | ordinal_mae | quadratic_kappa |
|---|---|---|---|---|---|
| fixed_log_HGB_regression_to_AQI_class | 0.477 | 0.461 | 0.701 | 0.334 | 0.793 |
| fixed_balanced_RF_direct_A4 | 0.517 | 0.565 | 0.682 | 0.379 | 0.783 |


The fixed comparison is **not** a procedure for choosing a new model from the previous v3 confirmation. Paired screening-cluster differences:

| metric | estimate | CI_low | CI_high |
|---|---|---|---|
| macro_F1_direct_RF_A4_minus_fixed_log_HGB_to_class | 0.040 | 0.012 | 0.064 |


## 5. All six classes, including severe/hazardous support and uncertainty

**Fixed** balanced RF A4 on pooled v4 screening OOF rows (not the best method chosen after searching):

| category | support | direct_RF_A4_F1 | F1_CI_low | F1_CI_high | direct_RF_A4_recall | recall_CI_low | recall_CI_high |
|---|---|---|---|---|---|---|---|
| Good | 6116 | 0.786 | 0.771 | 0.800 | 0.844 | 0.826 | 0.863 |
| Moderate | 5455 | 0.642 | 0.625 | 0.659 | 0.574 | 0.551 | 0.596 |
| Unhealthy for Sensitive Groups | 976 | 0.415 | 0.384 | 0.444 | 0.450 | 0.410 | 0.488 |
| Unhealthy | 1296 | 0.620 | 0.582 | 0.656 | 0.593 | 0.543 | 0.644 |
| Very Unhealthy | 148 | 0.266 | 0.180 | 0.330 | 0.304 | 0.186 | 0.419 |
| Hazardous | 121 | 0.371 | 0.277 | 0.471 | 0.628 | 0.500 | 0.731 |


Refer to `results/confusion_matrix.csv` and `results/figures/class_errors_and_calibration.png` for the complete 6×6 table and reliability diagram. Positive class support and severe misclassification must accompany macro-F1. Bootstrap intervals condition on the existing fitted models/folds and do not establish deployment recall.

### Safety-sensitive differences hidden by macro-F1

The two close competitors make **different high-risk errors**. On identical screening OOF rows, balanced RF A4 obtained **Hazardous recall 0.628** and **Unhealthy for Sensitive Groups recall 0.450**; LightGBM A4 obtained **Hazardous recall 0.372** and **sensitive-group recall 0.226**. Although LightGBM has slightly higher macro-F1 and accuracy, it missed more of these cases. `results/safety_tradeoffs_by_class.csv` compares per-class precision, recall, F1 and support for both models and XGBoost A5. Declare the intended loss from missed high-risk AQI categories before choosing a model; this is screening-only error analysis, not an independent public-health safety claim.

## 6. Calibration, threshold error and ordinal diagnostic

- Fixed balanced RF A4 multiclass Brier (sum-of-six-class formulation): **0.457**; ten-bin maximum-confidence ECE **0.161**. These probabilities are not demonstrably calibrated.
- **4.82%** of OOF RF predictions differed from the true class by two or more AQI categories.
- Near the five label boundaries (within 5 AQI units): **51.8%** accuracy on **2,797** rows; away from thresholds **72.2%** accuracy on **11,315** rows. This *descriptive* grouping uses numeric PM2.5 targets only after out-of-fold predictions to stratify errors, never as a feature.
- HGB A4 absolute residual 90th percentile: **35.32 AQI points**, a screening residual descriptor **not** a conformal prediction guarantee.

## 7. Geographic transfer, Country importance, ablation and seed checks

Country-grouped stress (predeclared reference estimators, not used in model selection):

| fold | task | n_validation | mae | macro_f1 | accuracy |
|---|---|---|---|---|---|
| 1 | regression | 6871 | 32.788 | — | — |
| 1 | classification | 6871 | — | 0.306 | 0.449 |
| 2 | regression | 1723 | 28.212 | — | — |
| 2 | classification | 1723 | — | 0.285 | 0.530 |
| 3 | regression | 5518 | 19.040 | — | — |
| 3 | classification | 5518 | — | 0.425 | 0.662 |


Fold-local permutation importance of **admitted raw predictors**, using negative MAE:

| feature | mae_increase | sd |
|---|---|---|
| CO AQI Value | 11.973 | 0.157 |
| Ozone AQI Value | 6.222 | 0.184 |
| NO2 AQI Value | 4.095 | 0.095 |
| Country | 15.717 | 0.120 |


The Country predictor has large predictive importance **within observed countries**, but poor country-held-out transfer suggests dependence on geographical identity; this is a correlational association, not a causal pollution mechanism. Feature-set ablation: `results/figures/feature_ablation.png` compares A1 (gases only), A3 (gases plus gas interactions), A4 (A3 plus Country), A5 (A4 plus safe transformations).

Seed robustness: fixed HGB log A4 and RF A4 on 2 folds each for seeds 42/43/44; raw scores:

| seed | fold | task | mae | macro_f1 | accuracy |
|---|---|---|---|---|---|
| 42 | 1 | regression | 17.272 | — | — |
| 42 | 1 | classification | — | 0.514 | 0.689 |
| 42 | 2 | regression | 18.067 | — | — |
| 42 | 2 | classification | — | 0.510 | 0.668 |
| 43 | 1 | regression | 17.464 | — | — |
| 43 | 1 | classification | — | 0.512 | 0.689 |
| 43 | 2 | regression | 17.909 | — | — |
| 43 | 2 | classification | — | 0.503 | 0.669 |
| 44 | 1 | regression | 17.623 | — | — |
| 44 | 1 | classification | — | 0.492 | 0.677 |
| 44 | 2 | regression | 17.961 | — | — |
| 44 | 2 | classification | — | 0.516 | 0.679 |


## 8. What can and cannot be claimed about improvement

- The **scientific workflow** is more comprehensive and more defensible than original post-test re-ranking; this is not automatically higher independent predictive accuracy.
- Within this exploratory screening protocol, A5 HGB moderately improved regression MAE over A4; the apparent LightGBM classification gain over RF was too uncertain to rely on.
- Original manuscript row-random historical HGB test MAE **16.549** and original classification macro-F1 **0.5258** were obtained under **different and already-inspected partitions**. Neither can be substituted for or compared directly with the v4 screening metrics.
- The v3 confirmation has already been examined; reusing it to choose models would be test/confirmation-driven selection. v4 intentionally leaves it and the outer holdout alone.
- Unmeasured time/site/weather/meteorological effects, dataset representativeness, small rare classes, near-unique city identifiers, hidden geographic dependencies and missing external validation remain.
- Using `AQI Value` would severely inflate accuracy: it equals the PM2.5 numeric target in **78.34%** of rows and approximately the maximum of available pollutant AQI columns in **99.40%**. Thus it is *not* admitted to predictive features.
- Phrases like “tomorrow's PM2.5 prediction”, “real-time forecasts” or “causal pollutant effects” are unsupported by the input dataset.

## 9. Article reporting checklist (after supervisor approval)

1. Define the **cross-sectional prediction** outcome and six integer AQI boundaries; include dataset source/provenance and original checksum.
2. State raw predictors and A1–A5 transformations; exclude global AQI/target/category proxy columns and document why.
3. Describe feature-only group-disjoint screening/confirmation/outer split, and note that confirmation was already inspected in v3 and **not reused** for v4 experiments.
4. Prespecify primary metric MAE for numeric regression and macro-F1 for class prediction; present RMSE, R², class-specific support/F1/recall, balanced accuracy, ordinal errors, reliability/ECE and Brier as secondary.
5. Specify all considered algorithms, software versions, seeds, fixed hyperparameters, fold design, and no test-set re-ranking.
6. Provide complete model ranking tables **without implying they are independent test performance estimates**.
7. Show all six-category confusion matrix, threshold analysis, rare-category uncertainty, country stress, and input importance caveats.
8. Describe uncertainty and repeated-fold multiple-comparisons limitations. Only state an improvement beyond exploratory screening with a truly independent, preregistered later test.
9. Contrast with Singh et al. (2026), tabular NN benchmarks and temporal Transformer/AQI literature. Methods relying on temporal sensors are not transferable to this dataset.
10. Report code availability, dependency versions, and data redistribution rights. No manuscript alteration was performed.

See `LITERATURE_AND_METHODS.md` for scientific references and transferability analysis. All numerical CSVs are in `results/`; all four figure PNGs are in `results/figures/`.
