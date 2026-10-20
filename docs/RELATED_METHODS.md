# External literature and method suitability (reviewed 2026-10-19)

The source dataset consists of 23,463 cross-sectional records, 3 independent gaseous AQI indicators, country, and PM2.5 numeric/class targets. It has **no timestamp, genuine free text, meteorology, station identifier, intervention, action, or reward**. The global `AQI Value` equals the PM2.5 numeric target in 78.34% of rows, and equals the largest supplied pollutant AQI in 99.40% of rows. Its inclusion would cause severe label-proxy leakage. The dataset's PM2.5 AQI *category* is deterministically constructed from its PM2.5 AQI *value*: direct classification and regression-derived classification are different representations of the SAME outcome.

## Method comparison with peer-reviewed literature

| Research | Methods/dataset | What transfers to this project | What **does not** transfer |
|---|---|---|---|
| Singh et al., Scientific Reports (2026), https://www.nature.com/articles/s41598-026-39232-w | GB, XGBoost, CatBoost, LightGBM, weighted ensemble, SHAP; Taiwan hourly monitoring with 6 pollutants, meteorology and time | Multiple GBDT families, fixed-fold comparisons, fold-local tuning, diagnostics | Their extremely high scores, temporal split, real forecasting, meteorological effects or external validity |
| Air Quality Forecasting Using Machine Learning, Water Air Soil Pollution (2025), https://link.springer.com/article/10.1007/s11270-025-08122-8 | LightGBM, CatBoost, SVR and Bayesian optimization; model ensembles | Model-family ablations, regression error comparisons, limited grouped search | Their optimized numeric scores and spatiotemporal claims |
| Stacked Transformer–XGBoost, iScience (2026), https://www.sciencedirect.com/science/article/pii/S2589004226017712 | Temporal feature extraction/temporal OOF stacks on chronological observations | OOF stacking principles | Transformer **temporal** features, because our rows do not form any time series |
| Grinsztajn et al., NeurIPS (2022), https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html | Large systematic tabular benchmarks: boosted trees competitive with neural nets on medium data | Properly tuned GBDT versus MLP, fair compute budgets | Claim that neural networks must be better |
| Hollmann et al., Nature (2025), https://www.nature.com/articles/s41586-024-08328-6 | TabPFN for tabular learning | Potential future comparison if weights, licensing, provenance and compute available | Treating pretrained weights as a fully local self-contained replacement (not bundled here) |

## Decision table (predeclared before running the v4 screening comparisons)

- **Implemented and tested:** XGBoost, LightGBM, CatBoost (both targets), CPU MLP baseline; A5 target-free feature expansion; 3 shared stratified grouped screening folds; preserved HGB and RF references; paired exact-pattern bootstrap; class-specific evaluation; reliability bins, multiclass Brier score; category-boundary and ordinal error analysis; country-held-out diagnostic; predetermined seed sensitivity; permutation importance of permitted RAW inputs.
- **Retained from v3:** Original 40-classifier experiment and 7 OOF stacking methods as audited reusable modules, but NOT rerun by default in this v4 *extension*; those historical exploratory comparisons can still be read in v3 results. The new screening ranks are **additional** explorations rather than replacements for independent confirmation.
- **Not scientifically justified:** NLP on `City` or `Country`. Cities are almost all unique, so identifiers are dangerous for memorizing rather than extracting transferable natural-language meaning; Country is represented as categorical. Sentiment/BERT embeddings on names are not pollutant measurements.
- **Not applicable:** Reinforcement Learning needs actions, rewards, a transition model and typically policy interventions; no such variables exist. No prospective causal emission-control policy may be inferred.
- **Not applicable as temporal forecasting:** RNN/LSTM, CNN-LSTM, temporal Transformer, time series cross-validation, spatiotemporal graph neural networks; there are no dates, observation histories, spatial links, station metadata or meteorological covariates.
- **Optional future benchmarking (not bundled):** TabPFN 2.5 (additional model-weights download and possible licensing), Bayesian optimization (larger compute budget), conformal prediction with a fresh valid calibration partition, calibrated rare-class classification, new external/temporal measurements. The previously used v3 confirmation partition has already been inspected and is **not reset or recycled** here.

## Safeguards and interpretation

1. Numeric PM2.5 values, categories, global AQI and the pollutant category columns are excluded by a strict whitelist. Derived A5 gas features are computed only from admitted input features inside pipelines.
2. The old, historically inspected 20%-random-row test and the v3 internal confirmation have historical results. The new feature-only pattern-disjoint outer holdout is not independently external. **Neither holdout nor v3 confirmation labels are passed to the v4 models, model selection, or statistical analyses.** Only the 14,112 screening records are used for v4 exploration.
3. All comparisons use the same 3 StratifiedGroupKFold folds and group keys. Every fitted pipeline (including MLP internal early stopping) only receives that fold's training rows.
4. Bootstrap intervals of out-of-fold screening prediction contrasts are *descriptive*. Their groups account for identical-input clustering but not tuning/search multiplicity, distribution shift, or validation-training uncertainty. Even CIs excluding zero cannot establish an independent-confirmation improvement after candidate screening.
5. Country grouping probes unseen-country transfer, not future-year transfer. Rare Hazardous n is limited; report support and confidence bounds before claiming its recall is reliable.
6. Error units: regression MAE in AQI points; macro-F1 unitless; never subtract the two. Original historical results and v4 CV use different training/validation protocols and are not directly comparable.
7. Do not change the manuscript's abstract/conclusions to assert improved independent model accuracy based on these exploratory comparisons. A new independent dataset or a genuinely future one-time prospective validation is required to substantiate such a claim.

## Reproducible citations

- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS. https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html
- Hollmann, N., et al. (2025). *Accurate predictions on small data with a tabular foundation model*. Nature 637. https://www.nature.com/articles/s41586-024-08328-6
- Singh, S., et al. (2026). *Ensemble learning for air quality index prediction: integrating gradient boosting, XGBoost, and stacking with SHAP-based interpretability*. Scientific Reports 16. https://www.nature.com/articles/s41598-026-39232-w
- *Air Quality Forecasting Using Machine Learning: Comparative Analysis and Ensemble Strategies for Enhanced Prediction*. (2025). Water, Air, & Soil Pollution. https://link.springer.com/article/10.1007/s11270-025-08122-8
- *Application of stacked Transformer-XGBoost model and SHAP interpretability in air quality index prediction in Linfen City*. (2026). iScience. https://www.sciencedirect.com/science/article/pii/S2589004226017712
