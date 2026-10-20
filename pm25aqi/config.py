"""Immutable study contract; additional models are exploratory, not paper benchmarks."""
from dataclasses import dataclass

SEED = 42
DATA_SHA256 = 'e1fc6e682edb8a2102d87e182b844fc8cde49e10763c0ef4293e510bbe240f69'
NUMERIC = ('CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value')
INPUTS = (*NUMERIC, 'Country')
TARGET = 'PM2.5 AQI Value'
CLASS_TARGET = 'PM2.5 AQI Category'
CATEGORIES = ('Good', 'Moderate', 'Unhealthy for Sensitive Groups',
              'Unhealthy', 'Very Unhealthy', 'Hazardous')
INTERACTIONS = ('CO_x_NO2', 'CO_x_Ozone')
FEATURE_SETS = {
    'A1': NUMERIC,
    'A2': (*NUMERIC, 'Country'),
    'A3': (*NUMERIC, *INTERACTIONS),
    'A4': (*NUMERIC, 'Country', *INTERACTIONS),
}
EXTENDED_FEATURE_SETS = {**FEATURE_SETS,
 'A5': (*NUMERIC, 'Country', *INTERACTIONS,
        *('log_' + c for c in NUMERIC), 'gas_max', 'gas_range', 'gas_sum',
        *('fraction_' + c for c in NUMERIC))}
HISTORICAL_MODEL = 'hgb_log_reference'
HISTORICAL_TEST = dict(mae=16.549035, rmse=29.643914, r2=0.702208)

@dataclass(frozen=True)
class ExperimentConfig:
    # Entirely development-only. Different protocol from the historical paper.
    outer_folds: int = 3
    inner_folds: int = 3
    seed: int = SEED
    feature_set: str = 'A4'
    regressors: tuple = ('dummy', 'ridge', 'elastic_net', 'random_forest', 'extra_trees', 'hgb_raw', 'hgb_log_reference')
    stacking_bases: tuple = ('ridge', 'extra_trees', 'hgb_log_reference')
    classification_candidates: tuple = ('dummy_majority','rf_balanced','rf_plain',
        'rf_balanced_subsample','extra_trees','logistic','logistic_plain',
        'knn','hist_gradient_boosting','hgb_balanced')
    classification_feature_sets: tuple = ('A1','A2','A3','A4')
    classifier_folds: int = 3
    classifier_confirmation_fraction: float = 0.25
    stackers: tuple = ('mean', 'inverse_mae', 'ridge', 'positive_ridge', 'huber', 'elastic_net', 'simplex')
