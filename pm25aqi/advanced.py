"""Optional evidence-based baselines for medium-sized tabular AQI data.

No model accepts City, overall AQI, target, pollutant category proxy or holdout.
Every estimator and scaler/encoder is constructed anew per *training* fold.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from .features import InteractionFeatures, RichInteractionFeatures
from .models import preprocessor, make_regressor, make_classifier
from .config import SEED, EXTENDED_FEATURE_SETS

REGRESSION_CANDIDATES = (
    ('log_hgb_A4', 'hgb_log_reference', 'A4'),
    ('log_hgb_A5', 'hgb_log_reference', 'A5'),
    ('catboost_A4', 'catboost', 'A4'),
    ('catboost_A5', 'catboost', 'A5'),
    ('xgboost_A4', 'xgboost', 'A4'),
    ('xgboost_A5', 'xgboost', 'A5'),
    ('lightgbm_A4', 'lightgbm', 'A4'),
    ('lightgbm_A5', 'lightgbm', 'A5'),
    ('mlp_A4', 'mlp', 'A4'),
    ('extra_trees_A4', 'extra_trees', 'A4'),
)
CLASSIFICATION_CANDIDATES = (
    ('rf_balanced_A4', 'rf_balanced', 'A4'),
    ('rf_balanced_A5', 'rf_balanced', 'A5'),
    ('hgb_balanced_A4', 'hgb_balanced', 'A4'),
    ('catboost_A4', 'catboost', 'A4'),
    ('catboost_A5', 'catboost', 'A5'),
    ('xgboost_A4', 'xgboost', 'A4'),
    ('xgboost_A5', 'xgboost', 'A5'),
    ('lightgbm_A4', 'lightgbm', 'A4'),
    ('lightgbm_A5', 'lightgbm', 'A5'),
    ('mlp_A4', 'mlp', 'A4'),
    ('rf_balanced_A1', 'rf_balanced', 'A1'),
    ('rf_balanced_A3', 'rf_balanced', 'A3'),
)


def _booster(name, kind):
    # Versioned optional libraries are explicitly in requirements.txt.
    if name == 'catboost':
        from catboost import CatBoostRegressor, CatBoostClassifier
        kw = dict(iterations=150, depth=5, learning_rate=.065, l2_leaf_reg=5,
                  random_seed=SEED, verbose=False, allow_writing_files=False,
                  thread_count=1)
        return CatBoostRegressor(loss_function='MAE', **kw) if kind=='reg' else CatBoostClassifier(loss_function='MultiClass', **kw)
    if name == 'xgboost':
        from xgboost import XGBRegressor, XGBClassifier
        kw = dict(n_estimators=175, max_depth=4, learning_rate=.06,
                  subsample=.85, colsample_bytree=.9, reg_lambda=3,
                  random_state=SEED, n_jobs=1, tree_method='hist')
        return XGBRegressor(objective='reg:squarederror', **kw) if kind=='reg' else XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', **kw)
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor, LGBMClassifier
        kw = dict(n_estimators=170, learning_rate=.055, num_leaves=15,
                  min_child_samples=35, colsample_bytree=.9,
                  reg_lambda=4, max_depth=-1, random_state=SEED,
                  n_jobs=1, verbosity=-1)
        return LGBMRegressor(objective='regression_l1', **kw) if kind=='reg' else LGBMClassifier(objective='multiclass', **kw)
    if name == 'mlp':
        kw = dict(hidden_layer_sizes=(48,24), alpha=2., learning_rate_init=.002,
                  max_iter=125, early_stopping=True, n_iter_no_change=12, random_state=SEED)
        return MLPRegressor(**kw) if kind=='reg' else MLPClassifier(**kw)
    raise ValueError(f'Unsupported booster {name}')


class BoostRegressionAdapter(RegressorMixin, BaseEstimator):
    """scikit-learn-compatible wrapper for optional third-party estimators.

    Some CatBoost/sklearn version combinations lack modern sklearn tags.
    Fitting always creates a new underlying model with training data only.
    """
    def __init__(self, name): self.name=name
    def fit(self, X, y):
        self.estimator_=_booster(self.name,'reg')
        self.estimator_.fit(np.asarray(X,dtype=float),y)
        return self
    def predict(self,X): return np.asarray(self.estimator_.predict(np.asarray(X,dtype=float)),dtype=float)

class BoostClassificationAdapter(ClassifierMixin,BaseEstimator):
    def __init__(self,name):self.name=name
    def fit(self,X,y):
        self.estimator_=_booster(self.name,'clf')
        self.estimator_.fit(np.asarray(X,dtype=float),y)
        self.classes_=np.asarray(self.estimator_.classes_,dtype=int)
        return self
    def predict(self,X):return np.asarray(self.estimator_.predict(np.asarray(X,dtype=float)),dtype=int).reshape(-1)
    def predict_proba(self,X):return np.asarray(self.estimator_.predict_proba(np.asarray(X,dtype=float)),dtype=float)


def make_advanced_regressor(name, features='A4'):
    if features not in EXTENDED_FEATURE_SETS: raise ValueError(features)
    if name in ('hgb_log_reference','extra_trees','hgb_raw','ridge','random_forest') and features != 'A5':
        return make_regressor(name, features)
    if name in ('hgb_log_reference','extra_trees','hgb_raw','ridge','random_forest') and features == 'A5':
        from sklearn.base import clone
        ref=make_regressor(name,'A4')
        if name=='hgb_log_reference':
            ref.regressor.set_params(features=RichInteractionFeatures(),prepare=preprocessor('A5'))
        else:
            ref.set_params(features=RichInteractionFeatures(),prepare=preprocessor('A5',scaled=name=='ridge'))
        return ref
    if name not in ('catboost','xgboost','lightgbm','mlp'):
        raise ValueError(f'Unknown regression method {name}')
    return Pipeline([('features',RichInteractionFeatures() if features=='A5' else InteractionFeatures()),
                     ('prepare',preprocessor(features,scaled=name=='mlp')),
                     ('model',BoostRegressionAdapter(name) if name!='mlp' else _booster(name,'reg'))])


def make_advanced_classifier(name, features='A4'):
    if features not in EXTENDED_FEATURE_SETS: raise ValueError(features)
    if name in ('rf_balanced','hgb_balanced','hist_gradient_boosting','extra_trees','logistic','knn'):
        ref=make_classifier(name, 'A4' if features=='A5' else features)
        if features=='A5': ref.set_params(features=RichInteractionFeatures(), prepare=preprocessor('A5',scaled=name in ('logistic','knn')))
        return ref
    if name not in ('catboost','xgboost','lightgbm','mlp'):
        raise ValueError(f'Unknown classification method {name}')
    return Pipeline([('features',RichInteractionFeatures() if features=='A5' else InteractionFeatures()),
                     ('prepare',preprocessor(features,scaled=name=='mlp')),
                     ('model',BoostClassificationAdapter(name) if name!='mlp' else _booster(name,'clf'))])
