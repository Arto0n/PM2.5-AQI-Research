"""All learned data transformations live inside fold-specific sklearn pipelines."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
    HistGradientBoostingRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from .config import FEATURE_SETS, EXTENDED_FEATURE_SETS, SEED
from .features import InteractionFeatures

# Fixed specifications: NOT calibrated to any final-holdout scores.
REGRESSORS = ('dummy', 'ridge', 'elastic_net', 'huber', 'knn', 'svr',
              'random_forest', 'extra_trees', 'gradient_boosting',
              'hgb_raw', 'hgb_log_reference', 'mlp')
CLASSIFIERS = ('dummy_majority', 'rf_balanced', 'rf_plain',
               'rf_balanced_subsample', 'extra_trees', 'extra_trees_plain',
               'logistic', 'logistic_plain', 'knn',
               'hist_gradient_boosting', 'hgb_balanced')


def preprocessor(feature_set='A4', scaled=False):
    if feature_set not in EXTENDED_FEATURE_SETS:
        raise ValueError(f'Unknown feature set: {feature_set}')
    chosen = EXTENDED_FEATURE_SETS[feature_set]
    nums = [c for c in chosen if c != 'Country']
    numeric_steps = [('fill', SimpleImputer(strategy='median'))]
    if scaled:
        numeric_steps.append(('scale', StandardScaler()))
    transformers = [('numeric', Pipeline(numeric_steps), nums)]
    if 'Country' in chosen:
        transformers.append(('country', Pipeline([
            ('fill', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), ['Country']))
    return ColumnTransformer(transformers, remainder='drop')


def make_regressor(name, feature_set='A4'):
    if name not in REGRESSORS:
        raise ValueError(f'Unknown regressor: {name}')
    scaling = name in {'ridge','elastic_net','huber','knn','svr','mlp'}
    estimators = {
        'dummy': lambda: DummyRegressor(strategy='median'),
        'ridge': lambda: Ridge(alpha=10.),
        'elastic_net': lambda: ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=2000, random_state=SEED),
        'huber': lambda: HuberRegressor(alpha=1., max_iter=300),
        'knn': lambda: KNeighborsRegressor(n_neighbors=25, weights='distance', n_jobs=1),
        'svr': lambda: SVR(C=30, epsilon=5, kernel='rbf'),
        'random_forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=15,
                            min_samples_leaf=5, random_state=SEED, n_jobs=1),
        'extra_trees': lambda: ExtraTreesRegressor(n_estimators=75, max_depth=15,
                            min_samples_leaf=4, random_state=SEED, n_jobs=1),
        'gradient_boosting': lambda: GradientBoostingRegressor(n_estimators=120,
                            learning_rate=.05, max_depth=3, random_state=SEED),
        'hgb_raw': lambda: HistGradientBoostingRegressor(learning_rate=.05, max_iter=150,
                            max_leaf_nodes=31, l2_regularization=1., random_state=SEED),
        'hgb_log_reference': lambda: HistGradientBoostingRegressor(learning_rate=.05, max_iter=150,
                            max_leaf_nodes=31, l2_regularization=1., random_state=SEED),
        'mlp': lambda: MLPRegressor(hidden_layer_sizes=(64,64), max_iter=130,
                            early_stopping=True, random_state=SEED),
    }
    pipe = Pipeline([('features', InteractionFeatures()),
                     ('prepare', preprocessor(feature_set, scaling)),
                     ('model', estimators[name]())])
    if name == 'hgb_log_reference':
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                          inverse_func=np.expm1, check_inverse=True)
    return pipe


def make_classifier(name, feature_set='A4'):
    if name not in CLASSIFIERS:
        raise ValueError(f'Unknown classifier: {name}')
    needs_scale = name in {'logistic','logistic_plain','knn'}
    estimators = {
        'dummy_majority': lambda: DummyClassifier(strategy='most_frequent'),
        'rf_balanced': lambda: RandomForestClassifier(n_estimators=250,max_depth=18,
                   min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=1),
        'rf_balanced_subsample': lambda: RandomForestClassifier(n_estimators=250,max_depth=18,
                   min_samples_leaf=2,class_weight='balanced_subsample',random_state=SEED,n_jobs=1),
        'rf_plain': lambda: RandomForestClassifier(n_estimators=250,max_depth=18,
                   min_samples_leaf=2,class_weight=None,random_state=SEED,n_jobs=1),
        'extra_trees': lambda: ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,
                   class_weight='balanced',random_state=SEED,n_jobs=1),
        'extra_trees_plain': lambda: ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,
                   class_weight=None,random_state=SEED,n_jobs=1),
        'logistic_plain': lambda: LogisticRegression(C=1.,max_iter=500,class_weight=None),
        'logistic': lambda: LogisticRegression(C=1.,max_iter=500,class_weight='balanced'),
        'knn': lambda: KNeighborsClassifier(n_neighbors=25,weights='distance',n_jobs=1),
        'hist_gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=100,
                   learning_rate=.07,random_state=SEED),
        'hgb_balanced': lambda: HistGradientBoostingClassifier(max_iter=100,
                   learning_rate=.07,class_weight='balanced',random_state=SEED),
    }
    return Pipeline([('features', InteractionFeatures()),
                     ('prepare', preprocessor(feature_set, needs_scale)),
                     ('model', estimators[name]())])
