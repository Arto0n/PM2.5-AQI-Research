"""Development-only model comparison with grouped, fold-local fitting."""
from dataclasses import asdict
import platform
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedGroupKFold
from .config import CATEGORIES, SEED
from .data import group_folds, pattern_groups, raw_predictors
from .models import make_regressor, make_classifier


def metrics(y, prediction):
    return dict(mae=float(mean_absolute_error(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                r2=float(r2_score(y, prediction)))


def _validate_xy(X, y):
    raw_predictors(X)
    if not isinstance(y, pd.Series) or not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError('X and y must have identical unique development row IDs')
    if y.isna().any() or not np.isfinite(np.asarray(y, dtype=float)).all():
        raise ValueError('Labels must be finite')


def _validate_folds(X, folds, grouping='pattern', require_full_coverage=True):
    groups = pattern_groups(X) if grouping == 'pattern' else X['Country'].fillna('__MISSING__').astype(str)
    seen = []
    for tr, va in folds:
        tr, va = np.asarray(tr, dtype=int), np.asarray(va, dtype=int)
        if (not len(tr) or not len(va) or len(set(tr)) != len(tr) or
                len(set(va)) != len(va) or np.any(tr<0) or np.any(va<0) or
                np.any(tr>=len(X)) or np.any(va>=len(X)) or
                not set(tr).isdisjoint(va)):
            raise AssertionError('Invalid train/validation fold')
        if not set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va])):
            raise AssertionError('Fold leaks exact groups')
        seen.extend(va.tolist())
    if require_full_coverage and (len(seen) != len(X) or set(seen) != set(range(len(X)))):
        raise AssertionError('Each development row must be predicted once')


def compare_regressors(X_dev, y_dev, candidates, *, feature_set='A4', folds=None):
    """Fixed candidates, scored only on held-out DEVELOPMENT folds."""
    _validate_xy(X_dev, y_dev)
    if folds is None:
        folds = group_folds(X_dev)
    folds = list(folds)
    _validate_folds(X_dev, folds)
    scores = []
    for name in candidates:
        for fold_id, (tr, va) in enumerate(folds, 1):
            est = make_regressor(name, feature_set)
            est.fit(X_dev.iloc[tr], y_dev.iloc[tr])
            scores.append({'model': name, 'kind': 'single', 'fold': fold_id,
                           'n_validation': len(va),
                           **metrics(y_dev.iloc[va], est.predict(X_dev.iloc[va]))})
    return pd.DataFrame(scores)


def compare_classifiers(X_dev, class_dev, candidates, feature_set='A4', n_splits=3):
    _validate_xy(X_dev, class_dev)
    if n_splits < 2:
        raise ValueError('Need at least two folds')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_dev, class_dev, groups=pattern_groups(X_dev)))
    _validate_folds(X_dev, folds)
    scores = []
    for name in candidates:
        for fold_id, (tr,va) in enumerate(folds, 1):
            est = make_classifier(name, feature_set)
            est.fit(X_dev.iloc[tr], class_dev.iloc[tr])
            pred = est.predict(X_dev.iloc[va])
            scores.append({'model': name, 'fold': fold_id, 'n_validation':len(va),
                           'accuracy':float(accuracy_score(class_dev.iloc[va], pred)),
                           'macro_f1':float(f1_score(class_dev.iloc[va], pred,
                                             labels=list(range(len(CATEGORIES))),
                                             average='macro',zero_division=0))})
    return pd.DataFrame(scores)


def summarize_regression(scores):
    if 'test_mae' in scores.columns or 'holdout_mae' in scores.columns:
        raise ValueError('Refusing any final-test ranking')
    return (scores.groupby(['kind','model'],as_index=False)
           .agg(mae=('mae','mean'),mae_sd=('mae','std'),rmse=('rmse','mean'),
                r2=('r2','mean'), folds=('fold','size'))
           .sort_values(['mae','rmse','model'],kind='stable').reset_index(drop=True))


def choose_development_winner(summary):
    if summary.empty or not {'model','kind','mae','rmse'}.issubset(summary.columns):
        raise ValueError('Needs a development-CV summary')
    if any('test' in s.lower() or 'holdout' in s.lower() for s in summary.columns):
        raise ValueError('Final test evidence prohibited in selection')
    if not np.isfinite(summary[['mae','rmse']].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite scores')
    best = summary.sort_values(['mae','rmse','model'],kind='stable').iloc[0]
    return str(best['kind']), str(best['model'])


def development_permutation_importance(X_dev, y_dev, *, estimator_name='hgb_log_reference',
                                       grouping='pattern', repeats=3):
    from sklearn.inspection import permutation_importance
    _validate_xy(X_dev,y_dev)
    (tr, va) = group_folds(X_dev, n_splits=3, grouping=grouping)[0]
    est = make_regressor(estimator_name).fit(X_dev.iloc[tr], y_dev.iloc[tr])
    result = permutation_importance(est, X_dev.iloc[va], y_dev.iloc[va],
                  scoring='neg_mean_absolute_error', n_repeats=repeats,
                  random_state=SEED, n_jobs=1)
    # RAW features are permuted, then interaction features recomputed inside Pipeline.
    return pd.DataFrame({'feature':X_dev.columns,
                         'mae_increase':result.importances_mean,
                         'sd':result.importances_std})


def group_transfer_sensitivity(X_dev,y_dev, *, grouping='country', model='hgb_log_reference',folds=3):
    _validate_xy(X_dev,y_dev)
    return compare_regressors(X_dev,y_dev,[model],folds=group_folds(
        X_dev,n_splits=folds,grouping=grouping)) if grouping == 'pattern' else _country_cv(X_dev,y_dev,model,folds)


def _country_cv(X,y,model,n_splits):
    folds=group_folds(X,n_splits=n_splits,grouping='country')
    _validate_folds(X,folds,grouping='country')
    records=[]
    for idx,(train,valid) in enumerate(folds,1):
        fitted=make_regressor(model).fit(X.iloc[train],y.iloc[train])
        records.append({'fold':idx,'grouping':'country',
                        'n_validation':len(valid),**metrics(y.iloc[valid],fitted.predict(X.iloc[valid]))})
    return pd.DataFrame(records)


def outlier_sensitivity(X_dev,y_dev,*,quantile=.99):
    """Any outlier cutoff is fitted on each TRAINING fold; never on validation."""
    if not 0 < quantile < 1:
        raise ValueError('Invalid quantile')
    _validate_xy(X_dev,y_dev)
    records=[]
    for fold,(tr,va) in enumerate(group_folds(X_dev),1):
        X_train=X_dev.iloc[tr]; y_train=y_dev.iloc[tr]
        cutoff=float(y_train.quantile(quantile))
        for label,mask in [('all_training',np.ones(len(tr),dtype=bool)),
                           ('trim_training_only',(y_train<=cutoff).to_numpy())]:
            model=make_regressor('hgb_log_reference').fit(X_train.iloc[mask],y_train.iloc[mask])
            score=metrics(y_dev.iloc[va],model.predict(X_dev.iloc[va]))
            records.append({'fold':fold,'policy':label,'training_threshold':cutoff,
                            'train_count':int(sum(mask)),**score})
    return pd.DataFrame(records)
