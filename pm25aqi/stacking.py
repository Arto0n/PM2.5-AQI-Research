"""Leakage-safe, nested grouped stacking. No holdout set is accepted here.

Outer fold validation is NEVER used to fit base learners or a meta-learner.
Inner OOF meta features are produced ONLY from the outer-training partition.
Each stacker is evaluated on outer-fold rows; all stackers reuse the same fits.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from .config import SEED
from .data import group_folds, pattern_groups
from .models import make_regressor
from .validation import _validate_folds, _validate_xy, metrics

STACKERS=('mean','inverse_mae','ridge','positive_ridge','huber','elastic_net','simplex')


@dataclass
class OOF:
    predictions: pd.DataFrame
    traces: list


def inner_oof(X_train, y_train, base_specs, folds=3, seed=SEED):
    _validate_xy(X_train,y_train)
    splits=group_folds(X_train,n_splits=folds,seed=seed)
    _validate_folds(X_train,splits)
    predictions=pd.DataFrame(index=X_train.index,columns=list(base_specs),dtype=float)
    traces=[]
    for fold,(tr,va) in enumerate(splits,1):
        for base, spec in base_specs.items():
            est=make_regressor(spec)
            est.fit(X_train.iloc[tr],y_train.iloc[tr])
            predictions.iloc[va,predictions.columns.get_loc(base)] = est.predict(X_train.iloc[va])
            traces.append({'fold':fold,'model':base,
                           'train_ids':tuple(X_train.index[tr]),
                           'validation_ids':tuple(X_train.index[va])})
    if not np.isfinite(predictions.to_numpy()).all() or predictions.isna().to_numpy().any():
        raise AssertionError('OOF predictions missing/nonfinite')
    for base in base_specs:
        ids=[row for item in traces if item['model']==base for row in item['validation_ids']]
        if len(ids)!=len(X_train) or set(ids)!=set(X_train.index):
            raise AssertionError('OOF every row exactly once')
    return OOF(predictions,traces)


class StackerMeta:
    """Mean, inverse-error weights, unconstrained/positive regressors, simplex."""
    def __init__(self, method):
        if method not in STACKERS:
            raise ValueError(f'Unknown stacker: {method}')
        self.method=method

    def fit(self, features, y):
        x=np.asarray(features,dtype=float); target=np.asarray(y,dtype=float)
        if x.ndim!=2 or len(x)!=len(target) or not np.isfinite(x).all():
            raise ValueError('Misaligned or missing OOF meta features')
        n=x.shape[1]
        if self.method=='mean':
            self.weights_=np.ones(n)/n
        elif self.method=='inverse_mae':
            errors=np.maximum(np.abs(x-target[:,None]).mean(axis=0),1e-8)
            weights=1/errors
            self.weights_=weights/weights.sum()
        elif self.method=='simplex':
            fit=minimize(lambda w: float(np.square(x@w-target).mean()),
                 np.ones(n)/n,method='SLSQP',bounds=[(0.,1.)]*n,
                 constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}],
                 options={'maxiter':500,'ftol':1e-10})
            if not fit.success:
                raise RuntimeError(f'Simplex optimizer failed: {fit.message}')
            self.weights_=np.maximum(fit.x,0.)
            self.weights_/=self.weights_.sum()
        else:
            models={
              'ridge':Ridge(alpha=10.),
              'positive_ridge':Ridge(alpha=10.,positive=True),
              'huber':HuberRegressor(max_iter=400,alpha=1.),
              'elastic_net':ElasticNet(alpha=.1,l1_ratio=.3,max_iter=3000,random_state=SEED)
            }
            self.model_=models[self.method].fit(x,target)
        self.n_features_=n
        return self

    def predict(self, features):
        x=np.asarray(features,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.n_features_:
            raise ValueError('Meta predictor shape does not match training')
        values = self.model_.predict(x) if hasattr(self,'model_') else x@self.weights_
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite stacked predictions')
        return values


class FittedStack:
    """A fully fitted development-only stack for inference, never a test-ranking API."""
    def __init__(self, names, fitted_models, meta):
        self.names=tuple(names)
        self.fitted_models=fitted_models
        self.meta=meta

    def predict(self, X):
        features=np.column_stack([self.fitted_models[k].predict(X) for k in self.names])
        return self.meta.predict(features)


def fit_stack_on_development(X_dev,y_dev,base_specs,*,method='ridge',inner_folds=3):
    _validate_xy(X_dev,y_dev)
    oof=inner_oof(X_dev,y_dev,base_specs,folds=inner_folds)
    meta=StackerMeta(method).fit(oof.predictions,y_dev)
    bases={name:make_regressor(spec).fit(X_dev,y_dev) for name,spec in base_specs.items()}
    return FittedStack(base_specs.keys(),bases,meta),oof


def compare_stackers_nested(X_dev,y_dev,base_specs,methods=STACKERS,*,outer_folds=3,inner_folds=3):
    """Nested grouped CV: all model comparison scores use OUTER development folds.

    Choosing a best candidate using these scores creates selection optimism;
    do not call that chosen score an external generalization estimate.
    """
    _validate_xy(X_dev,y_dev)
    if not 2<=outer_folds or not 2<=inner_folds:
        raise ValueError('Need 2+ folds')
    fold_splits=group_folds(X_dev,n_splits=outer_folds)
    _validate_folds(X_dev,fold_splits)
    scores=[]; diagnostics=[]
    for fold,(tr,va) in enumerate(fold_splits,1):
        Xtr,ytr=X_dev.iloc[tr],y_dev.iloc[tr]
        Xva,yva=X_dev.iloc[va],y_dev.iloc[va]
        oof=inner_oof(Xtr,ytr,base_specs,folds=inner_folds,seed=SEED+fold)
        refitted={name:make_regressor(spec).fit(Xtr,ytr) for name,spec in base_specs.items()}
        meta_validation=pd.DataFrame({name:est.predict(Xva) for name,est in refitted.items()}, index=Xva.index)
        for method in methods:
            meta=StackerMeta(method).fit(oof.predictions,ytr)
            pred=meta.predict(meta_validation)
            scores.append({'kind':'stacking','model':method,'fold':fold,
                           'n_validation':len(va),**metrics(yva,pred)})
        diagnostics.append({'outer_fold':fold, 'outer_train_rows':len(tr),
                            'outer_valid_rows':len(va),
                            'oof_train_rows':len(oof.predictions),
                            'oof_overlap':any(set(row['train_ids'])&set(row['validation_ids']) for row in oof.traces),
                            'outer_train_ids':tuple(Xtr.index),
                            'outer_validation_ids':tuple(Xva.index)})
    return pd.DataFrame(scores), diagnostics
