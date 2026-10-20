"""Scientific sensitivity suite. ALWAYS screening data; confirmation/holdout labels not used.

Methods are prespecified, grouped and paired. Do not treat OOF-screening rankings
as independent final generalization estimates; all model exploration is ongoing.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from .config import CATEGORIES, SEED
from .data import (load_data, raw_predictors, prospective_split,
                   development_only, assert_group_disjoint, group_folds, pattern_groups,
                   categories_from_aqi)
from .classification import screening_confirmation_split, classifier_scores
from .advanced import (REGRESSION_CANDIDATES, CLASSIFICATION_CANDIDATES,
                       make_advanced_regressor, make_advanced_classifier)
from .validation import metrics

LABELS = tuple(range(6))

@dataclass
class ScreeningCohort:
    X: pd.DataFrame
    y: pd.Series
    categories: pd.Series
    n_development: int
    n_confirmation_sealed: int
    n_outer_sealed: int


def load_screening_cohort(csv_path):
    """Only screening targets are extracted. No final or confirmation labels exposed.

    The source has been audited; its fingerprint is checked by load_data.
    """
    data = load_data(csv_path)
    X = raw_predictors(data)
    dev_ids, outer_ids = prospective_split(X)
    screen_ids, confirm_ids = screening_confirmation_split(X.loc[dev_ids])
    assert_group_disjoint(X.loc[screen_ids], X.loc[confirm_ids])
    assert_group_disjoint(X.loc[screen_ids], X.loc[outer_ids])
    x, y, cls = development_only(data, screen_ids)
    return ScreeningCohort(x, y, cls, len(dev_ids), len(confirm_ids), len(outer_ids))


def study_folds(cohort, n_splits=3, seed=SEED):
    """Identical grouped folds for all regressors and classifiers."""
    g = pattern_groups(cohort.X)
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    folds=list(cv.split(cohort.X,cohort.categories,groups=g))
    for train,val in folds:
        if set(g.iloc[train]) & set(g.iloc[val]):
            raise AssertionError('Fold group overlap')
        if len(set(train)&set(val)):
            raise AssertionError('Fold row overlap')
    return folds


def _threshold_classes(numeric):
    # AQI class labels correspond to integer-valued AQI (0..500).
    scores=np.clip(np.rint(np.asarray(numeric,dtype=float)), 0, 500)
    return pd.Categorical(categories_from_aqi(scores),categories=CATEGORIES).codes.astype(int)


def evaluation_tables(cohort, regression_specs=REGRESSION_CANDIDATES,
                      classification_specs=CLASSIFICATION_CANDIDATES,
                      progress=None):
    """Out-of-fold comparisons using only screening observations.

    Returns records + predictions. ALL candidates share exactly the same folds;
    preprocessing and early-stopping internal splits are always inside train folds.
    No confirmation or outer-holdout input argument exists.
    """
    X, yr, yc=cohort.X, cohort.y, cohort.categories
    folds=study_folds(cohort)
    reg_rows=[]; cls_rows=[]; pred_reg={}; pred_cls={}; prob_cls={}
    n=len(X)
    def emit(msg):
        if progress: progress(msg)
    for label,model,features in regression_specs:
        started=time.monotonic(); oof=np.full(n,np.nan)
        for f,(tr,va) in enumerate(folds,1):
            est=make_advanced_regressor(model,features)
            est.fit(X.iloc[tr],yr.iloc[tr]); pred=est.predict(X.iloc[va]); oof[va]=pred
            reg_rows.append({'name':label,'method':model,'features':features,'fold':f,
                             'n_validation':len(va), **metrics(yr.iloc[va],pred)})
        if not np.isfinite(oof).all():raise AssertionError('Incomplete regression OOF')
        pred_reg[label]=oof
        emit(f'Regression {label} completed in {time.monotonic()-started:.1f}s')
    for label,model,features in classification_specs:
        started=time.monotonic(); oof=np.full(n,-1,dtype=int)
        probs=np.full((n,6),np.nan)
        for f,(tr,va) in enumerate(folds,1):
            est=make_advanced_classifier(model,features)
            est.fit(X.iloc[tr],yc.iloc[tr]); p=np.asarray(est.predict(X.iloc[va]),dtype=int)
            oof[va]=p
            prob=est.predict_proba(X.iloc[va])
            probs[np.ix_(va,est.classes_.astype(int))]=prob
            # Fill absent classes with 0, if training split lacks a class.
            absent=np.setdiff1d(np.arange(6),est.classes_.astype(int))
            if len(absent): probs[np.ix_(va,absent)]=0.
            cls_rows.append({'name':label,'method':model,'features':features,'fold':f,
                             'n_validation':len(va), **classifier_scores(yc.iloc[va],p)})
        if (oof<0).any() or not np.isfinite(probs).all():raise AssertionError('Incomplete class OOF')
        pred_cls[label]=oof;prob_cls[label]=probs
        emit(f'Classification {label} completed in {time.monotonic()-started:.1f}s')
    return (pd.DataFrame(reg_rows),pd.DataFrame(cls_rows),
            pd.DataFrame(pred_reg,index=X.index),pd.DataFrame(pred_cls,index=X.index),prob_cls)


def oof_regression_to_category(cohort, folds=None):
    """FIXED before comparing with direct classifiers; same validation rows."""
    X, y=cohort.X,cohort.y
    folds=study_folds(cohort) if folds is None else folds
    numeric=np.full(len(X),np.nan)
    for tr,va in folds:
        est=make_advanced_regressor('hgb_log_reference','A4').fit(X.iloc[tr],y.iloc[tr])
        numeric[va]=est.predict(X.iloc[va])
    if not np.isfinite(numeric).all():raise AssertionError('Incomplete OOF')
    return _threshold_classes(numeric),numeric


def summarize_models(reg_folds, cls_folds):
    r=(reg_folds.groupby(['name','method','features'],as_index=False)
       .agg(mae=('mae','mean'),mae_sd=('mae','std'),rmse=('rmse','mean'),r2=('r2','mean'))
       .sort_values(['mae','name']).reset_index(drop=True))
    c=(cls_folds.groupby(['name','method','features'],as_index=False)
       .agg(macro_f1=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
            accuracy=('accuracy','mean'),balanced_accuracy=('balanced_accuracy','mean'),
            ordinal_mae=('ordinal_mae','mean'))
       .sort_values(['macro_f1','name'],ascending=[False,True]).reset_index(drop=True))
    return r,c


def uncertainty_and_error(cohort, reg_oof, cls_oof, probs, cls_baseline='rf_balanced_A4'):
    """Descriptive, no hyperparameter selection. OOF is NOT independent from selection."""
    y=cohort.categories.to_numpy(dtype=int)
    raw_y=cohort.y.to_numpy(dtype=float)
    # Fixed direct reference, reported irrespective of the top-ranked method.
    if cls_baseline not in cls_oof:raise KeyError(cls_baseline)
    p=cls_oof[cls_baseline].to_numpy(dtype=int)
    cm=confusion_matrix(y,p,labels=LABELS)
    pr,rc,f1,sup=precision_recall_fscore_support(y,p,labels=LABELS,zero_division=0)
    classes=pd.DataFrame({'category':CATEGORIES,'n':sup,'precision':pr,'recall':rc,'f1':f1})
    prob=probs[cls_baseline]
    bins=np.linspace(0,1,11); conf=prob.max(axis=1)
    pred=prob.argmax(axis=1)
    cal=[]
    for i,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
        mask=(conf>=lo)&(conf<=hi if i==9 else conf<hi)
        cal.append({'bin':i+1,'n':int(mask.sum()),'mean_confidence':float(conf[mask].mean()) if mask.any() else np.nan,
                    'accuracy':float((pred[mask]==y[mask]).mean()) if mask.any() else np.nan})
    uncertainty={'ece_10bin':float(sum(a['n']/len(y)*abs(a['mean_confidence']-a['accuracy']) for a in cal if a['n'])),
                 'multiclass_brier':float(np.sum((prob-np.eye(6)[y])**2,axis=1).mean()),
                 'ordinal_error_ge2':float((np.abs(p-y)>=2).mean()),
                 'hazardous_support':int((y==5).sum())}
    # Inputs label numeric PM2.5 AQI used ONLY for this post-hoc *screening* diagnostic.
    near=np.min(np.abs(raw_y[:,None]-np.array([50.5,99.5,149.5,200.5,299.5])[None,:]),axis=1)<=5
    for label,mask in [('within_5_AQI_of_boundary',near),('away_from_boundaries',~near)]:
        uncertainty[label+'_n']=int(mask.sum())
        uncertainty[label+'_accuracy']=float((p[mask]==y[mask]).mean()) if mask.any() else float('nan')
    # Regression residuals are not exchangeable across location, do not claim predictive coverage.
    ref=reg_oof['log_hgb_A4'].to_numpy()
    uncertainty['log_hgb_screening_mae']=float(np.mean(np.abs(raw_y-ref)))
    uncertainty['log_hgb_screening_residual_abs_q90']=float(np.quantile(np.abs(raw_y-ref),.90))
    return classes,cm,pd.DataFrame(cal),uncertainty


def _f1_from_confusion(cm):
    tp=np.diag(cm); denom=cm.sum(axis=0)+cm.sum(axis=1)
    return float(np.mean(np.divide(2*tp,denom,out=np.zeros(6),where=denom>0)))


def cluster_paired_bootstrap(cohort, reg_oof, cls_oof, *, n_boot=350, seed=SEED):
    """Cluster bootstrap paired predictions, fixed contrasts; exploratory CIs.

    The class contrast is *not* a calibrated test: screening CV was used to
    explore these model families and uncertainty ignores fold-training variance.
    """
    y=cohort.y.to_numpy(dtype=float); codes=cohort.categories.to_numpy(dtype=int)
    groups=pattern_groups(cohort.X).to_numpy()
    unique,inverse=np.unique(groups,return_inverse=True)
    k=len(unique)
    rng=np.random.default_rng(seed)
    contrasts=[]
    for a,b in [('catboost_A4','log_hgb_A4'),('xgboost_A4','log_hgb_A4'),
                ('lightgbm_A4','log_hgb_A4'),('log_hgb_A5','log_hgb_A4')]:
        if a not in reg_oof or b not in reg_oof:continue
        errors=np.abs(y-reg_oof[a].to_numpy())-np.abs(y-reg_oof[b].to_numpy())
        total=np.bincount(inverse,weights=errors,minlength=k)
        counts=np.bincount(inverse,minlength=k)
        vals=np.empty(n_boot)
        for i in range(n_boot):
            sampled=rng.integers(0,k,k)
            vals[i]=total[sampled].sum()/counts[sampled].sum()
        contrasts.append({'task':'regression','a':a,'b':b,
            'metric':'MAE(a)-MAE(b)','estimate':float(errors.mean()),
            'CI_low':float(np.quantile(vals,.025)),'CI_high':float(np.quantile(vals,.975))})
    for a,b in [('catboost_A4','rf_balanced_A4'),('xgboost_A4','rf_balanced_A4'),
                ('lightgbm_A4','rf_balanced_A4'),('rf_balanced_A5','rf_balanced_A4')]:
        if a not in cls_oof or b not in cls_oof:continue
        grouped=[]
        for name in (a,b):
            pred=cls_oof[name].to_numpy(dtype=int)
            encoded=inverse*36+codes*6+pred
            grouped.append(np.bincount(encoded,minlength=k*36).reshape(k,36))
        vals=[]
        for i in range(n_boot):
            s=rng.integers(0,k,k)
            ca=grouped[0][s].sum(axis=0).reshape(6,6)
            cb=grouped[1][s].sum(axis=0).reshape(6,6)
            vals.append(_f1_from_confusion(ca)-_f1_from_confusion(cb))
        observed=_f1_from_confusion(grouped[0].sum(axis=0).reshape(6,6))-_f1_from_confusion(grouped[1].sum(axis=0).reshape(6,6))
        contrasts.append({'task':'classification','a':a,'b':b,
            'metric':'macro-F1(a)-macro-F1(b)','estimate':float(observed),
            'CI_low':float(np.quantile(vals,.025)),'CI_high':float(np.quantile(vals,.975))})
    return pd.DataFrame(contrasts)


def seed_robustness(cohort, seeds=(42,43,44)):
    """Prespecified log-HGB and balanced RF, only on screening, 2 folds/seed."""
    rows=[]
    for seed in seeds:
        folds=study_folds(cohort,n_splits=2,seed=seed)
        for fold,(tr,va) in enumerate(folds,1):
            for kind in ('regression','classification'):
                if kind=='regression':
                    e=make_advanced_regressor('hgb_log_reference','A4').fit(cohort.X.iloc[tr],cohort.y.iloc[tr])
                    outcome=metrics(cohort.y.iloc[va],e.predict(cohort.X.iloc[va]))
                else:
                    e=make_advanced_classifier('rf_balanced','A4').fit(cohort.X.iloc[tr],cohort.categories.iloc[tr])
                    outcome=classifier_scores(cohort.categories.iloc[va],e.predict(cohort.X.iloc[va]))
                rows.append({'seed':seed,'fold':fold,'task':kind,'n_validation':len(va),**outcome})
    return pd.DataFrame(rows)


def geographical_stress(cohort):
    """Fixed predeclared models; no country-fold outcome is used to choose models."""
    rows=[]
    for fold,(tr,va) in enumerate(group_folds(cohort.X,n_splits=3,grouping='country'),1):
        for kind in ('regression','classification'):
            if kind=='regression':
                e=make_advanced_regressor('hgb_log_reference','A4').fit(cohort.X.iloc[tr],cohort.y.iloc[tr])
                s=metrics(cohort.y.iloc[va],e.predict(cohort.X.iloc[va]))
            else:
                e=make_advanced_classifier('rf_balanced','A4').fit(cohort.X.iloc[tr],cohort.categories.iloc[tr])
                s=classifier_scores(cohort.categories.iloc[va],e.predict(cohort.X.iloc[va]))
            rows.append({'fold':fold,'task':kind,'n_validation':len(va),**s})
    return pd.DataFrame(rows)


def class_specific_cluster_intervals(cohort, cls_oof, numeric_reg_oof, *, n_boot=350, seed=SEED):
    """Screening-only class-conditional F1/recall ranges + fixed direct-vs-derived.

    Resamples exact raw input patterns with replacement; does not estimate training
    variance or account for searching model specifications (descriptive ONLY).
    """
    y=np.asarray(cohort.categories,dtype=int)
    a=np.asarray(cls_oof['rf_balanced_A4'],dtype=int)
    b=_threshold_classes(numeric_reg_oof['log_hgb_A4'])
    groups=pattern_groups(cohort.X).to_numpy()
    unique,inverse=np.unique(groups,return_inverse=True)
    k=len(unique)
    agg=[]
    for p in (a,b):
        encoded=inverse*36+y*6+p
        agg.append(np.bincount(encoded,minlength=k*36).reshape(k,36))
    rng=np.random.default_rng(seed)
    direct_f1=[]; derived_f1=[]; direct_recall=[]; deltas=[]
    def per(cm):
        tp=np.diag(cm)
        f1=np.divide(2*tp,cm.sum(axis=0)+cm.sum(axis=1),out=np.zeros(6),where=(cm.sum(axis=0)+cm.sum(axis=1))>0)
        rc=np.divide(tp,cm.sum(axis=1),out=np.zeros(6),where=cm.sum(axis=1)>0)
        return f1,rc
    for i in range(n_boot):
        s=rng.integers(0,k,k)
        ca=agg[0][s].sum(axis=0).reshape(6,6)
        cb=agg[1][s].sum(axis=0).reshape(6,6)
        fa,ra=per(ca);fb,_=per(cb)
        direct_f1.append(fa);derived_f1.append(fb);direct_recall.append(ra)
        deltas.append(np.mean(fa)-np.mean(fb))
    ca=agg[0].sum(axis=0).reshape(6,6);cb=agg[1].sum(axis=0).reshape(6,6)
    fa,ra=per(ca);fb,_=per(cb)
    f1arr=np.asarray(direct_f1);rarr=np.asarray(direct_recall)
    per_class=pd.DataFrame({'category':CATEGORIES,'support':ca.sum(axis=1).astype(int),
            'direct_RF_A4_F1':fa,'F1_CI_low':np.quantile(f1arr,.025,axis=0),
            'F1_CI_high':np.quantile(f1arr,.975,axis=0),
            'direct_RF_A4_recall':ra,'recall_CI_low':np.quantile(rarr,.025,axis=0),
            'recall_CI_high':np.quantile(rarr,.975,axis=0)})
    contrasts=pd.DataFrame([{'metric':'macro_F1_direct_RF_A4_minus_fixed_log_HGB_to_class',
           'estimate':np.mean(fa)-np.mean(fb),
           'CI_low':np.quantile(deltas,.025),'CI_high':np.quantile(deltas,.975),
           'samples':n_boot}])
    return per_class,contrasts
