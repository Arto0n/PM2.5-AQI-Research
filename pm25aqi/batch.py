"""Frozen, single-pass end-stage scientific evaluation.

Two deliberate boundaries:
  (1) screening and its nested grouped CV can be rerun freely;
  (2) once the locked, selected specifications have been fit on development,
      holdout evaluation writes a receipt and is never repeated in normal runs.

No output can be fed back to select final models in this program.
"""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from .config import CATEGORIES, DATA_SHA256
from .data import load_data, raw_predictors, prospective_split, pattern_groups, assert_group_disjoint
from .classification import screening_confirmation_split, classifier_scores
from .advanced import make_advanced_classifier, make_advanced_regressor
from .stacking import (compare_stackers_nested, fit_stack_on_development,
                       STACKERS)
from .research import (load_screening_cohort, study_folds, _threshold_classes)
from .validation import metrics

BASES = {'ridge':'ridge', 'extra_trees':'extra_trees', 'hgb_log_reference':'hgb_log_reference'}
_REG_NAMES = ('log_hgb_A5','log_hgb_A4','huber_stack')
_CLASS_NAMES = ('lightgbm_A4','rf_balanced_A4','hgb_A5_to_class')
_LABELS = np.arange(6)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode('utf-8')


def protocol_and_split(root):
    """Validate pre-existing lock; split uses features only. The dataset loader audits the frozen numeric/class labels for consistency but they never determine membership."""
    root=Path(root)
    path=root/'LOCKED_PROTOCOL.json'
    if not path.is_file():
        raise RuntimeError('No prespecified LOCKED_PROTOCOL.json; stop before final scoring')
    lock=json.loads(path.read_text(encoding='utf-8'))
    if lock.get('data_sha256')!=DATA_SHA256:
        raise RuntimeError('Wrong fixed dataset in locked protocol')
    desired={'regression_primary':('log_hgb_A5','hgb_log_reference','A5'),
             'classification_primary':('lightgbm_A4','lightgbm','A4')}
    for k,(label,model,features) in desired.items():
        c=lock['selection_precommitted'][k]
        if (c['name'],c['method'],c['features'])!=(label,model,features):
            raise RuntimeError('Primary model changed from the locked protocol')
    sec=lock['selection_precommitted']['stacking_secondary']
    if (sec['method'],sec['bases'])!=('huber',list(BASES)):
        raise RuntimeError('Stacking specification changed from lock')
    for f,expect in lock['screening_ranking_sha256'].items():
        if file_sha(root/'results'/f)!=expect:
            raise RuntimeError(f'Screening evidence changed after protocol lock: {f}')
    # Feature-only membership is resolved BEFORE accessing target labels.
    # Reading raw CSV does materialize targets in RAM but no target is used to derive groups/splits.
    df=load_data(root/'global_air_pollution_dataset.csv')
    x=raw_predictors(df)
    dev,outer=prospective_split(x)
    screen,confirm=screening_confirmation_split(x.loc[dev])
    assert_group_disjoint(x.loc[dev],x.loc[outer])
    assert_group_disjoint(x.loc[screen],x.loc[confirm])
    if hashlib.sha256(','.join(map(str,sorted(outer))).encode()).hexdigest()!=lock['outer_row_index_sha256']:
        raise RuntimeError('Outer holdout members differ from frozen protocol')
    sizes=lock['split_sizes']
    if (len(screen),len(confirm),len(outer),len(dev))!=(sizes['screening'],sizes['prior_v3_confirmation'],sizes['outer_holdout'],sizes['development_for_final_training']):
        raise RuntimeError('Frozen split sizes changed')
    return lock,path,df,x,dev,outer,screen,confirm


def _joined_screen_results(root,cohort):
    reg=pd.read_csv(Path(root)/'results/regression_OOF_screening.csv').set_index('row_id')
    cls=pd.read_csv(Path(root)/'results/classification_OOF_screening.csv').set_index('row_id')
    if not reg.index.is_unique or not cls.index.is_unique or not reg.index.equals(cohort.X.index) or not cls.index.equals(cohort.X.index):
        raise ValueError('OOF row indexes are not exactly screening membership in original order')
    if reg.isna().any().any() or cls.isna().any().any():
        raise ValueError('Incomplete screening OOF data')
    return reg,cls


def _confusion_f1(m):
    d=m.sum(axis=0)+m.sum(axis=1)
    return float(np.divide(2*np.diag(m),d,out=np.zeros(6),where=d>0).mean())


def group_bootstrap_deltas(y_reg,y_cls,reg_pred,cls_pred,groups,*,n_boot=700,seed=42):
    """Paired cluster bootstrap for fixed, PRECOMMITTED contrasts.

    Descriptive conditional-on-fitted-model uncertainty. NOT correction for the
    historical dataset exploration, country dependence or model selection.
    """
    yy=np.asarray(y_reg,dtype=float)
    cats=np.asarray(y_cls,dtype=int)
    unique,inv=np.unique(np.asarray(groups),return_inverse=True)
    n=len(unique)
    if n<2:raise ValueError('Need multiple predictor-pattern clusters')
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,n,size=(n_boot,n))
    rows=[]
    for a,b in [('log_hgb_A5','log_hgb_A4'),('huber_stack','log_hgb_A5')]:
        delta=np.abs(yy-reg_pred[a])-np.abs(yy-reg_pred[b])
        numerator=np.bincount(inv,weights=delta,minlength=n)
        denominator=np.bincount(inv,minlength=n)
        vals=(numerator[samples].sum(axis=1)/denominator[samples].sum(axis=1))
        rows.append({'task':'regression','a':a,'b':b,'metric':'MAE(a)-MAE(b)',
                     'point_estimate':float(delta.mean()),'ci_low':float(np.quantile(vals,.025)),
                     'ci_high':float(np.quantile(vals,.975))})
    matrices={}
    for key,pred in cls_pred.items():
        enc=inv*36+cats*6+np.asarray(pred,dtype=int)
        matrices[key]=np.bincount(enc,minlength=n*36).reshape(n,36)
    for a,b in [('lightgbm_A4','rf_balanced_A4'),('lightgbm_A4','hgb_A5_to_class')]:
        ma=matrices[a];mb=matrices[b]
        vals=np.empty(n_boot)
        for i,sel in enumerate(samples):
            vals[i]=_confusion_f1(ma[sel].sum(axis=0).reshape(6,6))-_confusion_f1(mb[sel].sum(axis=0).reshape(6,6))
        obs=_confusion_f1(ma.sum(axis=0).reshape(6,6))-_confusion_f1(mb.sum(axis=0).reshape(6,6))
        rows.append({'task':'classification','a':a,'b':b,'metric':'Macro-F1(a)-Macro-F1(b)',
                     'point_estimate':obs,'ci_low':float(np.quantile(vals,.025)),
                     'ci_high':float(np.quantile(vals,.975))})
    ma=matrices['lightgbm_A4'];mb=matrices['rf_balanced_A4']
    recall_d=[]
    for sel in samples:
        ca=ma[sel].sum(axis=0).reshape(6,6)
        cb=mb[sel].sum(axis=0).reshape(6,6)
        da=ca[5,5]/ca[5].sum() if ca[5].sum() else np.nan
        db=cb[5,5]/cb[5].sum() if cb[5].sum() else np.nan
        recall_d.append(da-db)
    ca=ma.sum(axis=0).reshape(6,6);cb=mb.sum(axis=0).reshape(6,6)
    vals=np.asarray(recall_d);vals=vals[np.isfinite(vals)]
    rows.append({'task':'safety','a':'lightgbm_A4','b':'rf_balanced_A4',
                 'metric':'Hazardous recall(a)-recall(b)',
                 'point_estimate':float(ca[5,5]/max(ca[5].sum(),1)-cb[5,5]/max(cb[5].sum(),1)),
                 'ci_low':float(np.quantile(vals,.025)) if len(vals) else np.nan,
                 'ci_high':float(np.quantile(vals,.975)) if len(vals) else np.nan})
    return pd.DataFrame(rows)


def _per_class(y,pred):
    prec,rec,f1,support=precision_recall_fscore_support(y,pred,labels=_LABELS,zero_division=0)
    return pd.DataFrame({'category':CATEGORIES,'support':support,'precision':prec,'recall':rec,'f1':f1})


def absolute_cluster_intervals(y_reg,y_cls,reg_pred,cls_pred,groups,*,n_boot=900,seed=72):
    """Conditional descriptive uncertainty for each frozen model, by input pattern.

    If a rare category is entirely absent in a bootstrap replicate its recall is
    undefined; that replicate is excluded from that category's quantiles.
    """
    y=np.asarray(y_reg,dtype=float)
    c=np.asarray(y_cls,dtype=int)
    unique,inv=np.unique(np.asarray(groups),return_inverse=True)
    n=len(unique)
    gen=np.random.default_rng(seed)
    indices=gen.integers(0,n,(n_boot,n))
    ct=np.bincount(inv,minlength=n)
    rows=[]
    for name,pred in reg_pred.items():
        ae=np.abs(y-np.asarray(pred,dtype=float))
        sums=np.bincount(inv,weights=ae,minlength=n)
        vals=sums[indices].sum(axis=1)/ct[indices].sum(axis=1)
        rows.append({'task':'regression','model':name,'metric':'MAE',
          'point_estimate':float(ae.mean()),'ci_low':float(np.quantile(vals,.025)),
          'ci_high':float(np.quantile(vals,.975))})
    for name,pred in cls_pred.items():
        enc=inv*36+c*6+np.asarray(pred,dtype=int)
        m=np.bincount(enc,minlength=n*36).reshape(n,36)
        f1=np.empty(n_boot);rec=np.empty(n_boot)
        for i,idx in enumerate(indices):
            mat=m[idx].sum(axis=0).reshape(6,6)
            f1[i]=_confusion_f1(mat)
            rec[i]=mat[5,5]/mat[5].sum() if mat[5].sum() else np.nan
        true_m=m.sum(axis=0).reshape(6,6)
        rows.append({'task':'classification','model':name,'metric':'Macro-F1',
          'point_estimate':_confusion_f1(true_m),
          'ci_low':float(np.quantile(f1,.025)),'ci_high':float(np.quantile(f1,.975))})
        valid=rec[np.isfinite(rec)]
        rows.append({'task':'classification','model':name,'metric':'Hazardous recall',
          'point_estimate':float(true_m[5,5]/max(true_m[5].sum(),1)),
          'ci_low':float(np.quantile(valid,.025)) if len(valid) else np.nan,
          'ci_high':float(np.quantile(valid,.975)) if len(valid) else np.nan})
    return pd.DataFrame(rows)


def _reliability(y,prob):
    prob=np.asarray(prob,dtype=float)
    if prob.shape!=(len(y),6) or not np.isfinite(prob).all():
        raise ValueError('Invalid probability matrix')
    confidence=prob.max(axis=1); predict=prob.argmax(axis=1)
    bins=[]
    for i,(lo,hi) in enumerate(zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:])):
        mask=(confidence>=lo)&((confidence<=hi) if i==9 else (confidence<hi))
        bins.append({'bin':i+1,'n':int(mask.sum()),
                     'mean_confidence':float(confidence[mask].mean()) if mask.any() else np.nan,
                     'accuracy':float((predict[mask]==y[mask]).mean()) if mask.any() else np.nan})
    ece=sum(float(b['n'])/len(y)*abs(b['mean_confidence']-b['accuracy']) for b in bins if b['n'])
    brier=float(np.sum((prob-np.eye(6)[np.asarray(y,dtype=int)])**2,axis=1).mean())
    return pd.DataFrame(bins),ece,brier


def _class_counts(y):
    v=np.bincount(np.asarray(y,dtype=int),minlength=6)
    return dict(zip(CATEGORIES,map(int,v)))


def screening_extensions(root, *, progress=print, fast=False):
    """Run only on screening data, never calls holdout accessor."""
    root=Path(root)
    _,_,_,_,_,_,screen,confirm=protocol_and_split(root)
    cohort=load_screening_cohort(root/'global_air_pollution_dataset.csv')
    if not cohort.X.index.equals(screen):raise RuntimeError('Screening membership changed')
    reg_oof,cls_oof=_joined_screen_results(root,cohort)
    out=root/'results'/'batch'
    out.mkdir(parents=True,exist_ok=True)
    grouped=pattern_groups(cohort.X)
    progress('Screening: 7 stacking/blending techniques under nested group CV')
    stacking,trace=compare_stackers_nested(cohort.X,cohort.y,BASES,methods=STACKERS,
                                           outer_folds=2 if fast else 3,inner_folds=2 if fast else 3)
    if any(q['oof_overlap'] for q in trace):raise RuntimeError('Stacker leaks training fold into OOF')
    stacking.to_csv(out/'stacking_development_nested_folds.csv',index=False)
    summary=stacking.groupby('model',as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean'),r2=('r2','mean'))
    summary.sort_values('mae').to_csv(out/'stacking_development_summary.csv',index=False)
    progress('Screening: paired uncertainties and error analysis')
    regpred={'log_hgb_A5':reg_oof.log_hgb_A5.to_numpy(),
             'log_hgb_A4':reg_oof.log_hgb_A4.to_numpy()}
    clspr={'lightgbm_A4':cls_oof.lightgbm_A4.to_numpy(dtype=int),
           'rf_balanced_A4':cls_oof.rf_balanced_A4.to_numpy(dtype=int),
           'hgb_A5_to_class':_threshold_classes(regpred['log_hgb_A5'])}
    # Development bootstrap contracts need no stacking metric (only screen comparisons).
    reg_ext={**regpred,'huber_stack':regpred['log_hgb_A5']}
    pd.DataFrame([{'kind':'screening','item':k,**classifier_scores(cohort.categories,v)}
                  for k,v in clspr.items()]).to_csv(out/'screening_fixed_class_comparison.csv',index=False)
    grouped_contrasts=group_bootstrap_deltas(cohort.y,cohort.categories,
                             reg_ext,clspr,grouped,n_boot=500 if fast else 700,seed=42)
    # Huber-vs-A5 is not compared unless same-fold stacked OOF predictions are available.
    grouped_contrasts=grouped_contrasts.loc[~((grouped_contrasts.a=='huber_stack') &
        (grouped_contrasts.b=='log_hgb_A5'))]
    grouped_contrasts.to_csv(out/'screening_fixed_paired_CIs.csv',index=False)
    for model,pr in clspr.items():
        _per_class(cohort.categories,pr).to_csv(out/f'screening_per_class_{model}.csv',index=False)
    progress('Screening: predeclared country-transfer and seed-robustness checks')
    from .data import group_folds
    country_rows=[]
    for fold,(tr,va) in enumerate(group_folds(cohort.X,n_splits=3,grouping='country'),1):
        for key,kind,features in [('log_hgb_A5','reg','A5'),('lightgbm_A4','cls','A4'),('rf_balanced_A4','cls','A4')]:
            if kind=='reg':
                model=make_advanced_regressor('hgb_log_reference',features).fit(cohort.X.iloc[tr],cohort.y.iloc[tr])
                stats=metrics(cohort.y.iloc[va],model.predict(cohort.X.iloc[va]))
            else:
                mode='lightgbm' if key=='lightgbm_A4' else 'rf_balanced'
                model=make_advanced_classifier(mode,'A4').fit(cohort.X.iloc[tr],cohort.categories.iloc[tr])
                stats=classifier_scores(cohort.categories.iloc[va],model.predict(cohort.X.iloc[va]))
            country_rows.append({'model':key,'fold':fold,'n_validation':len(va),**stats})
    pd.DataFrame(country_rows).to_csv(out/'country_transfer_fixed_models.csv',index=False)
    seed_rows=[]
    for seed in (42,43) if fast else (42,43,44):
        from .research import study_folds
        folds=study_folds(cohort,n_splits=2,seed=seed)
        for fold,(tr,va) in enumerate(folds,1):
            for key,kind,features in [('log_hgb_A5','reg','A5'),('lightgbm_A4','cls','A4'),('rf_balanced_A4','cls','A4')]:
                if kind=='reg':
                    m=make_advanced_regressor('hgb_log_reference',features).fit(cohort.X.iloc[tr],cohort.y.iloc[tr])
                    stat=metrics(cohort.y.iloc[va],m.predict(cohort.X.iloc[va]))
                else:
                    method='lightgbm' if key=='lightgbm_A4' else 'rf_balanced'
                    m=make_advanced_classifier(method,features).fit(cohort.X.iloc[tr],cohort.categories.iloc[tr])
                    stat=classifier_scores(cohort.categories.iloc[va],m.predict(cohort.X.iloc[va]))
                seed_rows.append({'model':key,'seed':seed,'fold':fold,'n_validation':len(va),**stat})
    pd.DataFrame(seed_rows).to_csv(out/'seed_robustness_fixed_models.csv',index=False)
    progress('Screening-only extensions finished')
    return {'nested_stacking':summary,'paired_screening':grouped_contrasts}


def evaluate_final_once(root, *, progress=print):
    """Pre-registered, one pass. Re-runs return existing receipt; no test retraining.

    Train everything before reading holdout targets. Never tune using holdout.
    """
    root=Path(root)
    result_dir=root/'results'/'batch'
    result_dir.mkdir(parents=True,exist_ok=True)
    receipt_path=result_dir/'FINAL_HOLDOUT_RECEIPT.json'
    started_path=result_dir/'HOLDOUT_IN_PROGRESS.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        protocol_file=root/'LOCKED_PROTOCOL.json'
        if receipt.get('protocol_sha256')!=file_sha(protocol_file):
            raise RuntimeError('Existing test receipt does not match locked protocol')
        for filename,checksum in receipt['result_sha256'].items():
            if file_sha(result_dir/filename)!=checksum:
                raise RuntimeError(f'An existing final result has been modified: {filename}')
        progress('Holdout has already been scored once. Skipping evaluation and reading verified receipt.')
        return receipt
    if started_path.exists():
        raise RuntimeError('A previous one-shot evaluation started but did not commit a receipt. Refusing automatic test rerun.')
    lock,protocol_file,df,x,dev,outer,screen,confirm=protocol_and_split(root)
    threadpool_limits(1)
    progress('Final training: all development rows, frozen regression and classification specs')
    # Extract ONLY development labels while training model candidates.
    ydev=df.loc[dev,'PM2.5 AQI Value']
    cdev=pd.Categorical(df.loc[dev,'PM2.5 AQI Category'],categories=CATEGORIES,ordered=True).codes
    if (cdev<0).any():raise RuntimeError('Unexpected development class')
    fitted_reg={
       'log_hgb_A5':make_advanced_regressor('hgb_log_reference','A5').fit(x.loc[dev],ydev),
       'log_hgb_A4':make_advanced_regressor('hgb_log_reference','A4').fit(x.loc[dev],ydev),
    }
    stack,_=fit_stack_on_development(x.loc[dev],ydev,BASES,method='huber',inner_folds=3)
    fitted_cls={
       'lightgbm_A4':make_advanced_classifier('lightgbm','A4').fit(x.loc[dev],cdev),
       'rf_balanced_A4':make_advanced_classifier('rf_balanced','A4').fit(x.loc[dev],cdev),
    }
    progress('All model fitting completed; scoring frozen candidates on outer holdout ONCE')
    Xtest=x.loc[outer]
    predreg={key:np.asarray(model.predict(Xtest),dtype=float) for key,model in fitted_reg.items()}
    predreg['huber_stack']=np.asarray(stack.predict(Xtest),dtype=float)
    predcls={key:np.asarray(model.predict(Xtest),dtype=int) for key,model in fitted_cls.items()}
    predcls['hgb_A5_to_class']=_threshold_classes(predreg['log_hgb_A5'])
    # Persist the irreversible evaluation boundary *before* accessing test targets.
    started_path.write_text(json.dumps({'protocol_sha256':file_sha(protocol_file),
      'status':'final_holdout_scoring_started','policy':'never automatically retry after interruption'},indent=2))
    # This is the ONLY location where holdout target arrays ENTER MODEL SCORING.
    # Earlier loader integrity checks validate the entire immutable CSV, but
    # cannot affect partitioning, fitting or model selection.
    ytest=df.loc[outer,'PM2.5 AQI Value'].to_numpy(dtype=float)
    ctest=pd.Categorical(df.loc[outer,'PM2.5 AQI Category'],categories=CATEGORIES,ordered=True).codes
    if (ctest<0).any():raise RuntimeError('Unexpected holdout labels')
    if len(ytest)!=lock['split_sizes']['outer_holdout']:raise RuntimeError('Wrong holdout row count')
    # No performance-dependent branching after this point, and no model fit or selection.
    regression=pd.DataFrame([{'model':name,'n':len(ytest),**metrics(ytest,yp)} for name,yp in predreg.items()])
    classification=pd.DataFrame([{'model':name,'n':len(ctest),**classifier_scores(ctest,yp)} for name,yp in predcls.items()])
    regression.to_csv(result_dir/'final_regression_fixed.csv',index=False)
    classification.to_csv(result_dir/'final_classification_fixed.csv',index=False)
    contrasts=group_bootstrap_deltas(ytest,ctest,predreg,predcls,pattern_groups(Xtest),n_boot=900,seed=402)
    contrasts.to_csv(result_dir/'final_paired_pattern_CIs.csv',index=False)
    absolute_cluster_intervals(ytest,ctest,predreg,predcls,pattern_groups(Xtest),n_boot=900,seed=724).to_csv(
       result_dir/'final_absolute_pattern_CIs.csv',index=False)
    rows=[]
    for name,pred in predcls.items():
        sub=_per_class(ctest,pred)
        sub.insert(0,'model',name)
        rows.append(sub)
        pd.DataFrame(confusion_matrix(ctest,pred,labels=_LABELS),index=CATEGORIES,columns=CATEGORIES).to_csv(
          result_dir/f'final_confusion_{name}.csv')
    pd.concat(rows,ignore_index=True).to_csv(result_dir/'final_six_class_metrics.csv',index=False)
    probs=np.asarray(fitted_cls['lightgbm_A4'].predict_proba(Xtest),dtype=float)
    bins,ece,brier=_reliability(ctest,probs)
    bins.to_csv(result_dir/'final_calibration_bins.csv',index=False)
    near=np.min(np.abs(ytest[:,None]-np.asarray([50.5,99.5,149.5,200.5,299.5])[None,:]),axis=1)<=5
    errors=[]
    for key,pred in predcls.items():
        for label,mask in [('within_5_AQI_of_boundary',near),('outside_boundary_band',~near)]:
            errors.append({'model':key,'zone':label,'n':int(mask.sum()),
                           'accuracy':float((pred[mask]==ctest[mask]).mean()),
                           'ordinal_ge2_rate':float((np.abs(pred[mask]-ctest[mask])>=2).mean())})
    pd.DataFrame(errors).to_csv(result_dir/'final_boundary_errors.csv',index=False)
    extras={'ece_10_bin_lightgbm_A4':ece,'multiclass_brier_lightgbm_A4':brier,
            'outer_n':len(ytest),'outer_pattern_groups':int(pattern_groups(Xtest).nunique()),
            'class_counts':_class_counts(ctest),'data_sha256':DATA_SHA256,
            'training_size_all_development':len(dev),
            'warning':'Prior historical test and v3 confirmation were already examined; not external validation.',
            'rule':'No model changes, tuning, or ranking after holdout scores.'}
    (result_dir/'final_supplementary.json').write_text(json.dumps(extras,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    names=('final_regression_fixed.csv','final_classification_fixed.csv','final_paired_pattern_CIs.csv',
           'final_six_class_metrics.csv','final_calibration_bins.csv','final_boundary_errors.csv',
           'final_absolute_pattern_CIs.csv',
           'final_supplementary.json',*(f'final_confusion_{s}.csv' for s in _CLASS_NAMES))
    receipt={'status':'evaluated_once','protocol_sha256':file_sha(protocol_file),
             'holdout_count':len(outer),'model_specs_prespecified':list(_REG_NAMES)+list(_CLASS_NAMES),
             'result_sha256':{f:file_sha(result_dir/f) for f in names},
             'disclosure':'Data previously used for historical test/model exploration; not independent external validation',
             'post_holdout_tuning':'prohibited'}
    # Receipt is the final operation; publication/figures use aggregate CSV only.
    receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    started_path.unlink(missing_ok=True)
    progress('One-shot fixed holdout scoring complete. Receipt written; any rerun skips holdout.')
    return receipt
