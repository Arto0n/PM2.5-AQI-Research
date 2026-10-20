import inspect
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pm25aqi.config import (INPUTS,CATEGORIES,DATA_SHA256,HISTORICAL_TEST,
                            TARGET,HISTORICAL_MODEL)
from pm25aqi.data import (categories_from_aqi,load_data,raw_predictors,
    pattern_groups,assert_group_disjoint,prospective_split,group_folds,development_only)
from pm25aqi.features import InteractionFeatures
from pm25aqi.models import make_regressor,make_classifier,REGRESSORS,CLASSIFIERS
from pm25aqi.validation import (compare_regressors,choose_development_winner,
    summarize_regression,_validate_folds,compare_classifiers,group_transfer_sensitivity)
from pm25aqi.stacking import (inner_oof,StackerMeta,STACKERS,
    compare_stackers_nested,fit_stack_on_development)
from pm25aqi.inference import predict_pm25


def test_category_boundaries():
    vals=[0,50,51,99,100,149,150,200,201,299,300,500]
    assert list(categories_from_aqi(vals))==[x for cat in CATEGORIES for x in (cat,cat)]

@pytest.mark.parametrize('bad',[-1,501,float('nan'),float('inf')])
def test_invalid_aqi(bad):
    with pytest.raises(ValueError): categories_from_aqi([bad])


@pytest.mark.requires_data
def test_fingerprint_dataset_reference():
    path=Path(__file__).resolve().parents[1]/'global_air_pollution_dataset.csv'
    x=load_data(path)
    assert x.shape==(23463,12)
    assert x['PM2.5 AQI Category'].nunique()==6


def test_blacklist_overall_aqi_and_target(toy):
    before=InteractionFeatures().fit_transform(toy)
    changed=toy.copy()
    for c in ['AQI Value','AQI Category','PM2.5 AQI Value','PM2.5 AQI Category']:
        changed[c]='UNTRUSTED_LABEL'
    pd.testing.assert_frame_equal(before,InteractionFeatures().fit_transform(changed))
    assert before.columns.tolist()==[*INPUTS,'CO_x_NO2','CO_x_Ozone']


def test_interactions(toy):
    feats=InteractionFeatures().fit_transform(toy)
    np.testing.assert_array_equal(feats['CO_x_NO2'],toy['CO AQI Value']*toy['NO2 AQI Value'])
    np.testing.assert_array_equal(feats['CO_x_Ozone'],toy['CO AQI Value']*toy['Ozone AQI Value'])


def test_missing_feature_and_infinite(toy):
    with pytest.raises(ValueError): raw_predictors(toy.drop(columns=['Country']))
    poisoned=toy.copy();poisoned.loc[0,'NO2 AQI Value']=np.inf
    with pytest.raises(ValueError): raw_predictors(poisoned)


def test_outer_split_feature_only_and_disjoint(toy):
    a,b=prospective_split(toy)
    assert len(a)+len(b)==len(toy)
    assert not set(a)&set(b)
    assert_group_disjoint(toy.loc[a],toy.loc[b])
    changed=toy.copy()
    changed.loc[b,TARGET]=500
    changed.loc[b,'PM2.5 AQI Category']='Hazardous'
    a2,b2=prospective_split(changed)
    assert list(a)==list(a2) and list(b)==list(b2)
    original=development_only(toy,a); changed_dev=development_only(changed,a)
    for x,y in zip(original,changed_dev):
        pd.testing.assert_series_equal(x,y) if isinstance(x,pd.Series) else pd.testing.assert_frame_equal(x,y)


def test_cv_group_disjoint_and_complete(toy):
    folds=group_folds(toy,n_splits=4)
    _validate_folds(toy,folds)
    for tr,va in folds:
        assert not set(pattern_groups(toy.iloc[tr])) & set(pattern_groups(toy.iloc[va]))


def test_cv_rejects_group_overlap_and_duplicate_rows(toy):
    with pytest.raises(AssertionError):
        _validate_folds(toy,[(np.arange(0,100),np.arange(95,144))])
    with pytest.raises(AssertionError):
        _validate_folds(toy,[(np.arange(0,50),np.arange(50,70))])


def test_no_test_in_selection_fields():
    scores=pd.DataFrame([{'model':'A','kind':'single','mae':1.,'rmse':3.},
                         {'model':'B','kind':'stacking','mae':2.,'rmse':2.}])
    assert choose_development_winner(scores)==('single','A')
    with pytest.raises(ValueError):
        choose_development_winner(scores.assign(test_mae=[0.,0.]))
    with pytest.raises(ValueError):
        summarize_regression(pd.DataFrame({'test_mae':[0.], 'model':['A']}))
    assert 'y_test' not in inspect.signature(compare_regressors).parameters
    assert 'X_test' not in inspect.signature(compare_stackers_nested).parameters


def test_fold_local_median_and_unseen_country(toy):
    x=raw_predictors(toy.iloc[:12]).copy();x['CO AQI Value']=float('nan')
    x.loc[x.index[:6],'CO AQI Value']=[0.,2.]*3
    x.loc[x.index[6:],'Country']='UNSEEN'
    m=make_regressor('dummy').fit(x.iloc[:6],toy[TARGET].iloc[:6])
    imputer=m.named_steps['prepare'].named_transformers_['numeric'].named_steps['fill']
    assert imputer.statistics_[0]==1.
    assert np.isfinite(m.predict(x.iloc[6:])).all()
    cats=m.named_steps['prepare'].named_transformers_['country'].named_steps['onehot'].categories_[0]
    assert 'UNSEEN' not in cats


def test_no_holdout_target_used_in_prediction(toy):
    train=toy.iloc[:100];val=toy.iloc[100:].copy()
    m=make_regressor('hgb_log_reference').fit(train,train[TARGET])
    original=m.predict(val)
    val['AQI Value']=-1;val['PM2.5 AQI Value']=-100;val['PM2.5 AQI Category']='BAD'
    np.testing.assert_allclose(original,m.predict(val),rtol=0,atol=0)


def test_compare_fixed_models_dev_only(toy):
    X=raw_predictors(toy);y=toy[TARGET]
    scores=compare_regressors(X,y,['dummy','ridge'],folds=group_folds(X))
    assert len(scores)==6 and set(scores['kind'])=={'single'}
    winner=choose_development_winner(summarize_regression(scores))
    assert winner[0]=='single'


def test_oof_no_train_validation_pattern_overlap(toy):
    x=raw_predictors(toy);y=toy[TARGET]
    oof=inner_oof(x,y,{'D':'dummy','R':'ridge'},folds=3)
    assert len(oof.predictions)==len(x)
    assert len(oof.traces)==6
    for t in oof.traces:
        assert set(t['train_ids']).isdisjoint(t['validation_ids'])
        assert_group_disjoint(x.loc[list(t['train_ids'])],x.loc[list(t['validation_ids'])])


def test_validation_target_change_does_not_change_its_oof(toy):
    x=raw_predictors(toy); y=toy[TARGET].copy()
    original=inner_oof(x,y,{'D':'dummy'})
    held_ids=original.traces[0]['validation_ids']
    changed=y.copy();changed.loc[list(held_ids)]=499
    altered=inner_oof(x,changed,{'D':'dummy'})
    np.testing.assert_allclose(original.predictions.loc[list(held_ids)],
                               altered.predictions.loc[list(held_ids)])

@pytest.mark.parametrize('strategy',STACKERS)
def test_stack_meta_stable_and_trainable(strategy):
    x=np.array([[1.,2.,3.],[2,1,3],[4,3,2],[7,8,6],[9,6,8],[4,5,6],[7,6,8],[10,11,9]],dtype=float)
    y=np.array([2,2,3,7,8,5,6,10],dtype=float)
    m=StackerMeta(strategy).fit(x,y)
    assert np.isfinite(m.predict(x)).all()
    if strategy=='simplex':
        assert np.isclose(m.weights_.sum(),1.) and (m.weights_>=0).all()


def test_nested_grouped_stacking_no_outer_leakage(toy):
    x=raw_predictors(toy);y=toy[TARGET]
    table, traces=compare_stackers_nested(x,y,{'D':'dummy','R':'ridge'},
                                     ['mean','ridge','simplex'],outer_folds=3,inner_folds=2)
    assert len(table)==9 and not table['mae'].isna().any()
    assert all(not row['oof_overlap'] for row in traces)
    assert all(set(row['outer_train_ids']).isdisjoint(row['outer_validation_ids']) for row in traces)


def test_fit_stacked_predict_and_pickle(toy):
    x=raw_predictors(toy);y=toy[TARGET]
    fitted,oof=fit_stack_on_development(x,y,{'D':'dummy','R':'ridge'},method='ridge',inner_folds=3)
    before=fitted.predict(x.iloc[:10]);after=pickle.loads(pickle.dumps(fitted)).predict(x.iloc[:10])
    np.testing.assert_allclose(before,after)
    result=predict_pm25(fitted,x.iloc[:10])
    assert result['PM2.5_AQI_estimate'].between(0,500).all()


def test_inference_rejects_invalid_and_accepts_unseen(toy):
    m=make_regressor('ridge').fit(toy.iloc[:100],toy[TARGET].iloc[:100])
    x=raw_predictors(toy.iloc[100:105]);x['Country']='COUNTRY_NOT_SEEN'
    assert len(predict_pm25(m,x))==5
    x.loc[x.index[0],'CO AQI Value']=1000
    with pytest.raises(ValueError):predict_pm25(m,x)


def test_classifier_pattern_grouped_dev(toy):
    x=raw_predictors(toy)
    classes=pd.Series(pd.Categorical(toy['PM2.5 AQI Category'],categories=CATEGORIES).codes,index=x.index)
    results=compare_classifiers(x,classes,['logistic'],n_splits=3)
    assert len(results)==3 and results['macro_f1'].between(0,1).all()


def test_registered_models_all_constructible():
    assert len(REGRESSORS)>=10 and len(CLASSIFIERS)>=5 and len(STACKERS)>=6
    for name in REGRESSORS: assert make_regressor(name) is not None
    for name in CLASSIFIERS: assert make_classifier(name) is not None


def test_no_post_test_ranking_notebook():
    import json
    notebook=json.loads((Path(__file__).resolve().parents[1]/'PM2.5_Research.ipynb').read_text())
    text='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert 'all_regression_candidates_test' not in text
    assert 'y_test' not in text and 'X_test' not in text
    assert 'train_test_split' not in text
    # v5 delegates nested stacking to a tested module instead of inlining it
    # in the notebook; verify the notebook->runner->stacking call chain.
    if 'compare_stackers_nested' not in text:
        import inspect
        from pm25aqi.batch import screening_extensions
        assert ('run_all()' in text or 'verify(' in text)
        assert 'compare_stackers_nested' in inspect.getsource(screening_extensions)


@pytest.mark.slow
@pytest.mark.requires_data
def test_historical_selected_hgb_reference():
    """Fixed historical protocol, ONE prespecified selected model; NOT a search."""
    df=load_data(Path(__file__).resolve().parents[1]/'global_air_pollution_dataset.csv')
    indices=np.arange(len(df))
    codes=pd.Categorical(df['PM2.5 AQI Category'],categories=CATEGORIES).codes
    dev,test=train_test_split(indices,test_size=.2,random_state=42,stratify=codes)
    est=make_regressor(HISTORICAL_MODEL).fit(raw_predictors(df.iloc[dev]),df.iloc[dev][TARGET])
    preds=est.predict(raw_predictors(df.iloc[test]));true=df.iloc[test][TARGET]
    assert abs(mean_absolute_error(true,preds)-HISTORICAL_TEST['mae'])<1e-5
    assert abs(np.sqrt(mean_squared_error(true,preds))-HISTORICAL_TEST['rmse'])<1e-5
    assert abs(r2_score(true,preds)-HISTORICAL_TEST['r2'])<1e-5
