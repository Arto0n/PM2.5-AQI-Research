"""Selection, group isolation, feature ablations and confirmation safeguards."""
import inspect
import numpy as np
import pandas as pd
import pytest
from pm25aqi.classification import (ClassifierChoice, screening_confirmation_split,
    classifier_scores, classifier_screen, summarize_classifier_screen,
    choose_classifier, confirm_frozen_classifier, multiclass_calibration,
    pattern_bootstrap_ci, classifier_country_sensitivity)
from pm25aqi.config import CATEGORIES, FEATURE_SETS, TARGET
from pm25aqi.data import raw_predictors, assert_group_disjoint, pattern_groups
from pm25aqi.models import CLASSIFIERS, make_classifier


def six_class_toy():
    rng=np.random.default_rng(321)
    n=288; pattern=np.arange(n)//2
    frame=pd.DataFrame({
        'CO AQI Value': (pattern*7 % 140).astype(float),
        'Ozone AQI Value': (pattern*13 % 110).astype(float),
        'NO2 AQI Value': (pattern*11 % 100).astype(float),
        'Country':np.where(pattern%3==0,'A',np.where(pattern%3==1,'B','C'))
    })
    cls=(pattern % 6).astype(int)
    frame['PM2.5 AQI Value']=cls*60.+10
    frame['PM2.5 AQI Category']=[CATEGORIES[i] for i in cls]
    frame['AQI Value']=cls*80
    X=raw_predictors(frame); y=pd.Series(cls,index=X.index,dtype=int)
    return X,y,frame


def test_grid_crosses_all_models_with_all_feature_sets():
    X,y,_=six_class_toy()
    models=('dummy_majority','logistic_plain')
    scores=classifier_screen(X,y,candidates=models,feature_sets=tuple(FEATURE_SETS),n_splits=3)
    assert len(scores)==len(models)*len(FEATURE_SETS)*3
    assert len(scores.groupby(['model','feature_set']))==8
    for c in ('accuracy','macro_f1','weighted_f1','balanced_accuracy'):
        assert scores[c].between(0,1).all()
    summary=summarize_classifier_screen(scores)
    assert len(summary)==8
    assert choose_classifier(summary).model in models


def test_dev_confirmation_disjoint_and_frozen_choice_ignores_confirmation_labels():
    X,y,_=six_class_toy()
    a,b=screening_confirmation_split(X)
    assert set(a).isdisjoint(b)
    assert len(a)+len(b)==len(X)
    assert_group_disjoint(X.loc[a],X.loc[b])
    changed=y.copy();changed.loc[b]=(changed.loc[b]+1)%6
    a2,b2=screening_confirmation_split(X)
    assert a.equals(a2) and b.equals(b2)
    # Split chosen once before experiment; selection is only a function of screening rows.
    params=dict(candidates=('dummy_majority','logistic_plain'),feature_sets=('A1','A4'))
    first=summarize_classifier_screen(classifier_screen(X.loc[a],y.loc[a],**params))
    second=summarize_classifier_screen(classifier_screen(X.loc[a],changed.loc[a],**params))
    pd.testing.assert_frame_equal(first,second)
    choice=choose_classifier(first)
    est,score,per,cm,pred=confirm_frozen_classifier(X,y,a,b,choice)
    assert cm.shape==(6,6) and cm.sum()==len(b)
    assert per.support.sum()==len(b)
    assert pred.index.equals(b)
    assert score['macro_f1']>=0
    assert choice==choose_classifier(second)
    ci=pattern_bootstrap_ci(pred,X.loc[b],n_boot=25)
    assert ci['lower_95'].notna().all()
    assert multiclass_calibration(est,X.loc[b],y.loc[b])['multiclass_brier']>=0


def test_confirmation_cannot_be_ranked_or_bypass_group_boundary():
    X,y,_=six_class_toy()
    a,b=screening_confirmation_split(X)
    leader=pd.DataFrame([{'model':'dummy_majority','feature_set':'A1',
                          'macro_f1':.1,'balanced_accuracy':.2,'ordinal_mae':2.}])
    assert choose_classifier(leader)==ClassifierChoice('dummy_majority','A1')
    with pytest.raises(ValueError): choose_classifier(leader.assign(confirmation_macro_f1=.5))
    with pytest.raises(ValueError):summarize_classifier_screen(leader.assign(test_accuracy=.5))
    with pytest.raises(AssertionError):
        confirm_frozen_classifier(X,y,a,b.append(a[:1]),ClassifierChoice('dummy_majority','A1'))
    with pytest.raises(AssertionError):
        confirm_frozen_classifier(X,y,a,b.append(b[:1]),ClassifierChoice('dummy_majority','A1'))


def test_strict_predictor_whitelist_and_confirmation_target_independence():
    X,y,frame=six_class_toy()
    a,b=screening_confirmation_split(X)
    model=make_classifier('logistic_plain','A4').fit(frame.loc[a],y.loc[a])
    original=model.predict(frame.loc[b])
    poison=frame.loc[b].copy()
    for target in ('PM2.5 AQI Value','PM2.5 AQI Category','AQI Value','AQI Category'):
        poison[target]='poison value'
    np.testing.assert_array_equal(original,model.predict(poison))
    assert {'A1','A2','A3','A4'}.issubset(FEATURE_SETS)
    assert {'rf_balanced','rf_plain','extra_trees','logistic','knn','hist_gradient_boosting'}.issubset(CLASSIFIERS)
    assert 'holdout' not in inspect.signature(classifier_screen).parameters
    assert 'holdout' not in inspect.signature(confirm_frozen_classifier).parameters


def test_group_ci_uses_repeated_patterns_and_rejects_misalignment():
    X,y,_=six_class_toy()
    pred=pd.DataFrame({'true':y,'predicted':y},index=X.index)
    ci=pattern_bootstrap_ci(pred,X,n_boot=15)
    assert ci.iloc[0].lower_95==pytest.approx(1.)
    with pytest.raises(ValueError):pattern_bootstrap_ci(pred.iloc[1:],X,n_boot=5)


def test_rare_class_is_part_of_macro_average():
    y=np.array([0]*19+[5]);pred=np.zeros_like(y)
    s=classifier_scores(y,pred)
    assert s['accuracy']==.95
    assert s['macro_f1']<.2 # all six classes counted, even absent
    assert s['ordinal_mae']==.25


def test_class_score_labels_rejected():
    with pytest.raises(ValueError): classifier_scores([0,1],[0,9])
    with pytest.raises(ValueError): classifier_scores([0],[0,1])


def test_country_cv_works_without_holdout():
    X,y,_=six_class_toy()
    a,_=screening_confirmation_split(X)
    result=classifier_country_sensitivity(X.loc[a],y.loc[a],ClassifierChoice('dummy_majority','A1'))
    assert len(result)==3 and result['macro_f1'].notna().all()
