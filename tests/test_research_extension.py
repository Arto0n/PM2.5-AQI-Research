"""No held-out targets or target-derived predictors can drive research extension."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pm25aqi.advanced import (make_advanced_classifier,make_advanced_regressor,
                              REGRESSION_CANDIDATES,CLASSIFICATION_CANDIDATES)
from pm25aqi.config import EXTENDED_FEATURE_SETS
from pm25aqi.data import raw_predictors, pattern_groups
from pm25aqi.features import RichInteractionFeatures
from pm25aqi.research import (load_screening_cohort,study_folds,_threshold_classes,
                               cluster_paired_bootstrap,ScreeningCohort)


def test_extended_features_never_read_target():
    x=pd.DataFrame({'CO AQI Value':[1.,3.,5.],'Ozone AQI Value':[10.,20.,5.],
                    'NO2 AQI Value':[4.,2.,8.],'Country':['X','Y','X'],
                    'AQI Value':[500]*3,'PM2.5 AQI Value':[500]*3,'City':['A','B','C']})
    a=RichInteractionFeatures().fit_transform(x)
    x['PM2.5 AQI Value']=0
    x['AQI Value']=-9999
    b=RichInteractionFeatures().fit_transform(x)
    pd.testing.assert_frame_equal(a,b)
    assert all(col in a.columns for col in EXTENDED_FEATURE_SETS['A5'])
    assert not any(c in a.columns for c in ['PM2.5 AQI Value','AQI Value','City'])


def test_enhanced_models_never_access_target_columns():
    x=pd.DataFrame({'CO AQI Value':np.arange(36)%4, 'Ozone AQI Value':np.arange(36)%11,
       'NO2 AQI Value':np.arange(36)%7,'Country':['X']*18+['Y']*18,
       'AQI Value':np.arange(36)+200,'PM2.5 AQI Value':np.arange(36)*2})
    for model in ('catboost','xgboost','lightgbm','mlp'):
        est=make_advanced_regressor(model,'A5')
        est.fit(x.iloc[:30],np.arange(30,dtype=float))
        p=est.predict(x.iloc[30:])
        mod=x.copy();mod['AQI Value']=-999.;mod['PM2.5 AQI Value']=-999.
        assert np.allclose(p,est.predict(mod.iloc[30:]),rtol=1e-5,atol=1e-5)


@pytest.mark.requires_data
def test_grouped_screening_only_on_unchanged_dataset():
    cohort=load_screening_cohort(Path(__file__).parents[1]/'global_air_pollution_dataset.csv')
    assert len(cohort.X)==14112
    assert cohort.n_outer_sealed==4707 and cohort.n_confirmation_sealed==4644
    assert 'AQI Value' not in cohort.X.columns and 'PM2.5 AQI Value' not in cohort.X.columns
    folds=study_folds(cohort)
    g=pattern_groups(cohort.X)
    for tr,va in folds:
        assert not(set(g.iloc[tr])&set(g.iloc[va]))
    assert sorted(np.concatenate([va for tr,va in folds]))==list(range(len(cohort.X)))


def test_grouped_bootstrap_fixed_estimands():
    x=pd.DataFrame({'CO AQI Value':np.arange(48)%4,'Ozone AQI Value':np.arange(48)%6,
       'NO2 AQI Value':np.arange(48)%7,'Country':['X']*48})
    cohort=ScreeningCohort(x,pd.Series(np.arange(48,dtype=float)),
        pd.Series(np.arange(48)%6),48,0,0)
    yr=pd.DataFrame({'catboost_A4':np.arange(48)+1.,'log_hgb_A4':np.arange(48),
       'xgboost_A4':np.arange(48)+2.,'lightgbm_A4':np.arange(48)+.5,
       'log_hgb_A5':np.arange(48)+.25})
    yc=pd.DataFrame({'catboost_A4':np.arange(48)%6,'rf_balanced_A4':np.arange(48)%6,
       'xgboost_A4':np.arange(48)%6,'lightgbm_A4':np.arange(48)%6,
       'rf_balanced_A5':np.arange(48)%6})
    res=cluster_paired_bootstrap(cohort,yr,yc,n_boot=15)
    assert len(res)==8 and np.isfinite(res[['estimate','CI_low','CI_high']]).all().all()
    assert (res[res.task=='classification'].estimate==0).all()


def test_no_target_based_feature_regression():
    assert set(REGRESSION_CANDIDATES)
    assert set(CLASSIFICATION_CANDIDATES)
    assert set(EXTENDED_FEATURE_SETS['A5']).isdisjoint({'AQI Value','AQI Category',
             'PM2.5 AQI Value','PM2.5 AQI Category','City'})
    assert _threshold_classes([0,50,51,99,100,149,150,200,201,299,300,500]).tolist()==[0,0,1,1,2,2,3,3,4,4,5,5]


def test_class_specific_paired_intervals_respect_grouping():
    from pm25aqi.research import class_specific_cluster_intervals
    x=pd.DataFrame({'CO AQI Value':np.arange(72)%5,'Ozone AQI Value':np.arange(72)%13,
       'NO2 AQI Value':np.arange(72)%8,'Country':['X']*36+['Y']*36})
    true=np.arange(72)%6
    cohort=ScreeningCohort(x,pd.Series(np.arange(72,dtype=float)*5),pd.Series(true),72,0,0)
    cls=pd.DataFrame({'rf_balanced_A4':true})
    reg=pd.DataFrame({'log_hgb_A4':np.arange(72,dtype=float)*5})
    out,comparison=class_specific_cluster_intervals(cohort,cls,reg,n_boot=15)
    assert len(out)==6 and len(comparison)==1
    assert (out['direct_RF_A4_F1']==1).all()
    assert np.isfinite(comparison[['estimate','CI_low','CI_high']]).all().all()
