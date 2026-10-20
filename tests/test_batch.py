"""One-way selection and holdout integrity tests (without scoring holdout labels)."""
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pm25aqi.batch import (protocol_and_split, group_bootstrap_deltas,
   evaluate_final_once, _reliability)
from pm25aqi.data import pattern_groups, assert_group_disjoint
from pm25aqi.config import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.requires_data
def test_locked_protocol_precedes_holdout_access():
    lock,path,df,x,dev,outer,screen,confirm=protocol_and_split(ROOT)
    assert path.is_file() and lock['locked_before_holdout_evaluation']
    assert (len(screen),len(confirm),len(outer))==(14112,4644,4707)
    assert_group_disjoint(x.loc[dev],x.loc[outer])
    assert_group_disjoint(x.loc[screen],x.loc[confirm])
    assert lock['selection_precommitted']['regression_primary']['name']=='log_hgb_A5'
    assert lock['selection_precommitted']['classification_primary']['name']=='lightgbm_A4'


@pytest.mark.requires_data
def test_protocol_does_not_use_holdout_labels_for_membership():
    from pm25aqi.data import prospective_split, raw_predictors
    lock,path,df,x,dev,outer,screen,confirm=protocol_and_split(ROOT)
    changed=df.copy()
    changed.loc[outer,'PM2.5 AQI Value']=499
    changed.loc[outer,'PM2.5 AQI Category']='Hazardous'
    dv,ho=prospective_split(raw_predictors(changed))
    assert np.array_equal(np.asarray(dev),np.asarray(dv))
    assert np.array_equal(np.asarray(outer),np.asarray(ho))


def test_bootstrap_metrics_are_paired_by_pattern():
    y=np.array([10.,11.,20.,21.,30.,31.,40.,41.])
    true=np.array([0,0,1,1,2,2,5,5])
    reg={'log_hgb_A5':y+2.,'log_hgb_A4':y+4.,'huber_stack':y+3.}
    pred={'lightgbm_A4':true.copy(),'rf_balanced_A4':np.zeros(8,dtype=int),
          'hgb_A5_to_class':true.copy()}
    df=group_bootstrap_deltas(y,true,reg,pred,np.array([1,1,2,2,3,3,4,4]),n_boot=30)
    assert len(df)==5
    assert np.isfinite(df[['point_estimate','ci_low','ci_high']].to_numpy()).all()
    assert df.iloc[0].point_estimate==-2.


def test_reliability_proportions():
    y=np.array([0,1,2,3,4,5])
    p=np.eye(6)*.9 + np.full((6,6),.1/6)
    p=p/p.sum(axis=1,keepdims=True)
    bins,ece,brier=_reliability(y,p)
    assert bins.n.sum()==6
    assert ece>=0 and brier>=0


def test_no_after_test_model_selection_branch():
    source=inspect.getsource(evaluate_final_once)
    assert "model_specs_prespecified" in source
    assert 'post_holdout_tuning' in source
    assert 'HOLDOUT_IN_PROGRESS.json' in source
    assert "results/batch" not in source or 'FINAL_HOLDOUT_RECEIPT.json' in source


def test_receipt_skip_without_refitting(tmp_path,monkeypatch):
    import pm25aqi.batch as batch
    from hashlib import sha256
    (tmp_path/'results'/'batch').mkdir(parents=True)
    (tmp_path/'LOCKED_PROTOCOL.json').write_text('{}')
    payload=(tmp_path/'results'/'batch'/'final_regression_fixed.csv')
    payload.write_text('example,data\n')
    rec={'protocol_sha256':batch.file_sha(tmp_path/'LOCKED_PROTOCOL.json'),
         'result_sha256':{'final_regression_fixed.csv':batch.file_sha(payload)}}
    (payload.parent/'FINAL_HOLDOUT_RECEIPT.json').write_text(json.dumps(rec))
    monkeypatch.setattr(batch,'protocol_and_split',lambda root: (_ for _ in ()).throw(AssertionError('test labels accessed')))
    actual=batch.evaluate_final_once(tmp_path,progress=lambda x:None)
    assert actual==rec


def test_incomplete_started_receipt_blocks_rerun(tmp_path):
    d=tmp_path/'results'/'batch';d.mkdir(parents=True)
    (d/'HOLDOUT_IN_PROGRESS.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Refusing automatic test rerun'):
        evaluate_final_once(tmp_path,progress=lambda x:None)


def test_holdout_not_being_written_rowwise_in_final_batch():
    s=inspect.getsource(evaluate_final_once)
    assert "ytest.to_csv" not in s and "ctest.to_csv" not in s
    assert "grid_search" not in s.lower() and "rank_test_models" not in s
