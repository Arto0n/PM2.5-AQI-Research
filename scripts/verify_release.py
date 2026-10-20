"""Cryptographically verify the locked analysis record WITHOUT dataset or model fitting."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'regression_primary':('log_hgb_A5','mae',16.936984717820586),
          'regression_huber':('huber_stack','mae',16.87248979035367),
          'classification_primary':('lightgbm_A4','macro_f1',0.4977033498920321),
          'classification_rf':('rf_balanced_A4','macro_f1',0.5537140055344066)}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(root=ROOT, require_data=False):
    root=Path(root)
    lock_path=root/'LOCKED_PROTOCOL.json'; rec_path=root/'results/batch/FINAL_HOLDOUT_RECEIPT.json'
    lock=json.loads(lock_path.read_text(encoding='utf-8'))
    receipt=json.loads(rec_path.read_text(encoding='utf-8'))
    assert lock['locked_before_holdout_evaluation'] is True, 'Protocol is not locked'
    assert receipt['status']=='evaluated_once', 'Expected the original one-shot receipt'
    assert receipt['protocol_sha256']==sha(lock_path),'Protocol digest mismatch'
    assert lock['selection_precommitted']['regression_primary']['name']=='log_hgb_A5'
    assert lock['selection_precommitted']['classification_primary']['name']=='lightgbm_A4'
    assert receipt['model_specs_prespecified']==[
        'log_hgb_A5','log_hgb_A4','huber_stack',
        'lightgbm_A4','rf_balanced_A4','hgb_A5_to_class']
    assert receipt['holdout_count']==4707
    for name,expected in receipt['result_sha256'].items():
        file=root/'results'/'batch'/name
        assert file.is_file(),f'Missing published result: {name}'
        assert sha(file)==expected,f'Archived published result has changed: {name}'
    reg=pd.read_csv(root/'results/batch/final_regression_fixed.csv').set_index('model')
    cls=pd.read_csv(root/'results/batch/final_classification_fixed.csv').set_index('model')
    for name,(model,metric,expected) in EXPECTED.items():
        actual=float((reg if metric=='mae' else cls).loc[model,metric])
        assert abs(actual-expected)<1e-11, f'{name} no longer matches published evidence'
    dataset=root/'global_air_pollution_dataset.csv'
    if dataset.exists():
        assert sha(dataset)==lock['data_sha256'], 'Local CSV is not the exact frozen data'
    elif require_data:
        raise FileNotFoundError('Data missing: run python scripts/download_data.py')
    return {'receipt_files_verified':len(receipt['result_sha256']),
            'holdout_rows_archived':receipt['holdout_count'],
            'dataset_available':dataset.exists(),
            'dataset_expected_sha256':lock['data_sha256']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-data',action='store_true')
    args=parser.parse_args()
    try:
        print(json.dumps(verify(require_data=args.require_data),indent=2))
        print('PASS: archived fixed results and protocol verified; no training or scoring performed.')
    except Exception as exc:
        print(f'FAIL: {exc}',file=sys.stderr);sys.exit(1)
