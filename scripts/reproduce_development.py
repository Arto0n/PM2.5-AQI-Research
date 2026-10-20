"""Rerun grouped development-only screening and/or nested stacking to a separate folder.

The inspected outer holdout is NEVER scored here. Frozen results are never overwritten.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def run(mode='all',dest=None):
    from scripts.verify_release import verify
    info=verify(require_data=True)
    output=Path(dest or ROOT/'artifacts/reproduced').resolve()
    output.mkdir(parents=True,exist_ok=True)
    config={'mode':mode,'source_sha256':info['dataset_expected_sha256'],
            'holdout_re_evaluated':False,'source_receipt_unchanged':True}
    # Isolated temp project root: run_research.py writes outputs under __file__.parent.
    with tempfile.TemporaryDirectory(prefix='pm25-dev-repro-') as tmp:
        work=Path(tmp)
        shutil.copytree(ROOT/'pm25aqi',work/'pm25aqi',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'global_air_pollution_dataset.csv',work/'global_air_pollution_dataset.csv')
        if mode in ('screening','all'):
            shutil.copy2(ROOT/'run_research.py',work/'run_research.py')
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            subprocess.run([sys.executable,'run_research.py'],cwd=work,env=env,check=True)
            shutil.copytree(work/'results',output/'screening',dirs_exist_ok=True)
        if mode in ('stacking','all'):
            from threadpoolctl import threadpool_limits
            from pm25aqi.research import load_screening_cohort
            from pm25aqi.batch import BASES, STACKERS
            from pm25aqi.stacking import compare_stackers_nested
            threadpool_limits(1)
            cohort=load_screening_cohort(work/'global_air_pollution_dataset.csv')
            folds,trace=compare_stackers_nested(cohort.X,cohort.y,BASES,
                                  methods=STACKERS,outer_folds=3,inner_folds=3)
            if any(t['oof_overlap'] for t in trace):raise AssertionError('OOF overlap in stacking')
            (output/'stacking').mkdir(exist_ok=True)
            folds.to_csv(output/'stacking/stacking_folds.csv',index=False)
            summary=folds.groupby('model',as_index=False).agg(mae=('mae','mean'),rmse=('rmse','mean'),r2=('r2','mean'))
            summary.to_csv(output/'stacking/stacking_summary.csv',index=False)
    (output/'run_summary.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    assert info==verify(require_data=True),'Archived release receipt changed during reproduction'
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['screening','stacking','all'],default='all')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(f'Development-only reproduction finished: {run(args.mode,args.output_dir)}')
