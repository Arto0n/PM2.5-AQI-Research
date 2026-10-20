"""Batch all scientifically authorized stages, with one-shot final-holdout guard.

Python run_batch.py
    - Validate lock and data
    - Use archived, verified v4 grouped-screening model results
    - Run seven nested stacked methods + fixed uncertainty/geography/seed studies
    - Execute automated tests
    - Fit frozen models on development, evaluate 4,707 holdout records once
    - Produce article-ready figures and methodology/results reports

A delivered package with FINAL_HOLDOUT_RECEIPT.json *never* re-evaluates the
holdout. It checks the receipt, prints saved results and runs tests instead.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from time import monotonic
import pandas as pd
from threadpoolctl import threadpool_limits
from pm25aqi.batch import protocol_and_split,screening_extensions,evaluate_final_once
from render_batch import render

ROOT=Path(__file__).resolve().parent


def _test():
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests','-m','not slow'],cwd=ROOT,check=False)
    if result.returncode:
        raise RuntimeError(f'Leakage/test suite failed (exit {result.returncode}); refusing holdout evaluation')


def run_all():
    threadpool_limits(1)
    started=monotonic()
    if not (ROOT/'global_air_pollution_dataset.csv').is_file():
        raise FileNotFoundError('Original CSV missing. Run python scripts/download_data.py, or use scripts/verify_release.py without data.')
    lock,*_=protocol_and_split(ROOT)
    print(f"LOCKED: {lock['split_sizes']} | dataset SHA verified",flush=True)
    receipt_path=ROOT/'results'/'batch'/'FINAL_HOLDOUT_RECEIPT.json'
    stage=ROOT/'results'/'batch'
    screening_files=['stacking_development_nested_folds.csv','stacking_development_summary.csv',
      'screening_fixed_class_comparison.csv','screening_fixed_paired_CIs.csv',
      'country_transfer_fixed_models.csv','seed_robustness_fixed_models.csv']
    screen_manifest=stage/'SCREENING_STAGE_RECEIPT.json'
    if not receipt_path.exists():
        # The original v4 exploratory screening results are complete and
        # preserved, checksum-verified, and provide the locked candidate choices.
        if screen_manifest.exists():
            meta=json.loads(screen_manifest.read_text())
            for name,digest in meta['sha256'].items():
                if not (stage/name).is_file() or hashlib.sha256((stage/name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Screening receipt mismatch: '+name)
            print('SCREENING: Verified saved completed stage; skipping redundant expensive retraining',flush=True)
        else:
            if not all((stage/name).is_file() for name in screening_files):
                screening_extensions(ROOT,progress=lambda msg:print(msg,flush=True))
            # Stage marker only certifies deterministic development-only artifacts.
            for name in screening_files:
                if not (stage/name).is_file() or (stage/name).stat().st_size<30:
                    raise RuntimeError('Incomplete screening stage: '+name)
            screen_manifest.write_text(json.dumps({'sha256':{name:hashlib.sha256((stage/name).read_bytes()).hexdigest()
             for name in screening_files},'holdout_labels_used':False},indent=2)+'\n')
            print('SCREENING: completed results recorded; no repeated fitting',flush=True)
        _test()
        evaluate_final_once(ROOT,progress=lambda msg:print(msg,flush=True))
    else:
        evaluate_final_once(ROOT,progress=lambda msg:print(msg,flush=True))
        _test()
    report=render()
    r=pd.read_csv(ROOT/'results'/'batch'/'final_regression_fixed.csv')
    c=pd.read_csv(ROOT/'results'/'batch'/'final_classification_fixed.csv')
    print('\nFROZEN REGRESSION (one-shot internal holdout):\n',r.to_string(index=False),flush=True)
    print('\nFROZEN CLASSIFICATION (one-shot internal holdout):\n',c.to_string(index=False),flush=True)
    print(f'\nBatch completed in {monotonic()-started:.1f}s | report: {report.name}',flush=True)
    return {'regression':r,'classification':c,'report':report}

if __name__=='__main__':run_all()
