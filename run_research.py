"""One-command, screening-only scientific experiment & figures.

NO final test evaluation; NO reused v3 confirmation scoring. The 14,112
screening rows are all that flow through ML fitting and reporting functions.
"""
from pathlib import Path
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from pm25aqi.config import CATEGORIES, DATA_SHA256
from pm25aqi.research import (load_screening_cohort, evaluation_tables, summarize_models,
    oof_regression_to_category, uncertainty_and_error, cluster_paired_bootstrap,
    geographical_stress, seed_robustness, class_specific_cluster_intervals)
from pm25aqi.classification import classifier_scores
from pm25aqi.validation import development_permutation_importance

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
FIGURES=RESULTS/'figures'


def run_full(print_progress=True):
    threadpool_limits(1)
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    started=time.monotonic()
    def emit(msg):
        if print_progress: print(msg,flush=True)
    cohort=load_screening_cohort(ROOT/'global_air_pollution_dataset.csv')
    emit(f'SEALED: {cohort.n_outer_sealed} outer-holdout; {cohort.n_confirmation_sealed} v3 confirmation; '
         f'NEW exploratory screening CV: {len(cohort.X)} only.')
    reg_cv,cls_cv,reg_pred,cls_pred,prob = evaluation_tables(cohort,progress=emit)
    regression,classification=summarize_models(reg_cv,cls_cv)
    reg_cv.to_csv(RESULTS/'regression_folds.csv',index=False)
    cls_cv.to_csv(RESULTS/'classification_folds.csv',index=False)
    regression.to_csv(RESULTS/'regression_ranking.csv',index=False)
    classification.to_csv(RESULTS/'classification_ranking.csv',index=False)
    emit('Computing fixed OOF regression-derived class comparator')
    derived, numerics=oof_regression_to_category(cohort)
    derived_scores=classifier_scores(cohort.categories,derived)
    direct_scores=classifier_scores(cohort.categories,cls_pred['rf_balanced_A4'])
    descriptors=pd.DataFrame([{'method':'fixed_log_HGB_regression_to_AQI_class',**derived_scores},
                             {'method':'fixed_balanced_RF_direct_A4',**direct_scores}])
    descriptors.to_csv(RESULTS/'fixed_direct_vs_derived.csv',index=False)
    rare_intervals, direct_derived_intervals=class_specific_cluster_intervals(cohort,cls_pred,reg_pred)
    rare_intervals.to_csv(RESULTS/'rare_class_cluster_CIs.csv',index=False)
    direct_derived_intervals.to_csv(RESULTS/'direct_vs_regression_paired_CI.csv',index=False)
    classes,confusion,reliability,extra=uncertainty_and_error(cohort,reg_pred,cls_pred,prob)
    classes.to_csv(RESULTS/'class_specific_metrics.csv',index=False)
    pd.DataFrame(confusion,index=CATEGORIES,columns=CATEGORIES).to_csv(RESULTS/'confusion_matrix.csv')
    reliability.to_csv(RESULTS/'calibration_bins.csv',index=False)
    emit('Paired exact-input-pattern bootstrap (descriptive, not post-selection significance)')
    intervals=cluster_paired_bootstrap(cohort,reg_pred,cls_pred,n_boot=350)
    intervals.to_csv(RESULTS/'paired_cluster_bootstrap.csv',index=False)
    emit('Country-grouped fixed baselines')
    countries=geographical_stress(cohort)
    countries.to_csv(RESULTS/'country_transfer.csv',index=False)
    emit('Predeclared seed sensitivity')
    sensitivity=seed_robustness(cohort)
    sensitivity.to_csv(RESULTS/'seed_robustness.csv',index=False)
    emit('Fold-local input importance: fixed log-HGB')
    importance=development_permutation_importance(cohort.X,cohort.y,estimator_name='hgb_log_reference',repeats=3)
    importance.to_csv(RESULTS/'permutation_importance.csv',index=False)
    # Preserve OOF audit trail. One CSV per task, no protected labels included.
    reg_pred.assign(row_id=cohort.X.index).to_csv(RESULTS/'regression_OOF_screening.csv',index=False)
    cls_pred.assign(row_id=cohort.X.index).to_csv(RESULTS/'classification_OOF_screening.csv',index=False)
    # Plots all from screening diagnostics; never display a test leaderboard.
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    v=regression.iloc[:8].iloc[::-1]
    axes[0].barh(v.name,v.mae,color='#347c99');axes[0].set_xlabel('Grouped screening CV MAE (lower is better)');axes[0].set_title('Regression (screening only)')
    v=classification.iloc[:10].iloc[::-1]
    axes[1].barh(v.name,v.macro_f1,color='#bd782f');axes[1].set_xlabel('Grouped screening CV macro-F1 (higher is better)');axes[1].set_title('Classification (screening only)')
    fig.savefig(FIGURES/'models_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    import matplotlib.colors as colors
    axes[0].imshow(confusion,cmap='Blues')
    axes[0].set(xlabel='Predicted category',ylabel='Actual category',title='Fixed balanced RF A4 — 3-fold OOF')
    for i in range(6):
      for j in range(6):axes[0].text(j,i,str(confusion[i,j]),ha='center',va='center',fontsize=8)
    axes[0].set_xticks(range(6),[str(x) for x in range(6)])
    axes[0].set_yticks(range(6),[str(x) for x in range(6)])
    rel=reliability[reliability.n>0]
    axes[1].plot([0,1],[0,1],'--',color='grey',label='Ideal')
    axes[1].plot(rel.mean_confidence,rel.accuracy,'o-',color='#bd782f',label='Balanced RF A4')
    axes[1].set(xlabel='Mean confidence',ylabel='Observed accuracy',title='Multiclass confidence reliability',xlim=(0,1),ylim=(0,1));axes[1].legend()
    fig.savefig(FIGURES/'class_errors_and_calibration.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    geo=countries[countries.task=='regression']
    axes[0].bar(geo.fold.astype(str),geo.mae,color='#347c99');axes[0].set(xlabel='Held-out country-fold',ylabel='MAE',title='Country transfer: fixed log-HGB')
    vals=countries[countries.task=='classification']
    axes[1].bar(vals.fold.astype(str),vals.macro_f1,color='#bd782f');axes[1].set(xlabel='Held-out country-fold',ylabel='Macro-F1',title='Country transfer: fixed RF')
    fig.savefig(FIGURES/'geographical_transfer.png',dpi=170);plt.close(fig)
    # Feature-set comparisons use fixed model family, all estimates from same folds.
    fig,ax=plt.subplots(figsize=(8,4))
    a=classification[classification.method=='rf_balanced'].sort_values('features')
    ax.bar(a.features,a.macro_f1,color='#347c99');ax.set(ylim=(0,max(.7,float(a.macro_f1.max())+.08)),
        ylabel='Macro-F1 (screening CV)',xlabel='Input feature ablation',title='Balanced RF: raw gases, interactions, country, engineered')
    fig.savefig(FIGURES/'feature_ablation.png',dpi=170);plt.close(fig)
    env={'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
         'dataset_SHA256':DATA_SHA256,'data_records':23463,
         'screening_records':len(cohort.X),'sealed_confirmation_records':cohort.n_confirmation_sealed,
         'sealed_outer_records':cohort.n_outer_sealed,'grouped_cv_folds':3,'candidate_regression':len(regression),
         'candidate_classification':len(classification),'elapsed_seconds':round(time.monotonic()-started,2),
         'diagnostics':extra,'regression_best_screening':regression.iloc[0][['name','mae']].to_dict(),
         'classification_best_screening':classification.iloc[0][['name','macro_f1']].to_dict()}
    (RESULTS/'audit_manifest.json').write_text(json.dumps(env,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    emit(f"DONE: {env['elapsed_seconds']:.1f}s | exploratory best regression: {regression.iloc[0]['name']} MAE {regression.iloc[0]['mae']:.3f} | best classification {classification.iloc[0]['name']} F1 {classification.iloc[0]['macro_f1']:.3f}")
    return {'regression':regression,'classification':classification,'paired_bootstrap':intervals,
      'per_class':classes,'class_intervals':rare_intervals,'direct_derived_interval':direct_derived_intervals,'diagnostics':extra,'fixed_comparison':descriptors,'geographic':countries,'seed_sensitivity':sensitivity,'env':env}

if __name__=='__main__':
    # A release is immutable. A direct command in the published checkout would
    # overwrite the screening hashes frozen in LOCKED_PROTOCOL.json.
    if (ROOT/'LOCKED_PROTOCOL.json').is_file() and (ROOT/'results/batch/FINAL_HOLDOUT_RECEIPT.json').is_file():
        raise SystemExit('Frozen research repository: use python scripts/reproduce_development.py --mode screening instead. Archived rankings must not be overwritten.')
    run_full()
