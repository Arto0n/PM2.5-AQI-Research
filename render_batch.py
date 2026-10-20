"""Aggregate-only publication tables, figures, and professor-facing reports.

This script NEVER fits or scores models and NEVER accesses holdout labels.
All final holdout figures are generated only from pre-registered summary files.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pm25aqi.config import CATEGORIES
from pm25aqi.batch import file_sha

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'/'batch'
FIG=RES/'figures'


def _plot():
    FIG.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(RES/'final_regression_fixed.csv')
    c=pd.read_csv(RES/'final_classification_fixed.csv')
    ci=pd.read_csv(RES/'final_absolute_pattern_CIs.csv')
    fig,ax=plt.subplots(1,2,figsize=(15,4.8),layout='constrained')
    order=['log_hgb_A4','log_hgb_A5','huber_stack']
    vals=r.set_index('model').loc[order]
    ax[0].barh(np.arange(len(order)),vals.mae,color=['#8a9ba8','#206288','#6d7592'])
    ax[0].set_yticks(range(len(order)),['HGB + A4 (reference)','HGB + A5 (primary)','Huber stack (comparator)'])
    ax[0].invert_yaxis();ax[0].set(xlabel='MAE (AQI points; lower is better)',title='One-shot outer holdout: regression',xlim=(0,20))
    for i,x in enumerate(vals.mae):ax[0].text(x+.1,i,f'{x:.2f}',va='center',fontsize=9)
    cl=['rf_balanced_A4','lightgbm_A4','hgb_A5_to_class']
    vals=c.set_index('model').loc[cl]
    ax[1].barh(np.arange(len(cl)),vals.macro_f1,color=['#8a9ba8','#206288','#6d7592'])
    ax[1].set_yticks(range(len(cl)),['Balanced RF (safety)','LightGBM (primary)','HGB A5 → category'])
    ax[1].invert_yaxis();ax[1].set(xlabel='Macro-F1 (higher is better)',title='One-shot outer holdout: classification',xlim=(0,max(.75,vals.macro_f1.max()+.12)))
    for i,x in enumerate(vals.macro_f1):ax[1].text(x+.01,i,f'{x:.3f}',va='center',fontsize=9)
    fig.savefig(FIG/'FINAL_fixed_models.png',dpi=190,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
    for ax,model,title in [(axes[0],'lightgbm_A4','Frozen LightGBM — direct classification'),
                            (axes[1],'rf_balanced_A4','Frozen balanced RF — safety comparator')]:
        cm=pd.read_csv(RES/f'final_confusion_{model}.csv',index_col=0).to_numpy(dtype=int)
        ax.imshow(cm,cmap='Blues')
        ax.set(title=title,xlabel='Predicted AQI category (0–5)',ylabel='True AQI category (0–5)')
        ax.set_xticks(range(6));ax.set_yticks(range(6))
        for i in range(6):
            for j in range(6):ax.text(j,i,str(cm[i,j]),va='center',ha='center',fontsize=8)
    fig.savefig(FIG/'FINAL_confusion_matrices.png',dpi=185,bbox_inches='tight');plt.close(fig)

    f=pd.read_csv(RES/'final_six_class_metrics.csv')
    fig,ax=plt.subplots(figsize=(12,4.6),layout='constrained')
    xx=np.arange(len(CATEGORIES)); w=.25
    for pos,model,label in [(-1,'lightgbm_A4','LightGBM'),(0,'rf_balanced_A4','Balanced RF'),(1,'hgb_A5_to_class','HGB-derived')]:
        sub=f[f.model==model].set_index('category').reindex(CATEGORIES)
        ax.bar(xx+pos*w,sub.recall.values,width=w,label=label)
    ax.set(ylim=(0,1),xticks=xx,xticklabels=['Good','Moderate','Sensitive','Unhealthy','Very unhealthy','Hazardous'],
           ylabel='Recall',title='Holdout recall by AQI category (descriptive)')
    ax.legend();fig.savefig(FIG/'FINAL_per_category_recall.png',dpi=185,bbox_inches='tight');plt.close(fig)

    cal=pd.read_csv(RES/'final_calibration_bins.csv')
    fig,ax=plt.subplots(figsize=(6.5,5.3),layout='constrained')
    ax.plot([0,1],[0,1],linestyle='--',color='grey',label='Perfect confidence calibration')
    valid=cal[cal.n>0]
    ax.scatter(valid.mean_confidence,valid.accuracy,s=25+3*valid.n,alpha=.8,label='Frozen LightGBM')
    ax.plot(valid.mean_confidence,valid.accuracy,alpha=.7)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted confidence',ylabel='Observed fraction correct',
           title='Holdout reliability (descriptive, uncalibrated)')
    ax.legend();fig.savefig(FIG/'FINAL_calibration.png',dpi=190,bbox_inches='tight');plt.close(fig)

    country=pd.read_csv(RES/'country_transfer_fixed_models.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,name,label,col in [(axes[0],'log_hgb_A5','mae','#206288'),
                               (axes[1],'lightgbm_A4','macro_f1','#8c7239')]:
        d=country[country.model==name]
        ax.bar(d.fold.astype(str),d[label],color=col)
        ax.set(title=f'Country-held-out CV: {name}',xlabel='Country-grouped fold',ylabel=label)
    fig.savefig(FIG/'DEVELOPMENT_country_transfer.png',dpi=190,bbox_inches='tight');plt.close(fig)

    stack=pd.read_csv(RES/'stacking_development_summary.csv').sort_values('mae')
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.barh(stack.model.iloc[::-1],stack.mae.iloc[::-1],color='#206288')
    ax.set(title='Seven nested stacking approaches (screening only)',xlabel='Mean outer-fold MAE')
    fig.savefig(FIG/'DEVELOPMENT_seven_stackers.png',dpi=185,bbox_inches='tight');plt.close(fig)


def _report():
    lock=json.loads((ROOT/'LOCKED_PROTOCOL.json').read_text(encoding='utf-8'))
    receipt=json.loads((RES/'FINAL_HOLDOUT_RECEIPT.json').read_text(encoding='utf-8'))
    r=pd.read_csv(RES/'final_regression_fixed.csv').set_index('model')
    c=pd.read_csv(RES/'final_classification_fixed.csv').set_index('model')
    bp=pd.read_csv(RES/'final_paired_pattern_CIs.csv')
    ci=pd.read_csv(RES/'final_absolute_pattern_CIs.csv')
    cl=pd.read_csv(RES/'final_six_class_metrics.csv')
    stack=pd.read_csv(RES/'stacking_development_summary.csv').sort_values('mae')
    geo=pd.read_csv(RES/'country_transfer_fixed_models.csv')
    seed=pd.read_csv(RES/'seed_robustness_fixed_models.csv')
    boundary=pd.read_csv(RES/'final_boundary_errors.csv')
    extra=json.loads((RES/'final_supplementary.json').read_text(encoding='utf-8'))
    p=ROOT/'SCIENTIFIC_BATCH_RESULTS.md'
    def tab(df,dec=4):return df.round(dec).to_markdown(index=False)
    pair=bp.copy()
    reg=r.reset_index();cls=c.reset_index()
    doc=f'''# PM2.5 AQI: prespecified batch validation and publication audit

## Scope and status

This is a *fixed-candidate single-pass outer-holdout evaluation*, not a newly tuned
all-model test leaderboard. Exact dataset fingerprint: `{lock['data_sha256']}`.

The locked protocol (`LOCKED_PROTOCOL.json`, SHA-256 `{receipt['protocol_sha256']}`)
preceded all new holdout-label access. It freezes the model families, feature sets,
primary metrics, safety contrasts, data partition and training plan. Do not use
these results to revise the candidates or modify the feature sets and then rescore
this same holdout. An evaluation receipt prevents automatic repetition.

**Critical provenance caveat:** the original historical random-row test was
already inspected in earlier research and the v3 confirmation labels were also
inspected. This new 4,707-row outer split is pattern-disjoint, but it comes from
the same dataset and can share original rows or countries with those earlier
inspected partitions. Its results are **not independent external validation**.

## Dataset, protocol and model-selection contract

- Dataset: **23,463** records; three gaseous pollutant AQIs and `Country` permitted.
- Outer holdout: **4,707** exact-pattern-disjoint rows; development: **18,756**.
- Development: screening **14,112** (candidate selection and all exploratory CV);
  prior v3 confirmation **4,644** (previously inspected, not used for v4 selection).
- Final fit: **all 18,756 development rows** after freezing candidate choices.
- Primary regression: **HGB on log target with A5 features**, primary metric MAE.
  Prespecified comparators: same HGB+A4, and Huber stacked ridge/ExtraTrees/HGB+A4.
- Primary direct classification: **LightGBM+A4**, primary metric six-class Macro-F1.
  Safety comparator: **balanced RF+A4**, including Hazardous recall;
  additional fixed comparator: HGB+A5 numerical prediction mapped to AQI categories.
- 95% intervals: input-pattern cluster bootstrap, **descriptive conditional intervals**;
  they do not account for hyperparameter screening, geographic clustering, the
  earlier historical test examination, or model-fitting uncertainty.
- No model is selected based on holdout outcomes, irrespective of which comparator wins.

## Table 1 — one-shot fixed regression holdout

{tab(reg[['model','n','mae','rmse','r2']])}

## Table 2 — one-shot fixed classification holdout

{tab(cls[['model','n','macro_f1','weighted_f1','accuracy','balanced_accuracy','ordinal_mae','quadratic_kappa']])}

## Table 3 — preregistered paired holdout comparisons

{tab(pair)}

**Reading paired intervals:** Regression contrasts are MAE(a)-MAE(b), so
negative is in favor of A. Macro-F1 and recall contrasts are A minus B,
so positive favors A. These intervals are exploratory/descriptive, not
proof of generalization superiority or independent hypothesis tests.

## Table 4 — absolute holdout metrics with conditional pattern-bootstrap intervals

{tab(ci)}

## Table 5 — six-class classification performance

{tab(cl[['model','category','support','precision','recall','f1']])}

## Table 6 — model performance near AQI category boundaries

{tab(boundary)}

## Table 7 — seven nested stacking methods on screening only

{tab(stack[['model','mae','rmse','r2']])}

These stacking scores were obtained on screening observations only.
Huber stacking was designated a fixed secondary comparator *before* test access;
its newer screening result is not allowed to change final primary-model selection.

## Table 8 — predeclared country-held-out development checks

{tab(geo)}

## Table 9 — predeclared seed checks (screening only)

{tab(seed)}

## Calibration, class rarity and geographic interpretation

- LightGBM 10-bin ECE on outer holdout: **{extra['ece_10_bin_lightgbm_A4']:.4f}**.
- LightGBM multiclass Brier on outer holdout: **{extra['multiclass_brier_lightgbm_A4']:.4f}**.
- Outer holdout class counts: {extra['class_counts']}.
- Severe categories have few examples; uncertainty is high. In addition,
  input-pattern cluster bootstrap does not capture dependence among countries.
- The model estimates category/AQI cross-sectionally. **This dataset lacks
  timestamps and station/meteorological covariates and cannot justify time-series
  forecasting or sensor-station external generalization claims.**
- Do not promote the best holdout comparator to a new selected model. If the
  primary model loses on a secondary metric, report that result unchanged.

## Paper-ready figure files

- `results/batch/figures/FINAL_fixed_models.png`
- `results/batch/figures/FINAL_confusion_matrices.png`
- `results/batch/figures/FINAL_per_category_recall.png`
- `results/batch/figures/FINAL_calibration.png`
- `results/batch/figures/DEVELOPMENT_country_transfer.png`
- `results/batch/figures/DEVELOPMENT_seven_stackers.png`

## Recommended reporting and reviewer considerations

1. Explain why exact-pattern-disjoint splitting was chosen, distinguish it from
   country-grouped validation and from the historical random-row protocol.
2. Give ALL six class supports and classwise precision/recall/F1; avoid citing
   accuracy alone for rare high-risk AQI labels.
3. Clearly state candidate-selection optimism: v4's 10 regression and 12
   classification candidates were searched on screening; paired screening CIs
   are conditional, not confirmatory.
4. The original paper's already-inspected test cannot be rehabilitated by
   splitting the same dataset again. Independent external monitoring data
   would still be needed for strong deployment/generalization claims.
5. Discuss that AQI thresholds are tied to numeric PM2.5 AQI; direct class
   and regression-derived predictions serve related but different objectives.
6. Share software/library versions and dataset fingerprint, and call the
   locked holdout pass a precommitted **internal** evaluation rather than
   claiming a never-before-seen external cohort.
7. Do not claim causal effects, future forecasting, or intervention policies.
8. Treat numerical model ranking on holdout as description of FIXED models,
   not as a basis for another tuning cycle.

## Reproduction safeguard

The package contains `LOCKED_PROTOCOL.json` and `results/batch/FINAL_HOLDOUT_RECEIPT.json`.
Calling the default batch runner again verifies these files and **skips holdout scoring**.
Test predictions and individual holdout labels are not exported for post-test tuning.
All processing other than the final fixed scoring used screening or frozen development rows.
'''
    p.write_text(doc,encoding='utf-8')
    fa=f'''# خلاصه برای استاد — نسخهٔ Batch نهایی PM2.5 AQI

## تصمیم‌های علمی

- پروتکل ارزیابی قبل از دسترسی به برچسب‌های Holdout ثبت و قفل شد.
- **مدل اصلی رگرسیون:** HGB+A5 (معیار اصلی MAE).
- **مدل اصلی طبقه‌بندی:** LightGBM+A4 (معیار اصلی Macro-F1).
- **مدل‌های مقایسه‌ای ثابت:** HGB+A4، Huber Stacking، Balanced RF+A4 و طبقه‌بندی حاصل از رگرسیون HGB+A5.
- **آزمایش‌ها:** هفت روش Stacking با Nested Group CV، حساسیت به Seed، مقایسهٔ جفتی و بازه‌های اطمینان، تحلیل کلاس‌ها، Calibration، ارزیابی کشورهای دیده‌نشده و خطاهای مرزی AQI.

## نتایج Holdout جدید (ارزیابی یک‌باره)

**رگرسیون:**

{tab(reg[['model','mae','rmse','r2']])}

**طبقه‌بندی:**

{tab(cls[['model','macro_f1','accuracy','balanced_accuracy','ordinal_mae']])}

**فاصلهٔ اطمینان اختلاف مدل‌ها:**

{tab(pair)}

## نتیجه‌گیری و محدودیت مهم

- داده‌های ورودی شامل CO AQI، Ozone AQI، NO2 AQI و Country هستند؛ PM2.5 و Overall AQI هرگز به‌عنوان ویژگی استفاده نشده‌اند.
- نتیجهٔ Holdout برای تنظیم دوبارهٔ مدل یا تغییر مدل منتخب استفاده نشده است.
- دادهٔ Confirmation نسخه v3 و تست تاریخی قبلاً بررسی شده بودند؛ بنابراین این Holdout **دادهٔ خارجیِ کاملاً مستقل نیست**.
- نتیجهٔ دقیق و بی‌طرفانهٔ مدل‌های ثابت را باید گزارش کرد، حتی اگر مدل مقایسه‌ای در یک معیار بهتر باشد.
- مقاله باید روش‌شناسی تقسیم گروهی و محدودیت تعمیم به کشورهای جدید و کلاس‌های کم‌نمونه را توضیح دهد.

**پیشنهاد برای تصمیم استاد:** مقاله با تمرکز بر اعتبارسنجی مقاوم‌تر، مقایسهٔ جامع‌تر و گزارش شفاف محدودیت‌ها بازبینی شود؛ ادعای «برتری قطعی مدل جدید» فقط در صورت پشتیبانی مستقل قابل طرح است.

برای تمام نتایج و نمودارها فایل `SCIENTIFIC_BATCH_RESULTS.md` را ببینید.
'''
    (ROOT/'PROFESSOR_SUMMARY_FA.md').write_text(fa,encoding='utf-8')


def render():
    if not (RES/'FINAL_HOLDOUT_RECEIPT.json').exists():
        raise RuntimeError('Cannot render final holdout results before locked one-shot evaluation')
    _plot();_report()
    return ROOT/'SCIENTIFIC_BATCH_RESULTS.md'

if __name__=='__main__':
    print(render())
