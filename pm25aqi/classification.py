"""Development-only direct-classification experiments with a sealed outer holdout.

Protocol: split the *development* partition into screening and confirmation groups;
select a method+feature set by grouped CV on SCREENING only; evaluate the frozen
choice ONCE on development confirmation; never accept the external holdout.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from .config import CATEGORIES, FEATURE_SETS, SEED
from .data import assert_group_disjoint, pattern_groups, raw_predictors
from .models import CLASSIFIERS, make_classifier
from .validation import _validate_folds, _validate_xy

LABELS = tuple(range(len(CATEGORIES)))


@dataclass(frozen=True)
class ClassifierChoice:
    model: str
    feature_set: str


def _validate_classes(X, y):
    _validate_xy(X, y)
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Class labels must be integer encodings, not target AQI values')
    if not y.isin(LABELS).all():
        raise ValueError('Invalid category code; expected 0..5')


def screening_confirmation_split(X_dev, *, confirmation_fraction=0.25, seed=SEED):
    """Feature-only development confirmation split; NO access to ANY class labels.

    Because this is group-disjoint, the row fraction can differ slightly from
    the requested group fraction. Stratification is used only INSIDE screening
    CV and cannot influence confirmation membership.
    """
    x = raw_predictors(X_dev)
    if not x.index.is_unique:
        raise ValueError('Unique row IDs required')
    if not 0 < confirmation_fraction < 1:
        raise ValueError('Confirmation fraction must be between zero and one')
    groups = pattern_groups(x)
    screen_pos, confirm_pos = next(GroupShuffleSplit(
        n_splits=1, test_size=confirmation_fraction, random_state=seed
    ).split(x, groups=groups))
    screen, confirm = x.index[screen_pos], x.index[confirm_pos]
    assert_group_disjoint(x.loc[screen], x.loc[confirm])
    if set(screen) & set(confirm) or len(screen) + len(confirm) != len(x):
        raise AssertionError('Bad development partition')
    return screen, confirm


def classifier_scores(y_true, y_pred):
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape or true.ndim != 1 or len(true) == 0:
        raise ValueError('Mismatched class labels')
    if not np.isin(true, LABELS).all() or not np.isin(pred, LABELS).all():
        raise ValueError('Invalid class code')
    recalls = precision_recall_fscore_support(true, pred, labels=LABELS, zero_division=0)[1]
    return {
        'macro_f1': float(f1_score(true, pred, labels=LABELS, average='macro',zero_division=0)),
        'weighted_f1': float(f1_score(true, pred, labels=LABELS, average='weighted',zero_division=0)),
        'balanced_accuracy': float(np.mean(recalls)),
        'accuracy': float(accuracy_score(true, pred)),
        'ordinal_mae': float(np.mean(np.abs(true - pred))),
        'quadratic_kappa': float(cohen_kappa_score(true, pred, labels=LABELS, weights='quadratic')),
    }


def classifier_screen(X_screen, y_screen, *, candidates, feature_sets,
                      n_splits=3, seed=SEED):
    """Compare ALL supplied model × feature specifications on the same CV folds.

    This returns only development-screening scores; no candidate is fitted on
    validation labels in the fold where that candidate is evaluated.
    """
    _validate_classes(X_screen, y_screen)
    candidates = tuple(candidates); feature_sets = tuple(feature_sets)
    if not candidates or not feature_sets or len(set(candidates)) != len(candidates) or len(set(feature_sets)) != len(feature_sets):
        raise ValueError('Nonempty, unique candidates and feature sets required')
    if set(candidates)-set(CLASSIFIERS) or set(feature_sets)-set(FEATURE_SETS):
        raise ValueError('Unknown model or feature set')
    groups = pattern_groups(X_screen)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X_screen, y_screen, groups=groups))
    _validate_folds(X_screen, folds)
    records=[]
    for model_name in candidates:
        for feature_set in feature_sets:
            for fold_idx,(train_idx,val_idx) in enumerate(folds,1):
                est = make_classifier(model_name,feature_set)
                est.fit(X_screen.iloc[train_idx],y_screen.iloc[train_idx])
                pred=est.predict(X_screen.iloc[val_idx])
                records.append({'model':model_name,'feature_set':feature_set,
                    'fold':fold_idx,'n_validation':len(val_idx),
                    **classifier_scores(y_screen.iloc[val_idx],pred)})
    return pd.DataFrame.from_records(records)


def summarize_classifier_screen(scores):
    """Rank only screening CV metrics. Do not use confirmation/holdout fields."""
    forbidden = ('test','holdout','confirm','external')
    if any(any(s in str(col).lower() for s in forbidden) for col in scores.columns):
        raise ValueError('Only screening-CV evidence may select classifiers')
    required = {'model','feature_set','fold','macro_f1','balanced_accuracy','ordinal_mae'}
    if scores.empty or not required.issubset(scores.columns):
        raise ValueError('Incomplete screening scores')
    outcome = (scores.groupby(['model','feature_set'],as_index=False)
        .agg(macro_f1=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
             balanced_accuracy=('balanced_accuracy','mean'),
             weighted_f1=('weighted_f1','mean'),accuracy=('accuracy','mean'),
             ordinal_mae=('ordinal_mae','mean'),quadratic_kappa=('quadratic_kappa','mean'),
             folds=('fold','nunique')))
    if not np.isfinite(outcome[['macro_f1','balanced_accuracy','ordinal_mae']].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite candidate scores')
    return outcome.sort_values(['macro_f1','balanced_accuracy','ordinal_mae','model','feature_set'],
                                ascending=[False,False,True,True,True],kind='stable').reset_index(drop=True)


def choose_classifier(screen_ranking):
    # Validates origin before selecting; confirmation labels are NEVER consulted.
    if any(any(s in str(col).lower() for s in ('test','holdout','confirm','external'))
           for col in screen_ranking.columns):
        raise ValueError('Confirmation/holdout score cannot determine model choice')
    if screen_ranking.empty or not {'model','feature_set','macro_f1','balanced_accuracy','ordinal_mae'}.issubset(screen_ranking.columns):
        raise ValueError('Need screening leaderboard')
    ordered=screen_ranking.sort_values(['macro_f1','balanced_accuracy','ordinal_mae','model','feature_set'],
                           ascending=[False,False,True,True,True],kind='stable')
    row=ordered.iloc[0]
    if not np.isfinite(row[['macro_f1','balanced_accuracy','ordinal_mae']].to_numpy(dtype=float)).all():
        raise ValueError('Invalid classifier score')
    return ClassifierChoice(str(row['model']),str(row['feature_set']))


def confirm_frozen_classifier(X_dev, y_dev, screening_ids, confirmation_ids, choice):
    """Evaluates ONE frozen candidate ONCE on development confirmation.

    The outer sealed holdout is not an argument. No auto-reselection occurs.
    """
    _validate_classes(X_dev,y_dev)
    screening_ids=pd.Index(screening_ids); confirmation_ids=pd.Index(confirmation_ids)
    if (not len(screening_ids) or not len(confirmation_ids) or
        not screening_ids.is_unique or not confirmation_ids.is_unique or
        set(screening_ids)&set(confirmation_ids) or
        set(screening_ids)|set(confirmation_ids)!=set(X_dev.index)):
        raise AssertionError('Confirmation boundary invalid')
    assert_group_disjoint(X_dev.loc[screening_ids],X_dev.loc[confirmation_ids])
    if choice.model not in CLASSIFIERS or choice.feature_set not in FEATURE_SETS:
        raise ValueError('Unknown frozen candidate')
    classifier = make_classifier(choice.model,choice.feature_set)
    classifier.fit(X_dev.loc[screening_ids],y_dev.loc[screening_ids])
    pred=np.asarray(classifier.predict(X_dev.loc[confirmation_ids]),dtype=int)
    true=y_dev.loc[confirmation_ids]
    cm=confusion_matrix(true,pred,labels=LABELS)
    precision,recall,f1,support=precision_recall_fscore_support(true,pred,labels=LABELS,zero_division=0)
    per_class=pd.DataFrame({'category':CATEGORIES, 'support':support.astype(int),
         'precision':precision, 'recall':recall,'f1':f1})
    predictions=pd.DataFrame({'true':true.to_numpy(dtype=int),'predicted':pred},index=confirmation_ids)
    return classifier, classifier_scores(true,pred),per_class,cm,predictions


def multiclass_calibration(estimator, X, y, *, bins=10):
    """Descriptive calibration of the SINGLE frozen confirmation model only."""
    raw_predictors(X)
    probs=np.asarray(estimator.predict_proba(X),dtype=float)
    mapping=list(estimator.classes_)
    full=np.zeros((len(X),len(LABELS)),dtype=float)
    for col,class_id in enumerate(mapping): full[:,int(class_id)]=probs[:,col]
    if not np.isfinite(full).all() or not np.allclose(full.sum(axis=1),1,atol=1e-6):
        raise ValueError('Invalid predicted probabilities')
    true=np.asarray(y,dtype=int)
    confidence=full.max(axis=1); guessed=full.argmax(axis=1)
    ece=0.
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        mask=(confidence>=lo)&(confidence<=hi if hi==1 else confidence<hi)
        if mask.any(): ece+=mask.mean()*abs((guessed[mask]==true[mask]).mean()-confidence[mask].mean())
    onehot=np.eye(len(LABELS))[true]
    return {'multiclass_brier':float(np.square(full-onehot).sum(axis=1).mean()),
            'ece_10bin':float(ece)}


def pattern_bootstrap_ci(predictions, X_confirmation, *, n_boot=300, seed=SEED):
    """95% *descriptive* cluster-bootstrap intervals, resample predictor patterns.

    These do not account for dataset selection, paper multiple-comparisons, or
    external deployment shift. In particular, they are NOT causal confidence.
    """
    if not predictions.index.equals(X_confirmation.index):
        raise ValueError('Index mismatch')
    groups=pattern_groups(X_confirmation).to_numpy()
    unique,inverse=np.unique(groups,return_inverse=True)
    lookup=[np.where(inverse==i)[0] for i in range(len(unique))]
    true=predictions.true.to_numpy(dtype=int);pred=predictions.predicted.to_numpy(dtype=int)
    rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(n_boot):
        which=rng.integers(0,len(unique),size=len(unique))
        rows=np.concatenate([lookup[i] for i in which])
        s=classifier_scores(true[rows],pred[rows])
        boot.append([s['macro_f1'],s['balanced_accuracy'],s['ordinal_mae']])
    estimates=np.quantile(np.asarray(boot),[.025,.975],axis=0)
    return pd.DataFrame({'metric':['macro_f1','balanced_accuracy','ordinal_mae'],
                         'lower_95':estimates[0], 'upper_95':estimates[1]})


def classifier_country_sensitivity(X_dev,y_dev, choice, *, n_splits=3):
    """Optional descriptive transfer study, NEVER used to select a candidate."""
    _validate_classes(X_dev,y_dev)
    from .data import group_folds
    folds=group_folds(X_dev,n_splits=n_splits,grouping='country')
    _validate_folds(X_dev,folds,grouping='country')
    rows=[]
    for fold,(tr,va) in enumerate(folds,1):
        est=make_classifier(choice.model,choice.feature_set).fit(X_dev.iloc[tr],y_dev.iloc[tr])
        s=classifier_scores(y_dev.iloc[va],est.predict(X_dev.iloc[va]))
        rows.append({'fold':fold,'n_validation':len(va),**s})
    return pd.DataFrame(rows)
