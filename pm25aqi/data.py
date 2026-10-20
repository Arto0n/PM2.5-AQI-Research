"""Immutable data handling, safe feature-only splits and column contracts."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from .config import INPUTS, NUMERIC, TARGET, CLASS_TARGET, CATEGORIES, DATA_SHA256, SEED


def categories_from_aqi(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 500)).any():
        raise ValueError('Numeric AQI must be finite in [0,500]')
    return np.asarray(pd.cut(values, [-np.inf,50,99,149,200,299,np.inf], labels=CATEGORIES).astype(str))


def load_data(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError('Data fingerprint changed; do not silently alter the benchmark dataset')
    with path.open(encoding='utf-8-sig', newline='') as fp:
        headers = next(csv.reader(fp))
    if len(headers) != len(set(headers)):
        raise ValueError('Duplicated CSV header')
    df = pd.read_csv(path)
    required = set(INPUTS) | {TARGET, CLASS_TARGET}
    if required - set(df.columns):
        raise ValueError(f'Missing: {sorted(required-set(df.columns))}')
    for col in (*NUMERIC, TARGET):
        x = pd.to_numeric(df[col], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(x).all() or ((x < 0) | (x > 500)).any():
            raise ValueError(f'Invalid AQI column: {col}')
    if not np.array_equal(categories_from_aqi(df[TARGET]), df[CLASS_TARGET].to_numpy()):
        raise ValueError('Target and class labels disagree')
    if not df.index.is_unique:
        raise ValueError('Row IDs must be unique')
    return df


def raw_predictors(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Use DataFrame with named raw predictor columns')
    missing = set(INPUTS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing predictor fields: {sorted(missing)}')
    # Strict whitelist; no overall AQI, PM2.5 target, City, or category proxy.
    x = frame.loc[:, list(INPUTS)].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors='raise')
        if np.isinf(x[col].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite feature values in {col}')
    return x


def pattern_groups(raw_x):
    x = raw_predictors(raw_x)
    x['Country'] = x['Country'].fillna('__MISSING__').astype(str)
    # Row-index independent group id, covers all admitted predictors.
    return pd.util.hash_pandas_object(x, index=False).astype('uint64')


def prospective_split(raw_x, test_fraction=0.20, seed=SEED):
    """Partition generated ONLY from raw features. Holdout y never accessed."""
    x = raw_predictors(raw_x)
    if not x.index.is_unique:
        raise ValueError('Unique original row index is required')
    groups = pattern_groups(x)
    a, b = next(GroupShuffleSplit(test_size=test_fraction, random_state=seed).split(x, groups=groups))
    train_ids, hold_ids = x.index[a], x.index[b]
    assert_group_disjoint(x.loc[train_ids], x.loc[hold_ids])
    if len(set(train_ids) & set(hold_ids)) or len(a) + len(b) != len(x):
        raise AssertionError('Bad train/holdout partition')
    return train_ids, hold_ids


def assert_group_disjoint(left, right):
    if set(pattern_groups(left).to_numpy()) & set(pattern_groups(right).to_numpy()):
        raise AssertionError('The same input pattern appears on both sides')


def group_folds(raw_x, n_splits=3, seed=SEED, *, grouping='pattern'):
    x = raw_predictors(raw_x)
    if grouping == 'pattern':
        groups = pattern_groups(x)
    elif grouping == 'country':
        groups = x['Country'].fillna('__MISSING__').astype(str)
    else:
        raise ValueError('grouping must be pattern or country')
    if groups.nunique() < n_splits:
        raise ValueError('Not enough distinct groups')
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for train, val in splitter.split(x, groups=groups):
        if set(groups.iloc[train]) & set(groups.iloc[val]):
            raise AssertionError('Grouped CV leakage')
        result.append((np.asarray(train), np.asarray(val)))
    return result


def development_only(df, development_ids):
    # Deliberately no holdout argument in the API.
    d = df.loc[development_ids]
    x = raw_predictors(d)
    y = d[TARGET].copy()
    classes = pd.Categorical(d[CLASS_TARGET], categories=CATEGORIES, ordered=True).codes
    if (classes < 0).any():
        raise ValueError('Unknown class')
    return x, y, pd.Series(classes, index=x.index, name='class_code')
