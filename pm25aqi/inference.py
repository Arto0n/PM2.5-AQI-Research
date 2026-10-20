"""Inference validation and safe model-output formatting."""
import numpy as np
import pandas as pd
from .config import INPUTS, CATEGORIES
from .data import raw_predictors, categories_from_aqi


def predict_pm25(regressor, observations):
    X=raw_predictors(pd.DataFrame(observations).copy())
    for col in INPUTS[:3]:
        if X[col].isna().any() or not X[col].between(0,500).all():
            raise ValueError(f'Missing/out-of-range numeric input: {col}')
    pred=np.asarray(regressor.predict(X),dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError('Invalid model predictions')
    clipped=np.clip(pred,0,500)
    return pd.DataFrame({'PM2.5_AQI_estimate':clipped,
                         'predicted_category':categories_from_aqi(clipped)},index=X.index)
