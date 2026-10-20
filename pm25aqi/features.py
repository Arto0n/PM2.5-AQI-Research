"""Raw-predictor-only feature engineering; never uses target or proxy columns."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .config import NUMERIC
from .data import raw_predictors

class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Original v3 feature contract, retained exactly for baseline comparability."""
    def fit(self, X, y=None):
        raw_predictors(X)
        return self
    def transform(self, X):
        z=raw_predictors(X)
        z['CO_x_NO2']=z[NUMERIC[0]]*z[NUMERIC[2]]
        z['CO_x_Ozone']=z[NUMERIC[0]]*z[NUMERIC[1]]
        return z

class RichInteractionFeatures(InteractionFeatures):
    """Additional deterministic features from admitted inputs; only enabled for A5."""
    def transform(self, X):
        z=super().transform(X)
        for col in NUMERIC:
            z['log_' + col] = np.log1p(np.maximum(z[col], 0))
        z['gas_max'] = z[list(NUMERIC)].max(axis=1)
        z['gas_range'] = z[list(NUMERIC)].max(axis=1)-z[list(NUMERIC)].min(axis=1)
        z['gas_sum'] = z[list(NUMERIC)].sum(axis=1)
        for col in NUMERIC:
            z['fraction_' + col] = z[col]/(1+z['gas_sum'])
        return z
