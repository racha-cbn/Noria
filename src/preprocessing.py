"""Preprocessing that learns from the data (step D of session 2).

Anti-leakage rule: the transformer returned by `build_preprocessor` must be
fitted on the TRAIN set only, then used to transform val and test.
Never call fit / fit_transform on val or test.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src import config


class PercentileCapper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to its [lower, upper] percentiles learned at fit time (NaN ignored)."""

    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lower_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def split_column_types(X, features):
    """Split `features` into (categorical, binary, continuous) using the dtypes of X.

    Binary = numeric columns whose non-missing values are all 0/1 (flags).
    Flags are not capped: on a rare flag the 1st/99th percentiles are both 0,
    so capping would erase the information.
    """
    categorical, binary, continuous = [], [], []
    for col in features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            categorical.append(col)
        elif X[col].dropna().isin([0, 1]).all():
            binary.append(col)
        else:
            continuous.append(col)
    return categorical, binary, continuous


def _to_object(X):
    """Cast string columns to object so the imputer and encoder treat NaN consistently."""
    return pd.DataFrame(X).astype(object).where(pd.DataFrame(X).notna(), np.nan)


def build_preprocessor(X_train, features, scale=False, cap_percentiles=(1.0, 99.0)):
    """ColumnTransformer for `features`, configured from the TRAIN dataframe dtypes.

    - continuous numeric: percentile capping -> median imputation -> StandardScaler (if scale)
    - binary flags: median imputation (-> StandardScaler if scale)
    - categorical: most-frequent imputation -> OneHotEncoder(handle_unknown="ignore")
    - categorical in config.MISSING_CATEGORY_COLUMNS: missing values become their own
      category (config.MISSING_LABEL) -> OneHotEncoder(handle_unknown="ignore")

    scale=True is meant for the logistic regression only (trees do not need it).
    The returned object is NOT fitted: call .fit(X_train[features]) yourself.
    """
    categorical, binary, continuous = split_column_types(X_train, features)
    missing_category = [c for c in categorical if c in config.MISSING_CATEGORY_COLUMNS]
    categorical = [c for c in categorical if c not in missing_category]
    low, high = cap_percentiles

    continuous_steps = [
        ("cap", PercentileCapper(lower=low, upper=high)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    binary_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        continuous_steps.append(("scale", StandardScaler()))
        binary_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if continuous:
        transformers.append(("continuous", Pipeline(continuous_steps), continuous))
    if binary:
        transformers.append(("binary", Pipeline(binary_steps), binary))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical))
    if missing_category:
        missing_steps = [
            ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=config.MISSING_LABEL)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("categorical_missing", Pipeline(missing_steps), missing_category))

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor.set_output(transform="pandas")
