"""SHAP explanations of the final LightGBM models.

SHAP values are computed by TreeExplainer on the model's raw output (log-odds, before
calibration). Calibration is monotonic, so it does not change which factors push a file
towards more or less risk; it only rescales the final probability.
"""
import numpy as np
import pandas as pd
import shap


def transform(pipeline, df, features):
    """Features as the LightGBM model sees them (preprocessed, sanitized names)."""
    return pipeline[:-1].transform(df[list(features)])


def readable_names(pipeline):
    """Preprocessed feature names before sanitizing (e.g. 'NAME_INCOME_TYPE_Working')."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def explain(pipeline, df, features):
    """SHAP values (log-odds) for the rows of df. Returns (values DataFrame, base value, X, raw margin)."""
    X = transform(pipeline, df, features)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):          # older SHAP: one array per class
        values = values[1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    margin = model.predict(X, raw_score=True)
    shap_df = pd.DataFrame(values, columns=readable_names(pipeline), index=df.index)
    return shap_df, base, X, margin


def additivity_error(shap_df, base, margin):
    """Max |base + sum of SHAP values - model raw output| over the rows (should be ~0)."""
    return float(np.max(np.abs(base + shap_df.sum(axis=1).to_numpy() - np.asarray(margin))))


def top_factors(shap_df, X, row, k=5):
    """The k features with the largest |SHAP| for one row: value, contribution and direction."""
    contrib = shap_df.loc[row]
    order = contrib.abs().sort_values(ascending=False).index[:k]
    values = pd.Series(X.loc[row].to_numpy(), index=shap_df.columns)
    return pd.DataFrame({
        "feature": order,
        "value (preprocessed)": [values[f] for f in order],
        "SHAP (log-odds)": [contrib[f] for f in order],
        "effect": ["raises the risk" if contrib[f] > 0 else "lowers the risk" for f in order],
    })


def global_importance(shap_df):
    """Mean |SHAP| per feature, sorted."""
    return shap_df.abs().mean().sort_values(ascending=False)
