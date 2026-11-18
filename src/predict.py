"""Loading and prediction for the API: one file per model in artifacts/.

    from src.predict import predict
    result = predict("client", {"AGE_YEARS": 26, "NAME_FAMILY_STATUS": "Single / not married", ...})

Returns the calibrated probability of default, the risk class (with its label) and the
SHAP contributions (log-odds of the raw model, before calibration).
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import explain, risk
from src.export import artifact_path, metadata_path
from src.final import sha256

TOP_K = 5


@lru_cache(maxsize=None)
def load(list_name):
    """(model, metadata) for "bank_no_history" or "client"; the SHA-256 is checked against the metadata."""
    meta = json.loads(metadata_path(list_name).read_text(encoding="utf-8"))
    path = artifact_path(list_name)
    if sha256(path) != meta["sha256"]:
        raise RuntimeError(f"{path.name}: SHA-256 does not match its metadata")
    return joblib.load(path), meta


def to_frame(record, meta):
    """One-row DataFrame with the expected features and types. Missing keys become missing values."""
    spec = {f["name"]: f for f in meta["features"]}
    unknown = set(record) - set(spec)
    if unknown:
        raise ValueError(f"unexpected features: {sorted(unknown)}")
    row = {}
    for name, f in spec.items():
        value = record.get(name)
        if f["type"] == "categorical":
            row[name] = None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
        else:
            row[name] = np.nan if value is None else float(value)
    df = pd.DataFrame([row], columns=list(spec))
    for name, f in spec.items():
        df[name] = df[name].astype(object if f["type"] == "categorical" else float)
    return df


def predict(list_name, record, top_k=TOP_K):
    """Calibrated probability, class, label and SHAP contributions for one applicant."""
    model, meta = load(list_name)
    df = to_frame(record, meta)
    p = float(model.predict_proba(df)[0])
    cls = str(risk.assign_class([p])[0])
    shap_df, base, X, margin = explain.explain(model.pipeline, df, model.features)
    contrib = shap_df.iloc[0]
    top = contrib.abs().sort_values(ascending=False).index[:top_k]
    return {
        "model": meta["name"], "version": meta["version"],
        "probability": p, "class": cls, "label": meta["classes"]["labels"][cls],
        "raw_log_odds": float(margin[0]), "shap_base_value": base,
        "shap_contributions": {k: float(v) for k, v in contrib.items()},
        "top_factors": [{"feature": f, "shap": float(contrib[f]),
                         "effect": "raises the risk" if contrib[f] > 0 else "lowers the risk"} for f in top],
    }
