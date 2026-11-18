"""Model training (session 3).

Every model is a scikit-learn Pipeline: preprocessing (src/preprocessing.py)
+ estimator, fitted on the TRAIN set only. The validation set is never used
to fit or to choose anything here; the test set is never loaded.
"""
import re

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src import config, data
from src.preprocessing import build_preprocessor

MODELS_DIR = config.ROOT / "models"

LOGREG_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": config.RANDOM_STATE,
}

# Reasonable defaults, no hyperparameter search in this session.
LGBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
CV_FOLDS = 5
MAX_TREES = 5000
EARLY_STOPPING_ROUNDS = 100


def sanitize_feature_names(X):
    """Rename columns to [A-Za-z0-9_] (LightGBM rejects names such as 'Industry: type 1')."""
    X = X.copy()
    X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_") for c in X.columns]
    if X.columns.duplicated().any():
        raise ValueError("feature names collide after sanitizing")
    return X


def _make_estimator(model_type, n_estimators=None, params=None):
    """params: LightGBM parameters (defaults to LGBM_PARAMS); ignored for logreg."""
    if model_type == "logreg":
        return LogisticRegression(**LOGREG_PARAMS)
    if model_type == "lgbm":
        return lgb.LGBMClassifier(n_estimators=n_estimators or MAX_TREES, **(params or LGBM_PARAMS))
    raise ValueError(f"unknown model_type: {model_type!r}")


def build_pipeline(X_train, features, model_type, n_estimators=None, params=None):
    """Unfitted pipeline: preprocessing (+ scaling for logreg only) -> estimator."""
    return Pipeline([
        ("preprocess", build_preprocessor(X_train, features, scale=(model_type == "logreg"))),
        ("names", FunctionTransformer(sanitize_feature_names)),
        ("model", _make_estimator(model_type, n_estimators, params)),
    ])


def prepare_cv_folds(train, features):
    """The 5 stratified CV folds of the TRAIN set, already preprocessed.

    In each fold the preprocessing is fitted on the fold's training part only.
    Returns a list of (X_fit, y_fit, X_stop, y_stop); reusable across parameter sets.
    """
    X, y = train[features], train[config.TARGET]
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    prepared = []
    for fit_idx, stop_idx in folds.split(X, y):
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_stop, y_stop = X.iloc[stop_idx], y.iloc[stop_idx]
        prep = Pipeline([
            ("preprocess", build_preprocessor(X_fit, features, scale=False)),
            ("names", FunctionTransformer(sanitize_feature_names)),
        ]).fit(X_fit)
        prepared.append((prep.transform(X_fit), y_fit, prep.transform(X_stop), y_stop))
    return prepared


def cv_early_stopping(folds, params=None):
    """Fit LightGBM on each prepared fold with early stopping on the fold's held-out AUC.

    Returns per-fold details: best iteration and held-out AUC at that iteration.
    """
    details = []
    for k, (X_fit, y_fit, X_stop, y_stop) in enumerate(folds, start=1):
        model = _make_estimator("lgbm", params=params).fit(
            X_fit, y_fit,
            eval_X=(X_stop,), eval_y=(y_stop,),
            eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        details.append({"fold": k, "best_iteration": int(model.best_iteration_),
                        "fold_auc": float(model.best_score_["valid_0"]["auc"])})
    return details


def select_n_estimators(train, features, params=None, folds=None):
    """Number of trees by early stopping in a 5-fold stratified CV on the TRAIN set.

    Returns (n_estimators = mean best iteration over the folds, per-fold details).
    """
    details = cv_early_stopping(folds if folds is not None else prepare_cv_folds(train, features), params)
    n_estimators = int(round(np.mean([d["best_iteration"] for d in details])))
    return n_estimators, details


def train_model(feature_list, model_type, train=None, params=None, folds=None):
    """Train `model_type` ("logreg" or "lgbm") on `feature_list`, on the TRAIN set only.

    lgbm: the number of trees comes from select_n_estimators (CV on train),
    then the model is refitted on the whole train set with that number.
    params: LightGBM parameters (default LGBM_PARAMS); folds: optional prepared CV folds.
    Returns (fitted pipeline, info dict).
    """
    if train is None:
        train = data.load_split("train")
    features = list(feature_list)
    info = {"model_type": model_type, "n_features": len(features), "train_rows": len(train)}
    n_estimators = None
    if model_type == "lgbm":
        n_estimators, cv = select_n_estimators(train, features, params, folds)
        info.update(n_estimators=n_estimators, cv_folds=cv,
                    cv_mean_auc=float(np.mean([d["fold_auc"] for d in cv])),
                    params=dict(params or LGBM_PARAMS))
    pipeline = build_pipeline(train, features, model_type, n_estimators, params)
    pipeline.fit(train[features], train[config.TARGET])
    info["train_auc"] = float(roc_auc_score(
        train[config.TARGET], pipeline.predict_proba(train[features])[:, 1]))
    return pipeline, info


def feature_importance(pipeline, importance_type="gain"):
    """LightGBM importances, indexed by the preprocessed feature names (before sanitizing)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    booster = pipeline.named_steps["model"].booster_
    values = booster.feature_importance(importance_type=importance_type)
    return pd.Series(values, index=names, name=importance_type).sort_values(ascending=False)


def save_model(pipeline, name):
    """Save a fitted pipeline to models/<name>.joblib (models/ is not versioned)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return path


def best_params(list_name):
    """LightGBM parameters retained for a feature list after E13 (tuned if kept, else defaults)."""
    import json
    metrics_dir = config.ROOT / "reports" / "metrics"
    e13 = json.loads((metrics_dir / "e13.json").read_text(encoding="utf-8"))
    if e13["lists"][list_name]["decision"] == "tuned parameters kept":
        return json.loads((metrics_dir / f"e13_best_params__{list_name}.json").read_text(encoding="utf-8"))["best_params"]
    return dict(LGBM_PARAMS)
