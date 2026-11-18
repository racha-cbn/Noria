"""Probability calibration (experiment E8).

The calibrators are fitted on the OUT-OF-FOLD predictions of the TRAIN set
(5-fold stratified CV, same parameters and number of trees as the final model),
never on the validation set. They are evaluated on the validation set.
"""
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src import config
from src import train as tr

EPS = 1e-6


def final_model_path(list_name):
    return tr.MODELS_DIR / f"final__lgbm__{list_name}.joblib"


def load_final_model(list_name, train=None):
    """Final session 4 model (tuned parameters). Retrained and saved if the file is missing."""
    path = final_model_path(list_name)
    if path.exists():
        return joblib.load(path)
    pipeline, _ = tr.train_model(config.FEATURE_SETS[list_name], "lgbm", train,
                                 params=tr.best_params(list_name))
    tr.save_model(pipeline, f"final__lgbm__{list_name}")
    return pipeline


def oof_predictions(train, features, params, n_estimators):
    """Out-of-fold predicted probabilities on TRAIN (each row predicted by a model that never saw it)."""
    X, y = train[features], train[config.TARGET]
    oof = np.zeros(len(train))
    folds = StratifiedKFold(n_splits=tr.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    for fit_idx, pred_idx in folds.split(X, y):
        pipe = tr.build_pipeline(X.iloc[fit_idx], features, "lgbm", n_estimators, params)
        pipe.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[pred_idx] = pipe.predict_proba(X.iloc[pred_idx])[:, 1]
    return oof


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Platt scaling: logistic regression on the log-odds of the raw score."""

    def fit(self, p_raw, y):
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw), np.asarray(y))
        return self

    def predict(self, p_raw):
        return self.model_.predict_proba(_logit(p_raw))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression of the outcome on the raw score (monotonic, piecewise constant)."""

    def fit(self, p_raw, y):
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p_raw), np.asarray(y))
        return self

    def predict(self, p_raw):
        return self.model_.predict(np.asarray(p_raw))


CALIBRATORS = {"isotonic": IsotonicCalibrator, "platt": PlattCalibrator}


class CalibratedModel:
    """A fitted pipeline + a calibrator: predict_proba returns calibrated probabilities of default."""

    def __init__(self, pipeline, features, calibrator, method):
        self.pipeline, self.features, self.calibrator, self.method = pipeline, list(features), calibrator, method

    def predict_raw(self, df):
        return self.pipeline.predict_proba(df[self.features])[:, 1]

    def predict_proba(self, df):
        return self.calibrator.predict(self.predict_raw(df))


def save_calibrated(model, list_name):
    tr.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = tr.MODELS_DIR / f"calibrated__lgbm__{list_name}.joblib"
    joblib.dump(model, path)
    return path
