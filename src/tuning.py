"""Hyperparameter tuning with Optuna (experiment E13).

Objective: mean AUC of a 5-fold stratified CV on the TRAIN set, with early
stopping in each fold. The validation set is never used here.
"""
import numpy as np
import optuna

from src import config
from src import train as tr

N_TRIALS = 50
SAMPLER_SEED = config.RANDOM_STATE

# Fixed parameters (not tuned).
FIXED_PARAMS = {
    "subsample_freq": 1,
    "class_weight": "balanced",
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


def suggest_params(trial):
    """Search space pre-registered in docs/experiments.md (E13)."""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    params.update(FIXED_PARAMS)
    return params


def sqlite_storage(name):
    """Persistent Optuna storage in models/ (not versioned), so an interrupted study can resume."""
    path = tr.MODELS_DIR / "optuna"
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path / (name + '.db')}"


def tune(train, features, n_trials=N_TRIALS, seed=SAMPLER_SEED, storage=None, study_name="noria_e13"):
    """Run the Optuna study on `features`. Returns (best params, study, prepared folds).

    storage: optional Optuna storage URL (e.g. sqlite_storage("e14")). With a storage, an
    existing study of the same name is resumed and only the missing trials are run.
    Long runs should use a storage and be launched under `caffeinate -ims`.
    E13 was run without storage (in memory).
    """
    folds = tr.prepare_cv_folds(train, features)  # preprocessing fitted once per fold, reused

    def objective(trial):
        details = tr.cv_early_stopping(folds, suggest_params(trial))
        trial.set_user_attr("mean_best_iteration", float(np.mean([d["best_iteration"] for d in details])))
        return float(np.mean([d["fold_auc"] for d in details]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name=study_name, storage=storage, load_if_exists=storage is not None)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_trials - done > 0:
        study.optimize(objective, n_trials=n_trials - done)
    best = dict(study.best_params)
    best.update(FIXED_PARAMS)
    return best, study, folds


def trials_table(study):
    """One row per trial: number, CV AUC, mean best iteration and the sampled parameters."""
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return df.rename(columns={"value": "cv_auc"})
