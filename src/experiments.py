"""Experiments E1-E3 and E10-E13 of docs/experiments.md, on the VALIDATION set only.

The decision rules below are copied from docs/experiments.md and must not be
changed after seeing the results (see the rules at the top of that file).
"""
from src import config, data
from src import evaluate as ev
from src import train as tr
from src import tuning

# --- Decision rules, as pre-registered in docs/experiments.md ----------------
E3_KEEP_THRESHOLD = 0.65     # >= 0.65: kept
E3_CAUTION_THRESHOLD = 0.60  # 0.60-0.65: kept with a stronger caution message; < 0.60: dropped

# --- Sanity checks requested for session 3 (not decision rules) -------------
LEAK_AUC_THRESHOLD = 0.85        # no-history model above this: suspect a leak
DOMINANT_FEATURE_SHARE = 0.50    # one feature above this share of the total gain: investigate
OVERFIT_GAP_WARNING = 0.05       # train AUC - validation AUC above this: flag overfitting

# Feature lists used by the experiments (the E3 client list is frozen for reproducibility).
# Session 3 lists are frozen (before the session 4 removal of PHONE_CHANGE_YEARS) so that
# notebook 03 reproduces the recorded results of E1-E3, E10 and E11.
SETS = {
    "client": config.CLIENT_FEATURES_E3,
    "bank_no_history": config.BANK_NO_HISTORY_FEATURES_S3,
    "bank_with_history": config.BANK_WITH_HISTORY_FEATURES_S3,
    "client_no_pricing": [f for f in config.CLIENT_FEATURES_E3 if f not in config.PRICING_FEATURES],
    "bank_no_history_no_pricing": [f for f in config.BANK_NO_HISTORY_FEATURES_S3
                                   if f not in config.PRICING_FEATURES],
}

MODELS = {
    # key: (feature set name in SETS, model type)
    "logreg__bank_no_history": ("bank_no_history", "logreg"),
    "lgbm__bank_no_history": ("bank_no_history", "lgbm"),
    "lgbm__bank_with_history": ("bank_with_history", "lgbm"),
    "lgbm__client": ("client", "lgbm"),
    "logreg__client": ("client", "logreg"),  # indicative only (E3)
    "lgbm__client_no_pricing": ("client_no_pricing", "lgbm"),                    # E10
    "lgbm__bank_no_history_no_pricing": ("bank_no_history_no_pricing", "lgbm"),  # E11
}
E1_E3_MODELS = ["logreg__bank_no_history", "lgbm__bank_no_history", "lgbm__bank_with_history",
                "lgbm__client", "logreg__client"]
E10_E11_MODELS = ["lgbm__client_no_pricing", "lgbm__bank_no_history_no_pricing"]


def train_and_evaluate(train, val, keys=E1_E3_MODELS):
    """Train each model on TRAIN, evaluate on VAL, save model (.joblib) and metrics (.json)."""
    results = {}
    for key in keys:
        set_name, model_type = MODELS[key]
        features = SETS[set_name]
        pipeline, info = tr.train_model(features, model_type, train)
        p_val = ev.predict_proba(pipeline, val, features)
        metrics = ev.evaluate(val[config.TARGET], p_val)
        metrics.update(feature_set=set_name, train_auc=info["train_auc"],
                       train_minus_val_auc=info["train_auc"] - metrics["roc_auc"], training=info)
        if model_type == "lgbm":
            gain = tr.feature_importance(pipeline)
            metrics["top_feature"] = gain.index[0]
            metrics["top_feature_gain_share"] = float(gain.iloc[0] / gain.sum())
        tr.save_model(pipeline, key)
        ev.save_metrics(metrics, key)
        results[key] = {"pipeline": pipeline, "p_val": p_val, "metrics": metrics, "features": features}
    return results


def run_e1(results, y_val):
    """E1 - Baseline: logistic regression vs LightGBM on BANK_NO_HISTORY_FEATURES."""
    lr, gbm = results["logreg__bank_no_history"], results["lgbm__bank_no_history"]
    diff = ev.paired_bootstrap_auc_diff(y_val, gbm["p_val"], lr["p_val"])
    beats = gbm["metrics"]["roc_auc"] > lr["metrics"]["roc_auc"]
    out = {
        "experiment": "E1",
        "logreg_auc": lr["metrics"]["roc_auc"], "lgbm_auc": gbm["metrics"]["roc_auc"],
        "lgbm_minus_logreg": diff,
        "rule": "LightGBM must beat the logistic regression in validation AUC; otherwise, investigate",
        "rule_satisfied": bool(beats),
        "decision": ("LightGBM beats the logistic regression: LightGBM stays the main model"
                     if beats else "LightGBM does not beat the logistic regression: investigate before going further"),
    }
    ev.save_metrics(out, "e1")
    return out


def run_e2(results, y_val):
    """E2 - Cost of the history: bank with history vs bank without history (paired bootstrap)."""
    with_h, no_h = results["lgbm__bank_with_history"], results["lgbm__bank_no_history"]
    diff = ev.paired_bootstrap_auc_diff(y_val, with_h["p_val"], no_h["p_val"])
    out = {
        "experiment": "E2",
        "with_history_auc": with_h["metrics"]["roc_auc"], "no_history_auc": no_h["metrics"]["roc_auc"],
        "with_minus_without": diff,
        "rule": "None: measurement to report",
        "decision": "Measurement reported (no decision rule)",
    }
    ev.save_metrics(out, "e2")
    return out


def run_e3(results):
    """E3 - Client model viability: thresholds applied to the LightGBM client model's validation AUC."""
    auc = results["lgbm__client"]["metrics"]["roc_auc"]
    if auc >= E3_KEEP_THRESHOLD:
        decision = "kept"
    elif auc >= E3_CAUTION_THRESHOLD:
        decision = "kept with a stronger caution message"
    else:
        decision = "dropped"
    out = {
        "experiment": "E3",
        "client_lgbm_auc": auc,
        "client_lgbm_ci": [results["lgbm__client"]["metrics"]["ci_low"],
                           results["lgbm__client"]["metrics"]["ci_high"]],
        "client_logreg_auc_indicative": results["logreg__client"]["metrics"]["roc_auc"],
        "rule": ">= 0.65 kept; 0.60-0.65 kept with a stronger caution message; < 0.60 dropped",
        "decision": decision,
    }
    ev.save_metrics(out, "e3")
    return out


def run_e10(results):
    """E10 - Client model without the pricing signal: same thresholds as E3 (pre-registered)."""
    m = results["lgbm__client_no_pricing"]["metrics"]
    auc = m["roc_auc"]
    if auc >= E3_KEEP_THRESHOLD:
        decision = "kept"
    elif auc >= E3_CAUTION_THRESHOLD:
        decision = "kept with a stronger caution message"
    else:
        decision = "dropped"
    out = {
        "experiment": "E10",
        "features": results["lgbm__client_no_pricing"]["features"],
        "client_no_pricing_auc": auc, "ci": [m["ci_low"], m["ci_high"]], "pr_auc": m["pr_auc"],
        "client_e3_auc_for_reference": results["lgbm__client"]["metrics"]["roc_auc"],
        "rule": ">= 0.65 kept; 0.60-0.65 kept with a stronger caution message; < 0.60 dropped (same as E3)",
        "decision": decision,
    }
    ev.save_metrics(out, "e10")
    return out


def run_e11(results, y_val):
    """E11 - Bank model without history, with vs without the pricing signal (paired bootstrap)."""
    with_p, without_p = results["lgbm__bank_no_history"], results["lgbm__bank_no_history_no_pricing"]
    diff = ev.paired_bootstrap_auc_diff(y_val, with_p["p_val"], without_p["p_val"])
    out = {
        "experiment": "E11",
        "with_pricing_auc": with_p["metrics"]["roc_auc"], "without_pricing_auc": without_p["metrics"]["roc_auc"],
        "with_minus_without": diff,
        "rule": "None: measurement to report",
        "decision": "Measurement reported (no decision rule)",
    }
    ev.save_metrics(out, "e11")
    return out


def run_e10_e11(run):
    """Train the two E10/E11 models with the same protocol and run E10 and E11.

    `run` is the output of run_all(): E11 reuses its LightGBM bank-without-history model.
    """
    new = train_and_evaluate(run["train"], run["val"], keys=E10_E11_MODELS)
    run["results"].update(new)
    y_val = run["val"][config.TARGET].to_numpy()
    run["e10"], run["e11"] = run_e10(run["results"]), run_e11(run["results"], y_val)
    run["sanity_e10_e11"] = sanity_checks(new)
    return run


def sanity_checks(results):
    """Leak and overfitting checks; returns one row per model with flags."""
    rows = []
    for key, r in results.items():
        m = r["metrics"]
        row = {"model": key, "train_auc": m["train_auc"],
               "cv_auc_on_train_folds": m["training"].get("cv_mean_auc"), "val_auc": m["roc_auc"],
               "train_minus_val": m["train_minus_val_auc"],
               "overfit_flag": m["train_minus_val_auc"] > OVERFIT_GAP_WARNING,
               "top_feature": m.get("top_feature"), "top_feature_gain_share": m.get("top_feature_gain_share"),
               "dominant_feature_flag": (m.get("top_feature_gain_share") or 0) > DOMINANT_FEATURE_SHARE,
               "leak_auc_flag": m["feature_set"] != "bank_with_history" and m["roc_auc"] > LEAK_AUC_THRESHOLD}
        rows.append(row)
    return rows


def run_all():
    """Train the five models and run E1, E2, E3 (train + validation only, never the test set)."""
    train, val = data.load_train_val()
    results = train_and_evaluate(train, val)
    y_val = val[config.TARGET].to_numpy()
    return {"train": train, "val": val, "results": results,
            "e1": run_e1(results, y_val), "e2": run_e2(results, y_val), "e3": run_e3(results),
            "sanity": sanity_checks(results)}


# --- Session 4: E12 (new features) and E13 (hyperparameter tuning) ----------

E12_LISTS = {
    # name: (reference list without new features, candidates added)
    "bank_no_history": (config.BANK_NO_HISTORY_FEATURES_PRE_E12, config.E12_BANK_CANDIDATES),
    "bank_with_history": (config.BANK_WITH_HISTORY_FEATURES_PRE_E12, config.E12_BANK_CANDIDATES),
    "client": (config.CLIENT_FEATURES_PRE_E12, config.E12_CLIENT_CANDIDATES),
}


def _fit_eval(train, val, features, name, params=None, folds=None):
    """Train a LightGBM on TRAIN (CV early stopping), evaluate on VAL, save its metrics."""
    pipeline, info = tr.train_model(features, "lgbm", train, params=params, folds=folds)
    p_val = ev.predict_proba(pipeline, val, features)
    metrics = ev.evaluate(val[config.TARGET], p_val)
    gain = tr.feature_importance(pipeline)
    metrics.update(features=list(features), n_features=len(features),
                   cv_mean_auc=info["cv_mean_auc"], train_auc=info["train_auc"],
                   train_minus_val_auc=info["train_auc"] - metrics["roc_auc"],
                   top_feature=gain.index[0], top_feature_gain_share=float(gain.iloc[0] / gain.sum()),
                   training=info)
    ev.save_metrics(metrics, name)
    return {"pipeline": pipeline, "p_val": p_val, "metrics": metrics, "features": list(features)}


def run_e12(train, val):
    """E12: LightGBM with vs without the new features, per list; pre-registered rule applied as written."""
    y_val = val[config.TARGET].to_numpy()
    out = {"experiment": "E12", "rule": ("kept if the mean 5-fold CV AUC on train increases AND the lower "
                                         "bound of the paired bootstrap CI on validation is > 0"),
           "lists": {}}
    models = {}
    for name, (reference, candidates) in E12_LISTS.items():
        base = _fit_eval(train, val, reference, f"e12_lgbm__{name}__reference")
        with_new = _fit_eval(train, val, list(reference) + list(candidates), f"e12_lgbm__{name}__with_new")
        diff = ev.paired_bootstrap_auc_diff(y_val, with_new["p_val"], base["p_val"])
        cv_gain = with_new["metrics"]["cv_mean_auc"] - base["metrics"]["cv_mean_auc"]
        keep = bool(cv_gain > 0 and diff["ci_low"] > 0)
        out["lists"][name] = {
            "candidates": list(candidates),
            "reference": {"n_features": len(reference), "cv_mean_auc": base["metrics"]["cv_mean_auc"],
                          "val_auc": base["metrics"]["roc_auc"],
                          "val_ci": [base["metrics"]["ci_low"], base["metrics"]["ci_high"]]},
            "with_new": {"n_features": len(reference) + len(candidates),
                         "cv_mean_auc": with_new["metrics"]["cv_mean_auc"],
                         "val_auc": with_new["metrics"]["roc_auc"],
                         "val_ci": [with_new["metrics"]["ci_low"], with_new["metrics"]["ci_high"]]},
            "cv_auc_gain": cv_gain,
            "val_auc_diff": diff,
            "decision": "kept" if keep else "dropped",
            "final_features": list(reference) + list(candidates) if keep else list(reference),
        }
        models[name] = {"reference": base, "with_new": with_new, "final": with_new if keep else base}
    ev.save_metrics(out, "e12")
    return out, models


def run_e13(train, val, e12_models, n_trials=tuning.N_TRIALS):
    """E13: Optuna tuning on the final E12 lists; tuned kept if validation AUC >= default."""
    y_val = val[config.TARGET].to_numpy()
    out = {"experiment": "E13", "n_trials": n_trials,
           "rule": "tuned parameters kept if their validation AUC is >= that of the default parameters",
           "lists": {}}
    models, studies = {}, {}
    for name, m in e12_models.items():
        default = m["final"]
        features = default["features"]
        best_params, study, folds = tuning.tune(train, features, n_trials=n_trials)
        tuned = _fit_eval(train, val, features, f"e13_lgbm__{name}__tuned", params=best_params, folds=folds)
        diff = ev.paired_bootstrap_auc_diff(y_val, tuned["p_val"], default["p_val"])
        keep = bool(tuned["metrics"]["roc_auc"] >= default["metrics"]["roc_auc"])
        ev.save_metrics({"list": name, "best_params": best_params, "best_cv_auc": study.best_value,
                         "best_trial": study.best_trial.number}, f"e13_best_params__{name}")
        tuning.trials_table(study).to_csv(ev.METRICS_DIR / f"e13_trials__{name}.csv", index=False)
        out["lists"][name] = {
            "n_features": len(features),
            "default": {"cv_mean_auc": default["metrics"]["cv_mean_auc"], "val_auc": default["metrics"]["roc_auc"],
                        "val_ci": [default["metrics"]["ci_low"], default["metrics"]["ci_high"]],
                        "n_estimators": default["metrics"]["training"]["n_estimators"]},
            "tuned": {"best_cv_auc_optuna": study.best_value, "cv_mean_auc": tuned["metrics"]["cv_mean_auc"],
                      "val_auc": tuned["metrics"]["roc_auc"],
                      "val_ci": [tuned["metrics"]["ci_low"], tuned["metrics"]["ci_high"]],
                      "n_estimators": tuned["metrics"]["training"]["n_estimators"], "params": best_params},
            "val_auc_diff_tuned_minus_default": diff,
            "decision": "tuned parameters kept" if keep else "default parameters kept",
        }
        final = tuned if keep else default
        tr.save_model(final["pipeline"], f"final__lgbm__{name}")
        models[name] = {"default": default, "tuned": tuned, "final": final}
        studies[name] = study
    # E2 re-measured with the two tuned bank models (and with the retained ones if different)
    e2 = {"experiment": "E2 updated after E13",
          "tuned_models": ev.paired_bootstrap_auc_diff(
              y_val, models["bank_with_history"]["tuned"]["p_val"], models["bank_no_history"]["tuned"]["p_val"]),
          "tuned_with_history_auc": models["bank_with_history"]["tuned"]["metrics"]["roc_auc"],
          "tuned_no_history_auc": models["bank_no_history"]["tuned"]["metrics"]["roc_auc"]}
    if any(models[n]["final"] is not models[n]["tuned"] for n in ("bank_with_history", "bank_no_history")):
        e2["retained_models"] = ev.paired_bootstrap_auc_diff(
            y_val, models["bank_with_history"]["final"]["p_val"], models["bank_no_history"]["final"]["p_val"])
    out["e2_updated"] = e2
    ev.save_metrics(e2, "e2_after_e13")
    ev.save_metrics(out, "e13")
    return out, models, studies


def session3_predictions(train, val, keys=("lgbm__bank_no_history", "lgbm__bank_with_history",
                                           "lgbm__client_no_pricing")):
    """Validation predictions of the session 3 models (before the PHONE_CHANGE_YEARS removal).

    Loads models/<key>.joblib when present, otherwise retrains it with the frozen session 3
    list. Nothing is saved: the recorded session 3 metrics stay untouched.
    """
    import joblib
    preds = {}
    for key in keys:
        set_name, model_type = MODELS[key]
        features = SETS[set_name]
        path = tr.MODELS_DIR / f"{key}.joblib"
        pipeline = joblib.load(path) if path.exists() else tr.train_model(features, model_type, train)[0]
        preds[key] = ev.predict_proba(pipeline, val, features)
    return preds
