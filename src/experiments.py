"""Experiments E1-E3 of docs/experiments.md (session 3), on the VALIDATION set only.

The decision rules below are copied from docs/experiments.md and must not be
changed after seeing the results (see the rules at the top of that file).
"""
from src import config, data
from src import evaluate as ev
from src import train as tr

# --- Decision rules, as pre-registered in docs/experiments.md ----------------
E3_KEEP_THRESHOLD = 0.65     # >= 0.65: kept
E3_CAUTION_THRESHOLD = 0.60  # 0.60-0.65: kept with a stronger caution message; < 0.60: dropped

# --- Sanity checks requested for session 3 (not decision rules) -------------
LEAK_AUC_THRESHOLD = 0.85        # no-history model above this: suspect a leak
DOMINANT_FEATURE_SHARE = 0.50    # one feature above this share of the total gain: investigate
OVERFIT_GAP_WARNING = 0.05       # train AUC - validation AUC above this: flag overfitting

MODELS = {
    # key: (feature set name, model type)
    "logreg__bank_no_history": ("bank_no_history", "logreg"),
    "lgbm__bank_no_history": ("bank_no_history", "lgbm"),
    "lgbm__bank_with_history": ("bank_with_history", "lgbm"),
    "lgbm__client": ("client", "lgbm"),
    "logreg__client": ("client", "logreg"),  # indicative only (E3)
}


def train_and_evaluate(train, val, keys=MODELS):
    """Train each model on TRAIN, evaluate on VAL, save model (.joblib) and metrics (.json)."""
    results = {}
    for key in keys:
        set_name, model_type = MODELS[key]
        features = config.FEATURE_SETS[set_name]
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
