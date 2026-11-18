"""Session 5 experiments (docs/experiments.md): E8 calibration, E14 risk classes, E4 region
features, E5 no-history clients, E6 target population, E9 fairness. VALIDATION set only.

Models: the final session 4 models (52, 65 and 7 features, tuned parameters).
AUCs are computed on the raw model scores (calibration is monotonic; isotonic steps would only
add ties). Classes and fairness use the calibrated probabilities.
"""
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import analysis, calibration, config, risk
from src import evaluate as ev
from src import train as tr

LISTS = ["bank_no_history", "bank_with_history", "client"]


def final_models(train, val):
    """Final pipelines and their raw validation scores."""
    out = {}
    for name in LISTS:
        pipe = calibration.load_final_model(name, train)
        features = config.FEATURE_SETS[name]
        out[name] = {"pipeline": pipe, "features": features,
                     "p_raw": ev.predict_proba(pipe, val, features)}
    return out


def run_e8(train, val, models):
    """E8: isotonic vs Platt fitted on out-of-fold TRAIN predictions; Brier on validation."""
    y_tr, y_val = train[config.TARGET].to_numpy(), val[config.TARGET].to_numpy()
    out = {"experiment": "E8", "rule": "keep the method with the best Brier score on validation; "
           "calibration adopted only if the Brier improvement over the uncalibrated model has a CI lower bound > 0",
           "lists": {}}
    for name, m in models.items():
        n_trees = m["pipeline"].named_steps["model"].n_estimators
        oof = calibration.oof_predictions(train, m["features"], tr.best_params(name), n_trees)
        probs = {"uncalibrated": m["p_raw"]}
        calibrators = {}
        for method, cls in calibration.CALIBRATORS.items():
            calibrators[method] = cls().fit(oof, y_tr)
            probs[method] = calibrators[method].predict(m["p_raw"])
        brier = {k: float(brier_score_loss(y_val, p)) for k, p in probs.items()}
        chosen = min(["isotonic", "platt"], key=lambda k: brier[k])
        improvement = ev.paired_bootstrap_metric_diff(y_val, probs["uncalibrated"], probs[chosen], brier_score_loss)
        other = "platt" if chosen == "isotonic" else "isotonic"
        vs_other = ev.paired_bootstrap_metric_diff(y_val, probs[other], probs[chosen], brier_score_loss)
        adopted = improvement["ci_low"] > 0
        cal_model = calibration.CalibratedModel(m["pipeline"], m["features"], calibrators[chosen], chosen)
        calibration.save_calibrated(cal_model, name)
        m.update(probs=probs, chosen=chosen, p_cal=probs[chosen], calibrated_model=cal_model, oof=oof)
        out["lists"][name] = {
            "n_trees": int(n_trees), "oof_auc_train": float(roc_auc_score(y_tr, oof)),
            "brier": brier,
            "mean_predicted": {k: float(np.mean(p)) for k, p in probs.items()},
            "observed_default_rate": float(y_val.mean()),
            "chosen_method": chosen,
            "brier_uncalibrated_minus_chosen": improvement,
            f"brier_{other}_minus_{chosen}": vs_other,
            "calibration_adopted": bool(adopted),
        }
    ev.save_metrics(out, "e8")
    return out


def run_e14(val, models):
    """E14: risk classes on the calibrated probabilities, bank without history and client."""
    y_val = val[config.TARGET].to_numpy()
    out = {"experiment": "E14", "thresholds": {"low": f"< {risk.LOW_THRESHOLD}", "medium":
           f"{risk.LOW_THRESHOLD}-{risk.HIGH_THRESHOLD}", "high": f">= {risk.HIGH_THRESHOLD}"}, "lists": {}}
    tables = {}
    for name, labels in [("bank_no_history", risk.BANK_RECOMMENDATION), ("client", risk.CLIENT_CATEGORY)]:
        table, checks = risk.class_table(y_val, models[name]["p_cal"], labels)
        tables[name] = table
        out["lists"][name] = {"method": models[name]["chosen"],
                              "classes": table.reset_index().to_dict(orient="records"), "checks": checks}
    ev.save_metrics(out, "e14")
    return out, tables


def run_e4(train, val, models):
    """E4: bank without history with vs without REGION_FEATURES, same tuned parameters."""
    y_val = val[config.TARGET].to_numpy()
    features = list(config.BANK_NO_HISTORY_FEATURES) + list(config.REGION_FEATURES)
    pipe, info = tr.train_model(features, "lgbm", train, params=tr.best_params("bank_no_history"))
    p_region = ev.predict_proba(pipe, val, features)
    base = models["bank_no_history"]["p_raw"]
    out = {"experiment": "E4", "n_features_with_region": len(features), "n_trees_with_region": info["n_estimators"],
           "without_region": ev.bootstrap_auc(y_val, base), "with_region": ev.bootstrap_auc(y_val, p_region),
           "with_minus_without": ev.paired_bootstrap_auc_diff(y_val, p_region, base),
           "rule": "not used in Noria whatever the result; gain reported as a lead for a real bank",
           "decision": "REGION_FEATURES not used in Noria"}
    ev.save_metrics(out, "e4")
    return out, pipe


def run_e5(val, models):
    """E5: validation clients absent from bureau.csv vs all validation clients."""
    y_val = val[config.TARGET].to_numpy()
    ids = analysis.bureau_ids()
    mask = analysis.no_history_mask(val, ids)
    out = {"experiment": "E5", "set": "validation",
           "subgroup_rows": int(mask.sum()), "subgroup_share": float(mask.mean()),
           "subgroup_default_rate": float(y_val[mask].mean()), "others_default_rate": float(y_val[~mask].mean()),
           "models": {}}
    for name, m in models.items():
        out["models"][name] = {"all": ev.bootstrap_auc(y_val, m["p_raw"]),
                               "no_bureau_record": ev.bootstrap_auc(y_val[mask], m["p_raw"][mask]),
                               "with_bureau_record": ev.bootstrap_auc(y_val[~mask], m["p_raw"][~mask])}
    ev.save_metrics(out, "e5")
    return out, mask


def run_e6(val, models):
    """E6: AUC by NAME_INCOME_TYPE and by age group (+ 20-35), bank without history and client."""
    y_val = val[config.TARGET].to_numpy()
    ages = val["AGE_YEARS"].to_numpy()
    extra = {"20-35 (aggregate)": (ages >= 20) & (ages < 35)}
    out, tables = {"experiment": "E6", "min_class_count": analysis.MIN_CLASS_COUNT, "lists": {}}, {}
    for name in ["bank_no_history", "client"]:
        p = models[name]["p_raw"]
        by_income = analysis.group_auc(y_val, p, val["NAME_INCOME_TYPE"].astype(str).to_numpy())
        by_age = analysis.group_auc(y_val, p, analysis.age_group(val["AGE_YEARS"]).astype(str).to_numpy(), extra)
        tables[name] = {"income_type": by_income, "age": by_age}
        out["lists"][name] = {"overall_auc": float(roc_auc_score(y_val, p)),
                              "income_type": by_income.reset_index().to_dict(orient="records"),
                              "age": by_age.reset_index().to_dict(orient="records")}
    ev.save_metrics(out, "e6")
    return out, tables


def run_e9(val, models):
    """E9: acceptance / FNR / FPR per CODE_GENDER and age group, on the calibrated E14 classes."""
    y_val = val[config.TARGET].to_numpy()
    sensitive = {"CODE_GENDER": val[config.FAIRNESS_COL].astype(str).to_numpy(),
                 "age_group": analysis.age_group(val["AGE_YEARS"]).astype(str).to_numpy()}
    out, frames = {"experiment": "E9", "acceptance": "class 'low' (< 5%)", "lists": {}}, {}
    for name in ["bank_no_history", "client"]:
        out["lists"][name], frames[name] = {}, {}
        for attr, groups in sensitive.items():
            by_group, overall, gaps = analysis.fairness_frame(y_val, models[name]["p_cal"], groups)
            frames[name][attr] = by_group
            out["lists"][name][attr] = {"by_group": by_group.reset_index().to_dict(orient="records"),
                                        "overall": {k: float(v) for k, v in overall.items()},
                                        "max_gap_between_groups": gaps}
    ev.save_metrics(out, "e9")
    return out, frames
