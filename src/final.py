"""Final evaluation on the test set (experiment E16) and model freeze.

The final models are frozen in reports/final/manifest.json (SHA-256 of each file).
The evaluation uses them as they are: no retraining, no tuning, no threshold change.
`evaluate_split` computes exactly the E16 measures; it is run on the validation set
and on the test set with the same code, so that every figure can be compared.
"""
import hashlib
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import analysis, config, risk
from src import evaluate as ev
from src import train as tr

FINAL_DIR = config.ROOT / "reports" / "final"
MANIFEST = FINAL_DIR / "manifest.json"
LISTS = ["bank_no_history", "bank_with_history", "client"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(list_name):
    return tr.MODELS_DIR / f"calibrated__lgbm__{list_name}.joblib"


def build_manifest():
    """Freeze the final calibrated models (pipeline + Platt calibrator) and the data splits."""
    models = {}
    for name in LISTS:
        path = model_path(name)
        m = joblib.load(path)
        models[name] = {
            "file": str(path.relative_to(config.ROOT)),
            "sha256": sha256(path),
            "features": m.features,
            "n_features": len(m.features),
            "lgbm_params": tr.best_params(name),
            "n_estimators": int(m.pipeline.named_steps["model"].n_estimators),
            "calibration": m.method,
            "class_thresholds": {"low": f"p < {risk.LOW_THRESHOLD}",
                                 "medium": f"{risk.LOW_THRESHOLD} <= p < {risk.HIGH_THRESHOLD}",
                                 "high": f"p >= {risk.HIGH_THRESHOLD}"},
        }
    splits = {name: {"file": f"data/processed/{name}.parquet",
                     "sha256": sha256(config.PROCESSED_DIR / f"{name}.parquet")}
              for name in ("train", "val", "test")}
    manifest = {"frozen_on": date.today().isoformat(), "experiment": "E16",
                "note": "These files must not change. The test set is used once, with these models as they are.",
                "models": models, "data_splits": splits}
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest():
    """Check every frozen file against its SHA-256. Raises if anything changed."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = {}
    for name, m in manifest["models"].items():
        checks[m["file"]] = sha256(config.ROOT / m["file"]) == m["sha256"]
    for name, s in manifest["data_splits"].items():
        checks[s["file"]] = sha256(config.ROOT / s["file"]) == s["sha256"]
    if not all(checks.values()):
        raise RuntimeError(f"Frozen files changed: {[k for k, ok in checks.items() if not ok]}")
    return manifest, checks


def load_frozen_models():
    """Load the frozen calibrated models, after verifying their SHA-256."""
    manifest, _ = verify_manifest()
    return {name: joblib.load(config.ROOT / m["file"]) for name, m in manifest["models"].items()}


def evaluate_split(df, models, ids_without_bureau):
    """All E16 measures on one split (validation or test). AUC on raw scores, Brier/classes on calibrated."""
    y = df[config.TARGET].to_numpy()
    out = {"rows": int(len(df)), "defaults": int(y.sum()), "default_rate": float(y.mean()), "models": {}}
    preds = {}
    for name, m in models.items():
        p_raw, p_cal = m.predict_raw(df), m.predict_proba(df)
        preds[name] = {"raw": p_raw, "cal": p_cal}
        frac, mean_p = calibration_curve(y, p_cal, n_bins=10, strategy="quantile")
        out["models"][name] = {
            "roc_auc": ev.bootstrap_auc(y, p_raw),
            "pr_auc": float(average_precision_score(y, p_raw)),
            "brier_calibrated": float(brier_score_loss(y, p_cal)),
            "mean_calibrated_p": float(p_cal.mean()),
            "calibration_curve": {"mean_predicted": mean_p.tolist(), "observed": frac.tolist()},
        }
    # E2: cost of the history (paired bootstrap)
    out["e2_with_minus_without"] = ev.paired_bootstrap_auc_diff(
        y, preds["bank_with_history"]["raw"], preds["bank_no_history"]["raw"])
    # E14: classes
    out["e14"] = {}
    for name, labels in [("bank_no_history", risk.BANK_RECOMMENDATION), ("client", risk.CLIENT_CATEGORY)]:
        table, checks = risk.class_table(y, preds[name]["cal"], labels)
        out["e14"][name] = {"classes": table.reset_index().to_dict(orient="records"), "checks": checks}
    # E5: clients absent from bureau.csv
    mask = analysis.no_history_mask(df, ids_without_bureau)
    out["e5"] = {"subgroup_rows": int(mask.sum()), "subgroup_share": float(mask.mean()),
                 "subgroup_default_rate": float(y[mask].mean()), "others_default_rate": float(y[~mask].mean()),
                 "models": {name: {"all": ev.bootstrap_auc(y, preds[name]["raw"]),
                                   "no_bureau_record": ev.bootstrap_auc(y[mask], preds[name]["raw"][mask])}
                            for name in models}}
    # E6: AUC by income type and age group
    ages = df["AGE_YEARS"].to_numpy()
    extra = {"20-35 (aggregate)": (ages >= 20) & (ages < 35)}
    out["e6"] = {}
    for name in ["bank_no_history", "client"]:
        p = preds[name]["raw"]
        out["e6"][name] = {
            "income_type": analysis.group_auc(y, p, df["NAME_INCOME_TYPE"].astype(str).to_numpy())
                                   .reset_index().to_dict(orient="records"),
            "age": analysis.group_auc(y, p, analysis.age_group(df["AGE_YEARS"]).astype(str).to_numpy(), extra)
                           .reset_index().to_dict(orient="records")}
    # E9: fairness (acceptance = class "low")
    sensitive = {"CODE_GENDER": df[config.FAIRNESS_COL].astype(str).to_numpy(),
                 "age_group": analysis.age_group(df["AGE_YEARS"]).astype(str).to_numpy()}
    out["e9"] = {}
    for name in ["bank_no_history", "client"]:
        out["e9"][name] = {}
        for attr, groups in sensitive.items():
            by_group, overall, gaps = analysis.fairness_frame(y, preds[name]["cal"], groups)
            out["e9"][name][attr] = {"by_group": by_group.reset_index().to_dict(orient="records"),
                                     "max_gap_between_groups": gaps}
    return out, preds


def comparison_table(val_out, test_out):
    """Validation vs test, for the headline figures of E16."""
    rows = []
    def add(measure, v, t):
        rows.append({"measure": measure, "validation": v, "test": t, "test - validation": t - v})
    for name in LISTS:
        vm, tm = val_out["models"][name], test_out["models"][name]
        add(f"{name}: ROC-AUC", vm["roc_auc"]["roc_auc"], tm["roc_auc"]["roc_auc"])
        add(f"{name}: PR-AUC", vm["pr_auc"], tm["pr_auc"])
        add(f"{name}: Brier (calibrated)", vm["brier_calibrated"], tm["brier_calibrated"])
        add(f"{name}: mean calibrated p", vm["mean_calibrated_p"], tm["mean_calibrated_p"])
    add("default rate", val_out["default_rate"], test_out["default_rate"])
    add("E2: AUC with - without history", val_out["e2_with_minus_without"]["auc_diff"],
        test_out["e2_with_minus_without"]["auc_diff"])
    for name in LISTS:
        add(f"E5 {name}: AUC no bureau record", val_out["e5"]["models"][name]["no_bureau_record"]["roc_auc"],
            test_out["e5"]["models"][name]["no_bureau_record"]["roc_auc"])
    add("E5: share without bureau record", val_out["e5"]["subgroup_share"], test_out["e5"]["subgroup_share"])
    for name in ["bank_no_history", "client"]:
        for v_row, t_row in zip(val_out["e14"][name]["classes"], test_out["e14"][name]["classes"]):
            add(f"E14 {name} {v_row['class']}: share", v_row["share"], t_row["share"])
            add(f"E14 {name} {v_row['class']}: default rate", v_row["default_rate"], t_row["default_rate"])
        for kind in ["income_type", "age"]:
            t_rows = {r["group"]: r for r in test_out["e6"][name][kind]}
            for r in val_out["e6"][name][kind]:
                if "roc_auc" in r and "roc_auc" in t_rows.get(r["group"], {}):
                    add(f"E6 {name} {r['group']}: AUC", r["roc_auc"], t_rows[r["group"]]["roc_auc"])
        for attr in ["CODE_GENDER", "age_group"]:
            t_rows = {r["sensitive_feature_0"]: r for r in test_out["e9"][name][attr]["by_group"]}
            for r in val_out["e9"][name][attr]["by_group"]:
                g = r["sensitive_feature_0"]
                for metric in ["acceptance_rate", "false_negative_rate", "false_positive_rate"]:
                    add(f"E9 {name} {attr}={g}: {metric}", r[metric], t_rows[g][metric])
    return pd.DataFrame(rows)
