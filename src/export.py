"""Export of the two application models (bank without history, client) to artifacts/.

Each artifact is the frozen calibrated model of E16 copied byte for byte (same SHA-256
as in reports/final/manifest.json): preprocessing + LightGBM + Platt calibrator in one
file, plus a JSON metadata file. Loading an artifact requires the `src` package
(the pickled objects reference src.preprocessing, src.train and src.calibration).
"""
import json
import shutil
from datetime import date
from importlib.metadata import version as pkg_version

import joblib

from src import config, risk
from src import final
from src.preprocessing import split_column_types

ARTIFACTS_DIR = config.ROOT / "artifacts"
APP_MODELS = {"bank_no_history": "bank", "client": "client"}
ARTIFACT_VERSION = "1.0.0"


def artifact_path(list_name):
    return ARTIFACTS_DIR / f"noria_{list_name}_v{ARTIFACT_VERSION}.joblib"


def metadata_path(list_name):
    return ARTIFACTS_DIR / f"noria_{list_name}_v{ARTIFACT_VERSION}.metadata.json"


def _feature_spec(model, train):
    """Expected features with their type (and allowed categories for categorical ones)."""
    categorical, binary, continuous = split_column_types(train, model.features)
    pre = model.pipeline.named_steps["preprocess"]
    categories = {}
    for name, transformer, cols in pre.transformers_:
        if name.startswith("categorical"):
            enc = transformer.named_steps["onehot"]
            for col, cats in zip(cols, enc.categories_):
                categories[col] = [str(c) for c in cats]
    spec = []
    for f in model.features:
        kind = "categorical" if f in categorical else "binary" if f in binary else "continuous"
        item = {"name": f, "type": kind, "missing_allowed": True}
        if kind == "categorical":
            item["categories"] = categories.get(f, [])
        spec.append(item)
    return spec


def export(train):
    """Copy the frozen models to artifacts/ and write their metadata. Returns the metadata dicts."""
    manifest, _ = final.verify_manifest()
    test_metrics = json.loads((final.FINAL_DIR / "e16_test.json").read_text(encoding="utf-8"))
    val_metrics = json.loads((final.FINAL_DIR / "e16_validation.json").read_text(encoding="utf-8"))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for list_name, role in APP_MODELS.items():
        frozen = manifest["models"][list_name]
        src_path = config.ROOT / frozen["file"]
        dst = artifact_path(list_name)
        shutil.copyfile(src_path, dst)
        digest = final.sha256(dst)
        assert digest == frozen["sha256"], "artifact differs from the frozen model"
        model = joblib.load(dst)
        labels = risk.BANK_RECOMMENDATION if role == "bank" else risk.CLIENT_CATEGORY
        tm, vm = test_metrics["models"][list_name], val_metrics["models"][list_name]
        meta = {
            "name": f"noria_{list_name}", "role": role, "version": ARTIFACT_VERSION,
            "exported_on": date.today().isoformat(), "frozen_on": manifest["frozen_on"],
            "file": str(dst.relative_to(config.ROOT)), "sha256": digest,
            "contents": "scikit-learn pipeline (preprocessing + LightGBM) wrapped with a Platt calibrator "
                        "(src.calibration.CalibratedModel); predict_proba returns the calibrated probability of default",
            "requires": {"python_package": "src (this repository)",
                         "versions": {p: pkg_version(p) for p in ["scikit-learn", "lightgbm", "pandas", "numpy", "shap"]}},
            "features": _feature_spec(model, train),
            "calibration": model.method,
            "lgbm_params": frozen["lgbm_params"], "n_estimators": frozen["n_estimators"],
            "classes": {"thresholds": {"low": risk.LOW_THRESHOLD, "high": risk.HIGH_THRESHOLD},
                        "rule": "low if p < 0.05; medium if 0.05 <= p < 0.12; high if p >= 0.12",
                        "labels": labels},
            "assumed_annual_interest_rate": config.ASSUMED_ANNUAL_INTEREST_RATE
            if role == "client" else None,
            "assumed_rate_use": "indicative monthly annuity shown to the client only, never in the score"
            if role == "client" else None,
            "metrics": {
                "test": {"roc_auc": tm["roc_auc"], "pr_auc": tm["pr_auc"], "brier_calibrated": tm["brier_calibrated"]},
                "validation": {"roc_auc": vm["roc_auc"], "pr_auc": vm["pr_auc"], "brier_calibrated": vm["brier_calibrated"]},
            },
            "model_card": "docs/model-card.md",
        }
        metadata_path(list_name).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        out[list_name] = meta
    return out
