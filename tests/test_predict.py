"""Check src.predict on 3 validation rows: same calibrated probabilities as the frozen models.

Run: .venv/bin/python tests/test_predict.py   (or with pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config, data, explain, final, predict, risk


def _rows(val):
    """A pensioner (missing employment length), the first row, and the riskiest client of the bank model."""
    pensioner = int(np.flatnonzero(val["IS_PENSIONER"].to_numpy() == 1)[0])
    return [pensioner, 0]


def test_predict_matches_frozen_models():
    _, val = data.load_train_val()
    frozen = final.load_frozen_models()
    for name in ["bank_no_history", "client"]:
        model = frozen[name]
        p_val = model.predict_proba(val)               # as computed in notebook 07 (whole validation set)
        rows = _rows(val) + [int(np.argmax(p_val))]
        for i in rows:
            record = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                      for k, v in val.iloc[i][model.features].to_dict().items()}
            out = predict.predict(name, record)
            assert out["probability"] == float(p_val[i]), (name, i, out["probability"], p_val[i])
            assert out["class"] == risk.assign_class([p_val[i]])[0]
            total = out["shap_base_value"] + sum(out["shap_contributions"].values())
            assert abs(total - out["raw_log_odds"]) < 1e-9
            print(f"{name} row {i}: p = {out['probability']:.6f} (notebook {p_val[i]:.6f}) · {out['class']} / "
                  f"{out['label']} · top factor {out['top_factors'][0]['feature']}")


if __name__ == "__main__":
    test_predict_matches_frozen_models()
    print("OK: predict() reproduces the frozen models exactly")
