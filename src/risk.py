"""Risk classes on the calibrated probability of default (experiment E14).

Thresholds pre-registered in docs/experiments.md; they are never tuned on the results.
The class is a recommendation only: the advisor decides.
"""
import numpy as np
import pandas as pd

LOW_THRESHOLD = 0.05    # p < 5%            -> low
HIGH_THRESHOLD = 0.12   # 5% <= p < 12%     -> medium ; p >= 12% -> high
CLASSES = ["low", "medium", "high"]

BANK_RECOMMENDATION = {"low": "accept", "medium": "accept with conditions", "high": "refuse"}
CLIENT_CATEGORY = {"low": "favorable", "medium": "to strengthen", "high": "fragile"}

MAX_CLASS_SHARE = 0.80  # above this share of clients in one class: reported as a limitation


def assign_class(p):
    """'low' / 'medium' / 'high' for each calibrated probability."""
    p = np.asarray(p, dtype=float)
    return np.where(p < LOW_THRESHOLD, "low", np.where(p < HIGH_THRESHOLD, "medium", "high"))


def class_table(y, p, labels=None):
    """Share of clients and actual default rate per class, plus the E14 limitation checks."""
    classes = pd.Categorical(assign_class(p), categories=CLASSES, ordered=True)
    df = pd.DataFrame({"class": classes, "y": np.asarray(y), "p": np.asarray(p)})
    t = df.groupby("class", observed=False).agg(clients=("y", "size"), defaults=("y", "sum"),
                                                 mean_predicted=("p", "mean"))
    t["share"] = t["clients"] / t["clients"].sum()
    t["default_rate"] = t["defaults"] / t["clients"].replace(0, np.nan)
    if labels:
        t.insert(0, "label", [labels[c] for c in t.index])
    rates = t["default_rate"].dropna().to_numpy()
    checks = {
        "max_class_share": float(t["share"].max()),
        "class_over_80_percent": bool(t["share"].max() > MAX_CLASS_SHARE),
        "empty_classes": [c for c in CLASSES if t.loc[c, "clients"] == 0],
        "default_rates_increase": bool(len(rates) == len(CLASSES) and np.all(np.diff(rates) > 0)),
    }
    checks["limitation"] = checks["class_over_80_percent"] or not checks["default_rates_increase"]
    return t, checks
