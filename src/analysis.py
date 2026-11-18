"""Subgroup analyses on the VALIDATION set: E5 (no-history clients), E6 (target population), E9 (fairness)."""
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate

from src import config
from src import evaluate as ev
from src import risk

BUREAU_DATA = config.ROOT / "data" / "raw" / "bureau.csv"

# Pre-registered in docs/experiments.md (E6, reused by E9).
AGE_BINS = [20, 25, 35, 45, 55, 70]
AGE_LABELS = ["20-25", "25-35", "35-45", "45-55", "55-70"]
MIN_CLASS_COUNT = 50  # fewer defaults or non-defaults than this in a group: too small to conclude


def age_group(age_years):
    return pd.cut(age_years, bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)


def bureau_ids(path=BUREAU_DATA):
    """SK_ID_CURR present in bureau.csv. Used ONLY to identify the no-history subgroup, never as a feature."""
    return set(pd.read_csv(path, usecols=["SK_ID_CURR"])["SK_ID_CURR"].unique())


def no_history_mask(df, ids):
    """True for clients absent from bureau.csv (no credit-bureau record)."""
    return ~df[config.ID_COL].isin(ids).to_numpy()


def group_auc(y, p, groups, extra_groups=None):
    """AUC with 95% bootstrap CI per group; groups too small to conclude are flagged.

    extra_groups: optional {name: boolean mask} for aggregates (e.g. ages 20-35).
    """
    y, p = np.asarray(y), np.asarray(p)
    masks = {str(g): (np.asarray(groups) == g) for g in sorted(pd.unique(pd.Series(groups).dropna()))}
    masks.update(extra_groups or {})
    rows = []
    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        n, d = int(m.sum()), int(y[m].sum())
        row = {"group": name, "rows": n, "defaults": d, "default_rate": d / n if n else np.nan,
               "too_small": bool(d < MIN_CLASS_COUNT or (n - d) < MIN_CLASS_COUNT)}
        if 0 < d < n:
            row.update({k: v for k, v in ev.bootstrap_auc(y[m], p[m]).items() if k != "n_bootstrap"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def fairness_frame(y, p_calibrated, sensitive):
    """E9 metrics per group: acceptance = class 'low'; predicted risky = not 'low'; positive = default."""
    y = np.asarray(y)
    risky = (risk.assign_class(p_calibrated) != "low").astype(int)
    mf = MetricFrame(
        metrics={"acceptance_rate": lambda yt, yp: float(np.mean(yp == 0)),
                 "false_negative_rate": false_negative_rate,
                 "false_positive_rate": false_positive_rate,
                 "rows": lambda yt, yp: len(yt),
                 "defaults": lambda yt, yp: int(np.sum(yt))},
        y_true=y, y_pred=risky, sensitive_features=np.asarray(sensitive))
    gaps = {m: float(mf.difference(method="between_groups")[m])
            for m in ["acceptance_rate", "false_negative_rate", "false_positive_rate"]}
    return mf.by_group, mf.overall, gaps
