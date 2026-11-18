"""Model evaluation on the VALIDATION set (session 3).

ROC-AUC (main metric), PR-AUC, bootstrap confidence intervals, and a paired
bootstrap for the AUC difference between two models.
"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config

METRICS_DIR = config.ROOT / "reports" / "metrics"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = config.RANDOM_STATE


def predict_proba(pipeline, df, features):
    """Predicted probability of default (class 1)."""
    return pipeline.predict_proba(df[list(features)])[:, 1]


def _bootstrap_indices(n, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Yield n_boot resamples of range(n); the same (n, n_boot, seed) always gives the same ones."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, size=n)


def bootstrap_auc(y, p, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """ROC-AUC with a 95% percentile bootstrap confidence interval."""
    y, p = np.asarray(y), np.asarray(p)
    aucs = [roc_auc_score(y[i], p[i]) for i in _bootstrap_indices(len(y), n_boot, seed)
            if 0 < y[i].sum() < len(i)]
    return {"roc_auc": float(roc_auc_score(y, p)),
            "ci_low": float(np.percentile(aucs, 2.5)),
            "ci_high": float(np.percentile(aucs, 97.5)),
            "n_bootstrap": len(aucs)}


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """AUC(a) - AUC(b) with a 95% CI; both models are scored on the SAME resamples."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    diffs = [roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i])
             for i in _bootstrap_indices(len(y), n_boot, seed) if 0 < y[i].sum() < len(i)]
    return {"auc_diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "share_of_resamples_above_0": float(np.mean(np.asarray(diffs) > 0)),
            "n_bootstrap": len(diffs)}


def evaluate(y, p):
    """ROC-AUC with bootstrap CI, and PR-AUC (average precision)."""
    metrics = bootstrap_auc(y, p)
    metrics.update(pr_auc=float(average_precision_score(y, p)),
                   rows=int(len(y)), positives=int(np.sum(y)),
                   positive_rate=float(np.mean(y)))
    return metrics


def save_metrics(metrics, name):
    """Write a metrics dict to reports/metrics/<name>.json."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def paired_bootstrap_metric_diff(y, p_a, p_b, metric, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """metric(a) - metric(b) with a 95% CI, both scored on the SAME resamples (any metric(y, p))."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    diffs = [metric(y[i], p_a[i]) - metric(y[i], p_b[i])
             for i in _bootstrap_indices(len(y), n_boot, seed) if 0 < y[i].sum() < len(i)]
    return {"diff": float(metric(y, p_a) - metric(y, p_b)),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "n_bootstrap": len(diffs)}
