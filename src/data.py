"""Deterministic data preparation (steps A and C of session 2).

Nothing here learns from the target or fits a statistic used by a model:
filtering, column removal, unit conversions and ratios are row-wise rules.
The only data-dependent step is the rare-category grouping (category
frequencies, no target involved), which the project brief places before the split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path=config.RAW_DATA):
    """Read application_train.csv (read-only, never written back)."""
    return pd.read_csv(path)


def filter_rows(df):
    """Apply the validated row filters and return (filtered_df, row counts per step)."""
    steps = [("raw", len(df))]
    df = df[df["NAME_CONTRACT_TYPE"] == config.KEPT_CONTRACT_TYPE]
    steps.append(("NAME_CONTRACT_TYPE == 'Cash loans'", len(df)))
    df = df[df["CODE_GENDER"] != "XNA"]
    steps.append(("CODE_GENDER != 'XNA'", len(df)))
    df = df[df["NAME_FAMILY_STATUS"] != "Unknown"]
    steps.append(("NAME_FAMILY_STATUS != 'Unknown'", len(df)))
    counts = pd.DataFrame(steps, columns=["step", "rows"]).set_index("step")
    counts["removed"] = (-counts["rows"].diff()).fillna(0).astype(int)
    return df.reset_index(drop=True), counts


def housing_columns(columns):
    """Normalized housing columns to drop (suffixes _AVG/_MODE/_MEDI + 4 categorical *_MODE)."""
    return [c for c in columns
            if c.endswith(config.HOUSING_SUFFIXES) or c in config.HOUSING_EXTRA_COLUMNS]


def safe_divide(numerator, denominator):
    """Element-wise division; division by zero or by NaN gives NaN (never inf)."""
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    return (num / den.where(den != 0)).replace([np.inf, -np.inf], np.nan)


def add_features(df):
    """IS_PENSIONER flag, DAYS_* in years, and the ratio features."""
    df = df.copy()
    anomaly = df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_ANOMALY
    df["IS_PENSIONER"] = anomaly.astype(int)
    df.loc[anomaly, "DAYS_EMPLOYED"] = np.nan

    for new_col, days_col in config.YEARS_FROM_DAYS.items():
        df[new_col] = -df[days_col] / 365.25 + 0.0  # + 0.0 turns -0.0 into 0.0

    df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["INCOME_PER_PERSON"] = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["EMPLOYMENT_AGE_RATIO"] = safe_divide(df["EMPLOYMENT_YEARS"], df["AGE_YEARS"])

    # Session 4 candidates (experiment E12): deterministic, no raw amount, no history.
    doc_flags = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOC_COUNT"] = df[doc_flags].sum(axis=1)
    df["CONTACT_COUNT"] = df[config.CONTACT_FLAGS].sum(axis=1)
    df["AGE_AT_EMPLOYMENT_START"] = df["AGE_YEARS"] - df["EMPLOYMENT_YEARS"]
    df["CHILDREN_RATIO"] = safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
    df["ID_PUBLISH_AGE_RATIO"] = safe_divide(df["ID_PUBLISH_YEARS"], df["AGE_YEARS"])
    df["REGISTRATION_AGE_RATIO"] = safe_divide(df["REGISTRATION_YEARS"], df["AGE_YEARS"])
    return df


def categorical_columns(df):
    """Non-numeric columns, excluding the fairness column (never grouped or encoded)."""
    return [c for c in df.select_dtypes(exclude="number").columns if c != config.FAIRNESS_COL]


def group_rare_categories(df, threshold=config.RARE_CATEGORY_THRESHOLD, label=config.RARE_LABEL):
    """Replace categories below `threshold` (share of rows) by `label`. Missing values stay missing.

    Returns (df, {column: [grouped categories]}).
    """
    df = df.copy()
    grouped = {}
    for col in categorical_columns(df):
        shares = df[col].value_counts(normalize=True, dropna=True)
        rare = shares[shares < threshold].index.tolist()
        if rare:
            df[col] = df[col].where(~df[col].isin(rare), label)
            grouped[col] = rare
    return df, grouped


def prepare(df_raw):
    """Full deterministic preparation. Returns (prepared_df, report dict)."""
    df, row_counts = filter_rows(df_raw)
    dropped = housing_columns(df.columns)
    df = df.drop(columns=dropped)
    df = add_features(df)
    df, grouped = group_rare_categories(df)
    report = {"row_counts": row_counts, "dropped_housing_columns": dropped, "rare_groups": grouped}
    return df, report


def split(df, random_state=config.RANDOM_STATE):
    """70/15/15 split stratified on TARGET. Returns (train, val, test)."""
    holdout_share = config.VAL_SIZE + config.TEST_SIZE
    train, holdout = train_test_split(
        df, test_size=holdout_share, stratify=df[config.TARGET], random_state=random_state)
    val, test = train_test_split(
        holdout, test_size=config.TEST_SIZE / holdout_share,
        stratify=holdout[config.TARGET], random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train, val, test, out_dir=config.PROCESSED_DIR):
    """Write train/val/test as parquet in data/processed/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in {"train": train, "val": val, "test": test}.items():
        paths[name] = out_dir / f"{name}.parquet"
        part.to_parquet(paths[name], index=False)
    return paths


TEST_USED_FILE = config.ROOT / "reports" / "final" / "TEST_USED"


class TestSetLockedError(RuntimeError):
    """Raised when the test set is requested without final=True."""


class TestSetAlreadyUsedError(RuntimeError):
    """Raised when the test set is requested again after the final evaluation (E16)."""


def load_split(name, final=False, reproduce_e16=False, in_dir=config.PROCESSED_DIR):
    """Read one split ("train", "val" or "test") from data/processed/.

    The test set is locked: it is used only once, for the final results
    (docs/experiments.md). Loading it requires final=True explicitly.
    Once the final evaluation has been run (reports/final/TEST_USED exists), loading
    it again also requires reproduce_e16=True, which is reserved for reproducing E16
    exactly as it was run (same frozen models, same code) and never for any new analysis.
    """
    if name not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {name!r}")
    if name == "test" and not final:
        raise TestSetLockedError(
            "The test set is locked until the final evaluation. "
            "Use load_split('test', final=True) only for the final results.")
    if name == "test" and TEST_USED_FILE.exists() and not reproduce_e16:
        raise TestSetAlreadyUsedError(
            f"The test set was already used for the final evaluation ({TEST_USED_FILE.read_text().strip()}). "
            "Pass reproduce_e16=True only to reproduce E16 as it was run.")
    return pd.read_parquet(in_dir / f"{name}.parquet")


def mark_test_used(note="E16 final evaluation"):
    """Record that the test set has been used (reports/final/TEST_USED, with the date)."""
    from datetime import datetime
    TEST_USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEST_USED_FILE.write_text(f"{datetime.now().isoformat(timespec='seconds')} · {note}\n", encoding="utf-8")
    return TEST_USED_FILE


def load_train_val(in_dir=config.PROCESSED_DIR):
    """Read the train and validation splits (never the test set)."""
    return load_split("train", in_dir=in_dir), load_split("val", in_dir=in_dir)
