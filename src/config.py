"""Project-wide constants: paths, preparation parameters and the three feature sets.

See docs/project-brief.md (sections 4 and 5) for the reasoning behind each choice.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "application_train.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Identifiers and target --------------------------------------------------
ID_COL = "SK_ID_CURR"  # kept in every split file (no-history subgroup lookup in bureau.csv)
TARGET = "TARGET"
FAIRNESS_COL = "CODE_GENDER"  # kept in the files for the fairness audit, never used as a feature

# --- Split -------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

# --- Cleaning parameters -----------------------------------------------------
KEPT_CONTRACT_TYPE = "Cash loans"
DAYS_EMPLOYED_ANOMALY = 365243
RARE_CATEGORY_THRESHOLD = 0.001  # categories below 0.1% of rows are grouped into RARE_LABEL
RARE_LABEL = "Other"
HOUSING_EXTRA_COLUMNS = [
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]
HOUSING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")

# Categorical columns whose missing values get their own category instead of the most frequent one
# (OCCUPATION_TYPE is missing for every pensioner: "most frequent" would turn them all into "Laborers").
MISSING_CATEGORY_COLUMNS = ["OCCUPATION_TYPE"]
MISSING_LABEL = "Missing"

# --- Units (assumption deduced from the data, the dataset does not state it) -
# AMT_INCOME_TOTAL and AMT_ANNUITY are read as MONTHLY amounts in the same unit:
# it is the only reading where the median annuity / income (0.17) is a realistic
# debt-to-income ratio and CREDIT_TERM (median ~20.7) is a number of months.
# The form's monthly income is therefore used as is (no x12 conversion).

# Assumed annual interest rate, used ONLY for the indicative estimate shown to the client:
# monthly annuity from (amount, duration) with the standard amortization formula, and its
# share of the monthly income. Since E10 the client features no longer use the annuity or the
# duration, so this rate never enters the score (E7, the sensitivity analysis, was cancelled).
ASSUMED_ANNUAL_INTEREST_RATE = 0.10

# DAYS_* columns converted to positive years (new name: source column)
YEARS_FROM_DAYS = {
    "AGE_YEARS": "DAYS_BIRTH",
    "EMPLOYMENT_YEARS": "DAYS_EMPLOYED",
    "REGISTRATION_YEARS": "DAYS_REGISTRATION",
    "ID_PUBLISH_YEARS": "DAYS_ID_PUBLISH",
    "PHONE_CHANGE_YEARS": "DAYS_LAST_PHONE_CHANGE",
}

# --- Feature sets ------------------------------------------------------------

# Variables carrying Home Credit's risk-based pricing (the annuity includes the interest
# rate set for the applicant). The public form computes the annuity with a fixed assumed
# rate, so this signal does not exist there (experiments E10 and E11).
PRICING_FEATURES = ["CREDIT_TERM", "ANNUITY_INCOME_RATIO"]

# Client feature list as evaluated in E3, before E10 (kept for reproducibility of notebook 03).
CLIENT_FEATURES_E3 = [
    "AGE_YEARS",
    "NAME_FAMILY_STATUS",
    "CNT_CHILDREN",
    "NAME_INCOME_TYPE",
    "EMPLOYMENT_YEARS",
    "IS_PENSIONER",
    "CREDIT_INCOME_RATIO",
    "CREDIT_TERM",
    "ANNUITY_INCOME_RATIO",
]

# Client model: only what the public form can produce, without the pricing signal.
# Since E10 (validation AUC 0.6189, "kept with a stronger caution message"), this is the
# E3 list minus PRICING_FEATURES. No raw amounts: ratios only, to stay currency-independent.
CLIENT_FEATURES = [
    "AGE_YEARS",             # form: age
    "NAME_FAMILY_STATUS",    # form: family status
    "CNT_CHILDREN",          # form: number of dependent children
    "NAME_INCOME_TYPE",      # form: employment type (mapped to the dataset categories)
    "EMPLOYMENT_YEARS",      # form: employment length (missing when IS_PENSIONER = 1)
    "IS_PENSIONER",          # form: employment type "retired" (see CLIENT_FORM_FORMULAS)
    "CREDIT_INCOME_RATIO",   # form: amount / income
]

# How each client feature is computed from the form inputs (documentation for the API).
# income = monthly income from the form (same unit as AMT_INCOME_TOTAL, see "Units" above).
# Employment type "unemployed": no score is computed (too few examples, see project-brief.md).
CLIENT_FORM_FORMULAS = {
    "AGE_YEARS": "age",
    "NAME_FAMILY_STATUS": "family_status",
    "CNT_CHILDREN": "children",
    "NAME_INCOME_TYPE": "employment_type mapped to a dataset category",
    "EMPLOYMENT_YEARS": "employment_length (NaN when IS_PENSIONER = 1)",
    "IS_PENSIONER": "1 if employment_type is retired, else 0",
    "CREDIT_INCOME_RATIO": "amount / income",
}

# Bank model without history: every relevant variable except the hidden history.
_BANK_CATEGORICAL = [
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]
# Bank models: ratios and variables without a monetary unit only (no raw amounts).
_BANK_NUMERIC = [
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OWN_CAR_AGE",
]

# Region ratings computed by Home Credit: kept apart and NOT included in the BANK_* sets by default.
# Unknown method (an Algerian wilaya rating would not share the scale), no data to validate a
# rating per wilaya, and rating a client by where they live raises a fairness issue.
# The bank model's AUC with and without this group is measured in experiment E4 (docs/experiments.md).
REGION_FEATURES = [
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
]
_CREATED = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "REGISTRATION_YEARS",
    "ID_PUBLISH_YEARS",
    "PHONE_CHANGE_YEARS",
    "IS_PENSIONER",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "GOODS_CREDIT_RATIO",
    "EMPLOYMENT_AGE_RATIO",
]
_CONTACT_AND_ADDRESS_FLAGS = [
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
]
_DOCUMENT_FLAGS = [
    f"FLAG_DOCUMENT_{i}" for i in range(2, 22)
    if i not in (7, 10)  # FLAG_DOCUMENT_7 and _10 are constant (all 0) among cash loans
]

BANK_NO_HISTORY_FEATURES = (
    _BANK_CATEGORICAL + _BANK_NUMERIC + _CREATED + _CONTACT_AND_ADDRESS_FLAGS + _DOCUMENT_FLAGS
)

# Excluded from every bank feature set (one-line reason each):
EXCLUDED_COLUMNS = {
    "SK_ID_CURR": "identifier, carries no information about the applicant",
    "TARGET": "label to predict",
    "CODE_GENDER": "gender excluded from all models, kept only for the fairness audit",
    "NAME_CONTRACT_TYPE": "constant after keeping cash loans only",
    "DAYS_BIRTH": "replaced by AGE_YEARS",
    "DAYS_EMPLOYED": "replaced by EMPLOYMENT_YEARS and IS_PENSIONER",
    "DAYS_REGISTRATION": "replaced by REGISTRATION_YEARS",
    "DAYS_ID_PUBLISH": "replaced by ID_PUBLISH_YEARS",
    "DAYS_LAST_PHONE_CHANGE": "replaced by PHONE_CHANGE_YEARS",
    "WEEKDAY_APPR_PROCESS_START": "describes when the file was entered, not the applicant",
    "HOUR_APPR_PROCESS_START": "describes when the file was entered, not the applicant",
    "FLAG_DOCUMENT_7": "constant (all 0) among cash loans",
    "FLAG_DOCUMENT_10": "constant (all 0) among cash loans",
    "AMT_INCOME_TOTAL": "raw monetary amount (currency-dependent), used through ratios only",
    "AMT_CREDIT": "raw monetary amount (currency-dependent), used through ratios only",
    "AMT_ANNUITY": "raw monetary amount (currency-dependent), used through ratios only",
    "AMT_GOODS_PRICE": "raw monetary amount (currency-dependent), used through GOODS_CREDIT_RATIO",
    "INCOME_PER_PERSON": "monetary amount (currency-dependent)",
    "EXT_SOURCE_1/2/3": "external credit scores are a form of credit history (with-history model only)",
    "AMT_REQ_CREDIT_BUREAU_*": "inquiries to the credit bureau are credit history (with-history model only)",
    "REGION_POPULATION_RELATIVE / REGION_RATING_CLIENT(_W_CITY)": "Home Credit region ratings (unknown method, not transferable to wilayas, fairness): separate REGION_FEATURES group, off by default",
    "*_CNT_SOCIAL_CIRCLE": "defaults observed in the social circle are a form of history (with-history model only)",
    "missing-value indicators of the above": "missingness of a history score is itself linked to default",
    "housing columns (_AVG/_MODE/_MEDI + 4 *_MODE)": "dropped during cleaning (empty on ~47% of rows)",
}

HISTORY_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    # Social circle: observations and defaults among the applicant's surroundings (history of the circle)
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
]

# Raw monetary columns: must never appear in a feature set (currency independence).
MONETARY_COLUMNS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "INCOME_PER_PERSON"]

# Bank model with history: reference model, to measure what the history adds.
BANK_WITH_HISTORY_FEATURES = BANK_NO_HISTORY_FEATURES + HISTORY_FEATURES

FEATURE_SETS = {
    "client": CLIENT_FEATURES,
    "bank_no_history": BANK_NO_HISTORY_FEATURES,
    "bank_with_history": BANK_WITH_HISTORY_FEATURES,
}
