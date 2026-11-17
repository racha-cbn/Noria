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

# DAYS_* columns converted to positive years (new name: source column)
YEARS_FROM_DAYS = {
    "AGE_YEARS": "DAYS_BIRTH",
    "EMPLOYMENT_YEARS": "DAYS_EMPLOYED",
    "REGISTRATION_YEARS": "DAYS_REGISTRATION",
    "ID_PUBLISH_YEARS": "DAYS_ID_PUBLISH",
    "PHONE_CHANGE_YEARS": "DAYS_LAST_PHONE_CHANGE",
}

# --- Feature sets ------------------------------------------------------------

# Client model: only what the public form can produce.
# Form inputs: age, family status, number of children, employment type,
# employment length, income, requested amount, duration (months).
CLIENT_FEATURES = [
    "AGE_YEARS",             # form: age
    "NAME_FAMILY_STATUS",    # form: family status
    "CNT_CHILDREN",          # form: number of dependent children
    "NAME_INCOME_TYPE",      # form: employment type (mapped to the dataset categories)
    "EMPLOYMENT_YEARS",      # form: employment length (missing when IS_PENSIONER = 1)
    "IS_PENSIONER",          # form: employment type "retired" (see CLIENT_FORM_FORMULAS)
    "CREDIT_INCOME_RATIO",   # form: amount / income
    "CREDIT_TERM",           # form: amount / (amount / duration) = duration
    "ANNUITY_INCOME_RATIO",  # form: (amount / duration) / income
]
# No raw amounts in the client model: ratios only, to stay currency-independent.

# How each client feature is computed from the form inputs (documentation for the API).
# income = income in the dataset's unit (see the income-unit note in project-brief.md).
CLIENT_FORM_FORMULAS = {
    "AGE_YEARS": "age",
    "NAME_FAMILY_STATUS": "family_status",
    "CNT_CHILDREN": "children",
    "NAME_INCOME_TYPE": "employment_type mapped to a dataset category",
    "EMPLOYMENT_YEARS": "employment_length (NaN when IS_PENSIONER = 1)",
    "IS_PENSIONER": "1 if employment_type has no current job, else 0",
    "CREDIT_INCOME_RATIO": "amount / income",
    "CREDIT_TERM": "amount / annuity = duration, with annuity = amount / duration",
    "ANNUITY_INCOME_RATIO": "(amount / duration) / income",
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
_BANK_NUMERIC = [
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "OWN_CAR_AGE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
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
    "INCOME_PER_PERSON",
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
    "EXT_SOURCE_1/2/3": "external credit scores are a form of credit history",
    "AMT_REQ_CREDIT_BUREAU_*": "inquiries to the credit bureau are credit history",
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
]

# Bank model with history: reference model, to measure what the history adds.
BANK_WITH_HISTORY_FEATURES = BANK_NO_HISTORY_FEATURES + HISTORY_FEATURES

FEATURE_SETS = {
    "client": CLIENT_FEATURES,
    "bank_no_history": BANK_NO_HISTORY_FEATURES,
    "bank_with_history": BANK_WITH_HISTORY_FEATURES,
}
