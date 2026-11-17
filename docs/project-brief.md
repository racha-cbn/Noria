# Noria — Project brief

> **Assessing and explaining the credit risk of profiles with no banking history.**
> Web + AI project · Bachelor's degree in computer science (SALEAM-IA), NIT · Racha CHABANE · Due end of December 2026
> Last updated: 17 November 2026 (after the data preparation, session 2)

---

## 1. Problem and solution

**Problem.** In Algeria, young workers, freelancers and self-employed entrepreneurs are often refused credit because they have no banking history. The refusal is a black box: they know neither why, nor what to improve. On the bank side, the advisor lacks objective tools to assess these atypical profiles.

**Solution.** A web application built on trained machine learning models:
- a **client** model, for a public and anonymous estimate;
- a **bank** model, more accurate, that computes an explained risk score for the advisor.

The tool recommends, the human decides: **no decision is automatic.**

---

## 2. Users

| Role | Persona | Need | On Noria |
|---|---|---|---|
| **Client** (public) | Amine, 26, freelance developer in Blida, irregular income, no past credit | "I want to know if I stand a chance before going in." | No account, on a phone: 3-step form → category (favorable / to strengthen / fragile) + concrete suggestions. Nothing is stored or sent to the bank. |
| **Advisor** | Samia, 38, advisor for 8 years, ~20 files per week | "If the tool says no, I must be able to explain it to the client." | Logged in: enters the verified information → probability of default, risk class, explanatory factors, recommendation → records her decision. |
| **Administrator** | Branch manager or IT manager | Control the tool | Creates advisor accounts, sees all files, consults the audit log. |

**11 pages:** 3 on the client side, 5 on the advisor side (login shared with the admin), 4 on the admin side. The two sides are independent: the client submits nothing to the bank from the application.

---

## 3. Data

**Source:** Home Credit Default Risk (Home Credit Group, Kaggle competition 2018). Table used: `application_train.csv`.

**Key figures (EDA, session 1):**
- 307,511 applications × 122 columns (106 numeric, 16 categorical)
- `TARGET`: **8.07% defaults** (ratio 1:11.4) → imbalanced classes
- 67 columns with missing values, 41 of them above 50% (mostly the housing columns)
- `EXT_SOURCE_1/2/3` are by far the variables most correlated with the target

**Relevance:** Home Credit lends precisely to populations with little or no credit history.

### Data choices
- **Only the main table is used for training.** The history tables (credit bureau, past payments) are excluded from the features, to stay consistent with the "no history" target.
- **`bureau.csv` is used for evaluation only:** test-set clients absent from `bureau.csv` form the "no history" subgroup on which performance is measured separately. It is never used for training.
- **Only standard loans (`Cash loans`) are kept.** Noria simulates a credit defined by an amount and a duration; for revolving loans (~10%), the duration has no meaning.

---

## 4. Data preparation

| Step | Content |
|---|---|
| **Filtering** | Cash loans only; removal of rows with `CODE_GENDER = XNA` (4) and `NAME_FAMILY_STATUS = Unknown` (2), all of which are revolving loans (0 left after the cash-loans filter). Result: 278,232 rows, 8.35% defaults |
| **Cleaning** | `DAYS_EMPLOYED = 365243` (55,374 rows, pensioners and unemployed) → `IS_PENSIONER` flag, then value set to missing; removal of the normalized housing columns (~47 columns); rare categories (< 0.1% of rows) grouped into "Other" |
| **Created variables** | Age (years), employment length (years), the other `DAYS_*` in years (registration, ID publication, last phone change), credit / income, annuity / income (debt-to-income ratio), duration ≈ credit ÷ annuity, income per person, employment length / age |
| **Frozen split** | Train 70% · validation 15% · test 15%, stratified on `TARGET`, `random_state = 42`, `SK_ID_CURR` kept |
| **Pipeline (fit on the train set only)** | Imputation (median / most frequent category), capping of extremes (1st–99th percentile, continuous variables only), one-hot encoding (unknown categories ignored), scaling for the logistic regression |

**Why ratios?** They make the model less dependent on the currency and the standard of living of the country of origin.

**Anti-leakage rule:** everything learned from the data (medians, percentiles, encoding) is learned on the train set only, in a reproducible scikit-learn pipeline.

---

## 5. The models

Three feature sets, defined in `src/config.py`:

| | **Client (public)** | **Bank without history** | **Bank with history** |
|---|---|---|---|
| **Role** | Quick estimate on declared, unverified data | Accurate score on verified data, without history | Reference: measure what the history adds |
| **Variables** | Age, family status, children, employment type, employment length, pensioner, credit / income, duration, annuity / income | All relevant variables, **except** `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*` and their missing-value indicators | Bank without history + `EXT_SOURCE_*` + `AMT_REQ_CREDIT_BUREAU_*` |
| **Output** | Category favorable / to strengthen / fragile, strengths and points to strengthen | Calibrated probability of default, risk class low / medium / high, recommendation accept / with conditions / refuse | Used for the comparison |

**Algorithms:** LightGBM (main model), compared with a baseline logistic regression that must be beaten. Bonus: XGBoost and CatBoost for a comparison table.

**Training:** class weighting, 5-fold stratified cross-validation, hyperparameter tuning, early stopping.

**Calibration:** isotonic or Platt, so that a displayed probability of 20% corresponds to about 20% actual defaults.

**Explainability:** SHAP. The 5 to 10 factors that weighed on each file, translated into plain French for the advisor, and into simple sentences for the client.

**Client model constraint:** each variable must be computable from the form inputs alone (the annuity equals amount ÷ duration). The dataset gives an **annual** income, the form asks for a **monthly** income: ×12 conversion in the API. **To be confirmed:** in 91.1% of rows the annuity is larger than income ÷ 12, which contradicts "annual income + monthly annuity" (session 2, notebook 02, section 5.1).

---

## 6. Evaluation

- **Main metric: AUC-ROC** (suited to imbalance, unlike accuracy), complemented by AUC-PR, recall on defaults and the confusion matrix.
- **Calibration curve.**
- **Test set used only once**, at the end.
- **Gap between bank with / without history / client measured and displayed: this is the original contribution of the project.** It shows what the lack of history and unverified information cost in accuracy.
- **Performance on the "no history" subgroup** (test clients absent from `bureau.csv`), compared with overall performance.
- **Performance by employment type and by age** (`Working`, `Commercial associate` as the profile closest to freelancers, 20-35 years old), to check that the model remains usable for the target population.
- **Decision rule for the client model, set before seeing the results** (AUC on the validation set): ≥ 0.65 → kept; between 0.60 and 0.65 → kept with a stronger caution message; < 0.60 → dropped, and the decision is documented.

---

## 7. Rigor and ethics

- **Gender excluded** from all model variables, kept only for the fairness audit (raw gap observed in the EDA: F 7.0% vs M 10.1% defaults).
- **Fairness audit** (Fairlearn) by sex and age group: acceptance rates and error rates compared.
- **Hidden history excluded** from the model without history: `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`, and the very fact that one of these values is missing (also linked to default).
- Data leakage control, reproducible pipeline, versioned models.

---

## 8. Application and security

**Architecture:** React + Tailwind front end · FastAPI API (Python) · PostgreSQL database. The models are loaded into memory when the API starts and served in real time (response in under 2 s).

**Security:** JWT authentication, passwords hashed with bcrypt, role-based access control on the server side, strict input validation (Pydantic), ORM against injections, audit log, login attempt limiting.

**Public side:** data minimization (no simulation is stored), rate limiting per IP, no public sign-up (only the admin creates accounts).

---

## 9. Client form (aligned with the model)

| Step | Fields |
|---|---|
| **1 · Profile** | Age, family status, number of dependent children |
| **2 · Activity** | Employment type (employee, freelancer, self-employed, unemployed, retired), employment length, monthly income (DA) |
| **3 · Project** | Requested amount (DA), duration (months) |

Removed compared with the initial version: **personal contribution** and **purpose of the credit** (absent from the dataset, so unusable by the model).

Mapping to the dataset: "freelancer" and "self-employed" are mapped to the closest category (`Commercial associate`). This is an approximation, documented in the limitations.

---

## 10. Scope and schedule

**MVP:** the models and SHAP, public simulation, complete advisor area (files, score, decision, dashboard), admin area, security.

**Bonus, in order:** fairness audit, XGBoost / CatBoost comparison, client range, improvement suggestions (smallest change in amount or duration that moves to the next category up), model transparency page, database encryption, PDF result.

| Weeks | Work |
|---|---|
| 1-4 | Data and models, in parallel with the mockup design |
| 5-6 | Backend and security |
| 7-8 | Frontend · **MVP complete on 6 December** |
| 9-10 | Bonus, tests, report, one buffer week |

---

## 11. Acknowledged limitations

- **The data is not Algerian.** Noria is a proof of concept: the method (ratios, double model, explainability, fairness) is transferable, but real-world use would require retraining on local data.
- **The target population is under-represented.** The dataset has no freelancer or self-employed category (only 10 "Businessman"), and 51.6% of applicants are employees. Predictions for profiles like Amine are therefore less reliable.
- **The model without history is less accurate**, since it loses the most predictive variables. This gap is measured and is a result of the project, not a hidden flaw.

---

## 12. Decision log

| Date | Decision | Reason |
|---|---|---|
| 17/11/2026 | **Personal contribution** variable dropped | Goods price lower than or equal to the credit in 99.9% of rows: "price − credit" is negative or zero almost everywhere |
| 17/11/2026 | **Cash loans** only | For revolving loans, credit ÷ annuity is exactly 20 at the median: it is not a real duration |
| 17/11/2026 | `AMT_REQ_CREDIT_BUREAU_*` treated as history | They are inquiries to the credit bureau |
| 17/11/2026 | `EXT_SOURCE` missing-value indicators excluded from the model without history | The absence of an `EXT_SOURCE` is itself linked to default (9.31% vs 7.77% for `EXT_SOURCE_3`) |
| 17/11/2026 | **`IS_PENSIONER`** flag created before erasing `365243` | These rows have a markedly lower default rate (5.40% vs 8.66%): the information matters |
| 17/11/2026 | Normalized housing columns removed | Empty on ~47% of rows and unknown to both the client and the advisor |
| 17/11/2026 | `bureau.csv` used **for evaluation only** | Prove performance on real no-history profiles, without contradicting the "main table only" choice |
| 17/11/2026 | **Purpose of the credit** removed from the form | Absent from the main table |
| 17/11/2026 | Client model **kept until measured** | Dropping it would be decided on figures (AUC rule, section 6), not on intuition |
| 17/11/2026 | No retraining on a freelancer dataset | No reliable public dataset identified (many are synthetic); performance by employment type is measured instead |
| 17/11/2026 | Follow-up as a final-year project (PFE) dropped | Local data (ANADE, ANGEM, microfinance) very hard to obtain; reframed as a future perspective |
| 17/11/2026 | Capping (1st–99th percentile) applied to **continuous variables only**, not to 0/1 flags | On a rare flag both percentiles are 0: capping would erase it |
| 17/11/2026 | Raw `DAYS_*` replaced by positive years (`AGE_YEARS`, `EMPLOYMENT_YEARS`, `REGISTRATION_YEARS`, `ID_PUBLISH_YEARS`, `PHONE_CHANGE_YEARS`) | Same information, readable units; keeping both would duplicate features |
| 17/11/2026 | `WEEKDAY_APPR_PROCESS_START` and `HOUR_APPR_PROCESS_START` excluded from the bank models | They describe when the file was entered, not the applicant |
| 17/11/2026 | `FLAG_DOCUMENT_7` and `FLAG_DOCUMENT_10` excluded | Constant (all 0) among cash loans |
| 17/11/2026 | Rare-category threshold set at 0.1% of rows, computed on the prepared data before the split | Uses category frequencies only (no target), as planned in the preparation step |
