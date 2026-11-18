# Noria — Project brief

> **Assessing and explaining the credit risk of profiles with no banking history.**
> Web + AI project · Bachelor's degree in computer science (SALEAM-IA), NIT · Racha CHABANE · Due end of December 2026
> Last updated: 18 November 2026 (after the pricing-signal experiments E10 and E11)

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
| **Created variables** | Age (years), employment length (years), the other `DAYS_*` in years (registration, ID publication, last phone change), credit / income, annuity / income (debt-to-income ratio), duration ≈ credit ÷ annuity, goods price / credit, income per person (created but not used: monetary amount), employment length / age |
| **Frozen split** | Train 70% · validation 15% · test 15%, stratified on `TARGET`, `random_state = 42`, `SK_ID_CURR` kept |
| **Pipeline (fit on the train set only)** | Imputation (median / most frequent category; a separate `Missing` category for `OCCUPATION_TYPE`), capping of extremes (1st–99th percentile, continuous variables only), one-hot encoding (unknown categories ignored), scaling for the logistic regression |

**Why ratios?** They make the model less dependent on the currency and the standard of living of the country of origin. This applies to all three models: no raw monetary amount (`AMT_*`, income per person) is used as a feature, only ratios and variables without a monetary unit.

**Anti-leakage rule:** everything learned from the data (medians, percentiles, encoding) is learned on the train set only, in a reproducible scikit-learn pipeline.

---

## 5. The models

Three feature sets, defined in `src/config.py`:

| | **Client (public)** | **Bank without history** | **Bank with history** |
|---|---|---|---|
| **Role** | Quick estimate on declared, unverified data | Accurate score on verified data, without history | Reference: measure what the history adds |
| **Variables** | Age, family status, children, employment type, employment length, pensioner, credit / income (since E10: no duration, no annuity / income, which carried Home Credit's pricing) | All relevant variables (ratios and variables without a monetary unit, including goods price / credit), **except** `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`, `*_CNT_SOCIAL_CIRCLE` and their missing-value indicators. The Home Credit region ratings (`REGION_FEATURES`) are left out by default (section 7) | Bank without history + `EXT_SOURCE_*` + `AMT_REQ_CREDIT_BUREAU_*` + `*_CNT_SOCIAL_CIRCLE` |
| **Output** | Category favorable / to strengthen / fragile, strengths and points to strengthen, with a **stronger caution message** (E10: validation AUC 0.6189, between 0.60 and 0.65) | Calibrated probability of default, risk class low / medium / high, recommendation accept / with conditions / refuse | Used for the comparison |

**Algorithms:** LightGBM (main model), compared with a baseline logistic regression that must be beaten (confirmed in E1: 0.7093 vs 0.6698 validation AUC). Bonus: XGBoost and CatBoost for a comparison table.

**Training:** class weighting (`class_weight="balanced"`), number of trees chosen by early stopping in a 5-fold stratified cross-validation **on the train set**, then refit on the whole train set; `random_state = 42` everywhere (`src/train.py`). Hyperparameter tuning: not done yet (session 3 uses reasonable defaults); it is scheduled for the next session, after E10, on the final feature lists. Trained models are saved in `models/` (not versioned); metrics are saved as JSON in `reports/metrics/` (versioned).

**Calibration:** isotonic or Platt, so that a displayed probability of 20% corresponds to about 20% actual defaults (choice: experiment E8 in [experiments.md](experiments.md)).

**Explainability:** SHAP. The 5 to 10 factors that weighed on each file, translated into plain French for the advisor, and into simple sentences for the client.

**Client model constraint:** each variable must be computable from the form inputs alone.
- **Income unit (assumption deduced from the data):** the dataset does not state the unit of `AMT_INCOME_TOTAL` and `AMT_ANNUITY`. They are read as **monthly** amounts in the same unit, the only coherent reading: in 91.1% of rows the annuity is larger than income ÷ 12, so an annual income is ruled out, while the monthly reading gives a realistic median debt-to-income ratio (17.2%) and a `CREDIT_TERM` in months (median 20.7). The form's monthly income is used as is, with no conversion.
- **Annuity and duration:** in the dataset the annuity includes interest. For the public client form only, the API computes the monthly annuity from the amount and the duration with the standard amortization formula and an assumed annual interest rate (10% by default, constant `ASSUMED_ANNUAL_INTEREST_RATE` in `src/config.py`), then computes the ratios exactly as in the dataset (credit ÷ annuity, annuity ÷ income, credit ÷ income). **Since E10, the client features no longer use the annuity or the duration** (`CREDIT_TERM` and `ANNUITY_INCOME_RATIO` removed): only credit ÷ income remains. What to do with the duration field and the assumed rate is an open question.
- **Sensitivity to the assumed rate:** experiment E7 in [experiments.md](experiments.md).

---

## 6. Evaluation

- **Main metric: AUC-ROC** (suited to imbalance, unlike accuracy), complemented by AUC-PR, recall on defaults and the confusion matrix.
- **Test set used only once**, at the end. Selection metrics are computed on the validation set. The test set is locked in the code: `load_split("test")` raises an error unless `final=True` is passed explicitly (`src/data.py`).
- **Gap between bank with / without history / client measured and displayed: this is the original contribution of the project.** It shows what the lack of history and unverified information cost in accuracy.
- **Every planned measurement, with its decision rule fixed in advance, is pre-registered in [experiments.md](experiments.md)** (E1 to E9: baseline, cost of the history, client model viability, region features, real no-history profiles, target population, interest-rate sensitivity, calibration, fairness audit), then completed with its result.

---

## 7. Rigor and ethics

- **Gender excluded** from all model variables, kept only for the fairness audit (raw gap observed in the EDA: F 7.0% vs M 10.1% defaults).
- **Fairness audit** (Fairlearn) by sex and age group: acceptance rates and error rates compared (experiment E9 in [experiments.md](experiments.md)).
- **Hidden history excluded** from the model without history: `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`, the social-circle counts `*_CNT_SOCIAL_CIRCLE` (history of the applicant's surroundings), and the very fact that one of these values is missing (also linked to default).
- **Region ratings left out by default** (`REGION_FEATURES` in `src/config.py`, separate from the `BANK_*` sets): (1) these ratings are specific to Home Credit and their method is unknown, so a rating created for the Algerian wilayas would not share the same scale; (2) no data would make it possible to validate a rating per wilaya; (3) rating a client according to their region raises a fairness issue: they would be penalized for where they live. The address-consistency flags (`REG_*` / `LIVE_*`) stay in the bank sets, since they can be filled in for any client.
- Data leakage control, reproducible pipeline, versioned models.

---

## 8. Application and security

**Architecture:** React + Tailwind front end · FastAPI API (Python) · PostgreSQL database. The models are loaded into memory when the API starts and served in real time (response in under 2 s).

**Security:** JWT authentication, passwords hashed with bcrypt, role-based access control on the server side, strict input validation (Pydantic), ORM against injections, audit log, login attempt limiting.

**Public side:** data minimization (no simulation is stored), rate limiting per IP, no public sign-up (only the admin creates accounts).

---

## 9. Forms (aligned with the models)

### Client form (public)

| Step | Fields |
|---|---|
| **1 · Profile** | Age, family status, number of dependent children |
| **2 · Activity** | Employment type (employee, freelancer, self-employed, unemployed, retired), employment length, monthly income (DA) |
| **3 · Project** | Requested amount (DA), duration (months) |

Removed compared with the initial version: **personal contribution** and **purpose of the credit** (absent from the dataset, so unusable by the model).

Mapping to the dataset: "freelancer" and "self-employed" are mapped to the closest category (`Commercial associate`). This is an approximation, documented in the limitations.

**"Unemployed" profile:** if the client chooses "unemployed", Noria displays no score, only a message explaining that this profile cannot be estimated reliably (too few examples in the data: 15 unemployed applicants among the cash loans).

### Advisor form

The advisor enters **verified** information, which feeds the bank models (`BANK_NO_HISTORY_FEATURES`, `BANK_WITH_HISTORY_FEATURES` in `src/config.py`).

| Step | Fields |
|---|---|
| **1 · Applicant** | Age, family status, number of children, number of family members, education, housing type, accompanied by (`NAME_TYPE_SUITE`), owns a car (and its age), owns real estate |
| **2 · Employment and income** | Employment type, occupation, employer's sector (`ORGANIZATION_TYPE`), employment length, verified monthly income (DA) |
| **3 · Loan offer** | **Goods / project price** (DA), **Total financed amount (including fees and insurance)** (DA), **monthly annuity of the bank's offer** (DA) |
| **4 · File and contact details** | Documents provided (`FLAG_DOCUMENT_*`), phone and e-mail flags, address consistency (region / city of residence, registration and work), years since registration, ID document issue and last phone change |

**Two distinct amount fields, with a help text:** "Goods / project price" is the price of what the client wants to finance; "Total financed amount (including fees and insurance)" is the full amount of the loan, which in the dataset is always greater than or equal to the price (`GOODS_CREDIT_RATIO` ≤ 1). Entering the price in both fields would wrongly give a ratio of 1.

**Real annuity:** the advisor enters the monthly annuity of the bank's actual offer (verified data). The amortization formula with an assumed interest rate is used **only** for the public client form.

The ratios are computed by the API exactly as in the dataset: total financed amount ÷ income, annuity ÷ income, total financed amount ÷ annuity, goods price ÷ total financed amount.

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
- **The model without history is less accurate**, since it loses the most predictive variables. This gap is measured and is a result of the project, not a hidden flaw (E2: −0.057 AUC).
- **`CREDIT_TERM` reflects Home Credit's pricing.** It was the top feature of the client and bank models (50% and 28% of the gain). Annuities follow a grid of 806 annuity / credit ratios whose default rates differ widely, so the variable most likely carries the lender's risk-based pricing. The public form cannot reproduce it, so the client model was re-evaluated without `CREDIT_TERM` and `ANNUITY_INCOME_RATIO` (E10): its validation AUC drops from 0.6883 to **0.6189**. In the bank model without history, these two variables are worth +0.0298 AUC (E11: 0.7093 vs 0.6795).
- **The client model is weak** (validation AUC 0.6189, close to the 0.60 threshold): it is shown with a stronger caution message, and it rests mostly on employment length, age and credit ÷ income.

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
| 17/11/2026 | `AMT_INCOME_TOTAL` and `AMT_ANNUITY` read as **monthly** amounts in the same unit; ×12 conversion removed | The dataset does not state the unit; it is the only coherent reading (annuity > income ÷ 12 in 91.1% of rows, median debt-to-income 17.2%, `CREDIT_TERM` in months) |
| 17/11/2026 | Client annuity computed with the amortization formula and an assumed annual rate of 10% (`ASSUMED_ANNUAL_INTEREST_RATE`) | The dataset annuity includes interest: ratios must be computed the same way; sensitivity at 5% / 10% / 15% in session 3 |
| 17/11/2026 | "Unemployed" client profile: no score, an explanatory message instead | Too few examples (15 among cash loans) for a reliable estimate; `IS_PENSIONER` kept as is |
| 17/11/2026 | `OCCUPATION_TYPE` missing values imputed with a `Missing` category | Missing for every pensioner (31.9% of rows): "most frequent" would turn them all into "Laborers" |
| 17/11/2026 | Raw amounts removed from the bank models (`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `INCOME_PER_PERSON`); `GOODS_CREDIT_RATIO` added | Only ratios and unit-free variables, so the model does not depend on the currency (the advisor enters amounts in DA) |
| 17/11/2026 | `*_CNT_SOCIAL_CIRCLE` (4 columns) moved to the bank model with history only | Defaults observed in the applicant's surroundings are a form of history |
| 17/11/2026 | Advisor form: the advisor enters the **real monthly annuity** of the bank's offer; the assumed-rate amortization formula is used for the public client form only | The advisor works on verified data: the real annuity is more faithful than an estimate |
| 17/11/2026 | Advisor form: two distinct fields, "Goods / project price" and "Total financed amount (including fees and insurance)", with a help text | In the dataset the credit is always ≥ the goods price (fees, insurance): entering the price twice would force `GOODS_CREDIT_RATIO` to 1 |
| 17/11/2026 | Region ratings (`REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT`, `REGION_RATING_CLIENT_W_CITY`) moved to a separate `REGION_FEATURES` group, left out of the `BANK_*` sets by default; `REG_*` / `LIVE_*` flags kept | Home Credit-specific ratings with an unknown method (a wilaya rating would not share the scale), no data to validate a rating per wilaya, and a fairness issue (penalized for where one lives); AUC with and without the group measured in session 3 |
| 18/11/2026 | Pre-registered experiment log created: [experiments.md](experiments.md) (E1 to E9); details of the planned measurements moved there from section 6 | Each measurement and its decision rule are written before the result is seen; a rule changed afterwards must be logged here with its reason |
| 18/11/2026 | Number of LightGBM trees chosen by early stopping in a 5-fold stratified CV **on the train set**, then refit on the whole train set; no hyperparameter search yet | The validation set stays untouched for the comparisons of the experiment log |
| 18/11/2026 | Test set locked in the code (`load_split("test")` refuses without `final=True`) | Guarantees the test set is used only once, for the final results |
| 18/11/2026 | Trained models saved in `models/` (not versioned), metrics saved as JSON in `reports/metrics/` (versioned) | Models are regenerated by the notebooks; metrics must stay traceable |
| 18/11/2026 | E1: LightGBM confirmed as the main model | Validation AUC 0.7093 vs 0.6698 for the logistic regression (rule in [experiments.md](experiments.md)) |
| 18/11/2026 | E3: client model kept | Validation AUC 0.6883 ≥ 0.65 (rule applied as written); caveat on `CREDIT_TERM` added to the limitations |
| 18/11/2026 | E10 and E11 pre-registered (commit "Pre-register E10 and E11") before being run; **E10, not E3, decides the fate of the client model** (E3 marked "Superseded by E10", result and rule unchanged) | Session 3 showed that `CREDIT_TERM` and `ANNUITY_INCOME_RATIO` carry Home Credit's risk-based pricing, absent from the public form: E3 measured the client model with a signal it will not have |
| 18/11/2026 | E10: client model **kept with a stronger caution message**; `CLIENT_FEATURES` reduced to 7 features (without `CREDIT_TERM` and `ANNUITY_INCOME_RATIO`); the E3 list is kept as `CLIENT_FEATURES_E3` for reproducibility | Validation AUC 0.6189 [0.6092, 0.6291], between 0.60 and 0.65 (rule applied as written) |
| 18/11/2026 | E11 reported: the pricing signal is worth +0.0298 AUC [+0.0249, +0.0349] in the bank model without history | Measurement to report (no decision rule); the bank feature lists are unchanged |
| 18/11/2026 | Hyperparameter tuning scheduled for the next session, after E10, on the final feature lists | Tuning on feature lists that could still change would be wasted |
