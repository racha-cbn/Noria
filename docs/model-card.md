# Noria — Model card

> Two models used by the application: the **bank model without history** (advisor) and the **client model** (public).
> Version **1.0.0** · frozen on 18 November 2026 · final evaluation on the test set: 18 November 2026 (experiment E16).
> Files: `artifacts/noria_bank_no_history_v1.0.0.joblib` and `artifacts/noria_client_v1.0.0.joblib` (+ `.metadata.json`), loaded with `src/predict.py`.
> Reference documents: [project-brief.md](project-brief.md), [experiments.md](experiments.md), `reports/final/final_results.md`.

---

## 1. Intended use

| | Bank model without history | Client model |
|---|---|---|
| **Users** | Bank advisors, logged in | Anyone, public and anonymous |
| **Purpose** | Estimate the probability of default of a cash-loan applicant from **verified** information, without any credit history, and explain it | Help a person **understand what banks look at**, from information they declare themselves |
| **Output** | Calibrated probability of default, risk class (low / medium / high) and recommendation (accept / accept with conditions / refuse), main factors (SHAP) | Indicative category (favorable / to strengthen / fragile) with an explicit caution message, main factors in plain language, estimated monthly annuity |
| **Decision** | **A recommendation only: the advisor decides** | No decision: nothing is stored or sent to a bank |

### Uses to avoid
- **Automatic decisions** (granting or refusing a loan without a human).
- Using the **client** result as a promise or a refusal ("your chances"); it is an indicative tool with a weak AUC (0.62), lower still for some target profiles.
- **Revolving loans** or credit types other than cash loans (the models were trained on cash loans only).
- **Applicants outside the training ranges** (e.g. under 20 or over 70) or with unverified data in the bank model.
- **Real-world deployment in Algeria without retraining** on local data: the data is not Algerian (proof of concept).
- Using the bank model **with** credit-history variables, region ratings or raw amounts: it was not trained with them.

---

## 2. Data

- **Source:** Home Credit Default Risk (Kaggle, 2018), `application_train.csv` only. `bureau.csv` was used **only** to identify the subgroup of clients without a credit-bureau record (E5), never as a feature.
- **Scope:** cash loans only. 278,232 applications, 8.35% defaults.
- **Split:** train 70% (194,762), validation 15% (41,735), test 15% (41,735), stratified on `TARGET`, `random_state = 42`. The test set was used **once** (E16), then locked (`reports/final/TEST_USED`).
- **Units:** `AMT_INCOME_TOTAL` and `AMT_ANNUITY` are read as monthly amounts in the same unit. This is an assumption deduced from the data; the dataset does not state it.

---

## 3. Variables

**Ratios and variables without a monetary unit only** (no raw amount), **no gender**, **no credit history**, **no region rating**, no column an Algerian advisor could not verify.

- **Bank model without history (52 features):**
  - profile: family status, children, family members, education, housing, accompanied by, car (and its age), real estate;
  - employment: income type, occupation (`Missing` category when unknown), employer's sector, employment length, `IS_PENSIONER`, employment length / age;
  - time: age, years since registration, ID document age;
  - loan ratios: credit / income, annuity / income, credit / annuity (`CREDIT_TERM`), goods price / credit;
  - flags: 6 contact flags, 6 address-consistency flags, 18 document flags.
- **Client model (7 features):** `AGE_YEARS`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `NAME_INCOME_TYPE`, `EMPLOYMENT_YEARS`, `IS_PENSIONER`, `CREDIT_INCOME_RATIO`. All of them can be computed from the public form. Age is kept deliberately (see the brief's Decision log and E15).
- **Excluded:** `CODE_GENDER` (kept only for the fairness audit), `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`, `*_CNT_SOCIAL_CIRCLE` (history), `REGION_*` ratings, raw amounts, `DAYS_LAST_PHONE_CHANGE`; `CREDIT_TERM` and `ANNUITY_INCOME_RATIO` are excluded from the client model because they carry Home Credit's pricing (E10).

---

## 4. Model

- **Algorithm:** LightGBM, `class_weight="balanced"`, hyperparameters tuned with Optuna (E13; 50 trials, objective = 5-fold CV AUC on train).
  - bank: 392 trees, `num_leaves` 52, `max_depth` 11, `learning_rate` 0.026;
  - client: 218 trees, `num_leaves` 33, `max_depth` 3, `learning_rate` 0.042.
- **Preprocessing** (fitted on train only): median / most-frequent imputation (`Missing` category for occupation), capping of continuous variables at the 1st-99th percentiles, one-hot encoding (unknown categories ignored).
- **Calibration:** Platt scaling fitted on out-of-fold train predictions (E8).
- **Explainability:** SHAP `TreeExplainer` on the raw log-odds (additivity checked: base + contributions = raw output).
- **Software:** scikit-learn 1.9.1, LightGBM 4.7.0, pandas 3.1.1, numpy 2.5.4, shap 0.54.1. Loading requires the `src` package of this repository.

---

## 5. Final performance (test set, E16)

| Metric | Bank without history | Client |
|---|---|---|
| ROC-AUC [95% CI] | **0.7067** [0.6979, 0.7156] | **0.6188** [0.6091, 0.6290] |
| ROC-AUC on validation | 0.7091 | 0.6207 |
| PR-AUC (random = 0.0835) | 0.1879 | 0.1246 |
| Brier score (calibrated) | 0.07274 | 0.07544 |
| Mean calibrated probability (actual rate 0.0835) | 0.0840 | 0.0837 |
| ROC-AUC, clients without a credit-bureau record (E5) | 0.6962 [0.6746, 0.7182] | 0.6241 [0.6032, 0.6470] |
| ROC-AUC, 20-35 year-olds (E6) | 0.6745 [0.6590, 0.6893] | 0.5774 [0.5610, 0.5934] |
| ROC-AUC, `Commercial associate` (E6) | 0.6942 | 0.5971 |

For comparison, the reference bank model **with** history reaches 0.7657 on the test set: the cost of the missing history is **+0.0590** AUC [+0.0527, +0.0658] (E2).

---

## 6. Calibration and classes (test set)

Thresholds on the calibrated probability, fixed in advance (E14): low < 5% · medium 5-12% · high ≥ 12%.

| Class | Bank: share / actual default rate | Client: share / actual default rate |
|---|---|---|
| Low — accept / favorable | 34.26% / 3.14% | 11.11% / 3.77% |
| Medium — accept with conditions / to strengthen | 44.79% / 7.91% | 72.87% / 7.78% |
| High — refuse / fragile | 20.94% / 17.80% | 16.02% / 14.09% |

The calibrated probabilities match the observed default rates (calibration curve in `reports/figures/19_final_calibration_val_vs_test.png`).

---

## 7. Fairness (test set, E9; acceptance = class "low")

| Group | Bank: acceptance / FNR / FPR | Client: "favorable" / FNR / FPR |
|---|---|---|
| Women | 37.39% / 15.30% / 60.90% | 12.30% / 5.13% / 87.14% |
| Men | 28.32% / 9.76% / 69.49% | 8.85% / 4.88% / 90.69% |
| 20-25 | 8.97% / 1.06% / 89.72% | 0.00% / 0.00% / 100.00% |
| 25-35 | 19.15% / 6.75% / 79.31% | 0.49% / 0.19% / 99.48% |
| 35-45 | 31.65% / 11.45% / 66.41% | 5.80% / 2.67% / 93.90% |
| 45-55 | 37.96% / 15.27% / 60.29% | 14.57% / 8.07% / 84.94% |
| 55-70 | 52.62% / 29.37% / 46.02% | 26.42% / 17.27% / 73.05% |

FNR = share of defaulters placed in class "low"; FPR = share of non-defaulters not placed in class "low". Gender is not a feature. The gaps follow the actual default rates (higher for men and for young applicants) through correlated variables, especially age and employment length.

---

## 8. Limitations

- **Not Algerian data**: proof of concept; real use requires retraining and validation on local data.
- **Client model is weak** (test AUC 0.6188) and weaker still for part of the target: 20-35 year-olds 0.5774, 20-25 year-olds 0.5166 (close to random). It is presented as a way to understand what banks look at, never as "your chances".
- **Young applicants**: the bank model is also less accurate for 20-35 year-olds (0.6745 on the test set, 0.7047 on validation) and accepts them much less often. Young applicants are almost never "favorable" in the client model; removing age would not change this (E15). For 20-25 year-olds the test AUC is 0.6345 (bank) and 0.5166 (client) on a small group (1,327 clients, 189 defaults): **below 25, the application shows a stronger caution message** on both pages (`low_reliability_age_threshold` in the artifact metadata, `low_reliability_warning` in the output of `src/predict.py`).
- **Pricing signal**: the bank model uses the real annuity (`CREDIT_TERM`, `ANNUITY_INCOME_RATIO`), which at Home Credit includes risk-based pricing (worth +0.030 AUC, E11). A bank that does not price according to risk could get a performance closer to 0.68.
- **Clients without a credit-bureau record**: bank model AUC 0.6962 on the test set vs 0.7067 overall (within the CI; 0.7174 on validation).
- **Under-represented target**: no freelancer or self-employed category in the data; "freelancer" and "self-employed" are mapped to `Commercial associate`; "unemployed" gets no score (15 examples).
- **Raw scores are not probabilities**: only the calibrated output (`predict_proba` of the artifact) must be shown.
