# Noria — Experiment log

The project's lab notebook. Each planned measurement is written here **before** it is run, with its decision rule fixed in advance, then completed with the actual result.

Reference: [project-brief.md](project-brief.md) (choices and decision log).

## Rules

- Any new experiment is added here before it is run.
- Selection metrics are computed on the **validation** set. The **test** set is used only once, at the end, for the final results.
- A decision rule is never changed after seeing the result. If it must change, the change and its reason are written in the Decision log of [project-brief.md](project-brief.md) (section 12).
- **Adoption rule (from session 5, 18/11/2026):** any decision to **adopt** a change (feature, tuning, method) requires the paired bootstrap 95% CI on the validation set to have a **lower bound > 0** (improvement direction of the metric concerned). It is not applied retroactively: the decisions of E1-E13 stay as recorded.

## Summary

| ID | Experiment | Status |
|---|---|---|
| E1 | Baseline: logistic regression vs LightGBM | done |
| E2 | Cost of the history (main contribution) | done (updated after E13) |
| E3 | Client model viability | done (superseded by E10) |
| E4 | Region features | done |
| E5 | Real no-history profiles | done on validation (test pending) |
| E6 | Target population | done |
| E7 | Sensitivity to the assumed interest rate | cancelled (since E10) |
| E8 | Calibration | done |
| E9 (bonus) | Fairness audit | done |
| E10 | Client model without the pricing signal | done |
| E11 | Weight of the pricing signal in the bank model | done |
| E12 | New features | done |
| E13 | Hyperparameter tuning (Optuna) | done |
| E14 | Risk classes | done |
| E15 | Cost and effect of removing age from the client model | planned |

---

## E1 — Baseline

| Field | Content |
|---|---|
| **ID** | E1 |
| **Question** | Does LightGBM do better than a simple logistic regression on the bank features without history? |
| **Comparison** | Logistic regression vs LightGBM, both on `BANK_NO_HISTORY_FEATURES` |
| **Metric** | AUC-ROC on the validation set |
| **Decision rule (fixed in advance)** | LightGBM must beat the logistic regression in validation AUC; otherwise, investigate before going further |
| **Status** | done (session 3, 18/11/2026) |
| **Result** | Validation AUC: logistic regression 0.6698 [0.6594, 0.6789] vs LightGBM **0.7093** [0.6993, 0.7184] (95% bootstrap CI, 1,000 resamples). Paired difference LightGBM − logreg: +0.0396 [+0.0338, +0.0459]. PR-AUC 0.1641 vs 0.1947. Details: `reports/metrics/e1.json`, notebook 03 |
| **Decision taken** | Rule satisfied: LightGBM beats the logistic regression. LightGBM stays the main model, no investigation needed |

---

## E2 — Cost of the history (main contribution)

| Field | Content |
|---|---|
| **ID** | E2 |
| **Question** | How much accuracy does the bank model lose without credit history? |
| **Comparison** | Bank model with history (`BANK_WITH_HISTORY_FEATURES`) vs bank model without history (`BANK_NO_HISTORY_FEATURES`) |
| **Metric** | AUC gap, with a bootstrap confidence interval |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Status** | done (session 3, 18/11/2026) |
| **Result** | Validation AUC: with history **0.7663** [0.7579, 0.7742] vs without history **0.7093** [0.6993, 0.7184]. Gap with − without: **+0.0569**, paired bootstrap 95% CI [+0.0502, +0.0647] (1,000 resamples, same resamples for both models). PR-AUC 0.2592 vs 0.1947. Details: `reports/metrics/e2.json`, notebook 03 |
| **Decision taken** | Measurement reported (no decision rule): losing the credit history costs about 0.057 AUC |
| **Updated after E13** | With the two tuned bank models (session 4 lists, `PHONE_CHANGE_YEARS` removed): with history 0.7663 vs without history 0.7091, gap **+0.0572**, paired bootstrap 95% CI [+0.0509, +0.0644]. The original result above is unchanged. Details: `reports/metrics/e2_after_e13.json` |

---

## E3 — Client model viability

| Field | Content |
|---|---|
| **ID** | E3 |
| **Question** | Is the client model, built only from what the public form provides, good enough to be shown? |
| **Comparison** | Client model (`CLIENT_FEATURES`) against the thresholds below |
| **Metric** | AUC-ROC on the validation set |
| **Decision rule (fixed in advance)** | ≥ 0.65 → kept; between 0.60 and 0.65 → kept with a stronger caution message; < 0.60 → dropped |
| **Status** | done (session 3, 18/11/2026) |
| **Result** | Validation AUC of the LightGBM client model: **0.6883** [0.6780, 0.6979], PR-AUC 0.1708. Logistic regression (indicative): 0.6181. Note: `CREDIT_TERM` carries 50.07% of the gain and most likely encodes Home Credit's risk-based pricing, which the public form (assumed interest rate) cannot reproduce; this AUC is therefore probably optimistic for real use (notebook 03, section 5.1). Details: `reports/metrics/e3.json` |
| **Decision taken** | ≥ 0.65: the client model is **kept** (rule applied as written) |
| **Note** | **Superseded by E10** (pre-registered 18/11/2026): E10, not E3, decides the fate of the client model |

---

## E4 — Region features

| Field | Content |
|---|---|
| **ID** | E4 |
| **Question** | What do the Home Credit region ratings add to the bank model without history? |
| **Comparison** | Bank model without history with vs without `REGION_FEATURES` |
| **Metric** | AUC gain |
| **Decision rule (fixed in advance)** | Whatever the result, they are not used in Noria (transferability and fairness); the measured gain is reported as a lead for a real bank |
| **Protocol (pre-registered 18/11/2026, before running)** | Validation set. Final bank model without history (52 features, tuned parameters of E13) vs the same list + `REGION_FEATURES` (55 features) with the **same tuned parameters**, number of trees by early stopping in a 5-fold CV on train. AUC gain with a paired bootstrap 95% CI |
| **Status** | done (session 5, 18/11/2026) |
| **Result** | Validation AUC without `REGION_FEATURES` 0.7091 [0.6990, 0.7177] vs with them **0.7138** [0.7042, 0.7229] (55 features, tuned parameters, 463 trees). Gain **+0.0047**, paired bootstrap 95% CI [+0.0021, +0.0073]. Details: `reports/metrics/e4.json`, notebook 05 section 3 |
| **Decision taken** | As pre-registered, **not used in Noria** (transferability and fairness); the +0.0047 gain is reported as a lead for a real bank with its own validated regional data |

---

## E5 — Real no-history profiles

| Field | Content |
|---|---|
| **ID** | E5 |
| **Question** | Does the model perform as well on clients who really have no credit history? |
| **Comparison** | Performance on the test clients absent from `bureau.csv` vs overall performance (`bureau.csv` is used only to identify this subgroup) |
| **Metric** | Performance on the subgroup vs overall |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Protocol (pre-registered 18/11/2026, before running)** | Session 5 measures it on the **validation** set: validation clients whose `SK_ID_CURR` is absent from `data/raw/bureau.csv` vs all validation clients, for the bank model without history, the bank model with history and the client model (final session 4 models), AUC with 95% bootstrap CI. `bureau.csv` is only read to get its `SK_ID_CURR` list, never as a feature. **The final measurement on the test set will be made at the final-evaluation session** |
| **Status** | done on validation (session 5, 18/11/2026); test measurement pending (final-evaluation session) |
| **Result** | Validation: 5,861 clients (14.0%) absent from `bureau.csv`, default rate 10.82% vs 7.94% for the others. AUC on this subgroup vs all validation clients: bank without history **0.7174** [0.6958, 0.7396] vs 0.7091; bank with history 0.7555 [0.7347, 0.7741] vs 0.7663; client 0.6240 [0.6001, 0.6474] vs 0.6207. The gap with − without history shrinks on these clients (≈ 0.038 vs 0.057 overall). Details: `reports/metrics/e5.json`, notebook 05 section 4 |
| **Decision taken** | Measurement reported (no decision rule): the model without history performs at least as well on real no-history profiles. **The final measurement on the test set remains to be done at the final-evaluation session** |

---

## E6 — Target population

| Field | Content |
|---|---|
| **ID** | E6 |
| **Question** | Does the model remain usable for the target population (freelancers, young workers)? |
| **Comparison** | AUC by `NAME_INCOME_TYPE` (including `Commercial associate`, the profile closest to freelancers) and by age group (including 20-35) |
| **Metric** | AUC-ROC per group |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Protocol (pre-registered 18/11/2026, before running)** | Validation set, final session 4 models, bank without history and client. Groups: each `NAME_INCOME_TYPE` category, and age groups 20-25, 25-35, 35-45, 45-55, 55-70 plus the aggregate 20-35. Per group: rows, defaults, AUC with 95% bootstrap CI (1,000 resamples). A group with fewer than 50 defaults or fewer than 50 non-defaults is flagged **too small to conclude** |
| **Status** | done (session 5, 18/11/2026) |
| **Result** | Validation AUC [95% CI]. Bank without history (overall 0.7091): `Commercial associate` 0.6944 [0.6740, 0.7127], `Working` 0.7066, `State servant` 0.7044, `Pensioner` 0.6571; ages 20-25 0.6995, 25-35 0.7043, 35-45 0.6970, 45-55 0.6925, 55-70 0.6704, **20-35 0.7047** [0.6905, 0.7195]. Client (overall 0.6207): `Commercial associate` **0.5891** [0.5687, 0.6093], `Working` 0.6069, `State servant` 0.5687, `Pensioner` 0.5550; within each age group 0.578-0.592, **20-35 0.5921** [0.5752, 0.6091]. `Other` (6 rows) too small to conclude; every other group has ≥ 50 defaults and ≥ 50 non-defaults. Details: `reports/metrics/e6.json`, notebook 05 section 5 |
| **Decision taken** | Measurement reported (no decision rule): the bank model stays usable for the target groups (0.69-0.71); the client model is **below 0.60** within `Commercial associate` (0.5891) and the 20-35 group (0.5921) |

---

## E7 — Sensitivity to the assumed interest rate

| Field | Content |
|---|---|
| **ID** | E7 |
| **Question** | Does the client category depend on the assumed interest rate used to compute the annuity? |
| **Comparison** | Client category obtained with an assumed rate of 5%, 10% and 15% |
| **Metric** | Share of profiles whose category changes between 5% and 15% |
| **Decision rule (fixed in advance)** | If more than 10% of profiles change category between 5% and 15%, report it as a limitation in the client interface |
| **Status** | cancelled (18/11/2026) |
| **Result** | Not run |
| **Decision taken** | Cancelled: Since E10, the client model no longer uses any variable that depends on the annuity (`CREDIT_TERM` and `ANNUITY_INCOME_RATIO` removed), so the assumed interest rate has no effect on the score |

---

## E8 — Calibration

| Field | Content |
|---|---|
| **ID** | E8 |
| **Question** | Which calibration method makes the displayed probabilities match the actual default rates? |
| **Comparison** | Calibration curve before and after calibration: isotonic vs Platt |
| **Metric** | Brier score on the validation set (and the calibration curve) |
| **Decision rule (fixed in advance)** | Keep the method with the best Brier score on the validation set |
| **Protocol (pre-registered 18/11/2026, before running)** | Final session 4 models (52, 65 and 7 features, tuned parameters). The calibrators (isotonic regression and Platt scaling, i.e. a logistic regression on the log-odds of the raw score) are fitted on the **out-of-fold predictions of the train set** (5-fold stratified CV, `random_state = 42`, same parameters and number of trees as the final model), **never on the validation set**. They are then applied to the final model's validation predictions; calibration curves and Brier scores are computed on the validation set. The rule above is unchanged. Under the adoption rule (top of this file), calibrating at all is adopted only if the paired bootstrap CI of the Brier improvement of the selected method over the uncalibrated model has a lower bound > 0 |
| **Status** | done (session 5, 18/11/2026) |
| **Result** | Raw scores are far too high (mean 0.435 / 0.396 / 0.480 vs an observed rate of 0.0835, because of `class_weight="balanced"`). Validation Brier, uncalibrated / isotonic / Platt: bank without history 0.20732 / 0.07253 / **0.07251**; bank with history 0.18584 / 0.06906 / **0.069057**; client 0.23762 / 0.075384 / **0.075381**. Platt minus isotonic: at most 0.00002, paired CIs including 0 (e.g. [−0.000006, +0.000048]). Brier improvement of Platt over uncalibrated: +0.1348 [+0.1324, +0.1370], +0.1168 [+0.1145, +0.1191], +0.1622 [+0.1599, +0.1643]. Mean calibrated probability 0.0843 / 0.0843 / 0.0834. Details: `reports/metrics/e8.json`, notebook 05 section 1 |
| **Decision taken** | **Platt selected for the three models** (best validation Brier, rule applied as written; the margin over isotonic is negligible). Calibration **adopted** under the adoption rule (CI lower bounds > 0). Calibrated models saved as `models/calibrated__lgbm__<list>.joblib` |

---

## E9 (bonus) — Fairness audit

| Field | Content |
|---|---|
| **ID** | E9 |
| **Question** | Does the model treat groups differently by sex and by age? |
| **Comparison** | Groups by sex and by age group (Fairlearn) |
| **Metric** | Acceptance rates and error rates per group |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Protocol (pre-registered 18/11/2026, before running)** | On the validation set, with the **calibrated** probabilities and the E14 classes, for the bank model without history and the client model. Acceptance = class "low" (< 5%). Positive = default (`TARGET = 1`); a client is predicted "risky" when not in class "low". Measures per group (Fairlearn `MetricFrame`): acceptance rate, false negative rate (defaulters placed in class "low", i.e. accepted) and false positive rate (non-defaulters not placed in class "low"), by `CODE_GENDER` and by age group (bins of E6) |
| **Status** | done (session 5, 18/11/2026) |
| **Result** | Acceptance = class "low"; validation; calibrated probabilities. Bank without history by sex: acceptance F **36.73%** vs M **28.27%**, false negative rate F 15.06% vs M 10.74%, false positive rate F 61.60% vs M 69.64%. By age: acceptance 10.00% (20-25), 18.98%, 31.28%, 36.72%, **52.72% (55-70)**; false positive rate from 88.66% to 46.04%. Client by sex: acceptance F 12.69% vs M 8.91%; by age: **0.00% (20-25)**, **0.42% (25-35)**, 6.20%, 14.70%, 27.29% (55-70). Observed validation default rates: F 7.16% vs M 10.63%, 20-25 13.14% vs 55-70 5.20%. Details: `reports/metrics/e9.json`, notebook 05 section 6 |
| **Decision taken** | Measurement reported (no decision rule): gender is not a feature, but acceptance differs by sex (8.5 points in the bank model) and strongly by age, following the observed default rates through correlated variables; young applicants are almost never "favorable" in the client model |

---

## E10 — Client model without the pricing signal

| Field | Content |
|---|---|
| **ID** | E10 |
| **Question** | Is the client model still good enough without the variables that carry Home Credit's risk-based pricing? |
| **Comparison** | LightGBM on `CLIENT_FEATURES` without `CREDIT_TERM` and `ANNUITY_INCOME_RATIO`. Both contain Home Credit's risk-based pricing (the annuity includes the interest rate set for the applicant), which is absent from the public form, where the annuity is computed with a fixed assumed rate. Same protocol as E1-E3 |
| **Metric** | AUC-ROC on the validation set (95% bootstrap CI) |
| **Decision rule (fixed in advance)** | Identical to E3: ≥ 0.65 → kept; between 0.60 and 0.65 → kept with a stronger caution message; < 0.60 → dropped. **E10, not E3, decides the fate of the client model**, and if the model is kept, the E10 feature list becomes `CLIENT_FEATURES` |
| **Status** | done (18/11/2026) |
| **Result** | Validation AUC of LightGBM on the 7 client features without `CREDIT_TERM` and `ANNUITY_INCOME_RATIO`: **0.6189** [0.6092, 0.6291] (95% bootstrap CI, 1,000 resamples), PR-AUC 0.1258, 43 trees. For reference, E3 with the pricing signal: 0.6883. Top features (share of gain): `EMPLOYMENT_YEARS` 45.41%, `AGE_YEARS` 25.90%, `CREDIT_INCOME_RATIO` 12.55%. Details: `reports/metrics/e10.json`, notebook 03 section 6 |
| **Decision taken** | 0.60 ≤ 0.6189 < 0.65: the client model is **kept with a stronger caution message** (rule applied as written). The E10 feature list becomes `CLIENT_FEATURES` (7 features) |

---

## E11 — Weight of the pricing signal in the bank model

| Field | Content |
|---|---|
| **ID** | E11 |
| **Question** | How much of the bank model's performance comes from the pricing signal? |
| **Comparison** | LightGBM on `BANK_NO_HISTORY_FEATURES` with vs without `CREDIT_TERM` and `ANNUITY_INCOME_RATIO`. Same protocol as E1-E3 |
| **Metric** | AUC difference, paired bootstrap 95% CI |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Status** | done (18/11/2026) |
| **Result** | Validation AUC of LightGBM bank without history: with `CREDIT_TERM` and `ANNUITY_INCOME_RATIO` **0.7093** [0.6993, 0.7184] vs without **0.6795** [0.6700, 0.6883]. Difference with − without: **+0.0298**, paired bootstrap 95% CI [+0.0249, +0.0349] (1,000 resamples). PR-AUC 0.1947 vs 0.1690. Details: `reports/metrics/e11.json`, notebook 03 section 6 |
| **Decision taken** | Measurement reported (no decision rule): the pricing signal is worth about 0.030 AUC in the bank model without history |

---

## E12 — New features

| Field | Content |
|---|---|
| **ID** | E12 |
| **Question** | Do new deterministic features improve the three models? |
| **Comparison** | LightGBM with vs without the new features, for `BANK_NO_HISTORY_FEATURES`, `BANK_WITH_HISTORY_FEATURES` and `CLIENT_FEATURES`. Candidates (deterministic, no raw amount, no history, no regional rating, verifiable by an advisor): `DOC_COUNT` (sum of the `FLAG_DOCUMENT_*`), `CONTACT_COUNT` (sum of the contact flags: `FLAG_MOBIL`, `FLAG_EMP_PHONE`, `FLAG_WORK_PHONE`, `FLAG_CONT_MOBILE`, `FLAG_PHONE`, `FLAG_EMAIL`), `AGE_AT_EMPLOYMENT_START` (`AGE_YEARS` − `EMPLOYMENT_YEARS`), `CHILDREN_RATIO` (`CNT_CHILDREN` / `CNT_FAM_MEMBERS`), `ID_PUBLISH_AGE_RATIO` and `REGISTRATION_AGE_RATIO` (`ID_PUBLISH_YEARS` and `REGISTRATION_YEARS` divided by `AGE_YEARS`). Bank lists: all six candidates. Client list: only the candidates computable from the public form, i.e. `AGE_AT_EMPLOYMENT_START` (age and employment length are in the form) and `CHILDREN_RATIO` (the family size is deducible from the form: `CNT_FAM_MEMBERS` = 1 + spouse (`Married` or `Civil marriage`) + `CNT_CHILDREN` on 100% of train rows, checked before this registration). The reference models "without new features" are re-measured with the lists corrected by the session 4 preliminary decision (`PHONE_CHANGE_YEARS` removed). Same protocol as E1-E3 (default LightGBM parameters, early stopping in a 5-fold stratified CV on train) |
| **Metric** | Mean 5-fold CV AUC on train, and validation AUC difference with − without (paired bootstrap 95% CI, 1,000 resamples) |
| **Decision rule (fixed in advance)** | For each list, the new features are kept if the mean 5-fold CV AUC on train increases **and** the lower bound of the paired bootstrap CI on validation is > 0. Otherwise they are dropped for that list |
| **Status** | done (session 4, 18/11/2026) |
| **Result** | Reference lists after the preliminary decision. Bank without history (52 → 58 features): mean CV AUC 0.7005 → 0.6999 (−0.0006); validation 0.7058 → 0.7056, paired difference −0.0002 [−0.0023, +0.0019]. Bank with history (65 → 71): CV 0.7615 → 0.7617 (+0.0002); validation 0.7641 → 0.7652, +0.0011 [−0.0003, +0.0027]. Client (7 → 9): CV 0.6190 → 0.6188 (−0.0002); validation 0.6189 → 0.6190, +0.0001 [−0.0012, +0.0014]. The new features carry at most 2.82% of the gain in the bank models; in the client model `AGE_AT_EMPLOYMENT_START` carries 7.63% without improving the AUC. Details: `reports/metrics/e12.json`, notebook 04 section 1 |
| **Decision taken** | **Dropped for all three lists** (rule applied as written: the CV AUC does not increase for bank without history and client; for bank with history the CV AUC increases but the CI lower bound is not > 0). Final lists unchanged: 52, 65 and 7 features |

---

## E13 — Hyperparameter tuning (Optuna)

| Field | Content |
|---|---|
| **ID** | E13 |
| **Question** | Does tuning the LightGBM hyperparameters improve the three models? |
| **Comparison** | For each final list from E12: tuned parameters vs default parameters (`LGBM_PARAMS` in `src/train.py`). Optuna, TPE sampler with a fixed seed (42), 50 trials per list. Objective: mean AUC of a 5-fold stratified CV on the **train** set (never the validation set), early stopping kept. Search space: `num_leaves` 15-255 (log), `max_depth` 3-12, `learning_rate` 0.02-0.2 (log), `min_child_samples` 20-500 (log), `subsample` 0.5-1.0, `colsample_bytree` 0.3-1.0, `reg_alpha` 0.001-10 (log), `reg_lambda` 0.001-10 (log); `class_weight="balanced"` and `subsample_freq=1` fixed. The number of trees of the final model comes from the CV early stopping, as in E1-E3 |
| **Metric** | Validation AUC (95% bootstrap CI) of the tuned vs default model, per list |
| **Decision rule (fixed in advance)** | The tuned parameters are kept if their validation AUC is ≥ that of the default parameters; otherwise the default parameters are kept. Then E2 is re-measured with the two tuned bank models (paired bootstrap) and the updated gap is reported in E2, without overwriting E2's original result |
| **Status** | done (session 4, 18/11/2026) |
| **Result** | 50 trials per list, TPE seed 42, objective = mean 5-fold CV AUC on train. Bank without history: CV 0.7005 → 0.7048; validation default 0.7058 vs tuned **0.7091** [0.6990, 0.7177], difference +0.0033 [+0.0012, +0.0053]. Bank with history: CV 0.7615 → 0.7640; validation 0.7641 vs **0.7663** [0.7580, 0.7741], +0.0023 [+0.0008, +0.0038]. Client: CV 0.6190 → 0.6206; validation 0.6189 vs **0.6207** [0.6107, 0.6307], +0.0017 [−0.0001, +0.0035]. Best parameters: `reports/metrics/e13_best_params__*.json`; trial histories: `reports/metrics/e13_trials__*.csv`. E2 re-measured with the two tuned bank models: +0.0572 [+0.0509, +0.0644] (see E2). Details: `reports/metrics/e13.json`, notebook 04 section 2 |
| **Decision taken** | Validation AUC tuned ≥ default for the three lists: **tuned parameters kept for all three** (rule applied as written). The client model stays below 0.65, so the E10 decision (kept with a stronger caution message) is unchanged |

---

## E14 — Risk classes

| Field | Content |
|---|---|
| **ID** | E14 |
| **Question** | How do the clients spread across the risk classes, and do the classes separate the actual default rates? |
| **Comparison** | Classes defined on the **calibrated** probability of default (method selected in E8). Bank model without history: low < 5% → recommendation "accept"; medium 5-12% → "accept with conditions"; high ≥ 12% → "refuse" (recommendation only: the advisor decides). Client model: same thresholds → "favorable" / "to strengthen" / "fragile" |
| **Metric** | On the validation set, per class: share of clients and actual default rate |
| **Decision rule (fixed in advance)** | The thresholds are not modified after seeing the results. If one class contains more than 80% of the clients, or if the actual default rates do not increase from low to high, it is reported as a limitation (in the brief and, for the client, in the displayed message), without changing the thresholds |
| **Status** | done (session 5, 18/11/2026) |
| **Result** | Bank without history (Platt): low/accept **33.83%** of clients (actual default rate 3.25%), medium/accept with conditions **45.22%** (7.38%), high/refuse **20.94%** (18.66%). Client (Platt): favorable **11.40%** (4.50%), to strengthen **73.14%** (7.72%), fragile **15.47%** (14.16%). Details: `reports/metrics/e14.json`, notebook 05 section 2 |
| **Decision taken** | Thresholds kept (5% / 12%). Default rates increase from low to high and no class exceeds 80% (largest: client "to strengthen", 73.14%): **no limitation triggered** by the rule |

---

## E15 — Cost and effect of removing age from the client model

| Field | Content |
|---|---|
| **ID** | E15 |
| **Question** | What would removing `AGE_YEARS` cost the client model, and how would it change the categories by age? |
| **Comparison** | Current LightGBM client model (`CLIENT_FEATURES`, 7 features) vs the same without `AGE_YEARS` (6 features). Same tuned parameters as E13 (`e13_best_params__client.json`), number of trees by early stopping in a 5-fold CV on train, same Platt calibration (fitted on out-of-fold train predictions), same E14 thresholds (5% / 12%) |
| **Metric** | On the validation set: overall AUC with 95% bootstrap CI and the paired bootstrap CI of the difference; AUC (95% CI) of the 20-35 group and of `Commercial associate`; share of favorable / to strengthen / fragile per age group (E6 bins), with the actual default rate per class |
| **Decision rule (fixed in advance)** | None: this is a measurement to report. The decision to keep age in the client model is already taken (see the Decision log of the brief); `CLIENT_FEATURES` does not change |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |
