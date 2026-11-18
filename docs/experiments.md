# Noria — Experiment log

The project's lab notebook. Each planned measurement is written here **before** it is run, with its decision rule fixed in advance, then completed with the actual result.

Reference: [project-brief.md](project-brief.md) (choices and decision log).

## Rules

- Any new experiment is added here before it is run.
- Selection metrics are computed on the **validation** set. The **test** set is used only once, at the end, for the final results.
- A decision rule is never changed after seeing the result. If it must change, the change and its reason are written in the Decision log of [project-brief.md](project-brief.md) (section 12).

## Summary

| ID | Experiment | Status |
|---|---|---|
| E1 | Baseline: logistic regression vs LightGBM | done |
| E2 | Cost of the history (main contribution) | done |
| E3 | Client model viability | done (superseded by E10) |
| E4 | Region features | planned |
| E5 | Real no-history profiles | planned |
| E6 | Target population | planned |
| E7 | Sensitivity to the assumed interest rate | planned |
| E8 | Calibration | planned |
| E9 (bonus) | Fairness audit | planned |
| E10 | Client model without the pricing signal | done |
| E11 | Weight of the pricing signal in the bank model | done |

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
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E5 — Real no-history profiles

| Field | Content |
|---|---|
| **ID** | E5 |
| **Question** | Does the model perform as well on clients who really have no credit history? |
| **Comparison** | Performance on the test clients absent from `bureau.csv` vs overall performance (`bureau.csv` is used only to identify this subgroup) |
| **Metric** | Performance on the subgroup vs overall |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E6 — Target population

| Field | Content |
|---|---|
| **ID** | E6 |
| **Question** | Does the model remain usable for the target population (freelancers, young workers)? |
| **Comparison** | AUC by `NAME_INCOME_TYPE` (including `Commercial associate`, the profile closest to freelancers) and by age group (including 20-35) |
| **Metric** | AUC-ROC per group |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E7 — Sensitivity to the assumed interest rate

| Field | Content |
|---|---|
| **ID** | E7 |
| **Question** | Does the client category depend on the assumed interest rate used to compute the annuity? |
| **Comparison** | Client category obtained with an assumed rate of 5%, 10% and 15% |
| **Metric** | Share of profiles whose category changes between 5% and 15% |
| **Decision rule (fixed in advance)** | If more than 10% of profiles change category between 5% and 15%, report it as a limitation in the client interface |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E8 — Calibration

| Field | Content |
|---|---|
| **ID** | E8 |
| **Question** | Which calibration method makes the displayed probabilities match the actual default rates? |
| **Comparison** | Calibration curve before and after calibration: isotonic vs Platt |
| **Metric** | Brier score on the validation set (and the calibration curve) |
| **Decision rule (fixed in advance)** | Keep the method with the best Brier score on the validation set |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E9 (bonus) — Fairness audit

| Field | Content |
|---|---|
| **ID** | E9 |
| **Question** | Does the model treat groups differently by sex and by age? |
| **Comparison** | Groups by sex and by age group (Fairlearn) |
| **Metric** | Acceptance rates and error rates per group |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

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
