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
| E1 | Baseline: logistic regression vs LightGBM | planned |
| E2 | Cost of the history (main contribution) | planned |
| E3 | Client model viability | planned |
| E4 | Region features | planned |
| E5 | Real no-history profiles | planned |
| E6 | Target population | planned |
| E7 | Sensitivity to the assumed interest rate | planned |
| E8 | Calibration | planned |
| E9 (bonus) | Fairness audit | planned |

---

## E1 — Baseline

| Field | Content |
|---|---|
| **ID** | E1 |
| **Question** | Does LightGBM do better than a simple logistic regression on the bank features without history? |
| **Comparison** | Logistic regression vs LightGBM, both on `BANK_NO_HISTORY_FEATURES` |
| **Metric** | AUC-ROC on the validation set |
| **Decision rule (fixed in advance)** | LightGBM must beat the logistic regression in validation AUC; otherwise, investigate before going further |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E2 — Cost of the history (main contribution)

| Field | Content |
|---|---|
| **ID** | E2 |
| **Question** | How much accuracy does the bank model lose without credit history? |
| **Comparison** | Bank model with history (`BANK_WITH_HISTORY_FEATURES`) vs bank model without history (`BANK_NO_HISTORY_FEATURES`) |
| **Metric** | AUC gap, with a bootstrap confidence interval |
| **Decision rule (fixed in advance)** | None: this is a measurement to report |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

---

## E3 — Client model viability

| Field | Content |
|---|---|
| **ID** | E3 |
| **Question** | Is the client model, built only from what the public form provides, good enough to be shown? |
| **Comparison** | Client model (`CLIENT_FEATURES`) against the thresholds below |
| **Metric** | AUC-ROC on the validation set |
| **Decision rule (fixed in advance)** | ≥ 0.65 → kept; between 0.60 and 0.65 → kept with a stronger caution message; < 0.60 → dropped |
| **Status** | planned |
| **Result** | |
| **Decision taken** | |

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
