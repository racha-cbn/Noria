# Noria — Final results (experiment E16)

Test set: 41,735 cash-loan applications (3,483 defaults, 8.35%), opened **once** on 18/11/2026 with the models frozen in `manifest.json`.
Validation set: 41,735 applications (same default rate). ROC-AUC with 95% bootstrap CI (1,000 resamples).
Source files: `e16_test.json`, `e16_validation.json`, `validation_vs_test.csv` (this folder), notebook `07_final_evaluation.ipynb`.

## 1. Models

| Model | Features | Test ROC-AUC [95% CI] | Validation ROC-AUC | Test PR-AUC | Test Brier (calibrated) |
|---|---|---|---|---|---|
| Bank without history (application) | 52 | **0.7067** [0.6979, 0.7156] | 0.7091 | 0.1879 | 0.07274 |
| Bank with history (reference) | 65 | **0.7657** [0.7579, 0.7741] | 0.7663 | 0.2555 | 0.06934 |
| Client (application) | 7 | **0.6188** [0.6091, 0.6290] | 0.6207 | 0.1246 | 0.07544 |

## 2. Main contribution: cost of the credit history (E2)

| | Test | Validation |
|---|---|---|
| AUC with history − without history (paired bootstrap) | **+0.0590** [+0.0527, +0.0658] | +0.0572 [+0.0509, +0.0644] |

## 3. Clients without a credit-bureau record (E5)

| | Test | Validation |
|---|---|---|
| Share of clients absent from `bureau.csv` | 14.1% (5,880) | 14.0% (5,861) |
| Their default rate vs others | 10.26% vs 8.03% | 10.82% vs 7.94% |
| Bank without history: AUC on them vs all | **0.6962** [0.6746, 0.7182] vs 0.7067 | 0.7174 vs 0.7091 |
| Bank with history: AUC on them vs all | 0.7357 [0.7157, 0.7574] vs 0.7657 | 0.7555 vs 0.7663 |
| Client: AUC on them vs all | 0.6241 [0.6032, 0.6470] vs 0.6188 | 0.6240 vs 0.6207 |
| History advantage on them (with − without) | 0.0395 | 0.0381 |

## 4. Risk classes (E14, calibrated probability; low < 5%, medium 5-12%, high ≥ 12%)

| Class | Bank test: share / default rate | Bank validation | Client test: share / default rate | Client validation |
|---|---|---|---|---|
| Low (accept / favorable) | 34.26% / 3.14% | 33.83% / 3.25% | 11.11% / 3.77% | 11.40% / 4.50% |
| Medium (with conditions / to strengthen) | 44.79% / 7.91% | 45.22% / 7.38% | 72.87% / 7.78% | 73.14% / 7.72% |
| High (refuse / fragile) | 20.94% / 17.80% | 20.94% / 18.66% | 16.02% / 14.09% | 15.47% / 14.16% |

## 5. Target population (E6, test ROC-AUC)

| Group | Bank without history | Client |
|---|---|---|
| `Commercial associate` (closest to freelancers) | 0.6942 | 0.5971 |
| `Working` | 0.7053 | 0.6028 |
| `State servant` | 0.7181 | 0.6267 |
| `Pensioner` | 0.6590 | 0.5695 |
| 20-25 | 0.6345 [0.5955, 0.6759] | 0.5166 [0.4687, 0.5599] |
| 25-35 | 0.6778 | 0.5817 |
| 35-45 | 0.7087 | 0.5932 |
| 45-55 | 0.7136 | 0.6027 |
| 55-70 | 0.6709 | 0.5728 |
| **20-35** | **0.6745** [0.6590, 0.6893] | **0.5774** [0.5610, 0.5934] |

## 6. Fairness (E9, test; acceptance = class "low")

| Group | Bank acceptance | Client "favorable" |
|---|---|---|
| Women / men | 37.39% / 28.32% | 12.30% / 8.85% |
| 20-25 / 25-35 / 35-45 / 45-55 / 55-70 | 8.97% / 19.15% / 31.65% / 37.96% / 52.62% | 0.00% / 0.49% / 5.80% / 14.57% / 26.42% |

Error rates per group are in `docs/model-card.md` (section 7) and `e16_test.json`.
