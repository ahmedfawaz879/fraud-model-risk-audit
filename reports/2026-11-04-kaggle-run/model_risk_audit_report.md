# Model risk audit: Gradient boosting fraud model

*Generated 2026-11-04 · data `creditcard.csv` (sha256 `76274b691b16a6c4`...) · cmvt 0.1.0 · seed 42*

## 1. Verdict

| Dimension | Result | Evidence (out-of-time unless stated) |
|---|---|---|
| Ranking | **HELD** | AUROC 0.958 in-time [0.925, 0.991] to 0.984 out-of-time [0.973, 0.994]; change +0.025, unpaired DeLong p = 0.148 |
| Calibration | **UNCERTAIN (interval straddles the tolerance)** | O/E 0.96 [0.79, 1.16], slope 1.02 [0.93, 1.11], intercept -0.12; in-time O/E 1.15, slope 0.96 |
| Money | realised loss was **0.45x** the loss implied by the probabilities | 95% interval [0.19, 0.99] (the interval excludes 1.0); net savings of the review rule 11,065 (raw) vs 11,260 (recalibrated) of at most 14,232 |
| Recalibration | **not sufficient** | after: O/E 1.13, slope 1.06; gain +194 [88, 295] (distinguishable from zero) |

**Recommendation.** The evidence is not decisive: gather more out-of-time events before acting. Recalibration is cheap and reversible if needed. Caution: the recalibration window holds only 34 events. Calibration was already off in-time, so part of the problem is model fitting (over-fitting is the usual cause), not only drift.

Tolerances (declared before scoring, illustrative): ranking is HELD if AUROC falls by at most 0.02, DEGRADED if it falls by more and the fall is significant, INCONCLUSIVE if it falls by more but is not significant. Calibration is WITHIN tolerance only if the whole 95% interval of O/E (0.8, 1.25) and of the slope (0.8, 1.2) lies inside the band, OUTSIDE if either interval lies wholly outside, otherwise UNCERTAIN.

## 2. Scope and data

284,807 transactions spanning 48.0 hours, 492 frauds. 1081 exact duplicate rows removed before splitting.
Split by time: development 156,049 rows (332 frauds), recalibration window 42,560 (34), out-of-time test 85,117 (107).
In-time number = random 20% hold-out inside the development period. Features: V1-V28 and log(Amount); `Time` excluded.
Integrity: 0 of 4 decidable checks failed (the patient-overlap check is uninformative without an entity id); exact feature duplicates of training rows found in the out-of-time set: 226.

## 3. Findings

**Ranking.** The model ranks transactions about as well later as it did in-time. Average precision 0.825 in-time to 0.812 out-of-time (no-skill baseline 0.0013).

**Calibration.** The intervals straddle the tolerance, so probabilities can be called neither inside nor outside it. Out-of-time the model over-predicts fraud (O/E 0.96) and is under-confident (slope 1.02).
Equal-count ECE 0.00034 in-time vs 0.00014 out-of-time (an average over bins, so it can hide a level or slope problem; the verdict rests on O/E and slope).

**What changed.** The fraud base rate moved (0.213% to 0.126%, p = 0.000); inputs moved (max feature PSI 1.52 on V1; a classifier separates the periods with AUROC 0.99).

**Money.** At a review cost of 5 per alert, realised fraud loss out-of-time was 14,565; the probabilities implied 32,287. The review rule saved 11,065 against no model (77.8% of the oracle bound); with recalibrated probabilities 11,260.
A fixed score cutoff tuned in-time produced 157 alerts per 100k transactions in-time and 101 out-of-time.

## 4. Remediation

Logistic recalibration (slope and intercept) fitted on the 42,560-row recalibration window (34 events). It is monotone, so AUROC is unchanged (0.9836 vs 0.9836).
The recalibration window holds fewer than 100 events, so the fitted slope and intercept are imprecise.
Isotonic and intercept-only variants are in the notebook as sensitivity checks.

## 5. Limitations

* **Horizon.** 48 hours of data: this tests short-term drift only, not seasonal or adversarial drift over months.
* **No entity id.** Cards cannot be identified, so bootstrap intervals resample rows; if one card has many rows the intervals are too narrow.
* **Anonymised features.** V1-V28 are PCA components; drift cannot be explained in business terms.
* **Few events.** 107 out-of-time frauds; dollar-weighted quantities depend on a handful of large frauds and carry wide intervals.
* **Cost model is an assumption** (review cost 5; reviewed fraud always stopped; no customer friction). See the sensitivity table.
* **Lightly regularised, not tuned.** Fixed settings plus early stopping on the fit rows, no hyper-parameter search; any residual over-fitting would show up as miscalibration in-time as well.
* **One split, one seed.** Optimisation variance of the models is not included.
* **Tolerances are illustrative**; the model-risk owner should set them.

## 6. Reproducibility

Python 3.13.18, numpy 2.1.3, pandas 2.3.3; every random choice is seeded with 42; the configuration block in the notebook is the full parameter set.
