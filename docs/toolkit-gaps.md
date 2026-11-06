# What this audit needed that `cmvt` v0.1.0 does not provide

These gaps were found by building the audit, not by reading the toolkit. Each is a small,
testable addition, and each is implemented in this repository in the meantime — which is why
`src/fraud_audit/metrics.py` and `recalibrate.py` exist at all rather than being thin wrappers.

| # | Gap | Where it is implemented here | Why it was needed |
|---|---|---|---|
| 1 | **Calibration slope, intercept (CITL) and O/E** with confidence intervals. `calibration_summary` returns Brier / ECE / base rate / mean prediction only. | `metrics.calibration_metrics` | These three are the *verdict*. Brier and ECE are averages that can hide a level or slope problem entirely. |
| 2 | **Equal-mass ECE.** `expected_calibration_error` uses equal-width bins. | `metrics.ece_equal_mass` | At a 0.2% event rate nearly every score falls in bin 0 and ECE collapses to roughly `\|mean(p) - mean(y)\|`. Both are reported so the difference is visible. |
| 3 | **Logit-scale recalibration.** `recalibrate(method="platt")` fits the logistic map on the raw probability. | `recalibrate.py` | On the logit it is the standard slope-and-intercept update; an intercept-only level is the right choice when events are few. |
| 4 | **Scaling.** `shift_report` calls `energy_distance`, whose large-sample fallback is a Python loop over one sample; `bootstrap_ci` with no `group_id` builds one index array per row. | `drift.py` calls `population_stability_index` directly; `metrics.bootstrap_statistic` and `money.bootstrap_totals` | At 100k+ rows per period the toolkit paths are impractical. A vectorised no-cluster path in `bootstrap_ci` would fix the second. |
| 5 | **A temporal-split helper** that enforces period ordering. | `split.temporal_split` + `TemporalSplit.check_ordering` | The notebook asserted it by hand. It is the single invariant the whole audit rests on, so it is now enforced and tested. |
| 6 | **A money module** — provisioning ratio, cost-based review rule, tuned-cutoff alert drift. | `money.py` | Needed if model-risk use outside healthcare is a goal. Calibration is abstract until a decision depends on the level of a probability. |

## Things the toolkit got right and are used unchanged

* `cmvt.metrics.delong_auc_var` — AUROC variance, validated against `sklearn.roc_auc_score`.
* `cmvt.metrics.delong_unpaired_test` — the correct test for two *different* cohorts. Using the
  paired test across periods would understate the variance; the toolkit makes the right one easy.
* `cmvt.metrics.expected_calibration_error` — reported alongside the equal-mass variant.
* `cmvt.integrity.leakage_scan` — the decidable checks, with one annotated as uninformative on
  this dataset (no entity id exists, so "no entity in both splits" is trivially true).
* `cmvt.shift.population_stability_index` — per-feature and score-distribution PSI.
