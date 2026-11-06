"""Self-tests for the economics.

The first test is the notebook's hand-computed example: four transactions whose cost under the
review rule can be worked out on paper. If the implementation and the arithmetic ever disagree,
the money section of the report is worthless, so it is checked against the paper answer rather
than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_audit.money import (
    bootstrap_totals,
    cost_at_cutoff,
    money_rows,
    money_summary,
    tune_cutoff,
)

# Four transactions: two frauds (100 and 2), two legitimate (50 and 10).
Y = np.array([1, 0, 1, 0])
AMOUNT = np.array([100.0, 50.0, 2.0, 10.0])
P = np.array([0.20, 0.05, 0.90, 0.40])
REVIEW_COST = 5.0


class TestHandComputedExample:
    """``p * Amount >= 5`` flags only transaction 1 (0.2 * 100 = 20). Fraud 3 slips through at 2."""

    def test_only_the_first_transaction_is_flagged(self):
        flag, _ = money_rows(Y, AMOUNT, P, REVIEW_COST)
        assert flag.tolist() == [True, False, False, False]

    def test_cost_is_the_review_plus_the_missed_fraud(self):
        summary = money_summary(Y, AMOUNT, P, REVIEW_COST)
        assert summary["cost"] == pytest.approx(7.0)  # 5 review + 2 unreviewed fraud
        assert summary["savings"] == pytest.approx(95.0)  # 102 realised loss - 7
        assert summary["realised_loss"] == pytest.approx(102.0)

    def test_the_oracle_would_not_have_reviewed_the_cheap_fraud(self):
        """Reviewing a fraud worth 2 costs 5, so the oracle lets it through: same 95 saving."""
        summary = money_summary(Y, AMOUNT, P, REVIEW_COST)
        assert summary["oracle_savings"] == pytest.approx(95.0)
        assert summary["capture"] == pytest.approx(1.0)


class TestMoneySummary:
    def test_provisioning_ratio_is_realised_over_predicted(self):
        summary = money_summary(Y, AMOUNT, P, REVIEW_COST)
        assert summary["realised_over_predicted"] == pytest.approx(102.0 / float((P * AMOUNT).sum()))

    def test_alerts_per_100k_scales_the_alert_count(self):
        summary = money_summary(Y, AMOUNT, P, REVIEW_COST)
        assert summary["alerts_per_100k"] == pytest.approx(0.25 * 1e5)

    def test_no_fraud_at_all_yields_nan_rather_than_a_division_by_zero(self):
        summary = money_summary(np.zeros(4, dtype=int), AMOUNT, P, REVIEW_COST)
        assert np.isnan(summary["savings_share"])
        assert np.isnan(summary["capture"])


class TestTuneCutoff:
    def test_the_tuned_cutoff_is_at_least_as_cheap_as_any_other(self):
        rng = np.random.default_rng(5)
        n = 2_000
        p = rng.beta(0.4, 8.0, n)
        y = (rng.random(n) < p).astype(int)
        amount = rng.lognormal(3.0, 1.0, n)
        tau, _ = tune_cutoff(y, amount, p, REVIEW_COST)
        best = cost_at_cutoff(y, amount, p, tau, REVIEW_COST)["cost_per_100k"]
        for other in np.quantile(p, [0.5, 0.9, 0.99, 0.999]):
            assert cost_at_cutoff(y, amount, p, float(other), REVIEW_COST)["cost_per_100k"] >= best - 1e-9

    def test_reviewing_nothing_is_reported_as_an_infinite_cutoff(self):
        """No fraud anywhere: every review is pure cost, so the optimum is to flag none."""
        tau, k = tune_cutoff(np.zeros(50, dtype=int), np.full(50, 100.0), np.linspace(0, 1, 50), REVIEW_COST)
        assert np.isinf(tau)
        assert k == 0

    def test_a_monotone_rescaling_of_scores_does_not_change_which_rows_are_flagged(self):
        """The invariant behind 'a recalibration cannot change a fixed score cutoff's decisions'."""
        rng = np.random.default_rng(6)
        n = 500
        p = rng.beta(0.5, 6.0, n)
        y = (rng.random(n) < p).astype(int)
        amount = rng.lognormal(3.0, 1.0, n)
        tau_a, k_a = tune_cutoff(y, amount, p, REVIEW_COST)
        tau_b, k_b = tune_cutoff(y, amount, np.sqrt(p), REVIEW_COST)
        assert k_a == k_b
        assert (p >= tau_a).sum() == (np.sqrt(p) >= tau_b).sum()


class TestBootstrapTotals:
    def test_every_quantity_is_resampled_on_the_same_indices(self):
        """Pairing is the point: ``a - b`` per replicate must equal the replicate of ``a - b``."""
        rows = dict(a=np.arange(100.0), b=np.arange(100.0) * 2)
        boot = bootstrap_totals(rows, n_boot=50, seed=2)
        assert np.allclose(boot["b"], 2 * boot["a"])

    def test_is_reproducible_for_a_fixed_seed(self):
        rows = dict(a=np.arange(100.0))
        first = bootstrap_totals(rows, 20, 4)["a"]
        second = bootstrap_totals(rows, 20, 4)["a"]
        assert np.array_equal(first, second)
