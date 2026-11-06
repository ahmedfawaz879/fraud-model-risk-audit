"""Self-tests for the audit's instruments.

An audit is only as good as its instruments, so each check below has a known right answer: a
perfectly calibrated sample, a planted level shift, a planted over-confidence, agreement with
``scikit-learn``, and agreement between two independent uncertainty estimates.

These are the notebook's section-2 self-tests, which used to run inside the analysis and abort it
on failure. Here they run before it, in CI, which is where a broken instrument should be caught.
"""

from __future__ import annotations

import numpy as np
import pytest
from cmvt.metrics import delong_auc_var
from sklearn.metrics import roc_auc_score

from fraud_audit.metrics import (
    auroc_delong,
    band_table,
    bootstrap_statistic,
    calibration_metrics,
    ci,
    ece_equal_mass,
    expit,
    logit,
    wilson,
)


def test_logit_expit_round_trip():
    p = np.array([1e-9, 0.01, 0.5, 0.99, 1 - 1e-9])
    assert np.allclose(expit(logit(p)), np.clip(p, 1e-6, 1 - 1e-6), atol=1e-9)


def test_logit_is_finite_at_the_boundaries():
    """Clipping is what lets a recalibrator be fitted on scores that hit 0 or 1 exactly."""
    assert np.all(np.isfinite(logit(np.array([0.0, 1.0]))))


class TestWilson:
    def test_zero_events_gives_a_lower_bound_of_zero(self):
        lo, hi = wilson(0, 100)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert 0 < hi < 0.05

    def test_interval_stays_inside_the_unit_range(self):
        for k, n in [(0, 10), (1, 10), (10, 10), (3, 7)]:
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0

    def test_empty_sample_is_nan_not_an_exception(self):
        assert all(np.isnan(v) for v in wilson(0, 0))


class TestCalibrationMetrics:
    def test_a_perfectly_calibrated_sample_passes_every_measure(self, calibrated_sample):
        y, p = calibrated_sample
        cal = calibration_metrics(y, p)
        assert 0.9 < cal["oe"] < 1.1
        assert abs(cal["citl"]) < 0.15
        assert 0.88 < cal["slope"] < 1.12

    def test_a_planted_level_shift_shows_up_as_citl_and_oe(self, calibrated_sample):
        """Over-predicting (every logit +1) must read as negative CITL and O/E below 1."""
        y, p = calibrated_sample
        cal = calibration_metrics(y, expit(logit(p) + 1.0))
        assert -1.2 < cal["citl"] < -0.8
        assert cal["oe"] < 0.6

    def test_a_planted_over_confidence_shows_up_as_a_slope_near_one_half(self, calibrated_sample):
        """Doubling every logit must halve the recovered slope."""
        y, p = calibrated_sample
        cal = calibration_metrics(y, expit(2.0 * logit(p)))
        assert 0.40 < cal["slope"] < 0.60

    @pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
    def test_metrics_degrade_to_nan_rather_than_crash_without_events(self):
        y = np.zeros(500, dtype=int)
        p = np.full(500, 0.01)
        cal = calibration_metrics(y, p)
        assert cal["events"] == 0
        assert np.isnan(cal["slope"]) or not np.isfinite(cal["slope_lo"])

    def test_equal_width_ece_is_blind_where_equal_mass_ece_is_not(self, calibrated_sample):
        """The reason the audit reports both: at a low event rate, equal-width bins collapse."""
        y, p = calibrated_sample
        rare = p * 0.02  # push every score into the first equal-width bin
        cal = calibration_metrics((np.random.default_rng(1).random(len(y)) < rare).astype(int), rare)
        assert cal["ece_equal_width"] == pytest.approx(abs(cal["mean_pred"] - cal["base_rate"]), abs=5e-4)


class TestEceEqualMass:
    def test_perfect_calibration_gives_near_zero(self, calibrated_sample):
        y, p = calibrated_sample
        assert ece_equal_mass(y, p) < 0.01

    def test_a_distortion_increases_it(self, calibrated_sample, distorted_sample):
        y, p = calibrated_sample
        _, p_bad = distorted_sample
        assert ece_equal_mass(y, p_bad) > ece_equal_mass(y, p)


class TestDiscrimination:
    def test_delong_auc_matches_sklearn_exactly(self, calibrated_sample):
        y, p = calibrated_sample
        auc, _ = delong_auc_var(y, p)
        assert auc == pytest.approx(roc_auc_score(y, p), abs=1e-9)

    def test_delong_standard_error_agrees_with_the_bootstrap(self, calibrated_sample):
        """Two independent routes to the same uncertainty; disagreement means one is wrong."""
        y, p = calibrated_sample
        rng = np.random.default_rng(3)
        sub = rng.choice(len(y), 6000, replace=False)
        ys, ps = y[sub], p[sub]
        se_delong = float(np.sqrt(delong_auc_var(ys, ps)[1]))
        se_boot = bootstrap_statistic(len(ys), lambda idx: roc_auc_score(ys[idx], ps[idx]), 300, 1).std(
            ddof=1
        )
        assert abs(se_delong / se_boot - 1) < 0.25

    def test_auroc_delong_interval_brackets_the_estimate(self, calibrated_sample):
        y, p = calibrated_sample
        auc, lo, hi, se = auroc_delong(y, p)
        assert lo < auc < hi
        assert se > 0


class TestBootstrap:
    def test_is_reproducible_for_a_fixed_seed(self):
        draws = [bootstrap_statistic(100, lambda idx: float(idx.mean()), 20, 11) for _ in range(2)]
        assert np.array_equal(draws[0], draws[1])

    def test_ci_brackets_the_median_of_the_draws(self):
        draws = np.random.default_rng(0).normal(5.0, 1.0, 2000)
        lo, hi = ci(draws)
        assert lo < np.median(draws) < hi


class TestBandTable:
    def test_bands_partition_the_rows(self, calibrated_sample):
        y, p = calibrated_sample
        table = band_table(y, p)
        assert table["n"].sum() == len(y)
        assert table["events"].sum() == int(y.sum())

    def test_observed_rate_sits_inside_its_own_wilson_interval(self, calibrated_sample):
        y, p = calibrated_sample
        table = band_table(y, p)
        assert ((table["obs_lo"] <= table["observed"]) & (table["observed"] <= table["obs_hi"])).all()
