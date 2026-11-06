"""Self-tests for recalibration.

Two properties are load-bearing for the audit's conclusions and are checked on a *planted*
distortion, where the correct repair is known:

* each method restores the expected event count (O/E back to 1);
* each method is monotone, so AUROC is untouched -- which is why the report can say a
  recalibration fixes probabilities "without changing the ranking" rather than hoping so.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fraud_audit.metrics import calibration_metrics
from fraud_audit.recalibrate import (
    RECALIBRATORS,
    InterceptRecalibrator,
    IsotonicRecalibrator,
    LogisticRecalibrator,
    Recalibrator,
    get_recalibrator,
)


@pytest.mark.parametrize("cls", RECALIBRATORS, ids=lambda c: c.name)
class TestEveryRecalibrator:
    def test_satisfies_the_protocol(self, cls):
        assert isinstance(cls(), Recalibrator)

    def test_restores_the_expected_event_count(self, cls, distorted_sample):
        y, p_bad = distorted_sample
        fitted = cls().fit(p_bad, y)
        assert fitted.predict(p_bad).sum() / y.sum() == pytest.approx(1.0, abs=1e-3)

    def test_is_monotone_so_auroc_is_unchanged(self, cls, distorted_sample):
        y, p_bad = distorted_sample
        fitted = cls().fit(p_bad, y)
        before = roc_auc_score(y, p_bad)
        after = roc_auc_score(y, fitted.predict(p_bad))
        # The parametric maps are strictly increasing, so AUROC is identical to floating point.
        # Isotonic is a step function: the ties it creates score 0.5 per pair, which can nudge
        # AUROC either way, so it gets a tolerance rather than an equality.
        tolerance = 0.02 if cls is IsotonicRecalibrator else 1e-9
        assert after == pytest.approx(before, abs=tolerance)

    def test_output_is_a_probability(self, cls, distorted_sample):
        y, p_bad = distorted_sample
        out = cls().fit(p_bad, y).predict(p_bad)
        assert ((out >= 0) & (out <= 1)).all()

    def test_predict_before_fit_is_an_error_not_a_wrong_answer(self, cls):
        with pytest.raises(RuntimeError):
            cls().predict(np.array([0.1, 0.2]))


class TestLogisticRecalibrator:
    def test_repairs_both_the_level_and_the_slope(self, distorted_sample):
        """The planted distortion was slope 2.0 plus a level shift; both must come back to 1 and 0."""
        y, p_bad = distorted_sample
        fixed = LogisticRecalibrator().fit(p_bad, y).predict(p_bad)
        cal = calibration_metrics(y, fixed)
        assert cal["slope"] == pytest.approx(1.0, abs=0.02)
        assert cal["citl"] == pytest.approx(0.0, abs=0.02)

    def test_recovers_the_planted_coefficients(self, distorted_sample):
        """Inverting ``logit(p') = 2 * logit(p) + 0.5`` means b ~ 0.5 and a ~ -0.25."""
        y, p_bad = distorted_sample
        fitted = LogisticRecalibrator().fit(p_bad, y)
        assert fitted.b == pytest.approx(0.5, abs=0.06)
        assert fitted.a == pytest.approx(-0.25, abs=0.08)


class TestInterceptRecalibrator:
    def test_corrects_the_level_but_leaves_the_slope_wrong(self, distorted_sample):
        """One parameter cannot fix two problems: that is why the audit reports both."""
        y, p_bad = distorted_sample
        fixed = InterceptRecalibrator().fit(p_bad, y).predict(p_bad)
        cal = calibration_metrics(y, fixed)
        assert cal["oe"] == pytest.approx(1.0, abs=0.01)
        assert cal["slope"] < 0.8


class TestRegistry:
    def test_names_are_unique(self):
        names = [cls.name for cls in RECALIBRATORS]
        assert len(names) == len(set(names))

    def test_lookup_by_declared_name(self):
        assert get_recalibrator("Logistic (slope + intercept)") is LogisticRecalibrator

    def test_unknown_name_is_rejected_loudly(self):
        with pytest.raises(KeyError, match="unknown recalibrator"):
            get_recalibrator("magic")
