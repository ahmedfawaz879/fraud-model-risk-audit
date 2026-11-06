"""Tests for the verdict rules.

These are the rules that turn evidence into a conclusion, so they are tested on hand-built
intervals rather than on a model's output. The case that matters most is the third state: an
interval that straddles its tolerance band must read as UNCERTAIN, never as a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_audit.config import AuditConfig, Tolerances
from fraud_audit.report import band_state, calibration_state, ranking_verdict

CFG = AuditConfig(tolerances=Tolerances(auc_drop=0.02, oe=(0.80, 1.25), slope=(0.80, 1.20)))


def metrics(oe: tuple[float, float], slope: tuple[float, float]) -> dict[str, float]:
    return dict(oe_lo=oe[0], oe_hi=oe[1], slope_lo=slope[0], slope_hi=slope[1])


class TestBandState:
    @pytest.mark.parametrize(
        "lo, hi, expected",
        [
            (0.90, 1.10, "in"),  # wholly inside
            (0.80, 1.25, "in"),  # exactly on the edges counts as inside
            (1.30, 1.60, "out"),  # wholly above
            (0.40, 0.70, "out"),  # wholly below
            (0.70, 1.10, "unclear"),  # straddles the lower edge
            (1.10, 1.40, "unclear"),  # straddles the upper edge
            (0.10, 9.00, "unclear"),  # uselessly wide: not evidence of anything
        ],
    )
    def test_three_outcomes(self, lo, hi, expected):
        assert band_state(lo, hi, (0.80, 1.25)) == expected

    def test_an_unestimable_metric_is_never_a_pass(self):
        """A GLM that would not fit returns nan; nan must not be read as 'inside the band'."""
        assert band_state(np.nan, np.nan, (0.80, 1.25)) == "unclear"
        assert band_state(0.9, np.nan, (0.80, 1.25)) == "unclear"


class TestCalibrationState:
    def test_both_inside_is_a_pass(self):
        assert calibration_state(metrics((0.9, 1.1), (0.9, 1.1)), CFG) == "IN"

    def test_either_wholly_outside_is_a_failure(self):
        assert calibration_state(metrics((1.4, 1.9), (0.9, 1.1)), CFG) == "OUT"
        assert calibration_state(metrics((0.9, 1.1), (0.2, 0.5)), CFG) == "OUT"

    def test_a_straddling_interval_is_uncertain_not_a_pass(self):
        assert calibration_state(metrics((0.7, 1.1), (0.9, 1.1)), CFG) == "UNCLEAR"

    def test_outside_beats_uncertain_when_they_disagree(self):
        """One decisively broken dimension is a failure even if the other is merely unclear."""
        assert calibration_state(metrics((0.7, 1.1), (0.2, 0.5)), CFG) == "OUT"


class TestRankingVerdict:
    def test_a_small_fall_is_held(self):
        assert ranking_verdict(0.01, 0.001, CFG) == "HELD"

    def test_an_improvement_is_held(self):
        assert ranking_verdict(-0.05, 0.01, CFG) == "HELD"

    def test_a_fall_at_exactly_the_tolerance_is_held(self):
        assert ranking_verdict(0.02, 0.001, CFG) == "HELD"

    def test_a_large_significant_fall_is_degraded(self):
        assert ranking_verdict(0.08, 0.001, CFG) == "DEGRADED"

    def test_a_large_fall_that_could_be_noise_is_inconclusive(self):
        """The honest third answer: the point estimate moved, the evidence does not support it."""
        assert ranking_verdict(0.08, 0.40, CFG) == "INCONCLUSIVE"

    def test_the_tolerance_is_configurable(self):
        strict = AuditConfig(tolerances=Tolerances(auc_drop=0.001))
        assert ranking_verdict(0.01, 0.001, strict) == "DEGRADED"
