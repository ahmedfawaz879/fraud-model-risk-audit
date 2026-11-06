"""Tests for the integrity checks and the drift diagnostics.

Both are tested with *planted* answers: duplicates inserted on purpose, and a covariate shift
built into the synthetic generator. A diagnostic that cannot find a shift you put there yourself
will not find one you did not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_transactions
from fraud_audit.config import AuditConfig
from fraud_audit.data import load_transactions
from fraud_audit.drift import DOMAIN_AUC_THRESHOLD, drift_report
from fraud_audit.integrity import UNINFORMATIVE_CHECK, cross_split_duplicates, integrity_scan
from fraud_audit.split import temporal_split


@pytest.fixture(scope="module")
def split(csv_path):
    cfg = AuditConfig(data_path=csv_path)
    return temporal_split(load_transactions(cfg).frame, cfg)


class TestIntegrityScan:
    def test_the_strict_split_passes_the_decidable_checks(self, split):
        scan = integrity_scan(split)
        assert bool(scan["passed"].all()), scan

    def test_the_uninformative_check_is_annotated_rather_than_silently_counted(self, split):
        """No card id exists, so 'no entity in both splits' is trivially true and means nothing."""
        scan = integrity_scan(split)
        note = scan.loc[scan["check"] == UNINFORMATIVE_CHECK, "note"].iloc[0]
        assert "NOT informative" in note


class TestCrossSplitDuplicates:
    def test_reports_one_row_per_later_split(self, split):
        table = cross_split_duplicates(split)
        assert list(table["split"]) == ["intime", "recal", "oot"]

    def test_counts_are_non_negative_and_bounded_by_the_split_size(self, split):
        table = cross_split_duplicates(split).set_index("split")
        for name in ("intime", "recal", "oot"):
            assert 0 <= table.loc[name, "rows_identical_to_a_fit_row"] <= len(split.frames[name])

    def test_finds_a_duplicate_planted_in_the_out_of_time_period(self, tmp_path):
        """A fit row copied into the later period is leakage a time split cannot prevent."""
        frame = make_transactions(n=4_000, duplicate_rows=0)
        cfg = AuditConfig(data_path=str(tmp_path / "c.csv"), dedupe_exact_rows=False)
        early = frame.iloc[:5].copy()
        early["Time"] = frame["Time"].max() + np.arange(1, 6)  # same features, later timestamps
        pd.concat([frame, early], ignore_index=True).to_csv(cfg.data_path, index=False)

        split = temporal_split(load_transactions(cfg).frame, cfg)
        found = cross_split_duplicates(split).set_index("split").loc["oot", "rows_identical_to_a_fit_row"]
        assert found >= 1


class TestDriftReport:
    @pytest.fixture(scope="class")
    def report(self, split):
        rng = np.random.default_rng(0)
        scores = (
            rng.random(len(split.frames["fit"])) * 0.01,
            rng.random(len(split.frames["oot"])) * 0.01,
        )
        return drift_report(split.dev, split.frames["oot"], split.features, scores, AuditConfig())

    def test_finds_the_covariate_shift_planted_in_v1(self, report):
        """The generator drifts V1 and nothing else, so V1 must top the PSI table."""
        assert report.max_psi_feature == "V1"
        assert report.inputs_moved

    def test_the_domain_classifier_can_tell_the_periods_apart(self, report):
        assert report.domain_auc > DOMAIN_AUC_THRESHOLD

    def test_psi_covers_every_feature_and_is_sorted(self, report, split):
        assert set(report.psi["feature"]) == set(split.features)
        assert report.psi["PSI"].is_monotonic_decreasing

    def test_serialises_for_run_config(self, report):
        import json

        json.dumps(report.to_dict())

    def test_an_unshifted_period_is_reported_as_stable(self, tmp_path):
        """The negative control: no planted drift, so nothing may be flagged as moved."""
        frame = make_transactions(n=6_000, drift=0.0, duplicate_rows=0, seed=11)
        path = tmp_path / "flat.csv"
        frame.to_csv(path, index=False)
        cfg = AuditConfig(data_path=str(path))
        split = temporal_split(load_transactions(cfg).frame, cfg)
        rng = np.random.default_rng(1)
        scores = (rng.random(len(split.frames["fit"])), rng.random(len(split.frames["oot"])))
        report = drift_report(split.dev, split.frames["oot"], split.features, scores, cfg)
        assert report.max_psi < 0.10
        assert report.domain_auc < 0.60
        assert not report.inputs_moved
