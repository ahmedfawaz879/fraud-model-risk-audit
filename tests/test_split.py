"""Tests for the split, where the whole audit's validity is decided.

If a single later row reaches the fitting stage, every out-of-time number in the report is
optimistic and nothing downstream can detect it. So the invariant is tested directly, including
the awkward case the notebook handled by hand: a boundary landing inside a block of transactions
sharing one timestamp.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_audit.config import AuditConfig
from fraud_audit.data import FEATURES, load_transactions
from fraud_audit.split import SPLIT_NAMES, snap_to_timestamp_boundary, temporal_split


@pytest.fixture(scope="module")
def dataset(csv_path):
    return load_transactions(AuditConfig(data_path=csv_path))


@pytest.fixture(scope="module")
def split(dataset):
    return temporal_split(dataset.frame, AuditConfig(data_path=""))


class TestSnapToTimestampBoundary:
    def test_moves_forward_off_a_tie(self):
        times = np.array([0, 1, 1, 1, 2, 3])
        assert snap_to_timestamp_boundary(times, 2) == 4

    def test_leaves_a_clean_boundary_alone(self):
        times = np.array([0, 1, 2, 3])
        assert snap_to_timestamp_boundary(times, 2) == 2

    def test_runs_off_the_end_when_every_timestamp_is_identical(self):
        """The caller must treat this as 'no room for a later period', not as a valid cut."""
        times = np.zeros(10)
        assert snap_to_timestamp_boundary(times, 5) == len(times)


class TestTemporalSplit:
    def test_produces_the_four_expected_splits(self, split):
        assert set(split.frames) == set(SPLIT_NAMES)

    def test_periods_are_strictly_ordered_in_time(self, split):
        assert split.dev["Time"].max() < split.frames["recal"]["Time"].min()
        assert split.frames["recal"]["Time"].max() < split.frames["oot"]["Time"].min()

    def test_no_row_is_both_fitted_on_and_evaluated(self, split):
        fit_ids = set(split.frames["fit"]["row_id"])
        for later in ("intime", "recal", "oot"):
            assert not fit_ids & set(split.frames[later]["row_id"])

    def test_every_row_lands_in_exactly_one_split(self, split, dataset):
        counts = sum(len(frame) for frame in split.frames.values())
        assert counts == len(dataset.frame)

    def test_time_is_not_among_the_features(self, split):
        assert "Time" not in split.features
        assert split.features == FEATURES

    def test_feature_matrix_matches_the_frame_row_for_row(self, split):
        for name in SPLIT_NAMES:
            assert split.X[name].shape == (len(split.frames[name]), len(FEATURES))
            assert np.array_equal(split.y[name], split.frames[name]["Class"].to_numpy())

    def test_check_ordering_catches_a_tampered_split(self, split):
        broken = type(split)(
            frames={**split.frames, "recal": split.frames["oot"], "oot": split.frames["recal"]},
            X=split.X,
            y=split.y,
            amount=split.amount,
            table=split.table,
            dev=split.dev,
            boundaries=split.boundaries,
            features=split.features,
        )
        with pytest.raises(ValueError, match="overlap in time"):
            broken.check_ordering()

    def test_unsorted_input_is_rejected(self, dataset):
        shuffled = dataset.frame.sample(frac=1.0, random_state=0)
        with pytest.raises(ValueError, match="sorted by Time"):
            temporal_split(shuffled, AuditConfig(data_path=""))

    def test_too_few_rows_is_a_clear_error_not_an_empty_split(self, dataset):
        with pytest.raises(ValueError, match="empty period"):
            temporal_split(dataset.frame.head(2), AuditConfig(data_path=""))

    def test_split_table_reports_the_sizes_it_actually_produced(self, split):
        table = split.table.set_index("split")
        for name in SPLIT_NAMES:
            assert table.loc[name, "rows"] == len(split.frames[name])
            assert table.loc[name, "frauds"] == split.events(name)
