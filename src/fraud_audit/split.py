"""The strict temporal split, and the one invariant the whole audit rests on.

Rows are ordered by ``Time`` and cut into three consecutive periods. Nothing from a later period
is ever used to fit, tune or choose anything for an earlier one:

===================  =============  ======================================================
Period               Rows           Used for
===================  =============  ======================================================
Development          earliest 55%   fit the models (80%) and the **in-time** hold-out (20%)
Recalibration        next 15%       the only place a recalibration is fitted
Out-of-time test     last 30%       touched once, at the end, for every number that matters
===================  =============  ======================================================

The notebook this package grew out of asserted the period ordering by hand.
:func:`temporal_split` enforces it, and :meth:`TemporalSplit.check_ordering` is re-run as a test.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import AuditConfig
from .data import FEATURES

__all__ = ["TemporalSplit", "SPLIT_NAMES", "temporal_split", "snap_to_timestamp_boundary"]

log = logging.getLogger(__name__)

SPLIT_NAMES: tuple[str, ...] = ("fit", "intime", "recal", "oot")


def snap_to_timestamp_boundary(times: np.ndarray, index: int) -> int:
    """Move ``index`` forward until it no longer cuts inside a block of identical timestamps.

    Transactions sharing a timestamp must land in the same period, otherwise the "strictly
    later" claim is false at the boundary. Returns ``len(times)`` if every remaining timestamp
    is identical, which the caller must treat as "no room left for a later period".
    """
    n = len(times)
    i = int(index)
    while 0 < i < n and times[i] == times[i - 1]:
        i += 1
    return i


@dataclass(frozen=True)
class TemporalSplit:
    """Four aligned views of the data: ``fit``, ``intime``, ``recal``, ``oot``."""

    frames: Mapping[str, pd.DataFrame]
    X: Mapping[str, np.ndarray]
    y: Mapping[str, np.ndarray]
    amount: Mapping[str, np.ndarray]
    table: pd.DataFrame
    dev: pd.DataFrame
    boundaries: tuple[int, int]
    features: list[str]

    @property
    def split_time(self) -> float:
        """Last timestamp of the development period: the go-live moment the audit simulates."""
        return float(self.dev["Time"].max())

    def events(self, name: str) -> int:
        return int(self.y[name].sum())

    def check_ordering(self) -> None:
        """Raise unless development < recalibration < out-of-time, strictly, in time."""
        dev, recal, oot = self.dev, self.frames["recal"], self.frames["oot"]
        if not (dev["Time"].max() < recal["Time"].min()):
            raise ValueError("development and recalibration periods overlap in time")
        if not (recal["Time"].max() < oot["Time"].min()):
            raise ValueError("recalibration and out-of-time periods overlap in time")
        fit_ids = set(self.frames["fit"]["row_id"])
        for later in ("intime", "recal", "oot"):
            shared = fit_ids & set(self.frames[later]["row_id"])
            if shared:
                raise ValueError(f"{len(shared)} rows appear in both 'fit' and '{later}'")


def temporal_split(frame: pd.DataFrame, cfg: AuditConfig) -> TemporalSplit:
    """Cut a time-ordered frame into the four audit splits and verify the ordering."""
    if not frame["Time"].is_monotonic_increasing:
        raise ValueError("frame must be sorted by Time before splitting")

    times = frame["Time"].to_numpy()
    n = len(frame)
    i1 = snap_to_timestamp_boundary(times, int(n * cfg.train_frac))
    i2 = snap_to_timestamp_boundary(times, int(n * (cfg.train_frac + cfg.recal_frac)))
    if not 0 < i1 < i2 < n:
        raise ValueError(
            f"the requested fractions leave an empty period (n={n}, boundaries={(i1, i2)}); "
            "too few rows, or too many share a timestamp"
        )

    dev, recal_df, oot_df = frame.iloc[:i1], frame.iloc[i1:i2], frame.iloc[i2:]
    stratify = dev["Class"] if int(dev["Class"].sum()) >= 2 else None
    fit_df, intime_df = train_test_split(
        dev, test_size=cfg.intime_val_frac, stratify=stratify, random_state=cfg.seed
    )

    frames = {"fit": fit_df, "intime": intime_df, "recal": recal_df, "oot": oot_df}
    table = pd.DataFrame(
        [
            dict(
                split=name,
                rows=len(part),
                frauds=int(part["Class"].sum()),
                fraud_rate=float(part["Class"].mean()),
                t_start_h=float(part["Time"].min()) / 3600.0,
                t_end_h=float(part["Time"].max()) / 3600.0,
            )
            for name, part in frames.items()
        ]
    )

    split = TemporalSplit(
        frames=frames,
        X={k: v[FEATURES].to_numpy() for k, v in frames.items()},
        y={k: v["Class"].to_numpy() for k, v in frames.items()},
        amount={k: v["Amount"].to_numpy() for k, v in frames.items()},
        table=table,
        dev=dev,
        boundaries=(i1, i2),
        features=list(FEATURES),
    )
    split.check_ordering()

    for name in ("recal", "oot"):
        if split.events(name) < 30:
            log.warning(
                "only %d frauds in the '%s' split: every interval will be wide",
                split.events(name),
                name,
            )
    return split
