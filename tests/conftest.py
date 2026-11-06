"""Shared fixtures.

Every test runs on synthetic data with a known truth. Nothing here downloads the Kaggle file, so
the suite is runnable on a clean checkout and in CI: the point of a self-test is that the right
answer is known in advance, which is only true of data you generated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_audit.config import AuditConfig
from fraud_audit.metrics import expit, logit

N_FEATURES = 28


@pytest.fixture(scope="session")
def calibrated_sample() -> tuple[np.ndarray, np.ndarray]:
    """A perfectly calibrated sample: ``p_true`` really is each row's event probability."""
    rng = np.random.default_rng(0)
    n = 40_000
    p_true = expit(rng.normal(-3.0, 1.4, n))
    y = (rng.random(n) < p_true).astype(int)
    return y, p_true


@pytest.fixture(scope="session")
def distorted_sample(calibrated_sample) -> tuple[np.ndarray, np.ndarray]:
    """The same labels with a planted distortion: slope 2.0 and a +0.5 level shift."""
    y, p_true = calibrated_sample
    return y, expit(2.0 * logit(p_true) + 0.5)


def make_transactions(
    n: int = 12_000,
    base_rate: float = 0.002,
    seed: int = 7,
    drift: float = 2.0,
    duplicate_rows: int = 5,
) -> pd.DataFrame:
    """A creditcard.csv-shaped frame: ``Time``, ``V1..V28``, ``Amount``, ``Class``.

    Deliberately includes the three things the pipeline must cope with: a genuine signal, a
    covariate shift in the later period (so the drift diagnostics have something to find), and a
    handful of exact duplicate rows (so the de-duplication step is exercised).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    fraction_late = np.linspace(0.0, 1.0, n)
    X[:, 0] += drift * fraction_late  # covariate shift concentrated in V1
    weights = np.zeros(N_FEATURES)
    weights[[0, 1, 2, 3, 4]] = [1.0, -0.9, 0.8, -0.6, 0.5]
    intercept = np.log(base_rate / (1 - base_rate))
    p = expit(intercept + X @ weights)
    y = (rng.random(n) < p).astype(int)

    frame = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, N_FEATURES + 1)])
    frame.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)).round(0))
    frame["Amount"] = np.round(rng.lognormal(3.0, 1.2, n), 2)
    frame["Class"] = y
    if duplicate_rows:
        frame = pd.concat([frame, frame.iloc[:duplicate_rows]], ignore_index=True)
        frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    return frame


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory, transactions) -> str:
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    transactions.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fast_config(csv_path, tmp_path) -> AuditConfig:
    """A config small enough to run the whole audit inside a test."""
    return AuditConfig(
        data_path=csv_path,
        output_dir=str(tmp_path / "out"),
        n_boot=25,
        review_cost_grid=(1.0, 5.0),
        make_figures=False,
    )
