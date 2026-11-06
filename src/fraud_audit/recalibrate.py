"""Recalibration: three levels of updating, simplest first.

After Steyerberg's updating hierarchy. All three act on the **logit** of the model's score and
are **monotone**, which has two consequences the audit relies on:

* they cannot change the ranking, so AUROC is unchanged to numerical precision;
* they cannot change the decisions of a fixed *score* cutoff.

They therefore matter only for rules that use the probability's *level* -- loss provisioning and
the ``p x Amount >= C`` review rule.

``cmvt.calibration.recalibrate(method="platt")`` fits the logistic map on the raw probability;
on the logit it is the standard slope-and-intercept update, which is what is wanted here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from .metrics import expit, logit

__all__ = [
    "Recalibrator",
    "InterceptRecalibrator",
    "LogisticRecalibrator",
    "IsotonicRecalibrator",
    "RECALIBRATORS",
    "get_recalibrator",
]


@runtime_checkable
class Recalibrator(Protocol):
    """Fit on a held-out window's ``(p, y)``, then map probabilities to probabilities."""

    name: str

    def fit(self, p, y) -> Recalibrator: ...

    def predict(self, p) -> np.ndarray: ...


class InterceptRecalibrator:
    """Shift only: corrects base-rate (prior) drift and nothing else.

    One parameter, so it is the only option worth fitting when the recalibration window holds
    few events.
    """

    name = "Intercept only"

    def __init__(self) -> None:
        self.a: float | None = None

    def fit(self, p, y) -> InterceptRecalibrator:
        model = sm.GLM(
            np.asarray(y),
            np.ones((len(np.asarray(y)), 1)),
            family=sm.families.Binomial(),
            offset=logit(p),
        ).fit()
        self.a = float(model.params[0])
        return self

    def predict(self, p) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("fit() before predict()")
        return expit(logit(p) + self.a)


class LogisticRecalibrator:
    """Slope and intercept: corrects the level *and* over/under-confidence.

    The audit's primary method, declared in advance in the config.
    """

    name = "Logistic (slope + intercept)"

    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None

    def fit(self, p, y) -> LogisticRecalibrator:
        model = sm.GLM(np.asarray(y), sm.add_constant(logit(p)), family=sm.families.Binomial()).fit()
        self.a, self.b = float(model.params[0]), float(model.params[1])
        return self

    def predict(self, p) -> np.ndarray:
        if self.a is None or self.b is None:
            raise RuntimeError("fit() before predict()")
        return expit(self.a + self.b * logit(p))


class IsotonicRecalibrator:
    """Free-form monotone map: the most flexible, and the hungriest for events.

    Being a step function it also creates ties, which flattens parts of the score distribution.
    A sensitivity check, not the primary method.
    """

    name = "Isotonic"

    def __init__(self) -> None:
        self.model: IsotonicRegression | None = None

    def fit(self, p, y) -> IsotonicRecalibrator:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            np.asarray(p, dtype=float), np.asarray(y)
        )
        return self

    def predict(self, p) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() before predict()")
        return np.asarray(self.model.predict(np.asarray(p, dtype=float)), dtype=float)


RECALIBRATORS: tuple[type, ...] = (
    InterceptRecalibrator,
    LogisticRecalibrator,
    IsotonicRecalibrator,
)


def get_recalibrator(name: str) -> type:
    """Look a recalibrator class up by its declared ``name``."""
    for cls in RECALIBRATORS:
        if cls.name == name:
            return cls
    raise KeyError(f"unknown recalibrator {name!r}; known: {[c.name for c in RECALIBRATORS]}")
