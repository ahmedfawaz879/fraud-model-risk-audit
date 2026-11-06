"""The models under audit, and the three comparators.

Two families, each trained two ways. The **audit subject** is declared in the config before any
out-of-time number is seen; the rest are comparators.

The *class-weighted* variants are included on purpose. Re-weighting classes to "fix imbalance" is
common, and is known to preserve ranking while destroying the probability scale, so their
calibration result is **expected by construction**: they demonstrate what this audit catches,
they are not a finding about the data.

The boosting models are small and regularised, with the number of trees chosen by early stopping
on 15% of the *fit* rows only. A deeper, unregularised booster on ~150 fraud examples is badly
miscalibrated even in-time, which would confound a study about drift.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .split import TemporalSplit

__all__ = ["MODEL_NAMES", "SAMPLE_WEIGHTED", "make_models", "fit_models"]

log = logging.getLogger(__name__)

MODEL_NAMES: tuple[str, ...] = (
    "Logistic regression",
    "Gradient boosting",
    "Logistic regression (class-weighted)",
    "Gradient boosting (class-weighted)",
)

SAMPLE_WEIGHTED: frozenset[str] = frozenset({"Gradient boosting (class-weighted)"})
"""Boosting has no ``class_weight``; it is re-weighted through ``sample_weight`` at fit time."""


def make_models(seed: int) -> dict[str, object]:
    """Build the four unfitted estimators. Fixed settings; no hyper-parameter search."""

    def linear(**kw):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, **kw))

    def boosted(**kw):
        return HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=8,
            min_samples_leaf=100,
            l2_regularization=10.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            scoring="loss",
            random_state=seed,
            **kw,
        )

    models = {
        "Logistic regression": linear(),
        "Gradient boosting": boosted(),
        "Logistic regression (class-weighted)": linear(class_weight="balanced"),
        "Gradient boosting (class-weighted)": boosted(),
    }
    assert tuple(models) == MODEL_NAMES, "MODEL_NAMES is out of step with make_models"
    return models


def fit_models(split: TemporalSplit, seed: int) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    """Fit every model on the ``fit`` rows and score all four splits.

    Returns ``(models, predictions)`` where ``predictions[model][split]`` is a probability array.
    Scoring the later splits here is safe: nothing reads an out-of-time prediction until the
    evaluation stage, and no fitting decision depends on one.
    """
    models = make_models(seed)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, model in models.items():
        if name in SAMPLE_WEIGHTED:
            weights = compute_sample_weight("balanced", split.y["fit"])
            model.fit(split.X["fit"], split.y["fit"], sample_weight=weights)
        else:
            model.fit(split.X["fit"], split.y["fit"])
        predictions[name] = {part: model.predict_proba(split.X[part])[:, 1] for part in split.frames}
        log.info("fitted: %s", name)
    return models, predictions


def require_subject(predictions: Mapping[str, object], subject: str) -> None:
    """Fail loudly if the configured audit subject is not one of the fitted models."""
    if subject not in predictions:
        raise ValueError(f"audit subject {subject!r} is not among the models {list(predictions)}")
