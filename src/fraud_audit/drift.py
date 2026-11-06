"""What changed, and which fix that implies.

A calibration problem has three possible causes with three different remedies:

1. **Prior shift** -- the fraud base rate moved. Fix: re-centre (intercept).
2. **Covariate shift** -- the inputs moved. Fix: depends; may need new training data.
3. **Concept shift** -- the input-to-fraud relationship moved. Fix: retrain.

Three diagnostics, none conclusive alone: a two-proportion test on the base rate; per-feature PSI
(conventional reading: <0.1 stable, 0.1-0.25 moderate, >0.25 major) with a KS p-value; and a
domain classifier trained to tell development rows from out-of-time rows, where an AUROC near 0.5
means it cannot. Concept shift is what remains when neither of the first two explains the
miscalibration.

``cmvt.shift.shift_report`` is not used: it calls ``energy_distance``, whose large-sample fallback
is a Python loop over one sample, which is impractical at 100k+ rows per period. PSI is called
directly instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from cmvt.shift import population_stability_index
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import AuditConfig
from .metrics import wilson

__all__ = ["DriftReport", "PSI_MODERATE", "DOMAIN_AUC_THRESHOLD", "drift_report"]

PSI_MODERATE = 0.10
"""Declared threshold above which a feature counts as having moved."""

DOMAIN_AUC_THRESHOLD = 0.60
"""Declared threshold above which the periods count as distinguishable."""

DOMAIN_SAMPLE_CAP = 30_000
"""Rows per period for the domain classifier: enough for a stable AUROC, cheap to fit."""


@dataclass(frozen=True)
class DriftReport:
    """Base-rate shift, input shift, and the score-distribution shift of the audited model."""

    rate_dev: float
    rate_oot: float
    ci_dev: tuple[float, float]
    ci_oot: tuple[float, float]
    prior_p: float
    psi: pd.DataFrame
    max_psi: float
    max_psi_feature: str
    score_psi: float
    domain_auc: float

    @property
    def base_rate_moved(self) -> bool:
        return self.prior_p < 0.05

    @property
    def inputs_moved(self) -> bool:
        return self.max_psi > PSI_MODERATE or self.domain_auc > DOMAIN_AUC_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return dict(
            prior=dict(
                rate_dev=self.rate_dev,
                rate_oot=self.rate_oot,
                ci_dev=list(self.ci_dev),
                ci_oot=list(self.ci_oot),
                p=self.prior_p,
            ),
            max_psi=self.max_psi,
            max_psi_feature=self.max_psi_feature,
            score_psi=self.score_psi,
            domain_auc=self.domain_auc,
        )


def drift_report(
    dev: pd.DataFrame,
    oot: pd.DataFrame,
    features: list[str],
    subject_scores: tuple[np.ndarray, np.ndarray],
    cfg: AuditConfig,
) -> DriftReport:
    """Run all three diagnostics between the development and out-of-time periods.

    ``subject_scores`` is ``(fit_scores, oot_scores)`` of the audited model: a score-distribution
    PSI can stay small while the features move a lot, which is itself informative.
    """
    k0, n0 = int(dev["Class"].sum()), len(dev)
    k1, n1 = int(oot["Class"].sum()), len(oot)
    pooled = (k0 + k1) / (n0 + n1)
    denom = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    z = (k1 / n1 - k0 / n0) / denom if denom > 0 else 0.0
    prior_p = float(2 * stats.norm.sf(abs(z)))

    psi = pd.DataFrame(
        [
            dict(
                feature=column,
                PSI=float(population_stability_index(dev[column].to_numpy(), oot[column].to_numpy())),
                KS_p=float(stats.ks_2samp(dev[column], oot[column]).pvalue),
            )
            for column in features
        ]
    ).sort_values("PSI", ascending=False, ignore_index=True)

    fit_scores, oot_scores = subject_scores
    score_psi = float(population_stability_index(np.asarray(fit_scores), np.asarray(oot_scores)))

    domain_auc = _domain_classifier_auc(dev, oot, features, cfg)

    return DriftReport(
        rate_dev=k0 / n0,
        rate_oot=k1 / n1,
        ci_dev=wilson(k0, n0),
        ci_oot=wilson(k1, n1),
        prior_p=prior_p,
        psi=psi,
        max_psi=float(psi["PSI"].max()),
        max_psi_feature=str(psi.iloc[0]["feature"]),
        score_psi=score_psi,
        domain_auc=domain_auc,
    )


def _domain_classifier_auc(
    dev: pd.DataFrame, oot: pd.DataFrame, features: list[str], cfg: AuditConfig
) -> float:
    """AUROC of a classifier asked to tell a development row from an out-of-time row."""
    m = min(DOMAIN_SAMPLE_CAP, len(dev), len(oot))
    a = dev.sample(m, random_state=cfg.seed)[features].to_numpy()
    b = oot.sample(m, random_state=cfg.seed)[features].to_numpy()
    X = np.vstack([a, b])
    y = np.r_[np.zeros(m), np.ones(m)]
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=cfg.seed
    )
    model = HistGradientBoostingClassifier(max_iter=150, random_state=cfg.seed)
    model.fit(X[train_idx], y[train_idx])
    return float(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))
