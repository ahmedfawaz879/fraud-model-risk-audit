"""Audit instruments: discrimination, calibration, and uncertainty.

Four of the things this audit needs are not in ``cmvt`` v0.1.0 and are implemented here
(each is noted in ``docs/toolkit-gaps.md``):

1. calibration slope, intercept (CITL) and O/E with confidence intervals;
2. equal-**mass** ECE, because the toolkit's equal-width bins put nearly every score of a
   0.2%-prevalence model into bin 0, collapsing ECE to roughly ``|mean(p) - mean(y)|``;
3. a vectorised row-level bootstrap;
4. Wilson intervals for the per-band observed rates.

The toolkit supplies DeLong AUROC variance and the unpaired DeLong test, equal-width ECE,
the decidable leakage checks, and PSI. Definitions of O/E, CITL and slope follow
Van Calster et al., *J Clin Epidemiol* (2019).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from cmvt.metrics import delong_auc_var, expected_calibration_error

__all__ = [
    "EPS",
    "BAND_EDGES",
    "auroc_delong",
    "band_table",
    "bootstrap_statistic",
    "calibration_metrics",
    "ci",
    "ece_equal_mass",
    "expit",
    "logit",
    "wilson",
]

EPS = 1e-6

BAND_EDGES = np.array([0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5, 1.0000001])
"""Risk bands for the reliability table. Log-spaced, because the scores are."""


def logit(p) -> np.ndarray:
    """Log-odds, clipped away from 0 and 1 so recalibration stays finite."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z) -> np.ndarray:
    """Inverse of :func:`logit`."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, which stays usable when ``k`` is 0 or ``k == n``.

    The bounds are clipped to [0, 1]: at ``k = 0`` the closed form lands a few times machine
    epsilon below zero, and a negative lower bound on a fraud rate is a reporting error.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    lo, hi = (centre - half) / denom, (centre + half) / denom
    return float(min(max(lo, 0.0), 1.0)), float(min(max(hi, 0.0), 1.0))


def auroc_delong(y, p) -> tuple[float, float, float, float]:
    """AUROC with a DeLong 95% interval. Returns ``(auc, lo, hi, se)``."""
    auc, var = delong_auc_var(np.asarray(y), np.asarray(p, dtype=float))
    se = float(np.sqrt(var))
    return float(auc), float(auc - 1.96 * se), float(auc + 1.96 * se), se


def bootstrap_statistic(
    n: int, stat_of_index: Callable[[np.ndarray], float], n_boot: int, seed: int
) -> np.ndarray:
    """Row-level bootstrap of ``stat_of_index``.

    The data carries no card identifier, so there is nothing to cluster on; if one card
    contributes many rows these intervals are too narrow (see Limitations in the report).
    """
    rng = np.random.default_rng(seed)
    return np.array([stat_of_index(rng.integers(0, n, n)) for _ in range(n_boot)])


def ci(values, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution."""
    lo, hi = np.percentile(np.asarray(values, dtype=float), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ece_equal_mass(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-**count** bins.

    Equal-width bins (the toolkit default) are uninformative at this event rate: nearly every
    score lands in the first bin, so the statistic degenerates to ``|mean(p) - mean(y)|``.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, n_bins)
    return float(sum(len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean()) for idx in parts if len(idx)))


def calibration_metrics(y, p) -> dict[str, float]:
    """O/E, CITL, calibration slope (each with a 95% interval), Brier, log-loss and both ECEs.

    * **O/E** -- observed frauds divided by expected frauds (the sum of predicted
      probabilities). 1.0 is right; 2.0 means the model under-predicts by half. The interval is
      the usual ``exp(+/- 1.96 / sqrt(O))`` approximation on ``log(O/E)`` and ignores the
      variance of ``E``.
    * **CITL** -- the intercept of a logistic fit with the model's logit as a fixed offset: the
      correction the *level* needs. 0 is right.
    * **Slope** -- the coefficient on the model's logit. Below 1 is over-confident, above 1
      under-confident. 1 is right.

    ``citl`` and ``slope`` come back as ``nan`` if the GLM cannot be fitted (perfect separation,
    or no events), which the report renders as a non-verdict rather than a pass.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = logit(p)

    out: dict[str, float] = dict(
        n=len(y),
        events=int(y.sum()),
        base_rate=float(y.mean()),
        mean_pred=float(p.mean()),
    )
    observed, expected = float(y.sum()), float(p.sum())
    out["oe"] = observed / expected if expected > 0 else float("nan")
    half = 1.96 / np.sqrt(max(observed, 1.0))
    out["oe_lo"] = float(out["oe"] * np.exp(-half))
    out["oe_hi"] = float(out["oe"] * np.exp(half))

    try:
        level = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit()
        out["citl"] = float(level.params[0])
        out["citl_lo"], out["citl_hi"] = (float(v) for v in level.conf_int()[0])
        full = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
        out["slope"] = float(full.params[1])
        out["slope_lo"], out["slope_hi"] = (float(v) for v in full.conf_int()[1])
    except Exception:  # noqa: BLE001 - e.g. perfect separation or a split with no events
        for key in ("citl", "citl_lo", "citl_hi", "slope", "slope_lo", "slope_hi"):
            out[key] = float("nan")

    out["brier"] = float(np.mean((p - y) ** 2))
    out["logloss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    out["ece_equal_width"] = float(expected_calibration_error(y, p, 10))
    out["ece_equal_mass"] = ece_equal_mass(y, p, 10)
    return out


def band_table(y, p, edges: np.ndarray = BAND_EDGES) -> pd.DataFrame:
    """Observed fraud rate against mean predicted probability, per risk band, with Wilson bounds.

    Empty bands are dropped. Bands with zero events are kept here (they constrain the upper
    bound) but omitted from the log-log reliability figure, where 0 cannot be drawn.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(y[mask].sum())
        obs_lo, obs_hi = wilson(k, n)
        rows.append(
            dict(
                band=f"[{lo:g}, {min(hi, 1):g})",
                n=n,
                events=k,
                mean_pred=float(p[mask].mean()),
                observed=k / n,
                obs_lo=obs_lo,
                obs_hi=obs_hi,
            )
        )
    return pd.DataFrame(rows)
