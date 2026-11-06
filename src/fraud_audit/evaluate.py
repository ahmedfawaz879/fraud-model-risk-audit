"""Table builders: the audit's evidence, one DataFrame per question.

Each function here takes the fitted predictions and returns a tidy table that is written to
``tables/*.csv`` and read back by :mod:`fraud_audit.report`. Nothing in this module makes a
judgement -- the verdicts live in :mod:`fraud_audit.report`, and they consume only these tables.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from cmvt.metrics import delong_unpaired_test
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import AuditConfig
from .metrics import auroc_delong, bootstrap_statistic, calibration_metrics, ci
from .money import cost_at_cutoff, money_summary, tune_cutoff
from .recalibrate import RECALIBRATORS
from .split import TemporalSplit

__all__ = [
    "EVAL_SPLITS",
    "calibration_table",
    "discrimination_table",
    "auroc_change_table",
    "fixed_cutoff_table",
    "money_table",
    "recalibration_table",
    "review_cost_sensitivity",
]

EVAL_SPLITS: tuple[str, str] = ("intime", "oot")
"""The two numbers that matter: the one a team would report, and the one that is true later."""

Predictions = Mapping[str, Mapping[str, np.ndarray]]


def discrimination_table(split: TemporalSplit, predictions: Predictions, cfg: AuditConfig) -> pd.DataFrame:
    """AUROC with DeLong intervals and AUPRC with bootstrap intervals, in-time and out-of-time.

    AUPRC is reported because AUROC is insensitive to the rare-class regime that matters here;
    its no-skill baseline is the fraud rate, which is why that baseline is a column.
    """
    rows = []
    for name in predictions:
        for part in EVAL_SPLITS:
            y, p = split.y[part], np.asarray(predictions[name][part])
            auc, lo, hi, _ = auroc_delong(y, p)
            boot = bootstrap_statistic(
                len(y),
                lambda idx, y=y, p=p: average_precision_score(y[idx], p[idx]),
                cfg.n_boot,
                cfg.seed,
            )
            ap_lo, ap_hi = ci(boot)
            rows.append(
                dict(
                    model=name,
                    split=part,
                    n=len(y),
                    events=int(y.sum()),
                    auroc=auc,
                    auroc_lo=lo,
                    auroc_hi=hi,
                    auprc=float(average_precision_score(y, p)),
                    auprc_lo=ap_lo,
                    auprc_hi=ap_hi,
                    auprc_baseline=float(y.mean()),
                )
            )
    return pd.DataFrame(rows)


def auroc_change_table(split: TemporalSplit, predictions: Predictions) -> pd.DataFrame:
    """Test the in-time to out-of-time AUROC change with the **unpaired** DeLong test.

    Unpaired is the correct choice: these are two different sets of transactions. Using the
    paired test across cohorts would understate the variance.
    """
    rows = []
    for name in predictions:
        test = delong_unpaired_test(
            split.y["intime"],
            np.asarray(predictions[name]["intime"]),
            split.y["oot"],
            np.asarray(predictions[name]["oot"]),
        )
        rows.append(
            dict(
                model=name,
                auroc_intime=float(test["auc_a"]),
                auroc_oot=float(test["auc_b"]),
                delta_oot_minus_intime=-float(test["delta"]),
                p_unpaired_delong=float(test["p"]),
            )
        )
    return pd.DataFrame(rows)


def calibration_table(split: TemporalSplit, predictions: Predictions) -> pd.DataFrame:
    """O/E, CITL, slope, Brier, log-loss and both ECEs for every model and both splits."""
    rows = []
    for name in predictions:
        for part in EVAL_SPLITS:
            rows.append(
                dict(
                    model=name,
                    split=part,
                    **calibration_metrics(split.y[part], predictions[name][part]),
                )
            )
    return pd.DataFrame(rows)


def money_table(split: TemporalSplit, predictions: Predictions, review_cost: float) -> pd.DataFrame:
    """Economics of the ``p x Amount >= C`` rule on raw probabilities, for all four models."""
    rows = []
    for name in predictions:
        for part in EVAL_SPLITS:
            rows.append(
                dict(
                    model=name,
                    split=part,
                    **money_summary(split.y[part], split.amount[part], predictions[name][part], review_cost),
                )
            )
    return pd.DataFrame(rows)


def fixed_cutoff_table(split: TemporalSplit, predictions: Predictions, cfg: AuditConfig) -> pd.DataFrame:
    """Alert-volume drift of a fixed score cutoff: the policy most teams actually deploy.

    Rows are per 100k transactions so in-time and out-of-time are comparable. A monotone
    recalibration cannot change this table -- it only moves decisions that use the level of a
    probability.
    """
    subject = cfg.subject
    cost = cfg.review_cost
    tau_old, _ = tune_cutoff(split.y["intime"], split.amount["intime"], predictions[subject]["intime"], cost)
    tau_new, _ = tune_cutoff(split.y["recal"], split.amount["recal"], predictions[subject]["recal"], cost)
    return pd.DataFrame(
        [
            dict(
                policy="cutoff tuned in-time, evaluated in-time",
                tau=tau_old,
                **cost_at_cutoff(
                    split.y["intime"],
                    split.amount["intime"],
                    predictions[subject]["intime"],
                    tau_old,
                    cost,
                ),
            ),
            dict(
                policy="same cutoff, evaluated out-of-time",
                tau=tau_old,
                **cost_at_cutoff(
                    split.y["oot"],
                    split.amount["oot"],
                    predictions[subject]["oot"],
                    tau_old,
                    cost,
                ),
            ),
            dict(
                policy="cutoff re-tuned on recalibration window, out-of-time",
                tau=tau_new,
                **cost_at_cutoff(
                    split.y["oot"],
                    split.amount["oot"],
                    predictions[subject]["oot"],
                    tau_new,
                    cost,
                ),
            ),
        ]
    )


def recalibration_table(
    split: TemporalSplit, predictions: Predictions, cfg: AuditConfig
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Fit every recalibrator on the recalibration window; score the out-of-time period.

    Returns the comparison table and ``recalibrated[model][method]`` out-of-time probabilities,
    including a ``(none: raw)`` row so the before/after comparison is in one place.
    """
    rows = []
    recalibrated: dict[str, dict[str, np.ndarray]] = {}
    for name in predictions:
        recalibrated[name] = {}
        for cls in RECALIBRATORS:
            fitted = cls().fit(predictions[name]["recal"], split.y["recal"])
            p_oot = fitted.predict(predictions[name]["oot"])
            recalibrated[name][cls.name] = p_oot
            rows.append(_recal_row(name, cls.name, split, p_oot, cfg.review_cost))
        rows.append(
            _recal_row(name, "(none: raw)", split, np.asarray(predictions[name]["oot"]), cfg.review_cost)
        )
    return pd.DataFrame(rows), recalibrated


def _recal_row(
    model: str, method: str, split: TemporalSplit, p_oot: np.ndarray, review_cost: float
) -> dict[str, object]:
    y, amount = split.y["oot"], split.amount["oot"]
    cal = calibration_metrics(y, p_oot)
    cash = money_summary(y, amount, p_oot, review_cost)
    keep = ("realised_over_predicted", "alerts_per_100k", "savings", "capture")
    return dict(
        model=model,
        method=method,
        auroc=float(roc_auc_score(y, p_oot)),
        oe=cal["oe"],
        citl=cal["citl"],
        slope=cal["slope"],
        brier=cal["brier"],
        ece_equal_mass=cal["ece_equal_mass"],
        **{f"money_{k}": cash[k] for k in keep},
    )


def review_cost_sensitivity(
    split: TemporalSplit, raw_p: np.ndarray, fixed_p: np.ndarray, cfg: AuditConfig
) -> pd.DataFrame:
    """Does the conclusion survive a different review cost? The cost model is an assumption."""
    y, amount = split.y["oot"], split.amount["oot"]
    rows = []
    for cost in cfg.review_cost_grid:
        raw = money_summary(y, amount, raw_p, cost)
        fixed = money_summary(y, amount, fixed_p, cost)
        rows.append(
            dict(
                review_cost=cost,
                savings_raw=raw["savings"],
                savings_recal=fixed["savings"],
                gain=fixed["savings"] - raw["savings"],
                alerts_raw_per_100k=raw["alerts_per_100k"],
                alerts_recal_per_100k=fixed["alerts_per_100k"],
            )
        )
    return pd.DataFrame(rows)
