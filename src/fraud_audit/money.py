"""What a calibration gap costs.

Calibration is abstract until a decision depends on the *level* of a probability. Two do:

* **Loss provisioning** -- expected loss is ``sum(p * Amount)``, compared with the loss that
  actually happened. A miscalibrated level is a mis-stated provision.
* **A risk-based review rule** -- send a transaction to manual review iff ``p * Amount >= C``,
  where ``C`` is the cost of one review. The cutoff is *derived* from costs rather than tuned,
  so it is only correct if ``p`` is a real probability.

A third policy, a **fixed score cutoff tuned in-time**, is also implemented, because it is what
most teams actually run; a monotone recalibration cannot change its decisions, only its drift.

Assumptions, all of them debatable and all of them in the config: a review costs ``C``; a
reviewed fraud is always stopped; an unreviewed fraud costs its full ``Amount``; customer
friction and chargeback fees are ignored. ``Amount`` is in the dataset's own (unstated) currency.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["cost_at_cutoff", "money_rows", "money_summary", "bootstrap_totals", "tune_cutoff"]


def money_rows(y, amount, p, review_cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row outcome of the rule 'review iff ``p * Amount >= C``'.

    Returns ``(flag, cost)``: a reviewed transaction costs ``C`` and any fraud in it is stopped;
    an unreviewed fraud costs its ``Amount``.
    """
    y = np.asarray(y)
    amount = np.asarray(amount, dtype=float)
    p = np.asarray(p, dtype=float)
    flag = (p * amount) >= review_cost
    cost = np.where(flag, review_cost, y * amount)
    return flag, cost


def money_summary(y, amount, p, review_cost: float) -> dict[str, float]:
    """Totals for the ``p x Amount`` rule, plus the provisioning ratio and the oracle bound.

    ``realised_over_predicted`` is 1.0 when the provision was right, above 1 when the model
    under-provisioned. ``capture`` is the share of the oracle's saving achieved, where the oracle
    knows every label and reviews a fraud only when reviewing is cheaper than losing the amount.
    """
    y = np.asarray(y)
    amount = np.asarray(amount, dtype=float)
    flag, cost = money_rows(y, amount, p, review_cost)
    realised = float((y * amount).sum())
    predicted = float((np.asarray(p, dtype=float) * amount).sum())
    oracle_cost = float((y * np.minimum(review_cost, amount)).sum())
    total_cost = float(cost.sum())
    return dict(
        predicted_loss=predicted,
        realised_loss=realised,
        realised_over_predicted=realised / predicted if predicted > 0 else float("nan"),
        alerts=int(flag.sum()),
        alerts_per_100k=float(flag.mean() * 1e5),
        cost=total_cost,
        savings=realised - total_cost,
        savings_share=(realised - total_cost) / realised if realised > 0 else float("nan"),
        oracle_savings=realised - oracle_cost,
        capture=(
            (realised - total_cost) / (realised - oracle_cost) if realised > oracle_cost else float("nan")
        ),
    )


def bootstrap_totals(rows: Mapping[str, np.ndarray], n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Resample rows once per replicate and total every per-row quantity on the same indices.

    Sharing the indices is what makes the *paired* gain from recalibrating ("same transactions,
    two policies") a legitimate interval rather than the difference of two independent ones.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(rows.values())))
    out = {key: np.empty(n_boot) for key in rows}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for key, values in rows.items():
            out[key][b] = np.asarray(values)[idx].sum()
    return out


def tune_cutoff(y, amount, p, review_cost: float) -> tuple[float, int]:
    """Score cutoff minimising realised cost on ``(y, amount, p)``.

    Flagging the top-``k`` scores, the cost is ``C * k + (loss not yet captured)``, so the exact
    optimum over all ``k`` follows from one cumulative sum. Returns ``(tau, k)``; ``tau`` is
    ``inf`` when reviewing nothing is cheapest.
    """
    y = np.asarray(y)
    amount = np.asarray(amount, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="stable")
    captured = np.cumsum((y * amount)[order])
    total = float((y * amount).sum())
    k_grid = np.arange(0, len(p) + 1)
    cost = review_cost * k_grid + total - np.r_[0.0, captured]
    k_star = int(np.argmin(cost))
    tau = float("inf") if k_star == 0 else float(p[order][k_star - 1])
    return tau, k_star


def cost_at_cutoff(y, amount, p, tau: float, review_cost: float) -> dict[str, float]:
    """Per-100k economics of a fixed score cutoff, so splits of different size compare."""
    y = np.asarray(y)
    amount = np.asarray(amount, dtype=float)
    p = np.asarray(p, dtype=float)
    flag = p >= tau
    cost = float(np.where(flag, review_cost, y * amount).sum())
    loss = float((y * amount).sum())
    return dict(
        alerts_per_100k=float(flag.mean() * 1e5),
        cost_per_100k=cost / len(y) * 1e5,
        no_model_loss_per_100k=loss / len(y) * 1e5,
        savings_share=(loss - cost) / loss if loss > 0 else float("nan"),
    )
