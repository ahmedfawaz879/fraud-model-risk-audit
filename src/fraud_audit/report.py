"""Verdicts, and the Markdown report generated from them.

Two rules govern this module:

1. **No number is hand-written.** Every figure in the prose is interpolated from the tables, so
   the text cannot disagree with the evidence, and re-running on different data rewrites it.
2. **Verdicts compare intervals to bands declared in advance**, not point estimates to
   expectations. A point estimate inside a tolerance band proves nothing when its interval spans
   the band, so :func:`band_state` has three outcomes rather than two, and "UNCERTAIN" is a real,
   reportable answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import AuditConfig
from .drift import DOMAIN_AUC_THRESHOLD, PSI_MODERATE, DriftReport

__all__ = ["Verdicts", "band_state", "calibration_state", "ranking_verdict", "build_report"]

_RANKING_PROSE = {
    "HELD": "The model ranks transactions about as well later as it did in-time.",
    "DEGRADED": (
        "Ranking quality fell out-of-time by more than the declared tolerance, and the fall is "
        "statistically significant."
    ),
    "INCONCLUSIVE": (
        "AUROC fell by more than the declared tolerance, but at this sample size the fall cannot "
        "be told from noise."
    ),
}

_CALIBRATION_PROSE = {
    "IN": "Probabilities remain inside the declared tolerance.",
    "OUT": "Probabilities are outside the declared tolerance.",
    "UNCLEAR": (
        "The intervals straddle the tolerance, so probabilities can be called neither inside nor outside it."
    ),
}

_CALIBRATION_HEADLINE = {
    "IN": "WITHIN TOLERANCE",
    "OUT": "OUTSIDE TOLERANCE",
    "UNCLEAR": "UNCERTAIN (interval straddles the tolerance)",
}


def band_state(lo: float, hi: float, tol: tuple[float, float]) -> str:
    """Is the whole interval inside the tolerance band, wholly outside it, or straddling it?

    ``nan`` bounds (an unfittable GLM) yield ``"unclear"``: a metric that could not be estimated
    is never evidence of a pass.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unclear"
    if lo >= tol[0] and hi <= tol[1]:
        return "in"
    if hi < tol[0] or lo > tol[1]:
        return "out"
    return "unclear"


def calibration_state(metrics: Mapping[str, float], cfg: AuditConfig) -> str:
    """Combine the O/E and slope interval states into ``IN`` / ``OUT`` / ``UNCLEAR``.

    One interval wholly outside its band is enough to fail; a pass needs both wholly inside.
    """
    states = [
        band_state(metrics["oe_lo"], metrics["oe_hi"], cfg.tolerances.oe),
        band_state(metrics["slope_lo"], metrics["slope_hi"], cfg.tolerances.slope),
    ]
    if "out" in states:
        return "OUT"
    return "IN" if all(state == "in" for state in states) else "UNCLEAR"


def ranking_verdict(auc_drop: float, p_value: float, cfg: AuditConfig) -> str:
    """``HELD`` within tolerance; ``DEGRADED`` if a larger fall is significant; else ``INCONCLUSIVE``."""
    if auc_drop <= cfg.tolerances.auc_drop:
        return "HELD"
    return "DEGRADED" if p_value < 0.05 else "INCONCLUSIVE"


@dataclass(frozen=True)
class Verdicts:
    """The audit's conclusions, separated from their rendering."""

    ranking: str
    calibration_oot: str
    calibration_intime: str
    calibration_recalibrated: str
    recommendation: str
    gain_distinguishable_from_zero: bool
    recalibration_window_underpowered: bool

    @property
    def calibration_headline(self) -> str:
        return _CALIBRATION_HEADLINE[self.calibration_oot]

    def to_dict(self) -> dict[str, Any]:
        return dict(
            ranking=self.ranking,
            calibration_oot=self.calibration_oot,
            calibration_intime=self.calibration_intime,
            calibration_recalibrated=self.calibration_recalibrated,
            recommendation=self.recommendation,
            gain_distinguishable_from_zero=self.gain_distinguishable_from_zero,
            recalibration_window_underpowered=self.recalibration_window_underpowered,
        )


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def _interval(lo: float, hi: float, fmt: str = "{:.3f}") -> str:
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def _cause_text(drift: DriftReport) -> str:
    """One sentence naming whichever shifts the declared thresholds actually detected."""
    causes = []
    if drift.base_rate_moved:
        causes.append(
            f"the fraud base rate moved ({_pct(drift.rate_dev, 3)} to {_pct(drift.rate_oot, 3)}, "
            f"p {_p(drift.prior_p)})"
        )
    if drift.inputs_moved:
        causes.append(
            f"inputs moved (max feature PSI {drift.max_psi:.2f} on {drift.max_psi_feature}; "
            f"a classifier separates the periods with AUROC {drift.domain_auc:.2f})"
        )
    if causes:
        return "; ".join(causes)
    return (
        "no base-rate shift at p < 0.05 and no input drift above the declared thresholds "
        f"(PSI > {PSI_MODERATE:.2f} or domain-classifier AUROC > {DOMAIN_AUC_THRESHOLD:.2f}; "
        f"largest PSI {drift.max_psi:.3f} on {drift.max_psi_feature}, domain AUROC "
        f"{drift.domain_auc:.2f}), so any change in calibration points to a change in the "
        "input-to-fraud relationship or to sampling noise"
    )


def _recommendation(
    state_oot: str, state_intime: str, state_fixed: str, underpowered: bool, n_events: int, cfg: AuditConfig
) -> str:
    if state_oot == "IN":
        text = "No recalibration needed on this evidence; keep monitoring."
    elif state_oot == "UNCLEAR":
        text = (
            "The evidence is not decisive: gather more out-of-time events before acting. "
            "Recalibration is cheap and reversible if needed."
        )
    elif state_fixed == "IN":
        text = "Recalibrate (logistic) and re-test on the next period."
    else:
        text = (
            "Recalibration on this window did not bring the model back within tolerance; retrain, "
            "or collect a larger and more recent recalibration sample."
        )
    if underpowered and state_oot != "IN":
        text += f" Caution: the recalibration window holds only {n_events} events."
    if state_intime != "IN":
        text += (
            " Calibration was already off in-time, so part of the problem is model fitting "
            "(over-fitting is the usual cause), not only drift."
        )
    return text


def build_report(result: Any) -> tuple[str, Verdicts]:
    """Render the audit report from an :class:`fraud_audit.pipeline.AuditResult`.

    Returns ``(markdown, verdicts)``. Kept separate from the pipeline so the verdict rules can be
    tested on hand-built inputs without fitting a model.
    """
    cfg: AuditConfig = result.config
    subject = cfg.subject
    disc, cal, drops = (
        result.tables["discrimination"],
        result.tables["calibration"],
        result.tables["auroc_change"],
    )
    fixed_cutoff = result.tables["fixed_cutoff"]

    d_in = disc[(disc.model == subject) & (disc.split == "intime")].iloc[0]
    d_oo = disc[(disc.model == subject) & (disc.split == "oot")].iloc[0]
    change = drops[drops.model == subject].iloc[0]
    c_in = cal[(cal.model == subject) & (cal.split == "intime")].iloc[0]
    c_oo = cal[(cal.model == subject) & (cal.split == "oot")].iloc[0]
    c_fx = result.calibration_recalibrated

    auc_drop = float(d_in.auroc) - float(d_oo.auroc)
    verdict_rank = ranking_verdict(auc_drop, float(change.p_unpaired_delong), cfg)
    state_in = calibration_state(c_in, cfg)
    state_oot = calibration_state(c_oo, cfg)
    state_fixed = calibration_state(c_fx, cfg)

    n_events = result.split.events("recal")
    underpowered = n_events < cfg.tolerances.min_events_recal
    gain_lo, gain_hi = result.money["gain_ci"]
    gain_significant = bool(gain_lo > 0 or gain_hi < 0)

    verdicts = Verdicts(
        ranking=verdict_rank,
        calibration_oot=state_oot,
        calibration_intime=state_in,
        calibration_recalibrated=state_fixed,
        recommendation=_recommendation(state_oot, state_in, state_fixed, underpowered, n_events, cfg),
        gain_distinguishable_from_zero=gain_significant,
        recalibration_window_underpowered=underpowered,
    )

    money = result.money
    prov_lo, prov_hi = money["prov_raw_ci"]
    prov_text = (
        "the interval excludes 1.0"
        if (prov_lo > 1 or prov_hi < 1)
        else "the interval includes 1.0, so a provisioning error is not established"
    )
    if state_oot == "IN":
        fix_text = "not needed" + (
            "; fitting it on this window would have pushed calibration out of tolerance, so a "
            "recalibration window must resemble the period it is used on"
            if state_fixed != "IN"
            else ""
        )
    else:
        fix_text = "**effective**" if state_fixed == "IN" else "**not sufficient**"

    dataset = result.dataset
    split = result.split
    dev, recal_df, oot_df = split.dev, split.frames["recal"], split.frames["oot"]
    scan, dup = result.tables["integrity"], result.tables["duplicates"]
    oot_dupes = int(dup.loc[dup.split == "oot", "rows_identical_to_a_fit_row"].iloc[0])
    tol = cfg.tolerances

    report = f"""# Model risk audit: {subject} fraud model

*Generated {date.today().isoformat()} · data `{dataset.path.name}` (sha256 `{dataset.sha256[:16]}`...) · cmvt {result.cmvt_version} · seed {cfg.seed}*

## 1. Verdict

| Dimension | Result | Evidence (out-of-time unless stated) |
|---|---|---|
| Ranking | **{verdict_rank}** | AUROC {d_in.auroc:.3f} in-time {_interval(d_in.auroc_lo, d_in.auroc_hi)} to {d_oo.auroc:.3f} out-of-time {_interval(d_oo.auroc_lo, d_oo.auroc_hi)}; change {-auc_drop:+.3f}, unpaired DeLong p {_p(float(change.p_unpaired_delong))} |
| Calibration | **{verdicts.calibration_headline}** | O/E {c_oo.oe:.2f} {_interval(c_oo.oe_lo, c_oo.oe_hi, "{:.2f}")}, slope {c_oo.slope:.2f} {_interval(c_oo.slope_lo, c_oo.slope_hi, "{:.2f}")}, intercept {c_oo.citl:+.2f}; in-time O/E {c_in.oe:.2f}, slope {c_in.slope:.2f} |
| Money | realised loss was **{money["prov_raw"]:.2f}x** the loss implied by the probabilities | 95% interval {_interval(prov_lo, prov_hi, "{:.2f}")} ({prov_text}); net savings of the review rule {money["savings_raw"]:,.0f} (raw) vs {money["savings_fixed"]:,.0f} (recalibrated) of at most {money["savings_oracle"]:,.0f} |
| Recalibration | {fix_text} | after: O/E {c_fx["oe"]:.2f}, slope {c_fx["slope"]:.2f}; gain {money["gain"]:+,.0f} {_interval(gain_lo, gain_hi, "{:,.0f}")} ({"distinguishable from zero" if gain_significant else "not distinguishable from zero"}) |

**Recommendation.** {verdicts.recommendation}

Tolerances (declared before scoring, illustrative): ranking is HELD if AUROC falls by at most {tol.auc_drop}, DEGRADED if it falls by more and the fall is significant, INCONCLUSIVE if it falls by more but is not significant. Calibration is WITHIN tolerance only if the whole 95% interval of O/E ({tol.oe[0]:g}, {tol.oe[1]:g}) and of the slope ({tol.slope[0]:g}, {tol.slope[1]:g}) lies inside the band, OUTSIDE if either interval lies wholly outside, otherwise UNCERTAIN.

## 2. Scope and data

{dataset.n_raw:,} transactions spanning {dataset.span_hours:.1f} hours, {dataset.frauds_raw} frauds. {dataset.n_removed} exact duplicate rows removed before splitting.
Split by time: development {len(dev):,} rows ({int(dev.Class.sum())} frauds), recalibration window {len(recal_df):,} ({n_events}), out-of-time test {len(oot_df):,} ({int(oot_df.Class.sum())}).
In-time number = random {_pct(cfg.intime_val_frac, 0)} hold-out inside the development period. Features: V1-V28 and log(Amount); `Time` excluded.
Integrity: {int((~scan.passed).sum())} of {len(scan)} decidable checks failed (the entity-overlap check is uninformative without a card id); exact feature duplicates of training rows found in the out-of-time set: {oot_dupes}.

## 3. Findings

**Ranking.** {_RANKING_PROSE[verdict_rank]} Average precision {d_in.auprc:.3f} in-time to {d_oo.auprc:.3f} out-of-time (no-skill baseline {d_oo.auprc_baseline:.4f}).

**Calibration.** {_CALIBRATION_PROSE[state_oot]} {"In-time they were already outside it, so this is not purely a drift effect. " if state_in == "OUT" else ""}Out-of-time the model {"under-predicts" if c_oo.oe > 1 else "over-predicts"} fraud (O/E {c_oo.oe:.2f}) and is {"over-confident" if c_oo.slope < 1 else "under-confident"} (slope {c_oo.slope:.2f}).
Equal-count ECE {c_in.ece_equal_mass:.5f} in-time vs {c_oo.ece_equal_mass:.5f} out-of-time (an average over bins, so it can hide a level or slope problem; the verdict rests on O/E and slope).

**What changed.** {_cause_text(result.drift)[0].upper() + _cause_text(result.drift)[1:]}.

**Money.** At a review cost of {cfg.review_cost:g} per alert, realised fraud loss out-of-time was {money["realised_loss"]:,.0f}; the probabilities implied {money["predicted_loss_raw"]:,.0f}. The review rule saved {money["savings_raw"]:,.0f} against no model ({_pct(money["savings_raw"] / money["savings_oracle"], 1)} of the oracle bound); with recalibrated probabilities {money["savings_fixed"]:,.0f}.
A fixed score cutoff tuned in-time produced {fixed_cutoff.alerts_per_100k.iloc[0]:.0f} alerts per 100k transactions in-time and {fixed_cutoff.alerts_per_100k.iloc[1]:.0f} out-of-time.

## 4. Remediation

{cfg.primary_recalibrator} recalibration fitted on the {len(recal_df):,}-row recalibration window ({n_events} events). It is monotone, so AUROC is unchanged ({money["auroc_fixed"]:.4f} vs {money["auroc_raw"]:.4f}).
{f"The recalibration window holds fewer than {tol.min_events_recal} events, so the fitted slope and intercept are imprecise." if underpowered else ""}
Isotonic and intercept-only variants are reported in `tables/table_recalibration.csv` as sensitivity checks.

## 5. Limitations

* **Horizon.** {dataset.span_hours:.0f} hours of data: this tests short-term drift only, not seasonal or adversarial drift over months.
* **No entity id.** Cards cannot be identified, so bootstrap intervals resample rows; if one card has many rows the intervals are too narrow.
* **Anonymised features.** V1-V28 are PCA components; drift cannot be explained in business terms.
* **Few events.** {int(oot_df.Class.sum())} out-of-time frauds; amount-weighted quantities depend on a handful of large frauds and carry wide intervals.
* **Cost model is an assumption** (review cost {cfg.review_cost:g}; reviewed fraud always stopped; no customer friction). See `tables/table_review_cost_sensitivity.csv`.
* **Lightly regularised, not tuned.** Fixed settings plus early stopping on the fit rows, no hyper-parameter search; any residual over-fitting would show up as miscalibration in-time as well.
* **One split, one seed.** Optimisation variance of the models is not included.
* **Tolerances are illustrative**; the model-risk owner should set them.

## 6. Reproducibility

Python {result.environment["python"]}, numpy {result.environment["numpy"]}, pandas {result.environment["pandas"]}, scikit-learn {result.environment["scikit-learn"]}; every random choice is seeded with {cfg.seed}; `run_config.json` next to this report is the full parameter set.
"""
    return report, verdicts


def summarise_tables(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Row counts per saved table: a cheap manifest for a reviewer checking nothing is missing."""
    return pd.DataFrame(
        [dict(table=name, rows=len(frame), columns=len(frame.columns)) for name, frame in tables.items()]
    )
