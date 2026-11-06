"""The audit, end to end.

``run_audit(cfg)`` is the whole study: load, split, check integrity, fit, measure discrimination
and calibration, attribute the drift, price the gap, recalibrate on a window the fix has to earn
its place on, and write a report whose every number comes from the tables beside it.

Order matters and is enforced by construction: the out-of-time period is scored in one place, at
the end, and no fitting or tuning decision upstream can see it.
"""

from __future__ import annotations

import json
import logging
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import evaluate, plots
from .config import AuditConfig
from .data import Dataset, load_transactions
from .drift import DriftReport, drift_report
from .integrity import cross_split_duplicates, integrity_scan
from .metrics import calibration_metrics, ci
from .models import fit_models, require_subject
from .money import bootstrap_totals, money_rows
from .recalibrate import get_recalibrator
from .report import Verdicts, build_report
from .split import TemporalSplit, temporal_split

__all__ = ["AuditResult", "run_audit", "package_versions"]

log = logging.getLogger(__name__)

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "cmvt")


def package_versions() -> dict[str, str]:
    """Versions of everything a result depends on, for the reproducibility section."""
    out = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unknown"
    return out


@dataclass
class AuditResult:
    """Everything one run produced, in memory. :meth:`save` puts it on disk."""

    config: AuditConfig
    dataset: Dataset
    split: TemporalSplit
    predictions: Mapping[str, Mapping[str, np.ndarray]]
    recalibrated: Mapping[str, Mapping[str, np.ndarray]]
    tables: dict[str, pd.DataFrame]
    drift: DriftReport
    money: dict[str, Any]
    calibration_recalibrated: dict[str, float]
    environment: dict[str, str]
    subject_raw_oot: np.ndarray
    subject_recalibrated_oot: np.ndarray
    report: str = ""
    verdicts: Verdicts | None = None
    figures: list[Path] = field(default_factory=list)

    @property
    def cmvt_version(self) -> str:
        return self.environment.get("cmvt", "unknown")

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Write the report, every table, the figures and ``run_config.json`` under ``output_dir``."""
        out = Path(output_dir or self.config.output_dir)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            frame.to_csv(out / "tables" / f"table_{name}.csv", index=False)
        (out / "model_risk_audit_report.md").write_text(self.report, encoding="utf-8")
        if self.config.make_figures:
            self.figures = _draw_figures(self, out / "figures")
        (out / "run_config.json").write_text(
            json.dumps(
                dict(
                    config=self.config.to_dict(),
                    environment=self.environment,
                    data_path=str(self.dataset.path),
                    data_sha256=self.dataset.sha256,
                    drift=self.drift.to_dict(),
                    verdicts=self.verdicts.to_dict() if self.verdicts else None,
                ),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s", out / "model_risk_audit_report.md")
        return out


def run_audit(cfg: AuditConfig) -> AuditResult:
    """Run the full audit and return its result. Pure with respect to disk until ``.save()``."""
    dataset = load_transactions(cfg)
    split = temporal_split(dataset.frame, cfg)
    log.info("split sizes: %s", dict(zip(split.table.split, split.table.rows, strict=True)))

    tables: dict[str, pd.DataFrame] = {
        "splits": split.table,
        "exclusions": dataset.exclusions,
        "integrity": integrity_scan(split),
        "duplicates": cross_split_duplicates(split),
    }

    _, predictions = fit_models(split, cfg.seed)
    require_subject(predictions, cfg.subject)
    subject = cfg.subject

    tables["discrimination"] = evaluate.discrimination_table(split, predictions, cfg)
    tables["auroc_change"] = evaluate.auroc_change_table(split, predictions)
    tables["calibration"] = evaluate.calibration_table(split, predictions)

    drift = drift_report(
        split.dev,
        split.frames["oot"],
        split.features,
        (predictions[subject]["fit"], predictions[subject]["oot"]),
        cfg,
    )
    tables["drift_psi"] = drift.psi

    tables["money_raw"] = evaluate.money_table(split, predictions, cfg.review_cost)
    tables["fixed_cutoff"] = evaluate.fixed_cutoff_table(split, predictions, cfg)
    tables["recalibration"], recalibrated = evaluate.recalibration_table(split, predictions, cfg)

    primary = get_recalibrator(cfg.primary_recalibrator).name
    raw_p = np.asarray(predictions[subject]["oot"])
    fixed_p = np.asarray(recalibrated[subject][primary])
    money, money_ci_table = _price_the_gap(split, raw_p, fixed_p, cfg)
    tables["money_ci"] = money_ci_table
    tables["review_cost_sensitivity"] = evaluate.review_cost_sensitivity(split, raw_p, fixed_p, cfg)

    result = AuditResult(
        config=cfg,
        dataset=dataset,
        split=split,
        predictions=predictions,
        recalibrated=recalibrated,
        tables=tables,
        drift=drift,
        money=money,
        calibration_recalibrated=calibration_metrics(split.y["oot"], fixed_p),
        environment=package_versions(),
        subject_raw_oot=raw_p,
        subject_recalibrated_oot=fixed_p,
    )
    result.report, result.verdicts = build_report(result)
    return result


def _price_the_gap(
    split: TemporalSplit, raw_p: np.ndarray, fixed_p: np.ndarray, cfg: AuditConfig
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Currency totals for the audited model, with paired bootstrap intervals.

    The raw and recalibrated policies are resampled on the *same* row indices, so the gain from
    recalibrating is a paired quantity and its interval is not the difference of two independent
    ones.
    """
    y, amount, cost = split.y["oot"], split.amount["oot"], cfg.review_cost
    _, cost_raw = money_rows(y, amount, raw_p, cost)
    _, cost_fixed = money_rows(y, amount, fixed_p, cost)
    per_row = dict(
        real=y * amount,
        pred_raw=raw_p * amount,
        pred_fixed=fixed_p * amount,
        cost_raw=cost_raw,
        cost_fixed=cost_fixed,
        oracle=y * np.minimum(cost, amount),
    )
    boot = bootstrap_totals(per_row, cfg.n_boot, cfg.seed)
    total = {key: float(values.sum()) for key, values in per_row.items()}

    savings = {
        "raw": total["real"] - total["cost_raw"],
        "fixed": total["real"] - total["cost_fixed"],
        "oracle": total["real"] - total["oracle"],
    }
    boot_savings = {
        "raw": boot["real"] - boot["cost_raw"],
        "fixed": boot["real"] - boot["cost_fixed"],
        "oracle": boot["real"] - boot["oracle"],
    }
    gain = total["cost_raw"] - total["cost_fixed"]
    boot_gain = boot["cost_raw"] - boot["cost_fixed"]
    prov_raw = total["real"] / total["pred_raw"]
    prov_fixed = total["real"] / total["pred_fixed"]
    boot_prov_raw = boot["real"] / boot["pred_raw"]
    boot_prov_fixed = boot["real"] / boot["pred_fixed"]

    money = dict(
        realised_loss=total["real"],
        predicted_loss_raw=total["pred_raw"],
        predicted_loss_fixed=total["pred_fixed"],
        prov_raw=prov_raw,
        prov_raw_ci=ci(boot_prov_raw),
        prov_fixed=prov_fixed,
        prov_fixed_ci=ci(boot_prov_fixed),
        savings_raw=savings["raw"],
        savings_fixed=savings["fixed"],
        savings_oracle=savings["oracle"],
        savings_raw_ci=ci(boot_savings["raw"]),
        savings_fixed_ci=ci(boot_savings["fixed"]),
        savings_oracle_ci=ci(boot_savings["oracle"]),
        gain=gain,
        gain_ci=ci(boot_gain),
        auroc_raw=float(roc_auc_score(y, raw_p)),
        auroc_fixed=float(roc_auc_score(y, fixed_p)),
        boot_savings=boot_savings,
    )

    table = pd.DataFrame(
        [
            dict(
                quantity="Realised fraud loss (out-of-time)", value=total["real"], **_lohi(ci(boot["real"]))
            ),
            dict(
                quantity="Realised / predicted loss, raw probabilities",
                value=prov_raw,
                **_lohi(ci(boot_prov_raw)),
            ),
            dict(
                quantity="Realised / predicted loss, recalibrated",
                value=prov_fixed,
                **_lohi(ci(boot_prov_fixed)),
            ),
            dict(
                quantity="Net savings vs no model, raw probabilities",
                value=savings["raw"],
                **_lohi(ci(boot_savings["raw"])),
            ),
            dict(
                quantity="Net savings vs no model, recalibrated",
                value=savings["fixed"],
                **_lohi(ci(boot_savings["fixed"])),
            ),
            dict(quantity="Gain from recalibrating (paired)", value=gain, **_lohi(ci(boot_gain))),
            dict(
                quantity="Oracle savings (upper bound)",
                value=savings["oracle"],
                **_lohi(ci(boot_savings["oracle"])),
            ),
        ]
    )
    return money, table


def _lohi(bounds: tuple[float, float]) -> dict[str, float]:
    return dict(lo=bounds[0], hi=bounds[1])


def _draw_figures(result: AuditResult, out: Path) -> list[Path]:
    """Render the four figures into ``out``."""
    cfg, split = result.config, result.split
    raw_p, fixed_p = result.subject_raw_oot, result.subject_recalibrated_oot
    plots.apply_style()
    paths = [
        plots.figure_discrimination(result.tables["discrimination"], out / "fig1_discrimination.png"),
        plots.figure_reliability_grid(split.y, result.predictions, out / "fig2_reliability.png"),
        plots.figure_recalibration(
            split.y["intime"],
            np.asarray(result.predictions[cfg.subject]["intime"]),
            split.y["oot"],
            raw_p,
            fixed_p,
            cfg.subject,
            out / "fig3_recalibration.png",
        ),
        plots.figure_money(
            [result.money["savings_raw"], result.money["savings_fixed"], result.money["savings_oracle"]],
            [
                result.money["boot_savings"]["raw"],
                result.money["boot_savings"]["fixed"],
                result.money["boot_savings"]["oracle"],
            ],
            cfg.review_cost,
            out / "fig4_money.png",
        ),
    ]
    log.info("wrote %d figures to %s", len(paths), out)
    return paths
