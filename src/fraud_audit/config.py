"""Audit configuration.

Everything a reviewer may want to challenge lives here, in one typed object loaded from
YAML. Nothing downstream reads a module-level constant for a decision; the config is passed
explicitly so that a run is fully described by the file plus the CLI overrides, both of which
are written to ``run_config.json`` next to the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

__all__ = ["AuditConfig", "Tolerances", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "credit_fraud.yaml"


@dataclass(frozen=True)
class Tolerances:
    """Pre-declared pass/fail bands, set *before* any out-of-time number is seen.

    The values shipped in ``configs/credit_fraud.yaml`` are illustrative. In a real
    engagement the model-risk owner sets them, and the fact that they were fixed in advance
    is what makes the verdict in the generated report a finding rather than a choice.
    """

    auc_drop: float = 0.02
    """Largest AUROC fall that still counts as ranking having HELD."""

    oe: tuple[float, float] = (0.80, 1.25)
    """Acceptable band for observed/expected events."""

    slope: tuple[float, float] = (0.80, 1.20)
    """Acceptable band for the calibration slope."""

    min_events_recal: int = 100
    """Rule-of-thumb floor for the number of events in a recalibration sample."""

    def __post_init__(self) -> None:
        if not self.auc_drop >= 0:
            raise ValueError(f"auc_drop must be non-negative, got {self.auc_drop}")
        for name in ("oe", "slope"):
            lo, hi = getattr(self, name)
            if not lo < hi:
                raise ValueError(f"tolerance {name} must be (lo, hi) with lo < hi, got {(lo, hi)}")
        if self.min_events_recal < 1:
            raise ValueError("min_events_recal must be >= 1")


@dataclass(frozen=True)
class AuditConfig:
    """A complete, reproducible description of one audit run."""

    seed: int = 42

    # ── Data ───────────────────────────────────────────────────────────────────────
    data_path: str | None = None
    """Explicit path to ``creditcard.csv``. When ``None``, ``data_candidates`` is searched."""

    data_candidates: tuple[str, ...] = (
        "data/creditcard.csv",
        "creditcard.csv",
        "/kaggle/input/creditcardfraud/creditcard.csv",
    )
    kaggle_input_root: str = "/kaggle/input"
    """Searched recursively for ``creditcard*.csv`` if no candidate path exists."""

    dedupe_exact_rows: bool = True
    """Drop exact duplicate rows before splitting. Logged in the exclusions table."""

    # ── Strict temporal split ──────────────────────────────────────────────────────
    train_frac: float = 0.55
    """Earliest fraction of transactions forming the development period."""

    recal_frac: float = 0.15
    """Next fraction: the recalibration window. The remainder is the out-of-time test."""

    intime_val_frac: float = 0.20
    """Random hold-out *inside* the development period: the "headline" number."""

    # ── Model under audit ──────────────────────────────────────────────────────────
    subject: str = "Gradient boosting"
    """Declared before any out-of-time result is seen. The other models are comparators."""

    primary_recalibrator: str = "Logistic (slope + intercept)"
    """Declared in advance; the other recalibrators are sensitivity checks."""

    # ── Economics (assumptions, not measurements) ──────────────────────────────────
    review_cost: float = 5.0
    """Currency units per transaction sent to manual review."""

    review_cost_grid: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0)
    """Review costs used for the sensitivity table."""

    # ── Uncertainty ────────────────────────────────────────────────────────────────
    n_boot: int = 500
    """Bootstrap resamples for AUPRC and every currency total."""

    tolerances: Tolerances = field(default_factory=Tolerances)

    # ── Output ─────────────────────────────────────────────────────────────────────
    output_dir: str = "audit_outputs"
    make_figures: bool = True

    def __post_init__(self) -> None:
        for name in ("train_frac", "recal_frac", "intime_val_frac"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {value}")
        if self.train_frac + self.recal_frac >= 1.0:
            raise ValueError(
                "train_frac + recal_frac must leave rows for the out-of-time test, got "
                f"{self.train_frac} + {self.recal_frac}"
            )
        if self.n_boot < 1:
            raise ValueError("n_boot must be >= 1")
        if self.review_cost <= 0:
            raise ValueError("review_cost must be positive")
        if not self.review_cost_grid:
            raise ValueError("review_cost_grid must not be empty")

    # ── Construction ───────────────────────────────────────────────────────────────
    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> AuditConfig:
        """Build a config from a plain dict, coercing YAML lists into tuples."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")

        kwargs = dict(mapping)
        tolerances = kwargs.pop("tolerances", None)
        if tolerances is not None:
            if isinstance(tolerances, Tolerances):
                kwargs["tolerances"] = tolerances
            else:
                tol_unknown = set(tolerances) - set(Tolerances.__dataclass_fields__)
                if tol_unknown:
                    raise ValueError(f"unknown tolerance keys: {sorted(tol_unknown)}")
                tol = dict(tolerances)
                for key in ("oe", "slope"):
                    if key in tol:
                        tol[key] = tuple(float(v) for v in tol[key])
                kwargs["tolerances"] = Tolerances(**tol)
        for key in ("data_candidates", "review_cost_grid"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides: Any) -> AuditConfig:
        """Load ``configs/credit_fraud.yaml`` (or ``path``) and apply keyword overrides.

        Overrides whose value is ``None`` are ignored, so a CLI can pass every flag through
        without having to know which ones the user actually set.
        """
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level")
        cfg = cls.from_mapping(raw)
        return cfg.with_overrides(**overrides)

    def with_overrides(self, **overrides: Any) -> AuditConfig:
        """Return a copy with the non-``None`` overrides applied."""
        applied = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(applied) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return replace(self, **applied)

    # ── Serialisation ──────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-ready view, used for ``run_config.json`` and for provenance."""
        out = asdict(self)
        out["data_candidates"] = list(self.data_candidates)
        out["review_cost_grid"] = list(self.review_cost_grid)
        out["tolerances"]["oe"] = list(self.tolerances.oe)
        out["tolerances"]["slope"] = list(self.tolerances.slope)
        return out
