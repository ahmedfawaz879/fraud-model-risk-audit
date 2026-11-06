"""Model risk audit for a credit-card fraud model under a strict temporal split.

A fraud model looks good on a random hold-out. Does it still *rank* well on later transactions,
and are its *probabilities* still trustworthy? If the probabilities drift, what does that cost in
money, and does recalibration fix it?

The package reports discrimination and calibration separately, with uncertainty, before and after
a strict temporal split; diagnoses what changed; translates the gap into currency; and tests the
fix on data the fix never saw.

Typical use::

    from fraud_audit import AuditConfig, run_audit

    cfg = AuditConfig.from_yaml("configs/credit_fraud.yaml", data_path="data/creditcard.csv")
    result = run_audit(cfg)
    result.save()
    print(result.verdicts.ranking, result.verdicts.calibration_headline)
"""

from __future__ import annotations

from .config import AuditConfig, Tolerances
from .pipeline import AuditResult, run_audit

__all__ = ["AuditConfig", "AuditResult", "Tolerances", "run_audit", "__version__"]

__version__ = "0.1.0"
