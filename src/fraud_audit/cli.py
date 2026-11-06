"""Command-line entry point: ``fraud-audit`` (or ``python -m fraud_audit``).

Every flag overrides a field of :class:`~fraud_audit.config.AuditConfig`, and the resulting
config is written to ``run_config.json`` beside the report, so the artefacts always say how they
were produced.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, AuditConfig
from .pipeline import run_audit

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-audit",
        description=(
            "Audit a credit-card fraud model under a strict temporal split: ranking and "
            "calibration with uncertainty, drift attribution, the cost in currency, and a "
            "recalibration tested out-of-time."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config (default: configs/credit_fraud.yaml)",
    )
    parser.add_argument("--data", dest="data_path", type=str, default=None, help="path to creditcard.csv")
    parser.add_argument(
        "--output", dest="output_dir", type=str, default=None, help="directory for report, tables and figures"
    )
    parser.add_argument("--seed", type=int, default=None, help="master random seed")
    parser.add_argument("--subject", type=str, default=None, help="model under audit")
    parser.add_argument(
        "--review-cost", dest="review_cost", type=float, default=None, help="currency units per manual review"
    )
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=None, help="bootstrap resamples")
    parser.add_argument(
        "--no-figures",
        dest="make_figures",
        action="store_const",
        const=False,
        default=None,
        help="skip figure rendering",
    )
    parser.add_argument("--print-report", action="store_true", help="echo the report to stdout")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {
        key: getattr(args, key)
        for key in ("data_path", "output_dir", "seed", "subject", "review_cost", "n_boot", "make_figures")
    }
    try:
        cfg = AuditConfig.from_yaml(args.config, **overrides)
    except (OSError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        result = run_audit(cfg)
    except FileNotFoundError as exc:
        # A missing or unreadable dataset is a usage problem, not a failed audit: exit 2, as for
        # a bad config, so a caller can tell "you pointed me somewhere wrong" from "it broke".
        print(f"{exc}", file=sys.stderr)
        return 2

    out = result.save()
    if args.print_report:
        print(result.report)
    verdicts = result.verdicts
    assert verdicts is not None
    print(
        f"\nranking: {verdicts.ranking} | calibration: {verdicts.calibration_headline}\n"
        f"artefacts: {out.resolve()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
