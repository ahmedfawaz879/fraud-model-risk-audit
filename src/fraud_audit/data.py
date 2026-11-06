"""Loading, validating and fingerprinting the transaction file.

The dataset is Kaggle *Credit Card Fraud Detection* (``mlg-ulb/creditcardfraud``): 284,807 card
transactions over two days, 492 frauds. ``V1..V28`` are anonymised PCA components, ``Amount`` is
the transaction value, ``Time`` is seconds since the first transaction.

``Time`` is deliberately **not** a feature: under a temporal split a model can only extrapolate it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AuditConfig

__all__ = [
    "Dataset",
    "V_COLUMNS",
    "FEATURES",
    "RAW_FEATURE_COLUMNS",
    "REQUIRED_COLUMNS",
    "load_transactions",
    "resolve_data_path",
    "sha256_file",
]

log = logging.getLogger(__name__)

V_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
FEATURES: list[str] = V_COLUMNS + ["log_amount"]
RAW_FEATURE_COLUMNS: list[str] = V_COLUMNS + ["Amount"]
REQUIRED_COLUMNS: list[str] = ["Time", "Amount", "Class"] + V_COLUMNS


@dataclass(frozen=True)
class Dataset:
    """A validated, time-ordered transaction table plus its provenance."""

    frame: pd.DataFrame
    path: Path
    sha256: str
    exclusions: pd.DataFrame
    n_raw: int
    frauds_raw: int

    @property
    def span_hours(self) -> float:
        return float(self.frame["Time"].max()) / 3600.0

    @property
    def n_removed(self) -> int:
        return int(self.exclusions.iloc[-1].get("removed", 0) or 0)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 so a report can name the exact bytes it audited."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(cfg: AuditConfig) -> Path:
    """First existing path among ``data_path``, ``data_candidates``, then a Kaggle-style search."""
    if cfg.data_path:
        path = Path(cfg.data_path)
        if not path.exists():
            raise FileNotFoundError(f"data_path does not exist: {path}")
        return path
    for candidate in cfg.data_candidates:
        path = Path(candidate)
        if path.exists():
            return path
    root = Path(cfg.kaggle_input_root)
    if root.exists():
        hits = sorted(root.rglob("creditcard*.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "creditcard.csv not found. Download the Kaggle 'Credit Card Fraud Detection' dataset "
        "(mlg-ulb/creditcardfraud) to data/creditcard.csv, or pass --data PATH. Searched: "
        f"{[cfg.data_path, *cfg.data_candidates]}"
    )


def load_transactions(cfg: AuditConfig) -> Dataset:
    """Read the CSV, assert the schema, optionally de-duplicate, and order by ``Time``.

    De-duplication happens *before* splitting: identical rows that straddle a split boundary are
    train/test leakage, and the count removed is reported rather than hidden.
    """
    path = resolve_data_path(cfg)
    digest = sha256_file(path)
    frame = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if not frame[REQUIRED_COLUMNS].notna().all().all():
        raise ValueError(f"{path} contains nulls in {REQUIRED_COLUMNS}")
    labels = set(pd.unique(frame["Class"]))
    if not labels <= {0, 1}:
        raise ValueError(f"Class must be binary 0/1, found {sorted(labels)}")

    n_raw, frauds_raw = len(frame), int(frame["Class"].sum())
    exclusions: list[dict[str, object]] = [dict(step="raw file", rows=n_raw, frauds=frauds_raw)]

    if cfg.dedupe_exact_rows:
        duplicated = frame.duplicated(keep="first")
        frame = frame.loc[~duplicated]
        exclusions.append(
            dict(
                step="drop exact duplicate rows",
                rows=len(frame),
                frauds=int(frame["Class"].sum()),
                removed=int(duplicated.sum()),
                removed_frauds=int(frauds_raw - int(frame["Class"].sum())),
            )
        )

    frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))
    frame["log_amount"] = np.log1p(frame["Amount"])

    log.info(
        "loaded %s rows (%s frauds) from %s | sha256 %s... | span %.1f hours",
        f"{len(frame):,}",
        int(frame["Class"].sum()),
        path,
        digest[:16],
        float(frame["Time"].max()) / 3600.0,
    )
    return Dataset(
        frame=frame,
        path=path,
        sha256=digest,
        exclusions=pd.DataFrame(exclusions),
        n_raw=n_raw,
        frauds_raw=frauds_raw,
    )
