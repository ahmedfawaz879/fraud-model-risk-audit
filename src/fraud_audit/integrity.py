"""Decidable data-integrity checks.

"Decidable" means the check has a yes/no answer from the data alone, with no modelling judgement:
an id appearing in two splits either does or does not. The toolkit's :func:`cmvt.integrity.leakage_scan`
supplies four of them; one of those four is *uninformative on this dataset* and is annotated as
such rather than quietly counted as a pass -- the file carries no card identifier, so "no entity
appears in both splits" is trivially true and says nothing.

The second check here is the one that does bite: rows whose feature values are byte-identical to a
training row, found in a later split. Those are duplicates the time split cannot separate.
"""

from __future__ import annotations

import pandas as pd
from cmvt.integrity import leakage_scan

from .data import RAW_FEATURE_COLUMNS
from .split import TemporalSplit

__all__ = ["UNINFORMATIVE_CHECK", "integrity_scan", "cross_split_duplicates"]

UNINFORMATIVE_CHECK = "patient_overlap_train_test"

_NOTES = {
    UNINFORMATIVE_CHECK: "no entity id exists, so card overlap is trivially 0 and NOT informative",
    "duplicate_ids_in_train": "row ids are unique by construction",
}


def integrity_scan(split: TemporalSplit) -> pd.DataFrame:
    """Run the toolkit's leakage checks between ``fit`` and ``oot``, annotated for this dataset."""
    scan = leakage_scan(
        split.frames["fit"],
        split.frames["oot"],
        id_col="row_id",
        time_col="Time",
        label_col="Class",
        split_time=split.split_time,
    )
    scan["note"] = [_NOTES.get(str(check), "") for check in scan["check"]]
    return scan


def cross_split_duplicates(split: TemporalSplit) -> pd.DataFrame:
    """Count rows in each later split whose raw features exactly match a ``fit`` row.

    Hashing ``V1..V28`` and ``Amount`` (not ``Time``) is the point: two transactions at different
    moments with identical feature vectors are indistinguishable to the model, so one of them
    being in training is leakage that a temporal split does not prevent.
    """

    def hashes(frame: pd.DataFrame):
        return pd.util.hash_pandas_object(frame[RAW_FEATURE_COLUMNS], index=False).to_numpy()

    fit_hashes = set(hashes(split.frames["fit"]).tolist())
    rows = []
    for name in ("intime", "recal", "oot"):
        frame = split.frames[name]
        mask = pd.Series(hashes(frame)).isin(fit_hashes).to_numpy()
        rows.append(
            dict(
                split=name,
                rows_identical_to_a_fit_row=int(mask.sum()),
                of_which_fraud=int(split.y[name][mask].sum()),
            )
        )
    return pd.DataFrame(rows)
