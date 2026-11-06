"""The four figures, and the one plot style they share.

Design notes, because they are decisions rather than defaults:

* **Three fixed hues, assigned by meaning, never cycled**: blue = in-time, orange = out-of-time,
  green = recalibrated. A reader who learns the key once can read all four figures. Grey is a
  neutral reserved for reference marks (the identity line, the oracle bound) and is never a series.
* The series hues are colorblind-safe as a set (worst adjacent-pair CVD dE 9.2, normal-vision
  dE 27.6). Green sits slightly under 3:1 contrast against the surface, so identity is carried
  **redundantly** -- a legend, a distinct marker shape per series, and a CSV in ``tables/``
  holding every plotted number.
* Reliability is drawn on **log-log** axes: the scores of a 0.2%-prevalence model span four
  decades, and on linear axes every point collapses into the corner.
* Bands with zero observed events are omitted from the figures (0 cannot be drawn on a log axis)
  but kept in ``band_table``, where they still constrain the upper bound.

Writing a figure never changes a number: these functions read the same arrays the tables do.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import band_table, ci  # noqa: E402

__all__ = [
    "SURFACE",
    "INK",
    "BLUE",
    "ORANGE",
    "GREEN",
    "GREY",
    "apply_style",
    "figure_discrimination",
    "figure_reliability_grid",
    "figure_recalibration",
    "figure_money",
]

# Neutral ink and surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# The three series hues, in fixed meaning order, plus the reserved neutral.
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8b8a85"

MARKERS = ("o", "s", "^")
"""Redundant identity: in-time, out-of-time, recalibrated keep these shapes everywhere."""


def apply_style() -> None:
    """Install the audit's plot style on the global rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def figure_discrimination(discrimination: pd.DataFrame, path: Path) -> Path:
    """Fig 1. AUROC with DeLong intervals, one row per model, in-time against out-of-time."""
    names = list(dict.fromkeys(discrimination["model"]))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    series = (
        ("intime", BLUE, MARKERS[0], "In-time (random hold-out)", 0.14),
        ("oot", ORANGE, MARKERS[1], "Out-of-time (strict)", -0.14),
    )
    for split, colour, marker, label, offset in series:
        part = discrimination[discrimination["split"] == split].set_index("model").loc[names]
        ax.errorbar(
            part["auroc"],
            np.arange(len(names)) + offset,
            xerr=[part["auroc"] - part["auroc_lo"], part["auroc_hi"] - part["auroc"]],
            fmt=marker,
            color=colour,
            ms=6,
            lw=1.6,
            capsize=0,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
        )
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("AUROC (95% DeLong interval)")
    ax.set_title("Ranking quality, before and after the time split", pad=26)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    return _save(fig, path)


def _plot_reliability(ax, series: Iterable[tuple[str, str, str, np.ndarray, np.ndarray]], title: str) -> None:
    """Observed rate against mean prediction, per risk band, with Wilson bars."""
    ax.plot([1e-4, 1], [1e-4, 1], color=GREY, lw=1, ls="--", zorder=1)
    for label, colour, marker, y, p in series:
        table = band_table(y, p)
        table = table[table["events"] > 0]
        ax.errorbar(
            table["mean_pred"],
            table["observed"],
            yerr=[
                table["observed"] - table["obs_lo"],
                table["obs_hi"] - table["observed"],
            ],
            fmt=f"{marker}-",
            color=colour,
            lw=1.4,
            ms=5.5,
            capsize=0,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-4, 1)
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")


def figure_reliability_grid(
    y: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    path: Path,
) -> Path:
    """Fig 2. Reliability as small multiples, one panel per model.

    The grid adapts to the number of models so adding a fifth comparator does not silently
    drop it off the figure while it still appears in every table.
    """
    models = list(predictions)
    columns = 2 if len(models) > 1 else 1
    rows = -(-len(models) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.5 * columns, 4 * rows), sharex=True, sharey=True, squeeze=False
    )
    flat = axes.ravel()
    for ax, name in zip(flat, models, strict=False):
        _plot_reliability(
            ax,
            [
                ("In-time", BLUE, MARKERS[0], y["intime"], predictions[name]["intime"]),
                ("Out-of-time", ORANGE, MARKERS[1], y["oot"], predictions[name]["oot"]),
            ],
            name,
        )
    for ax in flat[len(models) :]:  # an odd model count leaves a blank cell; do not draw a frame
        ax.set_visible(False)
    flat[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Reliability by risk band: on the dashed line, predictions equal reality",
        y=0.995,
        fontsize=11,
    )
    return _save(fig, path)


def figure_recalibration(
    y_intime: np.ndarray,
    p_intime: np.ndarray,
    y_oot: np.ndarray,
    p_raw: np.ndarray,
    p_fixed: np.ndarray,
    subject: str,
    path: Path,
) -> Path:
    """Fig 3. The audited model's reliability before and after recalibration."""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    _plot_reliability(
        ax,
        [
            ("In-time", BLUE, MARKERS[0], y_intime, p_intime),
            ("Out-of-time, raw", ORANGE, MARKERS[1], y_oot, p_raw),
            ("Out-of-time, recalibrated", GREEN, MARKERS[2], y_oot, p_fixed),
        ],
        f"{subject}: before and after recalibration",
    )
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, path)


def figure_money(
    values: Sequence[float],
    boot: Sequence[np.ndarray],
    review_cost: float,
    path: Path,
) -> Path:
    """Fig 4. Net savings of the review rule: raw, recalibrated, and the oracle upper bound.

    Identity comes from the axis labels and a direct value label on every bar, so the grey
    oracle bar is a reference mark rather than a fourth series.
    """
    labels = ["Raw probabilities", "Recalibrated", "Oracle (upper bound)"]
    colours = [ORANGE, GREEN, GREY]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for i, (value, draws, colour) in enumerate(zip(values, boot, colours, strict=True)):
        lo, hi = ci(draws)
        ax.barh(i, value, color=colour, height=0.55)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
        ax.text(
            max(hi, value) + 0.01 * max(values),
            i,
            f"{value:,.0f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "Net savings vs no model, out-of-time\n"
        f"(currency units; review cost {review_cost:g}; whisker = 95% bootstrap)"
    )
    ax.set_title("What the probability rule earns")
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
