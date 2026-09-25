"""Captions for the regenerated ladder figures (Figures 23 and 24).

Shared by src/paper/ladder_figures.py (which embeds them in the PNGs) and
src/paper/figure_renumbering.py (which writes them to figure_map.csv), so the
two cannot drift apart. Every number is read from the corrected tables
(ladder_ng.csv, ladder_nyc.csv, nyc_headline_identical_trips.csv). Pandas
only, so it runs on the host as well as in the container.
"""
from __future__ import annotations

import pandas as pd

from config import settings

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
REF_CONFIG = "reference (no model fitted)"
ORIGINAL = "original (untuned, 300 trees)"
CAPACITY = "capacity-controlled (CV-selected on V0)"


def ref_mae(ladder: pd.DataFrame, rung: str) -> float:
    row = ladder[(ladder["configuration"] == REF_CONFIG) & (ladder["rung"] == rung)]
    return float(row["MAE_mean"].iloc[0])


def nyc_v0_gated() -> dict[str, dict[str, float]]:
    """Per configuration: V0's MAE on the gated test trips (mean, sd across seeds) and on its own ungated set."""
    h = pd.read_csv(PAPER_DIR / "nyc_headline_identical_trips.csv")
    out = {}
    for config in (ORIGINAL, CAPACITY):
        m = h[(h["configuration"] == config) & (h["record_type"] == "mean_across_seeds")].iloc[0]
        s = h[(h["configuration"] == config) & (h["record_type"] == "sd_across_seeds")].iloc[0]
        out[config] = {"gated_mean": float(m["v0_gated_set_mae"]), "gated_sd": float(s["v0_gated_set_mae"]),
                       "ungated_mean": float(m["v0_ungated_mae"])}
    return out


def nigeria_caption() -> str:
    ladder = pd.read_csv(PAPER_DIR / "ladder_ng.csv")
    n_seeds = int(ladder[ladder["configuration"] == ORIGINAL]["n_seeds"].iloc[0])
    return (
        "Nigerian petrol price, one month ahead: mean absolute error (NGN/litre) by ladder rung, for the "
        "original (300 trees) and capacity-controlled (12 trees) configurations. "
        f"Points are the mean of {n_seeds} seeds (LightGBM random_state 1-{n_seeds}); error bars are one standard "
        "deviation across seeds. Task A does no sampling, so the spread is model randomness only. "
        f"Dashed line: random walk (REF_heuristic, MAE {ref_mae(ladder, 'REF_heuristic'):,.2f}); dotted line: "
        f"training mean (REF_mean, MAE {ref_mae(ladder, 'REF_mean'):,.2f}); both fit no model. "
        "V3a (hollow markers) is computed with future months present and is shown only for comparison with V3b. "
        "Paired comparisons with intervals are in paired_comparisons_ng_v2.csv."
    )


def nyc_caption() -> str:
    ladder = pd.read_csv(PAPER_DIR / "ladder_nyc.csv")
    n_seeds = int(ladder[ladder["configuration"] == ORIGINAL]["n_seeds"].iloc[0])
    v0 = nyc_v0_gated()
    o, c = v0[ORIGINAL], v0[CAPACITY]
    return (
        "NYC trip duration, pickup-time features: mean absolute error (seconds) by ladder rung, for the "
        "original (300 trees) and capacity-controlled (182 trees) configurations. "
        f"Points are the mean of {n_seeds} seeds (each seed draws its own content-addressed bucket of about "
        "1.96 million training and 421,000 test trips and uses its own random_state); error bars are one "
        "standard deviation across seeds. "
        f"*V0 is plotted at its MAE restricted to exactly the trips in the gated test set ({o['gated_mean']:,.2f} s "
        f"original, {c['gated_mean']:,.2f} s capacity-controlled). Scored on its own, ungated test set, V0's MAE "
        f"is {o['ungated_mean']:,.2f} s and {c['ungated_mean']:,.2f} s. V1 onward, and the reference predictors, "
        "are scored on the gated test set. "
        f"Dashed line: distance over average speed (REF_heuristic, MAE {ref_mae(ladder, 'REF_heuristic'):,.2f} s); "
        f"dotted line: training mean (REF_mean, MAE {ref_mae(ladder, 'REF_mean'):,.2f} s); both fit no model. "
        "The y-axis is broken and does not start at zero: the reference lines lie far above every rung. "
        "V3a (hollow markers) is computed with future months present and is shown only for comparison with V3b."
    )
