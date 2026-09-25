"""Regenerates the two ladder figures from the corrected tables:

  - Nigeria (ladder_ng.csv, 20 seeds)
  - NYC (ladder_nyc.csv, 5 seeds), with V0 plotted at its MAE restricted to
    the trips in the GATED test set (nyc_headline_identical_trips.csv)

Both plot the mean absolute error per rung as a point with a +/-1 SD error bar
across seeds, one series per hyperparameter configuration, with the random-walk /
heuristic and training-mean predictors as horizontal reference lines. Points
rather than bars: the NYC axis is broken and does not start at zero, which a
bar would misrepresent.

Writes outputs/paper/figures/fig_model_ladder_nigeria.png and
fig_model_ladder_nyc.png (the originals under outputs/figures/ are untouched);
src/paper/figure_renumbering.py copies them into figures_renumbered/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.viz import style as vizstyle  # selects the Agg backend before pyplot is used
import matplotlib.pyplot as plt
from src.paper.ladder_captions import (
    PAPER_DIR, REF_CONFIG, ORIGINAL, CAPACITY, ref_mae, nyc_v0_gated,
    nigeria_caption, nyc_caption,
)

FIG_DIR = PAPER_DIR / "figures"
CONFIG_STYLE = {
    ORIGINAL: dict(colour=vizstyle.PALETTE[0], marker="o", label="original (300 trees)", offset=-0.12),
    CAPACITY: dict(colour=vizstyle.PALETTE[2], marker="s", label="capacity-controlled", offset=0.12),
}


def series(ladder: pd.DataFrame, config: str, rungs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sub = ladder[ladder["configuration"] == config].set_index("rung")
    return sub.loc[rungs, "MAE_mean"].to_numpy(float), sub.loc[rungs, "MAE_sd"].to_numpy(float)


def draw_points(ax, x, rungs, config, mean, sd, label=True) -> None:
    st = CONFIG_STYLE[config]
    hollow = np.array([r == "V3a" for r in rungs])
    ax.errorbar(x + st["offset"], mean, yerr=sd, fmt="none", ecolor=st["colour"], elinewidth=1.3,
                capsize=3, zorder=3)
    ax.scatter((x + st["offset"])[~hollow], mean[~hollow], marker=st["marker"], s=46, color=st["colour"],
               zorder=4, label=st["label"] if label else None)
    ax.scatter((x + st["offset"])[hollow], mean[hollow], marker=st["marker"], s=46, facecolors="white",
               edgecolors=st["colour"], linewidths=1.6, zorder=4)


def nigeria_figure() -> None:
    ladder = pd.read_csv(PAPER_DIR / "ladder_ng.csv")
    rungs = ["V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b"]
    x = np.arange(len(rungs))
    ref_walk, ref_mean = ref_mae(ladder, "REF_heuristic"), ref_mae(ladder, "REF_mean")

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    top = ref_mean
    bottom = ref_walk
    for config in (ORIGINAL, CAPACITY):
        mean, sd = series(ladder, config, rungs)
        draw_points(ax, x, rungs, config, mean, sd)
        top, bottom = max(top, float((mean + sd).max())), min(bottom, float((mean - sd).min()))
    ax.axhline(ref_walk, color=vizstyle.PALETTE[3], linestyle="--", linewidth=1.4,
               label=f"random walk ({ref_walk:,.1f})")
    ax.axhline(ref_mean, color=vizstyle.MUTED_COLOUR, linestyle=":", linewidth=1.4,
               label=f"training mean ({ref_mean:,.1f})")
    ax.set_ylim(bottom - 15, top + 60)  # headroom so the legend clears the training-mean line
    ax.set_xticks(x)
    ax.set_xticklabels([r if r != "V3a" else "V3a\n(leaky)" for r in rungs], fontsize=10)
    ax.set_xlabel("Ladder rung", fontsize=10)
    ax.set_ylabel("MAE on held-out test set (NGN/litre)", fontsize=10)
    ax.set_title("Nigerian petrol price, one month ahead: MAE by rung, with seed spread",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()
    vizstyle.finish(fig, FIG_DIR / "fig_model_ladder_nigeria.png",
                    "Source: this platform's own gold layer (fact_fuel_price_monthly); "
                    "table: outputs/paper/ladder_ng.csv", nigeria_caption())


def nyc_figure() -> None:
    ladder = pd.read_csv(PAPER_DIR / "ladder_nyc.csv")
    rungs = ["V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b", "V5"]
    x = np.arange(len(rungs))
    ref_walk, ref_mean = ref_mae(ladder, "REF_heuristic"), ref_mae(ladder, "REF_mean")
    v0 = nyc_v0_gated()

    fig, (ax_top, ax) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 6.4),
                                     gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.08})
    lo, hi = np.inf, -np.inf
    for config in (ORIGINAL, CAPACITY):
        mean, sd = series(ladder, config, rungs)
        mean[0], sd[0] = v0[config]["gated_mean"], v0[config]["gated_sd"]
        draw_points(ax, x, rungs, config, mean, sd)
        lo, hi = min(lo, float((mean - sd).min())), max(hi, float((mean + sd).max()))
    ax.set_ylim(lo - 4, hi + 9)
    ax_top.set_ylim(ref_walk - 18, ref_mean + 18)
    ax_top.axhline(ref_walk, color=vizstyle.PALETTE[3], linestyle="--", linewidth=1.4,
                   label=f"distance / average speed ({ref_walk:,.1f} s)")
    ax_top.axhline(ref_mean, color=vizstyle.MUTED_COLOUR, linestyle=":", linewidth=1.4,
                   label=f"training mean ({ref_mean:,.1f} s)")

    ax_top.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
              color=vizstyle.TEXT_COLOUR, mec=vizstyle.TEXT_COLOUR, mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kw)

    ax.set_xticks(x)
    ax.set_xticklabels(["V0*" if r == "V0" else ("V3a\n(leaky)" if r == "V3a" else r) for r in rungs],
                       fontsize=10)
    ax.set_xlabel("Ladder rung", fontsize=10)
    ax.set_ylabel("MAE on held-out test set (seconds)", fontsize=10)
    ax.yaxis.set_label_coords(-0.075, 0.70)
    ax_top.set_title("NYC trip duration, pickup-time features: MAE by rung, with seed spread",
                     fontsize=11, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    th, tl = ax_top.get_legend_handles_labels()
    ax.legend(handles + th, labels + tl, fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()
    vizstyle.finish(fig, FIG_DIR / "fig_model_ladder_nyc.png",
                    "Source: this platform's own gold layer (fact_trip, fact_weather_daily) and the raw NYC "
                    "TLC 2024 corpus (V0 only)\nTables: outputs/paper/ladder_nyc.csv, "
                    "nyc_headline_identical_trips.csv", nyc_caption())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vizstyle.apply_style()
    nigeria_figure()
    nyc_figure()


if __name__ == "__main__":
    main()
