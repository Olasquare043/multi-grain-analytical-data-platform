"""Publication figures for analyses A1-A11.

Every figure is built from the published CSV in ``outputs/tables/``, never from
a fresh query. That is deliberate: the figure in the paper and the table in the
appendix are then provably the same numbers, and a reader can reproduce any
chart from the CSV alone.

Each figure carries its units, its source attribution and -- where the analysis
has a caveat that changes how the chart should be read -- that caveat in the
footer rather than buried in the prose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from config import settings
from src.utils.logging_setup import get_logger, stage
from src.viz import style

LOG = get_logger(__name__)

NYC = settings.FIGURE_ATTRIBUTION_NYC
WFP = settings.FIGURE_ATTRIBUTION_WFP
NBS = settings.FIGURE_ATTRIBUTION_NBS


def _table(stem: str) -> pd.DataFrame | None:
    """Load an analysis result table, or warn and return None."""
    path = settings.TABLES_DIR / f"{stem}.csv"
    if not path.exists():
        LOG.warning("figure skipped: %s not found. Run 'make analysis' first.",
                    path.name)
        return None
    return pd.read_csv(path)


def _fig(path_stem: str) -> Path:
    return settings.FIGURES_DIR / f"{path_stem}.{settings.FIGURE_FORMAT}"


def _real_boroughs(frame: pd.DataFrame, column: str = "borough") -> pd.DataFrame:
    """Drop the unattributable pseudo-borough from a chart (kept in the CSV)."""
    return frame[~frame[column].isin(["Unattributed", "Unknown", "N/A"])]


# --------------------------------------------------------------------------- #
# A1
# --------------------------------------------------------------------------- #
def fig_a1() -> list[Path]:
    frame = _table("A1_demand_profile")
    if frame is None:
        return []
    frame = _real_boroughs(frame)
    written: list[Path] = []

    # ---- heatmap: hour x day-of-week, one panel per borough
    boroughs = (
        frame.groupby("borough")["trip_count"].sum().sort_values(ascending=False)
        .index.tolist()
    )
    # 2 columns (not 3): keeps each panel wide enough that its 9pt+ tick and
    # colorbar labels stay legible once the whole figure is printed at
    # roughly 8.5in wide, at the cost of an extra row.
    ncols = min(2, len(boroughs))
    nrows = int(np.ceil(len(boroughs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             squeeze=False)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for index, borough in enumerate(boroughs):
        ax = axes[index // ncols][index % ncols]
        sub = frame[frame["borough"] == borough]
        grid = (
            sub.pivot_table(index="day_of_week", columns="pickup_hour",
                            values="pct_of_borough_trips", aggfunc="sum")
            .reindex(index=range(1, 8), columns=range(24))
        )
        mesh = ax.imshow(grid.values, aspect="auto", cmap=style.SEQUENTIAL_CMAP,
                         origin="upper", interpolation="nearest")
        ax.set_title(f"{borough} ({int(sub['trip_count'].sum()):,} trips)",
                     fontsize=11)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=9)
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_labels, fontsize=9)
        ax.set_xlabel("Hour of pickup (local)", fontsize=9.5)
        ax.grid(False)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=9)
        bar.set_label("% of borough trips", fontsize=9)

    for spare in range(len(boroughs), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")

    fig.suptitle("Yellow taxi demand by hour and weekday, 2024",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    written.append(style.finish(
        fig, _fig("A1_demand_profile_heatmap"), NYC,
        "Each panel is normalised within its own borough, so shapes are "
        "comparable across boroughs of very different size. Trips with an "
        "unattributable pickup zone are excluded from the figure and retained "
        "in A1_demand_profile.csv.",
    ))

    # ---- line chart: hourly profile, weekday vs weekend
    fig, ax = plt.subplots(figsize=(8.5, 5))
    frame["is_weekend"] = frame["day_of_week"] >= 6
    for index, borough in enumerate(boroughs[:6]):
        sub = frame[frame["borough"] == borough]
        for weekend, dashes in ((False, None), (True, (2, 1.5))):
            series = (
                sub[sub["is_weekend"] == weekend]
                .groupby("pickup_hour")["trip_count"].sum()
                .reindex(range(24), fill_value=0)
            )
            total = series.sum()
            if not total:
                continue
            line, = ax.plot(
                series.index, 100.0 * series.values / total,
                color=style.PALETTE[index % len(style.PALETTE)],
                linewidth=1.8 if not weekend else 1.4,
                label=f"{borough} ({'weekend' if weekend else 'weekday'})",
            )
            if dashes:
                line.set_dashes(dashes)

    ax.set_xlabel("Hour of pickup (local time)")
    ax.set_ylabel("Share of that borough's trips (%)")
    ax.set_title("Diurnal demand profile by borough, weekday versus weekend, 2024")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    written.append(style.finish(
        fig, _fig("A1_demand_profile_lines"), NYC,
        "Solid = Monday-Friday, dashed = Saturday-Sunday. Each series sums to "
        "100% within its own borough and day type.",
    ))
    return written


# --------------------------------------------------------------------------- #
# A2
# --------------------------------------------------------------------------- #
def fig_a2() -> list[Path]:
    frame = _table("A2_od_corridors")
    if frame is None or frame.empty:
        return []
    frame = frame.sort_values("trip_count", ascending=True)
    labels = [
        f"{row.pickup_zone} → {row.dropoff_zone}"
        + ("  (intra-zone)" if row.is_intra_zone else "")
        for row in frame.itertuples()
    ]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(8.5, 9.5), gridspec_kw={"width_ratios": [2.1, 1]},
        sharey=True,
    )
    colours = [
        style.PALETTE[1] if intra else style.PALETTE[0]
        for intra in frame["is_intra_zone"]
    ]
    ax.barh(labels, frame["trip_count"], color=colours, height=0.72)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Trips in 2024 (count)")
    ax.set_title("Volume by corridor", fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(style.thousands))
    for y, value in enumerate(frame["trip_count"]):
        ax.text(value, y, f" {int(value):,}", va="center", fontsize=9,
                color=style.MUTED_COLOUR)
    ax.set_xlim(0, frame["trip_count"].max() * 1.15)

    ax2.barh(labels, frame["mean_fare"], color=style.PALETTE[2], height=0.45,
             label="Mean fare")
    ax2.barh(labels, frame["median_fare"], color=style.PALETTE[7], height=0.2,
             label="Median fare")
    ax2.set_xlabel("Fare (USD)")
    ax2.set_title("Mean vs median fare", fontsize=11)
    ax2.legend(fontsize=9)

    fig.suptitle("Directed corridor volume and fare profile, NYC yellow taxi 2024 "
                 "-- top 25 by trip count", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    fig.subplots_adjust(wspace=0.12)
    return [style.finish(
        fig, _fig("A2_od_corridors"), NYC,
        "Corridors are DIRECTED: A→B and B→A are separate rows. "
        "Orange bars are intra-zone trips. Non-geographic zones are excluded. "
        "Mean above median indicates right-skewed fares, chiefly airport flat "
        "fares.",
    )]


# --------------------------------------------------------------------------- #
# A3
# --------------------------------------------------------------------------- #
def fig_a3() -> list[Path]:
    frame = _table("A3_congestion_proxy")
    if frame is None or frame.empty:
        return []
    hourly = _real_boroughs(frame[frame["grouping_level"] == "borough_hour"])
    monthly = _real_boroughs(frame[frame["grouping_level"] == "borough_month"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.6))
    boroughs = (
        hourly.groupby("borough")["trip_count"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    for index, borough in enumerate(boroughs):
        sub = hourly[hourly["borough"] == borough].sort_values("pickup_hour")
        opts = style.series_style(index)
        ax1.plot(sub["pickup_hour"], sub["mean_speed_mph"],
                 color=opts["color"], linestyle=opts["linestyle"],
                 marker=opts["marker"], markersize=3.2, linewidth=1.6,
                 label=borough)
    ax1.set_xlabel("Hour of pickup (local time)")
    ax1.set_ylabel("Mean journey speed (mph)")
    ax1.set_title("Diurnal speed profile by borough", fontsize=11)
    ax1.set_xticks(range(0, 24, 3))
    ax1.legend(fontsize=9, ncol=2)

    for index, borough in enumerate(boroughs):
        sub = monthly[monthly["borough"] == borough].sort_values("year_month")
        opts = style.series_style(index)
        ax2.plot(sub["year_month"], sub["mean_speed_mph"],
                 color=opts["color"], linestyle=opts["linestyle"],
                 marker=opts["marker"], markersize=3.2, linewidth=1.6,
                 label=borough)
    ax2.set_xlabel("Month of 2024")
    ax2.set_ylabel("Mean journey speed (mph)")
    ax2.set_title("Month-over-month trend", fontsize=11)
    ax2.tick_params(axis="x", rotation=60, labelsize=9)

    fig.suptitle("Journey speed as a congestion proxy, NYC yellow taxi 2024",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return [style.finish(
        fig, _fig("A3_congestion_proxy"), NYC,
        "PROXY ONLY. Journey speed confounds route choice, trip-length mix and "
        "time stopped for non-traffic reasons; a fall is consistent with "
        "congestion but is not a measurement of it. No causal claim is made. "
        "Trips with speeds above 100 mph (~0.03%) are excluded.",
    )]


# --------------------------------------------------------------------------- #
# A4
# --------------------------------------------------------------------------- #
def fig_a4() -> list[Path]:
    frame = _table("A4_tipping_behaviour")
    if frame is None or frame.empty:
        return []
    interpretable = frame[frame["is_interpretable"]]
    interpretable = _real_boroughs(interpretable)
    if interpretable.empty:
        LOG.warning("A4 figure skipped: no interpretable rows")
        return []

    order = (
        interpretable[["distance_band", "distance_band_order"]]
        .drop_duplicates().sort_values("distance_band_order")["distance_band"]
        .tolist()
    )
    boroughs = (
        interpretable.groupby("borough")["trip_count"].sum()
        .sort_values(ascending=False).index.tolist()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.6))
    width = 0.8 / max(len(boroughs), 1)
    positions = np.arange(len(order))
    for index, borough in enumerate(boroughs):
        sub = interpretable[interpretable["borough"] == borough]
        values = [
            sub.loc[sub["distance_band"] == band, "aggregate_tip_pct_of_fare"]
            .pipe(lambda s: s.iloc[0] if len(s) else np.nan)
            for band in order
        ]
        ax1.bar(positions + index * width, values, width * 0.92,
                color=style.PALETTE[index % len(style.PALETTE)], label=borough)
    ax1.set_xticks(positions + width * (len(boroughs) - 1) / 2)
    ax1.set_xticklabels(order, fontsize=9)
    ax1.set_xlabel("Trip distance band")
    ax1.set_ylabel("Tip as % of fare (aggregate)")
    ax1.set_title("Credit-card tipping by distance and borough", fontsize=11)
    ax1.legend(fontsize=9, ncol=2)

    by_payment = (
        frame.groupby(["payment_type_name", "is_interpretable"], as_index=False)
        .agg(trips=("trip_count", "sum"),
             tipped=("trips_with_recorded_tip", "sum"))
    )
    by_payment["pct_tipped"] = 100.0 * by_payment["tipped"] / by_payment["trips"]
    by_payment = by_payment.sort_values("trips", ascending=True)
    colours = [
        style.PALETTE[0] if ok else style.PALETTE[4]
        for ok in by_payment["is_interpretable"]
    ]
    ax2.barh(by_payment["payment_type_name"], by_payment["pct_tipped"],
             color=colours, height=0.65)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_xlabel("Trips with a RECORDED tip (%)")
    ax2.set_title("Recorded tipping by payment type", fontsize=11)
    for y, (value, trips) in enumerate(zip(by_payment["pct_tipped"],
                                           by_payment["trips"])):
        ax2.text(value, y, f"  {value:.1f}%  (n={int(trips):,})", va="center",
                 fontsize=9, color=style.MUTED_COLOUR)
    ax2.set_xlim(0, max(by_payment["pct_tipped"].max() * 1.6, 5))

    fig.suptitle("Tipping behaviour, NYC yellow taxi 2024",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return [style.finish(
        fig, _fig("A4_tipping_behaviour"), NYC,
        "The TLC meter records CARD tips only; cash tips are not captured. Blue "
        "= interpretable (credit card); orange = structural zeros that are "
        "evidence about the recording system, not about passengers. The left "
        "panel therefore shows credit-card trips only.",
    )]


# --------------------------------------------------------------------------- #
# A5
# --------------------------------------------------------------------------- #
def fig_a5() -> list[Path]:
    frame = _table("A5_revenue_concentration")
    if frame is None or frame.empty:
        return []
    frame = frame.sort_values("revenue_rank")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(frame["revenue_rank"], frame["pct_of_revenue"],
           color=style.PALETTE[0], width=1.0, label="Share of revenue (%)")
    ax.set_xlabel("Pickup zone, ranked by revenue")
    ax.set_ylabel("Share of 2024 revenue (%)")
    ax.set_xlim(0, len(frame) + 1)

    twin = ax.twinx()
    twin.plot(frame["revenue_rank"], frame["cumulative_pct_revenue"],
              color=style.PALETTE[1], linewidth=2.0,
              label="Cumulative revenue (%)")
    twin.plot(frame["revenue_rank"], frame["cumulative_pct_trips"],
              color=style.PALETTE[2], linewidth=1.6, linestyle="--",
              label="Cumulative trips (%)")
    twin.set_ylabel("Cumulative share (%)")
    twin.set_ylim(0, 101)
    twin.grid(False)

    within = frame[frame["within_top_80pct_revenue"]]
    if len(within):
        cutoff = int(within["revenue_rank"].max())
        pct_zones = 100.0 * cutoff / len(frame)
        twin.axhline(80, color=style.MUTED_COLOUR, linewidth=0.9, linestyle=":")
        twin.axvline(cutoff, color=style.MUTED_COLOUR, linewidth=0.9,
                     linestyle=":")
        twin.annotate(
            f"{cutoff} zones ({pct_zones:.0f}% of zones)\ncarry 80% of revenue",
            xy=(cutoff, 80), xytext=(cutoff + len(frame) * 0.06, 55),
            fontsize=9.5, color=style.TEXT_COLOUR,
            arrowprops={"arrowstyle": "->", "color": style.MUTED_COLOUR,
                        "linewidth": 0.9},
        )

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="center right", fontsize=9.5)
    ax.set_title("Revenue concentration across pickup zones, NYC yellow taxi 2024",
                 fontsize=11.5)
    fig.tight_layout()
    return [style.finish(
        fig, _fig("A5_revenue_concentration"), NYC,
        "Revenue is total_amount: what passengers paid, including tolls, "
        "surcharges, taxes and recorded tips. It is not operator net revenue. "
        "Non-geographic zones are retained so the concentration of real zones "
        "is not overstated.",
    )]


# --------------------------------------------------------------------------- #
# A6
# --------------------------------------------------------------------------- #
def _monthly(sub: pd.DataFrame, start: pd.Timestamp,
             end: pd.Timestamp) -> pd.DataFrame:
    """Reindex one series onto every calendar month, so an absent month or a
    chain break draws as a GAP rather than a line bridging it."""
    indexed = sub.assign(month=pd.to_datetime(sub["year_month"] + "-01")) \
        .set_index("month")
    return indexed.reindex(pd.date_range(start, end, freq="MS"))


def fig_a6() -> list[Path]:
    frame = _table("A6_food_price_index")
    if frame is None or frame.empty:
        return []
    frame = frame[
        (frame["year_month"] >= settings.PRICE_INDEX_BASE_YEAR_MONTH)
        & (frame["category"].str.lower() != "unknown")
    ].copy()
    start = pd.Timestamp(settings.PRICE_INDEX_BASE_YEAR_MONTH + "-01")
    end = pd.to_datetime(frame["year_month"] + "-01").max()

    linked = frame.dropna(subset=["chained_index"])
    linked_months = linked.groupby("category")["chained_index"].count()
    # A category linked to the base for only a handful of months has no
    # series to draw; it is named in the caption instead of plotted as a dot.
    categories = (
        linked[linked["category"].isin(linked_months[linked_months >= 12].index)]
        .groupby("category")["matched_cells"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    unlinkable = sorted(set(frame["category"]) - set(categories))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8.5, 9.8), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.35, 1]},
    )

    for index, category in enumerate(categories):
        series = _monthly(frame[frame["category"] == category], start, end)
        opts = style.series_style(index)
        last = series["chained_index"].last_valid_index()
        ax1.plot(series.index, series["chained_index"], color=opts["color"],
                 linestyle=opts["linestyle"], linewidth=1.8,
                 label=f"{category} (linked to {last:%Y-%m})")
    ax1.axhline(100, color=style.MUTED_COLOUR, linewidth=0.8, linestyle="--")
    ax1.set_yscale("log")
    ax1.set_ylabel(f"Chained index\n({settings.PRICE_INDEX_BASE_YEAR_MONTH} = 100, log)")
    ax1.set_title("Nigerian food prices by category: chained matched-model index",
                  fontsize=11.5)
    ax1.legend(fontsize=9, ncol=2, loc="upper left")

    headline = "cereals and tubers"
    series = _monthly(frame[frame["category"] == headline], start, end)
    ax2.plot(series.index, series["chained_index"], color=style.PALETTE[0],
             linewidth=2.0, label="chained index (composition held constant)")
    ax2.plot(series.index, series["naive_mean_index"], color=style.PALETTE[1],
             linewidth=1.5, linestyle="--",
             label="naive mean of whatever reported (same rows)")
    ax2.axhline(100, color=style.MUTED_COLOUR, linewidth=0.8, linestyle=":")
    ax2.set_yscale("log")
    ax2.set_ylabel("Index (log)")
    ax2.set_title(f"Composition bias, {headline}: the two series use identical "
                  f"observations", fontsize=10.5)
    ax2.legend(fontsize=9, loc="upper left")

    for index, category in enumerate(categories):
        series = _monthly(frame[frame["category"] == category], start, end)
        ax3.plot(series.index, series["matched_cells"],
                 color=style.PALETTE[index % len(style.PALETTE)], linewidth=1.2)
    ax3.axhline(settings.PRICE_INDEX_MIN_MATCHED_CELLS, color=style.PALETTE[7],
                linewidth=1.0, linestyle="--",
                label=f"break threshold ({settings.PRICE_INDEX_MIN_MATCHED_CELLS} "
                      f"cells)")
    ax3.set_ylabel("Matched cells\nper link")
    ax3.set_xlabel("Month")
    ax3.set_title("Panel depth behind each link", fontsize=10.5)
    ax3.legend(fontsize=9, loc="upper left")
    ax3.tick_params(axis="x", labelsize=9)

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return [style.finish(
        fig, _fig("A6_food_price_index"), WFP,
        "Chained matched-model (Jevons) index: each monthly link compares only "
        "cells observed in both periods (a cell may link to its own observation "
        f"up to {settings.PRICE_INDEX_MAX_LINK_GAP_MONTHS} months old). Gaps are "
        f"chain breaks -- links on fewer than "
        f"{settings.PRICE_INDEX_MIN_MATCHED_CELLS} matched cells -- and nothing "
        "is carried across one, so each category is drawn only while it links "
        "unbroken to January 2016. Not drawn, because they cannot be linked to "
        f"the base for 12 months or more: {', '.join(unlinkable) or 'none'}. "
        "The panel contracts to three north-eastern states from January 2023. "
        "Unweighted: WFP publishes no quantities.",
    )]


# --------------------------------------------------------------------------- #
# A7
# --------------------------------------------------------------------------- #
def fig_a7() -> list[Path]:
    frame = _table("A7_subsidy_structural_break")
    if frame is None or frame.empty:
        return []
    frame = frame.sort_values("months_since_break")
    row = frame.iloc[0]
    n_markets, n_states = int(row["panel_markets"]), int(row["panel_states"])
    window = int(row["window_months_each_side"])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for segment, colour in (("pre", style.PALETTE[0]), ("post", style.PALETTE[1])):
        sub = frame[frame["segment"] == segment]
        if sub.empty:
            continue
        ax.plot(sub["months_since_break"], sub["staple_index"], color=colour,
                marker="o", markersize=5, linewidth=1.8,
                label=f"{segment}-break index (fixed panel)")
        ax.plot(sub["months_since_break"], sub["fitted_index"], color=colour,
                linestyle="--", linewidth=2.0,
                label=f"{segment}-break OLS trend (n={len(sub)})")

    ax.axvline(0.5, color=style.PALETTE[7], linewidth=1.4)
    ax.text(0.6, 0.03, f"subsidy removal announced {row['break_event_date']}",
            transform=ax.get_xaxis_transform(), fontsize=9.5,
            color=style.TEXT_COLOUR, va="bottom")
    ax.axhline(100, color=style.MUTED_COLOUR, linewidth=0.8, linestyle=":")

    summary = [
        f"WITHIN-CELL LEVEL CHANGE (robust): "
        f"{row['within_cell_level_change_pct']:+.1f}%",
        f"  same market-commodity cells, after vs before; n = "
        f"{int(row['cells_on_both_sides'])} cells",
        f"  interquartile range {row['within_cell_change_p25_pct']:+.1f}% to "
        f"{row['within_cell_change_p75_pct']:+.1f}%",
        "",
        "Segment trends (fragile: only "
        f"{window} points each):",
    ]
    for segment in ("pre", "post"):
        sub = frame[frame["segment"] == segment]
        if len(sub):
            summary.append(
                f"  {segment}: {sub['segment_slope_per_month'].iloc[0]:+.2f} pts/month "
                f"(R²={sub['segment_r2'].iloc[0]:.2f}, n={int(sub['segment_n_months'].iloc[0])})"
            )
    summary.append(f"  slope change {row['slope_change_per_month']:+.2f} pts/month; "
                   f"level shift {row['level_shift_at_break']:+.2f} pts")
    ax.text(0.015, 0.97, "\n".join(summary), transform=ax.transAxes, va="top",
            fontsize=9, color=style.TEXT_COLOUR, family="DejaVu Sans",
            bbox={"facecolor": "white", "edgecolor": style.GRID_COLOUR,
                  "boxstyle": "round,pad=0.5"})

    ax.set_xticks(frame["months_since_break"])
    ax.set_xticklabels([f"{m}\n(t={t:+d})" for m, t in
                        zip(frame["year_month"], frame["months_since_break"])],
                       fontsize=9)
    ax.set_xlabel("Month (t = months since the announcement month)")
    ax.set_ylabel(f"Staple food index ({row['break_month']} = 100)")
    ax.set_title(
        f"WITHIN-MARKET: staple prices in a fixed panel of {n_markets} markets "
        f"in {n_states} states ({row['panel_state_list'].replace('; ', ', ')})",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="center left")
    fig.tight_layout()
    return [style.finish(
        fig, _fig("A7_subsidy_structural_break"), WFP,
        f"WITHIN-MARKET ANALYSIS, NOT NATIONAL. Fixed panel: the {n_markets} "
        f"markets in {n_states} states that report retail staple prices in "
        f"every month from {frame['year_month'].min()} to "
        f"{frame['year_month'].max()} -- {row['panel_market_list']}. "
        f"A wider window would contain no continuously reporting market. "
        "ASSOCIATION ONLY -- NO CAUSAL CLAIM: the window also spans the June "
        "2023 naira devaluation, the northern lean season and insecurity in "
        "these markets. Fuel prices could not be used: NBS petrol data begins "
        "2023-11, after this window closes.",
    )]


# --------------------------------------------------------------------------- #
# A8
# --------------------------------------------------------------------------- #
def fig_a8() -> list[Path]:
    frame = _table("A8_price_dispersion")
    if frame is None or frame.empty:
        return []
    frame = frame[frame["meets_market_threshold"]].copy()
    if frame.empty:
        LOG.warning("A8 figure skipped: no cells meet the market threshold")
        return []

    zones = sorted(frame["geopolitical_zone"].dropna().unique())
    top_commodities = (
        frame.groupby("commodity_label")["reporting_markets"].sum()
        .sort_values(ascending=False).head(5).index.tolist()
    )

    # 2 columns: with 4-6 zones this is a clean 2xN grid (no empty cells to
    # crop away) at a width where the rotated month ticks stay >=9pt.
    ncols = min(2, len(zones))
    nrows = int(np.ceil(len(zones) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows),
                             squeeze=False, sharey=True)

    for index, zone in enumerate(zones):
        ax = axes[index // ncols][index % ncols]
        sub = frame[frame["geopolitical_zone"] == zone]
        for c_index, commodity in enumerate(top_commodities):
            series = (sub[sub["commodity_label"] == commodity]
                      .sort_values("year_month"))
            if series.empty:
                continue
            # Plotted as real dates, not raw strings: a categorical string
            # axis registers each commodity's month set in first-seen order,
            # which silently misplaces ticks once commodities cover
            # different date ranges within the same zone.
            dates = pd.to_datetime(series["year_month"] + "-01")
            ax.plot(dates, series["cv_pct"],
                    color=style.PALETTE[c_index % len(style.PALETTE)],
                    linewidth=1.3, label=commodity)
        ax.set_title(zone, fontsize=11)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        if index % ncols == 0:
            ax.set_ylabel("Cross-market CV (%)")

    for spare in range(len(zones), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Cross-market price dispersion by geopolitical zone "
        "(coefficient of variation)", fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return [style.finish(
        fig, _fig("A8_price_dispersion"), WFP,
        f"Coefficient of variation = sd / mean across reporting markets within "
        f"a zone, for one commodity and unit at a time. Scale-free, so a rise "
        f"means prices genuinely diverged rather than merely rose. Cells with "
        f"fewer than {settings.DISPERSION_MIN_MARKETS} reporting markets are "
        f"excluded. A high CV is not by itself evidence of market failure.",
    )]


# --------------------------------------------------------------------------- #
# A9
# --------------------------------------------------------------------------- #
def fig_a9() -> list[Path]:
    frame = _table("A9_fuel_food_passthrough")
    if frame is None or frame.empty:
        return []

    zones = [z for z in frame["zone"].unique() if z != "ALL ZONES"]
    order = ["ALL ZONES"] + sorted(zones)
    lags = sorted(frame["lag_months"].unique())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    width = 0.8 / max(len(order), 1)
    positions = np.arange(len(lags))

    for index, zone in enumerate(order):
        sub = frame[frame["zone"] == zone].set_index("lag_months")
        values = [sub["corr_log_changes"].get(lag, np.nan) for lag in lags]
        colour = style.PALETTE[7] if zone == "ALL ZONES" else \
            style.PALETTE[index % len(style.PALETTE)]
        ax1.bar(positions + index * width, values, width * 0.9, color=colour,
                label=zone, edgecolor="white", linewidth=0.4)

    ax1.axhline(0, color=style.MUTED_COLOUR, linewidth=1.0)
    # The critical |r| for nominal 5% significance at the smallest n reported,
    # BEFORE any allowance for serial correlation or clustering in 3 states.
    n_min = int(frame["n_pairs"].min())
    t_crit = 2.0
    r_crit = t_crit / np.sqrt(n_min - 2 + t_crit ** 2)
    for bound in (r_crit, -r_crit):
        ax1.axhline(bound, color=style.PALETTE[1], linewidth=1.0, linestyle=":")
    ax1.annotate(f" |r| = {r_crit:.2f}: nominal 5% bar (n={n_min}), unadjusted ",
                 xy=(len(lags) - 0.55, r_crit), xytext=(0, -7),
                 textcoords="offset points", fontsize=9, va="top", ha="right",
                 color=style.PALETTE[1])
    ax1.set_ylim(-max(0.3, r_crit * 1.45), max(0.3, r_crit * 1.45))
    ax1.set_ylabel("Correlation of monthly log changes")
    ax1.set_title(
        "EXPLORATORY -- NULL RESULT: petrol vs staple food price changes, by lag",
        color="#8B0000", fontsize=11,
    )
    ax1.legend(fontsize=9, ncol=3, loc="lower right")

    for index, zone in enumerate(order):
        sub = frame[frame["zone"] == zone].set_index("lag_months")
        pairs = [sub["n_pairs"].get(lag, 0) for lag in lags]
        colour = style.PALETTE[7] if zone == "ALL ZONES" else \
            style.PALETTE[index % len(style.PALETTE)]
        ax2.bar(positions + index * width, pairs, width * 0.9, color=colour,
                edgecolor="white", linewidth=0.4)
    ax2.axhline(settings.PASSTHROUGH_MIN_PAIRS, color=style.PALETTE[1],
                linewidth=1.1, linestyle="--",
                label=f"minimum {settings.PASSTHROUGH_MIN_PAIRS} pairs to report")
    ax2.set_ylabel("Sample size\n(state-month pairs)")
    ax2.set_xlabel("Lag applied to the petrol series (months)")
    ax2.set_xticks(positions + width * (len(order) - 1) / 2)
    ax2.set_xticklabels([f"lag {lag}" for lag in lags], fontsize=9)
    ax2.legend(fontsize=9)

    row = frame.iloc[0]
    coverage = (
        f"COVERAGE: {int(row['states_in_overlap'])} of "
        f"{int(row['federating_units_total'])} federating units "
        f"({row['pct_of_federating_units']}%), all North East.\n"
        f"Petrol covers {int(row['states_with_fuel_data'])} states; WFP food "
        f"covers {int(row['states_with_food_data'])} historically but only "
        f"{int(row['states_in_overlap'])} in this window."
    )
    ax1.text(0.015, 0.97, coverage, transform=ax1.transAxes, va="top",
             fontsize=9, color=style.TEXT_COLOUR,
             bbox={"facecolor": "#FFF4E6", "edgecolor": style.PALETTE[1],
                   "boxstyle": "round,pad=0.45"})

    fig.suptitle("EXPLORATORY ANALYSIS -- NOT A FINDING -- SUPPORTS NO INFERENCE",
                 fontsize=11.5, fontweight="bold", color="#8B0000", y=1.01)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return [style.finish(
        fig, _fig("A9_fuel_food_passthrough"), f"{NBS}; {WFP}",
        "EXPLORATORY -- NULL RESULT. WFP had stopped reporting 11 of the 14 "
        "states it monitored by January 2023, so this analysis rests on 3 "
        "states, all located in the North East (Borno, Yobe, Adamawa); 'ALL "
        "ZONES' and 'North East' are the same three states. The correlations "
        "describe conflict-affected north-eastern markets in those states only. "
        "Every one is under +/-0.09, far inside even the unadjusted nominal "
        "significance bar shown, and cannot support any inference. Retained for "
        "transparency; not presented in the results.",
    )]


# --------------------------------------------------------------------------- #
# A10
# --------------------------------------------------------------------------- #
def fig_a10() -> list[Path]:
    frame = _table("A10_grain_comparison")
    if frame is None or frame.empty:
        return []
    frame = frame.sort_values("row_count", ascending=True)
    labels = frame["fact_table"].tolist()

    # Stacked 1x3 -> 3x1: at ~8.5in wide, three side-by-side log-scale bar
    # panels leave no room for 9pt value labels; stacking keeps each panel
    # full width instead.
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.2))
    axes[0].barh(labels, frame["row_count"], color=style.PALETTE[0], height=0.6)
    axes[0].set_xscale("log")
    axes[0].tick_params(axis="y", labelsize=9.5)
    axes[0].set_xlabel("Rows (log scale)")
    axes[0].set_title("Grain: rows per fact table", fontsize=11)
    for y, value in enumerate(frame["row_count"]):
        axes[0].text(value, y, f" {int(value):,}", va="center", fontsize=9.5,
                     color=style.MUTED_COLOUR)

    axes[1].barh(labels, frame["bytes_measured"], color=style.PALETTE[2],
                 height=0.6)
    axes[1].set_xscale("log")
    axes[1].tick_params(axis="y", labelsize=9.5)
    axes[1].set_xlabel("On-disk bytes (log scale)")
    axes[1].set_title("Storage footprint", fontsize=11)
    for y, value in enumerate(frame["bytes_human_measured"]):
        axes[1].text(frame["bytes_measured"].iloc[y], y, f" {value}",
                     va="center", fontsize=9.5, color=style.MUTED_COLOUR)

    axes[2].barh(labels, frame["national_aggregate_seconds_measured"],
                 color=style.PALETTE[1], height=0.6)
    axes[2].tick_params(axis="y", labelsize=9.5)
    axes[2].set_xlabel("Seconds (median of repeats)")
    axes[2].set_title("Time to a national monthly aggregate", fontsize=11)
    for y, value in enumerate(frame["national_aggregate_seconds_measured"]):
        if pd.notna(value):
            axes[2].text(value, y, f" {value:.3f}s", va="center", fontsize=9.5,
                         color=style.MUTED_COLOUR)

    fig.suptitle(
        "The engineering cost of serving multiple grains from one conformed model",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    return [style.finish(
        fig, _fig("A10_grain_comparison"),
        "Source: this platform's own gold layer",
        "Row counts and dimension members are measured in SQL; bytes and "
        "timings are filesystem and wall-clock measurements taken by the "
        "analysis runner. Timings are the median of "
        f"{settings.BENCHMARK_REPEATS} repeats inside the declared "
        f"{settings.CPU_LIMIT}-core / {settings.MEM_LIMIT_GB} GB container.",
    )]


# --------------------------------------------------------------------------- #
# A11
# --------------------------------------------------------------------------- #
def fig_a11() -> list[Path]:
    frame = _table("A11_missingness_structure")
    if frame is None or frame.empty:
        return []
    patterns = frame[frame["record_type"] == "pattern"].copy()
    vendor_month = frame[frame["record_type"] == "vendor_month"].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.6))

    patterns = patterns.sort_values("trip_count", ascending=True)
    ax1.barh(patterns["missingness_pattern"], patterns["pct_of_trips"],
             color=[style.PALETTE[0] if p == "----" else style.PALETTE[1]
                    for p in patterns["missingness_pattern"]], height=0.6)
    ax1.tick_params(axis="y", labelsize=9.5)
    ax1.set_xlabel("Share of 2024 trips (%)")
    ax1.set_ylabel("Missingness signature\n(P/C/R/A = field absent)")
    ax1.set_title(f"Observed patterns: {len(patterns)} of 16 possible", fontsize=11)
    for y, (value, count) in enumerate(zip(patterns["pct_of_trips"],
                                           patterns["trip_count"])):
        ax1.text(value, y, f"  {value:.2f}%  (n={int(count):,})", va="center",
                 fontsize=9.5, color=style.MUTED_COLOUR)
    ax1.set_xlim(0, max(patterns["pct_of_trips"].max() * 1.5, 5))

    for index, vendor in enumerate(sorted(vendor_month["vendor_name"].unique())):
        sub = vendor_month[vendor_month["vendor_name"] == vendor] \
            .sort_values("year_month")
        opts = style.series_style(index)
        ax2.plot(sub["year_month"], sub["pct_missing_passenger"],
                 color=opts["color"], linestyle="-", marker=opts["marker"],
                 markersize=3.4, linewidth=1.7, label=vendor)
        ax2.plot(sub["year_month"], sub["pct_all_three_missing"],
                 color=opts["color"], linestyle=":", linewidth=2.4, alpha=0.75)
    ax2.set_xlabel("Month of 2024")
    ax2.set_ylabel("Trips with the field absent (%)")
    ax2.set_title("Block failure rate by vendor over time", fontsize=11)
    ax2.tick_params(axis="x", rotation=60, labelsize=9)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Fields do not go missing independently:\na structural signature in the 2024 TLC corpus",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.91))
    return [style.finish(
        fig, _fig("A11_missingness_structure"), NYC,
        "SUPPLEMENTARY ANALYSIS (not in the original specification). Signature "
        "letters: P = passenger_count, C = congestion_surcharge, "
        "R = RatecodeID, A = airport_fee. Only 2 of the 16 possible "
        "combinations occur: all present, or all four absent together. In the "
        "right panel a dotted line for 'all absent together' is drawn for each "
        "vendor and is hidden exactly beneath the solid line -- that "
        "coincidence is the finding. The fields fail as a block, so no "
        "per-column imputation is defensible.",
    )]


# --------------------------------------------------------------------------- #
# A12 -- centrepiece of the Nigerian analysis
# --------------------------------------------------------------------------- #
A12_STEM = "A12_petrol_price_geography"
ZONE_ORDER = ("North Central", "North East", "North West",
              "South East", "South South", "South West")
#: One fixed colour and dash per zone, shared by every A12 figure.
ZONE_STYLE = {
    zone: {"color": style.PALETTE[i], "linestyle": style.LINE_STYLES[i],
           "marker": style.MARKERS[i]}
    for i, zone in enumerate(ZONE_ORDER)
}


def _months(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["year_month"] + "-01")


def _a12_national(national: pd.DataFrame) -> Path:
    national = national.sort_values("year_month")
    x = _months(national)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8.5, 8.8), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1, 1]},
    )

    ax1.fill_between(x, national["min_state_price_ngn"],
                     national["max_state_price_ngn"], color=style.PALETTE[5],
                     alpha=0.18, linewidth=0, label="range across 37 states")
    ax1.fill_between(x, national["p25_ngn"], national["p75_ngn"],
                     color=style.PALETTE[5], alpha=0.42, linewidth=0,
                     label="interquartile range across states")
    ax1.plot(x, national["national_mean_ngn"], color=style.PALETTE[0],
             linewidth=2.4, marker="o", markersize=3.5, label="national mean")
    ax1.plot(x, national["national_median_ngn"], color=style.PALETTE[7],
             linewidth=1.1, linestyle="--", label="national median")

    first, last = national.iloc[0], national.iloc[-1]
    change = 100.0 * (last["national_mean_ngn"] / first["national_mean_ngn"] - 1.0)
    for point, offset in ((first, (8, 18)), (last, (-8, 14))):
        ax1.annotate(
            f"{point['year_month']}\n₦{point['national_mean_ngn']:,.2f}",
            xy=(pd.Timestamp(point["year_month"] + "-01"),
                point["national_mean_ngn"]),
            xytext=offset, textcoords="offset points", fontsize=9.5,
            ha="left" if offset[0] > 0 else "right",
            arrowprops={"arrowstyle": "-", "color": style.MUTED_COLOUR,
                        "linewidth": 0.7},
        )
    ax1.set_ylabel("NGN per litre")
    ax1.set_title(
        f"National mean pump price of petrol (PMS): ₦{first['national_mean_ngn']:,.0f}"
        f" to ₦{last['national_mean_ngn']:,.0f} ({change:+.0f}%) across "
        f"{len(national)} months", fontsize=10.5,
    )
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax1.legend(fontsize=9, loc="upper left")

    ax2.bar(x, national["cv_pct"], width=22, color=style.PALETTE[1], alpha=0.85)
    ax2.set_ylabel("Cross-state CV (%)")
    ax2.set_title("Dispersion across states (coefficient of variation, "
                  "population sd / mean)", fontsize=10)

    ax3.plot(x, national["zone_share_of_variance_pct"], color=style.PALETTE[2],
             linewidth=1.8, marker="s", markersize=3.2)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Explained by\nzones (%)")
    ax3.set_xlabel("Month")
    ax3.set_title("Share of cross-state variance lying BETWEEN the six "
                  "geopolitical zones", fontsize=10)
    ax3.tick_params(axis="x", labelsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return style.finish(
        fig, _fig("A12_petrol_price_national"), NBS,
        "National mean = unweighted mean of the 37 state prices, the Bureau's own "
        "definition; it reconciles to 0.00 NGN against the published national "
        "averages. Dispersion uses the population sd, because the 37 federating "
        "units are a census, not a sample. The series begins in November 2023, "
        "six months after the 29 May 2023 subsidy removal, so it describes the "
        "post-removal market. Descriptive only.",
    )


def _a12_premiums(states: pd.DataFrame) -> Path:
    states = states.sort_values("mean_premium_vs_national_pct")
    y = np.arange(len(states))
    fig, ax = plt.subplots(figsize=(8.5, 12.5))

    for pos, row in zip(y, states.itertuples()):
        zs = ZONE_STYLE.get(row.zone, {"color": style.MUTED_COLOUR, "marker": "o"})
        ax.hlines(pos, row.min_premium_pct, row.max_premium_pct,
                  color=zs["color"], alpha=0.30, linewidth=1.2)
        ax.hlines(pos, row.mean_premium_vs_national_pct - row.sd_premium_pp,
                  row.mean_premium_vs_national_pct + row.sd_premium_pp,
                  color=zs["color"], alpha=0.85, linewidth=3.2)
        persistent = row.persistence_class != "no persistent position"
        ax.plot(row.mean_premium_vs_national_pct, pos, marker=zs["marker"],
                markersize=8, markeredgecolor=zs["color"], markeredgewidth=1.5,
                markerfacecolor=zs["color"] if persistent else "white",
                linestyle="none", zorder=3)
        ax.text(1.005, pos,
                f"{row.share_months_above * 100:3.0f}% above  "
                f"(run {row.longest_run_above_months}/{row.longest_run_below_months})",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                color=style.TEXT_COLOUR)

    ax.axvline(0, color=style.TEXT_COLOUR, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(states["state"], fontsize=9.5)
    ax.set_ylim(-0.8, len(states) - 0.2)
    ax.set_xlabel("Premium over the national mean (%)")
    ax.set_title("Persistent state premiums and discounts, petrol, "
                 f"{int(states['months'].iloc[0])} months", fontsize=11)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ZONE_STYLE[z]["color"], marker=ZONE_STYLE[z]["marker"],
                      linestyle="none", markersize=7, label=z) for z in ZONE_ORDER]
    handles += [
        Line2D([0], [0], color=style.MUTED_COLOUR, marker="o", linestyle="none",
               markerfacecolor=style.MUTED_COLOUR, markersize=7,
               label=f"filled: persistent (≥{settings.A12_PERSISTENCE_SHARE:.0%} "
                     "of months one side)"),
        Line2D([0], [0], color=style.MUTED_COLOUR, marker="o", linestyle="none",
               markerfacecolor="white", markersize=7, label="hollow: no persistent position"),
        Line2D([0], [0], color=style.MUTED_COLOUR, linewidth=3.2,
               label="thick bar: mean ± 1 sd over time"),
        Line2D([0], [0], color=style.MUTED_COLOUR, linewidth=1.2, alpha=0.4,
               label="thin line: min to max monthly premium"),
    ]
    # Legend below the plot area so it never covers a state's line. A wider
    # right-margin reservation than the 12in original: the same annotation
    # text now sits at a larger font on a narrower figure.
    fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=2,
               bbox_to_anchor=(0.45, 0.0))
    fig.tight_layout(rect=(0, 0.11, 0.72, 1))
    return style.finish(
        fig, _fig("A12_petrol_price_state_premiums"), NBS,
        "Premium = state price / national mean - 1. Dot = mean monthly premium; "
        "states sorted by it. Right-hand text: share of months above the "
        "national mean, and the longest unbroken run of months above / below "
        "it. Premiums sum to zero across states in every month. Descriptive "
        "only: distance from import terminals, transport cost and security are "
        "plausible reasons for a premium, and none is tested here.",
    )


def _a12_ranks(main: pd.DataFrame, national: pd.DataFrame,
               states: pd.DataFrame) -> Path:
    order = states.sort_values("mean_rank_most_expensive_first")["state"].tolist()
    grid = main.pivot_table(index="state", columns="year_month",
                            values="rank_most_expensive_first").reindex(order)
    national = national.sort_values("year_month")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 13.5), gridspec_kw={"height_ratios": [3.3, 1]},
    )
    mesh = ax1.imshow(grid.values, aspect="auto", cmap="viridis_r",
                      interpolation="nearest", vmin=1, vmax=len(order))
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order, fontsize=9)
    cols = list(grid.columns)
    step = 4
    ax1.set_xticks(range(0, len(cols), step))
    ax1.set_xticklabels(cols[::step], rotation=60, fontsize=9)
    ax1.grid(False)
    ax1.set_title("Monthly rank of each state (1 = most expensive); states "
                  "ordered by mean rank", fontsize=11)
    bar = fig.colorbar(mesh, ax=ax1, fraction=0.03, pad=0.015)
    bar.ax.tick_params(labelsize=9)
    bar.set_label("rank (1 = dearest, 37 = cheapest)", fontsize=9)

    x = np.arange(len(national))
    ax2.plot(x, national["spearman_vs_previous_month"], color=style.PALETTE[0],
             linewidth=2.0, marker="o", markersize=3.5,
             label="vs previous month (churn)")
    ax2.plot(x, national["spearman_vs_first_month"], color=style.PALETTE[1],
             linewidth=2.0, linestyle="--", marker="s", markersize=3.2,
             label=f"vs first month, {national['year_month'].iloc[0]} (decay)")
    ax2.axhline(0, color=style.MUTED_COLOUR, linewidth=0.8)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(national["year_month"].iloc[::step], rotation=60, fontsize=9)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(min(-0.2, float(national[["spearman_vs_previous_month",
                                           "spearman_vs_first_month"]].min().min()) - 0.05),
                 1.05)
    ax2.set_ylabel("Spearman's rho")
    ax2.set_title("Stability of the ranking of states over time", fontsize=10.5)
    ax2.legend(fontsize=9, loc="lower left")

    fig.tight_layout(rect=(0, 0.015, 1, 1))
    return style.finish(
        fig, _fig("A12_petrol_price_rank_stability"), NBS,
        "Ranks use average ranks for tied prices (11 tie groups, none larger than "
        "3 states), so rho is an exact Spearman coefficient. Churn: correlation "
        "of each month's ranking with the previous month's. Decay: correlation "
        "with the first month's ranking; a falling line means the initial "
        "geography of petrol prices dissolves over time.",
    )


def _a12_zones(zones: pd.DataFrame) -> Path:
    zones = zones.sort_values("year_month")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.2),
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    for zone in ZONE_ORDER:
        sub = zones[zones["zone"] == zone]
        if sub.empty:
            continue
        zs = ZONE_STYLE[zone]
        ax1.plot(_months(sub), sub["zone_premium_vs_national_pct"],
                 color=zs["color"], linestyle=zs["linestyle"], marker=zs["marker"],
                 markersize=3.2, linewidth=1.6,
                 label=f"{zone} ({int(sub['states_in_zone'].iloc[0])} states)")
    ax1.axhline(0, color=style.TEXT_COLOUR, linewidth=0.9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.set_ylabel("Zone premium over the national mean (%)")
    ax1.set_xlabel("Month")
    ax1.set_title("Geopolitical zones relative to the national mean", fontsize=11)
    ax1.legend(fontsize=9, ncol=2, loc="best")

    period = (zones.groupby("zone")
              .agg(premium=("zone_mean_premium_over_period_pct", "first"),
                   within_cv=("within_zone_cv_pct", "mean"),
                   n=("states_in_zone", "first"))
              .reindex(ZONE_ORDER).dropna().sort_values("premium"))
    # Zone name and its internal dispersion go in the tick label; only the
    # short value sits beside the bar, so negative bars cannot collide with
    # the labels.
    labels = [
        f"{zone}\n({int(row['n'])} states; within-zone CV {row['within_cv']:.1f}%)"
        for zone, row in period.iterrows()
    ]
    ax2.barh(labels, period["premium"],
             color=[ZONE_STYLE[z]["color"] for z in period.index], height=0.6)
    ax2.axvline(0, color=style.TEXT_COLOUR, linewidth=0.9)
    for pos, value in enumerate(period["premium"]):
        ax2.text(value, pos, f" {value:+.2f}% ", va="center",
                 ha="left" if value >= 0 else "right",
                 fontsize=9, color=style.TEXT_COLOUR)
    lo, hi = period["premium"].min(), period["premium"].max()
    pad = max(abs(lo), abs(hi)) * 1.7
    ax2.set_xlim(-pad, pad)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_xlabel("Mean premium over the period (%)")
    ax2.set_title("Period average by zone", fontsize=11)

    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    return style.finish(
        fig, _fig("A12_petrol_price_zones"), NBS,
        "Zone price = unweighted mean of its member states. Within-zone CV = "
        "population sd / mean across the zone's states, averaged over months; "
        "set against the zone premiums, it shows whether zones differ more from "
        "each other than their own states differ internally. Descriptive only.",
    )


def fig_a12() -> list[Path]:
    main = _table(A12_STEM)
    national = _table(f"{A12_STEM}_national")
    states = _table(f"{A12_STEM}_states")
    zones = _table(f"{A12_STEM}_zones")
    if any(frame is None or frame.empty for frame in (main, national, states, zones)):
        return []
    return [
        _a12_national(national),
        _a12_premiums(states),
        _a12_ranks(main, national, states),
        _a12_zones(zones),
    ]


FIGURE_BUILDERS: dict[str, Callable[[], list[Path]]] = {
    "A1": fig_a1, "A2": fig_a2, "A3": fig_a3, "A4": fig_a4, "A5": fig_a5,
    "A6": fig_a6, "A7": fig_a7, "A8": fig_a8, "A9": fig_a9, "A10": fig_a10,
    "A11": fig_a11, "A12": fig_a12,
}


def run(only: str | None = None) -> dict[str, Any]:
    """Build every figure, tolerating an analysis whose table is absent."""
    with stage("figures: publication charts", LOG):
        style.apply_style()
        settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        selected = {only: FIGURE_BUILDERS[only]} if only else FIGURE_BUILDERS
        written: dict[str, Any] = {}
        for analysis_id, builder in selected.items():
            try:
                paths = builder()
            except Exception as exc:  # noqa: BLE001 - one bad chart must not
                LOG.error("figure %s failed: %s: %s", analysis_id,
                          type(exc).__name__, exc)
                written[analysis_id] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            written[analysis_id] = [p.name for p in paths]

        total = sum(len(v) for v in written.values() if isinstance(v, list))
        LOG.info("wrote %d figure(s) at %d dpi", total, settings.FIGURE_DPI)
        return written
