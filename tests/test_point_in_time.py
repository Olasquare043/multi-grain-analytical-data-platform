"""Proves the leakage fix actually does something, not just that it runs.

Each test builds a tiny synthetic panel with a deliberate late jump in the
target, computes a historical feature both the leaky way and the
point-in-time-correct way, and asserts that an EARLY row's two feature
values differ -- specifically that the leaky version has been pulled toward
the future jump while the point-in-time version has not. A leakage fix that
silently degenerated into a no-op (e.g. an off-by-one that let the current
or a future row back into the "prior" window) would make these assertions
fail, which is the point: this test fails unless the fix is real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modelling.encoders import target_encode_pit
from src.modelling.features import (
    add_leaky_state_avg_nigeria,
    add_leaky_zone_hour_avg_nyc,
    add_pit_state_avg_nigeria,
    add_pit_zone_hour_avg_nyc,
)


def test_nigeria_state_avg_leaky_vs_pit_differ_on_a_late_jump() -> None:
    # One state, four months. The price jumps sharply in the LAST month only.
    # A row-2 feature that "knows about" the row-4 jump is leaking.
    frame = pd.DataFrame({
        "geo_key": ["NG-X"] * 4,
        "month_key": [20230101, 20230201, 20230301, 20230401],
        "price_ngn": [100.0, 100.0, 100.0, 1000.0],
    })

    leaky = add_leaky_state_avg_nigeria(frame)
    pit = add_pit_state_avg_nigeria(frame)

    row2_leaky = leaky.loc[leaky["month_key"] == 20230201, "state_avg_price_leaky"].iloc[0]
    row2_pit = pit.loc[pit["month_key"] == 20230201, "state_avg_price_pit"].iloc[0]

    # Leaky: mean of all 4 months, pulled up by the future jump.
    assert row2_leaky == pytest.approx((100 + 100 + 100 + 1000) / 4)
    # Point-in-time: mean of only the strictly-prior month (row 1) = 100.
    assert row2_pit == pytest.approx(100.0)
    # The whole point: they must not agree once the future actually differs.
    assert row2_leaky != pytest.approx(row2_pit)

    # The very first row has no prior data at all under the correct builder.
    row1_pit = pit.loc[pit["month_key"] == 20230101, "state_avg_price_pit"].iloc[0]
    assert np.isnan(row1_pit)
    # ...but the leaky builder happily gives it a "history" that never existed.
    row1_leaky = leaky.loc[leaky["month_key"] == 20230101, "state_avg_price_leaky"].iloc[0]
    assert not np.isnan(row1_leaky)


def test_nigeria_state_avg_pit_matches_leaky_when_the_future_never_diverges() -> None:
    # Sanity check on the other direction: if nothing changes across time,
    # there is nothing for the leaky version to leak, and (after the first
    # row, which is always NaN under PIT) the two should agree.
    frame = pd.DataFrame({
        "geo_key": ["NG-X"] * 4,
        "month_key": [20230101, 20230201, 20230301, 20230401],
        "price_ngn": [100.0, 100.0, 100.0, 100.0],
    })
    leaky = add_leaky_state_avg_nigeria(frame)
    pit = add_pit_state_avg_nigeria(frame)

    last_leaky = leaky["state_avg_price_leaky"].iloc[-1]
    last_pit = pit.sort_values("month_key")["state_avg_price_pit"].iloc[-1]
    assert last_leaky == pytest.approx(last_pit) == pytest.approx(100.0)


def test_nyc_zone_hour_avg_leaky_vs_pit_differ_on_a_late_jump() -> None:
    # One (zone, hour) cell, three months, with a duration jump in month 3.
    frame = pd.DataFrame({
        "pickup_geo_key": [7, 7, 7, 7, 7, 7],
        "pickup_hour": [8, 8, 8, 8, 8, 8],
        "month": [1, 1, 2, 2, 3, 3],
        "trip_duration_seconds": [300, 300, 300, 300, 3000, 3000],
    })

    leaky = add_leaky_zone_hour_avg_nyc(frame)
    pit = add_pit_zone_hour_avg_nyc(frame)

    month2_leaky = leaky.loc[leaky["month"] == 2, "zone_hour_avg_duration_leaky"].iloc[0]
    month2_pit = pit.loc[pit["month"] == 2, "zone_hour_avg_duration_pit"].iloc[0]

    # Leaky: mean over all 6 rows across all 3 months, pulled up by month 3.
    assert month2_leaky == pytest.approx((300 * 4 + 3000 * 2) / 6)
    # Point-in-time: mean of only month 1's rows (the strictly-prior month) = 300.
    assert month2_pit == pytest.approx(300.0)
    assert month2_leaky != pytest.approx(month2_pit)

    # Month 1 (the cell's first observed month) has no prior month under PIT.
    month1_pit = pit.loc[pit["month"] == 1, "zone_hour_avg_duration_pit"].iloc[0]
    assert np.isnan(month1_pit)


def test_target_encode_pit_differs_from_a_naive_whole_column_encoding() -> None:
    # Same shape of leakage, applied to the V4 encoding rung: a category
    # whose only late observation is a huge spike must not influence an
    # EARLIER row's encoded value for that same category. Two categories (so
    # the "prior" window at month_key=3 is genuinely populated by both,
    # exercising the tie-safe bucket aggregation, not just a single series),
    # each present in every one of 4 months; state Y spikes only in month 4.
    frame = pd.DataFrame({
        "state": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
        "month_key": [1, 2, 3, 4, 1, 2, 3, 4],
        "price": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10_000.0],
    })
    # Naive whole-column-per-category target mean (what an off-the-shelf,
    # non-time-aware target encoder would compute for every row of state Y,
    # including month 3, before the spike has even happened).
    naive = frame.groupby("state")["price"].transform("mean")
    naive_y_row3 = naive[(frame["state"] == "Y") & (frame["month_key"] == 3)].iloc[0]

    pit = target_encode_pit(frame, cat_col="state", target_col="price",
                             time_col="month_key", smoothing=1.0)
    pit_y_row3 = pit.loc[
        (pit["state"] == "Y") & (pit["month_key"] == 3), "state_target_enc"
    ].iloc[0]

    # Naive: state Y's month-3 encoding is dragged up by its own month-4 spike.
    assert naive_y_row3 == pytest.approx((100 + 100 + 100 + 10_000) / 4)
    # Point-in-time: month 3's encoding uses only months 1-2 of state Y
    # (both 100) blended with the contemporaneous cross-state prior mean
    # (also 100 -- state X never spikes) -- nowhere near the naive figure.
    assert pit_y_row3 == pytest.approx(100.0)
    assert pit_y_row3 != pytest.approx(naive_y_row3)
