"""Point-in-time-correct and deliberately-leaky historical feature builders.

Section 6 of the v3 prompt requires the leaky and correct versions of each
historical-aggregate feature to be two clearly separate, named functions, not
one function behind a boolean flag: a reader should be able to see the two
code paths side by side and understand the difference without reading the
notebook. That is why every pair below is laid out leaky-then-correct, with
the same docstring shape, so the diff between them is the whole point.

Both tasks reuse the same idea at different grains:
  - Task A (Nigeria): "this state's historical average petrol price."
  - Task B (NYC):      "this pickup zone and hour's historical average trip
                         duration."

`frame` in every function below is expected to be the FULL population the
feature will be attached to (train and test rows together, already
concatenated). That is deliberate, not an oversight: a leaky feature's
whole failure mode is that it can see rows from the test period, so it must
be computed with the test rows physically present in `frame`. A
point-in-time-correct feature computed the same way is safe, because for
every test row its "prior" window is entirely inside the training period by
construction of the time split -- it just isn't the whole frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Task A: Nigeria -- state-level historical average petrol price
# --------------------------------------------------------------------------- #
def add_leaky_state_avg_nigeria(
    frame: pd.DataFrame,
    state_col: str = "geo_key",
    price_col: str = "price_ngn",
    out_col: str = "state_avg_price_leaky",
) -> pd.DataFrame:
    """LEAKY. "This state's average price," computed over EVERY month in
    `frame`, including months after the row being featurised. A row in
    2023-11 receives a value that was partly computed from 2026-05, a month
    that had not happened yet. Labelled LEAKY everywhere it is used;
    never trusted as a real number. See add_pit_state_avg_nigeria for the
    correct version of the same idea.
    """
    frame = frame.copy()
    frame[out_col] = frame.groupby(state_col)[price_col].transform("mean")
    return frame


def add_pit_state_avg_nigeria(
    frame: pd.DataFrame,
    state_col: str = "geo_key",
    month_key_col: str = "month_key",
    price_col: str = "price_ngn",
    out_col: str = "state_avg_price_pit",
) -> pd.DataFrame:
    """Point-in-time-correct. "This state's average price using only months
    strictly before the month being predicted," an expanding mean recomputed
    per row from that state's own prior rows only. A state's first observed
    month has no prior data and is NaN by construction (LightGBM handles
    missing values natively; nothing is imputed here). See
    add_leaky_state_avg_nigeria for the leaky version of the same idea.
    """
    frame = frame.sort_values([state_col, month_key_col]).copy()
    grp = frame.groupby(state_col)[price_col]
    frame[out_col] = grp.transform(lambda s: s.shift(1).expanding().mean())
    return frame


# --------------------------------------------------------------------------- #
# Task B: NYC -- (pickup zone, pickup hour) historical average trip duration
# --------------------------------------------------------------------------- #
def add_leaky_zone_hour_avg_nyc(
    frame: pd.DataFrame,
    zone_col: str = "pickup_geo_key",
    hour_col: str = "pickup_hour",
    duration_col: str = "trip_duration_seconds",
    out_col: str = "zone_hour_avg_duration_leaky",
) -> pd.DataFrame:
    """LEAKY. "This zone and hour's average trip duration," computed over the
    ENTIRE dataset in `frame`, including trips from months after the row
    being featurised. A January row receives a value partly computed from
    December. Labelled LEAKY everywhere it is used. See
    add_pit_zone_hour_avg_nyc for the correct version of the same idea.
    """
    frame = frame.copy()
    frame[out_col] = frame.groupby([zone_col, hour_col])[duration_col].transform("mean")
    return frame


def add_pit_zone_hour_avg_nyc(
    frame: pd.DataFrame,
    zone_col: str = "pickup_geo_key",
    hour_col: str = "pickup_hour",
    month_col: str = "month",
    duration_col: str = "trip_duration_seconds",
    out_col: str = "zone_hour_avg_duration_pit",
) -> pd.DataFrame:
    """Point-in-time-correct. "This zone and hour's average trip duration
    using only trips from strictly earlier calendar months," computed at
    month granularity (per section 3.4 / 4.4 of the v3 prompt) rather than
    per-row, because that is the natural update cadence of an aggregate a
    production system would actually refresh.

    Implementation: build a small (zone, hour, month) summary table of
    per-cell sum and count, then take each cell's CUMULATIVE sum/count over
    STRICTLY PRIOR months only (current month's own sum/count subtracted back
    out of its cumulative total), and merge that lookup onto every row by
    (zone, hour, month). A (zone, hour) pair's first observed month has no
    prior data and is NaN by construction, exactly as in the Nigeria version.
    See add_leaky_zone_hour_avg_nyc for the leaky version of the same idea.
    """
    monthly = (
        frame.groupby([zone_col, hour_col, month_col])[duration_col]
        .agg(month_sum="sum", month_count="count")
        .reset_index()
        .sort_values([zone_col, hour_col, month_col])
    )
    grp = monthly.groupby([zone_col, hour_col])
    cum_sum = grp["month_sum"].cumsum()
    cum_count = grp["month_count"].cumsum()
    prior_sum = cum_sum - monthly["month_sum"]
    prior_count = cum_count - monthly["month_count"]
    monthly[out_col] = prior_sum / prior_count.replace(0, np.nan)

    lookup = monthly[[zone_col, hour_col, month_col, out_col]]
    return frame.merge(lookup, on=[zone_col, hour_col, month_col], how="left")
