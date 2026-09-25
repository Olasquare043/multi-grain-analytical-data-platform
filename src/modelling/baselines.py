"""Reference predictors that fit no model at all.

The ladder reports each rung's error relative to the rung below it, which
answers "did this data engineering step help?" but never answers "is this
level of error any good?". A ladder can improve steadily and still sit below
what a practitioner would get from one line of arithmetic.

These two references per task supply that missing anchor:

  * REF_mean -- predict the training set's mean target for every test row.
    This is the floor. A model that cannot beat it has learned nothing at
    all from its features, whatever its ladder position says.
  * REF_heuristic -- the obvious domain shortcut someone would reach for
    with no model: distance over average speed for a trip duration, and
    last-observed-price-carried-forward (a random walk) for a price series,
    which is the standard naive benchmark in forecasting.

They are reported beside the rungs in the ladder CSVs, under `rung` values
REF_mean and REF_heuristic, but they are NOT rungs: nothing is stacked on
top of them and they are excluded from rung-to-rung comparisons. They are
scored through `ladder.score_predictions`, the same code every fitted rung's
metrics come from, so the comparison is like-for-like.

No published literature figure is quoted anywhere in this layer as a
comparison point. A published NYC trip-duration error is not comparable to
this one unless the test window, the filtering, the sampling and the target
definition all match, and they do not; these internal references serve that
role instead, and honestly (docs/modelling_notes.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600.0


def predict_train_mean(y_train: pd.Series, n_test: int) -> np.ndarray:
    """Predict the training mean for every test row (the learn-nothing floor)."""
    return np.full(n_test, float(y_train.mean()), dtype=float)


def predict_distance_over_mean_speed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    distance_col: str = "trip_distance_miles",
    duration_col: str = "trip_duration_seconds",
) -> np.ndarray:
    """Task B's no-model heuristic: trip distance / the fleet's average speed.

    Average speed is computed as total training distance over total training
    time, NOT as the mean of per-trip speeds. The mean of per-trip ratios is
    dominated by very short trips, where a few seconds of error produces an
    enormous implied speed; the aggregate ratio is what "the average speed of
    a New York taxi" actually means and is what a practitioner would compute.

    Trips with zero recorded distance predict zero seconds. That is left as
    it is rather than patched: this is meant to be the naive heuristic, and
    hiding its weaknesses would overstate the bar the ladder has to clear.
    """
    total_hours = float(train[duration_col].sum()) / SECONDS_PER_HOUR
    if total_hours <= 0:
        raise ValueError("training duration sums to zero; cannot form a mean speed")
    mean_speed_mph = float(train[distance_col].sum()) / total_hours
    return (test[distance_col].to_numpy(dtype=float) / mean_speed_mph) * SECONDS_PER_HOUR


def predict_last_value_carried_forward(
    test: pd.DataFrame, current_value_col: str = "price_ngn"
) -> np.ndarray:
    """Task A's no-model heuristic: next month's price = this month's price.

    The random-walk benchmark, the standard naive comparator for a price
    series. Task A's framing makes it a one-liner: the model's V0 feature is
    already this month's price and the target is next month's, so carrying
    the value forward unchanged is exactly this column, unmodified.

    Worth noting when reading the ladder against it: V0 is a gradient
    booster given this same single column and asked to learn the mapping
    from it. Any gap between V0 and this reference is the model's learned
    adjustment on top of pure persistence -- which, on a series still
    trending upward in the test window, is not guaranteed to be positive.
    """
    return test[current_value_col].to_numpy(dtype=float)
