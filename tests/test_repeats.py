"""Proves the repeated-run statistics behave as docs/modelling_notes.md claims.

These numbers are now the modelling layer's central claim -- which rung-to-rung
differences survive repetition and which dissolve -- so the arithmetic that
produces them is asserted here rather than trusted. The load-bearing property
is that rungs are compared WITHIN a seed: a bug that compared marginal
distributions instead would quietly turn a real effect into "lost in the
noise", or the reverse, with nothing visibly wrong in the output.
"""
from __future__ import annotations

import numpy as np

from src.modelling.ladder import RungResult
from src.modelling.repeats import (
    Pair,
    adjacent_pairs,
    aggregate_repeats,
    best_rung_by_mean_mae,
    paired_comparisons,
)

VARIANT = "test-variant"


def _result(rung: str, seed: int, mae: float, mape: float = 10.0) -> RungResult:
    return RungResult(
        rung=rung, description=f"{rung} description", mae=mae, mape=mape,
        notes="", n_train=100, n_test=20, seconds=0.1, variant=VARIANT, seed=seed,
    )


def test_pairing_is_within_seed_not_between_marginals() -> None:
    """The property the whole comparison rests on.

    Rungs A and B here swing wildly across seeds -- a spread of 40 -- but B is
    always exactly 1.0 better than A *on the same seed*. A within-seed
    comparison sees a perfectly consistent effect; comparing marginal
    distributions would drown it in a standard deviation of ~15.8 and call it
    nothing. The ladder's conclusions depend on getting this right.
    """
    results = []
    for seed, base in zip(range(1, 6), [10.0, 20.0, 30.0, 40.0, 50.0]):
        results.append(_result("A", seed, base))
        results.append(_result("B", seed, base - 1.0))

    table = paired_comparisons(results, [Pair("A", "B")], task="unit")
    row = table.iloc[0]

    assert row["mean_diff_mae"] == -1.0, "B is better than A by exactly 1.0 every time"
    assert row["sd_diff_mae"] == 0.0, "the paired differences are identical, so spread is zero"
    assert row["n_same_direction"] == 5
    assert bool(row["direction_consistent"]) is True
    assert row["better_rung"] == "B"

    # And confirm the marginal spread really is large, so the test is not
    # passing for the trivial reason that the data barely varies.
    marginal_sd = float(np.std([10.0, 20.0, 30.0, 40.0, 50.0], ddof=1))
    assert marginal_sd > 15.0


def test_inconsistent_direction_is_reported_as_inconsistent() -> None:
    """An effect whose sign flips across repeats must not look established."""
    diffs = [-5.0, -4.0, +6.0, -3.0, +5.0]  # mean is negative, two repeats disagree
    results = []
    for seed, diff in zip(range(1, 6), diffs):
        results.append(_result("A", seed, 100.0))
        results.append(_result("B", seed, 100.0 + diff))

    row = paired_comparisons(results, [Pair("A", "B")], task="unit").iloc[0]

    assert row["mean_diff_mae"] < 0, "mean leans toward B"
    assert row["n_same_direction"] == 3, "three of five repeats agree with the mean's sign"
    assert bool(row["direction_consistent"]) is False, (
        "a sign that flips in two of five repeats must never be flagged consistent"
    )
    assert abs(row["mean_over_sd"]) < 1.0, (
        "the mean difference is smaller than the spread of the differences"
    )


def test_aggregate_reports_spread_and_keeps_first_repeat_as_the_single_run() -> None:
    maes = [10.0, 12.0, 14.0, 16.0, 18.0]
    results = [_result("V0", seed, mae) for seed, mae in zip(range(1, 6), maes)]

    row = aggregate_repeats(results, mae_col="mae_x").iloc[0]

    assert row["mae_x"] == 10.0, "the single-run column carries the first repeat (seed 1)"
    assert row["mae_mean"] == 14.0
    assert row["mae_min"] == 10.0
    assert row["mae_max"] == 18.0
    assert row["n_repeats"] == 5
    assert row["seeds"] == "1,2,3,4,5"
    assert np.isclose(row["mae_sd"], np.std(maes, ddof=1)), "sample sd, ddof=1"


def test_best_rung_ignores_the_leaky_rung_and_the_reference_rows() -> None:
    """V3a is knowingly leaky and REF_* are not rungs; neither may win."""
    results = []
    for seed in range(1, 6):
        results.append(_result("V0", seed, 100.0))
        results.append(_result("V3a", seed, 1.0))        # leaky, lowest error
        results.append(_result("REF_mean", seed, 0.5))   # reference, lower still
        results.append(_result("V4a", seed, 90.0))       # the real winner
    summary = aggregate_repeats(results, mae_col="mae_x")

    assert best_rung_by_mean_mae(summary, VARIANT) == "V4a"


def test_adjacent_pairs_walks_the_ladder_in_order() -> None:
    pairs = adjacent_pairs(["V0", "V1", "V2"])
    assert [(p.rung_a, p.rung_b) for p in pairs] == [("V0", "V1"), ("V1", "V2")]
