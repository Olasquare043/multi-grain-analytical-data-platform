"""A12 petrol_price_geography: the centrepiece of the Nigerian analysis.

These tests run the shipped SQL against the built warehouse and check every
property the paper will lean on: the national mean is the reconciled NBS figure,
premiums are defined so that they sum to zero, ranks are complete and exact
under ties, rank correlations are valid, persistence classes follow from their
definition, and the zone decomposition is internally consistent.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from config import settings
from src.analysis.run_analysis import _render, execute_analysis, render_params
from tests.conftest import require_table

N_STATES = 37


@pytest.fixture(scope="module")
def a12(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    require_table(con, "fact_fuel_price_monthly")
    return execute_analysis(con, _render("A12_petrol_price_geography", render_params()))


def test_all_four_outputs_are_produced(a12) -> None:
    assert set(a12) == {"main", "national", "states", "zones"}


def test_panel_is_complete(a12) -> None:
    main = a12["main"]
    months = main["year_month"].nunique()
    assert main["state"].nunique() == N_STATES
    assert len(main) == months * N_STATES
    assert (a12["national"]["states_reporting"] == N_STATES).all()


@pytest.mark.parametrize("month,published", sorted(
    settings.NBS_RECONCILIATION_TARGETS.items()))
def test_national_mean_is_the_reconciled_nbs_figure(a12, month, published) -> None:
    national = a12["national"].set_index("year_month")
    assert abs(national.loc[month, "national_mean_ngn"] - published) \
        <= settings.NBS_RECONCILIATION_TOLERANCE


def test_premiums_sum_to_zero_every_month(a12) -> None:
    """Premiums are relative to a mean that includes the state itself."""
    totals = a12["main"].groupby("year_month")["premium_vs_national_pct"].sum()
    assert (totals.abs() < 0.01).all(), totals[totals.abs() >= 0.01]


def test_ranks_are_complete_and_exact_under_ties(a12) -> None:
    """Average ranks must still sum to 1 + 2 + ... + 37 in every month."""
    sums = a12["main"].groupby("year_month")["rank_most_expensive_first"].sum()
    assert (sums == N_STATES * (N_STATES + 1) / 2).all()
    ranks = a12["main"]["rank_most_expensive_first"]
    assert ranks.min() >= 1 and ranks.max() <= N_STATES


def test_rank_correlations_are_valid(a12) -> None:
    national = a12["national"]
    rho_first = national["spearman_vs_first_month"]
    assert rho_first.iloc[0] == pytest.approx(1.0)
    assert rho_first.between(-1, 1).all()
    rho_prev = national["spearman_vs_previous_month"]
    assert pd.isna(rho_prev.iloc[0]), "no previous month exists for the first"
    assert rho_prev.iloc[1:].between(-1, 1).all()


def test_dispersion_uses_the_population_sd(a12) -> None:
    """37 federating units are the whole population, not a sample."""
    main, national = a12["main"], a12["national"].set_index("year_month")
    for month, group in main.groupby("year_month"):
        expected = 100.0 * group["price_ngn"].std(ddof=0) / group["price_ngn"].mean()
        assert national.loc[month, "cv_pct"] == pytest.approx(expected, abs=1e-3)


def test_state_summary_follows_its_definitions(a12) -> None:
    states = a12["states"]
    months = a12["main"]["year_month"].nunique()
    assert len(states) == N_STATES
    assert (states["months"] == months).all()
    assert (states["share_months_above"]
            == (states["months_above_national_mean"] / months).round(4)).all()
    share = settings.A12_PERSISTENCE_SHARE
    for row in states.itertuples():
        if row.share_months_above >= share:
            assert row.persistence_class == "persistent premium"
        elif 1 - row.share_months_above >= share:
            assert row.persistence_class == "persistent discount"
        else:
            assert row.persistence_class == "no persistent position"
        assert 0 <= row.longest_run_above_months <= row.months_above_national_mean
        assert 0 <= row.longest_run_below_months <= row.months_at_or_below_mean


def test_zone_decomposition_is_consistent(a12) -> None:
    zones = a12["zones"]
    assert zones["zone"].nunique() == 6
    assert (zones.groupby("year_month")["states_in_zone"].sum() == N_STATES).all()
    share = a12["national"]["zone_share_of_variance_pct"]
    assert share.between(0, 100).all()
