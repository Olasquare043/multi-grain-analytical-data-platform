"""A7 is a within-market analysis on a fixed panel; its stated scope must be true.

The SQL header states the exact number of markets and states in the panel. That
sentence goes into the paper, so this test recomputes the panel from the data and
fails if the header, the query output and the continuity rule ever disagree.
"""
from __future__ import annotations

import re

import duckdb
import pandas as pd
import pytest

from config import settings
from src.analysis.run_analysis import _render, execute_analysis, render_params
from src.utils.db import read_sql_file
from tests.conftest import require_table

STEM = "A7_subsidy_structural_break"


@pytest.fixture(scope="module")
def a7(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    require_table(con, "fact_market_price_monthly")
    return execute_analysis(con, _render(STEM, render_params()))["main"]


def test_header_counts_match_the_computed_panel(a7) -> None:
    header = read_sql_file(settings.SQL_ANALYSIS_DIR / f"{STEM}.sql")
    match = re.search(r"FIXED PANEL of (\d+) markets in (\d+) states", header)
    assert match, "the A7 header must state the panel's market and state counts"
    stated_markets, stated_states = int(match.group(1)), int(match.group(2))

    assert a7["panel_markets"].nunique() == 1
    assert int(a7["panel_markets"].iloc[0]) == stated_markets, (
        "the SQL header states a market count the query no longer produces; "
        "update the header, the figure caption and methodology_notes.md together"
    )
    assert int(a7["panel_states"].iloc[0]) == stated_states


def test_every_panel_market_reports_in_every_window_month(con, a7) -> None:
    """The continuity rule, re-checked independently of the query."""
    window = settings.SUBSIDY_WINDOW_MONTHS
    brk = int(settings.SUBSIDY_BREAK_YEAR_MONTH[:4]) * 12 \
        + int(settings.SUBSIDY_BREAK_YEAR_MONTH[5:7])
    markets = [
        entry.split(": ", 1)[1]
        for entry in a7["panel_market_list"].iloc[0].split("; ")
    ]
    categories = tuple(settings.NIGERIA_STAPLE_CATEGORIES)
    coverage = con.execute(
        f"""
        SELECT g.geo_name AS market,
               count(DISTINCT (f.month_key // 10000) * 12
                              + (f.month_key // 100) % 100) AS months
        FROM fact_market_price_monthly f
        JOIN dim_commodity c ON c.commodity_key = f.commodity_key
        JOIN dim_geography g ON g.geo_key = f.geo_key
        WHERE c.category IN {categories} AND NOT c.is_fuel
          AND f.price_type = ? AND f.price_ngn > 0
          AND (f.month_key // 10000) * 12 + (f.month_key // 100) % 100
              BETWEEN ? AND ?
        GROUP BY 1
        """,
        [settings.NIGERIA_PRICE_TYPE, brk - window + 1, brk + window],
    ).df().set_index("market")["months"]

    for market in markets:
        assert coverage.get(market, 0) == 2 * window, (
            f"{market} is in the fixed panel but does not report in all "
            f"{2 * window} window months"
        )


def test_output_is_labelled_within_market(a7) -> None:
    assert (a7["analysis_scope"].str.contains("NOT a national")).all()
    window = settings.SUBSIDY_WINDOW_MONTHS
    assert set(a7["segment"]) <= {"pre", "post"}
    assert a7["months_since_break"].between(-window + 1, window).all()
