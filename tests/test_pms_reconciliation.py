"""The pipeline's proof of correctness against an external authority.

Section 3, Source D makes this test mandatory: the assembled national mean must
match figures published independently by the National Bureau of Statistics, to
within 0.05 NGN. **No figure may be produced from a run in which this fails.**

It is the only check in the platform that exercises the whole chain at once --
HTML scrape, heterogeneous Excel parse, month-header inference, de-duplication,
state-name normalisation, silver cleansing and dimensional load -- against a
number nobody involved in building the pipeline chose.

Published reference figures (NGN per litre, national mean retail price of
Premium Motor Spirit):

    2023-11   648.93
    2023-12   671.86
    2024-05   769.62
"""
from __future__ import annotations

import duckdb
import pytest

from config import settings
from tests.conftest import require_table

TARGETS = settings.NBS_RECONCILIATION_TARGETS
TOLERANCE = settings.NBS_RECONCILIATION_TOLERANCE


def _national_means(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """National mean petrol price per month, from the GOLD fact table.

    Computed from ``fact_fuel_price_monthly`` rather than an intermediate, so a
    defect introduced anywhere between the spreadsheet and the star schema will
    surface here.
    """
    rows = con.execute(
        """
        SELECT printf('%04d-%02d', month_key // 10000, (month_key // 100) % 100)
                                             AS year_month,
               round(avg(price_ngn), 2)      AS national_mean
        FROM fact_fuel_price_monthly
        GROUP BY 1
        """
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows}


@pytest.mark.parametrize("month,published", sorted(TARGETS.items()))
def test_national_mean_matches_published_figure(
    con: duckdb.DuckDBPyConnection, month: str, published: float
) -> None:
    """The assembled national mean equals the NBS published figure."""
    require_table(con, "fact_fuel_price_monthly")
    observed = _national_means(con)

    assert month in observed, (
        f"{month} is absent from fact_fuel_price_monthly. The NBS catalogue "
        f"publishes it; a missing month means the workbook parser failed to "
        f"find a header row. Check the per-file diagnostics in "
        f"docs/run_manifest.json under extraction.nbs_pms."
    )
    delta = observed[month] - published
    assert abs(delta) <= TOLERANCE, (
        f"RECONCILIATION FAILED for {month}: assembled {observed[month]:.2f} "
        f"NGN/litre against the published {published:.2f} "
        f"(delta {delta:+.2f}, tolerance {TOLERANCE}). No figure may be "
        f"published from this run until the cause is found."
    )


def test_panel_is_rectangular(con: duckdb.DuckDBPyConnection) -> None:
    """Every federating unit reports in every published month.

    The NBS release is a complete state panel. A month with fewer than the full
    complement means the parser dropped rows -- which is precisely how the three
    phantom "missing months" described in docs/methodology_notes.md were
    originally masked.
    """
    require_table(con, "fact_fuel_price_monthly")
    rows = con.execute(
        """
        SELECT count(*)                   AS observations,
               count(DISTINCT month_key)  AS months,
               count(DISTINCT geo_key)    AS states
        FROM fact_fuel_price_monthly
        """
    ).fetchone()
    observations, months, states = int(rows[0]), int(rows[1]), int(rows[2])

    assert observations == months * states, (
        f"the fuel panel is not rectangular: {observations} observations but "
        f"{months} months x {states} states = {months * states}. Some "
        f"(month, state) cells are missing."
    )


def test_no_unexpected_month_gaps(con: duckdb.DuckDBPyConnection) -> None:
    """There is no calendar gap between the first and last published month.

    Gaps are never interpolated; this test asserts there are none to interpolate.
    If NBS genuinely stops publishing a month, this test should fail and a human
    should record the gap rather than the pipeline papering over it.
    """
    require_table(con, "fact_fuel_price_monthly")
    months = [
        row[0] for row in con.execute(
            """
            SELECT DISTINCT (month_key // 10000) * 12 + ((month_key // 100) % 100)
            FROM fact_fuel_price_monthly ORDER BY 1
            """
        ).fetchall()
    ]
    expected = list(range(min(months), max(months) + 1))
    missing = sorted(set(expected) - set(months))
    assert not missing, (
        f"{len(missing)} calendar month(s) are absent from the fuel panel "
        f"(dense month indices {missing}). Gaps are reported, never "
        f"interpolated -- but an unexpected gap means the parser should be "
        f"checked before the gap is accepted as real."
    )


def test_all_states_are_canonical(con: duckdb.DuckDBPyConnection) -> None:
    """No geopolitical-zone subtotal or Grand Total row reached the fact table."""
    require_table(con, "fact_fuel_price_monthly")
    from config.nigeria_states import CANONICAL_STATES

    names = [
        row[0] for row in con.execute(
            """
            SELECT DISTINCT g.geo_name
            FROM fact_fuel_price_monthly f
            JOIN dim_geography g ON g.geo_key = f.geo_key
            """
        ).fetchall()
    ]
    unexpected = sorted(set(names) - set(CANONICAL_STATES))
    assert not unexpected, (
        f"non-state label(s) reached fact_fuel_price_monthly: {unexpected}. "
        f"The NBS workbooks mix zone subtotals and a Grand Total row into the "
        f"state column; these must be filtered and reported, never loaded."
    )
    assert len(names) == len(CANONICAL_STATES), (
        f"expected all {len(CANONICAL_STATES)} federating units, found "
        f"{len(names)}"
    )
