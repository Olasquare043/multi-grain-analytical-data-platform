"""Offline tests for the optional cloud module.

None of these tests touches the network or BigQuery. They pin down the parts of
the module whose failure would cost money or leak into the core run: the SQL
assembled for the scale steps, the byte-budget guard, credential handling, and
the isolation of the core pipeline from the cloud dependencies.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from config import settings
from src.cloud import run_cloud
from src.cloud.bq_client import BudgetExceeded, ByteBudget, credentials_status


def _inventory() -> pd.DataFrame:
    return pd.DataFrame([
        {"table_id": "tlc_yellow_trips_2014", "num_rows": 300, "pickup_datetime_type": "TIMESTAMP"},
        {"table_id": "tlc_yellow_trips_2016", "num_rows": 200, "pickup_datetime_type": "DATETIME"},
        {"table_id": "tlc_yellow_trips_2019", "num_rows": 100, "pickup_datetime_type": "TIMESTAMP"},
        {"table_id": "tlc_yellow_trips_2023", "num_rows": 0, "pickup_datetime_type": "TIMESTAMP"},
        {"table_id": "tlc_green_trips_2019", "num_rows": 50, "pickup_datetime_type": "TIMESTAMP"},
        {"table_id": "taxi_zone_geom", "num_rows": 263, "pickup_datetime_type": None},
    ])


def test_c2_plan_selects_the_right_tables() -> None:
    plan = run_cloud.c2_plan(_inventory())
    assert [p["step"] for p in plan] == [
        "one month", "one year", "largest single table", "full yellow archive"]
    assert plan[0]["tables"] == plan[1]["tables"] == [settings.BQ_COMPARISON_TABLE]
    assert plan[2]["tables"] == ["tlc_yellow_trips_2014"]
    archive = plan[3]["tables"]
    assert "tlc_yellow_trips_2023" not in archive, "empty tables are excluded"
    assert "tlc_green_trips_2019" not in archive, "only yellow trips are scaled"
    assert len(archive) == 3


def test_c2_sql_is_well_formed() -> None:
    """Regression test: a placeholder in the header comment once received the
    multi-line FROM clause, turning all but its first line into live SQL."""
    for step in run_cloud.c2_plan(_inventory()):
        sql = step["sql"]
        lines = sql.splitlines()
        assert "{" not in sql and "}" not in sql, "an unfilled placeholder remains"
        assert not any(line.lstrip().startswith("--") and "SELECT CAST" in line
                       for line in lines), "SQL was substituted into a comment"
    month = run_cloud.c2_plan(_inventory())[0]["sql"]
    assert "TIMESTAMP '2019-01-01'" in month and "TIMESTAMP '2019-02-01'" in month
    archive = run_cloud.c2_plan(_inventory())[3]["sql"]
    # Count separators in executable lines only: the header comment itself
    # mentions UNION ALL when describing how the FROM clause is built.
    live = [line for line in archive.splitlines() if not line.lstrip().startswith("--")]
    assert sum("UNION ALL" in line for line in live) == 2
    assert "CAST(pickup_datetime AS DATETIME)" in archive


def test_budget_blocks_a_query_before_it_spends() -> None:
    budget = ByteBudget(session_limit=100)
    budget.check(60, "fits")
    budget.spent = 60
    with pytest.raises(BudgetExceeded):
        budget.check(50, "would exceed the session allowance")


def test_allowance_respects_remaining_free_tier() -> None:
    budget = ByteBudget(
        session_limit=10 ** 15,
        monthly_used_before=settings.BQ_FREE_TIER_BYTES_PER_MONTH - 1000,
    )
    assert budget.allowance() == int(settings.BQ_ABORT_THRESHOLD_FRACTION * 1000)


def test_missing_credentials_are_reported_not_raised(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    ok, message = credentials_status()
    assert ok is False
    assert "not set" in message


def test_implied_cost_uses_tebibytes() -> None:
    assert run_cloud.implied_cost(1024 ** 4, 6.25) == pytest.approx(6.25)


def test_core_pipeline_never_imports_bigquery() -> None:
    """The core run must succeed without the cloud dependencies installed."""
    import src.pipeline  # noqa: F401

    assert "google.cloud.bigquery" not in sys.modules
