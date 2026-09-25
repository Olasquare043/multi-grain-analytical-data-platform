"""Proves Task B's sample selection returns identical rows on every run.

This is the test the previous sampling approach would have failed. DuckDB's
`USING SAMPLE n ROWS (reservoir, 796)` fixes the sampler's randomness but not
the order rows arrive in, and reservoir sampling's output depends on arrival
order too -- so two nominally identical executions of notebook 02 drew
different rows and reported different metrics, which made a 3.4% effect
indistinguishable from run-to-run noise.

The replacement buckets rows by a hash of their own content
(`src/modelling/splits.trip_bucket_predicate`), so selection cannot depend on
scan order. Two properties are asserted here:

  1. ORDER INDEPENDENCE -- the same rows presented in a different physical
     order select the identical subset. This is the property that broke.
  2. CROSS-PROCESS STABILITY -- the real predicate, run against the real
     fact_trip table in two SEPARATE OS processes (fresh DuckDB instance,
     fresh page cache state), returns the same row count and the same
     checksum of a numeric column.

The predicate is what decides *which* rows are selected; the surrounding
SELECT list in the notebooks' pull queries cannot change that, so exercising
the predicate here is a faithful test of the notebooks' row selection.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import duckdb
import pytest

from config import settings
from src.modelling.splits import (
    NYC_BUCKET_COUNT,
    REPEAT_SEEDS,
    bucket_for_seed,
    trip_bucket_predicate,
)


def test_bucket_selection_is_independent_of_row_order() -> None:
    """The failure mode that motivated this change, reproduced as an assertion.

    Two tables holding the identical rows in deliberately different physical
    orders must yield the identical selected subset -- not merely the same
    number of rows, the same rows.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE forward AS
        SELECT md5(CAST(i AS VARCHAR)) AS trip_id, i AS payload
        FROM range(20000) t(i)
    """)
    # Same content, deliberately different arrival order.
    con.execute("CREATE TABLE reversed AS SELECT * FROM forward ORDER BY payload DESC")

    predicate = trip_bucket_predicate(3)
    forward_rows = con.execute(
        f"SELECT payload FROM forward WHERE {predicate} ORDER BY payload"
    ).fetchall()
    reversed_rows = con.execute(
        f"SELECT payload FROM reversed WHERE {predicate} ORDER BY payload"
    ).fetchall()
    con.close()

    assert forward_rows, "the predicate selected nothing -- test proves nothing"
    assert forward_rows == reversed_rows, (
        "bucket selection changed when the rows arrived in a different order; "
        "this is exactly the reservoir-sampling defect this approach replaced"
    )


def test_every_row_lands_in_exactly_one_bucket() -> None:
    """The buckets partition the data: no row is droppable or double-counted."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE t AS
        SELECT md5(CAST(i AS VARCHAR)) AS trip_id FROM range(20000) t(i)
    """)
    total = con.execute("SELECT count(*) FROM t").fetchone()[0]

    selected = 0
    for bucket in range(NYC_BUCKET_COUNT):
        selected += con.execute(
            f"SELECT count(*) FROM t WHERE {trip_bucket_predicate(bucket)}"
        ).fetchone()[0]
    con.close()

    assert selected == total, (
        f"buckets covered {selected} of {total} rows -- a partition must cover "
        "every row exactly once"
    )


def test_repeat_seeds_map_to_distinct_buckets() -> None:
    """The five repeats must draw five different samples, not the same one."""
    buckets = [bucket_for_seed(s) for s in REPEAT_SEEDS]
    assert len(set(buckets)) == len(REPEAT_SEEDS), (
        f"seeds {REPEAT_SEEDS} collapsed onto buckets {buckets}; repeats would "
        "not be measuring sampling variation at all"
    )
    assert all(0 <= b < NYC_BUCKET_COUNT for b in buckets)


def _selection_probe_source(bucket: int) -> str:
    """A self-contained script printing 'count,checksum' for one bucket."""
    return textwrap.dedent(f"""
        import duckdb
        con = duckdb.connect({str(settings.WAREHOUSE_DB)!r}, read_only=True)
        con.execute("SET enable_progress_bar=false")  # keeps stdout to one line
        row = con.execute('''
            SELECT count(*), CAST(sum(trip_duration_seconds) AS BIGINT)
            FROM fact_trip
            WHERE year = 2024 AND month BETWEEN 1 AND 10
              AND {trip_bucket_predicate(bucket)}
        ''').fetchone()
        con.close()
        print(f"{{row[0]}},{{row[1]}}")
    """)


@pytest.mark.skipif(
    not settings.WAREHOUSE_DB.exists(),
    reason="warehouse not built; run the pipeline first",
)
def test_selection_is_identical_across_separate_processes() -> None:
    """Run the real selection twice, in two separate OS processes.

    A same-process repeat could be satisfied by a cached result; two
    subprocesses each build their own DuckDB instance and scan afresh.
    """
    probe = _selection_probe_source(bucket_for_seed(1))
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, check=True,
        )
        outputs.append(completed.stdout.strip())

    count, checksum = outputs[0].split(",")
    assert int(count) > 0, "selection returned no rows -- test proves nothing"
    assert outputs[0] == outputs[1], (
        f"two separate processes selected different data: {outputs[0]} vs "
        f"{outputs[1]} (count,checksum of trip_duration_seconds)"
    )


def _fit_probe_source(order_by: str) -> str:
    """A script that pulls one month, fits a booster, and prints a checksum.

    One month keeps the test to a few seconds while staying large enough that
    LightGBM builds histograms across several threads -- the regime where
    row order actually changes the fitted model.
    """
    ordering = f"ORDER BY {order_by}" if order_by else ""
    return textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(settings.ROOT)!r})
        import duckdb
        from lightgbm import LGBMRegressor
        from src.modelling.ladder import LGBM_PARAMS

        con = duckdb.connect({str(settings.WAREHOUSE_DB)!r}, read_only=True)
        con.execute("SET enable_progress_bar=false")
        df = con.execute('''
            SELECT t.pickup_hour, t.pickup_geo_key, t.trip_distance_miles,
                   t.passenger_count, t.trip_duration_seconds
            FROM fact_trip t
            WHERE t.year = 2024 AND t.month = 1
              AND {trip_bucket_predicate(0, "t.trip_id")}
            {ordering}
        ''').df()
        con.close()

        feats = ["pickup_hour", "pickup_geo_key", "trip_distance_miles", "passenger_count"]
        model = LGBMRegressor(**{{**LGBM_PARAMS, "n_estimators": 40}})
        model.fit(df[feats], df["trip_duration_seconds"])
        print(f"{{len(df)}},{{model.predict(df[feats]).sum():.6f}}")
    """)


@pytest.mark.skipif(
    not settings.WAREHOUSE_DB.exists(),
    reason="warehouse not built; run the pipeline first",
)
def test_fitted_model_reproduces_across_processes_when_rows_are_ordered() -> None:
    """A deterministic row SET is not enough; the row ORDER matters too.

    Gradient boosting is order-sensitive -- bagging selects rows by position
    and histogram split ties break by arrival order -- while DuckDB's parallel
    scan returns rows in no guaranteed order. An earlier version of notebook
    02 selected a provably identical set of rows (same count, same target
    checksum, same sum of distances) and still fitted different models
    between runs, which moved one reported rung-to-rung conclusion.

    Measured at the notebook's full scale (1,955,364 rows) while diagnosing
    that: two processes on an UNORDERED pull produced test-prediction sums of
    202,240,914 and 203,933,890, while the same pull with `ORDER BY trip_id`
    produced exactly 208,095,622.694010 in both. This test guards the fix at
    a smaller scale so it stays fast. The instability itself is scale- and
    schedule-dependent, so it is documented from that measurement rather than
    re-demonstrated here, where one month of data may happen to reproduce
    without any ordering at all.
    """
    outputs = []
    for _ in range(2):
        done = subprocess.run(
            [sys.executable, "-c", _fit_probe_source("t.trip_id")],
            capture_output=True, text=True, check=True,
        )
        outputs.append(done.stdout.strip())

    assert outputs[0].split(",")[0] != "0", "probe pulled no rows"
    assert outputs[0] == outputs[1], (
        "two processes fitted different models from an ORDERED pull of "
        f"identical rows: {outputs[0]} vs {outputs[1]}. Every notebook 02 "
        "query must carry an explicit ORDER BY for its numbers to reproduce."
    )
