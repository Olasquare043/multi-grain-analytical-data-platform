"""Time-based train/test splits, shared by both ladder notebooks.

Every split in this module is by TIME, never by random shuffling. This is the
single most important technical requirement in the v3 build: a held-out test
set built by shuffling rows lets a model see information from the future
during training (a lagged feature for a January row computed from a March
row, for instance), which makes it look accurate in evaluation and fail in
real deployment. tests/test_time_split.py asserts, for every split this
module produces, that the maximum training timestamp is strictly earlier
than the minimum test timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------- #
# Task A: Nigerian petrol price forecasting (fact_fuel_price_monthly)
# --------------------------------------------------------------------------- #
# month_key is INTEGER YYYYMM01 (the first day of the month) -- see
# docs/data_dictionary.md. Comparing it as an integer sorts chronologically
# with no string-parsing edge cases.
NIGERIA_TRAIN_END_KEY = 20251101   # last month INCLUDED in training: 2025-11
NIGERIA_TEST_START_KEY = 20251201  # first month INCLUDED in test: 2025-12
NIGERIA_TEST_END_KEY = 20260501    # last month INCLUDED in test: 2026-05

# --------------------------------------------------------------------------- #
# Task B: NYC trip duration (fact_trip)
# --------------------------------------------------------------------------- #
NYC_YEAR = 2024
NYC_TRAIN_MONTHS: tuple[int, ...] = tuple(range(1, 11))  # January - October 2024
NYC_TEST_MONTHS: tuple[int, ...] = (11, 12)              # November - December 2024
#: The sample sizes the superseded reservoir draw targeted. Kept because the
#: bucket count below is chosen to land near them, which is what keeps every
#: figure comparable to the earlier single-run ladder; nothing samples to an
#: exact row count any more, and the notebooks report the realised counts.
NYC_TRAIN_SAMPLE_SIZE = 2_000_000
NYC_TEST_SAMPLE_SIZE = 400_000

# --------------------------------------------------------------------------- #
# Deterministic, content-addressed sampling (replaces DuckDB `USING SAMPLE`)
# --------------------------------------------------------------------------- #
# WHY THIS REPLACED RESERVOIR SAMPLING. `USING SAMPLE n ROWS (reservoir, 796)`
# fixes the sampler's own randomness but NOT which rows reach the sampler in
# which order, and reservoir sampling's output depends on arrival order as
# well as on the seed. Reading through a `read_parquet('*.parquet')` glob does
# not guarantee a stable scan order between container runs, so two nominally
# identical executions drew different rows and produced different metrics
# (docs/methodology_notes.md records the symptom and this resolution).
#
# The replacement selects on CONTENT, not on position or arrival order:
# fact_trip's `trip_id` is already a deterministic MD5 over each trip's
# business attributes (sql/ddl/silver_nyc_trip.sql), so its leading hex digits
# are a uniformly distributed, row-stable pseudo-random number. Taking that
# value modulo NYC_BUCKET_COUNT assigns every trip to a fixed bucket, for all
# time, independent of scan order, thread count, or file layout. Selecting one
# bucket therefore returns byte-identical rows on every run
# (tests/test_sampling_determinism.py proves it across separate processes).
#
# The leading 8 hex digits are parsed directly rather than calling DuckDB's
# `hash()`: the former depends only on the stored string, while the latter
# depends on a library-internal hash function that a DuckDB upgrade could
# legitimately change.
NYC_BUCKET_COUNT = 17
#: 17 buckets over the 2024 windows realises ~1,956,000 training rows and
#: ~421,000 test rows per bucket -- close to the 2,000,000 / 400,000 sizes the
#: reservoir draw targeted, which keeps every figure comparable to the earlier
#: single-run ladder. Exact realised counts are reported by the notebooks
#: rather than assumed, because a bucket's size is a property of the data, not
#: a number this module gets to choose.

#: The five repeats. Seeds 1-5 drive BOTH the model's `random_state` and (Task
#: B only) which bucket is drawn, so a repeat varies the data sample and the
#: model's randomness together rather than isolating one of them.
REPEAT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)


def bucket_for_seed(seed: int) -> int:
    """Which deterministic sample bucket repeat `seed` draws (seeds 1-5 -> 0-4).

    Buckets are assigned by an MD5-derived value, so consecutive bucket
    indices are no more related to each other than any other pair; there is
    nothing to gain from scattering them across the range.
    """
    return (seed - 1) % NYC_BUCKET_COUNT


def trip_bucket_predicate(bucket: int, trip_id_col: str = "trip_id") -> str:
    """SQL predicate selecting one deterministic bucket of trips by content hash.

    Returned as a SQL fragment rather than applied here because the two
    callers select from different sources (the gated `fact_trip` table and the
    raw TLC Parquet, which has no `trip_id` and recomputes the identical MD5
    inline), and both must bucket by exactly the same rule to be comparable.
    """
    if not 0 <= bucket < NYC_BUCKET_COUNT:
        raise ValueError(f"bucket must be in [0, {NYC_BUCKET_COUNT}), got {bucket}")
    return f"('0x' || substr({trip_id_col}, 1, 8))::UBIGINT % {NYC_BUCKET_COUNT} = {bucket}"


#: The raw TLC Parquet carries no trip_id -- the gate assigns it -- so V0's
#: un-gated draw recomputes the identical MD5 expression inline from the raw
#: columns it hashes (sql/ddl/silver_nyc_trip.sql lines 90-100). Same rule,
#: same bucketing, same determinism, on a source that has not been through
#: the quality gate.
RAW_TRIP_ID_SQL = """md5(concat_ws('|',
        CAST(VendorID AS VARCHAR),
        CAST(tpep_pickup_datetime AS VARCHAR),
        CAST(tpep_dropoff_datetime AS VARCHAR),
        CAST(PULocationID AS VARCHAR),
        CAST(DOLocationID AS VARCHAR),
        CAST(trip_distance AS VARCHAR),
        CAST(fare_amount AS VARCHAR),
        CAST(total_amount AS VARCHAR),
        CAST(payment_type AS VARCHAR)
    ))"""


@dataclass(frozen=True)
class Split:
    """A train/test pair plus a human-readable statement of the boundary used."""

    train: pd.DataFrame
    test: pd.DataFrame
    boundary_description: str


def nigeria_time_split(frame: pd.DataFrame, month_key_col: str = "month_key") -> Split:
    """Split fact_fuel_price_monthly by month_key: train <=2025-11, test 2025-12..2026-05.

    25 training months (2023-11..2025-11), 6 test months (2025-12..2026-05),
    matching the boundary stated in section 3.2 of the v3 prompt and repeated
    in docs/modelling_notes.md.
    """
    train = frame[frame[month_key_col] <= NIGERIA_TRAIN_END_KEY].copy()
    test = frame[
        (frame[month_key_col] >= NIGERIA_TEST_START_KEY)
        & (frame[month_key_col] <= NIGERIA_TEST_END_KEY)
    ].copy()
    description = (
        f"train: month_key <= {NIGERIA_TRAIN_END_KEY} "
        f"({train[month_key_col].nunique()} months, {len(train)} rows); "
        f"test: {NIGERIA_TEST_START_KEY} <= month_key <= {NIGERIA_TEST_END_KEY} "
        f"({test[month_key_col].nunique()} months, {len(test)} rows)"
    )
    return Split(train, test, description)


def nyc_time_split(frame: pd.DataFrame, year_col: str = "year",
                    month_col: str = "month") -> Split:
    """Split fact_trip by calendar month within 2024: train Jan-Oct, test Nov-Dec.

    Rows outside year 2024 (the 55 out-of-window pickups documented in
    methodology_notes.md section 7) are dropped from both sides rather than
    silently sorted into either one.
    """
    in_year = frame[frame[year_col] == NYC_YEAR]
    train = in_year[in_year[month_col].isin(NYC_TRAIN_MONTHS)].copy()
    test = in_year[in_year[month_col].isin(NYC_TEST_MONTHS)].copy()
    description = (
        f"train: {NYC_YEAR} months {NYC_TRAIN_MONTHS[0]}-{NYC_TRAIN_MONTHS[-1]} "
        f"({len(train)} rows); test: {NYC_YEAR} months {NYC_TEST_MONTHS} "
        f"({len(test)} rows)"
    )
    return Split(train, test, description)


def assert_split_is_time_ordered(split: Split, time_col: str) -> None:
    """Raise if any training timestamp is not strictly earlier than every test
    timestamp. Used by tests/test_time_split.py; also safe to call from a
    notebook cell as a live, visible assertion rather than a hidden invariant.
    """
    if split.train.empty or split.test.empty:
        raise AssertionError(
            f"split is time-ordered only vacuously: train has {len(split.train)} "
            f"rows, test has {len(split.test)} rows -- at least one side is empty"
        )
    max_train = split.train[time_col].max()
    min_test = split.test[time_col].min()
    if not (max_train < min_test):
        raise AssertionError(
            f"leakage: max training {time_col} ({max_train}) is not strictly "
            f"earlier than min test {time_col} ({min_test})"
        )
