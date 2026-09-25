"""Task 3/4 support: re-fit the NYC ladder once, at the SAME 5 seeds already
published (notebooks/02_nyc_trip_duration.ipynb), purely to capture
per-observation absolute test-set errors that were never persisted from the
original run. Task 3's own aggregate numbers (model_ladder_nyc.csv) are
NOT regenerated in place -- this writes a separate outputs/paper/
ladder_nyc_refit.csv, which Task 3's export script then cross-checks
cell-by-cell against the original (outputs/paper/refit_verification.csv).

To keep this under the user-approved ~50 minute budget, the capacity-
controlled hyperparameters are HARDCODED from docs/modelling_notes.md
("Task B: ... The winner: num_leaves=31, learning_rate=0.10,
min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0, frozen at
n_estimators=182") rather than re-run through the ~12-minute expanding-
window CV search notebooks/02 performs once. Everything else -- queries,
feature builders, encoders, splits, hyperparameters, seeds -- is identical
to notebooks/02, reusing the existing src/modelling/* modules rather than
re-implementing them.

Row pairing across rungs (needed for Task 4's per-observation paired
bootstrap) uses a compact int64 `row_key`: the same leading-8-hex-digit
MD5 prefix of `trip_id` that src/modelling/splits.py already uses for
content-addressed bucketing, computed identically on the raw (V0) and
gated (V1-V5) sides. A physical trip therefore carries the same row_key
whether it is read from the raw TLC Parquet or from fact_trip, so V0-vs-V1
(the one adjacent pair that crosses the raw/gated boundary, on DIFFERENT
row sets of different sizes) can still be paired on the intersection.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.utils.db import connect
from src.modelling.splits import (
    nyc_time_split, assert_split_is_time_ordered,
    NYC_TRAIN_MONTHS, NYC_TEST_MONTHS, NYC_BUCKET_COUNT, bucket_for_seed,
    trip_bucket_predicate, RAW_TRIP_ID_SQL,
)
from src.modelling.features import add_leaky_zone_hour_avg_nyc, add_pit_zone_hour_avg_nyc
from src.modelling.encoders import one_hot_encode, target_encode_pit
from src.modelling.ladder import (
    run_rung, score_predictions, LGBM_PARAMS,
    VARIANT_ORIGINAL, VARIANT_CAPACITY_CONTROLLED, VARIANT_REFERENCE,
)
from src.modelling.baselines import predict_train_mean, predict_distance_over_mean_speed
from src.modelling.repeats import aggregate_repeats

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
ERRORS_DIR = PAPER_DIR / "errors" / "nyc"
REPEAT_SEEDS = (1, 2, 3, 4, 5)          # identical to the published run
TARGET = "trip_duration_seconds"
ROW_KEY_SQL = "('0x' || substr({expr}, 1, 8))::UBIGINT"

# Hardcoded from docs/modelling_notes.md "Task B: Two model-capacity variants
# here too" -- the frozen result of notebooks/02's one-off CV search, reused
# verbatim here to avoid a ~12-minute re-search that changes nothing this
# script needs (per-observation errors, not a fresh hyperparameter choice).
CAPACITY_CONTROLLED_PARAMS = dict(
    num_leaves=31, learning_rate=0.10, min_child_samples=50,
    reg_alpha=1.0, reg_lambda=1.0, n_estimators=182,
    max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=796, n_jobs=4, verbosity=-1,
)

BYTES_PER_ROW_ESTIMATE = 24  # row_key int64 (8) + abs_error f32 (4) + seed i16 (2) + parquet/dict overhead margin
SIZE_LIMIT_BYTES = int(1.5 * 1024 ** 3)
SUBSAMPLE_CAP = 2_000_000


def pull_raw(con, months, bucket):
    month_list = ", ".join(str(m) for m in months)
    row_key_expr = ROW_KEY_SQL.format(expr=RAW_TRIP_ID_SQL)
    query = f'''
        SELECT
            PULocationID                                              AS pickup_zone_raw,
            EXTRACT(hour  FROM tpep_pickup_datetime)::TINYINT          AS pickup_hour,
            date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime)
                                                                        AS trip_duration_seconds,
            trip_distance                                             AS trip_distance_miles,
            passenger_count,
            {row_key_expr}                                            AS row_key
        FROM read_parquet('{str(settings.RAW_NYC / "*.parquet")}')
        WHERE EXTRACT(year FROM tpep_pickup_datetime) = 2024
          AND EXTRACT(month FROM tpep_pickup_datetime) IN ({month_list})
          AND {trip_bucket_predicate(bucket, RAW_TRIP_ID_SQL)}
        ORDER BY {RAW_TRIP_ID_SQL}
    '''
    return con.execute(query).df()


def pull_gated(con, months, bucket):
    month_list = ", ".join(str(m) for m in months)
    row_key_expr = ROW_KEY_SQL.format(expr="t.trip_id")
    query = f'''
        SELECT
            t.pickup_geo_key, t.pickup_hour, t.trip_distance_miles, t.passenger_count,
            t.trip_duration_seconds,
            g.parent_geo_name  AS borough, g.region_group AS service_zone,
            d.day_of_week, d.is_weekend, d.year, d.month_number AS month,
            w.precipitation_mm, w.temp_max_c,
            {row_key_expr}     AS row_key
        FROM fact_trip t
        JOIN dim_geography g ON g.geo_key = t.pickup_geo_key
        JOIN dim_date d      ON d.date_key = t.pickup_date_key
        LEFT JOIN fact_weather_daily w ON w.date_key = t.pickup_date_key
        WHERE d.year = 2024 AND d.month_number IN ({month_list})
          AND {trip_bucket_predicate(bucket, "t.trip_id")}
        ORDER BY t.trip_id
    '''
    return con.execute(query).df()


class RepeatData:
    def __init__(self, seed):
        self.seed = seed
        self.bucket = bucket_for_seed(seed)
        t0 = time.perf_counter()
        con = connect(settings.WAREHOUSE_DB, read_only=True, memory_limit="2GB", threads=4)
        self.raw_train = pull_raw(con, NYC_TRAIN_MONTHS, self.bucket)
        self.raw_test = pull_raw(con, NYC_TEST_MONTHS, self.bucket)
        gated_train = pull_gated(con, NYC_TRAIN_MONTHS, self.bucket)
        gated_test = pull_gated(con, NYC_TEST_MONTHS, self.bucket)
        con.close()

        modeling = pd.concat([gated_train, gated_test], ignore_index=True)
        del gated_train, gated_test
        gc.collect()

        for col in ("borough", "service_zone"):
            modeling[col] = modeling[col].astype("category")
        for col in ("trip_distance_miles", "precipitation_mm", "temp_max_c"):
            modeling[col] = modeling[col].astype("float32")

        modeling = add_leaky_zone_hour_avg_nyc(modeling)
        modeling = add_pit_zone_hour_avg_nyc(modeling)
        modeling = target_encode_pit(
            modeling, cat_col="pickup_geo_key", target_col=TARGET,
            time_col="month", smoothing=50.0, out_col="zone_target_enc",
        )
        self.split = nyc_time_split(modeling)
        assert_split_is_time_ordered(self.split, time_col="month")
        del modeling
        gc.collect()

        self.y_train = self.split.train[TARGET]
        self.y_test = self.split.test[TARGET]
        self.seconds = time.perf_counter() - t0

    def build_one_hot(self, common_cols):
        return one_hot_encode(
            self.split.train[common_cols + ["pickup_geo_key"]],
            self.split.test[common_cols + ["pickup_geo_key"]],
            col="pickup_geo_key",
        )


def reference_results(data, seed):
    ref_mean = score_predictions(
        "REF_mean", "reference: predict the training mean trip duration for every test row",
        data.y_test, predict_train_mean(data.y_train, len(data.y_test)),
        n_train=len(data.y_train), notes="no model, no features",
        variant=VARIANT_REFERENCE, seed=seed,
    )
    ref_speed = score_predictions(
        "REF_heuristic", "reference: trip distance / the training set's average speed",
        data.y_test, predict_distance_over_mean_speed(data.split.train, data.split.test),
        n_train=len(data.y_train), notes="no-model domain heuristic",
        variant=VARIANT_REFERENCE, seed=seed,
    )
    return [ref_mean, ref_speed]


def main() -> None:
    started = time.perf_counter()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    RUNG_SPECS: list[dict] = []

    def add_rung(rung, description, notes, feature_cols=None, kind="columns"):
        RUNG_SPECS.append({"rung": rung, "description": description, "notes": notes,
                            "feature_cols": feature_cols, "kind": kind})

    FEATURES_V0 = ["pickup_hour", "pickup_zone_raw", "trip_distance_miles", "passenger_count"]
    add_rung("V0", "raw baseline, un-gated source", "raw TLC source", FEATURES_V0, kind="raw")
    FEATURES_V1 = ["pickup_hour", "pickup_geo_key", "trip_distance_miles", "passenger_count"]
    add_rung("V1", "quality-gated", "gated fact_trip", FEATURES_V1)
    FEATURES_V2 = ["pickup_hour", "trip_distance_miles", "passenger_count",
                   "borough", "service_zone", "day_of_week", "is_weekend"]
    add_rung("V2", "+ dimensional features", "dim_geography/dim_date", FEATURES_V2)
    FEATURES_V3A = FEATURES_V2 + ["zone_hour_avg_duration_leaky"]
    add_rung("V3a", "LEAKY, DO NOT TRUST THIS NUMBER", "leaky historical avg", FEATURES_V3A)
    FEATURES_V3B = FEATURES_V2 + ["zone_hour_avg_duration_pit"]
    add_rung("V3b", "+ zone_hour_avg_duration_pit", "point-in-time correct", FEATURES_V3B)
    COMMON_V4 = FEATURES_V3B
    add_rung("V4a", "+ pickup zone identity, ONE-HOT", "one-hot scipy.sparse", kind="onehot")
    FEATURES_V4B = COMMON_V4 + ["zone_target_enc"]
    add_rung("V4b", "+ pickup zone identity, TARGET encoded", "point-in-time target encoding", FEATURES_V4B)
    FEATURES_V5 = FEATURES_V4B + ["precipitation_mm", "temp_max_c"]
    add_rung("V5", "+ external enrichment (weather)", "fact_weather_daily", FEATURES_V5)

    VARIANTS = ((VARIANT_ORIGINAL, LGBM_PARAMS),
                (VARIANT_CAPACITY_CONTROLLED, CAPACITY_CONTROLLED_PARAMS))

    all_results = []
    # error_frames[(variant, rung)] -> list of small DataFrames (row_key, seed, abs_error)
    error_frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    total_error_rows = 0

    for seed in REPEAT_SEEDS:
        t_seed = time.perf_counter()
        data = RepeatData(seed)
        print(f"seed {seed} (bucket {data.bucket}): loaded in {data.seconds:.1f}s "
              f"[{len(data.split.train):,} train / {len(data.split.test):,} gated test / "
              f"{len(data.raw_test):,} raw test rows]", flush=True)

        seed_results = list(reference_results(data, seed))
        train_oh, test_oh = data.build_one_hot(COMMON_V4)

        for spec in RUNG_SPECS:
            t_rung = time.perf_counter()
            if spec["kind"] == "onehot":
                X_train, X_test = train_oh, test_oh
                y_tr, y_te = data.y_train, data.y_test
                row_key_test = data.split.test["row_key"].to_numpy()
            elif spec["kind"] == "raw":
                cols = spec["feature_cols"]
                X_train, X_test = data.raw_train[cols], data.raw_test[cols]
                y_tr, y_te = data.raw_train[TARGET], data.raw_test[TARGET]
                row_key_test = data.raw_test["row_key"].to_numpy()
            else:
                cols = spec["feature_cols"]
                X_train, X_test = data.split.train[cols], data.split.test[cols]
                y_tr, y_te = data.y_train, data.y_test
                row_key_test = data.split.test["row_key"].to_numpy()

            for variant, params in VARIANTS:
                result = run_rung(
                    spec["rung"], spec["description"], X_train, y_tr, X_test, y_te,
                    notes=spec["notes"], params=params, variant=variant, seed=seed,
                )
                seed_results.append(result)
                abs_err = np.abs(np.asarray(result.predictions) - y_te.to_numpy()).astype(np.float32)
                frame = pd.DataFrame({
                    "row_key": row_key_test.astype(np.int64),
                    "seed": np.int16(seed),
                    "abs_error": abs_err,
                })
                key = (variant, spec["rung"])
                error_frames.setdefault(key, []).append(frame)
                total_error_rows += len(frame)

            print(f"  rung {spec['rung']:4s} (both variants) done in "
                  f"{time.perf_counter() - t_rung:.1f}s", flush=True)

        all_results.extend(seed_results)
        del data, train_oh, test_oh
        gc.collect()
        print(f"seed {seed} complete in {time.perf_counter() - t_seed:.1f}s", flush=True)

    print(f"\n{len(all_results)} results total "
          f"({len(RUNG_SPECS)} rungs x 2 variants x {len(REPEAT_SEEDS)} seeds, plus references)")

    # --- disk footprint estimate, BEFORE writing ---
    est_bytes = total_error_rows * BYTES_PER_ROW_ESTIMATE
    print(f"\nper-observation error footprint estimate: {total_error_rows:,} row-instances "
          f"x ~{BYTES_PER_ROW_ESTIMATE} bytes = {est_bytes / (1024**3):.3f} GiB "
          f"(limit {SIZE_LIMIT_BYTES / (1024**3):.2f} GiB)")

    subsampled = False
    allowed_keys: set[int] | None = None
    if est_bytes > SIZE_LIMIT_BYTES:
        subsampled = True
        # Deterministic content-addressed subsample: the SUBSAMPLE_CAP row_keys
        # with the smallest value modulo a large prime, identical across every
        # rung/config/seed since row_key itself is content-derived (trip MD5
        # prefix), not run-order-derived.
        all_keys = sorted(set(
            int(k) for frames in error_frames.values() for f in frames for k in f["row_key"].to_numpy()
        ))
        allowed_keys = set(all_keys[:SUBSAMPLE_CAP])
        print(f"footprint exceeds limit -- subsampling to a fixed set of "
              f"{len(allowed_keys):,} row_keys (cap {SUBSAMPLE_CAP:,})")

    written_rows = 0
    for (variant, rung), frames in error_frames.items():
        out = pd.concat(frames, ignore_index=True)
        if subsampled:
            out = out[out["row_key"].isin(allowed_keys)]
        variant_slug = "original" if variant == VARIANT_ORIGINAL else "capacity_controlled"
        path = ERRORS_DIR / f"{variant_slug}__{rung}.parquet"
        out.to_parquet(path, index=False)
        written_rows += len(out)
    actual_bytes = sum(f.stat().st_size for f in ERRORS_DIR.glob("*.parquet"))
    print(f"wrote per-observation errors: {len(error_frames)} files, {written_rows:,} rows, "
          f"{actual_bytes / (1024**3):.3f} GiB on disk under {ERRORS_DIR} "
          f"(subsampled={subsampled}"
          + (f", n={len(allowed_keys):,}" if subsampled else "") + ")")

    # --- aggregate table, cross-checked separately against the published one ---
    table = aggregate_repeats(all_results, mae_col="mae_seconds")
    out_path = PAPER_DIR / "ladder_nyc_refit.csv"
    table.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(table)} rows)")

    elapsed = time.perf_counter() - started
    print(f"\nnyc error-capture refit wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
