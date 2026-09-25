"""Task 5 (paper repair pass): experimental configuration disclosure.

Writes outputs/paper/split_config.csv (exact train/val/test windows, row
counts, NYC's sampling fraction and realised n, CV fold boundaries, and
target min/max per window per task) and outputs/paper/hyperparameters.csv
(every LightGBM parameter for both configurations of both tasks, the search
grid used during capacity selection, and the selected values).

Read-only against the existing warehouse (data/warehouse.duckdb) and raw NYC
Parquet files. Fits no model and writes nothing outside outputs/paper/.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from config import settings
from src.modelling.splits import (
    NIGERIA_TRAIN_END_KEY, NIGERIA_TEST_START_KEY, NIGERIA_TEST_END_KEY,
    NYC_YEAR, NYC_TRAIN_MONTHS, NYC_TEST_MONTHS, NYC_TRAIN_SAMPLE_SIZE,
    NYC_TEST_SAMPLE_SIZE, NYC_BUCKET_COUNT, REPEAT_SEEDS,
)
from src.modelling.tuning import expanding_window_folds, CANDIDATE_PARAMS
from src.modelling.ladder import LGBM_PARAMS

PAPER_DIR = settings.OUTPUTS_DIR / "paper"

NIGERIA_CV_MIN_TRAIN_MONTHS = 15
NIGERIA_CV_VAL_BLOCK = 2
NYC_CV_MIN_TRAIN_MONTHS = 6
NYC_CV_VAL_BLOCK = 1

# Confirmed by this pass's own re-runs / docs/modelling_notes.md (the frozen,
# already-published result of each task's once-only CV search).
NIGERIA_SELECTED = dict(num_leaves=7, learning_rate=0.10, min_child_samples=5,
                         reg_alpha=0.0, reg_lambda=0.0, n_estimators=12)
NYC_SELECTED = dict(num_leaves=31, learning_rate=0.10, min_child_samples=50,
                     reg_alpha=1.0, reg_lambda=1.0, n_estimators=182)

CAPACITY_COMMON = dict(max_depth=-1, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=796, n_jobs=4, verbosity=-1)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)

    # ------------------------------------------------------------------ #
    # Nigeria: panel, split, target range
    # ------------------------------------------------------------------ #
    raw = con.execute('''
        SELECT f.month_key, g.geo_name AS state, f.price_ngn
        FROM fact_fuel_price_monthly f
        JOIN dim_geography g ON g.geo_key = f.geo_key AND g.is_current
        ORDER BY g.geo_name, f.month_key
    ''').df()
    raw = raw.sort_values(["state", "month_key"]).reset_index(drop=True)
    raw["target_price_next_month"] = raw.groupby("state")["price_ngn"].shift(-1)
    modeling = raw.dropna(subset=["target_price_next_month"]).copy()

    ng_train = modeling[modeling["month_key"] <= NIGERIA_TRAIN_END_KEY]
    ng_test = modeling[
        (modeling["month_key"] >= NIGERIA_TEST_START_KEY)
        & (modeling["month_key"] <= NIGERIA_TEST_END_KEY)
    ]
    ng_train_months = sorted(ng_train["month_key"].unique())
    ng_folds = expanding_window_folds(ng_train_months, NIGERIA_CV_MIN_TRAIN_MONTHS, NIGERIA_CV_VAL_BLOCK)

    # ------------------------------------------------------------------ #
    # NYC: full (unsampled) gated population within the train/test windows,
    # for a target range that is a property of the WINDOW, not of any one
    # seed's content-addressed bucket draw.
    # ------------------------------------------------------------------ #
    nyc_train_months_sql = ", ".join(str(m) for m in NYC_TRAIN_MONTHS)
    nyc_test_months_sql = ", ".join(str(m) for m in NYC_TEST_MONTHS)
    nyc_train_stats = con.execute(f'''
        SELECT count(*) AS n, min(trip_duration_seconds) AS min_t, max(trip_duration_seconds) AS max_t
        FROM fact_trip t JOIN dim_date d ON d.date_key = t.pickup_date_key
        WHERE d.year = {NYC_YEAR} AND d.month_number IN ({nyc_train_months_sql})
    ''').df().iloc[0]
    nyc_test_stats = con.execute(f'''
        SELECT count(*) AS n, min(trip_duration_seconds) AS min_t, max(trip_duration_seconds) AS max_t
        FROM fact_trip t JOIN dim_date d ON d.date_key = t.pickup_date_key
        WHERE d.year = {NYC_YEAR} AND d.month_number IN ({nyc_test_months_sql})
    ''').df().iloc[0]

    con.close()

    # Realised per-seed bucket sizes -- published in docs/modelling_notes.md
    # ("Task B: Sampling" table), reproduced verbatim rather than re-derived,
    # since re-deriving them means re-running the same DuckDB pulls the
    # concurrent NYC refit (src/paper/ladder_nyc_refit.py) is already doing
    # against the same file.
    realised_bucket_sizes = {
        1: {"bucket": 0, "train": 1_955_364, "test": 420_169},
        2: {"bucket": 1, "train": 1_957_342, "test": 420_267},
        3: {"bucket": 2, "train": 1_955_147, "test": 421_283},
        4: {"bucket": 3, "train": 1_959_559, "test": 421_535},
        5: {"bucket": 4, "train": 1_957_859, "test": 421_747},
    }
    mean_train_n = sum(v["train"] for v in realised_bucket_sizes.values()) / len(realised_bucket_sizes)
    mean_test_n = sum(v["test"] for v in realised_bucket_sizes.values()) / len(realised_bucket_sizes)

    # ------------------------------------------------------------------ #
    # split_config.csv
    # ------------------------------------------------------------------ #
    rows = []
    rows.append({
        "task": "nigeria", "window": "train",
        "start": "2023-11 (month_key 20231101)", "end": f"2025-11 (month_key {NIGERIA_TRAIN_END_KEY})",
        "n_rows": len(ng_train), "notes": "25 feature months, no sampling; full panel",
        "target_min": round(float(ng_train["target_price_next_month"].min()), 2),
        "target_max": round(float(ng_train["target_price_next_month"].max()), 2),
    })
    rows.append({
        "task": "nigeria", "window": "test",
        "start": f"2025-12 (month_key {NIGERIA_TEST_START_KEY})", "end": f"2026-04 (last feature month with a target; "
                                                                          f"panel month_key ceiling {NIGERIA_TEST_END_KEY} "
                                                                          f"but 2026-05 rows have no next-month target and are dropped)",
        "n_rows": len(ng_test), "notes": "5 feature months (2025-12..2026-04), predicting target months 2026-01..2026-05",
        "target_min": round(float(ng_test["target_price_next_month"].min()), 2),
        "target_max": round(float(ng_test["target_price_next_month"].max()), 2),
    })
    rows.append({
        "task": "nigeria", "window": "cv_validation (5 expanding-window folds, within training period only)",
        "start": "", "end": "",
        "n_rows": "", "notes": "see nigeria_cv_folds section below", "target_min": "", "target_max": "",
    })
    rows.append({
        "task": "nyc", "window": "train",
        "start": f"{NYC_YEAR}-01", "end": f"{NYC_YEAR}-10",
        "n_rows": int(nyc_train_stats["n"]), "notes": "full gated population in window, before per-seed sampling",
        "target_min": float(nyc_train_stats["min_t"]), "target_max": float(nyc_train_stats["max_t"]),
    })
    rows.append({
        "task": "nyc", "window": "test",
        "start": f"{NYC_YEAR}-11", "end": f"{NYC_YEAR}-12",
        "n_rows": int(nyc_test_stats["n"]), "notes": "full gated population in window, before per-seed sampling",
        "target_min": float(nyc_test_stats["min_t"]), "target_max": float(nyc_test_stats["max_t"]),
    })
    split_df = pd.DataFrame(rows)
    split_df.to_csv(PAPER_DIR / "split_config.csv", index=False)

    # NYC sampling: one content-addressed bucket per seed, 1/17 of the
    # (already date-filtered) population.
    sampling_rows = []
    for seed in REPEAT_SEEDS:
        b = realised_bucket_sizes[seed]
        sampling_rows.append({
            "task": "nyc", "seed": seed, "bucket": b["bucket"],
            "bucket_count": NYC_BUCKET_COUNT,
            "target_train_n": NYC_TRAIN_SAMPLE_SIZE, "target_test_n": NYC_TEST_SAMPLE_SIZE,
            "realised_train_n": b["train"], "realised_test_n": b["test"],
            "train_fraction_of_full_window": round(b["train"] / int(nyc_train_stats["n"]), 6),
            "test_fraction_of_full_window": round(b["test"] / int(nyc_test_stats["n"]), 6),
        })
    pd.DataFrame(sampling_rows).to_csv(PAPER_DIR / "split_config_nyc_sampling.csv", index=False)

    # CV fold boundaries
    ng_fold_rows = [{
        "task": "nigeria", "fold": i + 1,
        "train_months_n": len(f.train_months), "train_end": f.train_months[-1],
        "val_months": ",".join(str(m) for m in f.val_months),
    } for i, f in enumerate(ng_folds)]
    pd.DataFrame(ng_fold_rows).to_csv(PAPER_DIR / "split_config_nigeria_cv_folds.csv", index=False)

    # NYC's CV search ran on the FIRST repeat's already-loaded training bucket
    # (seed 1, bucket 0, 1,955,364 rows) over its 10 training months
    # (docs/modelling_notes.md, "Task B: Two model-capacity variants here
    # too"). Fold boundaries by month count alone are deterministic and do
    # not depend on which bucket was drawn.
    nyc_folds = expanding_window_folds(list(NYC_TRAIN_MONTHS), NYC_CV_MIN_TRAIN_MONTHS, NYC_CV_VAL_BLOCK)
    nyc_fold_rows = [{
        "task": "nyc", "fold": i + 1,
        "train_months_n": len(f.train_months), "train_end": f.train_months[-1],
        "val_months": ",".join(str(m) for m in f.val_months),
    } for i, f in enumerate(nyc_folds)]
    pd.DataFrame(nyc_fold_rows).to_csv(PAPER_DIR / "split_config_nyc_cv_folds.csv", index=False)

    print(f"wrote split_config.csv ({len(split_df)} rows), "
          f"split_config_nyc_sampling.csv ({len(sampling_rows)} rows), "
          f"split_config_nigeria_cv_folds.csv ({len(ng_fold_rows)} rows), "
          f"split_config_nyc_cv_folds.csv ({len(nyc_fold_rows)} rows)")

    # ------------------------------------------------------------------ #
    # hyperparameters.csv
    # ------------------------------------------------------------------ #
    hp_rows = []
    for task in ("nigeria", "nyc"):
        for param, value in LGBM_PARAMS.items():
            hp_rows.append({"task": task, "configuration": "original (untuned, 300 trees)",
                             "parameter": param, "value": value, "role": "fixed, applied to every rung"})
    for task, selected in (("nigeria", NIGERIA_SELECTED), ("nyc", NYC_SELECTED)):
        full = {**selected, **CAPACITY_COMMON}
        for param, value in full.items():
            hp_rows.append({"task": task, "configuration": "capacity-controlled (CV-selected on V0)",
                             "parameter": param, "value": value,
                             "role": "selected by expanding-window CV" if param in selected
                                     else "fixed (not searched), matches the original variant's non-capacity settings"})
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(PAPER_DIR / "hyperparameters.csv", index=False)

    grid_rows = []
    for task in ("nigeria", "nyc"):
        for i, cand in enumerate(CANDIDATE_PARAMS):
            row = {"task": task, "candidate_index": i, **cand}
            grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(PAPER_DIR / "hyperparameters_search_grid.csv", index=False)

    print(f"wrote hyperparameters.csv ({len(hp_df)} rows), "
          f"hyperparameters_search_grid.csv ({len(grid_df)} rows)")


if __name__ == "__main__":
    main()
