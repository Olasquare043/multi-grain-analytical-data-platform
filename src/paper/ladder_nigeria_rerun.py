"""Task 3 (paper repair pass): re-run the Nigerian petrol-price ladder at 20
seeds instead of notebooks/01's 5, and additionally persist every rung's
per-observation absolute test-set errors so Task 4's paired bootstrap can run
without a further re-fit.

This mirrors notebooks/01_nigeria_petrol_forecasting.ipynb cell for cell --
same query, same feature builders, same two hyperparameter variants, same
CV-based capacity selection -- and changes only two things: REPEAT_SEEDS
(1..20 instead of 1..5) and the addition of per-row error capture. It imports
the existing src/modelling/* modules rather than re-implementing them, per
the constraint that new code lives under src/paper/ and existing modules are
not modified.

Existing outputs (outputs/tables/model_ladder_nigeria.csv,
outputs/tables/ladder_paired_comparisons_nigeria.csv, docs/, anything under
outputs/ outside outputs/paper/) are never written to by this script.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import settings
from src.modelling.splits import nigeria_time_split, assert_split_is_time_ordered
from src.modelling.features import add_leaky_state_avg_nigeria, add_pit_state_avg_nigeria
from src.modelling.encoders import one_hot_encode, target_encode_pit
from src.modelling.ladder import (
    run_rung, score_predictions, LGBM_PARAMS,
    VARIANT_ORIGINAL, VARIANT_CAPACITY_CONTROLLED, VARIANT_REFERENCE,
)
from src.modelling.tuning import expanding_window_folds, select_hyperparameters_cv
from src.modelling.baselines import predict_train_mean, predict_last_value_carried_forward
from src.modelling.repeats import aggregate_repeats

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
ERRORS_DIR = PAPER_DIR / "errors" / "ng"
REPEAT_SEEDS_20 = tuple(range(1, 21))
TARGET = "target_price_next_month"


def main() -> None:
    started = time.perf_counter()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)
    raw = con.execute('''
        SELECT f.month_key,
               f.geo_key,
               g.geo_name       AS state,
               g.region_group   AS region_group,
               d.month_number   AS month_of_year,
               f.price_ngn
        FROM fact_fuel_price_monthly f
        JOIN dim_geography g ON g.geo_key = f.geo_key AND g.is_current
        JOIN dim_date d      ON d.date_key = f.month_key
        ORDER BY g.geo_name, f.month_key
    ''').df()
    con.close()
    assert len(raw) == 1147, f"expected 1,147-row panel, got {len(raw)}"

    raw = raw.sort_values(["state", "month_key"]).reset_index(drop=True)
    raw["target_price_next_month"] = raw.groupby("state")["price_ngn"].shift(-1)
    modeling = raw.dropna(subset=["target_price_next_month"]).copy()
    modeling["region_group"] = modeling["region_group"].astype("category")

    split = nigeria_time_split(modeling)
    assert_split_is_time_ordered(split, time_col="month_key")

    modeling = add_leaky_state_avg_nigeria(modeling, state_col="state")
    modeling = add_pit_state_avg_nigeria(modeling, state_col="state")
    modeling = target_encode_pit(
        modeling, cat_col="state", target_col=TARGET,
        time_col="month_key", smoothing=10.0, out_col="state_target_enc",
    )
    split = nigeria_time_split(modeling)
    assert_split_is_time_ordered(split, time_col="month_key")
    y_train, y_test = split.train[TARGET], split.test[TARGET]

    # Row key for pairing per-observation errors across rungs -- fixed across
    # every seed and rung for Task A (no sampling; the panel is byte-identical
    # every repeat), so a plain (state, month_key) pair is a safe, stable key.
    row_keys = split.test[["state", "month_key"]].reset_index(drop=True)

    print(f"train: {len(split.train)} rows, test: {len(split.test)} rows "
          f"({split.test['month_key'].nunique()} months)")

    FEATURES_V0 = ["price_ngn"]
    train_months = sorted(split.train["month_key"].unique())
    cv_folds = expanding_window_folds(train_months, min_train_months=15, val_block=2)
    tuning = select_hyperparameters_cv(
        split.train, feature_cols=FEATURES_V0, target_col=TARGET,
        month_col="month_key", folds=cv_folds,
    )
    print(f"CV search: {len(tuning.cv_table)} candidates x up to {tuning.n_folds} folds "
          f"in {tuning.seconds:.1f}s -> best_params={tuning.best_params}, "
          f"n_estimators={tuning.frozen_n_estimators}")

    CAPACITY_CONTROLLED_PARAMS = {
        **tuning.best_params,
        "n_estimators": tuning.frozen_n_estimators,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 796,
        "n_jobs": 4,
        "verbosity": -1,
    }

    def reference_results(seed):
        ref_mean = score_predictions(
            "REF_mean", "reference: predict the training mean price for every test row",
            y_test, predict_train_mean(y_train, len(y_test)), n_train=len(y_train),
            notes="no model, no features", variant=VARIANT_REFERENCE, seed=seed,
        )
        ref_walk = score_predictions(
            "REF_heuristic", "reference: random walk -- next month's price = this month's price",
            y_test, predict_last_value_carried_forward(split.test, "price_ngn"), n_train=len(y_train),
            notes="deterministic; zero spread across repeats", variant=VARIANT_REFERENCE, seed=seed,
        )
        return [ref_mean, ref_walk]

    RUNG_SPECS: list[dict] = []

    def add_rung(rung, description, notes, feature_cols=None, kind="columns"):
        RUNG_SPECS.append({"rung": rung, "description": description, "notes": notes,
                            "feature_cols": feature_cols, "kind": kind})

    add_rung("V0", "naive baseline: current month's raw price only",
              "no data engineering", FEATURES_V0)
    FEATURES_V1 = FEATURES_V0
    add_rung("V1", "quality-gated: identical features, sourced from fact_fuel_price_monthly "
                   "after the quality gate (0 of 1,147 rows rejected)",
              "V0 and V1 read identical validated rows", FEATURES_V1)
    FEATURES_V2 = FEATURES_V1 + ["region_group", "month_of_year"]
    add_rung("V2", "+ dimensional features: region_group (dim_geography) and month_of_year (dim_date)",
              "conformed-dimension context", FEATURES_V2)
    FEATURES_V3A = FEATURES_V2 + ["state_avg_price_leaky"]
    add_rung("V3a", "LEAKY, DO NOT TRUST THIS NUMBER: + state_avg_price_leaky "
                    "(this state's average price over ALL months, including future ones)",
              "LEAKY -- future months present", FEATURES_V3A)
    FEATURES_V3B = FEATURES_V2 + ["state_avg_price_pit"]
    add_rung("V3b", "+ state_avg_price_pit (this state's average price, strictly-prior months only)",
              "point-in-time-correct version of V3a", FEATURES_V3B)
    COMMON_V4 = FEATURES_V3B
    train_oh, test_oh = one_hot_encode(
        split.train[COMMON_V4 + ["state"]], split.test[COMMON_V4 + ["state"]], col="state"
    )
    n_onehot_cols = train_oh.shape[1] - len(COMMON_V4)
    add_rung("V4a", f"+ state identity, ONE-HOT encoded ({n_onehot_cols} binary columns)",
              "one-hot encoding of the 37-state categorical", kind="onehot")
    FEATURES_V4B = COMMON_V4 + ["state_target_enc"]
    add_rung("V4b", "+ state identity, POINT-IN-TIME TARGET encoded (1 numeric column)",
              "smoothed, point-in-time-correct target encoding (smoothing=10)", FEATURES_V4B)

    feature_set_by_rung = {}
    for spec in RUNG_SPECS:
        if spec["kind"] == "onehot":
            feature_set_by_rung[spec["rung"]] = (
                COMMON_V4 + [f"state (one-hot, {n_onehot_cols} categories)"],
                len(COMMON_V4) + n_onehot_cols,
            )
        else:
            feature_set_by_rung[spec["rung"]] = (spec["feature_cols"], len(spec["feature_cols"]))

    VARIANTS = ((VARIANT_ORIGINAL, LGBM_PARAMS),
                (VARIANT_CAPACITY_CONTROLLED, CAPACITY_CONTROLLED_PARAMS))

    all_results = []
    # error_rows[(variant, rung)] -> list of per-seed abs-error arrays, aligned to row_keys
    error_rows: dict[tuple[str, str], list[np.ndarray]] = {}

    for seed in REPEAT_SEEDS_20:
        t0 = time.perf_counter()
        seed_results = list(reference_results(seed))
        for spec in RUNG_SPECS:
            if spec["kind"] == "onehot":
                X_train, X_test = train_oh, test_oh
            else:
                X_train = split.train[spec["feature_cols"]]
                X_test = split.test[spec["feature_cols"]]
            for variant, params in VARIANTS:
                result = run_rung(
                    spec["rung"], spec["description"], X_train, y_train, X_test, y_test,
                    notes=spec["notes"], params=params, variant=variant, seed=seed,
                )
                seed_results.append(result)
                abs_err = np.abs(result.predictions - y_test.to_numpy()).astype(np.float32)
                key = (variant, spec["rung"])
                error_rows.setdefault(key, []).append(abs_err)
        all_results.extend(seed_results)
        fitted = [r for r in seed_results if r.variant != VARIANT_REFERENCE]
        print(f"seed {seed}/20: {len(fitted)} fits in {time.perf_counter() - t0:.2f}s")

    print(f"\n{len(all_results)} results total "
          f"({len(RUNG_SPECS)} rungs x 2 variants x {len(REPEAT_SEEDS_20)} seeds, plus references)")

    # --- write per-observation error parquet, one file per (variant, rung) ---
    total_error_rows = 0
    for (variant, rung), arrays in error_rows.items():
        frame = row_keys.copy()
        long_rows = []
        for seed, arr in zip(REPEAT_SEEDS_20, arrays):
            block = frame.copy()
            block["seed"] = np.int16(seed)
            block["abs_error"] = arr
            long_rows.append(block)
        out = pd.concat(long_rows, ignore_index=True)
        out["abs_error"] = out["abs_error"].astype(np.float32)
        variant_slug = "original" if variant == VARIANT_ORIGINAL else "capacity_controlled"
        path = ERRORS_DIR / f"{variant_slug}__{rung}.parquet"
        out.to_parquet(path, index=False)
        total_error_rows += len(out)
    print(f"wrote per-observation errors: {len(error_rows)} files, {total_error_rows:,} rows total, "
          f"under {ERRORS_DIR}")

    # --- aggregate ladder table (Task 3's rerun deliverable) ---
    table = aggregate_repeats(all_results, mae_col="mae_ngn")
    table["feature_set"] = table["rung"].map(
        lambda r: ",".join(feature_set_by_rung[r][0]) if r in feature_set_by_rung else ""
    )
    table["n_features"] = table["rung"].map(
        lambda r: feature_set_by_rung[r][1] if r in feature_set_by_rung else np.nan
    )
    out_path = PAPER_DIR / "ladder_nigeria_20seed_raw.csv"
    table.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(table)} rows)")

    elapsed = time.perf_counter() - started
    print(f"\nnigeria 20-seed rerun wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
