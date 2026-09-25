"""Follow-up Task A: characterise the real Nigerian undershoot mechanism.

split_config.csv shows the test window's target max (1722.91) does NOT
exceed the training window's target max (1730.00), so the paper's stated
explanation ("trees cannot extrapolate beyond the training target range") is
wrong on the numbers and must be replaced. The candidate mechanism: a tree's
prediction is bounded by the range of LEAF MEANS it learned, not by the raw
range of training targets, and leaf means -- being averages over the rows
that land in each leaf -- span a strictly narrower range than the raw
targets that fed them. This script tests that directly rather than asserting
it: it refits V0 (both configurations, all 20 seeds -- reusing the same
data-loading and feature logic as src/paper/ladder_nigeria_rerun.py and
extrapolation_ceiling.py) and reports the actual predicted-value range
against the training and test target ranges.

Writes:
  - outputs/paper/undershoot_mechanism.csv (a leading comment row states the
    verdict; do not treat the hypothesis as confirmed unless the numbers
    below it actually show a compressed prediction range)
  - outputs/paper/nigeria_national_mean_series.csv (full panel, every month,
    train/test flag, month-on-month % change)
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import settings
from src.modelling.splits import (
    nigeria_time_split, assert_split_is_time_ordered,
    NIGERIA_TRAIN_END_KEY, NIGERIA_TEST_START_KEY, NIGERIA_TEST_END_KEY,
)
from src.modelling.ladder import LGBM_PARAMS

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
REPEAT_SEEDS_20 = tuple(range(1, 21))
TARGET = "target_price_next_month"

# Identical to src/paper/split_config_and_hyperparams.py's NIGERIA_SELECTED /
# CAPACITY_COMMON -- this pass's confirmed re-run of notebooks/01's once-only
# CV search.
CAPACITY_CONTROLLED_PARAMS = dict(
    num_leaves=7, learning_rate=0.10, min_child_samples=5, reg_alpha=0.0, reg_lambda=0.0,
    n_estimators=12, max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=796, n_jobs=4, verbosity=-1,
)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)
    raw = con.execute('''
        SELECT f.month_key, g.geo_name AS state, f.price_ngn
        FROM fact_fuel_price_monthly f
        JOIN dim_geography g ON g.geo_key = f.geo_key AND g.is_current
        ORDER BY g.geo_name, f.month_key
    ''').df()

    # -------------------------------------------------------------- #
    # nigeria_national_mean_series.csv -- the full panel, every month
    # -------------------------------------------------------------- #
    national = con.execute('''
        SELECT f.month_key, avg(f.price_ngn) AS national_mean_price_ngn, count(*) AS n_states
        FROM fact_fuel_price_monthly f
        GROUP BY f.month_key
        ORDER BY f.month_key
    ''').df()
    con.close()

    national["month_key"] = national["month_key"].astype(int)
    national = national.sort_values("month_key").reset_index(drop=True)
    national["window"] = np.where(
        national["month_key"] <= NIGERIA_TRAIN_END_KEY, "train",
        np.where(
            (national["month_key"] >= NIGERIA_TEST_START_KEY)
            & (national["month_key"] <= NIGERIA_TEST_END_KEY),
            "test", "outside_train_and_test",
        ),
    )
    national["mom_pct_change"] = national["national_mean_price_ngn"].pct_change() * 100.0
    national_path = PAPER_DIR / "nigeria_national_mean_series.csv"
    national.to_csv(national_path, index=False)

    train_mom = national.loc[national["window"] == "train", "mom_pct_change"].dropna()
    test_mom = national.loc[national["window"] == "test", "mom_pct_change"].dropna()
    print(f"wrote {national_path} ({len(national)} months)")
    print(f"train month-on-month %% change: mean={train_mom.mean():.4f}, "
          f"sd={train_mom.std(ddof=1):.4f}, max_abs={train_mom.abs().max():.4f}")
    print(f"test  month-on-month %% change: mean={test_mom.mean():.4f}, "
          f"sd={test_mom.std(ddof=1):.4f}, max_abs={test_mom.abs().max():.4f}")

    # -------------------------------------------------------------- #
    # Refit V0 (both configs, 20 seeds) to get actual predicted values
    # -------------------------------------------------------------- #
    raw = raw.sort_values(["state", "month_key"]).reset_index(drop=True)
    raw["target_price_next_month"] = raw.groupby("state")["price_ngn"].shift(-1)
    modeling = raw.dropna(subset=["target_price_next_month"]).copy()

    split = nigeria_time_split(modeling)
    assert_split_is_time_ordered(split, time_col="month_key")
    y_train, y_test = split.train[TARGET], split.test[TARGET]
    FEATURES_V0 = ["price_ngn"]
    X_train, X_test = split.train[FEATURES_V0], split.test[FEATURES_V0]

    train_min, train_max = float(y_train.min()), float(y_train.max())
    test_min, test_max = float(y_test.min()), float(y_test.max())
    n_test = len(y_test)
    y_test_arr = y_test.to_numpy()

    VARIANTS = {
        "original (untuned, 300 trees)": LGBM_PARAMS,
        "capacity-controlled (CV-selected on V0)": CAPACITY_CONTROLLED_PARAMS,
    }

    rows = []
    for variant, base_params in VARIANTS.items():
        per_seed_preds = []
        per_seed_min_max = []
        for seed in REPEAT_SEEDS_20:
            params = dict(base_params)
            params["random_state"] = seed
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            per_seed_preds.append(preds)
            per_seed_min_max.append((seed, float(preds.min()), float(preds.max())))

        pooled_preds = np.concatenate(per_seed_preds)
        pooled_max = float(pooled_preds.max())
        pooled_min = float(pooled_preds.min())
        n_actual_above_pooled_max_ceiling = int((y_test_arr > pooled_max).sum())
        max_actual_missed_by = float(y_test_arr.max() - pooled_max) if n_actual_above_pooled_max_ceiling else 0.0

        rows.append({
            "configuration": variant,
            "n_seeds": len(REPEAT_SEEDS_20),
            "n_test_rows": n_test,
            "pooled_pred_min": round(pooled_min, 4),
            "pooled_pred_max": round(pooled_max, 4),
            "train_target_min": round(train_min, 4),
            "train_target_max": round(train_max, 4),
            "test_target_min": round(test_min, 4),
            "test_target_max": round(test_max, 4),
            "pred_max_below_train_max_by": round(train_max - pooled_max, 4),
            "pred_max_below_test_max_by": round(test_max - pooled_max, 4),
            "n_test_actuals_above_pooled_pred_max": n_actual_above_pooled_max_ceiling,
            "pct_test_actuals_above_pooled_pred_max": round(
                100.0 * n_actual_above_pooled_max_ceiling / n_test, 4),
            "max_actual_missed_by": round(max_actual_missed_by, 4),
            "per_seed_pred_min_max": ";".join(
                f"seed{seed}:[{mn:.2f},{mx:.2f}]" for seed, mn, mx in per_seed_min_max
            ),
        })
        print(f"{variant}: pooled predicted range [{pooled_min:.2f}, {pooled_max:.2f}]; "
              f"train target range [{train_min:.2f}, {train_max:.2f}]; "
              f"{n_actual_above_pooled_max_ceiling}/{n_test} test actuals exceed the "
              f"pooled prediction ceiling")

    out = pd.DataFrame(rows)

    # Verdict, computed from the numbers above, not asserted independently of
    # them: the leaf-mean-ceiling hypothesis predicts pooled_pred_max should
    # sit BELOW train_target_max (predictions bounded by leaf MEANS, which
    # are narrower than the raw target range that produced them) -- a
    # stronger and different claim than "test exceeds train", which the
    # numbers already contradict.
    supported = bool((out["pooled_pred_max"] < out["train_target_max"]).all())
    verdict = (
        "SUPPORTED: for both configurations, the pooled predicted maximum across all "
        "20 seeds sits BELOW the training target maximum itself (not just below the "
        "test target maximum) -- predictions never reach even the most extreme value "
        "seen in training, consistent with predictions being bounded by leaf MEANS "
        "(averages over the rows in each leaf), which are strictly narrower than the "
        "raw training target range that produced them, rather than by the training "
        "target range directly."
        if supported else
        "NOT SUPPORTED: at least one configuration's pooled predicted maximum reaches "
        "or exceeds the training target maximum, which contradicts the leaf-mean-"
        "ceiling hypothesis as stated. Do not assert it in the paper on this evidence."
    )
    print(f"\nVERDICT: {verdict}")

    out_path = PAPER_DIR / "undershoot_mechanism.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {verdict}\n")
    out.to_csv(out_path, mode="a", index=False)
    print(f"wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
