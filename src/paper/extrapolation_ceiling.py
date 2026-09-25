"""Task 7 (paper repair pass): make the extrapolation-ceiling finding visible.

The paper's most persuasive Nigerian result is currently buried in prose: V0
(a gradient-boosted tree given essentially the lag-1 price) performs WORSE
than REF_heuristic (simply carrying that price forward), because the test
window sits at price levels the training window never reached and trees
cannot extrapolate. This script re-fits ONLY V0 (both hyperparameter
configurations, all 20 seeds now used for Task 3/4) to get predictions,
averages each configuration's prediction across seeds per test row (a single
seed would make an arbitrary choice among 20 equally-valid fits), and plots
predicted vs. actual national mean petrol price over the test window against
the random walk and the training-period ceiling.

Reuses the identical data-loading and feature logic as
src/paper/ladder_nigeria_rerun.py (itself mirroring notebooks/01) rather than
re-deriving it a third time.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import settings
from src.viz import style as vizstyle
from src.modelling.splits import nigeria_time_split, assert_split_is_time_ordered
from src.modelling.ladder import LGBM_PARAMS
from src.modelling.baselines import predict_last_value_carried_forward

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
FIG_DIR = PAPER_DIR / "figures"
REPEAT_SEEDS_20 = tuple(range(1, 21))
TARGET = "target_price_next_month"

# Identical to src/paper/split_config_and_hyperparams.py's NIGERIA_SELECTED /
# CAPACITY_COMMON (this pass's confirmed re-run of notebooks/01's once-only
# CV search: num_leaves=7, learning_rate=0.10, min_child_samples=5, no L1/L2,
# n_estimators=12).
CAPACITY_CONTROLLED_PARAMS = dict(
    num_leaves=7, learning_rate=0.10, min_child_samples=5, reg_alpha=0.0, reg_lambda=0.0,
    n_estimators=12, max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=796, n_jobs=4, verbosity=-1,
)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)
    raw = con.execute('''
        SELECT f.month_key, g.geo_name AS state, f.price_ngn
        FROM fact_fuel_price_monthly f
        JOIN dim_geography g ON g.geo_key = f.geo_key AND g.is_current
        ORDER BY g.geo_name, f.month_key
    ''').df()
    con.close()

    raw = raw.sort_values(["state", "month_key"]).reset_index(drop=True)
    raw["target_price_next_month"] = raw.groupby("state")["price_ngn"].shift(-1)
    modeling = raw.dropna(subset=["target_price_next_month"]).copy()

    split = nigeria_time_split(modeling)
    assert_split_is_time_ordered(split, time_col="month_key")
    y_train, y_test = split.train[TARGET], split.test[TARGET]
    FEATURES_V0 = ["price_ngn"]
    X_train, X_test = split.train[FEATURES_V0], split.test[FEATURES_V0]

    VARIANTS = {
        "original (untuned, 300 trees)": LGBM_PARAMS,
        "capacity-controlled (CV-selected on V0)": CAPACITY_CONTROLLED_PARAMS,
    }

    preds_by_variant: dict[str, np.ndarray] = {}
    for variant, base_params in VARIANTS.items():
        seed_preds = []
        for seed in REPEAT_SEEDS_20:
            params = dict(base_params)
            params["random_state"] = seed
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            seed_preds.append(model.predict(X_test))
        preds_by_variant[variant] = np.mean(np.vstack(seed_preds), axis=0)
        print(f"fit {variant}: {len(REPEAT_SEEDS_20)} seeds averaged")

    ref_heuristic_pred = predict_last_value_carried_forward(split.test, "price_ngn")

    test_frame = split.test[["state", "month_key"]].reset_index(drop=True).copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["pred_original"] = preds_by_variant["original (untuned, 300 trees)"]
    test_frame["pred_capacity_controlled"] = preds_by_variant["capacity-controlled (CV-selected on V0)"]
    test_frame["pred_ref_heuristic"] = ref_heuristic_pred

    monthly = test_frame.groupby("month_key", as_index=False).agg(
        actual_national_mean=("actual", "mean"),
        pred_original_national_mean=("pred_original", "mean"),
        pred_capacity_controlled_national_mean=("pred_capacity_controlled", "mean"),
        pred_ref_heuristic_national_mean=("pred_ref_heuristic", "mean"),
        n_states=("actual", "size"),
    )
    monthly = monthly.sort_values("month_key").reset_index(drop=True)

    train_min, train_max = float(y_train.min()), float(y_train.max())
    test_min, test_max = float(y_test.min()), float(y_test.max())
    monthly["train_target_min"] = train_min
    monthly["train_target_max"] = train_max
    monthly["test_target_min"] = test_min
    monthly["test_target_max"] = test_max

    csv_path = PAPER_DIR / "extrapolation_ceiling.csv"
    monthly.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(monthly)} rows)")
    print(f"train target range: {train_min:.2f} - {train_max:.2f}")
    print(f"test target range:  {test_min:.2f} - {test_max:.2f}")

    # ------------------------------------------------------------------ #
    # Figure -- reuses the project's own established, colour-blind-safe
    # (Okabe-Ito) matplotlib style module, for visual consistency with
    # every other figure in the paper.
    # ------------------------------------------------------------------ #
    vizstyle.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    months = monthly["month_key"].astype(str).str[:6]
    x = np.arange(len(monthly))

    ax.plot(x, monthly["actual_national_mean"], label="Actual national mean price",
            **vizstyle.series_style(0), linewidth=2.0, markersize=6)
    ax.plot(x, monthly["pred_original_national_mean"],
            label="V0 prediction (original, 300 trees, mean of 20 seeds)",
            **vizstyle.series_style(1), linewidth=1.6, markersize=5)
    ax.plot(x, monthly["pred_capacity_controlled_national_mean"],
            label="V0 prediction (capacity-controlled, mean of 20 seeds)",
            **vizstyle.series_style(2), linewidth=1.6, markersize=5)
    ax.plot(x, monthly["pred_ref_heuristic_national_mean"],
            label="REF_heuristic (random walk)",
            **vizstyle.series_style(3), linewidth=1.6, markersize=5)

    ax.axhline(train_max, color=vizstyle.MUTED_COLOUR, linestyle=(0, (4, 2)), linewidth=1.4,
               label=f"Max training-period target (NGN {train_max:,.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=0)
    ax.set_xlabel("Test-window target month (state-level rows averaged to national mean)", fontsize=10)
    ax.set_ylabel("Petrol price (NGN/litre)", fontsize=10)
    ax.set_title("Nigeria: predicted vs. actual national mean petrol price, test window\n"
                 "(the extrapolation ceiling: V0 cannot reach price levels never seen in training)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    fig_path = vizstyle.finish(
        fig, FIG_DIR / "fig_extrapolation_ceiling.png",
        "Source: this platform's own gold layer (fact_fuel_price_monthly)",
        f"Model predictions are the mean of {len(REPEAT_SEEDS_20)} seeds (1-{REPEAT_SEEDS_20[-1]}), "
        "each varying only LightGBM's random_state; REF_heuristic is deterministic. The dashed "
        "horizontal line marks the highest target value observed anywhere in the training period "
        f"(NGN {train_max:,.2f}/litre); the test period's actual mean rises above it in every month "
        "shown, which is the mechanical reason a tree-based model, bounded by the leaf values it "
        "was trained on, cannot fully track the rise and a naive carry-forward heuristic can. "
        "outputs/paper/extrapolation_ceiling.csv carries the full underlying series.",
    )
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
