"""Task C (analysis-only follow-up): Nigerian inference redone with the
dependence structure respected, plus a full trace of the V0 model.

Defects being corrected (see paired_comparisons.csv / diebold_mariano_ng.csv):
  1. The Nigerian paired bootstrap pooled 20 seeds x the SAME 185 test
     observations as 3,700 independent draws (~20x overstated precision).
  2. The Nigerian Diebold-Mariano test used lag 0 and treated 185 state-months
     as independent although errors are correlated across states within a month.

Everything here uses the per-observation errors already saved under
outputs/paper/errors/ng/ (averaged over the 20 seeds to 185 values per rung)
plus read-only warehouse queries. The single exception, approved for this
task: one diagnostic LightGBM fit per configuration (seed 1, V0 features) to
trace the fitted response curve of V0, each checked to reproduce the saved
seed-1 test errors exactly before it is used. No ladder output is regenerated
or overwritten.

Writes (all new): paired_comparisons_ng_v2.csv, diebold_mariano_ng_v2.csv,
nigeria_by_month_and_state.csv, ceiling_training_support.csv,
v0_response_curve.csv, v0_plateau_summary.csv, v0_training_support_by_band.csv,
figures/fig_v0_response_curve.png.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from config import settings
from src.viz import style as vizstyle
from src.modelling.splits import nigeria_time_split, assert_split_is_time_ordered
from src.modelling.ladder import LGBM_PARAMS, VARIANT_ORIGINAL, VARIANT_CAPACITY_CONTROLLED

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
ERRORS_DIR = PAPER_DIR / "errors" / "ng"
FIG_DIR = PAPER_DIR / "figures"
RUNGS = ("V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b")
N_BOOT = 10_000
RNG_SEED = 796
TARGET = "target_price_next_month"
CONFIGS = ((VARIANT_ORIGINAL, "original"), (VARIANT_CAPACITY_CONTROLLED, "capacity_controlled"))

CAPACITY_CONTROLLED_PARAMS = dict(
    num_leaves=7, learning_rate=0.10, min_child_samples=5, reg_alpha=0.0, reg_lambda=0.0,
    n_estimators=12, max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=796, n_jobs=4, verbosity=-1,
)
PARAMS_BY_SLUG = {"original": LGBM_PARAMS, "capacity_controlled": CAPACITY_CONTROLLED_PARAMS}

CURVE_GRID = np.arange(550, 1805, 5)
BAND_EDGES = list(range(500, 1900, 100))


def sig4(x: float) -> float:
    return float(f"{x:.4g}")


def load_matrix(slug: str, rung: str, states, months) -> np.ndarray:
    df = pd.read_parquet(ERRORS_DIR / f"{slug}__{rung}.parquet")
    df["abs_error"] = df["abs_error"].astype("float64")
    counts = df.groupby(["state", "month_key"])["abs_error"].size()
    assert (counts == 20).all(), f"{slug}/{rung}: expected 20 seeds per observation"
    avg = df.groupby(["state", "month_key"])["abs_error"].mean().unstack("month_key")
    avg = avg.reindex(index=states, columns=months)
    assert not avg.isna().any().any()
    return avg.to_numpy()


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def dm_row(d: np.ndarray) -> dict:
    """d: (states, months) matrix of loss differentials (rung minus reference)."""
    n = d.size
    dbar = float(d.mean())
    resid = d - dbar
    out = {"n_observations": n, "mean_loss_diff": dbar}

    var0 = float(resid.ravel().var(ddof=1))
    se0 = np.sqrt(var0 / n)
    t0 = dbar / se0
    out.update(dm_lag0_stat=t0, dm_lag0_df=n - 1, dm_lag0_p=2 * stats.t.sf(abs(t0), n - 1))

    for name, axis, g in (("month", 0, d.shape[1]), ("state", 1, d.shape[0])):
        cluster_sums = resid.sum(axis=axis)
        var = g / (g - 1) * float((cluster_sums ** 2).sum()) / n ** 2
        se = np.sqrt(var)
        t = dbar / se
        out.update({
            f"dm_{name}_cluster_stat": t,
            f"dm_{name}_cluster_df": g - 1,
            f"dm_{name}_cluster_p": 2 * stats.t.sf(abs(t), g - 1),
            f"n_{name}_clusters": g,
        })
    return out


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)
    raw = con.execute('''
        SELECT f.month_key, g.geo_name AS state, f.price_ngn
        FROM fact_fuel_price_monthly f
        JOIN dim_geography g ON g.geo_key = f.geo_key AND g.is_current
        ORDER BY g.geo_name, f.month_key
    ''').df()
    con.close()
    raw = raw.sort_values(["state", "month_key"]).reset_index(drop=True)
    raw[TARGET] = raw.groupby("state")["price_ngn"].shift(-1)
    modeling = raw.dropna(subset=[TARGET]).copy()
    split = nigeria_time_split(modeling)
    assert_split_is_time_ordered(split, time_col="month_key")
    train, test = split.train.copy(), split.test.copy()

    states = sorted(test["state"].unique())
    months = sorted(test["month_key"].unique())
    assert len(states) == 37 and len(months) == 5 and len(test) == 185

    ref = test.assign(ref_abs_error=(test[TARGET] - test["price_ngn"]).abs())
    ref_mat = (ref.set_index(["state", "month_key"])["ref_abs_error"]
               .unstack("month_key").reindex(index=states, columns=months).to_numpy())

    E = {(slug, rung): load_matrix(slug, rung, states, months)
         for _, slug in CONFIGS for rung in RUNGS}

    # ------------------------------------------------------------------ #
    # C.1  paired comparisons, three intervals
    # ------------------------------------------------------------------ #
    v1 = pd.read_csv(PAPER_DIR / "paired_comparisons.csv")
    v1 = v1[v1["task"] == "nigeria"].set_index(["configuration", "comparison"])

    rows = []
    for config, slug in CONFIGS:
        for rung_a, rung_b in zip(RUNGS[:-1], RUNGS[1:]):
            d = E[(slug, rung_b)] - E[(slug, rung_a)]
            mean = float(d.mean())
            flat = d.ravel()
            obs = flat[rng.integers(0, flat.size, (N_BOOT, flat.size))].mean(axis=1)
            st = d[rng.integers(0, d.shape[0], (N_BOOT, d.shape[0]))].mean(axis=(1, 2))
            mo = d[:, rng.integers(0, d.shape[1], (N_BOOT, d.shape[1]))].mean(axis=(0, 2))
            comp = f"{rung_b} vs {rung_a}"
            old = v1.loc[(config, comp)]
            assert abs(mean - old["mean_diff_mae"]) < 1e-3, (comp, mean, old["mean_diff_mae"])
            lo_o, hi_o = percentile_ci(obs)
            lo_s, hi_s = percentile_ci(st)
            lo_m, hi_m = percentile_ci(mo)
            rows.append({
                "task": "nigeria", "configuration": config, "comparison": comp,
                "rung_a": rung_a, "rung_b": rung_b,
                "n_observations": int(d.size), "n_seeds_averaged": 20,
                "mean_diff_mae": round(mean, 4),
                "ci_obs_bootstrap_lower": round(lo_o, 4), "ci_obs_bootstrap_upper": round(hi_o, 4),
                "ci_state_cluster_lower": round(lo_s, 4), "ci_state_cluster_upper": round(hi_s, 4),
                "n_state_clusters": 37,
                "ci_month_cluster_lower": round(lo_m, 4), "ci_month_cluster_upper": round(hi_m, 4),
                "n_month_clusters": 5,
                "month_cluster_note": "UNRELIABLE: only 5 clusters",
                "obs_excludes_zero": bool(lo_o > 0 or hi_o < 0),
                "state_cluster_excludes_zero": bool(lo_s > 0 or hi_s < 0),
                "month_cluster_excludes_zero": bool(lo_m > 0 or hi_m < 0),
                "v1_pooled_ci_lower": old["bootstrap_ci_lower_95"],
                "v1_pooled_ci_upper": old["bootstrap_ci_upper_95"],
                "v1_pooled_excludes_zero": bool(old["bootstrap_ci_lower_95"] > 0 or old["bootstrap_ci_upper_95"] < 0),
                "n_bootstrap_resamples": N_BOOT,
            })
    pc = pd.DataFrame(rows)
    pc.to_csv(PAPER_DIR / "paired_comparisons_ng_v2.csv", index=False)
    print(f"wrote paired_comparisons_ng_v2.csv ({len(pc)} rows)")

    # ------------------------------------------------------------------ #
    # C.2  Diebold-Mariano vs REF_heuristic on seed-averaged errors
    # ------------------------------------------------------------------ #
    dm_rows = []
    for config, slug in CONFIGS:
        for rung in RUNGS:
            d = E[(slug, rung)] - ref_mat
            row = {"task": "nigeria", "configuration": config, "rung": rung}
            row.update(dm_row(d))
            row["rung_mae"] = float(E[(slug, rung)].mean())
            row["ref_mae"] = float(ref_mat.mean())
            dm_rows.append(row)
    dm = pd.DataFrame(dm_rows)
    for c in dm.columns:
        if c.endswith("_stat") or c in ("mean_loss_diff", "rung_mae", "ref_mae"):
            dm[c] = dm[c].round(4)
        elif c.endswith("_p"):
            dm[c] = dm[c].map(sig4)
    for name in ("lag0", "month_cluster", "state_cluster"):
        dm[f"dm_{name}_significant_0.05"] = dm[f"dm_{name}_p"] < 0.05
    dm["loss_diff_definition"] = "seed-averaged |error| of rung minus |error| of REF_heuristic; positive = rung worse"
    dm.to_csv(PAPER_DIR / "diebold_mariano_ng_v2.csv", index=False)
    print(f"wrote diebold_mariano_ng_v2.csv ({len(dm)} rows)")

    # ------------------------------------------------------------------ #
    # C.3  descriptive: by month and by state
    # ------------------------------------------------------------------ #
    desc_rows = []
    for config, slug in CONFIGS:
        for rung in RUNGS:
            m = E[(slug, rung)]
            row = {"configuration": config, "rung": rung,
                   "mae_all_months": round(float(m.mean()), 4),
                   "ref_mae_all_months": round(float(ref_mat.mean()), 4)}
            for j, mk in enumerate(months):
                row[f"rung_mae_fm{mk // 100}"] = round(float(m[:, j].mean()), 4)
                row[f"ref_mae_fm{mk // 100}"] = round(float(ref_mat[:, j].mean()), 4)
            row["n_months_ref_better_of_5"] = int((ref_mat.mean(axis=0) < m.mean(axis=0)).sum())
            row["n_states_ref_better_of_37"] = int((ref_mat.mean(axis=1) < m.mean(axis=1)).sum())
            desc_rows.append(row)
    pd.DataFrame(desc_rows).to_csv(PAPER_DIR / "nigeria_by_month_and_state.csv", index=False)
    print("wrote nigeria_by_month_and_state.csv "
          "(fm = feature month; the target is the following month's price)")

    # ------------------------------------------------------------------ #
    # C.4a  training rows whose target lies above each prediction ceiling
    # ------------------------------------------------------------------ #
    ceilings = pd.read_csv(PAPER_DIR / "undershoot_mechanism.csv", comment="#")
    ceilings = ceilings.set_index("configuration")["pooled_pred_max"].to_dict()

    def to_month(k):
        return pd.to_datetime(k.astype(str), format="%Y%m%d")

    train = train.assign(
        feature_month=to_month(train["month_key"]).dt.strftime("%Y-%m"),
        target_month=(to_month(train["month_key"]) + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m"),
    )
    cols = ["record_type", "configuration", "ceiling", "n_training_rows_total",
            "n_training_rows_above_ceiling", "pct_training_rows_above_ceiling",
            "n_distinct_states", "n_distinct_feature_months", "feature_months_involved",
            "n_test_rows_above_ceiling", "state", "feature_month", "target_month",
            "price_ngn", "target_price_next_month"]
    ceil_rows = []
    for config, slug in CONFIGS:
        ceiling = float(ceilings[config])
        above = train[train[TARGET] > ceiling].sort_values(TARGET, ascending=False)
        ceil_rows.append({
            "record_type": "summary", "configuration": config, "ceiling": ceiling,
            "n_training_rows_total": len(train),
            "n_training_rows_above_ceiling": len(above),
            "pct_training_rows_above_ceiling": round(100 * len(above) / len(train), 4),
            "n_distinct_states": above["state"].nunique(),
            "n_distinct_feature_months": above["feature_month"].nunique(),
            "feature_months_involved": ";".join(sorted(above["feature_month"].unique())),
            "n_test_rows_above_ceiling": int((test[TARGET] > ceiling).sum()),
        })
        for _, r in above.iterrows():
            ceil_rows.append({
                "record_type": "row", "configuration": config, "ceiling": ceiling,
                "state": r["state"], "feature_month": r["feature_month"],
                "target_month": r["target_month"], "price_ngn": r["price_ngn"],
                "target_price_next_month": r[TARGET],
            })
    pd.DataFrame(ceil_rows, columns=cols).to_csv(PAPER_DIR / "ceiling_training_support.csv", index=False)
    print("wrote ceiling_training_support.csv")

    # ------------------------------------------------------------------ #
    # C.4b  diagnostic fits (seed 1, V0) -> response curve
    # ------------------------------------------------------------------ #
    X_train, X_test = train[["price_ngn"]], test[["price_ngn"]]
    curves, plateau_rows, models = [], [], {}
    for config, slug in CONFIGS:
        params = dict(PARAMS_BY_SLUG[slug])
        params["random_state"] = 1
        model = LGBMRegressor(**params)
        model.fit(X_train, train[TARGET])
        preds = model.predict(X_test)
        mine = pd.DataFrame({"state": test["state"].to_numpy(), "month_key": test["month_key"].to_numpy(),
                             "mine": np.abs(preds - test[TARGET].to_numpy()).astype(np.float32)})
        saved = pd.read_parquet(ERRORS_DIR / f"{slug}__V0.parquet")
        saved = saved[saved["seed"] == 1][["state", "month_key", "abs_error"]]
        joined = mine.merge(saved, on=["state", "month_key"], how="inner")
        assert len(joined) == 185
        max_diff = float(np.abs(joined["mine"].to_numpy() - joined["abs_error"].to_numpy()).max())
        assert np.array_equal(joined["mine"].to_numpy(), joined["abs_error"].to_numpy()), \
            f"{slug}: diagnostic fit does NOT reproduce saved seed-1 errors (max abs diff {max_diff})"
        print(f"{slug}: diagnostic seed-1 fit reproduces saved test errors exactly (max abs diff {max_diff})")
        models[slug] = model

        grid_pred = model.predict(pd.DataFrame({"price_ngn": CURVE_GRID.astype(float)}))
        for p, y in zip(CURVE_GRID, grid_pred):
            curves.append({"configuration": config, "price_ngn": int(p),
                           "predicted_next_price": float(y), "random_walk_price": float(p)})

        thresholds = model.booster_.trees_to_dataframe()["threshold"].dropna()
        last_threshold = float(thresholds.max())
        level = float(model.predict(pd.DataFrame({"price_ngn": [last_threshold + 1.0]}))[0])
        final = grid_pred[-1]
        const_from = CURVE_GRID[[bool(np.all(np.abs(grid_pred[i:] - final) < 1e-9))
                                 for i in range(len(CURVE_GRID))]].min()
        first_max = CURVE_GRID[int(np.argmax(grid_pred))]
        diffs = np.diff(grid_pred)
        plateau_rows.append({
            "configuration": config, "seed": 1, "n_estimators": params["n_estimators"],
            "last_split_threshold_price_ngn": round(last_threshold, 4),
            "plateau_level_above_last_threshold": round(level, 4),
            "plateau_start_price_grid_5ngn": int(const_from),
            "max_prediction_on_grid": round(float(grid_pred.max()), 4),
            "first_grid_price_attaining_max": int(first_max),
            "grid_curve_monotone_nondecreasing": bool((diffs >= -1e-9).all()),
            "train_price_ngn_max": float(train["price_ngn"].max()),
            "train_price_ngn_min": float(train["price_ngn"].min()),
            "train_target_max": float(train[TARGET].max()),
            "n_test_rows_price_above_plateau_start": int((test["price_ngn"] > last_threshold).sum()),
            "n_test_rows_price_above_train_price_max": int((test["price_ngn"] > train["price_ngn"].max()).sum()),
            "n_test_actuals_above_plateau_level": int((test[TARGET] > level).sum()),
            "max_test_prediction_seed1": round(float(preds.max()), 4),
            "verified_against_saved_seed1_errors": True,
        })
    pd.DataFrame(curves).to_csv(PAPER_DIR / "v0_response_curve.csv", index=False)
    plateau = pd.DataFrame(plateau_rows)
    plateau.to_csv(PAPER_DIR / "v0_plateau_summary.csv", index=False)
    print(plateau.T.to_string())

    # ------------------------------------------------------------------ #
    # C.4c  training support by price band
    # ------------------------------------------------------------------ #
    def band_table(frame, lo, hi):
        sel = frame[(frame["price_ngn"] >= lo) & (frame["price_ngn"] < hi)]
        return len(sel), (float(sel[TARGET].mean()) if len(sel) else np.nan)

    band_rows = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        n_tr, mean_tr = band_table(train, lo, hi)
        n_te, mean_te = band_table(test, lo, hi)
        band_rows.append({"band": f"[{lo}, {hi})", "band_lo": lo, "band_hi": hi,
                          "n_train_rows": n_tr, "mean_next_target_train": round(mean_tr, 4) if n_tr else np.nan,
                          "n_test_rows": n_te, "mean_next_actual_test": round(mean_te, 4) if n_te else np.nan})
    lo_edge, hi_edge = BAND_EDGES[0], BAND_EDGES[-1]
    for label, mask_tr, mask_te in (
        (f"below {lo_edge}", train["price_ngn"] < lo_edge, test["price_ngn"] < lo_edge),
        (f"{hi_edge} and above", train["price_ngn"] >= hi_edge, test["price_ngn"] >= hi_edge),
    ):
        if mask_tr.any() or mask_te.any():
            band_rows.append({"band": label, "n_train_rows": int(mask_tr.sum()),
                              "mean_next_target_train": round(float(train.loc[mask_tr, TARGET].mean()), 4) if mask_tr.any() else np.nan,
                              "n_test_rows": int(mask_te.sum()),
                              "mean_next_actual_test": round(float(test.loc[mask_te, TARGET].mean()), 4) if mask_te.any() else np.nan})
    bands = pd.DataFrame(band_rows)
    assert bands["n_train_rows"].sum() == len(train) and bands["n_test_rows"].sum() == len(test)
    bands.to_csv(PAPER_DIR / "v0_training_support_by_band.csv", index=False)
    print("wrote v0_training_support_by_band.csv")

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    vizstyle.apply_style()
    import matplotlib.pyplot as plt

    curve_df = pd.DataFrame(curves)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), sharex=True, sharey=True)
    titles = {"original": "Original configuration (300 trees)",
              "capacity_controlled": "Capacity-controlled configuration (12 trees)"}
    for ax, (config, slug), colour in zip(axes, CONFIGS, (vizstyle.PALETTE[1], vizstyle.PALETTE[2])):
        c = curve_df[curve_df["configuration"] == config]
        ax.set_xlim(550, 1800)
        ax.set_ylim(550, 1800)
        ax.vlines(train["price_ngn"], 0, 0.035, transform=ax.get_xaxis_transform(),
                  color=vizstyle.MUTED_COLOUR, alpha=0.35, linewidth=0.6,
                  label="training price_ngn values (rug)")
        ax.plot(c["price_ngn"], c["random_walk_price"], color=vizstyle.PALETTE[3], linestyle=":",
                linewidth=1.6, label="random walk (predicted = current price)")
        ax.plot(c["price_ngn"], c["predicted_next_price"], color=colour, linewidth=2.0,
                drawstyle="steps-post", label="fitted V0 model, seed 1")
        ax.scatter(test["price_ngn"], test[TARGET], s=16, color=vizstyle.PALETTE[0], alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=4,
                   label="test observations (actual next price)")
        ax.axvline(train["price_ngn"].max(), color=vizstyle.MUTED_COLOUR, linestyle=(0, (4, 2)),
                   linewidth=1.1, label="highest training price_ngn")
        ax.set_title(titles[slug], fontsize=11, fontweight="bold")
        ax.set_xlabel("Current-month price_ngn (NGN/litre)", fontsize=10)
    axes[0].set_ylabel("Next-month price (NGN/litre)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    p_o = plateau[plateau["configuration"] == VARIANT_ORIGINAL].iloc[0]
    p_c = plateau[plateau["configuration"] == VARIANT_CAPACITY_CONTROLLED].iloc[0]
    caption = (
        "Each curve is one model (V0, seed 1) whose test errors were verified to reproduce the saved "
        "seed-1 errors exactly. The dotted line is the random walk. The rug marks the training values "
        "of price_ngn; the dashed vertical line is the highest of them "
        f"(NGN {p_o['train_price_ngn_max']:,.2f}). Points are the {len(test)} test observations. "
        f"Original curve: exactly flat above price_ngn {p_o['last_split_threshold_price_ngn']:,.2f}, "
        f"at NGN {p_o['plateau_level_above_last_threshold']:,.2f}; {int(p_o['n_test_actuals_above_plateau_level'])} "
        f"of {len(test)} actual test values exceed that level. Capacity-controlled curve: flat above "
        f"{p_c['last_split_threshold_price_ngn']:,.2f}, at NGN {p_c['plateau_level_above_last_threshold']:,.2f}; "
        f"{int(p_c['n_test_actuals_above_plateau_level'])} of {len(test)} exceed it. "
        f"{int(p_o['n_test_rows_price_above_train_price_max'])} test observations have a current price "
        "above the highest training price."
    )
    vizstyle.finish(fig, FIG_DIR / "fig_v0_response_curve.png",
                    "Source: this platform's own gold layer (fact_fuel_price_monthly)", caption)
    print("wrote figures/fig_v0_response_curve.png")


if __name__ == "__main__":
    main()
