"""Task 4 (paper repair pass): Diebold-Mariano test of every Nigerian ladder
rung against the random-walk reference predictor (REF_heuristic), on
absolute-error loss. This is the formal test behind the paper's central null
result (no rung beats the random walk) and was previously asserted in prose
only.

REF_heuristic is deterministic (its prediction does not depend on the
model's random_state), while every fitted rung's predictions vary across the
20 seeds re-run for this pass (src/paper/ladder_nigeria_rerun.py). A single
DM test needs one forecast series, not twenty, so this script runs one DM
test PER SEED (using that seed's per-observation absolute errors, already
persisted under outputs/paper/errors/ng/) and reports both the per-seed
detail and an across-seed summary (mean/sd of the DM statistic, mean
p-value, share of seeds significant at 0.05) -- rather than averaging 20
sets of predictions first, which would silently change the loss series being
tested.

DM statistic: for observations i = 1..n (n = 185 state-month test rows,
paired by (state, month_key)), d_i = |e_rung,i| - |e_ref,i|.
DM = mean(d) / sqrt(var(d, ddof=1) / n), two-sided p-value from a
t-distribution with n-1 degrees of freedom. No Newey-West / HAC lag
adjustment is applied beyond the plain sample variance: these are one-step-
ahead (h=1) forecasts, for which the standard DM prescription is lag = h-1 = 0.
Cross-sectional (across-state, same-month) correlation in d_i is not modelled;
this is stated as a limitation, not corrected for.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from config import settings
from src.modelling.ladder import VARIANT_ORIGINAL, VARIANT_CAPACITY_CONTROLLED

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
ERRORS_DIR = PAPER_DIR / "errors" / "ng"
REPEAT_SEEDS_20 = tuple(range(1, 21))
RUNGS = ("V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b")


def main() -> None:
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

    from src.modelling.splits import nigeria_time_split
    split = nigeria_time_split(modeling)
    ref = split.test[["state", "month_key", "price_ngn", "target_price_next_month"]].copy()
    ref["ref_abs_error"] = (ref["target_price_next_month"] - ref["price_ngn"]).abs()
    ref_key = ref[["state", "month_key", "ref_abs_error"]]

    per_seed_rows = []
    for variant, variant_slug in ((VARIANT_ORIGINAL, "original"),
                                   (VARIANT_CAPACITY_CONTROLLED, "capacity_controlled")):
        for rung in RUNGS:
            path = ERRORS_DIR / f"{variant_slug}__{rung}.parquet"
            errs = pd.read_parquet(path)  # columns: state, month_key, seed, abs_error
            merged = errs.merge(ref_key, on=["state", "month_key"], how="inner")
            assert len(merged) == len(errs), (
                f"row mismatch joining {rung}/{variant_slug} errors to REF_heuristic errors: "
                f"{len(errs)} vs {len(merged)} after join"
            )
            for seed in REPEAT_SEEDS_20:
                s = merged[merged["seed"] == seed]
                d = s["abs_error"].to_numpy(dtype=float) - s["ref_abs_error"].to_numpy(dtype=float)
                n = len(d)
                mean_d = float(d.mean())
                var_d = float(d.var(ddof=1))
                if var_d > 0:
                    dm_stat = mean_d / np.sqrt(var_d / n)
                    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
                else:
                    dm_stat, p_value = np.nan, np.nan
                per_seed_rows.append({
                    "task": "nigeria", "configuration": variant, "rung": rung, "seed": seed,
                    "n": n, "mean_loss_diff": round(mean_d, 4),
                    "dm_statistic": round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
                    "p_value": round(p_value, 6) if not np.isnan(p_value) else np.nan,
                    "rung_beats_ref": bool(mean_d < 0),
                    "significant_at_0.05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                })

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed_path = PAPER_DIR / "diebold_mariano_ng_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows = []
    for (config, rung), g in per_seed.groupby(["configuration", "rung"], sort=False):
        summary_rows.append({
            "task": "nigeria", "configuration": config, "rung": rung,
            "n_per_seed": int(g["n"].iloc[0]),
            "n_seeds": len(g),
            "mean_dm_statistic": round(float(g["dm_statistic"].mean()), 4),
            "sd_dm_statistic": round(float(g["dm_statistic"].std(ddof=1)), 4),
            "mean_p_value": round(float(g["p_value"].mean()), 6),
            "n_seeds_rung_beats_ref_heuristic": int(g["rung_beats_ref"].sum()),
            "n_seeds_significant_at_0.05": int(g["significant_at_0.05"].sum()),
            "method": "per-seed DM test (h=1, lag=0 / no HAC adjustment), absolute-error loss, "
                      "REF_heuristic (deterministic random walk) as the reference forecast; "
                      "summary is the mean/sd of 20 independent per-seed DM statistics, not a "
                      "single test on averaged predictions.",
        })
    summary = pd.DataFrame(summary_rows).sort_values(["configuration", "rung"])
    out_path = PAPER_DIR / "diebold_mariano_ng.csv"
    summary.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(summary)} rows) and {per_seed_path} ({len(per_seed)} rows)")
    print(summary[["configuration", "rung", "mean_dm_statistic", "mean_p_value",
                    "n_seeds_rung_beats_ref_heuristic", "n_seeds_significant_at_0.05"]].to_string(index=False))


if __name__ == "__main__":
    main()
