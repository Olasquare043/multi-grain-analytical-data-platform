"""Task 3 (paper repair pass): final export of outputs/paper/ladder_nyc.csv
and outputs/paper/ladder_ng.csv in the exact schema requested (task,
configuration, rung, description, feature_set, n_features, MAE_mean, MAE_sd,
MAE_min, MAE_max, MAPE_mean, MAPE_sd, n_seeds), and the refit cross-check
(outputs/paper/refit_verification.csv) the user's re-fit conditions required.

Nigeria: read from outputs/paper/ladder_nigeria_20seed_raw.csv (this pass's
own fresh 20-seed re-run, src/paper/ladder_nigeria_rerun.py). Already carries
feature_set/n_features.

NYC: read from the EXISTING, untouched outputs/tables/model_ladder_nyc.csv
(5 seeds, already has MAPE, so Task 3's own instructions say not to re-run
it) and add feature_set/n_features, which that file does not carry, from the
literal FEATURES_V* lists in notebooks/02_nyc_trip_duration.ipynb.

Pure pandas; no docker/duckdb/lightgbm needed.
"""
from __future__ import annotations

import pandas as pd

from config import settings

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
TABLES_DIR = settings.OUTPUTS_DIR / "tables"

REQUIRED_COLS = ["task", "configuration", "rung", "description", "feature_set",
                  "n_features", "MAE_mean", "MAE_sd", "MAE_min", "MAE_max",
                  "MAPE_mean", "MAPE_sd", "n_seeds"]

# Verbatim from notebooks/02_nyc_trip_duration.ipynb (this pass reads it, does
# not modify it).
NYC_FEATURES = {
    "V0": (["pickup_hour", "pickup_zone_raw", "trip_distance_miles", "passenger_count"], 4),
    "V1": (["pickup_hour", "pickup_geo_key", "trip_distance_miles", "passenger_count"], 4),
    "V2": (["pickup_hour", "trip_distance_miles", "passenger_count", "borough",
             "service_zone", "day_of_week", "is_weekend"], 7),
    "V3a": (["pickup_hour", "trip_distance_miles", "passenger_count", "borough",
              "service_zone", "day_of_week", "is_weekend", "zone_hour_avg_duration_leaky"], 8),
    "V3b": (["pickup_hour", "trip_distance_miles", "passenger_count", "borough",
              "service_zone", "day_of_week", "is_weekend", "zone_hour_avg_duration_pit"], 8),
    "V4a": (["pickup_hour", "trip_distance_miles", "passenger_count", "borough",
              "service_zone", "day_of_week", "is_weekend", "zone_hour_avg_duration_pit",
              "pickup_geo_key (one-hot, up to 265 categories)"], "8 + up to 265 one-hot columns (varies by seed's realised zone set)"),
    "V4b": (["pickup_hour", "trip_distance_miles", "passenger_count", "borough",
              "service_zone", "day_of_week", "is_weekend", "zone_hour_avg_duration_pit",
              "zone_target_enc"], 9),
    "V5": (["pickup_hour", "trip_distance_miles", "passenger_count", "borough",
             "service_zone", "day_of_week", "is_weekend", "zone_hour_avg_duration_pit",
             "zone_target_enc", "precipitation_mm", "temp_max_c"], 11),
}


def export_nigeria() -> pd.DataFrame:
    src = pd.read_csv(PAPER_DIR / "ladder_nigeria_20seed_raw.csv")
    out = pd.DataFrame({
        "task": "nigeria",
        "configuration": src["variant"],
        "rung": src["rung"],
        "description": src["description"],
        "feature_set": src["feature_set"],
        "n_features": src["n_features"],
        "MAE_mean": src["mae_mean"],
        "MAE_sd": src["mae_sd"],
        "MAE_min": src["mae_min"],
        "MAE_max": src["mae_max"],
        "MAPE_mean": src["mape_mean"],
        "MAPE_sd": src["mape_sd"],
        "n_seeds": src["n_repeats"],
    })
    out.to_csv(PAPER_DIR / "ladder_ng.csv", index=False)
    return out


def export_nyc() -> pd.DataFrame:
    src = pd.read_csv(TABLES_DIR / "model_ladder_nyc.csv")  # existing, untouched
    feature_sets, n_features = [], []
    for rung in src["rung"]:
        if rung in NYC_FEATURES:
            fs, nf = NYC_FEATURES[rung]
            feature_sets.append(",".join(fs))
            n_features.append(nf)
        else:  # REF_mean / REF_heuristic: no features, fits no model
            feature_sets.append("")
            n_features.append("")
    out = pd.DataFrame({
        "task": "nyc",
        "configuration": src["variant"],
        "rung": src["rung"],
        "description": src["description"],
        "feature_set": feature_sets,
        "n_features": n_features,
        "MAE_mean": src["mae_mean"],
        "MAE_sd": src["mae_sd"],
        "MAE_min": src["mae_min"],
        "MAE_max": src["mae_max"],
        "MAPE_mean": src["mape_mean"],
        "MAPE_sd": src["mape_sd"],
        "n_seeds": src["n_repeats"],
    })
    out.to_csv(PAPER_DIR / "ladder_nyc.csv", index=False)
    return out


def refit_verification() -> pd.DataFrame:
    original = pd.read_csv(TABLES_DIR / "model_ladder_nyc.csv")
    refit_path = PAPER_DIR / "ladder_nyc_refit.csv"
    if not refit_path.exists():
        print("NOTE: outputs/paper/ladder_nyc_refit.csv not found yet -- "
              "run src/paper/ladder_nyc_refit.py first, then re-run this script.")
        return pd.DataFrame()
    refit = pd.read_csv(refit_path)

    merged = original.merge(
        refit, on=["variant", "rung"], suffixes=("_original", "_refit"), how="outer",
        indicator=True,
    )
    merged["abs_diff_mae_mean"] = (merged["mae_mean_original"] - merged["mae_mean_refit"]).abs()
    merged["exact_match"] = merged["abs_diff_mae_mean"] < 1e-9
    out = merged[["variant", "rung", "mae_mean_original", "mae_mean_refit",
                  "abs_diff_mae_mean", "exact_match", "_merge"]].rename(
        columns={"_merge": "row_presence"}
    )
    out.to_csv(PAPER_DIR / "refit_verification.csv", index=False)

    n_total = len(out)
    n_exact = int(out["exact_match"].sum())
    if n_exact == n_total and n_total > 0:
        print(f"REFIT VERIFICATION: all {n_total} rung/configuration cells match "
              f"EXACTLY (abs diff < 1e-9) between the original published "
              f"model_ladder_nyc.csv and this pass's independent re-fit -- a "
              f"reproducibility result in its own right.")
    else:
        mismatches = out[~out["exact_match"]]
        print(f"REFIT VERIFICATION: {n_exact} of {n_total} cells match exactly. "
              f"{len(mismatches)} MISMATCH(ES) -- reported loudly, not adopted quietly:")
        print(mismatches.to_string(index=False))
    return out


def main() -> None:
    ng = export_nigeria()
    nyc = export_nyc()
    print(f"wrote outputs/paper/ladder_ng.csv ({len(ng)} rows)")
    print(f"wrote outputs/paper/ladder_nyc.csv ({len(nyc)} rows)")
    refit_verification()


if __name__ == "__main__":
    main()
