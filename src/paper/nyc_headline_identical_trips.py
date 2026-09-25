"""NYC headline (V0 -> V4a) on identical trips.

The ladder scores V0 on the UNGATED test window (raw TLC rows, before the
quality gate) and V1 onward on the GATED one, so V0-vs-later-rung differences
mix a model effect with a test-set composition effect. This restricts V0's
saved per-observation errors to exactly the trips in the gated test set and
recomputes the V0 -> V4a improvement on those identical trips, in MAE and, for
V0 and V4a, in MAPE (actual durations come from the same reconstructed pulls;
V0's ungated MAPE is checked against the published value before any gated
MAPE is used).

Matching is on full trip_id (k-th occurrence within a trip_id as tiebreak),
using the same reconstructed ordered pulls as src/paper/nyc_day_cluster_v2.py
(imported, not rewritten); alignment to the saved error files is asserted row
for row. No model is fitted. Writes only outputs/paper/nyc_headline_identical_trips.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src.utils.db import connect
from src.modelling.splits import (
    bucket_for_seed, NYC_TEST_MONTHS, NYC_BUCKET_COUNT, RAW_TRIP_ID_SQL,
)
from src.paper.nyc_day_cluster_v2 import SEEDS, ERRORS_DIR, CONFIGS, PAPER_DIR

#: Published V0 MAPE (model_ladder_nyc.csv, mape_mean / mape_sd), used only to
#: check that the reconstructed durations reproduce V0's own ungated MAPE.
PUBLISHED_V0_MAPE = {"original": (78.0921, 1.7086), "capacity_controlled": (77.5639, 2.417)}


def pull_gated_test(con) -> pd.DataFrame:
    """Same pull as nyc_day_cluster_v2.pull_gated_test, plus the trip duration."""
    months = ", ".join(str(m) for m in NYC_TEST_MONTHS)
    buckets = ", ".join(str(bucket_for_seed(s)) for s in SEEDS)
    key = "('0x' || substr(t.trip_id, 1, 8))::UBIGINT"
    return con.execute(f'''
        SELECT t.trip_id AS trip_id, t.trip_duration_seconds AS duration,
               {key} AS row_key, {key} % {NYC_BUCKET_COUNT} AS bucket
        FROM fact_trip t
        JOIN dim_geography g ON g.geo_key = t.pickup_geo_key
        JOIN dim_date d      ON d.date_key = t.pickup_date_key
        LEFT JOIN fact_weather_daily w ON w.date_key = t.pickup_date_key
        WHERE d.year = 2024 AND d.month_number IN ({months})
          AND {key} % {NYC_BUCKET_COUNT} IN ({buckets})
        ORDER BY t.trip_id
    ''').df()


def pull_raw_test(con) -> pd.DataFrame:
    """Same pull as nyc_day_cluster_v2.pull_raw_test, plus the trip duration."""
    months = ", ".join(str(m) for m in NYC_TEST_MONTHS)
    buckets = ", ".join(str(bucket_for_seed(s)) for s in SEEDS)
    glob = str(settings.RAW_NYC / "*.parquet").replace("\\", "/")
    return con.execute(f'''
        SELECT trip_id, duration, row_key, row_key % {NYC_BUCKET_COUNT} AS bucket
        FROM (
            SELECT trip_id, duration, ('0x' || substr(trip_id, 1, 8))::UBIGINT AS row_key
            FROM (
                SELECT {RAW_TRIP_ID_SQL} AS trip_id,
                       date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) AS duration
                FROM read_parquet('{glob}')
                WHERE EXTRACT(year FROM tpep_pickup_datetime) = 2024
                  AND EXTRACT(month FROM tpep_pickup_datetime) IN ({months})
            )
        )
        WHERE row_key % {NYC_BUCKET_COUNT} IN ({buckets})
        ORDER BY trip_id
    ''').df()


def mape(abs_err: np.ndarray, actual: np.ndarray) -> float:
    """Same definition as src.modelling.ladder.mape: percent, rows with actual == 0 excluded."""
    keep = actual != 0
    return float(np.mean(abs_err[keep] / np.abs(actual[keep])) * 100.0)

REF_NOTE = (
    "scored on the GATED test set: reference_results() scores against data.y_test = "
    "split.test[trip_duration_seconds], the gated fact_trip pull (notebooks/02 cell 11; "
    "identical code in src/paper/ladder_nyc_refit.py, whose re-fit reproduced every "
    "published REF cell exactly). {detail}"
)


def main() -> None:
    con = connect(settings.WAREHOUSE_DB, read_only=True, memory_limit="3GB", threads=4)
    gated = pull_gated_test(con)
    raw = pull_raw_test(con)
    con.close()
    gated["row_key"] = gated["row_key"].astype("int64")
    raw["row_key"] = raw["row_key"].astype("int64")

    G, R, match = {}, {}, {}
    for s in SEEDS:
        b = bucket_for_seed(s)
        G[s] = gated[gated["bucket"] == b].reset_index(drop=True)
        R[s] = raw[raw["bucket"] == b].reset_index(drop=True)
        r = R[s][["trip_id"]].copy()
        r["k"] = r.groupby("trip_id").cumcount()
        r["pos_r"] = np.arange(len(r))
        g = G[s][["trip_id"]].copy()
        g["k"] = g.groupby("trip_id").cumcount()
        g["pos_g"] = np.arange(len(g))
        m = r.merge(g, on=["trip_id", "k"], how="inner")
        assert len(m) == len(g), f"seed {s}: {len(g) - len(m)} gated trips have no raw counterpart"
        match[s] = m.sort_values("pos_g")

    rows = []
    for config, slug in CONFIGS:
        errs = {}
        for rung in ("V0", "V4a"):
            df = pd.read_parquet(ERRORS_DIR / f"{slug}__{rung}.parquet")
            errs[rung] = {}
            for s in SEEDS:
                part = df[df["seed"] == s]
                ref = R[s] if rung == "V0" else G[s]
                assert np.array_equal(part["row_key"].to_numpy(), ref["row_key"].to_numpy())
                errs[rung][s] = part["abs_error"].to_numpy(dtype=np.float64)

        per_seed = []
        for s in SEEDS:
            v0_all = errs["V0"][s]
            v0_gated = v0_all[match[s]["pos_r"].to_numpy()]
            v4a = errs["V4a"][s]
            assert len(v0_gated) == len(v4a)
            y_raw = R[s]["duration"].to_numpy(dtype=np.float64)
            y_gated = G[s]["duration"].to_numpy(dtype=np.float64)
            assert np.array_equal(y_raw[match[s]["pos_r"].to_numpy()], y_gated), \
                f"seed {s}: matched trips disagree on duration"
            u, g_, f = float(v0_all.mean()), float(v0_gated.mean()), float(v4a.mean())
            per_seed.append({
                "configuration": config, "record_type": "seed", "seed": s,
                "n_gated_trips": len(v4a), "n_ungated_trips_v0": len(v0_all),
                "v0_ungated_mae": u, "v0_gated_set_mae": g_,
                "test_set_composition_effect_s": u - g_,
                "v4a_gated_set_mae": f,
                "improvement_v0_ungated_to_v4a_s": u - f,
                "improvement_v0_ungated_to_v4a_pct_of_ungated": 100 * (u - f) / u,
                "improvement_v0_gated_to_v4a_s": g_ - f,
                "improvement_v0_gated_to_v4a_pct_of_gated": 100 * (g_ - f) / g_,
                "share_of_headline_due_to_test_set_composition_pct": 100 * (u - g_) / (u - f),
                "v0_ungated_mape": mape(v0_all, y_raw),
                "v0_gated_set_mape": mape(v0_gated, y_gated),
                "v4a_gated_set_mape": mape(v4a, y_gated),
            })
        frame = pd.DataFrame(per_seed)
        pub_mean, pub_sd = PUBLISHED_V0_MAPE[slug]
        assert abs(frame["v0_ungated_mape"].mean() - pub_mean) < 1e-3 \
            and abs(frame["v0_ungated_mape"].std(ddof=1) - pub_sd) < 1e-3, \
            f"{slug}: reconstructed V0 ungated MAPE does not reproduce the published value"
        print(f"{slug}: reconstructed V0 ungated MAPE {frame['v0_ungated_mape'].mean():.4f} "
              f"+/- {frame['v0_ungated_mape'].std(ddof=1):.4f} matches published {pub_mean} +/- {pub_sd}",
              flush=True)
        numeric = [c for c in frame.columns if c not in ("configuration", "record_type", "seed")]
        mean_row = {"configuration": config, "record_type": "mean_across_seeds", "seed": np.nan}
        sd_row = {"configuration": config, "record_type": "sd_across_seeds", "seed": np.nan}
        for c in numeric:
            mean_row[c] = frame[c].mean()
            sd_row[c] = frame[c].std(ddof=1)
        rows.extend(per_seed + [mean_row, sd_row])

    out = pd.DataFrame(rows)
    for c in out.columns:
        if c.startswith(("v0_", "v4a_", "improvement_", "test_set_", "share_")):
            out[c] = out[c].round(4)

    out["note"] = ""
    for name, detail in (
        ("REF_mean", "The training mean is the mean of the GATED training set."),
        ("REF_heuristic", "Its average speed is total distance over total time of the GATED training set."),
    ):
        out.loc[len(out)] = {"configuration": "reference (no model fitted)",
                             "record_type": "reference_note",
                             "note": f"{name}: " + REF_NOTE.format(detail=detail)}

    out.to_csv(PAPER_DIR / "nyc_headline_identical_trips.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out["record_type"].isin(["mean_across_seeds", "sd_across_seeds"])]
          .drop(columns=["seed", "note"]).T.to_string())
    print("wrote nyc_headline_identical_trips.csv")


if __name__ == "__main__":
    main()
