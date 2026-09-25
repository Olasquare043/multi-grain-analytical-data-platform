"""Task D (analysis-only follow-up): NYC paired comparisons with a day-level
cluster bootstrap.

The earlier NYC bootstrap (paired_comparisons.csv) treated trips as
independent, although trips on the same day share weather and traffic. This
script resamples the 61 test days (2024-11-01 .. 2024-12-31) with all of each
day's trips, 10,000 times, over ALL paired observations (no subsampling).

The saved per-observation errors carry `row_key` (an 8-hex-digit MD5 prefix of
trip_id) but neither trip_id nor the pickup date, so the identical ordered test
pulls are reconstructed here -- same SQL joins, same content-addressed bucket
rule, ORDER BY trip_id, exactly as src/paper/ladder_nyc_refit.py did -- with
NO model fitted. Alignment to the saved errors is asserted row for row
(length and row_key sequence) for all 16 saved files before any date is used.

row_key is NOT unique (8 hex digits, ~350 collisions per seed), so pairs are
built positionally for the gated rungs V1..V5 (identical row order by
construction) and on full trip_id (with a k-th-occurrence tiebreak) for the
one pair that crosses the raw/gated boundary, V0 vs V1. The earlier merge on
row_key was many-to-many for the duplicated keys; the size of that effect is
reported in the output.

Read-only against the warehouse and raw Parquet. Writes only
outputs/paper/paired_comparisons_nyc_v2.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import settings
from src.utils.db import connect
from src.modelling.splits import (
    NYC_TEST_MONTHS, NYC_BUCKET_COUNT, bucket_for_seed, RAW_TRIP_ID_SQL,
)

PAPER_DIR = settings.OUTPUTS_DIR / "paper"
ERRORS_DIR = PAPER_DIR / "errors" / "nyc"
SEEDS = (1, 2, 3, 4, 5)
RUNGS = ["V0", "V1", "V2", "V3a", "V3b", "V4a", "V4b", "V5"]
CONFIGS = (("original (untuned, 300 trees)", "original"),
           ("capacity-controlled (CV-selected on V0)", "capacity_controlled"))
N_BOOT = 10_000
RNG_SEED = 796


def pull_gated_test(con) -> pd.DataFrame:
    months = ", ".join(str(m) for m in NYC_TEST_MONTHS)
    buckets = ", ".join(str(bucket_for_seed(s)) for s in SEEDS)
    key = "('0x' || substr(t.trip_id, 1, 8))::UBIGINT"
    return con.execute(f'''
        SELECT t.trip_id AS trip_id, t.pickup_date_key AS date_key,
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
    months = ", ".join(str(m) for m in NYC_TEST_MONTHS)
    buckets = ", ".join(str(bucket_for_seed(s)) for s in SEEDS)
    glob = str(settings.RAW_NYC / "*.parquet").replace("\\", "/")
    return con.execute(f'''
        SELECT trip_id, date_key, row_key, row_key % {NYC_BUCKET_COUNT} AS bucket
        FROM (
            SELECT trip_id, date_key, ('0x' || substr(trip_id, 1, 8))::UBIGINT AS row_key
            FROM (
                SELECT {RAW_TRIP_ID_SQL} AS trip_id,
                       CAST(strftime(tpep_pickup_datetime, '%Y%m%d') AS INTEGER) AS date_key
                FROM read_parquet('{glob}')
                WHERE EXTRACT(year FROM tpep_pickup_datetime) = 2024
                  AND EXTRACT(month FROM tpep_pickup_datetime) IN ({months})
            )
        )
        WHERE row_key % {NYC_BUCKET_COUNT} IN ({buckets})
        ORDER BY trip_id
    ''').df()


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(RNG_SEED)

    con = connect(settings.WAREHOUSE_DB, read_only=True, memory_limit="3GB", threads=4)
    gated = pull_gated_test(con)
    print(f"gated test pull: {len(gated):,} rows in {time.perf_counter() - started:.0f}s", flush=True)
    raw = pull_raw_test(con)
    con.close()
    print(f"raw test pull:   {len(raw):,} rows in {time.perf_counter() - started:.0f}s", flush=True)

    gated["row_key"] = gated["row_key"].astype("int64")
    raw["row_key"] = raw["row_key"].astype("int64")
    days = np.array(sorted(gated["date_key"].unique()))
    assert len(days) == 61, f"expected 61 test days, found {len(days)}"
    day_of = {int(d): i for i, d in enumerate(days)}

    G, R = {}, {}
    for s in SEEDS:
        b = bucket_for_seed(s)
        G[s] = gated[gated["bucket"] == b].reset_index(drop=True)
        R[s] = raw[raw["bucket"] == b].reset_index(drop=True)
        G[s]["day"] = G[s]["date_key"].map(day_of).to_numpy()
        print(f"seed {s} bucket {b}: gated {len(G[s]):,} (dup row_key {int(G[s]['row_key'].duplicated().sum())}), "
              f"raw {len(R[s]):,}", flush=True)

    # ---- load saved errors and assert row-for-row alignment ---------------
    err = {}
    for _, slug in CONFIGS:
        for rung in RUNGS:
            df = pd.read_parquet(ERRORS_DIR / f"{slug}__{rung}.parquet")
            per_seed = {}
            for s in SEEDS:
                part = df[df["seed"] == s]
                ref = R[s] if rung == "V0" else G[s]
                assert len(part) == len(ref), (slug, rung, s, len(part), len(ref))
                assert np.array_equal(part["row_key"].to_numpy(), ref["row_key"].to_numpy()), \
                    f"row_key sequence mismatch {slug}/{rung}/seed {s}"
                per_seed[s] = part["abs_error"].to_numpy(dtype=np.float64)
            err[(slug, rung)] = per_seed
    print("alignment verified: all 16 saved error files match the reconstructed pulls row for row", flush=True)

    # ---- V0 vs V1 matching on full trip_id ---------------------------------
    match = {}
    for s in SEEDS:
        r = R[s][["trip_id", "date_key"]].copy()
        r["k"] = r.groupby("trip_id").cumcount()
        r["pos_r"] = np.arange(len(r))
        g = G[s][["trip_id", "date_key", "day"]].copy()
        g["k"] = g.groupby("trip_id").cumcount()
        g["pos_g"] = np.arange(len(g))
        m = r.merge(g, on=["trip_id", "k"], how="inner", suffixes=("_r", "_g"))
        assert (m["date_key_r"] == m["date_key_g"]).all()
        match[s] = m
        print(f"seed {s}: V0/V1 matched {len(m):,} of {len(r):,} raw and {len(g):,} gated rows "
              f"({len(g) - len(m):,} gated rows unmatched)", flush=True)

    # ---- comparisons -------------------------------------------------------
    old = pd.read_csv(PAPER_DIR / "paired_comparisons.csv")
    old = old[old["task"] == "nyc"].set_index(["configuration", "comparison"])

    rows = []
    for config, slug in CONFIGS:
        for rung_a, rung_b in zip(RUNGS[:-1], RUNGS[1:]):
            S = np.zeros(61)
            N = np.zeros(61)
            Q = np.zeros(61)
            seed_means = []
            for s in SEEDS:
                if (rung_a, rung_b) == ("V0", "V1"):
                    m = match[s]
                    d = err[(slug, "V1")][s][m["pos_g"].to_numpy()] - err[(slug, "V0")][s][m["pos_r"].to_numpy()]
                    day = m["day"].to_numpy()
                else:
                    d = err[(slug, rung_b)][s] - err[(slug, rung_a)][s]
                    day = G[s]["day"].to_numpy()
                S += np.bincount(day, weights=d, minlength=61)
                N += np.bincount(day, minlength=61)
                Q += np.bincount(day, weights=d * d, minlength=61)
                seed_means.append(float(d.mean()))

            n = float(N.sum())
            mean = float(S.sum() / n)
            idx = rng.integers(0, 61, (N_BOOT, 61))
            boot = S[idx].sum(axis=1) / N[idx].sum(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            var_iid = (Q.sum() - n * mean ** 2) / (n - 1)
            se_iid = float(np.sqrt(var_iid / n))
            se_cluster = float(boot.std(ddof=1))

            comp = f"{rung_b} vs {rung_a}"
            o = old.loc[(config, comp)]
            old_excl = bool(o["bootstrap_ci_lower_95"] > 0 or o["bootstrap_ci_upper_95"] < 0)
            new_excl = bool(lo > 0 or hi < 0)
            rows.append({
                "task": "nyc", "configuration": config, "comparison": comp,
                "rung_a": rung_a, "rung_b": rung_b,
                "n_pairs": int(n), "n_test_days": 61, "n_seeds": len(SEEDS),
                "mean_diff_pooled": round(mean, 4),
                "mean_diff_seed_level": round(float(np.mean(seed_means)), 4),
                "sd_diff_seed_level": round(float(np.std(seed_means, ddof=1)), 4),
                "day_cluster_ci_lower": round(float(lo), 4),
                "day_cluster_ci_upper": round(float(hi), 4),
                "day_cluster_excludes_zero": new_excl,
                "n_bootstrap_resamples": N_BOOT,
                "iid_full_data_ci_lower": round(mean - 1.96 * se_iid, 4),
                "iid_full_data_ci_upper": round(mean + 1.96 * se_iid, 4),
                "se_ratio_day_cluster_vs_iid": round(se_cluster / se_iid, 3),
                "v1_ci_lower": o["bootstrap_ci_lower_95"],
                "v1_ci_upper": o["bootstrap_ci_upper_95"],
                "v1_excludes_zero": old_excl,
                "v1_mean_diff_mae": o["mean_diff_mae"],
                "v1_n_observations_pooled": int(o["n_observations_pooled"]),
                "v1_n_observations_bootstrapped": int(o["n_observations_bootstrapped"]),
                "v1_pool_was_subsampled": bool(o["pool_was_subsampled"]),
                "conclusion_changed_vs_v1": old_excl != new_excl,
            })

    out = pd.DataFrame(rows)
    out.to_csv(PAPER_DIR / "paired_comparisons_nyc_v2.csv", index=False)
    print(f"\nwrote paired_comparisons_nyc_v2.csv ({len(out)} rows) in {time.perf_counter() - started:.0f}s")
    print(out[["configuration", "comparison", "n_pairs", "v1_n_observations_pooled", "mean_diff_pooled",
               "v1_mean_diff_mae", "day_cluster_ci_lower", "day_cluster_ci_upper",
               "v1_ci_lower", "v1_ci_upper", "se_ratio_day_cluster_vs_iid",
               "conclusion_changed_vs_v1"]].to_string(index=False))


if __name__ == "__main__":
    main()
