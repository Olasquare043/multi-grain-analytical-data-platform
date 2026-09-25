"""Task 1 (paper repair pass): recompute every benchmark ratio quoted in the
paper from the FULL-PRECISION values already stored in
outputs/benchmarks/benchmark_results.csv and outputs/tables/
A10_grain_comparison.csv, and compare against what the paper reported.

No benchmark is re-run: every stored value already carries far more
precision (6 decimal places for timings, exact integer byte counts) than the
2-3 significant figures the paper printed, which is the root cause of the
reviewer's complaint -- 27.23s / 0.21s (the paper's own printed precision)
recomputes to 129.7x, not the 131x the paper states, purely because printing
"0.21" throws away the third significant figure of 0.207308. Reporting every
value to 4 significant figures, as this script does, is the fix.

Run with plain host Python (pandas only) -- deliberately NOT inside the
project's Docker container, so it does not compete for the same 4 CPU cores
as the concurrent NYC ladder refit (src/paper/ladder_nyc_refit.py) this pass
also runs. Nothing here touches modelling code, duckdb, or any pinned
container-only dependency.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_CSV = ROOT / "outputs" / "benchmarks" / "benchmark_results.csv"
A10_CSV = ROOT / "outputs" / "tables" / "A10_grain_comparison.csv"
OUT_PATH = ROOT / "outputs" / "paper" / "benchmarks_precise.csv"


def sig4(x: float) -> float:
    """Round to 4 significant figures."""
    if x == 0:
        return 0.0
    from math import floor, log10
    d = 4 - int(floor(log10(abs(x)))) - 1
    return round(x, d)


def main() -> None:
    bench = pd.read_csv(BENCH_CSV)
    a10 = pd.read_csv(A10_CSV)

    def row(bench_df, variant):
        r = bench_df[bench_df["variant"] == variant]
        assert len(r) == 1, f"expected exactly one row for variant={variant!r}, got {len(r)}"
        return r.iloc[0]

    b1 = bench[bench["benchmark"] == "B1_format_comparison"]
    parquet = row(b1, "parquet")
    csv = row(b1, "csv")

    b2 = bench[bench["benchmark"] == "B2_partition_pruning"]
    pruned = row(b2, "partitioned_pruned")
    consolidated = row(b2, "consolidated_single_file")

    b3 = bench[bench["benchmark"] == "B3_aggregate_navigation"]
    atomic = row(b3, "from_atomic_fact")
    agg = row(b3, "from_daily_aggregate")

    b4 = bench[bench["benchmark"] == "B4_engine_comparison"]
    duckdb_row = row(b4, "duckdb")
    pandas_row = row(b4, "pandas")

    fact_trip = a10[a10["fact_table"] == "fact_trip"].iloc[0]
    fact_fuel = a10[a10["fact_table"] == "fact_fuel_price_monthly"].iloc[0]

    rows = []

    def add(benchmark, a_label, a_val, b_label, b_val, unit, reported, reported_source, note=""):
        ratio = a_val / b_val
        differs = abs(ratio - reported) / ratio > 0.01 if reported is not None else None
        rows.append({
            "benchmark": benchmark,
            "condition_a_label": a_label,
            "condition_a_value": sig4(a_val),
            "condition_b_label": b_label,
            "condition_b_value": sig4(b_val),
            "unit": unit,
            "ratio_unrounded_4sf": sig4(ratio),
            "ratio_as_previously_reported": reported if reported is not None else "",
            "previously_reported_source": reported_source,
            "differs_gt_1pct": differs if differs is not None else "",
            "note": note,
        })

    add(
        "columnar_storage_size_parquet_vs_csv",
        "csv_bytes", float(csv["bytes"]), "parquet_bytes", float(parquet["bytes"]), "bytes",
        None, "no single ratio number is quoted for storage size anywhere in "
              "outputs/benchmarks/; only per-format sizes and bytes_per_row are "
              "reported there (105.3 MB vs 407.9 MB, 37.7289 vs 146.1637 bytes/row). "
              "Left blank rather than invented -- check directly against the paper's "
              "own wording for this figure.",
        note="ratio is CSV bytes / Parquet bytes (how many times larger CSV is).",
    )
    add(
        "columnar_single_column_query_speedup_parquet_vs_csv",
        "csv_single_column_seconds", float(csv["single_column_median_seconds"]),
        "parquet_single_column_seconds", float(parquet["single_column_median_seconds"]), "seconds",
        36.164, "outputs/benchmarks/benchmark_results.csv, column "
                "csv_vs_parquet_single_column_ratio (same run, same code path as this "
                "recomputation -- expected to match exactly).",
    )
    add(
        "partition_pruning_partitioned_vs_consolidated",
        "partitioned_pruned_seconds", float(pruned["median_seconds"]),
        "consolidated_seconds", float(consolidated["median_seconds"]), "seconds",
        1.81, "outputs/benchmarks/benchmark_results.md prose: 'The pruned query was "
              "1.81x SLOWER than the single consolidated file.' Ratio here is "
              "partitioned/consolidated (>1 means pruning was slower), matching that "
              "sentence's direction -- NOT the stored speedup_from_pruning column "
              "(0.5514), which is the reciprocal (consolidated/partitioned).",
    )
    add(
        "aggregate_navigation_atomic_vs_daily_agg",
        "from_atomic_fact_seconds", float(atomic["median_seconds"]),
        "from_daily_aggregate_seconds", float(agg["median_seconds"]), "seconds",
        131, "paper prose, as quoted by the user commissioning this repair pass: "
             "'27.23 s / 0.21 s was reported as \"131 times\"'. Recomputing from "
             "those PRINTED (rounded) values gives 27.23/0.21 = 129.7, not 131 -- "
             "the reviewer's complaint. Recomputing from the UNROUNDED stored values "
             "(27.230352 / 0.207308) gives 131.35, which the paper's '131x' rounds "
             "to correctly. The defect is the printed precision (3 sig figs on the "
             "denominator, 0.21, is not enough to reproduce the ratio the paper "
             "itself states), not the underlying arithmetic.",
    )
    add(
        "engine_comparison_wall_clock_pandas_vs_duckdb",
        "pandas_seconds", float(pandas_row["median_seconds"]),
        "duckdb_seconds", float(duckdb_row["median_seconds"]), "seconds",
        1.606, "outputs/benchmarks/benchmark_results.csv, column pandas_vs_duckdb_ratio "
               "(same run, same code path as this recomputation).",
    )
    add(
        "engine_comparison_peak_memory_pandas_vs_duckdb",
        "pandas_peak_rss_gb", float(pandas_row["peak_rss_gb"]),
        "duckdb_peak_rss_gb", float(duckdb_row["peak_rss_gb"]), "GB",
        None, "no peak-memory ratio is stored anywhere in outputs/benchmarks/ -- only "
              "the wall-clock ratio (pandas_vs_duckdb_ratio) was computed there. Left "
              "blank rather than invented.",
        note="ratio is pandas peak RSS / DuckDB peak RSS.",
    )
    add(
        "grain_comparison_national_aggregate_fact_trip_vs_fact_fuel_price_monthly",
        "fact_trip_seconds", float(fact_trip["national_aggregate_seconds_measured"]),
        "fact_fuel_price_monthly_seconds", float(fact_fuel["national_aggregate_seconds_measured"]),
        "seconds",
        3268, "paper prose, as quoted by the user commissioning this repair pass: "
              "'14.03 s / 0.004 s was reported as \"3,268 times\"'. Recomputing from "
              "those PRINTED (rounded) values gives 14.03/0.004 = 3507.5 (~3,508), "
              "not 3,268 -- the reviewer's complaint. Recomputing from the UNROUNDED "
              "stored values (14.029315 / 0.004293, outputs/tables/A10_grain_"
              "comparison.csv) gives 3267.6, which the paper's '3,268x' rounds to "
              "correctly. Root cause: 0.004 has only 1 significant figure against "
              "the true value 0.004293 -- 4-sig-fig printing (this file) fixes it.",
    )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out)} rows)")
    print(out[["benchmark", "ratio_unrounded_4sf", "ratio_as_previously_reported",
                "differs_gt_1pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
