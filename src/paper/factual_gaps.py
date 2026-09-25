"""Task 6 (paper repair pass): three factual gaps the paper leaves open.

(a) Names the fourth co-missing field (airport_fee, alongside passenger_count,
    congestion_surcharge and rate_code) and exports the full 16-row 2^4
    presence/absence table, with which vendor the all-missing pattern belongs
    to.
(b) Explains dim_trip_flags' 25 observed rows (not <= 2^4 = 16): a query, not
    an assertion.
(c) Exports dim_geography's exact composition by country/admin_level plus the
    Unknown member, and states the WFP market-level count explicitly.

Read-only against the existing warehouse. Writes only to outputs/paper/.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from config import settings

PAPER_DIR = settings.OUTPUTS_DIR / "paper"


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)

    # ------------------------------------------------------------------ #
    # (a) co-missingness: the four fields, the full 2^4 pattern table, and
    #     which vendor the all-missing pattern belongs to.
    # ------------------------------------------------------------------ #
    flagged_and_vendor = con.execute('''
        WITH flagged AS (
            SELECT
                v.vendor_name,
                (f.passenger_count      IS NULL) AS missing_passenger_count,
                (f.congestion_surcharge IS NULL) AS missing_congestion_surcharge,
                (f.rate_key             = -1)    AS missing_rate_code,
                (f.airport_fee          IS NULL) AS missing_airport_fee
            FROM fact_trip f
            JOIN dim_vendor v ON v.vendor_key = f.vendor_key
            WHERE f.year = 2024
        )
        SELECT missing_passenger_count, missing_congestion_surcharge,
               missing_rate_code, missing_airport_fee, vendor_name,
               count(*) AS trip_count
        FROM flagged
        GROUP BY ALL
        ORDER BY trip_count DESC
    ''').df()

    # Full 2^4 = 16-row scaffold of every presence/absence combination,
    # left-joined against what is actually observed (collapsed across
    # vendor, since a pattern can in principle span more than one vendor).
    combos = []
    for p in range(2):
        for c in range(2):
            for r in range(2):
                for a in range(2):
                    combos.append((bool(p), bool(c), bool(r), bool(a)))
    scaffold = pd.DataFrame(combos, columns=[
        "missing_passenger_count", "missing_congestion_surcharge",
        "missing_rate_code", "missing_airport_fee",
    ])
    observed_by_pattern = (
        flagged_and_vendor.groupby(
            ["missing_passenger_count", "missing_congestion_surcharge",
             "missing_rate_code", "missing_airport_fee"], as_index=False
        )["trip_count"].sum()
    )
    vendors_by_pattern = (
        flagged_and_vendor[flagged_and_vendor["trip_count"] > 0]
        .groupby(["missing_passenger_count", "missing_congestion_surcharge",
                   "missing_rate_code", "missing_airport_fee"])["vendor_name"]
        .apply(lambda s: ";".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"vendor_name": "vendor_name(s)_present_in_this_pattern"})
    )
    table16 = scaffold.merge(observed_by_pattern, how="left", on=list(scaffold.columns))
    table16 = table16.merge(vendors_by_pattern, how="left", on=list(scaffold.columns))
    table16["trip_count"] = table16["trip_count"].fillna(0).astype("int64")
    table16["vendor_name(s)_present_in_this_pattern"] = (
        table16["vendor_name(s)_present_in_this_pattern"].fillna("")
    )
    total = int(table16["trip_count"].sum())
    table16["pct_of_trips"] = (100.0 * table16["trip_count"] / total).round(6)
    table16["pattern_signature"] = table16.apply(
        lambda r: "".join([
            "P" if r["missing_passenger_count"] else "-",
            "C" if r["missing_congestion_surcharge"] else "-",
            "R" if r["missing_rate_code"] else "-",
            "A" if r["missing_airport_fee"] else "-",
        ]), axis=1,
    )
    table16 = table16.sort_values("trip_count", ascending=False).reset_index(drop=True)
    table16.to_csv(PAPER_DIR / "comissingness_16_patterns.csv", index=False)
    occurring = table16[table16["trip_count"] > 0]
    print(f"(a) fourth co-missing field: airport_fee (alongside passenger_count, "
          f"congestion_surcharge, rate_code). {len(occurring)} of 16 patterns actually "
          f"occur:")
    print(occurring[["pattern_signature", "trip_count", "pct_of_trips",
                      "vendor_name(s)_present_in_this_pattern"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # (b) dim_trip_flags: distinct value counts per column, and row count.
    # ------------------------------------------------------------------ #
    flags = con.execute("SELECT * FROM dim_trip_flags").df()
    distinct_counts = {
        col: {
            "n_distinct_including_null": int(flags[col].nunique(dropna=False)),
            "n_distinct_excluding_null": int(flags[col].nunique(dropna=True)),
            "n_nulls": int(flags[col].isna().sum()),
            "values": sorted([str(v) for v in flags[col].dropna().unique()]),
        }
        for col in ["flag_key", "store_and_fwd_flag", "is_airport_trip", "has_tip", "has_toll"]
    }
    n_total_rows = len(flags)
    n_unknown_member = int((flags["flag_key"] == settings.UNKNOWN_KEY).sum())
    n_observed_combos = n_total_rows - n_unknown_member
    sf_vals = distinct_counts["store_and_fwd_flag"]["n_distinct_including_null"]
    explanation = (
        f"dim_trip_flags holds {n_total_rows} rows total: {n_observed_combos} observed "
        f"combinations of 4 attributes plus 1 Unknown member (flag_key={settings.UNKNOWN_KEY}). "
        f"The apparent '25 > 2^4=16' discrepancy assumes all 4 attributes are binary; "
        f"store_and_fwd_flag is actually {sf_vals}-valued (Y / N / NULL, "
        f"sql/ddl/dim_trip_flags.sql's own header states the true theoretical maximum as "
        f"3 x 2 x 2 x 2 = 24), so {n_observed_combos} of 24 possible combinations are "
        f"observed in the data, plus the dimension's standard Unknown member row makes "
        f"{n_total_rows}."
    )
    print(f"\n(b) {explanation}")

    dtf_rows = [{"column": col, **stats} for col, stats in distinct_counts.items()]
    dtf_df = pd.DataFrame([{**r, "values": ";".join(r["values"])} for r in dtf_rows])
    dtf_df["n_total_dim_rows"] = n_total_rows
    dtf_df["n_unknown_member_rows"] = n_unknown_member
    dtf_df["n_observed_combinations"] = n_observed_combos
    dtf_df["theoretical_max_combinations"] = 24
    dtf_df.to_csv(PAPER_DIR / "dim_trip_flags_composition.csv", index=False)

    # ------------------------------------------------------------------ #
    # (c) dim_geography: current-row composition by country/admin_level,
    #     the Unknown member, and the WFP market-level count.
    # ------------------------------------------------------------------ #
    geo_composition = con.execute('''
        SELECT country, admin_level, count(*) AS n_rows
        FROM dim_geography
        WHERE is_current
        GROUP BY ALL
        ORDER BY country, admin_level
    ''').df()
    geo_total = con.execute("SELECT count(*) AS n FROM dim_geography WHERE is_current").df().iloc[0]["n"]
    unknown_member = con.execute(f'''
        SELECT count(*) AS n FROM dim_geography
        WHERE is_current AND geo_key = {settings.UNKNOWN_KEY}
    ''').df().iloc[0]["n"]
    market_count = con.execute('''
        SELECT count(*) AS n FROM dim_geography
        WHERE is_current AND admin_level = 'market'
    ''').df().iloc[0]["n"]
    geo_composition.to_csv(PAPER_DIR / "dim_geography_composition.csv", index=False)
    print(f"\n(c) dim_geography: {int(geo_total)} current rows total "
          f"(including {int(unknown_member)} Unknown member), "
          f"of which {int(market_count)} are WFP market-level entries "
          f"(admin_level='market').")
    print(geo_composition.to_string(index=False))
    assert int(geo_total) == 422, (
        f"expected 422 current dim_geography rows per the paper's stated figure, "
        f"got {int(geo_total)} -- reporting the discrepancy rather than silently "
        f"adopting either number."
    )

    con.close()


if __name__ == "__main__":
    main()
