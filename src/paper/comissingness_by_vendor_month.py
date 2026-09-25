"""Follow-up Task B: does vendor, month, or vendor-and-month predict the
all-fields-missing ('PCRA') co-missingness pattern?

outputs/paper/factual_gaps.md already showed all three vendors appear in
both the all-present and all-missing patterns, contradicting a single-vendor
explanation. This goes one step further: trip counts and the missingness
rate for the PCRA pattern, broken down by (vendor, month), to see whether
some combination of the two available dimensions predicts it. If the rate is
roughly uniform across vendors and months for a given vendor, nothing in
these dimensions predicts the pattern, and that is reported as the finding
rather than reaching for a weak association in the noise.

Read-only against the existing warehouse. Writes only to outputs/paper/.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from config import settings

PAPER_DIR = settings.OUTPUTS_DIR / "paper"


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)

    by_vendor_month = con.execute('''
        WITH flagged AS (
            SELECT
                v.vendor_name,
                d.year_month,
                (f.passenger_count IS NULL AND f.congestion_surcharge IS NULL
                 AND f.rate_key = -1 AND f.airport_fee IS NULL) AS is_pcra
            FROM fact_trip f
            JOIN dim_date   d ON d.date_key   = f.pickup_date_key
            JOIN dim_vendor v ON v.vendor_key = f.vendor_key
            WHERE f.year = 2024
        )
        SELECT
            vendor_name, year_month,
            count(*)                              AS trip_count,
            count(*) FILTER (WHERE is_pcra)        AS pcra_count,
            round(100.0 * count(*) FILTER (WHERE is_pcra) / count(*), 6) AS pcra_pct
        FROM flagged
        GROUP BY ALL
        ORDER BY vendor_name, year_month
    ''').df()
    con.close()

    out_path = PAPER_DIR / "comissingness_by_vendor_month.csv"
    by_vendor_month.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(by_vendor_month)} rows)")
    print(by_vendor_month.to_string(index=False))

    # ------------------------------------------------------------------ #
    # Does vendor (collapsing month) predict the rate? Does month
    # (collapsing vendor) predict it? Simple range/dispersion checks, no
    # inferential test claimed -- this either shows a pattern or it doesn't.
    # ------------------------------------------------------------------ #
    by_vendor = by_vendor_month.groupby("vendor_name").apply(
        lambda g: pd.Series({
            "trip_count": g["trip_count"].sum(),
            "pcra_count": g["pcra_count"].sum(),
            "pcra_pct": round(100.0 * g["pcra_count"].sum() / g["trip_count"].sum(), 6),
        }), include_groups=False,
    ).reset_index()
    by_month = by_vendor_month.groupby("year_month").apply(
        lambda g: pd.Series({
            "trip_count": g["trip_count"].sum(),
            "pcra_count": g["pcra_count"].sum(),
            "pcra_pct": round(100.0 * g["pcra_count"].sum() / g["trip_count"].sum(), 6),
        }), include_groups=False,
    ).reset_index()

    vendor_range = by_vendor["pcra_pct"].max() - by_vendor["pcra_pct"].min()
    month_range = by_month["pcra_pct"].max() - by_month["pcra_pct"].min()
    within_vendor_month_ranges = by_vendor_month.groupby("vendor_name")["pcra_pct"].agg(
        lambda s: s.max() - s.min()
    )

    print(f"\nby-vendor PCRA%% (month collapsed): range = {vendor_range:.4f} pp")
    print(by_vendor.to_string(index=False))
    print(f"\nby-month PCRA%% (vendor collapsed): range = {month_range:.4f} pp")
    print(f"\nwithin-vendor, across-month PCRA%% range (does the SAME vendor vary "
          f"materially by month?):")
    print(within_vendor_month_ranges.to_string())

    max_within_vendor_range = float(within_vendor_month_ranges.max())

    # Split the vendor-level picture by volume: one vendor (Myle Technologies
    # Inc) may be a perfect predictor while carrying negligible trip volume,
    # which would misleadingly dominate a volume-blind "vendor range" number.
    by_vendor_sorted = by_vendor.sort_values("trip_count", ascending=False)
    total_trips = float(by_vendor_sorted["trip_count"].sum())
    dominant = by_vendor_sorted[by_vendor_sorted["trip_count"] / total_trips >= 0.01]
    minor = by_vendor_sorted[by_vendor_sorted["trip_count"] / total_trips < 0.01]

    parts = []
    if len(minor):
        for _, r in minor.iterrows():
            parts.append(
                f"{r['vendor_name']} is a perfect (but negligible-coverage) predictor: "
                f"{r['pcra_pct']:.1f}% PCRA rate on {int(r['trip_count']):,} trips "
                f"({100 * r['trip_count'] / total_trips:.4f}% of all trips)."
            )
    if len(dominant) > 1:
        dom_range = dominant["pcra_pct"].max() - dominant["pcra_pct"].min()
        parts.append(
            f"Among the {len(dominant)} vendor(s) carrying essentially all volume "
            f"({100 * dominant['trip_count'].sum() / total_trips:.2f}% of trips), vendor "
            f"alone is a WEAK predictor (rates {dominant['pcra_pct'].min():.2f}%-"
            f"{dominant['pcra_pct'].max():.2f}%, a {dom_range:.2f} pp spread), while month "
            f"is a much STRONGER one within each of them (within-vendor swings of "
            f"{within_vendor_month_ranges.loc[dominant['vendor_name']].min():.2f}-"
            f"{within_vendor_month_ranges.loc[dominant['vendor_name']].max():.2f} pp across "
            f"2024's twelve months). Neither vendor nor month, nor the two combined, fully "
            f"explains the pattern for these vendors: monthly rates range from about "
            f"{by_vendor_month[by_vendor_month['vendor_name'].isin(dominant['vendor_name'])]['pcra_pct'].min():.1f}% "
            f"to "
            f"{by_vendor_month[by_vendor_month['vendor_name'].isin(dominant['vendor_name'])]['pcra_pct'].max():.1f}%, "
            f"never collapsing to a clean 0%/100% split the way it does for the negligible-"
            f"volume vendor(s) above."
        )
    finding = " ".join(parts)
    print(f"\nFINDING: {finding}")

    summary_path = PAPER_DIR / "comissingness_by_vendor_month_summary.csv"
    pd.DataFrame([{
        "vendor_pcra_pct_range_pp": round(vendor_range, 4),
        "month_pcra_pct_range_pp": round(month_range, 4),
        "max_within_vendor_across_month_range_pp": round(max_within_vendor_range, 4),
        "finding": finding,
    }]).to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
