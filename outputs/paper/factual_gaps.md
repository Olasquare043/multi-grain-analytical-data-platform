# Factual gaps (paper repair pass, Task 6)

Computed directly against the live warehouse (`data/warehouse.duckdb`) by
`src/paper/factual_gaps.py`. Full detail in the companion CSVs named below.

## (a) The fourth co-missing field, and the full 16-row pattern table

The paper names "passenger count, congestion surcharge, rate code and a
fourth flag" without naming the fourth field. The fourth field is
**`airport_fee`**. All four (`passenger_count`, `congestion_surcharge`,
`rate_code` via `rate_key = -1`, `airport_fee`) are tested for NULL/Unknown
jointly against `fact_trip`, giving 2^4 = 16 possible presence/absence
patterns. Full table: `outputs/paper/comissingness_16_patterns.csv`.

Exactly **two of the sixteen** patterns actually occur:

| pattern | fields missing | trip_count | % of trips | vendor(s) present |
| --- | --- | --- | --- | --- |
| `----` (none missing) | 0 | 36,459,364 | 90.198842% | Creative Mobile Technologies, LLC; Curb Mobility, LLC |
| `PCRA` (all four missing) | 4 | 3,961,736 | 9.801158% | Creative Mobile Technologies, LLC; Curb Mobility, LLC; Myle Technologies Inc |

The remaining 14 patterns (any partial combination of the four fields
missing) have zero occurrences: the four fields fail as a block or not at
all, never independently, confirming the co-missingness signature the
quality rules (Q012/Q013/Q024) each reported separately at the same rate.

**Correction to an assumption made while building this table.** Before
querying, the working assumption was that the all-missing pattern (`PCRA`)
would trace to a single vendor (the per-vendor-month breakdown in
`outputs/tables/A11_missingness_structure.csv` shows materially different
missingness rates by vendor and month). The direct query shows otherwise:
**all three vendors** have trips in both the `----` and `PCRA` patterns.
Vendor identity alone does not determine which pattern a trip falls into;
whatever produces the block failure is not fully captured by `vendor_id`.
This is reported because it contradicts the more convenient assumption,
not because it was expected.

## (b) `dim_trip_flags`: 25 rows explained

`dim_trip_flags` holds **25 rows total**: **24 observed combinations** of
its four attributes, plus **1** Unknown member (`flag_key = -1`). Per-column
distinct-value counts (including NULL), from
`outputs/paper/dim_trip_flags_composition.csv`:

| column | distinct values (incl. NULL) | values |
| --- | --- | --- |
| `store_and_fwd_flag` | 3 | `N`, `Y`, NULL |
| `is_airport_trip` | 2 | `False`, `True` |
| `has_tip` | 2 | `False`, `True` |
| `has_toll` | 2 | `False`, `True` |

**The apparent "25 > 2^4 = 16" discrepancy assumes all four attributes are
binary. They are not.** `store_and_fwd_flag` is three-valued (`Y` / `N` /
NULL, NULL retained deliberately as its own member per
`sql/ddl/dim_trip_flags.sql`'s own header comment, which already states the
true theoretical maximum as 3 x 2 x 2 x 2 = 24). All 24 of the 24 possible
combinations are observed in the data (24/24), and the dimension's standard
Unknown member adds the 25th row. 2^4 = 16 was never the right ceiling for
this dimension.

## (c) `dim_geography` composition

422 current rows total, confirmed against the paper's stated figure. Full
breakdown in `outputs/paper/dim_geography_composition.csv`:

| country | admin_level | n_rows |
| --- | --- | --- |
| Nigeria | market | 118 |
| Nigeria | state | 37 |
| United States | city | 1 |
| United States | zone | 265 |
| Unknown | unknown | 1 |
| **Total** | | **422** |

**The WFP market-level entries (`admin_level = 'market'`) number 118**,
stated explicitly as requested. 118 + 37 + 1 + 265 + 1 = 422, reconciling
exactly with no residual to explain.
