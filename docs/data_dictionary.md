# Data Dictionary

CSC 796 Advanced Data Engineering -- *Engineering a Multi-Grain Analytical Data Platform for the Economics of Movement*.

Generated from the live warehouse catalogue on 2026-09-14 17:35:30 UTC. Column names and types are read from the schema on every run, so this document cannot drift from the tables it describes.

## Conventions

- Every dimension carries an **Unknown member** at surrogate key `-1`. Facts route unmatched codes there rather than losing rows to an inner join, which is what makes referential integrity total.
- Surrogate keys are integers. Natural keys are retained as attributes.
- `dim_geography` is **Type 2** slowly changing; every other dimension is Type 1. The reasoning is in `docs/scd_strategy.md`.
- Money is USD for New York, NGN for Nigeria. No cross-currency measure is computed anywhere.

## Contents

- [`dim_date`](#dim-date)
- [`dim_payment_type`](#dim-payment-type)
- [`dim_rate_code`](#dim-rate-code)
- [`dim_vendor`](#dim-vendor)
- [`dim_transport_mode`](#dim-transport-mode)
- [`dim_commodity`](#dim-commodity)
- [`dim_trip_flags`](#dim-trip-flags)
- [`dim_geography`](#dim-geography)
- [`fact_trip`](#fact-trip)
- [`fact_market_price_monthly`](#fact-market-price-monthly)
- [`fact_fuel_price_monthly`](#fact-fuel-price-monthly)
- [`fact_trip_daily_agg`](#fact-trip-daily-agg)
- [`fact_weather_daily`](#fact-weather-daily)

## dim_date

**Dimension (conformed, Type 1)**

- **Grain:** One row per calendar day, 2016-01-01 to 2026-12-31, plus one Unknown member at date_key = -1.
- **Source:** Generated. No input data; cannot carry a fabricated value.
- **Rows:** 4,019
- **On disk:** 23.1 KB (5.881 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `date_key` | INTEGER | Surrogate key, INTEGER YYYYMMDD. -1 = Unknown member. |
| `full_date` | DATE | Calendar date. NULL on the Unknown member. |
| `year` | SMALLINT | Hive partition column: year of pickup. |
| `quarter` | TINYINT |  |
| `month_number` | TINYINT |  |
| `month_name` | VARCHAR |  |
| `day_of_month` | TINYINT |  |
| `day_of_week` | TINYINT | ISO numbering: 1 = Monday .. 7 = Sunday. |
| `day_name` | VARCHAR |  |
| `is_weekend` | BOOLEAN |  |
| `week_of_year` | TINYINT |  |
| `year_month` | VARCHAR | 'YYYY-MM' string, for grouping and labelling. |
| `is_month_end` | BOOLEAN | True on the last calendar day of the month. |

## dim_payment_type

**Dimension (Type 1, decoded dictionary)**

- **Grain:** One row per TLC payment_type code.
- **Source:** NYC TLC Yellow Trips data dictionary.
- **Rows:** 8
- **On disk:** 1.0 KB (131.25 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `payment_key` | INTEGER | dim_payment_type surrogate. |
| `payment_type_id` | INTEGER |  |
| `payment_type_name` | VARCHAR |  |
| `payment_description` | VARCHAR |  |
| `is_tip_observable` | BOOLEAN | True only for credit card, the one payment type whose tips the meter records. |

## dim_rate_code

**Dimension (Type 1, decoded dictionary)**

- **Grain:** One row per TLC RatecodeID.
- **Source:** NYC TLC Yellow Trips data dictionary.
- **Rows:** 8
- **On disk:** 1.0 KB (133.875 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `rate_key` | INTEGER | dim_rate_code surrogate. |
| `rate_code_id` | INTEGER |  |
| `rate_code_name` | VARCHAR |  |
| `rate_description` | VARCHAR |  |
| `is_airport_rate` | BOOLEAN |  |

## dim_vendor

**Dimension (Type 1, decoded dictionary)**

- **Grain:** One row per licensed technology provider code.
- **Source:** NYC TLC Yellow Trips data dictionary.
- **Rows:** 5
- **On disk:** 955 B (191.0 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `vendor_key` | INTEGER | dim_vendor surrogate. |
| `vendor_id` | INTEGER |  |
| `vendor_name` | VARCHAR |  |
| `vendor_description` | VARCHAR |  |

## dim_transport_mode

**Dimension (Type 1)**

- **Grain:** One row per transport mode. Currently one real member.
- **Source:** Defined by the platform, not sourced.
- **Rows:** 2
- **On disk:** 1011 B (505.5 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `mode_key` | INTEGER | dim_transport_mode surrogate. |
| `mode_code` | VARCHAR |  |
| `mode_name` | VARCHAR |  |
| `mode_category` | VARCHAR |  |
| `country_scope` | VARCHAR |  |
| `is_metered` | BOOLEAN |  |

## dim_commodity

**Dimension (Type 1)**

- **Grain:** One row per (commodity, unit). Unit is part of the identity: 1 KG of rice and 100 KG of rice are different products.
- **Source:** Derived from silver_wfp_prices (Source C).
- **Rows:** 67
- **On disk:** 2.5 KB (37.806 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `commodity_key` | INTEGER | dim_commodity surrogate. |
| `commodity_code` | VARCHAR |  |
| `commodity_name` | VARCHAR |  |
| `category` | VARCHAR |  |
| `unit` | VARCHAR | Published unit of sale. Part of dim_commodity's grain. |
| `is_fuel` | BOOLEAN | True for the two WFP fuel commodities; excluded from every food index. |

## dim_trip_flags

**Dimension (Type 1, **junk dimension**)**

- **Grain:** One row per OBSERVED combination of four low-cardinality trip attributes, plus an Unknown member.
- **Source:** Derived from silver_nyc_trip (Source A).
- **Rows:** 25
- **On disk:** 736 B (29.44 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `flag_key` | INTEGER | dim_trip_flags surrogate (junk dimension). |
| `store_and_fwd_flag` | VARCHAR |  |
| `is_airport_trip` | BOOLEAN |  |
| `has_tip` | BOOLEAN |  |
| `has_toll` | BOOLEAN |  |

## dim_geography

**Dimension (conformed, ragged hierarchy, **Type 2 SCD**)**

- **Grain:** One row per (geo_code, country, admin_level) VERSION. A natural key may hold several versions, of which exactly one is current.
- **Source:** NYC taxi zone lookup (Source B), WFP market register and price panel (Source C), and config/nigeria_states.py for the 36 states plus FCT.
- **Rows:** 422
- **On disk:** 7.0 KB (16.865 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `geo_key` | INTEGER | Surrogate key into dim_geography. -1 = Unknown member. |
| `geo_code` | VARCHAR | Natural key within (country, admin_level): a TLC LocationID, a WFP market id, or a canonical Nigerian state name. |
| `geo_name` | VARCHAR |  |
| `country` | VARCHAR |  |
| `admin_level` | VARCHAR | zone | market | state | city. 'city' is a documented extension carrying the single New York City member that fact_weather_daily references. |
| `parent_geo_name` | VARCHAR | Parent in the ragged hierarchy: borough for a zone, state for a market, 'Nigeria' for a state. Type 2 tracked. |
| `region_group` | VARCHAR | TLC service zone for a zone; geopolitical zone for a Nigerian state or market. Type 2 tracked. |
| `valid_from` | DATE | Start of this version's validity (Type 2). |
| `valid_to` | DATE | End of validity; 9999-12-31 while current (Type 2). |
| `is_current` | BOOLEAN | True on exactly one version per natural key (Type 2). |

## fact_trip

**Fact (atomic, transactional)**

- **Grain:** **ONE ROW PER COMPLETED TAXI TRIP.** Parquet, Hive-partitioned year=YYYY/month=M.
- **Source:** NYC TLC Yellow Taxi trip records 2024 (Source A), after structural rejection at the silver boundary.
- **Rows:** 40,421,155
- **On disk:** 1.4 GB (37.985 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `trip_id` | VARCHAR | Degenerate dimension. Deterministic MD5 of the row's business content; stable across reruns, never random. TLC publishes no trip identifier. |
| `pickup_date_key` | INTEGER | dim_date surrogate for the pickup date. -1 where the pickup falls outside the generated calendar, which the official 2024 files genuinely contain. |
| `dropoff_date_key` | INTEGER | dim_date surrogate for the dropoff date. |
| `pickup_hour` | TINYINT | Hour of pickup, 0-23, local New York time. |
| `pickup_geo_key` | INTEGER | dim_geography surrogate for the PICKUP zone. |
| `dropoff_geo_key` | INTEGER | dim_geography surrogate for the DROPOFF zone. |
| `mode_key` | INTEGER | dim_transport_mode surrogate. |
| `vendor_key` | INTEGER | dim_vendor surrogate. |
| `payment_key` | INTEGER | dim_payment_type surrogate. |
| `rate_key` | INTEGER | dim_rate_code surrogate. |
| `flag_key` | INTEGER | dim_trip_flags surrogate (junk dimension). |
| `passenger_count` | SMALLINT | As transmitted by the vendor. NULL for ~9.8% of 2024 trips; never defaulted to zero. See analysis A11. |
| `trip_distance_miles` | DOUBLE | Metered distance in miles. |
| `trip_duration_seconds` | INTEGER | Dropoff minus pickup, in seconds. Strictly positive and at most 86,400 by construction. |
| `avg_speed_mph` | DOUBLE | Derived: distance / duration. A congestion PROXY only. |
| `fare_amount` | DOUBLE | Metered fare, USD. Non-negative by construction. |
| `extra` | DOUBLE | Miscellaneous extras and surcharges, USD. |
| `mta_tax` | DOUBLE | MTA tax, USD. |
| `tip_amount` | DOUBLE | Tip, USD. CARD TIPS ONLY -- the meter does not record cash tips. A zero on a cash trip is evidence about the recording system, not about the passenger. |
| `tolls_amount` | DOUBLE | Tolls, USD. |
| `improvement_surcharge` | DOUBLE | Improvement surcharge, USD. |
| `congestion_surcharge` | DOUBLE | Congestion surcharge, USD. NULL on the same rows that lack passenger_count (see A11). |
| `airport_fee` | DOUBLE | Airport pickup fee, USD. |
| `total_amount` | DOUBLE | Total charged to the passenger, USD. Not operator net revenue. |
| `month` | BIGINT | Hive partition column: month of pickup, 1-12. |
| `year` | BIGINT | Hive partition column: year of pickup. |

## fact_market_price_monthly

**Fact (periodic snapshot)**

- **Grain:** **ONE ROW PER (market, commodity, price type, month).**
- **Source:** WFP Nigeria market food prices via HDX (Source C).
- **Rows:** 87,384
- **On disk:** 434.0 KB (5.086 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `month_key` | INTEGER | INTEGER YYYYMM01, the first day of the month. |
| `geo_key` | INTEGER | Surrogate key into dim_geography. -1 = Unknown member. |
| `commodity_key` | INTEGER | dim_commodity surrogate. |
| `price_type` | VARCHAR | Retail or Wholesale, as published by WFP. |
| `price_ngn` | DOUBLE | Price in Nigerian naira, as published. |
| `price_usd` | DOUBLE | WFP's own USD conversion. Carried but never used to build an index, since the exchange rate would then drive the series. |
| `unit` | VARCHAR | Published unit of sale. Part of dim_commodity's grain. |
| `observation_count` | INTEGER | Source rows that collapsed into this cell. Normally 1; a rise would reveal a change in publication frequency rather than being silently averaged. |

## fact_fuel_price_monthly

**Fact (periodic snapshot)**

- **Grain:** **ONE ROW PER (state, month)** for Premium Motor Spirit.
- **Source:** NBS PMS Price Watch (Source D).
- **Rows:** 1,147
- **On disk:** 20.3 KB (18.101 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `month_key` | INTEGER | INTEGER YYYYMM01, the first day of the month. |
| `geo_key` | INTEGER | Surrogate key into dim_geography. -1 = Unknown member. |
| `fuel_type` | VARCHAR | 'PMS' (Premium Motor Spirit, petrol). |
| `price_ngn` | DOUBLE | Price in Nigerian naira, as published. |
| `mom_pct_change` | DOUBLE | Month-over-month % change, COMPUTED from the loaded series, never read from the published sheet. NULL where the prior month is absent; never interpolated. |
| `yoy_pct_change` | DOUBLE | Year-over-year % change, computed on the same basis. |
| `source_file` | VARCHAR | Originating file, for provenance. |

## fact_trip_daily_agg

**Fact (aggregate, derived)**

- **Grain:** **ONE ROW PER (date, pickup zone, mode).** Redundant by design; exists so benchmark B3 can measure what aggregate navigation buys.
- **Source:** Derived entirely from fact_trip.
- **Rows:** 84,910
- **On disk:** 1.8 MB (22.821 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `date_key` | INTEGER | Surrogate key, INTEGER YYYYMMDD. -1 = Unknown member. |
| `pickup_geo_key` | INTEGER | dim_geography surrogate for the PICKUP zone. |
| `mode_key` | INTEGER | dim_transport_mode surrogate. |
| `trip_count` | BIGINT | Additive count of trips. |
| `total_distance_miles` | DOUBLE | Additive. |
| `total_revenue` | DOUBLE | Additive. |
| `avg_fare` | DOUBLE | NOT re-aggregable: averaging across zones weights every zone equally. Roll up with total_revenue / trip_count instead. |
| `median_fare` | DOUBLE | Computed exactly, not approximated. Not re-aggregable. |
| `avg_duration_seconds` | DOUBLE | Not re-aggregable. |
| `avg_speed_mph` | DOUBLE | Derived: distance / duration. A congestion PROXY only. |

## fact_weather_daily

**Fact (periodic snapshot, OPTIONAL)**

- **Grain:** **ONE ROW PER (date, geography).** In practice one row per date: Open-Meteo publishes a single point series.
- **Source:** Open-Meteo archive API (Source E). Non-blocking; the core pipeline succeeds without it.
- **Rows:** 366
- **On disk:** 5.2 KB (14.689 bytes/row)

| Column | Type | Description |
| --- | --- | --- |
| `date_key` | INTEGER | Surrogate key, INTEGER YYYYMMDD. -1 = Unknown member. |
| `geo_key` | INTEGER | Surrogate key into dim_geography. -1 = Unknown member. |
| `temp_max_c` | DOUBLE | Daily maximum air temperature, degrees Celsius. |
| `temp_min_c` | DOUBLE | Daily minimum air temperature, degrees Celsius. |
| `precipitation_mm` | DOUBLE | Daily precipitation total, millimetres. |
| `snowfall_mm` | DOUBLE | Daily snowfall total, millimetres. |
| `wind_speed_max_kmh` | DOUBLE | Daily maximum 10 m wind speed, km/h. |

