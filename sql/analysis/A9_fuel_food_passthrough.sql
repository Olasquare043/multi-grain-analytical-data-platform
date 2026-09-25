-- ===========================================================================
-- A9  fuel_food_passthrough
--
-- ****************************************************************************
-- *  EXPLORATORY ANALYSIS -- NULL RESULT -- NOT A FINDING                     *
-- *                                                                          *
-- *  WFP had stopped reporting 11 of the 14 states it monitored by January  *
-- *  2023 (ten in January 2023 itself; Sokoto in April 2019).                *
-- *  This analysis therefore rests on 3 states, all located in the North    *
-- *  East (Borno, Yobe, Adamawa). The correlations describe conflict-       *
-- *  affected north-eastern markets in those three states only. They are   *
-- *  under +/-0.09 at every lag and cannot support any inference.           *
-- *  The output is retained                                                 *
-- *  for transparency only; it is not presented in the results. See         *
-- *  docs/methodology_notes.md, Limitations.                                *
-- ****************************************************************************
--
-- Question   : Over the window where both sources genuinely overlap, is the
--              month-to-month movement in a state's petrol price associated
--              with the movement in its staple food prices, contemporaneously
--              or at a lag of one to three months?
-- Grain      : one row per (geopolitical zone, lag). A national row per lag is
--              also emitted, labelled zone = 'ALL ZONES'.
-- Output     : outputs/tables/A9_fuel_food_passthrough.csv
-- Figure     : A9_fuel_food_passthrough.png
--
-- ############################################################################
-- # LIMITATION 1 -- STATE COVERAGE. READ BEFORE USING ANY NUMBER BELOW.      #
-- #                                                                          #
-- # Nigeria has 37 federating units (36 states + FCT).                        #
-- #   Source D (NBS petrol)  covers ALL 37, 2023-11 onward.                   #
-- #   Source C (WFP food)    covers 14 states historically -- BUT 11 of them  #
-- #   stop reporting by January 2023 (Sokoto by 2019). Across the overlap    #
-- #   window only THREE states report: Borno, Yobe and Adamawa, all in the    #
-- #   North East, where WFP's humanitarian monitoring continued.              #
-- #                                                                           #
-- # The join therefore rests on 3 of 37 federating units (about 8 per cent),  #
-- # in ONE geopolitical zone. The 'ALL ZONES' row and the 'North East' row    #
-- # are the same three states. Five of six zones have no food data at all.    #
-- #                                                                           #
-- # Every output row carries n_states, n_months and n_pairs, plus the         #
-- # coverage columns, precisely so this cannot be overlooked; rows below the  #
-- # minimum pair threshold carry a NULL correlation rather than a number      #
-- # computed from too little data. The result describes conflict-affected     #
-- # north-eastern markets and does NOT generalise to Nigeria as a whole.      #
-- ############################################################################
--
-- LIMITATION 2 -- INDEPENDENT SOURCES, DIFFERENT METHODS.
--   The two series are collected by different organisations for different
--   purposes: the National Bureau of Statistics surveys fuel retail outlets and
--   publishes a state mean; the World Food Programme surveys selected markets
--   and publishes per-commodity quotes. They share no sampling frame, no
--   reference week, and no collection instrument. A correlation between them is
--   a correlation between two independently constructed estimates, and part of
--   any observed relationship may be common seasonality or common exposure to
--   the exchange rate rather than any transmission from fuel to food.
--
-- LIMITATION 3 -- ASSOCIATION ONLY, NEVER CAUSATION.
--   Nothing here identifies a causal effect. There is no instrument, no control
--   group and no exogenous variation. A positive lagged correlation is
--   consistent with fuel costs passing through to food prices; it is equally
--   consistent with both responding to a common macroeconomic shock. The paper
--   must report association and say so.
--
-- Method and assumptions:
--   1. Both series are differenced in LOGS before correlating. Price LEVELS in
--      this window are strongly trending, and correlating levels would yield a
--      large coefficient that reflects nothing but shared trend. The level
--      correlation is reported alongside, labelled as spurious-prone, so the
--      contrast is visible rather than hidden.
--   2. The food movement per state-month is the Jevons link of A6 computed
--      WITHIN the state: the geometric mean of price relatives over staple
--      cells observed in both adjacent months. Composition is thus held
--      constant inside every state-month observation.
--   3. Lags are applied to FUEL: food change at month m is paired with the fuel
--      change at month m-k. Pairs are formed only on calendar-adjacent months;
--      no gap is bridged.
--   4. Retail quotes and staple categories only (see config/settings.py).
--   5. A cell reporting fewer than {min_pairs} pairs gets NULL for every
--      correlation. Reporting a correlation from four points would be
--      misleading precision.
-- ===========================================================================
WITH state_dim AS (
    SELECT geo_code AS state, region_group AS geopolitical_zone
    FROM dim_geography
    WHERE admin_level = 'state' AND country = 'Nigeria' AND is_current
),
fuel AS (
    SELECT
        sd.state,
        sd.geopolitical_zone,
        (f.month_key // 10000) * 12 + ((f.month_key // 100) % 100)  AS month_index,
        printf('%04d-%02d', f.month_key // 10000,
               (f.month_key // 100) % 100)                          AS year_month,
        f.price_ngn                                                AS fuel_price_ngn
    FROM fact_fuel_price_monthly f
    JOIN dim_geography g ON g.geo_key = f.geo_key
    JOIN state_dim sd    ON sd.state  = g.geo_code
    WHERE f.price_ngn > 0
),
fuel_changes AS (
    SELECT
        state, geopolitical_zone, month_index, year_month, fuel_price_ngn,
        CASE WHEN lag(month_index) OVER w = month_index - 1
             THEN ln(fuel_price_ngn / lag(fuel_price_ngn) OVER w)
        END                                                        AS d_log_fuel
    FROM fuel
    WINDOW w AS (PARTITION BY state ORDER BY month_index)
),
-- ---- state-level staple food movement, composition held constant -----------
food_cells AS (
    SELECT
        market_geo.parent_geo_name                                 AS state,
        f.geo_key, f.commodity_key, f.price_type,
        (f.month_key // 10000) * 12 + ((f.month_key // 100) % 100)   AS month_index,
        f.price_ngn
    FROM fact_market_price_monthly f
    JOIN dim_commodity c ON c.commodity_key = f.commodity_key
    JOIN dim_geography market_geo
                         ON market_geo.geo_key     = f.geo_key
                        AND market_geo.admin_level = 'market'
    WHERE c.category IN {staple_categories}
      AND NOT c.is_fuel
      AND f.price_type = '{price_type}'
      AND f.price_ngn > 0
),
food_relatives AS (
    SELECT
        state, month_index, price_ngn,
        lag(price_ngn)   OVER w AS prev_price,
        lag(month_index) OVER w AS prev_month_index
    FROM food_cells
    WINDOW w AS (PARTITION BY state, geo_key, commodity_key, price_type
                 ORDER BY month_index)
),
food_changes AS (
    SELECT
        state,
        month_index,
        printf('%04d-%02d', (month_index - 1) // 12, ((month_index - 1) % 12) + 1)
                                                                   AS year_month,
        count(*) FILTER (WHERE prev_price > 0
                           AND prev_month_index = month_index - 1)  AS matched_cells,
        avg(ln(price_ngn / prev_price))
            FILTER (WHERE prev_price > 0
                      AND prev_month_index = month_index - 1)       AS d_log_food,
        avg(price_ngn)                                              AS mean_staple_price_ngn
    FROM food_relatives
    GROUP BY ALL
),
lags(lag_months) AS (VALUES {lag_values}),
-- ---- pair food at m with fuel at m-k ---------------------------------------
pairs AS (
    SELECT
        fc.state,
        fu.geopolitical_zone,
        l.lag_months,
        fc.year_month                                              AS food_month,
        fu.year_month                                              AS fuel_month,
        fc.d_log_food,
        fu.d_log_fuel,
        fc.mean_staple_price_ngn,
        fu.fuel_price_ngn,
        fc.matched_cells
    FROM food_changes fc
    CROSS JOIN lags l
    JOIN fuel_changes fu
           ON fu.state       = fc.state
          AND fu.month_index = fc.month_index - l.lag_months
    WHERE fc.d_log_food IS NOT NULL
      AND fu.d_log_fuel IS NOT NULL
      AND fc.matched_cells > 0
),
by_zone AS (
    SELECT
        geopolitical_zone                                          AS zone,
        lag_months,
        count(*)                                                   AS n_pairs,
        count(DISTINCT state)                                      AS n_states,
        count(DISTINCT food_month)                                 AS n_months,
        min(food_month)                                            AS first_food_month,
        max(food_month)                                            AS last_food_month,
        round(corr(d_log_food, d_log_fuel), 6)                     AS corr_log_changes,
        round(corr(mean_staple_price_ngn, fuel_price_ngn), 6)      AS corr_levels_spurious_prone,
        round(avg(d_log_food) * 100, 6)                            AS mean_food_pct_change,
        round(avg(d_log_fuel) * 100, 6)                            AS mean_fuel_pct_change,
        round(sum(matched_cells) * 1.0 / count(*), 3)              AS mean_matched_cells_per_pair
    FROM pairs
    GROUP BY ALL

    UNION ALL

    SELECT
        'ALL ZONES'                                                AS zone,
        lag_months,
        count(*), count(DISTINCT state), count(DISTINCT food_month),
        min(food_month), max(food_month),
        round(corr(d_log_food, d_log_fuel), 6),
        round(corr(mean_staple_price_ngn, fuel_price_ngn), 6),
        round(avg(d_log_food) * 100, 6),
        round(avg(d_log_fuel) * 100, 6),
        round(sum(matched_cells) * 1.0 / count(*), 3)
    FROM pairs
    GROUP BY ALL
),
coverage AS (
    SELECT
        count(DISTINCT state) AS states_in_overlap,
        array_to_string(list_sort(list_distinct(list(state))), ', ')
                              AS overlap_state_list,
        (SELECT count(*) FROM state_dim)              AS federating_units_total,
        (SELECT count(DISTINCT state) FROM fuel)      AS states_with_fuel_data,
        (SELECT count(DISTINCT state) FROM food_cells) AS states_with_food_data
    FROM pairs
)
SELECT
    'EXPLORATORY - null result; supports no inference'    AS analysis_status,
    z.zone,
    z.lag_months,
    z.n_pairs,
    z.n_states,
    z.n_months,
    z.first_food_month,
    z.last_food_month,
    -- Correlations are suppressed, not rounded away, below the pair threshold.
    CASE WHEN z.n_pairs >= {min_pairs} THEN z.corr_log_changes END
                                                          AS corr_log_changes,
    CASE WHEN z.n_pairs >= {min_pairs} THEN z.corr_levels_spurious_prone END
                                                          AS corr_levels_spurious_prone,
    (z.n_pairs >= {min_pairs})                            AS meets_pair_threshold,
    {min_pairs}                                           AS min_pairs_required,
    z.mean_food_pct_change,
    z.mean_fuel_pct_change,
    z.mean_matched_cells_per_pair,
    -- Coverage limitation carried on EVERY row, so no downstream consumer of
    -- this CSV can use a correlation without seeing what it rests on.
    c.states_in_overlap,
    c.states_with_fuel_data,
    c.states_with_food_data,
    c.federating_units_total,
    round(100.0 * c.states_in_overlap / c.federating_units_total, 2)
                                                          AS pct_of_federating_units,
    -- Built from the measured coverage, so the caveat can never drift from the
    -- data it describes.
    'association only; not causal; describes conflict-affected north-eastern '
        || 'markets (' || c.overlap_state_list || ') only; rests on '
        || CAST(c.states_in_overlap AS VARCHAR)
        || ' of ' || CAST(c.federating_units_total AS VARCHAR)
        || ' federating units'
                                                          AS interpretation_note
FROM by_zone z
CROSS JOIN coverage c
ORDER BY (z.zone = 'ALL ZONES') DESC, z.zone, z.lag_months;
