-- ===========================================================================
-- A7  subsidy_structural_break          WITHIN-MARKET ANALYSIS -- NOT NATIONAL
--
-- Question   : Within a fixed set of markets that report continuously on both
--              sides of the petrol subsidy removal announced on 29 May 2023,
--              did staple food prices change in level or trend?
-- Scope      : a FIXED PANEL of 10 markets in 2 states -- Borno (6: Abba
--              Gamaram, Baga Road, Budum, Bullunkutu, Custom, Monday) and Yobe
--              (4: Geidam, Gujba (Buni Yadi), Yunusari, Yusufari). All in the
--              North East. These counts are asserted against the query's own
--              output by tests/test_a7_scope.py, so they cannot drift.
-- Grain      : one row per month in the analysis window, carrying the staple
--              index, its segment, that segment's fitted trend, and the panel
--              and within-cell summaries.
-- Output     : outputs/tables/A7_subsidy_structural_break.csv
-- Figure     : A7_subsidy_structural_break.png (break date marked)
--
-- WHY THIS IS NOT A NATIONAL BEFORE-AND-AFTER
--   WFP's Nigerian retail panel changes composition at almost exactly the
--   break: 14 states report up to January 2023, only Borno, Yobe and Adamawa
--   afterwards. A before-and-after over the whole panel would compare a
--   14-state panel with a 3-state one. This version compares the SAME markets
--   with themselves, and nothing else. It says nothing about Nigeria.
--
-- WHY THE WINDOW IS {window} MONTHS EACH SIDE
--   "Continuously" means: at least one retail staple observation in EVERY month
--   of the window. North-eastern reporting has a panel-wide hole from June 2022
--   to January 2023, so no market is continuous over any window reaching back
--   before February 2023. February to September 2023 -- {window} months either
--   side -- is the longest symmetric window in which a continuous panel exists.
--   Pre-break: t = -{window}+1 .. 0 (May 2023, the announcement month, is the
--   last pre-break month). Post-break: t = 1 .. {window}.
--
-- NO CAUSAL CLAIM IS MADE OR IMPLIED.
--   The window also spans the June 2023 naira devaluation, the lean season in
--   the north, and ongoing insecurity affecting these specific markets. The
--   analysis reports association in time only. Fuel prices cannot be used as
--   the explanatory series: NBS petrol data begins 2023-11, after this window
--   closes, and the WFP fuel series has a 2023 coverage gap.
--
-- Method:
--   1. Fixed panel: markets with a retail staple observation in every one of
--      the 2 x {window} months. Panel membership is computed, not listed by hand.
--   2. A chained matched-model (Jevons) index over the panel's cells, with the
--      same linking rules as A6 (relative to the cell's own prior observation up
--      to {max_link_gap} months old; a link on fewer than {min_matched_cells}
--      matched cells breaks the chain; only the segment containing the break
--      month is used). Rebased to 100 at the break month.
--   3. OLS trend per segment. With {window} points per segment the slopes are
--      FRAGILE; n and R-squared are published so the reader can see it.
--   4. The more robust number: a WITHIN-CELL LEVEL CHANGE. For every
--      (market, commodity+unit) cell observed on both sides, the difference
--      between its mean log price after and before the break; the geometric
--      mean across cells, expressed in per cent. Each cell is its own control.
-- ===========================================================================
WITH window_months AS (
    SELECT range AS month_index
    FROM range({break_month_index} - {window} + 1, {break_month_index} + {window} + 1)
),
staple_obs AS (
    SELECT
        f.geo_key,
        f.commodity_key,
        f.price_type,
        (f.month_key // 10000) * 12 + ((f.month_key // 100) % 100)   AS month_index,
        f.price_ngn
    FROM fact_market_price_monthly f
    JOIN dim_commodity c ON c.commodity_key = f.commodity_key
    WHERE c.category IN {staple_categories}
      AND NOT c.is_fuel
      AND f.price_type = '{price_type}'
      AND f.price_ngn > 0
      AND f.geo_key <> -1
),
-- ---- THE FIXED PANEL: markets reporting in every month of the window -------
market_coverage AS (
    SELECT o.geo_key, count(DISTINCT o.month_index) AS months_reported
    FROM staple_obs o
    JOIN window_months w ON w.month_index = o.month_index
    GROUP BY o.geo_key
),
fixed_panel AS (
    SELECT mc.geo_key, g.geo_name AS market, g.parent_geo_name AS state
    FROM market_coverage mc
    JOIN dim_geography g ON g.geo_key = mc.geo_key
    WHERE mc.months_reported = 2 * {window}
),
panel_summary AS (
    SELECT
        count(*)                                                    AS panel_markets,
        count(DISTINCT state)                                       AS panel_states,
        array_to_string(list_sort(list_distinct(list(state))), '; ') AS panel_state_list,
        array_to_string(list_sort(list(state || ': ' || market)), '; ')
                                                                    AS panel_market_list
    FROM fixed_panel
),
priced AS (
    SELECT o.*
    FROM staple_obs o
    JOIN fixed_panel p ON p.geo_key = o.geo_key
    WHERE o.month_index BETWEEN {break_month_index} - {window} + 1
                            AND {break_month_index} + {window}
),
-- ---- the within-cell level change: each cell is its own control -----------
cell_levels AS (
    SELECT
        geo_key, commodity_key,
        avg(ln(price_ngn)) FILTER (WHERE month_index <= {break_month_index}) AS pre_log,
        avg(ln(price_ngn)) FILTER (WHERE month_index >  {break_month_index}) AS post_log
    FROM priced
    GROUP BY 1, 2
),
level_change AS (
    SELECT
        count(*) FILTER (WHERE pre_log IS NOT NULL AND post_log IS NOT NULL)
                                                        AS cells_on_both_sides,
        100.0 * (exp(avg(post_log - pre_log)) - 1.0)    AS within_cell_level_change_pct,
        100.0 * (exp(quantile_cont(post_log - pre_log, 0.25)) - 1.0)
                                                        AS within_cell_change_p25_pct,
        100.0 * (exp(quantile_cont(post_log - pre_log, 0.75)) - 1.0)
                                                        AS within_cell_change_p75_pct
    FROM cell_levels
),
-- ---- the chained index over the fixed panel --------------------------------
relatives AS (
    SELECT
        month_index,
        geo_key,
        price_ngn,
        lag(price_ngn)   OVER w AS prev_price,
        lag(month_index) OVER w AS prev_month_index
    FROM priced
    WINDOW w AS (PARTITION BY geo_key, commodity_key, price_type
                 ORDER BY month_index)
),
links AS (
    SELECT
        month_index,
        count(*) FILTER (WHERE prev_price > 0
                           AND prev_month_index >= month_index - {max_link_gap})
                                                                   AS matched_cells,
        exp(avg(ln(price_ngn / prev_price))
            FILTER (WHERE prev_price > 0
                      AND prev_month_index >= month_index - {max_link_gap}))
                                                                   AS link_relative,
        count(*)                                                   AS cells_reporting,
        count(DISTINCT geo_key)                                    AS markets_reporting
    FROM relatives
    GROUP BY ALL
),
chained AS (
    SELECT
        links.*,
        (matched_cells < {min_matched_cells})                      AS is_chain_break,
        sum(CASE WHEN matched_cells < {min_matched_cells} THEN 1 ELSE 0 END)
            OVER (ORDER BY month_index
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS chain_segment
    FROM links
),
levelled AS (
    SELECT
        chained.*,
        exp(sum(CASE WHEN is_chain_break THEN 0.0 ELSE ln(link_relative) END)
            OVER (PARTITION BY chain_segment ORDER BY month_index
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS chain_level
    FROM chained
),
anchored AS (
    SELECT
        levelled.*,
        month_index - {break_month_index}                          AS t,
        max(chain_level)   FILTER (WHERE month_index = {break_month_index}) OVER ()
                                                                   AS break_chain_level,
        max(chain_segment) FILTER (WHERE month_index = {break_month_index}) OVER ()
                                                                   AS break_segment
    FROM levelled
),
windowed AS (
    SELECT
        printf('%04d-%02d', (month_index - 1) // 12, ((month_index - 1) % 12) + 1)
                                                                   AS year_month,
        t,
        matched_cells,
        cells_reporting,
        markets_reporting,
        round(100.0 * chain_level / nullif(break_chain_level, 0), 6) AS staple_index,
        CASE WHEN t <= 0 THEN 'pre' ELSE 'post' END                  AS segment
    FROM anchored
    WHERE break_chain_level IS NOT NULL
      AND chain_segment = break_segment
),
fitted AS (
    SELECT
        w.*,
        regr_slope(staple_index, t)     OVER s  AS segment_slope_per_month,
        regr_intercept(staple_index, t) OVER s  AS segment_value_at_break,
        regr_r2(staple_index, t)        OVER s  AS segment_r2,
        count(*)                        OVER s  AS segment_n_months,
        min(year_month)                 OVER s  AS segment_first_month,
        max(year_month)                 OVER s  AS segment_last_month
    FROM windowed w
    WINDOW s AS (PARTITION BY segment)
)
SELECT
    'within-market fixed panel; NOT a national comparison'     AS analysis_scope,
    f.year_month,
    f.t                                                        AS months_since_break,
    f.segment,
    f.markets_reporting,
    f.matched_cells,
    f.cells_reporting,
    round(f.staple_index, 4)                                   AS staple_index,
    round(f.segment_slope_per_month, 6)                        AS segment_slope_per_month,
    round(f.segment_value_at_break, 6)                         AS segment_fitted_at_break,
    round(f.segment_r2, 6)                                     AS segment_r2,
    f.segment_n_months,
    f.segment_first_month,
    f.segment_last_month,
    round(f.segment_value_at_break + f.segment_slope_per_month * f.t, 6)
                                                               AS fitted_index,
    -- Segment contrasts, repeated on every row so the CSV is self-contained.
    round(max(f.segment_slope_per_month) FILTER (WHERE f.segment = 'post') OVER ()
        - max(f.segment_slope_per_month) FILTER (WHERE f.segment = 'pre')  OVER (), 6)
                                                               AS slope_change_per_month,
    round(max(f.segment_value_at_break) FILTER (WHERE f.segment = 'post') OVER ()
        - max(f.segment_value_at_break) FILTER (WHERE f.segment = 'pre')  OVER (), 6)
                                                               AS level_shift_at_break,
    -- The robust summary: same cells, after versus before.
    round(lc.within_cell_level_change_pct, 4)                  AS within_cell_level_change_pct,
    round(lc.within_cell_change_p25_pct, 4)                    AS within_cell_change_p25_pct,
    round(lc.within_cell_change_p75_pct, 4)                    AS within_cell_change_p75_pct,
    lc.cells_on_both_sides,
    -- The panel, stated on every row.
    ps.panel_markets,
    ps.panel_states,
    ps.panel_state_list,
    ps.panel_market_list,
    {window}                                                   AS window_months_each_side,
    '{break_year_month}'                                       AS break_month,
    '{break_date}'                                             AS break_event_date,
    'association only; no causal claim'                        AS interpretation_note
FROM fitted f
CROSS JOIN panel_summary ps
CROSS JOIN level_change lc
ORDER BY f.t;
