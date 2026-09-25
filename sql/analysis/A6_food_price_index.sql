-- ===========================================================================
-- A6  food_price_index
--
-- Question   : How have Nigerian food prices moved by category since January
--              2016, once the changing composition of the reporting panel is
--              held constant?
-- Grain      : one row per (category, month in which the category reports).
-- Output     : outputs/tables/A6_food_price_index.csv
-- Figure     : A6_food_price_index.png (chained index vs naive mean)
--
-- THE PROBLEM THIS QUERY EXISTS TO SOLVE
--   The WFP panel is UNBALANCED: the number of markets reporting in a month
--   varies substantially across the series. A national AVG(price) therefore
--   moves when the panel composition moves, even if no price changed anywhere.
--   If expensive urban markets enter the panel one month, the naive mean rises;
--   that is a measurement artefact, not inflation.
--
-- METHOD: CHAINED MATCHED-MODEL (JEVONS) INDEX
--   A true fixed basket anchored to January 2016 is not constructible on this
--   panel: very few (market, commodity, unit, price type) cells report
--   continuously from 2016 to 2026, so a fixed basket would discard most of the
--   data and its remaining cells would not be representative.
--
--   Instead, for each month the query forms a price relative for every cell
--   against that cell's own most recent OBSERVED price -- provided that
--   observation is at most {max_link_gap} months old -- takes the unweighted
--   geometric mean of those relatives (the Jevons formula, standard in
--   official statistics for elementary aggregates without quantity weights),
--   and chains the resulting links into a continuous series rebased so that
--   {base_year_month} = 100. Composition never enters a single link.
--
-- CHAIN BREAKS -- NOTHING IS CARRIED ACROSS ONE
--   A month whose link rests on fewer than {min_matched_cells} matched cells is
--   a CHAIN BREAK, not a price change. The chain restarts there as a new
--   segment. chained_index is published ONLY for the segment that links
--   unbroken to the base month; elsewhere it is NULL, and chain_segment /
--   is_linked_to_base say why.
--
--   An earlier draft of this query instead treated a missing link as a
--   relative of 1.0 -- "no change" -- which is carry-forward interpolation. It
--   produced a perfectly flat eleven-year stretch in one category and was
--   removed. On the live panel the rule above leaves ONLY cereals and tubers
--   linked unbroken from January 2016 to the latest month. Pulses and nuts,
--   and oil and fats, link to January 2023 and break when the WFP panel
--   contracts to the North East. Meat/fish/eggs, milk and dairy, vegetables
--   and fruits, and miscellaneous food do not report continuously in the
--   months after January 2016 and cannot be linked to the base at all; their
--   rows remain in the output with is_linked_to_base = false.
--
-- Assumptions and limits, all carried into docs/methodology_notes.md:
--   1. A "cell" is (market, commodity+unit, price type). Unit is part of the
--      identity: 1 KG of rice and 100 KG of rice are different products.
--   2. A bridged relative (a cell whose previous observation is two or three
--      months old) is attributed to the month of the new observation. The chain
--      LEVEL is exact; the single-month change column is then a change since the
--      prior observation, and longest_bridge_months shows when that happens.
--   3. The index is UNWEIGHTED. WFP publishes no consumption quantities, so the
--      index describes price behaviour in the sampled markets, not the cost of a
--      representative household's basket.
--   4. Fuel commodities are EXCLUDED (dim_commodity.is_fuel).
--   5. matched_cells is published on every row; a thin link is visible as such.
--   6. naive_mean_index is computed on the same rows by the same rebasing, so
--      the divergence between the two is attributable solely to the
--      composition correction.
-- ===========================================================================
WITH priced AS (
    SELECT
        c.category,
        f.geo_key,
        f.commodity_key,
        f.price_type,
        -- Months as a dense integer so "consecutive" is expressible.
        (f.month_key // 10000) * 12 + ((f.month_key // 100) % 100)  AS month_index,
        printf('%04d-%02d', f.month_key // 10000,
               (f.month_key // 100) % 100)                          AS year_month,
        f.price_ngn
    FROM fact_market_price_monthly f
    JOIN dim_commodity c ON c.commodity_key = f.commodity_key
    WHERE NOT c.is_fuel
      AND f.price_ngn > 0
      AND f.geo_key <> -1
),
relatives AS (
    SELECT
        category,
        month_index,
        year_month,
        price_ngn,
        lag(price_ngn)   OVER w  AS prev_price,
        lag(month_index) OVER w  AS prev_month_index
    FROM priced
    WINDOW w AS (
        PARTITION BY geo_key, commodity_key, price_type
        ORDER BY month_index
    )
),
links AS (
    SELECT
        category,
        month_index,
        year_month,
        count(*) FILTER (
            WHERE prev_price > 0
              AND prev_month_index >= month_index - {max_link_gap}
        )                                                     AS matched_cells,
        -- Jevons link: geometric mean of matched, OBSERVED price relatives.
        exp(avg(ln(price_ngn / prev_price)) FILTER (
            WHERE prev_price > 0
              AND prev_month_index >= month_index - {max_link_gap}
        ))                                                    AS link_relative,
        max(month_index - prev_month_index) FILTER (
            WHERE prev_price > 0
              AND prev_month_index >= month_index - {max_link_gap}
        )                                                     AS longest_bridge_months,
        count(*)                                              AS cells_reporting,
        avg(price_ngn)                                        AS naive_mean_price
    FROM relatives
    GROUP BY ALL
),
segmented AS (
    SELECT
        links.*,
        (matched_cells < {min_matched_cells})                 AS is_chain_break,
        -- Running count of breaks: every row between two breaks shares a
        -- segment, and a chain may only be read within one segment.
        sum(CASE WHEN matched_cells < {min_matched_cells} THEN 1 ELSE 0 END)
            OVER (PARTITION BY category ORDER BY month_index
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                              AS chain_segment
    FROM links
),
chained AS (
    SELECT
        segmented.*,
        -- The break row anchors its segment at 1.0; every later link in the
        -- segment multiplies onto it. No link is ever assumed.
        exp(sum(CASE WHEN is_chain_break THEN 0.0 ELSE ln(link_relative) END)
            OVER (PARTITION BY category, chain_segment ORDER BY month_index
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW))
                                                              AS chain_level
    FROM segmented
),
based AS (
    SELECT
        chained.*,
        max(chain_segment)    FILTER (WHERE year_month = '{base_year_month}')
            OVER (PARTITION BY category)                      AS base_segment,
        max(chain_level)      FILTER (WHERE year_month = '{base_year_month}')
            OVER (PARTITION BY category)                      AS base_chain_level,
        max(naive_mean_price) FILTER (WHERE year_month = '{base_year_month}')
            OVER (PARTITION BY category)                      AS base_naive_price
    FROM chained
)
SELECT
    category,
    year_month,
    matched_cells,
    cells_reporting,
    longest_bridge_months,
    is_chain_break,
    chain_segment,
    coalesce(chain_segment = base_segment, FALSE)             AS is_linked_to_base,
    CASE WHEN chain_segment = base_segment AND NOT is_chain_break
         THEN round(link_relative, 8) END                     AS monthly_link_relative,
    CASE WHEN chain_segment = base_segment AND NOT is_chain_break
         THEN round(100.0 * (link_relative - 1.0), 6) END     AS pct_change_since_prior_observation,
    -- The composition-controlled series, published only where it links
    -- unbroken to the base month.
    CASE WHEN chain_segment = base_segment
         THEN round(100.0 * chain_level / nullif(base_chain_level, 0), 4)
    END                                                       AS chained_index,
    -- The naive series, shown precisely so the divergence can be reported.
    round(100.0 * naive_mean_price / nullif(base_naive_price, 0), 4)
                                                              AS naive_mean_index,
    round(naive_mean_price, 4)                                AS naive_mean_price_ngn,
    CASE WHEN chain_segment = base_segment
         THEN round(100.0 * naive_mean_price / nullif(base_naive_price, 0)
                  - 100.0 * chain_level / nullif(base_chain_level, 0), 4)
    END                                                       AS naive_minus_chained,
    '{base_year_month}'                                       AS index_base_period
FROM based
ORDER BY category, year_month;
