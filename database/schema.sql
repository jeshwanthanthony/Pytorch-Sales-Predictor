-- Schema for the forecast warehouse.
--
-- Storage rules, applied consistently:
--   * Money is INTEGER cents. Never a float — 0.1 + 0.2 has no place in a ledger.
--   * Timestamps are TEXT, Square's RFC 3339 UTC ("2026-07-30T18:15:00Z").
--     ISO-8601 sorts and compares correctly as text, so a range scan on an
--     index works without any conversion.
--   * Dates are TEXT "YYYY-MM-DD", same reasoning.
--   * Booleans are INTEGER 0/1.
--   * Every table is STRICT, so SQLite rejects a wrong-typed value at write
--     time instead of silently storing "4521" as text in a money column.
--   * Primary keys are the upstream ids, which is what makes re-loading the
--     same raw file a no-op instead of a duplicate.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- provenance

-- Which raw files have already been loaded, so a re-run does no redundant work.
-- Keyed on the filename, which the collector already makes unique per run.
CREATE TABLE IF NOT EXISTS ingested_files (
    path        TEXT PRIMARY KEY,
    entity      TEXT NOT NULL,
    rows        INTEGER NOT NULL,
    loaded_at   TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_ingested_entity ON ingested_files (entity);

-- ------------------------------------------------------------------- square

CREATE TABLE IF NOT EXISTS locations (
    location_id     TEXT PRIMARY KEY,
    name            TEXT,
    status          TEXT,
    currency        TEXT,
    country         TEXT,
    state           TEXT,
    city            TEXT,
    postal_code     TEXT,
    address_line_1  TEXT,
    latitude        REAL,
    longitude       REAL,
    timezone        TEXT,
    business_name   TEXT,
    type            TEXT,
    created_at      TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS orders (
    order_id              TEXT PRIMARY KEY,
    location_id           TEXT,
    created_at            TEXT,
    updated_at            TEXT,
    closed_at             TEXT,
    -- Local calendar day the order belongs to, cutoff-adjusted: a 1:30am
    -- Saturday order is Friday's business. Computed once at load time so no
    -- query ever has to redo timezone maths.
    business_date         TEXT,
    business_hour         INTEGER,
    state                 TEXT,
    currency              TEXT,
    revenue_cents         INTEGER NOT NULL DEFAULT 0,
    discount_cents        INTEGER NOT NULL DEFAULT 0,
    tax_cents             INTEGER NOT NULL DEFAULT 0,
    tip_cents             INTEGER NOT NULL DEFAULT 0,
    service_charge_cents  INTEGER NOT NULL DEFAULT 0,
    net_sales_cents       INTEGER NOT NULL DEFAULT 0,
    source                TEXT,
    fulfillment_type      TEXT,
    fulfillment_state     TEXT,
    customer_id           TEXT,
    payment_types         TEXT,   -- JSON array
    tender_count          INTEGER NOT NULL DEFAULT 0,
    line_item_count       INTEGER NOT NULL DEFAULT 0,
    item_quantity         REAL NOT NULL DEFAULT 0,
    has_returns           INTEGER NOT NULL DEFAULT 0,
    ticket_name           TEXT,
    version               INTEGER,
    ingested_at           TEXT NOT NULL
) STRICT;

-- The daily rollup's covering index: it answers date-range scans per location
-- from the index alone.
CREATE INDEX IF NOT EXISTS ix_orders_date_loc
    ON orders (business_date, location_id, state, net_sales_cents);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders (customer_id)
    WHERE customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_orders_updated ON orders (updated_at);

CREATE TABLE IF NOT EXISTS order_items (
    order_id          TEXT NOT NULL,
    line_item_uid     TEXT NOT NULL,
    line_number       INTEGER,
    location_id       TEXT,
    created_at        TEXT,
    business_date     TEXT,
    catalog_object_id TEXT,
    item_name         TEXT,
    variation_name    TEXT,
    quantity          REAL NOT NULL DEFAULT 0,
    unit              TEXT,
    base_price_cents  INTEGER NOT NULL DEFAULT 0,
    gross_sales_cents INTEGER NOT NULL DEFAULT 0,
    discount_cents    INTEGER NOT NULL DEFAULT 0,
    tax_cents         INTEGER NOT NULL DEFAULT 0,
    total_cents       INTEGER NOT NULL DEFAULT 0,
    item_type         TEXT,
    note              TEXT,
    PRIMARY KEY (order_id, line_item_uid)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_items_date_name
    ON order_items (business_date, item_name, quantity);
CREATE INDEX IF NOT EXISTS ix_items_catalog ON order_items (catalog_object_id);

-- Modifiers get their own table rather than a JSON blob: "how often is Butter
-- Chicken ordered extra spicy" is then an index scan, not a full-table JSON parse.
CREATE TABLE IF NOT EXISTS order_item_modifiers (
    order_id          TEXT NOT NULL,
    line_item_uid     TEXT NOT NULL,
    modifier_uid      TEXT NOT NULL,
    catalog_object_id TEXT,
    name              TEXT,
    price_cents       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (order_id, line_item_uid, modifier_uid)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_modifiers_name ON order_item_modifiers (name);

CREATE TABLE IF NOT EXISTS payments (
    payment_id           TEXT PRIMARY KEY,
    order_id             TEXT,
    location_id          TEXT,
    customer_id          TEXT,
    created_at           TEXT,
    updated_at           TEXT,
    business_date        TEXT,
    status               TEXT,
    amount_cents         INTEGER NOT NULL DEFAULT 0,
    tip_cents            INTEGER NOT NULL DEFAULT 0,
    app_fee_cents        INTEGER NOT NULL DEFAULT 0,
    refunded_cents       INTEGER NOT NULL DEFAULT 0,
    approved_cents       INTEGER NOT NULL DEFAULT 0,
    currency             TEXT,
    processing_fee_cents INTEGER NOT NULL DEFAULT 0,
    source_type          TEXT,
    card_brand           TEXT,
    card_type            TEXT,
    last_4               TEXT,
    entry_method         TEXT,
    receipt_number       TEXT,
    team_member_id       TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS ix_payments_date ON payments (business_date, source_type);
CREATE INDEX IF NOT EXISTS ix_payments_order ON payments (order_id);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id            TEXT PRIMARY KEY,
    payment_id           TEXT,
    order_id             TEXT,
    location_id          TEXT,
    created_at           TEXT,
    updated_at           TEXT,
    business_date        TEXT,
    status               TEXT,
    amount_cents         INTEGER NOT NULL DEFAULT 0,
    currency             TEXT,
    processing_fee_cents INTEGER NOT NULL DEFAULT 0,
    reason               TEXT,
    destination_type     TEXT,
    team_member_id       TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS ix_refunds_date ON refunds (business_date);

CREATE TABLE IF NOT EXISTS customers (
    customer_id        TEXT PRIMARY KEY,
    created_at         TEXT,
    updated_at         TEXT,
    given_name         TEXT,
    family_name        TEXT,
    email              TEXT,
    phone              TEXT,
    birthday           TEXT,
    reference_id       TEXT,
    company_name       TEXT,
    creation_source    TEXT,
    group_ids          TEXT,   -- JSON array
    segment_ids        TEXT,   -- JSON array
    email_unsubscribed INTEGER,
    postal_code        TEXT,
    note               TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS catalog_items (
    catalog_object_id TEXT PRIMARY KEY,
    item_name         TEXT,
    description       TEXT,
    category_id       TEXT,
    category_name     TEXT,
    product_type      TEXT,
    is_archived       INTEGER NOT NULL DEFAULT 0,
    is_deleted        INTEGER NOT NULL DEFAULT 0,
    updated_at        TEXT,
    version           INTEGER,
    modifier_list_ids TEXT    -- JSON array
) STRICT;

CREATE INDEX IF NOT EXISTS ix_catalog_category ON catalog_items (category_name);

-- Variations are what actually carry a price and a stock count, so they are
-- rows, not JSON on the item.
CREATE TABLE IF NOT EXISTS catalog_variations (
    variation_id      TEXT PRIMARY KEY,
    catalog_object_id TEXT NOT NULL,
    name              TEXT,
    sku               TEXT,
    price_cents       INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE INDEX IF NOT EXISTS ix_variations_item ON catalog_variations (catalog_object_id);

CREATE TABLE IF NOT EXISTS inventory_counts (
    catalog_object_id   TEXT NOT NULL,
    location_id         TEXT NOT NULL,
    state               TEXT NOT NULL,
    catalog_object_type TEXT,
    quantity            REAL NOT NULL DEFAULT 0,
    calculated_at       TEXT,
    PRIMARY KEY (catalog_object_id, location_id, state)
) STRICT;

CREATE TABLE IF NOT EXISTS team_members (
    team_member_id  TEXT PRIMARY KEY,
    reference_id    TEXT,
    status          TEXT,
    given_name      TEXT,
    family_name     TEXT,
    is_owner        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT,
    updated_at      TEXT,
    assignment_type TEXT,
    location_ids    TEXT     -- JSON array
) STRICT;

CREATE TABLE IF NOT EXISTS shifts (
    shift_id            TEXT PRIMARY KEY,
    team_member_id      TEXT,
    location_id         TEXT,
    start_at            TEXT,
    end_at              TEXT,
    business_date       TEXT,
    hours               REAL NOT NULL DEFAULT 0,
    status              TEXT,
    job_id              TEXT,
    job_title           TEXT,
    hourly_rate_cents   INTEGER NOT NULL DEFAULT 0,
    labor_cost_cents    INTEGER NOT NULL DEFAULT 0,
    declared_tips_cents INTEGER NOT NULL DEFAULT 0,
    timezone            TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS ix_shifts_date ON shifts (business_date, location_id);

-- ----------------------------------------------------------------- external

CREATE TABLE IF NOT EXISTS weather (
    date                TEXT PRIMARY KEY,
    latitude            REAL,
    longitude           REAL,
    source              TEXT,   -- archive | forecast
    temp_max_f          REAL,
    temp_min_f          REAL,
    temp_mean_f         REAL,
    feels_like_max_f    REAL,
    feels_like_min_f    REAL,
    precipitation_in    REAL,
    rain_in             REAL,
    snowfall_in         REAL,
    precipitation_hours REAL,
    wind_max_mph        REAL,
    wind_gust_mph       REAL,
    humidity_mean       REAL,
    weather_code        INTEGER,
    sunrise             TEXT,
    sunset              TEXT,
    is_rainy            INTEGER NOT NULL DEFAULT 0,
    is_snowy            INTEGER NOT NULL DEFAULT 0,
    is_stormy           INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE TABLE IF NOT EXISTS calendar_days (
    date                 TEXT PRIMARY KEY,
    day_of_week          INTEGER NOT NULL,
    day_name             TEXT,
    month                INTEGER NOT NULL,
    month_name           TEXT,
    year                 INTEGER NOT NULL,
    day_of_month         INTEGER NOT NULL,
    day_of_year          INTEGER NOT NULL,
    week_of_year         INTEGER NOT NULL,
    quarter              INTEGER NOT NULL,
    is_weekend           INTEGER NOT NULL DEFAULT 0,
    is_weekend_night     INTEGER NOT NULL DEFAULT 0,
    is_holiday           INTEGER NOT NULL DEFAULT 0,
    holiday_name         TEXT,
    is_holiday_eve       INTEGER NOT NULL DEFAULT 0,
    is_day_after_holiday INTEGER NOT NULL DEFAULT 0,
    observance           TEXT,
    is_observance        INTEGER NOT NULL DEFAULT 0,
    school_break         TEXT,
    is_school_break      INTEGER NOT NULL DEFAULT 0,
    is_month_start       INTEGER NOT NULL DEFAULT 0,
    is_month_end         INTEGER NOT NULL DEFAULT 0,
    is_payday_window     INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE TABLE IF NOT EXISTS events (
    date                TEXT NOT NULL,
    name                TEXT NOT NULL,
    category            TEXT,
    venue               TEXT,
    distance_miles      REAL,
    expected_attendance INTEGER,
    start_time          TEXT,
    is_multi_day        INTEGER NOT NULL DEFAULT 0,
    day_index           INTEGER NOT NULL DEFAULT 0,
    notes               TEXT,
    PRIMARY KEY (date, name)
) STRICT;

CREATE TABLE IF NOT EXISTS promotions (
    date           TEXT NOT NULL,
    name           TEXT NOT NULL,
    channel        TEXT,
    discount_type  TEXT,
    discount_value REAL,
    spend_usd      REAL,
    applies_to     TEXT,
    day_index      INTEGER NOT NULL DEFAULT 0,
    notes          TEXT,
    PRIMARY KEY (date, name)
) STRICT;

-- -------------------------------------------------------------------- views
--
-- Views, not materialized tables: SQLite runs these against the indexes above
-- fast enough that a stale copy would be the bigger problem.

-- Orders rolled up to one row per business day per location.
-- Canceled orders are excluded — they were never revenue.
DROP VIEW IF EXISTS daily_sales;
CREATE VIEW daily_sales AS
SELECT
    business_date,
    location_id,
    COUNT(*)                                  AS order_count,
    SUM(revenue_cents)                        AS revenue_cents,
    SUM(net_sales_cents)                      AS net_sales_cents,
    SUM(discount_cents)                       AS discount_cents,
    SUM(tax_cents)                            AS tax_cents,
    SUM(tip_cents)                            AS tip_cents,
    SUM(service_charge_cents)                 AS service_charge_cents,
    SUM(item_quantity)                        AS item_count,
    COUNT(DISTINCT customer_id)               AS known_customer_count,
    CAST(AVG(revenue_cents) AS INTEGER)       AS avg_ticket_cents,
    SUM(CASE WHEN fulfillment_type = 'PICKUP'   THEN 1 ELSE 0 END) AS pickup_orders,
    SUM(CASE WHEN fulfillment_type = 'DELIVERY' THEN 1 ELSE 0 END) AS delivery_orders,
    SUM(CASE WHEN fulfillment_type IS NULL      THEN 1 ELSE 0 END) AS dine_in_orders
FROM orders
WHERE state <> 'CANCELED' AND business_date IS NOT NULL
GROUP BY business_date, location_id;

-- Per-item daily demand: the target for a future menu-prep model.
DROP VIEW IF EXISTS daily_item_sales;
CREATE VIEW daily_item_sales AS
SELECT
    i.business_date,
    i.location_id,
    i.item_name,
    c.category_name,
    SUM(i.quantity)     AS quantity,
    SUM(i.total_cents)  AS revenue_cents,
    COUNT(*)            AS times_ordered
FROM order_items i
LEFT JOIN catalog_variations v ON v.variation_id = i.catalog_object_id
LEFT JOIN catalog_items      c ON c.catalog_object_id = v.catalog_object_id
WHERE i.business_date IS NOT NULL
GROUP BY i.business_date, i.location_id, i.item_name, c.category_name;

DROP VIEW IF EXISTS daily_labor;
CREATE VIEW daily_labor AS
SELECT
    business_date,
    location_id,
    COUNT(DISTINCT team_member_id) AS staff_count,
    SUM(hours)                     AS labor_hours,
    SUM(labor_cost_cents)          AS labor_cost_cents
FROM shifts
WHERE business_date IS NOT NULL
GROUP BY business_date, location_id;

DROP VIEW IF EXISTS daily_payments;
CREATE VIEW daily_payments AS
SELECT
    business_date,
    location_id,
    SUM(CASE WHEN source_type = 'CASH'      THEN amount_cents ELSE 0 END) AS cash_cents,
    SUM(CASE WHEN source_type = 'CARD'      THEN amount_cents ELSE 0 END) AS card_cents,
    SUM(CASE WHEN source_type = 'GIFT_CARD' THEN amount_cents ELSE 0 END) AS gift_card_cents,
    SUM(CASE WHEN card_brand = 'VISA'             THEN amount_cents ELSE 0 END) AS visa_cents,
    SUM(CASE WHEN card_brand = 'MASTERCARD'       THEN amount_cents ELSE 0 END) AS mastercard_cents,
    SUM(CASE WHEN card_brand = 'AMERICAN_EXPRESS' THEN amount_cents ELSE 0 END) AS amex_cents,
    SUM(processing_fee_cents) AS processing_fee_cents
FROM payments
WHERE status = 'COMPLETED' AND business_date IS NOT NULL
GROUP BY business_date, location_id;

DROP VIEW IF EXISTS daily_refunds;
CREATE VIEW daily_refunds AS
SELECT
    business_date,
    location_id,
    COUNT(*)           AS refund_count,
    SUM(amount_cents)  AS refund_cents
FROM refunds
WHERE business_date IS NOT NULL
GROUP BY business_date, location_id;

DROP VIEW IF EXISTS daily_events;
CREATE VIEW daily_events AS
SELECT
    date,
    COUNT(*)                       AS event_count,
    SUM(COALESCE(expected_attendance, 0)) AS event_attendance,
    MIN(distance_miles)            AS nearest_event_miles,
    SUM(CASE WHEN category = 'sports'   THEN 1 ELSE 0 END) AS sports_events,
    SUM(CASE WHEN category = 'concert'  THEN 1 ELSE 0 END) AS concert_events,
    SUM(CASE WHEN category = 'festival' THEN 1 ELSE 0 END) AS festival_events
FROM events
GROUP BY date;

DROP VIEW IF EXISTS daily_promotions;
CREATE VIEW daily_promotions AS
SELECT
    date,
    COUNT(*)                    AS promotion_count,
    SUM(COALESCE(spend_usd, 0)) AS promotion_spend_usd,
    MAX(CASE WHEN discount_type = 'percent' THEN discount_value ELSE 0 END) AS max_percent_off,
    MAX(CASE WHEN channel = 'facebook_ads' THEN 1 ELSE 0 END) AS has_facebook_ads,
    MAX(CASE WHEN channel = 'ubereats'     THEN 1 ELSE 0 END) AS has_delivery_promo,
    MAX(CASE WHEN channel = 'email'        THEN 1 ELSE 0 END) AS has_email_promo,
    MAX(CASE WHEN channel = 'in_store'     THEN 1 ELSE 0 END) AS has_in_store_promo
FROM promotions
GROUP BY date;

-- The training table. One row per day per location, every column numeric or
-- trivially encodable, lags and rolling means computed in SQL.
--
-- The spine is calendar_days, not orders: a day the restaurant was closed must
-- still appear as a zero, and future dates must appear so tomorrow can be
-- predicted. `is_observed` separates "closed, sold nothing" from "hasn't
-- happened yet" — training filters on it, inference selects the rows without it.
DROP VIEW IF EXISTS daily_forecast_features;
CREATE VIEW daily_forecast_features AS
WITH spine AS (
    SELECT c.date AS business_date, l.location_id
    FROM calendar_days c
    CROSS JOIN locations l
    WHERE l.status = 'ACTIVE' OR l.status IS NULL
),
last_observed AS (
    SELECT MAX(business_date) AS d FROM daily_sales
),
joined AS (
    SELECT
        s.business_date,
        s.location_id,
        (s.business_date <= (SELECT d FROM last_observed)) AS is_observed,

        cal.day_of_week, cal.month, cal.year, cal.day_of_month, cal.week_of_year,
        cal.quarter, cal.is_weekend, cal.is_weekend_night, cal.is_holiday,
        cal.is_holiday_eve, cal.is_day_after_holiday, cal.is_observance,
        cal.is_school_break, cal.is_month_start, cal.is_month_end,
        cal.is_payday_window, cal.holiday_name, cal.day_name,

        w.temp_max_f, w.temp_min_f, w.temp_mean_f, w.feels_like_max_f,
        w.precipitation_in, w.snowfall_in, w.precipitation_hours,
        w.wind_max_mph, w.humidity_mean, w.is_rainy, w.is_snowy, w.is_stormy,
        w.source AS weather_source,

        COALESCE(ev.event_count, 0)           AS event_count,
        COALESCE(ev.event_attendance, 0)      AS event_attendance,
        ev.nearest_event_miles,
        COALESCE(ev.sports_events, 0)         AS sports_events,

        COALESCE(pr.promotion_count, 0)       AS promotion_count,
        COALESCE(pr.promotion_spend_usd, 0)   AS promotion_spend_usd,
        COALESCE(pr.max_percent_off, 0)       AS max_percent_off,
        COALESCE(pr.has_facebook_ads, 0)      AS has_facebook_ads,
        COALESCE(pr.has_delivery_promo, 0)    AS has_delivery_promo,
        (COALESCE(pr.promotion_count, 0) > 0) AS promotion_active,

        COALESCE(ds.order_count, 0)           AS order_count,
        COALESCE(ds.item_count, 0)            AS item_count,
        COALESCE(ds.known_customer_count, 0)  AS customer_count,
        COALESCE(ds.avg_ticket_cents, 0)      AS avg_ticket_cents,
        COALESCE(ds.pickup_orders, 0)         AS pickup_orders,
        COALESCE(ds.delivery_orders, 0)       AS delivery_orders,
        COALESCE(ds.tip_cents, 0)             AS tip_cents,
        COALESCE(ds.discount_cents, 0)        AS discount_cents,
        COALESCE(rf.refund_cents, 0)          AS refund_cents,
        COALESCE(lb.labor_hours, 0)           AS labor_hours,
        COALESCE(lb.labor_cost_cents, 0)      AS labor_cost_cents,
        COALESCE(lb.staff_count, 0)           AS staff_count,
        COALESCE(pm.cash_cents, 0)            AS cash_cents,
        COALESCE(pm.card_cents, 0)            AS card_cents,

        -- The target. NULL on future dates, 0 on a day that was simply closed.
        CASE
            WHEN s.business_date <= (SELECT d FROM last_observed)
            THEN COALESCE(ds.net_sales_cents, 0)
        END AS target_sales_cents
    FROM spine s
    JOIN calendar_days cal    ON cal.date = s.business_date
    LEFT JOIN weather w       ON w.date = s.business_date
    LEFT JOIN daily_events ev ON ev.date = s.business_date
    LEFT JOIN daily_promotions pr ON pr.date = s.business_date
    LEFT JOIN daily_sales ds  ON ds.business_date = s.business_date AND ds.location_id = s.location_id
    LEFT JOIN daily_refunds rf ON rf.business_date = s.business_date AND rf.location_id = s.location_id
    LEFT JOIN daily_labor lb  ON lb.business_date = s.business_date AND lb.location_id = s.location_id
    LEFT JOIN daily_payments pm ON pm.business_date = s.business_date AND pm.location_id = s.location_id
)
SELECT
    *,
    -- History features. Every one looks strictly backwards (the frame ends at
    -- 1 PRECEDING), so nothing here can leak the day's own answer into training.
    LAG(target_sales_cents, 1) OVER win AS sales_lag_1_cents,
    LAG(target_sales_cents, 2) OVER win AS sales_lag_2_cents,
    LAG(target_sales_cents, 7) OVER win AS sales_lag_7_cents,
    LAG(target_sales_cents, 14) OVER win AS sales_lag_14_cents,
    LAG(order_count, 1) OVER win        AS orders_lag_1,
    LAG(customer_count, 1) OVER win     AS customers_lag_1,
    CAST(AVG(target_sales_cents) OVER (
        PARTITION BY location_id ORDER BY business_date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS INTEGER) AS sales_avg_7_cents,
    CAST(AVG(target_sales_cents) OVER (
        PARTITION BY location_id ORDER BY business_date
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) AS INTEGER) AS sales_avg_30_cents,
    CAST(AVG(order_count) OVER (
        PARTITION BY location_id ORDER BY business_date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS INTEGER) AS orders_avg_7
FROM joined
WINDOW win AS (PARTITION BY location_id ORDER BY business_date);
