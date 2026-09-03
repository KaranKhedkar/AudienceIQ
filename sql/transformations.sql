-- AudienceIQ — SQL-Side Transformation Views
-- Intermediate and aggregated relational structures for analytics

-- 1. Customer Order Summary View
CREATE OR REPLACE VIEW audienceiq.v_customer_summary AS
SELECT 
    user_id,
    COUNT(DISTINCT order_id) AS total_orders,
    MIN(order_number) AS first_order_number,
    MAX(order_number) AS max_order_number,
    AVG(days_since_prior_order) AS avg_days_between_orders,
    AVG(order_hour_of_day) AS avg_order_hour
FROM audienceiq.orders
GROUP BY user_id;

-- 2. Order Basket Metrics View
CREATE OR REPLACE VIEW audienceiq.v_order_baskets AS
SELECT 
    o.order_id,
    o.user_id,
    o.order_number,
    o.order_dow,
    o.order_hour_of_day,
    o.days_since_prior_order,
    COUNT(op.product_id) AS basket_size,
    SUM(op.reordered) AS reordered_count,
    ROUND(SUM(op.reordered)::NUMERIC / NULLIF(COUNT(op.product_id), 0), 4) AS reorder_ratio
FROM audienceiq.orders o
JOIN audienceiq.order_products op ON o.order_id = op.order_id
GROUP BY o.order_id, o.user_id, o.order_number, o.order_dow, o.order_hour_of_day, o.days_since_prior_order;
