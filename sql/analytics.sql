-- AudienceIQ — Reusable Analytical Queries for BI & Stakeholders

-- 1. Top 20 Most Frequently Ordered Products & Reorder Percentages
SELECT 
    p.product_id,
    p.product_name,
    d.department,
    a.aisle,
    COUNT(*) AS total_purchases,
    SUM(op.reordered) AS total_reorders,
    ROUND(SUM(op.reordered)::numeric / COUNT(*), 4) AS reorder_rate
FROM audienceiq.order_products op
JOIN audienceiq.products p ON op.product_id = p.product_id
JOIN audienceiq.departments d ON p.department_id = d.department_id
JOIN audienceiq.aisles a ON p.aisle_id = a.aisle_id
GROUP BY p.product_id, p.product_name, d.department, a.aisle
ORDER BY total_purchases DESC
LIMIT 20;

-- 2. Peak Purchase Days and Hours
SELECT 
    order_dow,
    order_hour_of_day,
    COUNT(DISTINCT order_id) AS order_volume
FROM audienceiq.orders
GROUP BY order_dow, order_hour_of_day
ORDER BY order_dow, order_hour_of_day;

-- 3. Customer Retention & Repeat Purchase Frequency
SELECT 
    total_orders,
    COUNT(user_id) AS customer_count,
    ROUND(AVG(avg_days_between_orders), 2) AS mean_days_between_orders
FROM audienceiq.v_customer_summary
GROUP BY total_orders
ORDER BY total_orders;

