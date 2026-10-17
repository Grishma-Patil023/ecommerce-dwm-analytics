-- 05_analytical_views.sql
-- Pre-aggregated analytical views for rapid BI querying

DROP VIEW IF EXISTS vw_customer_rfm_summary;
CREATE VIEW vw_customer_rfm_summary AS
SELECT
    c.customer_key,
    c.customer_id,
    c.full_name,
    c.country,
    c.age_group,
    COUNT(f.order_fact_key) AS total_orders,
    COALESCE(SUM(f.total_amount), 0) AS total_spend,
    COALESCE(AVG(f.total_amount), 0) AS average_order_value,
    COALESCE(MAX(f.date_key), 0) AS last_order_date_key
FROM dim_customer c
LEFT JOIN fact_orders f ON c.customer_key = f.customer_key
GROUP BY c.customer_key, c.customer_id, c.full_name, c.country, c.age_group;

DROP VIEW IF EXISTS vw_category_performance;
CREATE VIEW vw_category_performance AS
SELECT
    p.category,
    COUNT(DISTINCT f.order_id) AS order_count,
    SUM(f.quantity) AS total_units_sold,
    SUM(f.total_amount) AS gross_revenue,
    AVG(f.unit_price) AS avg_unit_price,
    AVG(r.rating) AS avg_customer_rating
FROM dim_product p
JOIN fact_orders f ON p.product_key = f.product_key
LEFT JOIN fact_reviews r ON p.product_key = r.product_key
GROUP BY p.category;

DROP VIEW IF EXISTS vw_monthly_executive_kpis;
CREATE VIEW vw_monthly_executive_kpis AS
SELECT
    d.year,
    d.month,
    d.month_name,
    COUNT(f.order_fact_key) AS total_transactions,
    SUM(f.total_amount) AS monthly_revenue,
    AVG(f.total_amount) AS aov,
    SUM(CASE WHEN s.order_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    ROUND(100.0 * SUM(CASE WHEN s.order_status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(f.order_fact_key), 2) AS cancellation_rate_pct
FROM fact_orders f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_order_status s ON f.status_key = s.status_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;
