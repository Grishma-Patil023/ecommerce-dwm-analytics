-- 06_olap_queries.sql
-- Demonstration of core Data Warehouse Multidimensional Operations

-- 1. ROLL-UP: Aggregate Revenue by Product Category up to Yearly Granularity
SELECT
    d.year,
    p.category,
    COUNT(f.order_fact_key) AS total_orders,
    SUM(f.total_amount) AS revenue,
    ROUND(AVG(f.total_amount), 2) AS aov
FROM fact_orders f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY d.year, p.category
ORDER BY d.year, revenue DESC;

-- 2. DRILL-DOWN: Decompose Country Revenue down into Customer Age Groups & Categories
SELECT
    c.country,
    c.age_group,
    p.category,
    COUNT(f.order_fact_key) AS order_volume,
    SUM(f.total_amount) AS segment_revenue
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY c.country, c.age_group, p.category
ORDER BY c.country, segment_revenue DESC;

-- 3. SLICE: Filter for High-Value Orders (>= $500) specifically in 'Electronics'
SELECT
    f.order_id,
    c.full_name,
    c.country,
    p.product_name,
    f.quantity,
    f.total_amount,
    s.order_status
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_order_status s ON f.status_key = s.status_key
WHERE p.category = 'Electronics' AND f.total_amount >= 500
ORDER BY f.total_amount DESC;

-- 4. DICE: Restrict query to 'Apparel' and 'Toys' in 'Germany' and 'France' for 2024
SELECT
    c.country,
    p.category,
    s.order_status,
    COUNT(f.order_fact_key) AS orders_count,
    SUM(f.total_amount) AS total_revenue
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_order_status s ON f.status_key = s.status_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE p.category IN ('Apparel', 'Toys')
  AND c.country IN ('Germany', 'France')
  AND d.year = 2024
GROUP BY c.country, p.category, s.order_status;

-- 5. PIVOT (Cross-Tabulation): Category Sales across Country Columns
SELECT
    p.category,
    SUM(CASE WHEN c.country = 'USA' THEN f.total_amount ELSE 0 END) AS USA_sales,
    SUM(CASE WHEN c.country = 'UK' THEN f.total_amount ELSE 0 END) AS UK_sales,
    SUM(CASE WHEN c.country = 'Germany' THEN f.total_amount ELSE 0 END) AS Germany_sales,
    SUM(CASE WHEN c.country = 'Canada' THEN f.total_amount ELSE 0 END) AS Canada_sales,
    SUM(CASE WHEN c.country = 'India' THEN f.total_amount ELSE 0 END) AS India_sales,
    SUM(f.total_amount) AS Total_Sales
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY p.category
ORDER BY Total_Sales DESC;
