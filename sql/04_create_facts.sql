-- 04_create_facts.sql
-- Analytical Fact Tables for E-Commerce Star Schema

-- Fact Orders (Grain: One row per customer order line item)
CREATE TABLE IF NOT EXISTS fact_orders (
    order_fact_key INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    customer_key INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    product_key INTEGER NOT NULL REFERENCES dim_product(product_key),
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    status_key INTEGER NOT NULL REFERENCES dim_order_status(status_key),
    payment_key INTEGER NOT NULL REFERENCES dim_payment_method(payment_key),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10, 2) NOT NULL,
    total_amount NUMERIC(12, 2) NOT NULL,
    estimated_margin NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact Reviews (Grain: One row per customer product feedback)
CREATE TABLE IF NOT EXISTS fact_reviews (
    review_fact_key INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id VARCHAR(50) UNIQUE NOT NULL,
    order_id VARCHAR(50),
    customer_key INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    product_key INTEGER NOT NULL REFERENCES dim_product(product_key),
    review_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    sentiment_score NUMERIC(5, 2),
    sentiment_label VARCHAR(20),
    review_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_fact_orders_cust ON fact_orders(customer_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_prod ON fact_orders(product_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_date ON fact_orders(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_status ON fact_orders(status_key);
CREATE INDEX IF NOT EXISTS idx_fact_reviews_prod ON fact_reviews(product_key);
