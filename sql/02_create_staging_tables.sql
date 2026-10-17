-- 02_create_staging_tables.sql
-- Raw staging table replicating 10,000 transactions directly from source CSV

CREATE TABLE IF NOT EXISTS stg_ecommerce_orders (
    stg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    gender VARCHAR(20),
    age_group VARCHAR(50),
    signup_date VARCHAR(20),
    country VARCHAR(100),
    product_id VARCHAR(50),
    product_name VARCHAR(200),
    category VARCHAR(100),
    quantity INTEGER,
    unit_price NUMERIC(10, 2),
    order_id VARCHAR(50),
    order_date VARCHAR(20),
    order_status VARCHAR(50),
    payment_method VARCHAR(50),
    rating INTEGER,
    review_text TEXT,
    review_id VARCHAR(50),
    review_date VARCHAR(20),
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stg_order_id ON stg_ecommerce_orders(order_id);
CREATE INDEX IF NOT EXISTS idx_stg_customer_id ON stg_ecommerce_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_stg_product_id ON stg_ecommerce_orders(product_id);
