-- 03_create_dimensions.sql
-- Conformed Dimension Tables for E-Commerce Star Schema

-- Date Dimension (Conformed integer key YYYYMMDD)
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    day INTEGER NOT NULL,
    day_name VARCHAR(20) NOT NULL,
    is_weekend INTEGER NOT NULL DEFAULT 0
);

-- Customer Dimension
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    full_name VARCHAR(200),
    gender VARCHAR(20),
    age_group VARCHAR(50),
    country VARCHAR(100),
    signup_date VARCHAR(20),
    signup_date_key INTEGER
);

-- Product Dimension
CREATE TABLE IF NOT EXISTS dim_product (
    product_key INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL,
    price_tier VARCHAR(50) NOT NULL
);

-- Order Status Dimension
CREATE TABLE IF NOT EXISTS dim_order_status (
    status_key INTEGER PRIMARY KEY AUTOINCREMENT,
    order_status VARCHAR(50) UNIQUE NOT NULL,
    is_completed INTEGER DEFAULT 0
);

-- Payment Method Dimension
CREATE TABLE IF NOT EXISTS dim_payment_method (
    payment_key INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_method VARCHAR(50) UNIQUE NOT NULL,
    is_digital INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_dim_cust_id ON dim_customer(customer_id);
CREATE INDEX IF NOT EXISTS idx_dim_prod_id ON dim_product(product_id);
CREATE INDEX IF NOT EXISTS idx_dim_prod_cat ON dim_product(category);
