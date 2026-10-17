"""
ml/db_loader.py — Loads star schema tables into clean pandas DataFrames for ML modules.
"""
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from etl.db import query_df, execute_query

def load_orders_ml_df() -> pd.DataFrame:
    """Loads conformed orders joined with customer and product dimensions."""
    sql = """
    SELECT
        f.order_id,
        f.customer_key,
        c.customer_id,
        c.gender,
        c.age_group,
        c.country,
        f.product_key,
        p.product_id,
        p.product_name,
        p.category,
        p.price_tier,
        f.quantity,
        f.unit_price,
        f.total_amount,
        f.estimated_margin,
        s.order_status,
        m.payment_method,
        d.year,
        d.month,
        d.day_name,
        d.is_weekend,
        r.rating,
        r.sentiment_score
    FROM fact_orders f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_product p ON f.product_key = p.product_key
    JOIN dim_order_status s ON f.status_key = s.status_key
    JOIN dim_payment_method m ON f.payment_key = m.payment_key
    JOIN dim_date d ON f.date_key = d.date_key
    LEFT JOIN fact_reviews r ON f.customer_key = r.customer_key AND f.product_key = r.product_key;
    """
    return query_df(sql)

def load_rfm_df() -> pd.DataFrame:
    """Loads customer aggregated metrics for clustering and RFM."""
    sql = """
    SELECT
        c.customer_id,
        c.country,
        c.age_group,
        c.gender,
        COUNT(f.order_fact_key) AS frequency,
        SUM(f.total_amount) AS monetary,
        AVG(f.total_amount) AS aov,
        AVG(f.quantity) AS avg_units_per_order,
        MAX(f.date_key) AS last_order_date_key
    FROM dim_customer c
    JOIN fact_orders f ON c.customer_key = f.customer_key
    GROUP BY c.customer_id, c.country, c.age_group, c.gender;
    """
    return query_df(sql)
