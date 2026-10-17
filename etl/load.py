"""
etl/load.py — Idempotent loading of staging, dimensions, and facts.
"""
from typing import Dict
import pandas as pd
from .db import get_connection, execute_batch, execute_script
from .config import SQL_DIR

def initialize_database():
    """Applies all schema DDL scripts in order."""
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        script = sql_file.read_text(encoding="utf-8")
        execute_script(script)

def load_staging(df_raw: pd.DataFrame) -> int:
    """Loads raw dataframe into stg_ecommerce_orders."""
    conn = get_connection()
    try:
        # Check if already populated
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM stg_ecommerce_orders;")
        count = cur.fetchone()[0]
        if count >= len(df_raw):
            return count
        
        # Load raw
        df_raw.to_sql("stg_ecommerce_orders", conn, if_exists="replace", index=False)
        return len(df_raw)
    finally:
        conn.close()

def load_star_schema(tables: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """Loads transformed DataFrames into conformed dimensions and fact tables."""
    conn = get_connection()
    results = {}
    try:
        # 1. Dimensions
        for dim_name in ["dim_date", "dim_customer", "dim_product", "dim_order_status", "dim_payment_method"]:
            df = tables[dim_name]
            df.to_sql(dim_name, conn, if_exists="replace", index=False)
            results[dim_name] = len(df)
        
        # 2. Facts
        for fact_name in ["fact_orders", "fact_reviews"]:
            df = tables[fact_name]
            df.to_sql(fact_name, conn, if_exists="replace", index=False)
            results[fact_name] = len(df)
            
        # Re-apply analytical views after tables loaded
        views_script = (SQL_DIR / "05_analytical_views.sql").read_text(encoding="utf-8")
        if hasattr(conn, "executescript"):
            conn.executescript(views_script)
            conn.commit()
    finally:
        conn.close()
        
    return results
