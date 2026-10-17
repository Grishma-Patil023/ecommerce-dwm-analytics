"""
product_analysis.py — Product catalog velocity, profit contribution, and ratings.
Run: python product_analysis.py
"""
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from etl.db import query_df

def run_product_analysis():
    print("\n" + "="*65)
    print("  E-COMMERCE PRODUCT CATALOG & MARGIN ANALYSIS")
    print("="*65 + "\n")
    
    sql = """
    SELECT
        p.category,
        p.price_tier,
        COUNT(DISTINCT p.product_id) AS products_in_catalog,
        SUM(f.quantity) AS units_sold,
        SUM(f.total_amount) AS revenue,
        SUM(f.estimated_margin) AS gross_profit,
        ROUND(AVG(r.rating), 2) AS avg_rating
    FROM dim_product p
    JOIN fact_orders f ON p.product_key = f.product_key
    LEFT JOIN fact_reviews r ON p.product_key = r.product_key
    GROUP BY p.category, p.price_tier
    ORDER BY revenue DESC;
    """
    df = query_df(sql)
    print(f"  {'Category':<16} {'Price Tier':<12} {'Units':>8} {'Revenue ($)':>14} {'Profit ($)':>12} {'Rating':>7}")
    print("  " + "-"*73)
    for _, r in df.iterrows():
        print(f"  {r['category']:<16} {r['price_tier']:<12} {r['units_sold']:>8,d} ${r['revenue']:>13,.2f} ${r['gross_profit']:>11,.2f} {r['avg_rating']:>7.2f}★")
        
    print("\n" + "="*65 + "\n")

if __name__ == "__main__":
    run_product_analysis()
