"""
customer_analysis.py — Deep-dive customer demographics, RFM distributions, and loyalty analysis.
Run: python customer_analysis.py
"""
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from etl.db import query_df

def run_customer_analysis():
    print("\n" + "="*65)
    print("  E-COMMERCE CUSTOMER DEMOGRAPHICS & COHORT ANALYSIS")
    print("="*65 + "\n")
    
    # 1. Country & Spend Summary
    sql = """
    SELECT
        c.country,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(f.order_fact_key) AS total_orders,
        SUM(f.total_amount) AS total_gmv,
        ROUND(AVG(f.total_amount), 2) AS aov
    FROM dim_customer c
    JOIN fact_orders f ON c.customer_key = f.customer_key
    GROUP BY c.country
    ORDER BY total_gmv DESC;
    """
    df_country = query_df(sql)
    print("  Top Markets by Gross Revenue:")
    print(f"  {'Country':<15} {'Customers':>10} {'Orders':>10} {'GMV ($)':>14} {'AOV ($)':>10}")
    print("  " + "-"*62)
    for _, r in df_country.iterrows():
        print(f"  {r['country']:<15} {r['total_customers']:>10,d} {r['total_orders']:>10,d} ${r['total_gmv']:>13,.2f} ${r['aov']:>9.2f}")
        
    # 2. Age Group Penetration
    sql_age = """
    SELECT
        c.age_group,
        COUNT(DISTINCT c.customer_id) as customers,
        SUM(f.total_amount) as revenue,
        ROUND(SUM(f.total_amount) * 100.0 / (SELECT SUM(total_amount) FROM fact_orders), 1) as share_pct
    FROM dim_customer c
    JOIN fact_orders f ON c.customer_key = f.customer_key
    GROUP BY c.age_group
    ORDER BY revenue DESC;
    """
    df_age = query_df(sql_age)
    print("\n  Demographic Age Cohorts:")
    print(f"  {'Age Group':<15} {'Customers':>10} {'Revenue ($)':>14} {'Share (%)':>10}")
    print("  " + "-"*52)
    for _, r in df_age.iterrows():
        print(f"  {r['age_group']:<15} {r['customers']:>10,d} ${r['revenue']:>13,.2f} {r['share_pct']:>9.1f}%")
        
    print("\n" + "="*65 + "\n")

if __name__ == "__main__":
    run_customer_analysis()
