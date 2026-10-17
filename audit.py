"""
audit.py — Audits E-Commerce Data Warehouse state, table row counts, and health.
Run: python audit.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from etl.db import query_df, execute_query

def audit_warehouse():
    print("\n" + "="*65)
    print("  E-COMMERCE DATA WAREHOUSE — STATE AUDIT")
    print("="*65 + "\n")
    
    tables = [
        ("stg_ecommerce_orders", "Staging", "Raw transaction ingestion"),
        ("dim_date", "Dimension", "Conformed Calendar & Time"),
        ("dim_customer", "Dimension", "Customer Demographics & Geographies"),
        ("dim_product", "Dimension", "Product Catalog & Pricing Tiers"),
        ("dim_order_status", "Dimension", "Fulfillment State Flags"),
        ("dim_payment_method", "Dimension", "Payment Channels"),
        ("fact_orders", "Fact", "Sales & Margin Transactions"),
        ("fact_reviews", "Fact", "Customer Ratings & Sentiments"),
    ]
    
    print(f"  {'Table Name':<25} {'Type':<12} {'Row Count':>12}  {'Status'}")
    print("  " + "-"*60)
    
    total_fact_rows = 0
    all_ok = True
    
    for tbl, t_type, desc in tables:
        try:
            res = execute_query(f"SELECT COUNT(*) AS cnt FROM {tbl};")
            cnt = res[0]["cnt"] if res else 0
            status = "✅ Active" if cnt > 0 else "⚠️ Empty"
            if cnt == 0 and t_type != "Staging":
                all_ok = False
            if t_type == "Fact":
                total_fact_rows += cnt
            print(f"  {tbl:<25} {t_type:<12} {cnt:>12,d}  {status}")
        except Exception as e:
            all_ok = False
            print(f"  {tbl:<25} {t_type:<12} {'ERROR':>12}  ❌ Not Found ({e})")
            
    # Revenue audit
    try:
        rev_res = execute_query("SELECT SUM(total_amount) as gmv, AVG(total_amount) as aov FROM fact_orders;")
        if rev_res and rev_res[0]["gmv"]:
            gmv = float(rev_res[0]["gmv"])
            aov = float(rev_res[0]["aov"])
            print("\n  Financial Metrics:")
            print(f"    - Gross Merchandise Value (GMV): ${gmv:,.2f}")
            print(f"    - Average Order Value (AOV):     ${aov:,.2f}")
    except Exception:
        pass
        
    print("\n" + "="*65)
    if all_ok:
        print("  🎉 AUDIT PASSED: All warehouse tables are populated and conformed.")
    else:
        print("  ⚠️ AUDIT NOTICE: Some tables are missing or empty.")
        print("  Run 'python run_warehouse.py' to build and populate the warehouse.")
    print("="*65 + "\n")
    return all_ok

if __name__ == "__main__":
    ok = audit_warehouse()
    sys.exit(0 if ok else 1)
