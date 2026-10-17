"""
validate_warehouse.py — Deep dimensional & referential integrity validation test.
Run: python validate_warehouse.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from etl.db import query_df, execute_query

def validate_all():
    print("\n" + "="*70)
    print("  E-COMMERCE DATA WAREHOUSE — DIMENSIONAL VALIDATION SUITE")
    print("="*70 + "\n")
    
    checks = []
    
    def check(name, assertion_fn):
        try:
            ok, msg = assertion_fn()
            status = "✅ PASS" if ok else "❌ FAIL"
            checks.append((ok, name, msg))
            print(f"  {status}  {name:<45} {msg}")
        except Exception as e:
            checks.append((False, name, str(e)))
            print(f"  ❌ FAIL  {name:<45} ERROR: {e}")
            
    # Check 1: Fact Orders Row Count == 10,000
    def c1():
        cnt = execute_query("SELECT COUNT(*) AS c FROM fact_orders;")[0]["c"]
        return cnt == 10000, f"Expected 10,000 rows, found {cnt:,}"
    check("1. Fact Orders Grain & Completeness", c1)
    
    # Check 2: Zero Orphaned Customer Keys
    def c2():
        sql = """SELECT COUNT(*) AS c FROM fact_orders f 
                 LEFT JOIN dim_customer c ON f.customer_key = c.customer_key 
                 WHERE c.customer_key IS NULL;"""
        cnt = execute_query(sql)[0]["c"]
        return cnt == 0, f"Orphaned customer keys: {cnt}"
    check("2. Customer Foreign Key Integrity", c2)
    
    # Check 3: Zero Orphaned Product Keys
    def c3():
        sql = """SELECT COUNT(*) AS c FROM fact_orders f 
                 LEFT JOIN dim_product p ON f.product_key = p.product_key 
                 WHERE p.product_key IS NULL;"""
        cnt = execute_query(sql)[0]["c"]
        return cnt == 0, f"Orphaned product keys: {cnt}"
    check("3. Product Foreign Key Integrity", c3)
    
    # Check 4: Zero Orphaned Date Keys
    def c4():
        sql = """SELECT COUNT(*) AS c FROM fact_orders f 
                 LEFT JOIN dim_date d ON f.date_key = d.date_key 
                 WHERE d.date_key IS NULL;"""
        cnt = execute_query(sql)[0]["c"]
        return cnt == 0, f"Orphaned date keys: {cnt}"
    check("4. Conformed Date Dimension Integrity", c4)
    
    # Check 5: Financial Reconciliation (quantity * unit_price == total_amount)
    def c5():
        sql = """SELECT COUNT(*) AS c FROM fact_orders 
                 WHERE ABS(total_amount - (quantity * unit_price)) > 0.01;"""
        cnt = execute_query(sql)[0]["c"]
        return cnt == 0, f"Discrepancies found: {cnt}"
    check("5. Revenue Line-Item Math Consistency", c5)
    
    # Check 6: Valid Rating Bounds [1, 5]
    def c6():
        sql = "SELECT MIN(rating) as min_r, MAX(rating) as max_r FROM fact_reviews;"
        res = execute_query(sql)[0]
        ok = res["min_r"] >= 1 and res["max_r"] <= 5
        return ok, f"Range: [{res['min_r']}, {res['max_r']}]"
    check("6. Customer Rating Range Enforcement", c6)
    
    # Check 7: Positive Quantities
    def c7():
        sql = "SELECT COUNT(*) as c FROM fact_orders WHERE quantity <= 0;"
        cnt = execute_query(sql)[0]["c"]
        return cnt == 0, f"Non-positive quantity orders: {cnt}"
    check("7. Quantity Clamping (>= 1)", c7)
    
    # Check 8: Conformed Dimensions Non-Empty
    def c8():
        c_cnt = execute_query("SELECT COUNT(*) as c FROM dim_customer;")[0]["c"]
        p_cnt = execute_query("SELECT COUNT(*) as c FROM dim_product;")[0]["c"]
        return c_cnt > 0 and p_cnt > 0, f"Customers: {c_cnt:,} | Products: {p_cnt:,}"
    check("8. Dimension Cardinality Verification", c8)
    
    # Check 9: Analytical Views Operational
    def c9():
        res = query_df("SELECT * FROM vw_category_performance LIMIT 5;")
        return len(res) > 0, f"vw_category_performance returned {len(res)} rows"
    check("9. OLAP Analytical Views Queryable", c9)
    
    # Check 10: Gross Merchandise Value > $5,000,000
    def c10():
        gmv = execute_query("SELECT SUM(total_amount) as s FROM fact_orders;")[0]["s"]
        return gmv > 5000000, f"Total GMV: ${gmv:,.2f}"
    check("10. Enterprise Revenue Scale Validation", c10)
    
    print("\n" + "="*70)
    passed = sum(1 for ok, _, _ in checks if ok)
    total = len(checks)
    print(f"  RESULTS: {passed}/{total} checks passed!")
    print("="*70 + "\n")
    return passed == total

if __name__ == "__main__":
    ok = validate_all()
    sys.exit(0 if ok else 1)
