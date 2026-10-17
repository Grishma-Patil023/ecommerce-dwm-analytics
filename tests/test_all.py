"""
tests/test_all.py — Comprehensive Automated Test Suite for E-Commerce Data Warehouse.
Run: python tests/test_all.py
"""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from etl.db import execute_query, query_df
from simulator import run_simulation

class TestEcommerceWarehouse(unittest.TestCase):
    
    def test_01_fact_orders_completeness(self):
        """Validates exactly 10,000 order records in fact table."""
        res = execute_query("SELECT COUNT(*) as c FROM fact_orders;")
        self.assertEqual(res[0]["c"], 10000, "fact_orders must contain exactly 10,000 rows.")

    def test_02_dimensions_populated(self):
        """Validates all five conformed dimensions contain records."""
        dims = ["dim_date", "dim_customer", "dim_product", "dim_order_status", "dim_payment_method"]
        for d in dims:
            res = execute_query(f"SELECT COUNT(*) as c FROM {d};")
            self.assertGreater(res[0]["c"], 0, f"Dimension {d} must not be empty.")

    def test_03_referential_integrity(self):
        """Ensures zero orphaned foreign keys in fact_orders."""
        sql = """
        SELECT COUNT(*) as orphans FROM fact_orders f
        LEFT JOIN dim_customer c ON f.customer_key = c.customer_key
        LEFT JOIN dim_product p ON f.product_key = p.product_key
        LEFT JOIN dim_date d ON f.date_key = d.date_key
        WHERE c.customer_key IS NULL OR p.product_key IS NULL OR d.date_key IS NULL;
        """
        res = execute_query(sql)
        self.assertEqual(res[0]["orphans"], 0, "No orphaned foreign keys allowed in fact_orders.")

    def test_04_revenue_reconciliation(self):
        """Verifies total revenue equals sum of (quantity * unit_price)."""
        sql = "SELECT SUM(total_amount) as total_rev, SUM(quantity * unit_price) as math_rev FROM fact_orders;"
        res = execute_query(sql)[0]
        self.assertAlmostEqual(float(res["total_rev"]), float(res["math_rev"]), places=1)

    def test_05_simulator_output_structure(self):
        """Verifies the E-Commerce Campaign Simulator returns expected decision attributes."""
        res = run_simulation(category="Electronics", discount_pct=10.0, quantity=1)
        self.assertIn("financial_projections", res)
        self.assertIn("behavioral_probabilities", res)
        self.assertIn("recommendation", res)
        self.assertGreater(res["financial_projections"]["net_revenue"], 0)

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  RUNNING FULL E-COMMERCE DWM TEST SUITE")
    print("="*65 + "\n")
    unittest.main(verbosity=2)
