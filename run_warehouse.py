"""
run_warehouse.py — Master runner to load and build the E-Commerce Data Warehouse.
Run: python run_warehouse.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from etl.run_pipeline import run_etl_pipeline
from audit import audit_warehouse

def main():
    print("\n" + "="*65)
    print("  E-COMMERCE DATA WAREHOUSE — MASTER BUILDER")
    print("█"*65)
    
    # 1. Run ETL Pipeline
    run_etl_pipeline()
    
    # 2. Audit Loaded Tables
    audit_warehouse()
    
    print("Warehouse build finished successfully.")

if __name__ == "__main__":
    main()
