"""
etl/run_pipeline.py — Master orchestrator for E-Commerce Data Warehouse ETL.
"""
import time
from .extract import extract_raw_data
from .transform import clean_and_transform
from .load import initialize_database, load_staging, load_star_schema

def run_etl_pipeline():
    print("=" * 65)
    print("  E-COMMERCE DATA WAREHOUSE — ETL PIPELINE EXECUTION")
    print("=" * 65)
    start_time = time.time()
    
    # Step 1: Initialize Database
    print("\n[STEP 1/4] Initializing Schema DDL & Constraints...")
    initialize_database()
    print("  ✓ Schemas, tables, and views successfully applied.")
    
    # Step 2: Extract
    print("\n[STEP 2/4] Extracting Raw E-Commerce Records...")
    df_raw, audit_meta = extract_raw_data()
    print(f"  ✓ Extracted {audit_meta['total_rows']:,} orders from {audit_meta['source_file']}")
    print(f"  ✓ {audit_meta['unique_customers']:,} unique customers | {audit_meta['unique_products']:,} unique products")
    
    # Load Staging
    stg_count = load_staging(df_raw)
    print(f"  ✓ Staging table loaded ({stg_count:,} rows)")
    
    # Step 3: Transform
    print("\n[STEP 3/4] Transforming Dimensions, Conformed Keys & Facts...")
    tables = clean_and_transform(df_raw)
    for name, df in tables.items():
        if name != "staging":
            print(f"  ✓ Transformed {name:<22} -> {len(df):,d} records")
            
    # Step 4: Load
    print("\n[STEP 4/4] Ingesting into Star Schema Tables...")
    loaded = load_star_schema(tables)
    print("  ✓ Star Schema tables loaded successfully:")
    for k, v in loaded.items():
        print(f"     - {k:<22} : {v:>8,d} rows")
        
    elapsed = time.time() - start_time
    print(f"\n✨ ETL Pipeline completed in {elapsed:.2f} seconds!")
    print("=" * 65 + "\n")
    return loaded

if __name__ == "__main__":
    run_etl_pipeline()
