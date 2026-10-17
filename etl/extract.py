"""
etl/extract.py — Extracts and audits raw 10,000 transaction CSV records.
"""
import pandas as pd
from typing import Dict, Any, Tuple
from .config import DATASET_PATH

EXPECTED_COLUMNS = [
    "customer_id", "first_name", "last_name", "gender", "age_group",
    "signup_date", "country", "product_id", "product_name", "category",
    "quantity", "unit_price", "order_id", "order_date", "order_status",
    "payment_method", "rating", "review_text", "review_id", "review_date"
]

def extract_raw_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reads source dataset and produces initial extraction telemetry."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found at: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Validation checks
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    audit_meta = {
        "source_file": str(DATASET_PATH.name),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_orders": df["order_id"].nunique(),
        "unique_customers": df["customer_id"].nunique(),
        "unique_products": df["product_id"].nunique(),
        "null_counts": df.isnull().sum().to_dict(),
        "countries": df["country"].unique().tolist(),
        "categories": df["category"].unique().tolist(),
    }
    
    return df, audit_meta

if __name__ == "__main__":
    df, meta = extract_raw_data()
    print(f"Extraction successful: {meta['total_rows']:,} records extracted.")
