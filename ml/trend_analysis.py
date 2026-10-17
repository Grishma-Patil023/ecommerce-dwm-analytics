"""
ml/trend_analysis.py — Temporal & Market Trend Trajectory Analysis.
"""
import json
from pathlib import Path
import pandas as pd

from .db_loader import load_orders_ml_df

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_trend_analysis():
    print("\n[ML] Running E-Commerce Trend Trajectory Analysis...")
    df = load_orders_ml_df()
    
    # 1. Category Revenue Trends
    cat_summary = df.groupby("category")["total_amount"].agg(["count", "sum", "mean"]).round(2).to_dict(orient="index")
    
    # 2. Country Performance
    country_summary = df.groupby("country")["total_amount"].agg(["count", "sum"]).round(2).to_dict(orient="index")
    
    # 3. Monthly velocity
    monthly = df.groupby(["year", "month"])["total_amount"].sum().round(2).to_dict()
    monthly_formatted = {f"{k[0]}-{k[1]:02d}": v for k, v in monthly.items()}
    
    summary = {
        "categories": cat_summary,
        "countries": country_summary,
        "monthly_revenue_timeline": monthly_formatted,
        "top_performing_category": df.groupby("category")["total_amount"].sum().idxmax(),
        "top_market": df.groupby("country")["total_amount"].sum().idxmax()
    }
    
    out_file = OUTPUT_DIR / "trend_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"  ✓ Trend analysis complete. Top category: {summary['top_performing_category']}")
    print(f"  ✓ Saved to: {out_file.name}")
    return summary

if __name__ == "__main__":
    run_trend_analysis()
