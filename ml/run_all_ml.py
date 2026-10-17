"""
ml/run_all_ml.py — Master orchestrator for the entire Machine Learning & Mining suite.
Run: python -m ml.run_all_ml
"""
import time
import json
from pathlib import Path
from .classification import run_classification
from .clustering import run_clustering
from .association_rules import run_association_rules
from .regression_revenue import run_revenue_regression
from .anomaly_detection import run_anomaly_detection
from .trend_analysis import run_trend_analysis

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"

def run_all_ml():
    print("="*65)
    print("  E-COMMERCE DATA MINING & ML SUITE — EXECUTION")
    print("="*65)
    start_time = time.time()
    
    cls_res = run_classification()
    cl_res = run_clustering()
    ar_res = run_association_rules()
    reg_res = run_revenue_regression()
    anom_res = run_anomaly_detection()
    tr_res = run_trend_analysis()
    
    elapsed = time.time() - start_time
    
    summary = {
        "execution_time_seconds": round(elapsed, 2),
        "modules_completed": [
            "Order Cancellation Classification (Random Forest & Decision Tree)",
            "Customer RFM Clustering (K-Means)",
            "Market Basket Analysis (Apriori & Association Rules)",
            "Revenue & CLV Regression (Random Forest & Ridge)",
            "Order Anomaly Detection (Isolation Forest)",
            "Temporal & Market Trend Analysis"
        ]
    }
    
    with open(OUTPUT_DIR / "ml_suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*65)
    print(f"✨ Full ML Suite completed in {elapsed:.2f} seconds!")
    print("="*65 + "\n")

if __name__ == "__main__":
    run_all_ml()
