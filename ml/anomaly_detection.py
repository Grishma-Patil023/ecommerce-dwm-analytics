"""
ml/anomaly_detection.py — Identifies abnormal order volumes and return/cancellation risk transactions.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest

from .db_loader import load_orders_ml_df

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_anomaly_detection(contamination=0.02):
    print("\n[ML] Running Anomaly Detection on Order Transactions...")
    df = load_orders_ml_df()
    
    features = ["quantity", "unit_price", "total_amount"]
    X = df[features].fillna(0)
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly_score"] = iso.fit_predict(X)
    # -1 is anomaly, 1 is normal
    anomalies = df[df["anomaly_score"] == -1].copy()
    
    csv_out = OUTPUT_DIR / "order_anomalies.csv"
    anomalies[[
        "order_id", "customer_id", "product_name", "category",
        "quantity", "unit_price", "total_amount", "country", "order_status"
    ]].to_csv(csv_out, index=False)
    
    summary = {
        "total_audited": len(df),
        "anomalies_detected": len(anomalies),
        "contamination_rate": contamination,
        "max_anomalous_spend": float(anomalies["total_amount"].max()) if not anomalies.empty else 0,
        "high_quantity_outliers": int((anomalies["quantity"] >= 4).sum()),
    }
    
    out_file = OUTPUT_DIR / "anomaly_detection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"  ✓ Flagged {len(anomalies)} anomalous transactions ({contamination*100}% threshold)")
    print(f"  ✓ Anomalies exported to: {csv_out.name}")
    return summary

if __name__ == "__main__":
    run_anomaly_detection()
