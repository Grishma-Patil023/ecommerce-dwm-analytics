"""
ml/classification.py — Order Fulfillment & Cancellation Propensity Classification.
Trains Decision Tree and Random Forest classifiers.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .db_loader import load_orders_ml_df

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_classification():
    print("\n[ML] Running Order Cancellation / Status Classification...")
    df = load_orders_ml_df()
    
    # Target: 1 if Cancelled, 0 if Delivered/Completed
    df["target"] = (df["order_status"] == "Cancelled").astype(int)
    
    # Feature Engineering
    features = pd.get_dummies(df[[
        "quantity", "unit_price", "total_amount", "category",
        "payment_method", "age_group", "is_weekend"
    ]], drop_first=True)
    
    X = features
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Feature Importance
    importances = dict(zip(X.columns, rf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
    
    results = {
        "dataset_size": len(df),
        "cancellation_rate": float(y.mean()),
        "decision_tree": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_dt)), 4),
            "precision": round(float(precision_score(y_test, y_pred_dt, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_dt, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_dt, zero_division=0)), 4),
        },
        "random_forest": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_rf)), 4),
            "precision": round(float(precision_score(y_test, y_pred_rf, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_rf, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_rf, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
        },
        "top_features": {k: round(float(v), 4) for k, v in top_features}
    }
    
    out_file = OUTPUT_DIR / "classification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"  ✓ Random Forest Accuracy: {results['random_forest']['accuracy']*100:.2f}% | F1: {results['random_forest']['f1']:.4f}")
    print(f"  ✓ Saved to: {out_file.name}")
    return results

if __name__ == "__main__":
    run_classification()
