"""
ml/regression_revenue.py — Customer Lifetime Value (CLV) & Revenue Forecasting Regression.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from .db_loader import load_orders_ml_df

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_revenue_regression():
    print("\n[ML] Running Revenue & CLV Regression Modeling...")
    df = load_orders_ml_df()
    
    # Target: total_amount
    X = pd.get_dummies(df[[
        "quantity", "unit_price", "category", "country", "age_group", "is_weekend"
    ]], drop_first=True)
    y = df["total_amount"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Save RF model
    model_path = OUTPUT_DIR / "revenue_model.joblib"
    joblib.dump(rf, model_path)
    
    results = {
        "ridge_regression": {
            "r2_score": round(float(r2_score(y_test, y_pred_ridge)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred_ridge)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_ridge))), 2),
        },
        "random_forest": {
            "r2_score": round(float(r2_score(y_test, y_pred_rf)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred_rf)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 2),
        },
        "model_file": str(model_path.name)
    }
    
    out_file = OUTPUT_DIR / "regression_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"  ✓ RF Regressor R² Score: {results['random_forest']['r2_score']:.4f} (MAE: ${results['random_forest']['mae']:.2f})")
    print(f"  ✓ Model saved to: {model_path.name}")
    return results

if __name__ == "__main__":
    run_revenue_regression()
