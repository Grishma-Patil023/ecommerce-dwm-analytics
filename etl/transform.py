"""
etl/transform.py — Data cleansing, conformed date key generation, and dimensional modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

def clean_and_transform(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Transforms raw transactions into conformed dimensional tables and fact records."""
    df = df_raw.copy()
    
    # 1. Clean string fields
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 2. Date parsing and conformed integer keys
    df["dt_order"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["dt_signup"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["dt_review"] = pd.to_datetime(df["review_date"], errors="coerce")
    
    df["date_key"] = df["dt_order"].dt.strftime("%Y%m%d").astype(int)
    df["signup_date_key"] = df["dt_signup"].dt.strftime("%Y%m%d").fillna(0).astype(int)
    df["review_date_key"] = df["dt_review"].dt.strftime("%Y%m%d").fillna(0).astype(int)
    
    # 3. Numeric calculations
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0).round(2)
    df["total_amount"] = (df["quantity"] * df["unit_price"]).round(2)
    
    # Cost & margin derivation (proxy: estimated margin ~ 35-45% based on category)
    margin_rates = {
        "Electronics": 0.28,
        "Apparel": 0.45,
        "Home & Kitchen": 0.38,
        "Beauty": 0.52,
        "Toys": 0.35,
        "Sports": 0.40,
        "Books": 0.30
    }
    df["margin_rate"] = df["category"].map(margin_rates).fillna(0.35)
    df["estimated_margin"] = (df["total_amount"] * df["margin_rate"]).round(2)
    
    # 4. Sentiment scoring derivation (based on rating and review text)
    sentiment_map = {5: 0.85, 4: 0.50, 3: 0.10, 2: -0.40, 1: -0.80}
    df["sentiment_score"] = df["rating"].map(sentiment_map).fillna(0.0)
    df["sentiment_label"] = np.where(df["rating"] >= 4, "Positive",
                             np.where(df["rating"] == 3, "Neutral", "Negative"))
    
    # ── DIMENSION 1: dim_date ──────────────────────────────────────────────
    all_dates = pd.concat([df["dt_order"], df["dt_signup"], df["dt_review"]]).dropna().drop_duplicates()
    dim_date = pd.DataFrame({"full_date": all_dates.dt.date.drop_duplicates()})
    dim_date["dt"] = pd.to_datetime(dim_date["full_date"])
    dim_date["date_key"] = dim_date["dt"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["dt"].dt.year
    dim_date["quarter"] = dim_date["dt"].dt.quarter
    dim_date["month"] = dim_date["dt"].dt.month
    dim_date["month_name"] = dim_date["dt"].dt.strftime("%B")
    dim_date["day"] = dim_date["dt"].dt.day
    dim_date["day_name"] = dim_date["dt"].dt.strftime("%A")
    dim_date["is_weekend"] = dim_date["dt"].dt.weekday.isin([5, 6]).astype(int)
    dim_date = dim_date.drop(columns=["dt"]).sort_values("date_key").reset_index(drop=True)
    
    # ── DIMENSION 2: dim_customer ──────────────────────────────────────────
    dim_customer = df[[
        "customer_id", "first_name", "last_name", "gender", "age_group", "country", "signup_date", "signup_date_key"
    ]].drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    dim_customer["full_name"] = dim_customer["first_name"] + " " + dim_customer["last_name"]
    dim_customer["customer_key"] = dim_customer.index + 1
    
    # ── DIMENSION 3: dim_product ───────────────────────────────────────────
    dim_product = df[[
        "product_id", "product_name", "category", "unit_price"
    ]].drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    
    # Price Tiering
    p_median = dim_product["unit_price"].median()
    p_75 = dim_product["unit_price"].quantile(0.75)
    dim_product["price_tier"] = np.where(dim_product["unit_price"] >= p_75, "Premium",
                                np.where(dim_product["unit_price"] >= p_median, "Mid-Range", "Budget"))
    dim_product["product_key"] = dim_product.index + 1
    
    # ── DIMENSION 4: dim_order_status ──────────────────────────────────────
    statuses = sorted(df["order_status"].dropna().unique().tolist())
    dim_status = pd.DataFrame({
        "status_key": range(1, len(statuses) + 1),
        "order_status": statuses,
        "is_completed": [1 if s == "Delivered" else 0 for s in statuses]
    })
    
    # ── DIMENSION 5: dim_payment_method ────────────────────────────────────
    payments = sorted(df["payment_method"].dropna().unique().tolist())
    dim_payment = pd.DataFrame({
        "payment_key": range(1, len(payments) + 1),
        "payment_method": payments,
        "is_digital": [0 if p == "Cash on Delivery" else 1 for p in payments]
    })
    
    # Map foreign keys back onto transactions for facts
    cust_key_map = dict(zip(dim_customer["customer_id"], dim_customer["customer_key"]))
    prod_key_map = dict(zip(dim_product["product_id"], dim_product["product_key"]))
    status_key_map = dict(zip(dim_status["order_status"], dim_status["status_key"]))
    payment_key_map = dict(zip(dim_payment["payment_method"], dim_payment["payment_key"]))
    
    df["customer_key"] = df["customer_id"].map(cust_key_map)
    df["product_key"] = df["product_id"].map(prod_key_map)
    df["status_key"] = df["order_status"].map(status_key_map)
    df["payment_key"] = df["payment_method"].map(payment_key_map)
    
    # ── FACT 1: fact_orders ────────────────────────────────────────────────
    fact_orders = df[[
        "order_id", "customer_key", "product_key", "date_key", "status_key",
        "payment_key", "quantity", "unit_price", "total_amount", "estimated_margin"
    ]].drop_duplicates(subset=["order_id"]).reset_index(drop=True)
    fact_orders["order_fact_key"] = fact_orders.index + 1
    
    # ── FACT 2: fact_reviews ───────────────────────────────────────────────
    fact_reviews = df[[
        "review_id", "order_id", "customer_key", "product_key", "review_date_key",
        "rating", "sentiment_score", "sentiment_label", "review_text"
    ]].drop_duplicates(subset=["review_id"]).reset_index(drop=True)
    fact_reviews["review_fact_key"] = fact_reviews.index + 1
    
    return {
        "staging": df_raw,
        "dim_date": dim_date,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
        "dim_order_status": dim_status,
        "dim_payment_method": dim_payment,
        "fact_orders": fact_orders,
        "fact_reviews": fact_reviews
    }
