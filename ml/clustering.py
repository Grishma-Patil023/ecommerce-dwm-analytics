"""
ml/clustering.py — Customer RFM Segmentation via K-Means Clustering.
Calculates Recency, Frequency, and Monetary scores and clusters customers into distinct personas.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .db_loader import load_rfm_df

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_clustering(n_clusters=4):
    print("\n[ML] Running Customer RFM K-Means Clustering...")
    df = load_rfm_df()
    
    # Calculate Recency: days since max date_key in dataset
    max_date = df["last_order_date_key"].max()
    # Treat integer YYYYMMDD difference proxy
    df["recency_proxy"] = (max_date - df["last_order_date_key"]).clip(lower=0)
    
    features = ["recency_proxy", "frequency", "monetary", "aov"]
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    sil_score = float(silhouette_score(X_scaled, df["cluster"]))
    
    # Persona mapping based on average monetary and frequency
    cluster_means = df.groupby("cluster")[features].mean()
    cluster_names = {}
    sorted_by_spend = cluster_means.sort_values("monetary", ascending=False).index.tolist()
    labels = ["Champions (High Value)", "Loyal Frequent Shoppers", "Recent / Promising", "At-Risk / Inactive"]
    for idx, c_id in enumerate(sorted_by_spend):
        cluster_names[int(c_id)] = labels[idx] if idx < len(labels) else f"Segment {c_id}"
        
    df["segment_name"] = df["cluster"].map(cluster_names)
    
    # Save clustered CSV
    csv_out = OUTPUT_DIR / "customer_segments.csv"
    df[["customer_id", "country", "age_group", "frequency", "monetary", "aov", "cluster", "segment_name"]].to_csv(csv_out, index=False)
    
    summary = {
        "total_customers": len(df),
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 4),
        "cluster_distribution": df["segment_name"].value_counts().to_dict(),
        "cluster_averages": cluster_means.round(2).to_dict(orient="index")
    }
    
    json_out = OUTPUT_DIR / "clustering_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"  ✓ Segmented {len(df):,d} customers into {n_clusters} clusters (Silhouette: {sil_score:.3f})")
    print(f"  ✓ Output saved to: {csv_out.name} and {json_out.name}")
    return summary

if __name__ == "__main__":
    run_clustering()
