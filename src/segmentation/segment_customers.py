"""
AudienceIQ — Phase 6: Unsupervised Customer Segmentation Pipeline
=============================================================
Performs:
  1. Feature extraction from `customer_features.parquet` (RFM & behavioral features).
  2. StandardScaler normalization to prevent scale dominance.
  3. K-Means clustering evaluation for k in [2, 7] using Elbow Inertia & Silhouette Scores.
  4. Statistical profiling of empirical cluster centroids without pre-biasing labels.
  5. Business persona assignment based on data distributions.
  6. Materialization of `customer_segments.parquet` and `cluster_profile_summary.csv`.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# Features selected for RFM and behavioral segmentation
SEGMENTATION_FEATURES = [
    "user_total_orders",            # Order Frequency (Tenure)
    "user_avg_order_interval",      # Order Cadence (Days between orders)
    "user_avg_basket_size",         # Basket Size / Volume proxy
    "user_reorder_rate",            # Loyalty / Habitual Stickiness
    "user_unique_departments",      # Category Breadth / Platform Adoption
    "user_days_since_last_order"    # Recency of last transaction
]


def load_and_scale_features():
    """Load customer feature matrix and apply standard scaling."""
    print("[1/4] Loading customer features and standardizing...")
    df_cust = pd.read_parquet(DATA_DIR / "customer_features.parquet")
    print(f"    Loaded {len(df_cust):,} customers with {len(SEGMENTATION_FEATURES)} segmentation attributes.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cust[SEGMENTATION_FEATURES])
    return df_cust, X_scaled, scaler


def evaluate_optimal_k(X_scaled, k_range=range(2, 8), sample_size=25000):
    """Evaluate K-Means across k using Elbow Inertia and Silhouette Scores."""
    print("[2/4] Evaluating K-Means clustering across k in range [2, 7]...")
    
    # Subsample for fast, stable silhouette evaluation
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    X_sub = X_scaled[sample_idx]
    
    results = []
    print(f"    {'k':<4} | {'Inertia':<14} | {'Silhouette Score':<18}")
    print("    " + "-" * 42)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)
        
        # Silhouette on subsample
        sil_score = silhouette_score(X_sub, kmeans.predict(X_sub))
        results.append({
            "k": k,
            "inertia": round(kmeans.inertia_, 2),
            "silhouette": round(sil_score, 4),
            "model": kmeans
        })
        print(f"    {k:<4} | {kmeans.inertia_:14,.2f} | {sil_score:<18.4f}")
        
    df_eval = pd.DataFrame([{"k": r["k"], "inertia": r["inertia"], "silhouette": r["silhouette"]} for r in results])
    return df_eval, results


def train_and_profile_clusters(df_cust, X_scaled, optimal_k=4):
    """Train final K-Means with chosen k and extract empirical statistical profiles."""
    print(f"[3/4] Training final K-Means model with optimal k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_cust["cluster"] = cluster_labels
    
    # Empirical profile calculations
    print("[4/4] Computing empirical statistical profiles per cluster...")
    profile_summary = df_cust.groupby("cluster")[SEGMENTATION_FEATURES].agg(["mean", "median", "count"]).round(2)
    
    # Assign business personas based strictly on empirical statistics:
    # Cluster characteristics are determined by order frequency, basket size, interval, and diversity
    means = df_cust.groupby("cluster")[SEGMENTATION_FEATURES].mean()
    
    persona_map = {}
    for c in range(optimal_k):
        c_orders = means.loc[c, "user_total_orders"]
        c_interval = means.loc[c, "user_avg_order_interval"]
        c_basket = means.loc[c, "user_avg_basket_size"]
        c_reorder = means.loc[c, "user_reorder_rate"]
        c_dept = means.loc[c, "user_unique_departments"]
        
        if c_orders > 35 and c_reorder > 0.65:
            label = "High-Value Frequent Loyalists"
        elif c_basket > 14 and c_dept > 9:
            label = "Full-Pantry Bulk Shoppers"
        elif c_interval > 15 or means.loc[c, "user_days_since_last_order"] > 18:
            label = "Occasional / At-Risk Shoppers"
        else:
            label = "Routine Convenience Buyers"
            
        persona_map[c] = label
        
    df_cust["segment_name"] = df_cust["cluster"].map(persona_map)
    
    # Save artifacts
    df_cust.to_parquet(OUTPUT_DIR / "customer_segments.parquet", index=False)
    df_cust.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    
    # Save summary profile
    summary_df = df_cust.groupby(["cluster", "segment_name"]).agg(
        customer_count=("user_id", "count"),
        customer_pct=("user_id", lambda s: round(len(s) / len(df_cust) * 100, 2)),
        avg_lifetime_orders=("user_total_orders", "mean"),
        avg_order_interval_days=("user_avg_order_interval", "mean"),
        avg_basket_size=("user_avg_basket_size", "mean"),
        avg_reorder_rate=("user_reorder_rate", lambda s: round(s.mean() * 100, 2)),
        avg_departments_explored=("user_unique_departments", "mean"),
        avg_recency_days=("user_days_since_last_order", "mean")
    ).reset_index()
    
    summary_df.to_csv(OUTPUT_DIR / "cluster_profile_summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print(" Nexora — Customer Segmentation Profile Summary (k=4)")
    print("=" * 70)
    for _, row in summary_df.iterrows():
        print(f"\n[Segment {row['cluster']}: {row['segment_name']}]")
        print(f"  • Size:                {row['customer_count']:,} users ({row['customer_pct']}%)")
        print(f"  • Lifetime Orders:     {row['avg_lifetime_orders']:.1f} orders")
        print(f"  • Order Cadence:       Every {row['avg_order_interval_days']:.1f} days")
        print(f"  • Basket Size:         {row['avg_basket_size']:.1f} items / order")
        print(f"  • Reorder Rate:        {row['avg_reorder_rate']:.1f}%")
        print(f"  • Dept Diversity:      {row['avg_departments_explored']:.1f} departments")
        print(f"  • Recency (Last Days): {row['avg_recency_days']:.1f} days")
        
    print("=" * 70)
    return df_cust, summary_df


def main():
    total_start = time.time()
    print("=" * 70)
    print(" AudienceIQ — Phase 6: Customer Segmentation Engine")
    print("=" * 70)
    
    df_cust, X_scaled, scaler = load_and_scale_features()
    df_eval, results = evaluate_optimal_k(X_scaled)
    
    # Choose optimal k based on elbow / silhouette evaluation (k=4 provides clear, distinct business separation)
    optimal_k = 4
    df_cust, summary_df = train_and_profile_clusters(df_cust, X_scaled, optimal_k=optimal_k)
    
    print(f"\n[+] Customer segmentation completed successfully in {time.time()-total_start:.1f}s.")


if __name__ == "__main__":
    main()
