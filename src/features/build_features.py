"""
AudienceIQ — Phase 5: Production Feature Engineering Engine
========================================================
Builds modular, reproducible, leak-free feature representations across:
  1. Customer Profile Features (Recency, Frequency, Diversity, Basket Velocity)
  2. Product Catalog Features (Volume, Stickiness, Cart Priority, Taxonomy)
  3. User-Product Interaction Features (Personal Reorder Propensity, Order Lag, Affinity)

Leakage Prevention Guarantee:
  - All historical features are computed EXCLUSIVELY on prior orders (`eval_set == 'prior'`).
  - Prediction targets are derived from the subsequent `train` order without contaminating historical aggregates.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def save_artifact(df: pd.DataFrame, base_name: str):
    """Save dataframe as CSV and attempt Parquet export."""
    csv_path = FEATURES_DIR / f"{base_name}.csv"
    df.to_csv(csv_path, index=False)
    
    parquet_path = FEATURES_DIR / f"{base_name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        # If parquet engine is not available, CSV remains primary
        pass


def build_product_features():
    """Build product-level aggregates using historical orders."""
    print("[1/3] Building Product Catalog Features (Historical Prior Orders)...")
    start = time.time()
    
    # Load products and categories
    df_prod = pd.read_csv(DATA_DIR / "products.csv")
    
    # Load prior order_products
    df_orders = pd.read_csv(DATA_DIR / "orders.csv", usecols=["order_id", "eval_set"])
    prior_order_ids = set(df_orders[df_orders["eval_set"] == "prior"]["order_id"])
    
    # Aggregate product statistics in chunks for memory efficiency
    prod_agg = {}
    print("    -> Processing historical order products for product metrics...")
    
    for chunk in pd.read_csv(DATA_DIR / "order_products.csv", chunksize=2000000):
        # Filter for prior orders to prevent data leakage
        chunk_prior = chunk[chunk["order_id"].isin(prior_order_ids)]
        
        # Groupby in chunk
        grp = chunk_prior.groupby("product_id").agg(
            chunk_orders=("reordered", "count"),
            chunk_reorders=("reordered", "sum"),
            chunk_cart_sum=("add_to_cart_order", "sum")
        )
        
        for pid, row in grp.iterrows():
            if pid not in prod_agg:
                prod_agg[pid] = [0, 0, 0]
            prod_agg[pid][0] += row["chunk_orders"]
            prod_agg[pid][1] += row["chunk_reorders"]
            prod_agg[pid][2] += row["chunk_cart_sum"]
            
    df_pfeats = pd.DataFrame([
        {
            "product_id": pid,
            "prod_total_orders": stats[0],
            "prod_reorder_rate": round(stats[1] / stats[0], 4) if stats[0] > 0 else 0.0,
            "prod_avg_add_to_cart": round(stats[2] / stats[0], 2) if stats[0] > 0 else 0.0
        }
        for pid, stats in prod_agg.items()
    ])
    
    # Merge with catalog taxonomy
    df_product_features = df_prod.merge(df_pfeats, on="product_id", how="left").fillna({
        "prod_total_orders": 0,
        "prod_reorder_rate": 0.0,
        "prod_avg_add_to_cart": 0.0
    })
    
    df_product_features["product_id"] = df_product_features["product_id"].astype(np.int32)
    df_product_features["aisle_id"] = df_product_features["aisle_id"].astype(np.int16)
    df_product_features["department_id"] = df_product_features["department_id"].astype(np.int16)
    df_product_features["prod_total_orders"] = df_product_features["prod_total_orders"].astype(np.int32)
    df_product_features["prod_reorder_rate"] = df_product_features["prod_reorder_rate"].astype(np.float32)
    df_product_features["prod_avg_add_to_cart"] = df_product_features["prod_avg_add_to_cart"].astype(np.float32)
    
    # Save product features
    save_artifact(df_product_features, "product_features")
    print(f"    [+] Saved product_features ({len(df_product_features):,} products in {time.time()-start:.1f}s)")
    return df_product_features


def build_customer_features():
    """Build customer-level behavioral and RFM features."""
    print("[2/3] Building Customer Profile & RFM Features (206,209 Users)...")
    start = time.time()
    
    df_orders = pd.read_csv(DATA_DIR / "orders.csv")
    df_prior_orders = df_orders[df_orders["eval_set"] == "prior"]
    
    # 1. Order-level temporal and frequency metrics per user
    print("    -> Aggregating order timing and frequency...")
    user_order_stats = df_prior_orders.groupby("user_id").agg(
        user_total_orders=("order_number", "max"),
        user_avg_order_interval=("days_since_prior_order", "mean"),
        user_std_order_interval=("days_since_prior_order", "std"),
        user_preferred_dow=("order_dow", lambda s: s.mode()[0] if not s.empty else 0),
        user_preferred_hour=("order_hour_of_day", lambda s: s.mode()[0] if not s.empty else 12),
        user_days_since_last_order=("days_since_prior_order", "last")
    ).reset_index()
    user_order_stats["user_std_order_interval"] = user_order_stats["user_std_order_interval"].fillna(0.0)
    
    # 2. Basket size and diversity metrics
    print("    -> Aggregating basket size, reorders, and department diversity...")
    # Load order products for prior orders
    prior_order_ids = set(df_prior_orders["order_id"])
    df_prod = pd.read_csv(DATA_DIR / "products.csv", usecols=["product_id", "department_id", "aisle_id"])
    
    # Map order_id to user_id
    order_to_user = df_prior_orders.set_index("order_id")["user_id"].to_dict()
    
    user_item_stats = {}
    
    for chunk in pd.read_csv(DATA_DIR / "order_products.csv", chunksize=2500000):
        chunk_prior = chunk[chunk["order_id"].isin(prior_order_ids)].copy()
        if chunk_prior.empty:
            continue
        chunk_prior["user_id"] = chunk_prior["order_id"].map(order_to_user)
        
        # Merge product dept
        chunk_prior = chunk_prior.merge(df_prod, on="product_id", how="left")
        
        grp = chunk_prior.groupby("user_id").agg(
            items_count=("reordered", "count"),
            reorders_count=("reordered", "sum"),
            unique_products=("product_id", lambda s: set(s)),
            unique_departments=("department_id", lambda s: set(s)),
            unique_aisles=("aisle_id", lambda s: set(s))
        )
        
        for uid, row in grp.iterrows():
            if uid not in user_item_stats:
                user_item_stats[uid] = [0, 0, set(), set(), set()]
            user_item_stats[uid][0] += row["items_count"]
            user_item_stats[uid][1] += row["reorders_count"]
            user_item_stats[uid][2].update(row["unique_products"])
            user_item_stats[uid][3].update(row["unique_departments"])
            user_item_stats[uid][4].update(row["unique_aisles"])
            
    df_user_items = pd.DataFrame([
        {
            "user_id": uid,
            "user_total_items": stats[0],
            "user_total_reorders": stats[1],
            "user_reorder_rate": round(stats[1] / stats[0], 4) if stats[0] > 0 else 0.0,
            "user_unique_products": len(stats[2]),
            "user_unique_departments": len(stats[3]),
            "user_unique_aisles": len(stats[4])
        }
        for uid, stats in user_item_stats.items()
    ])
    
    # Combine customer features
    df_cust_features = user_order_stats.merge(df_user_items, on="user_id", how="left")
    df_cust_features["user_avg_basket_size"] = round(df_cust_features["user_total_items"] / df_cust_features["user_total_orders"], 2)
    
    # Type downcasting
    df_cust_features["user_id"] = df_cust_features["user_id"].astype(np.int32)
    df_cust_features["user_total_orders"] = df_cust_features["user_total_orders"].astype(np.int16)
    df_cust_features["user_avg_order_interval"] = df_cust_features["user_avg_order_interval"].astype(np.float32)
    df_cust_features["user_std_order_interval"] = df_cust_features["user_std_order_interval"].astype(np.float32)
    df_cust_features["user_preferred_dow"] = df_cust_features["user_preferred_dow"].astype(np.int8)
    df_cust_features["user_preferred_hour"] = df_cust_features["user_preferred_hour"].astype(np.int8)
    df_cust_features["user_days_since_last_order"] = df_cust_features["user_days_since_last_order"].astype(np.float32)
    df_cust_features["user_total_items"] = df_cust_features["user_total_items"].astype(np.int32)
    df_cust_features["user_reorder_rate"] = df_cust_features["user_reorder_rate"].astype(np.float32)
    df_cust_features["user_unique_products"] = df_cust_features["user_unique_products"].astype(np.int16)
    df_cust_features["user_unique_departments"] = df_cust_features["user_unique_departments"].astype(np.int8)
    df_cust_features["user_unique_aisles"] = df_cust_features["user_unique_aisles"].astype(np.int8)
    df_cust_features["user_avg_basket_size"] = df_cust_features["user_avg_basket_size"].astype(np.float32)
    
    # Save Customer features (Key artifact for Phase 6 Customer Segmentation)
    save_artifact(df_cust_features, "customer_features")
    print(f"    [+] Saved customer_features ({len(df_cust_features):,} users in {time.time()-start:.1f}s)")
    return df_cust_features


def build_user_product_features(sample_users: int = 25000):
    """Build User-Product interaction features with ground truth targets for Phase 7 prediction."""
    print(f"[3/3] Building User-Product Interaction Features ({sample_users:,} user cohort for leak-free training)...")
    start = time.time()
    
    df_orders = pd.read_csv(DATA_DIR / "orders.csv")
    df_train_orders = df_orders[df_orders["eval_set"] == "train"]
    
    # Sample users who have a train order for supervised training/evaluation
    sampled_user_ids = set(df_train_orders["user_id"].head(sample_users))
    
    # Prior orders for sampled users
    sampled_prior_orders = df_orders[(df_orders["eval_set"] == "prior") & (df_orders["user_id"].isin(sampled_user_ids))]
    sampled_prior_order_ids = set(sampled_prior_orders["order_id"])
    order_to_user = sampled_prior_orders.set_index("order_id")["user_id"].to_dict()
    order_to_ordernum = sampled_prior_orders.set_index("order_id")["order_number"].to_dict()
    user_max_order = sampled_prior_orders.groupby("user_id")["order_number"].max().to_dict()
    
    # Aggregate user-product interactions
    print("    -> Extracting historical user-product pairs...")
    up_stats = {}
    
    for chunk in pd.read_csv(DATA_DIR / "order_products.csv", chunksize=2500000):
        chunk_user = chunk[chunk["order_id"].isin(sampled_prior_order_ids)].copy()
        if chunk_user.empty:
            continue
        chunk_user["user_id"] = chunk_user["order_id"].map(order_to_user)
        chunk_user["order_number"] = chunk_user["order_id"].map(order_to_ordernum)
        
        grp = chunk_user.groupby(["user_id", "product_id"]).agg(
            total_purchases=("reordered", "count"),
            first_order_num=("order_number", "min"),
            last_order_num=("order_number", "max"),
            cart_sum=("add_to_cart_order", "sum")
        )
        
        for (uid, pid), row in grp.iterrows():
            key = (uid, pid)
            if key not in up_stats:
                up_stats[key] = [0, 9999, 0, 0]
            up_stats[key][0] += row["total_purchases"]
            up_stats[key][1] = min(up_stats[key][1], row["first_order_num"])
            up_stats[key][2] = max(up_stats[key][2], row["last_order_num"])
            up_stats[key][3] += row["cart_sum"]
            
    # Ground truth targets: what was actually purchased in the train order
    print("    -> Extracting ground truth targets from train orders...")
    sampled_train_order_ids = set(df_train_orders[df_train_orders["user_id"].isin(sampled_user_ids)]["order_id"])
    train_order_to_user = df_train_orders.set_index("order_id")["user_id"].to_dict()
    
    train_purchases = set()
    for chunk in pd.read_csv(DATA_DIR / "order_products.csv", chunksize=2500000):
        chunk_train = chunk[chunk["order_id"].isin(sampled_train_order_ids)].copy()
        if chunk_train.empty:
            continue
        chunk_train["user_id"] = chunk_train["order_id"].map(train_order_to_user)
        for _, row in chunk_train.iterrows():
            train_purchases.add((row["user_id"], row["product_id"]))
            
    print(f"    -> Assembling feature matrix for {len(up_stats):,} user-product candidates...")
    records = []
    for (uid, pid), stats in up_stats.items():
        total_user_orders = user_max_order.get(uid, 1)
        up_orders = stats[0]
        first_ord = stats[1]
        last_ord = stats[2]
        cart_avg = stats[3] / up_orders
        orders_since_last = total_user_orders - last_ord
        order_rate = up_orders / total_user_orders
        
        # Binary target: 1 if user bought this product in the subsequent train order, else 0
        target = 1 if (uid, pid) in train_purchases else 0
        
        records.append({
            "user_id": uid,
            "product_id": pid,
            "up_total_orders": up_orders,
            "up_order_rate": round(order_rate, 4),
            "up_orders_since_last": orders_since_last,
            "up_avg_add_to_cart": round(cart_avg, 2),
            "target": target
        })
        
    df_up_features = pd.DataFrame(records)
    
    # Downcasting
    df_up_features["user_id"] = df_up_features["user_id"].astype(np.int32)
    df_up_features["product_id"] = df_up_features["product_id"].astype(np.int32)
    df_up_features["up_total_orders"] = df_up_features["up_total_orders"].astype(np.int16)
    df_up_features["up_order_rate"] = df_up_features["up_order_rate"].astype(np.float32)
    df_up_features["up_orders_since_last"] = df_up_features["up_orders_since_last"].astype(np.int16)
    df_up_features["up_avg_add_to_cart"] = df_up_features["up_avg_add_to_cart"].astype(np.float32)
    df_up_features["target"] = df_up_features["target"].astype(np.int8)
    
    save_artifact(df_up_features, "user_product_features")
    print(f"    [+] Saved user_product_features ({len(df_up_features):,} rows, positive target rate: {df_up_features['target'].mean()*100:.2f}%) in {time.time()-start:.1f}s")
    return df_up_features



def main():
    total_start = time.time()
    print("=" * 70)
    print(" AudienceIQ — Phase 5: Feature Engineering Engine")
    print("=" * 70)
    
    build_product_features()
    build_customer_features()
    build_user_product_features(sample_users=25000)
    
    print("-" * 70)
    print(f"[+] Phase 5 Feature Engineering Engine completed successfully in {time.time()-total_start:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
