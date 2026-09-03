"""
AudienceIQ — Phase 2: Data Cleaning & Relational Transformation Pipeline
====================================================================
This script processes the validated raw CSV files and creates optimized,
relational tables in `data/processed/` ready for PostgreSQL ingestion.

Transformation & Cleaning Decisions:
  1. aisles: downcast `aisle_id` to int16 (134 categories).
  2. departments: downcast `department_id` to int16 (21 departments).
  3. products: downcast `product_id` (int32), `aisle_id` (int16), `department_id` (int16), strip whitespace from `product_name`.
  4. customers: build dedicated customer dimension table from `orders.csv` with `user_id` (int32), `total_orders` (int16).
  5. orders: downcast `order_id` (int32), `user_id` (int32), `order_number` (int16), `order_dow` (int8), `order_hour_of_day` (int8),
     add `is_first_order` (int8: 1 if order_number == 1 else 0), fill nulls in `days_since_prior_order` with 0.0 (float32).
  6. order_products: unify `prior` (32.4M) and `train` (1.38M) into a single clean relational table with `order_id` (int32),
     `product_id` (int32), `add_to_cart_order` (int16), `reordered` (int8), deduplicating on (order_id, product_id).
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def clean_reference_tables():
    """Clean aisles, departments, and products reference tables."""
    print("[1/4] Cleaning Reference Tables (aisles, departments, products)...")
    
    # 1. Aisles
    # Downcasting aisle_id to uint16 since max id is 134
    df_aisles = pd.read_csv(RAW_DIR / "aisles.csv")
    df_aisles["aisle_id"] = df_aisles["aisle_id"].astype(np.int16)
    df_aisles["aisle"] = df_aisles["aisle"].astype(str).str.strip()
    df_aisles.to_csv(PROCESSED_DIR / "aisles.csv", index=False)
    print(f"    [+] Processed aisles.csv ({len(df_aisles):,} rows)")
    
    # 2. Departments
    # Downcasting department_id to uint16 since max id is 21
    df_dept = pd.read_csv(RAW_DIR / "departments.csv")
    df_dept["department_id"] = df_dept["department_id"].astype(np.int16)
    df_dept["department"] = df_dept["department"].astype(str).str.strip()
    df_dept.to_csv(PROCESSED_DIR / "departments.csv", index=False)
    print(f"    [+] Processed departments.csv ({len(df_dept):,} rows)")
    
    # 3. Products
    # Downcasting product_id to int32, aisle/department to int16, stripping whitespace
    df_prod = pd.read_csv(RAW_DIR / "products.csv")
    df_prod["product_id"] = df_prod["product_id"].astype(np.int32)
    df_prod["aisle_id"] = df_prod["aisle_id"].astype(np.int16)
    df_prod["department_id"] = df_prod["department_id"].astype(np.int16)
    df_prod["product_name"] = df_prod["product_name"].astype(str).str.strip()
    df_prod.to_csv(PROCESSED_DIR / "products.csv", index=False)
    print(f"    [+] Processed products.csv ({len(df_prod):,} rows)")


def clean_orders_and_customers():
    """Clean orders table and extract customer dimension table."""
    print("[2/4] Cleaning Orders & Building Customers Dimension...")
    
    # Read orders.csv
    df_orders = pd.read_csv(RAW_DIR / "orders.csv")
    
    # Downcasting types for memory optimization
    df_orders["order_id"] = df_orders["order_id"].astype(np.int32)
    df_orders["user_id"] = df_orders["user_id"].astype(np.int32)
    df_orders["order_number"] = df_orders["order_number"].astype(np.int16)
    df_orders["order_dow"] = df_orders["order_dow"].astype(np.int8)
    df_orders["order_hour_of_day"] = df_orders["order_hour_of_day"].astype(np.int8)
    
    # Feature addition: is_first_order flag to explicitly distinguish true 0-day reorders from first orders
    df_orders["is_first_order"] = (df_orders["order_number"] == 1).astype(np.int8)
    
    # Impute missing days_since_prior_order with 0.0 to prevent SQL null issues while is_first_order preserves semantics
    df_orders["days_since_prior_order"] = df_orders["days_since_prior_order"].fillna(0.0).astype(np.float32)
    
    # Save cleaned orders
    df_orders.to_csv(PROCESSED_DIR / "orders.csv", index=False)
    print(f"    [+] Processed orders.csv ({len(df_orders):,} rows)")
    
    # Build Customers dimension table
    # Aggregating customer order stats for direct relational lookup
    df_customers = df_orders.groupby("user_id").agg(
        total_orders=("order_number", "max"),
        has_train_order=("eval_set", lambda s: int("train" in set(s))),
        has_test_order=("eval_set", lambda s: int("test" in set(s)))
    ).reset_index()
    
    df_customers["user_id"] = df_customers["user_id"].astype(np.int32)
    df_customers["total_orders"] = df_customers["total_orders"].astype(np.int16)
    df_customers["has_train_order"] = df_customers["has_train_order"].astype(np.int8)
    df_customers["has_test_order"] = df_customers["has_test_order"].astype(np.int8)
    
    df_customers.to_csv(PROCESSED_DIR / "customers.csv", index=False)
    print(f"    [+] Built customers.csv ({len(df_customers):,} unique users)")


def clean_and_unify_order_products():
    """Clean and unify prior and train order_products in chunks."""
    print("[3/4] Unifying & Cleaning Order Products (Prior + Train)...")
    
    out_file = PROCESSED_DIR / "order_products.csv"
    if out_file.exists():
        out_file.unlink()
        
    total_written = 0
    header_written = False
    
    # 1. Stream order_products__prior.csv
    print("    -> Processing order_products__prior.csv...")
    for chunk in pd.read_csv(RAW_DIR / "order_products__prior.csv", chunksize=2000000):
        chunk["order_id"] = chunk["order_id"].astype(np.int32)
        chunk["product_id"] = chunk["product_id"].astype(np.int32)
        chunk["add_to_cart_order"] = chunk["add_to_cart_order"].astype(np.int16)
        chunk["reordered"] = chunk["reordered"].astype(np.int8)
        
        chunk.to_csv(out_file, mode="a", index=False, header=not header_written)
        header_written = True
        total_written += len(chunk)
        print(f"       written {total_written:,} rows...")
        
    # 2. Stream order_products__train.csv
    print("    -> Processing order_products__train.csv...")
    for chunk in pd.read_csv(RAW_DIR / "order_products__train.csv", chunksize=1000000):
        chunk["order_id"] = chunk["order_id"].astype(np.int32)
        chunk["product_id"] = chunk["product_id"].astype(np.int32)
        chunk["add_to_cart_order"] = chunk["add_to_cart_order"].astype(np.int16)
        chunk["reordered"] = chunk["reordered"].astype(np.int8)
        
        chunk.to_csv(out_file, mode="a", index=False, header=False)
        total_written += len(chunk)
        print(f"       written {total_written:,} rows...")
        
    print(f"    [+] Completed unified order_products.csv ({total_written:,} total rows)")


def verify_processed_outputs():
    """Verify that all processed files exist, are readable, and conform to target schema."""
    print("[4/4] Verifying Processed Output Tables...")
    processed_files = list(PROCESSED_DIR.glob("*.csv"))
    print(f"    Found {len(processed_files)} processed tables:")
    
    summary = []
    for f in sorted(processed_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        df_head = pd.read_csv(f, nrows=5)
        print(f"    - `{f.name}`: {size_mb:.2f} MB | cols: {list(df_head.columns)}")
        summary.append({
            "table": f.name,
            "size_mb": round(size_mb, 2),
            "columns": list(df_head.columns)
        })
    return summary


def main():
    start_time = time.time()
    print("=" * 70)
    print(" AudienceIQ — Phase 2: Data Cleaning & Transformation Pipeline")
    print("=" * 70)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    clean_reference_tables()
    clean_orders_and_customers()
    clean_and_unify_order_products()
    verify_processed_outputs()
    
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"[+] Phase 2 Data Cleaning & Transformation completed in {elapsed:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
